"""Unit tests for the label robustness checks, on hand-built grades."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest

from arbiter.eval import robustness
from arbiter.eval.labels import CELLS, KEY, LabelTable, build_label_table, cell_index


def _hits(*indices: int) -> list[bool]:
    return [index in indices for index in range(len(CELLS))]


def _rows(
    prompt_id: str,
    family: str,
    correct: Sequence[bool],
    sample: int = 0,
    variant: str = "base",
    capped: Sequence[int] = (),
    tokens: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    """One generation per cell; output tokens default to ten times the cell's position."""
    return [
        {
            "prompt_id": prompt_id,
            "family": family,
            "source": "src",
            "subset": "pilot",
            "size": size,
            "tag": f"qwen3:{size}",
            "mode": mode,
            "sample": sample,
            "variant": variant,
            "correct": bool(correct[index]),
            "done_reason": "length" if index in capped else "stop",
            "prompt_eval_count": 0,
            "eval_count": (tokens or {}).get(index, 10 * (index + 1)),
            "wall_s": 1.0 + index,
            "contaminated": False,
        }
        for index, (size, mode) in enumerate(CELLS)
    ]


def _rerun(
    prompt_id: str, family: str, index: int, correct: bool, capped: bool, tokens: int
) -> dict[str, Any]:
    """One generation of a single cell under the longer cap."""
    rows = _rows(
        prompt_id,
        family,
        _hits(index) if correct else _hits(),
        variant="cap8192",
        capped=(index,) if capped else (),
        tokens={index: tokens},
    )
    return next(r for r in rows if cell_index(r["size"], r["mode"]) == index)


def test_label_noise_counts_disagreement_and_cstar_changes() -> None:
    """Sample disagreement on one cell moves c* for one sample but not the majority."""
    rows = []
    for sample, first in enumerate([True, False, True]):
        rows += _rows("p0", "math", _hits(*([0, 7] if first else [7])), sample=sample)
        rows += _rows("p1", "code", _hits(2, 7), sample=sample)
    cells, cstar = robustness.label_noise(pd.DataFrame(rows))

    c1 = cells[(cells["scope"] == "all") & (cells["cell"] == "c1")].iloc[0]
    assert c1["success_rate"] == pytest.approx(2 / 6)
    assert c1["varies_rate"] == pytest.approx(0.5)
    assert c1["pairwise_disagreement"] == pytest.approx(1 / 3)
    assert c1["majority_differs"] == 0.0

    overall = cstar.set_index("scope").loc["all"]
    assert overall["cstar_pairwise_agreement"] == pytest.approx(4 / 6)
    assert overall["cstar_first_vs_majority"] == 1.0
    assert overall["servability_varies"] == 0.0
    assert overall["cell_pairwise_disagreement"] == pytest.approx((2 / 3) / 16)
    assert cstar.set_index("scope").loc["math", "cstar_pairwise_agreement"] == pytest.approx(1 / 3)


def test_cap_sensitivity_rescues_and_moves_cstar() -> None:
    """Rescued cells move c* only when cheaper; reruns of uncapped cells are ignored."""
    base = (
        _rows("p0", "math", _hits(7), capped=(1, 3))
        + _rows("p1", "code", _hits(), capped=(3,))
        + _rows("p2", "mcq", _hits(0), capped=(5,))
    )
    reruns = [
        _rerun("p0", "math", 1, correct=True, capped=False, tokens=60),  # 60 < 80: moves
        _rerun("p0", "math", 3, correct=False, capped=True, tokens=8192),
        _rerun("p1", "code", 3, correct=True, capped=False, tokens=900),  # newly servable
        _rerun("p2", "mcq", 5, correct=True, capped=False, tokens=900),  # dearer than c1
        _rerun("p2", "mcq", 0, correct=True, capped=False, tokens=5),  # base not capped
    ]
    table = robustness.cap_sensitivity(pd.DataFrame(base + reruns))
    overall = table[(table["family"] == "all") & (table["size"] == "all")].iloc[0]
    assert overall["n_rerun"] == 4
    assert overall["cap_rate"] == pytest.approx(4 / 24)
    assert overall["rescued_rate"] == pytest.approx(3 / 4)
    assert overall["still_capped_rate"] == pytest.approx(1 / 4)
    assert overall["cstar_moved_rate"] == pytest.approx(2 / 4)
    assert overall["newly_servable_rate"] == pytest.approx(1 / 4)

    cell = table[(table["size"] == "1.7b") & (table["mode"] == "think")].iloc[0]
    assert (cell["n_rerun"], cell["rescued_rate"]) == (2, 0.5)
    assert cell["cap_rate"] == pytest.approx(2 / 3)
    assert table.set_index("family").loc["mcq", "cstar_moved_rate"] == 0.0

    with pytest.raises(ValueError):
        robustness.cap_sensitivity(pd.DataFrame(base))


def test_apportioning_check_compares_cstar_and_cell_means() -> None:
    """A swap in the cheapest successful cell shows up as c* disagreement."""
    wall = np.arange(16.0).reshape(2, 8) + 10.0
    tokens = 2.0 * wall
    wall[0, 2], tokens[0, 0], tokens[0, 2] = 5.0, 4.0, 6.0
    success = np.zeros((2, 8), dtype=bool)
    success[0, [0, 2]] = True
    success[1, 7] = True
    table = LabelTable(
        prompts=pd.DataFrame({"prompt_id": ["p0", "p1"], "family": ["math", "code"]}),
        success=success,
        costs={"energy_j": wall, "energy_token_j": tokens},
        capped=np.zeros((2, 8), dtype=bool),
        contaminated=np.zeros((2, 8), dtype=bool),
    )
    result = robustness.apportioning_check(table).set_index("scope")
    assert result.loc["all", "cstar_agreement"] == 0.5
    assert result.loc["code", "cstar_agreement"] == 1.0
    assert result.loc["all", "ratio_c8"] == pytest.approx(2.0)
    assert result.loc["all", "ratio_c1"] == pytest.approx(20.0 / 14.0)


def test_label_table_adds_token_share_energy_only_when_present() -> None:
    """The token-share unit appears only if attribution wrote it."""
    grades = pd.DataFrame(_rows("p0", "math", _hits(7)))
    energy = grades[KEY].assign(energy_dyn_j=1.0, energy_board_j=2.0)
    assert "energy_token_j" not in build_label_table(grades, energy).costs
    table = build_label_table(grades, energy.assign(energy_dyn_token_j=3.0))
    assert np.all(table.costs["energy_token_j"] == 3.0)


def test_energy_drift_pairs_samples_and_ignores_unpaired_prompts() -> None:
    """Energy per token after over before, on prompts and cells present in both groups."""
    before = pd.DataFrame(_rows("p0", "math", _hits(0)) + _rows("p1", "code", _hits(1)))
    after = pd.DataFrame(_rows("p0", "math", _hits(0), sample=1))
    grades = pd.concat([before, after], ignore_index=True)
    # Sample 1 spends 1.5 times the joules of sample 0 on the same tokens; baseline rises.
    energy = grades[KEY].assign(
        energy_dyn_j=np.where(grades["sample"] == 1, 1.5, 1.0) * grades["eval_count"] * 0.2,
        baseline_w=np.where(grades["sample"] == 1, 4.0, 3.0),
    )
    result = robustness.energy_drift(grades, energy, n_boot=50).set_index("cell")
    assert len(result) == len(CELLS)
    assert (result["n_prompts"] == 1).all()
    assert result.loc["c1", "j_per_token_before"] == pytest.approx(0.2)
    assert result.loc["c8", "ratio"] == pytest.approx(1.5)
    assert result.loc["c8", "ratio_lo"] == pytest.approx(1.5)
    assert result.loc["c3", "baseline_w_after"] == pytest.approx(4.0)


def test_energy_drift_needs_pairs() -> None:
    """With no prompt measured in both groups there is nothing to compare."""
    grades = pd.DataFrame(_rows("p0", "math", _hits(0)))
    energy = grades[KEY].assign(energy_dyn_j=1.0, baseline_w=3.0)
    with pytest.raises(ValueError, match="before and after"):
        robustness.energy_drift(grades, energy)
