"""Unit tests for label tables, minimum sufficient configurations and splits."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from arbiter.eval import labels


def _grades(prompt_ids: list[str], drop_last_cell_of: str | None = None) -> pd.DataFrame:
    rows = []
    for pid in prompt_ids:
        for index, (size, mode) in enumerate(labels.CELLS):
            if pid == drop_last_cell_of and index == 7:
                continue
            rows.append(
                {
                    "prompt_id": pid, "family": "math", "source": "gsm8k", "subset": "pilot",
                    "size": size, "tag": f"qwen3:{size}", "mode": mode, "sample": 0,
                    "variant": "base", "correct": index >= 2, "done_reason": "stop",
                    "prompt_eval_count": 10, "eval_count": 100 * (index + 1),
                    "wall_s": float(index + 1), "contaminated": False,
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


def test_build_label_table_pivots_cells_into_grid_order() -> None:
    """Success and costs land in c1..c8 order; incomplete prompts are dropped."""
    grades = _grades(["a", "b"], drop_last_cell_of="b")
    energy = grades[labels.KEY].assign(
        energy_dyn_j=np.arange(len(grades), dtype=float), energy_board_j=1.0
    )
    table = labels.build_label_table(grades, energy, price_in=1.0, price_out=2.0)
    assert table.prompts["prompt_id"].tolist() == ["a"]
    assert table.success[0].tolist() == [False, False] + [True] * 6
    assert table.costs["tokens"][0, 0] == 10 + 2 * 100
    assert table.costs["latency_s"][0].tolist() == [float(i + 1) for i in range(8)]
    assert table.costs["energy_j"][0].tolist() == [float(i) for i in range(8)]
    assert not table.capped.any()


def test_build_label_table_takes_latency_from_the_energy_measurement() -> None:
    """A cell rerun for its energy uses the rerun's wall time; cells without energy keep theirs."""
    grades = _grades(["a"])
    energy = grades[[*labels.KEY, "wall_s"]].assign(energy_dyn_j=1.0, energy_board_j=2.0)
    energy.loc[3, "wall_s"] = 0.25
    energy = energy.drop(index=5)
    table = labels.build_label_table(grades, energy)
    expected = [float(i + 1) for i in range(8)]
    expected[3] = 0.25
    assert table.costs["latency_s"][0].tolist() == expected
    assert np.isnan(table.costs["energy_j"][0, 5])


def test_build_label_table_rejects_duplicate_cells() -> None:
    """The same prompt and cell twice is a data error."""
    grades = _grades(["a"])
    with pytest.raises(ValueError):
        labels.build_label_table(pd.concat([grades, grades.iloc[:1]]), None)


def test_minimum_sufficient_and_cstar_cost() -> None:
    """c* is the cheapest successful cell, not the lowest index; unservable is -1."""
    success = np.array([[0, 1, 1, 0, 0, 0, 0, 0], [0] * 8], dtype=bool)
    cost = np.array([[1, 9, 3, 4, 5, 6, 7, 8], [1] * 8], dtype=float)
    assert labels.minimum_sufficient(success, cost).tolist() == [2, -1]
    best = labels.cstar_cost(success, cost)
    assert best[0] == 3.0
    assert np.isnan(best[1])


def test_assign_splits_is_stratified_and_deterministic() -> None:
    """Each stratum splits 60/20/20 and the assignment is reproducible."""
    ids = [f"p{i}" for i in range(200)]
    strata = ["x"] * 100 + ["y"] * 100
    splits = labels.assign_splits(ids, strata, seed=1)
    for stratum in ("x", "y"):
        counts = Counter(s for s, st in zip(splits, strata, strict=True) if st == stratum)
        assert counts == {"train": 60, "val": 20, "test": 20}
    assert (splits == labels.assign_splits(ids, strata, seed=1)).all()
    assert not (splits == labels.assign_splits(ids, strata, seed=2)).all()
