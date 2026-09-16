"""Unit tests for scripts/select_energy_reruns.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_energy_reruns", REPO_ROOT / "scripts" / "select_energy_reruns.py"
)
assert SPEC is not None and SPEC.loader is not None
select_energy_reruns = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(select_energy_reruns)


def _row(pid: str, reason: str, sample: int = 0, variant: str = "base") -> dict[str, object]:
    return {"prompt_id": pid, "size": "8b", "mode": "think", "sample": sample,
            "variant": variant, "window_id": f"w-{pid}", "energy_excluded": reason}  # fmt: skip


def test_excluded_cells_lists_only_excluded_generations() -> None:
    """Measured generations are not rerun; the list is sorted and names the reason."""
    energy = pd.DataFrame([_row("b", "on_battery"), _row("a", ""), _row("c", "sampling_gap")])
    cells = select_energy_reruns.excluded_cells(energy)
    assert cells["prompt_id"].tolist() == ["b", "c"]
    assert cells["energy_excluded"].tolist() == ["on_battery", "sampling_gap"]
    assert {"prompt_id", "size", "mode"} <= set(cells.columns)


def test_excluded_cells_refuses_generations_the_stage_cannot_reproduce() -> None:
    """Only sample 0 of the base variant can be rerun with its original seed."""
    energy = pd.DataFrame([_row("a", "on_battery", sample=1)])
    with pytest.raises(ValueError, match="non-base"):
        select_energy_reruns.excluded_cells(energy)
