"""Unit tests for the labelling harness helpers that do not touch the GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

from arbiter.data.corpus import PromptItem
from arbiter.energy.telemetry import TelemetrySample
from arbiter.labeling import harness

REPO_ROOT = Path(__file__).resolve().parents[1]


def _item(prompt_id: str, family: str, subset: str = "pilot", noise: bool = False) -> PromptItem:
    return PromptItem(prompt_id, family, "src", "text", None, subset, noise)


def _sample(util: int | None) -> TelemetrySample:
    return TelemetrySample(0.0, 0.0, 1000, util, 40, 900.0)


def test_cell_seed_is_deterministic_and_distinct() -> None:
    """The same cell gives the same seed; sample index and variant change it."""
    seed = harness.cell_seed(1, "gsm8k-1", "qwen3:8b", "think", 0, "base")
    assert seed == harness.cell_seed(1, "gsm8k-1", "qwen3:8b", "think", 0, "base")
    assert seed != harness.cell_seed(1, "gsm8k-1", "qwen3:8b", "think", 1, "base")
    assert seed != harness.cell_seed(1, "gsm8k-1", "qwen3:8b", "think", 0, "cap8192")
    assert 0 <= seed < 2**31


def test_chunks_keeps_the_remainder() -> None:
    """The last chunk holds whatever is left."""
    assert harness.chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError):
        harness.chunks([1], 0)


def test_tail_contaminated_uses_mean_utilisation() -> None:
    """Sustained utilisation above the ceiling flags the window; no data does not."""
    assert harness.tail_contaminated([_sample(0), _sample(4)], 15) == (False, 2.0)
    assert harness.tail_contaminated([_sample(90), _sample(60)], 15) == (True, 75.0)
    assert harness.tail_contaminated([_sample(None)], 15) == (False, None)


def test_power_contaminated_flags_battery_and_stalls() -> None:
    """Any battery reading or a stall past the limit flags the window; unknown power does not."""
    assert harness.power_contaminated([True, True], 0.0, 1.0) == (False, False)
    assert harness.power_contaminated([True, False, True], 0.0, 1.0) == (True, False)
    assert harness.power_contaminated([None, None], 0.0, 1.0) == (False, False)
    assert harness.power_contaminated([True], 1.0, 1.0) == (False, False)
    assert harness.power_contaminated([True], 846.5, 1.0) == (False, True)


def test_select_items_filters_limits_and_shuffles_deterministically() -> None:
    """Selection honours subset, noise and per-family limits with a stable order."""
    config = harness.load_config(REPO_ROOT / "configs" / "labelling.yaml")
    items = [_item(f"m{i}", "math", noise=i < 2) for i in range(5)]
    items += [_item(f"c{i}", "code") for i in range(5)]
    items += [_item(f"x{i}", "math", subset="main") for i in range(3)]

    pilot = harness.resolve_stage(config, "pilot", REPO_ROOT)
    chosen = harness.select_items(items, pilot, 7)
    assert {item.prompt_id for item in chosen} == {f"m{i}" for i in range(5)} | {
        f"c{i}" for i in range(5)
    }
    assert [i.prompt_id for i in chosen] == [
        i.prompt_id for i in harness.select_items(items, pilot, 7)
    ]

    noise = harness.resolve_stage(config, "noise", REPO_ROOT)
    assert {item.prompt_id for item in harness.select_items(items, noise, 7)} == {"m0", "m1"}

    smoke = harness.resolve_stage(config, "smoke", REPO_ROOT)
    limited = harness.select_items(items, smoke, 7)
    assert sorted(item.prompt_id for item in limited) == ["c0", "c1", "m0", "m1"]


def test_resolve_stage_merges_overrides() -> None:
    """Stage overrides replace base settings; unknown stages raise."""
    config = harness.load_config(REPO_ROOT / "configs" / "labelling.yaml")
    smoke = harness.resolve_stage(config, "smoke", REPO_ROOT)
    assert smoke.batch_size == 3
    assert smoke.sizes == ("0.6b", "1.7b")
    assert smoke.decoding["think"]["num_predict"] == 4096
    assert smoke.decoding["no_think"]["num_predict"] == 2048
    assert smoke.raw_dir == REPO_ROOT / "results" / "raw" / "labelling_smoke"
    with pytest.raises(KeyError):
        harness.resolve_stage(config, "nope", REPO_ROOT)


def test_cells_stage_restricts_prompts_and_cells(tmp_path: Path) -> None:
    """A cells stage keeps listed prompts and runs only the listed cells."""
    cells_file = tmp_path / "cells.csv"
    cells_file.write_text("prompt_id,size,mode\nm1,8b,think\nc2,0.6b,no_think\n", encoding="utf-8")
    config = harness.load_config(REPO_ROOT / "configs" / "labelling.yaml")
    config["stages"]["rerun"] = {
        "select": "cells",
        "cells_file": str(cells_file),
        "variant": "cap8192",
        "num_ctx": 10240,
        "decoding": {"think": {"num_predict": 8192}},
    }
    stage = harness.resolve_stage(config, "rerun", REPO_ROOT)
    items = [_item(f"m{i}", "math") for i in range(3)] + [_item(f"c{i}", "code") for i in range(3)]
    assert {item.prompt_id for item in harness.select_items(items, stage, 7)} == {"m1", "c2"}
    assert stage.allows("m1", "8b", "think")
    assert not stage.allows("m1", "8b", "no_think")
    assert stage.num_ctx == 10240
    assert stage.decoding["think"]["num_predict"] == 8192
    pilot = harness.resolve_stage(config, "pilot", REPO_ROOT)
    assert pilot.allows("anything", "4b", "think")
    assert pilot.num_ctx == config["num_ctx"]
    del config["stages"]["rerun"]["cells_file"]
    with pytest.raises(ValueError):
        harness.resolve_stage(config, "rerun", REPO_ROOT)
