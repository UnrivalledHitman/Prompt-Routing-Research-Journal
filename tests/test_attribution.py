"""Unit tests for window energy attribution."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from arbiter.energy import attribution

IDLE_W, BUSY_W = 5.0, 25.0


def _trace() -> tuple[np.ndarray, np.ndarray]:
    """Idle power, then 25 W between t = 10 and t = 30, then idle again."""
    t = np.round(np.arange(0.0, 50.0001, 0.1), 4)
    p = np.where((t >= 10.0) & (t <= 30.0), BUSY_W, IDLE_W)
    return t, p


def test_baseline_power_uses_accepted_baseline_windows_only() -> None:
    """Rejected windows and batch windows do not enter the baseline."""
    t, p = _trace()
    windows = [
        {"kind": "baseline_start", "accepted": True, "start_perf": 1.0, "end_perf": 9.0},
        {"kind": "baseline_start", "accepted": False, "start_perf": 12.0, "end_perf": 20.0},
        {"kind": "batch", "accepted": True, "start_perf": 10.0, "end_perf": 30.0},
    ]
    assert attribution.baseline_power_w(t, p, windows) == pytest.approx(IDLE_W)
    with pytest.raises(ValueError):
        attribution.baseline_power_w(t, p, windows[1:])


def test_attribute_window_splits_board_and_dynamic_energy_by_wall_time() -> None:
    """Dynamic energy removes idle power over the window and preserves the total."""
    t, p = _trace()
    window = {"window_id": "w", "session_id": "s", "start_perf": 10.0, "end_perf": 30.0}
    records = [
        {"prompt_id": f"p{i}", "tag": "q", "size": "1b", "mode": "think", "sample": 0,
         "variant": "base", "wall_s": wall, "eval_count": tokens}
        for i, (wall, tokens) in enumerate([(1.0, 300), (3.0, 100)])
    ]  # fmt: skip
    rows = attribution.attribute_window(t, p, window, records, IDLE_W)
    assert rows[0]["window_board_j"] == pytest.approx(BUSY_W * 20.0)
    assert [r["energy_dyn_j"] for r in rows] == pytest.approx([100.0, 300.0])
    assert [r["energy_dyn_token_j"] for r in rows] == pytest.approx([300.0, 100.0])
    assert sum(r["energy_board_j"] for r in rows) == pytest.approx(rows[0]["window_board_j"])
    for record in records:
        del record["eval_count"]
    rows = attribution.attribute_window(t, p, window, records, IDLE_W)
    assert [r["energy_dyn_token_j"] for r in rows] == pytest.approx([100.0, 300.0])


def test_attribute_stage_reads_a_stage_directory(tmp_path: Path) -> None:
    """Accepted windows are attributed; rejected attempts are ignored."""
    t, p = _trace()
    (tmp_path / "telemetry").mkdir()
    with (tmp_path / "telemetry" / "s.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["perf_s", "epoch_s", "power_mw", "util_pct", "temp_c", "mem_used_mb"])
        writer.writerows([ti, ti, int(pi * 1000), 0, 40, 900] for ti, pi in zip(t, p, strict=True))
    base = {"session_id": "s", "telemetry_file": "telemetry/s.csv"}
    windows = [
        {**base, "window_id": "b0", "kind": "baseline_start", "accepted": True,
         "start_perf": 0.0, "end_perf": 9.0},
        {**base, "window_id": "w-a1", "kind": "batch", "accepted": False, "reason": "contaminated",
         "start_perf": 10.0, "end_perf": 20.0, "n_generations": 1},
        {**base, "window_id": "w-a2", "kind": "batch", "accepted": True, "contaminated": False,
         "start_perf": 10.0, "end_perf": 30.0, "n_generations": 2},
    ]  # fmt: skip
    generations = [
        {"prompt_id": f"p{i}", "tag": "q", "size": "1b", "mode": "no_think", "sample": 0,
         "variant": "base", "wall_s": 2.0, "window_id": "w-a2"}
        for i in range(2)
    ]  # fmt: skip
    (tmp_path / "windows.jsonl").write_text("".join(json.dumps(w) + "\n" for w in windows))
    (tmp_path / "generations.jsonl").write_text("".join(json.dumps(g) + "\n" for g in generations))
    rows = attribution.attribute_stage(tmp_path)
    assert [r["energy_dyn_j"] for r in rows] == pytest.approx([200.0, 200.0])
    assert {r["window_id"] for r in rows} == {"w-a2"}
    assert {r["energy_excluded"] for r in rows} == {""}


def test_window_exclusion_reasons() -> None:
    """Sampling gaps, harness flags and battery periods each exclude a window."""
    t, _ = _trace()
    window = {"start_perf": 10.0, "end_perf": 30.0}
    assert attribution.window_exclusion(t, window) is None
    gappy = t[(t < 15.0) | (t > 16.5)]
    assert attribution.window_exclusion(gappy, window) == "sampling_gap"
    assert attribution.window_exclusion(gappy, window, max_gap_s=2.0) is None
    assert attribution.window_exclusion(t, {**window, "stall_s": 3.0}) == "sampling_gap"
    assert attribution.window_exclusion(t, {**window, "on_battery": True}) == "on_battery"
    clock = (t, t + 1000.0)  # wall clock runs 1000 s ahead of perf
    assert attribution.window_exclusion(t, window, clock, [(1029.0, 1040.0)]) == "on_battery"
    assert attribution.window_exclusion(t, window, clock, [(1031.0, 1040.0)]) is None
    with pytest.raises(ValueError, match="clock"):
        attribution.window_exclusion(t, window, None, [(1029.0, 1040.0)])


def test_attribute_window_with_an_exclusion_has_no_energy() -> None:
    """An excluded window keeps its rows, with NaN energy and the reason."""
    t, p = _trace()
    window = {"window_id": "w", "session_id": "s", "start_perf": 10.0, "end_perf": 30.0}
    records = [
        {"prompt_id": "p0", "tag": "q", "size": "1b", "mode": "think", "sample": 0,
         "variant": "base", "wall_s": 2.0, "eval_count": 10}
    ]  # fmt: skip
    (row,) = attribution.attribute_window(t, p, window, records, IDLE_W, "on_battery")
    assert row["energy_excluded"] == "on_battery"
    for key in ("energy_board_j", "energy_dyn_j", "energy_dyn_token_j", "window_board_j"):
        assert np.isnan(row[key])
    assert row["wall_s"] == 2.0


def test_attribute_stage_excludes_gaps_and_battery_instead_of_raising(tmp_path: Path) -> None:
    """A suspend-length gap and a battery period mark windows; the rest are attributed."""
    t = np.round(np.arange(0.0, 90.0001, 0.1), 4)
    p = np.where((t >= 10.0) & (t <= 30.0), BUSY_W, IDLE_W)
    keep = (t < 40.0) | (t > 45.0)  # telemetry stalls for 5 s inside window w2
    (tmp_path / "telemetry").mkdir()
    with (tmp_path / "telemetry" / "s.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["perf_s", "epoch_s", "power_mw", "util_pct", "temp_c", "mem_used_mb"])
        writer.writerows(
            [ti, ti + 1000.0, int(pi * 1000), 0, 40, 900]
            for ti, pi in zip(t[keep], p[keep], strict=True)
        )
    base = {"session_id": "s", "telemetry_file": "telemetry/s.csv", "kind": "batch",
            "accepted": True, "n_generations": 1}  # fmt: skip
    windows = [
        {**base, "window_id": "b0", "kind": "baseline_start", "start_perf": 0.0, "end_perf": 9.0},
        {**base, "window_id": "w1", "start_perf": 10.0, "end_perf": 30.0},
        {**base, "window_id": "w2", "start_perf": 35.0, "end_perf": 50.0},
        {**base, "window_id": "w3", "start_perf": 60.0, "end_perf": 70.0},
    ]
    generations = [
        {"prompt_id": w, "tag": "q", "size": "1b", "mode": "think", "sample": 0,
         "variant": "base", "wall_s": 2.0, "window_id": w}
        for w in ("w1", "w2", "w3")
    ]  # fmt: skip
    (tmp_path / "windows.jsonl").write_text("".join(json.dumps(w) + "\n" for w in windows))
    (tmp_path / "generations.jsonl").write_text("".join(json.dumps(g) + "\n" for g in generations))
    rows = {r["window_id"]: r for r in attribution.attribute_stage(tmp_path, [(1065.0, 1066.0)])}
    assert rows["w1"]["energy_excluded"] == ""
    assert rows["w1"]["energy_dyn_j"] == pytest.approx((BUSY_W - IDLE_W) * 20.0)
    assert rows["w2"]["energy_excluded"] == "sampling_gap"
    assert rows["w3"]["energy_excluded"] == "on_battery"
    assert np.isnan(rows["w2"]["energy_dyn_j"]) and np.isnan(rows["w3"]["energy_dyn_j"])
