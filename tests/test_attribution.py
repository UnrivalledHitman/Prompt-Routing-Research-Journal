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
