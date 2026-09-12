"""Tests for the progress board's arithmetic and its projection of unstarted cells."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("progress", REPO_ROOT / "scripts" / "progress.py")
progress = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(progress)


def response(
    size: str, mode: str, tokens: int, wall: float, capped: bool = False
) -> dict[str, Any]:
    """Return one generation record with the fields the board reads."""
    return {
        "subset": "main",
        "size": size,
        "mode": mode,
        "eval_count": tokens,
        "wall_s": wall,
        "done_reason": "length" if capped else "stop",
    }


def test_bar_is_empty_full_and_clamped() -> None:
    """The bar fills in proportion and never overruns its width."""
    assert progress.bar(0, 10, width=4) == "...."
    assert progress.bar(10, 10, width=4) == "####"
    assert progress.bar(20, 10, width=4) == "####"
    assert progress.bar(1, 0, width=4) == "----"


def test_clock_reads_as_hours_and_minutes() -> None:
    """Durations format as hours and minutes, and a non-positive one is not a time."""
    assert progress.clock(0) == "--"
    assert progress.clock(-5) == "--"
    assert progress.clock(90) == "2m"
    assert progress.clock(3600 + 120) == "1h02m"


def test_size_params_reads_the_tag() -> None:
    """A size tag names its parameter count in billions."""
    assert progress.size_params("0.6b") == pytest.approx(0.6)
    assert progress.size_params("8B") == pytest.approx(8.0)


def test_throughput_measures_what_it_can_and_fits_the_rest() -> None:
    """Measured sizes keep their own rate; an unstarted size is fitted and marked as such."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {
        ("0.6b", "no_think"): [response("0.6b", "no_think", 200, 1.0)] * 50,
        ("1.7b", "no_think"): [response("1.7b", "no_think", 100, 1.0)] * 50,
        ("4b", "no_think"): [response("4b", "no_think", 50, 1.0)] * 50,
    }
    speed, measured = progress.throughput(cells, floor=10)
    assert measured == {"0.6b", "1.7b", "4b"}
    assert speed["0.6b"] == pytest.approx(200.0)
    # The fit is exact on a power law, and 8b must land below the smallest measured rate.
    assert "8b" in speed
    assert speed["8b"] < speed["4b"]


def test_throughput_fits_nothing_from_one_size() -> None:
    """With a single measured size there is no slope, so no size is filled in."""
    cells = {("0.6b", "no_think"): [response("0.6b", "no_think", 200, 1.0)] * 50}
    speed, measured = progress.throughput(cells, floor=10)
    assert measured == {"0.6b"}
    assert set(speed) == {"0.6b"}


def write_run(directory: Path, records: list[dict[str, Any]], target: int) -> None:
    """Write the four files the board reads for a labelling stage."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "generations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )
    (directory / "runs.jsonl").write_text(
        json.dumps({"stage": "main", "n_prompts": target}), encoding="utf-8"
    )
    (directory / "sessions.jsonl").write_text("", encoding="utf-8")
    (directory / "windows.jsonl").write_text("", encoding="utf-8")


def test_labelling_projects_a_running_cell_from_its_own_rate(tmp_path: Path) -> None:
    """Half a cell at 2 s a response leaves the other half at 2 s a response."""
    records = [response("0.6b", "no_think", 100, 2.0) for _ in range(5)]
    write_run(tmp_path, records, target=10)
    lines, remaining = progress.labelling(tmp_path, "main")
    assert remaining == pytest.approx(5 * 2.0)  # no finished session, so overhead is 1.0
    assert "RUNNING" in "\n".join(lines)


def test_labelling_reports_nothing_for_a_stage_with_no_records(tmp_path: Path) -> None:
    """A stage that has not run is said to have no records rather than projected."""
    write_run(tmp_path, [], target=10)
    lines, remaining = progress.labelling(tmp_path, "main")
    assert remaining == 0.0
    assert "no main-stage records" in lines[0]


def test_hosted_is_silent_without_a_ledger(tmp_path: Path) -> None:
    """The hosted block disappears rather than printing an empty frame."""
    assert progress.hosted(tmp_path, "main") == []


def test_hosted_totals_the_ledger(tmp_path: Path) -> None:
    """Spend is the sum of settled costs, and an unsettled batch contributes nothing."""
    rows = [
        {"actual_usd": 1.5, "cap_usd": 8.0, "label": "main_w1"},
        {"actual_usd": None, "cap_usd": 8.0, "label": "main_w2"},
    ]
    (tmp_path / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    line = progress.hosted(tmp_path, "main")[0]
    assert "$1.5000 of $8.00" in line


def test_paper_counts_placeholders(tmp_path: Path) -> None:
    """Both placeholder macros count, and a clean section says so."""
    (tmp_path / "01.tex").write_text("a \\tbd{x} b \\decision{y}\n", encoding="utf-8")
    (tmp_path / "02.tex").write_text("nothing pending\n", encoding="utf-8")
    lines = progress.paper(tmp_path)
    assert "2 placeholder(s)" in lines[1]
    assert "clear" in lines[2]
    assert lines[-1] == "  2 placeholder(s) left"


def test_queue_dates_each_stage_after_the_grid(tmp_path: Path) -> None:
    """Stages are dated cumulatively from the moment the grid finishes."""
    config = tmp_path / "progress.yaml"
    config.write_text(
        "queue:\n"
        "  - {name: first, hours: 1.0, detail: d}\n"
        "  - {name: second, hours: 2.0, detail: d}\n",
        encoding="utf-8",
    )
    lines = progress.queue(config, after=0.0)
    assert len(lines) == 3
    assert " 1.0 h" in lines[1]
    assert " 2.0 h" in lines[2]


def test_queue_is_silent_without_a_config(tmp_path: Path) -> None:
    """A missing queue file removes the block instead of failing."""
    assert progress.queue(tmp_path / "absent.yaml", after=0.0) == []


def test_throughput_fit_recovers_a_power_law() -> None:
    """Rates generated by a power law are reproduced by the fit at the missing size."""
    rates = {"0.6b": 300.0, "1.7b": 150.0, "4b": 75.0}
    cells = {
        (size, "no_think"): [response(size, "no_think", int(rate), 1.0)] * 50
        for size, rate in rates.items()
    }
    speed, _ = progress.throughput(cells, floor=10)
    xs = [math.log(progress.size_params(s)) for s in rates]
    ys = [math.log(v) for v in rates.values()]
    mean_x, mean_y = sum(xs) / 3, sum(ys) / 3
    spread = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / spread
    expected = math.exp(mean_y + slope * (math.log(8.0) - mean_x))
    assert speed["8b"] == pytest.approx(expected)
