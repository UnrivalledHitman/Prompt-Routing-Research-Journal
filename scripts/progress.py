"""Print a progress board: which stage is running, how far it has got, how long it needs.

Everything about the labelling grid is read from the run's own records. A cell that has
started is projected from its own measured seconds per response; a cell that has not is
projected from the decode throughput of the sizes that have finished, extrapolated in model
size, which is the only predictor the run itself supplies. Session overhead -- telemetry
windows, cooldowns and windows rejected for other GPU work -- is carried as the measured
ratio of session wall clock to generation wall clock over the sizes that have finished.

The queue of stages that follow the grid is read from ``configs/progress.yaml``, because
those hours are estimates rather than measurements and should be editable as such.

Usage::

    python scripts/progress.py
    python scripts/progress.py --watch --every 300
    python scripts/progress.py --labelling ../main-checkout/results/raw/labelling
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from arbiter.eval.labels import CELLS  # noqa: E402

BAR_WIDTH = 26
IDLE_S = 600.0  # a gap longer than this is worth showing; a thinking window can take minutes


def bar(done: float, total: float, width: int = BAR_WIDTH) -> str:
    """Return a text progress bar.

    Parameters
    ----------
    done, total
        Progress and its target. A non-positive ``total`` gives an empty bar.
    width
        Characters in the bar.
    """
    if total <= 0:
        return "-" * width
    filled = int(round(width * min(done / total, 1.0)))
    return "#" * filled + "." * (width - filled)


def clock(seconds: float) -> str:
    """Format a duration as hours and minutes, or ``--`` if it is not positive."""
    if seconds <= 0:
        return "--"
    hours, minutes = divmod(int(round(seconds / 60)), 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file, returning an empty list if it does not exist."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def size_params(size: str) -> float:
    """Return the parameter count in billions named by a size tag such as ``1.7b``."""
    return float(size.rstrip("bB"))


def throughput(
    cells: dict[tuple[str, str], list[dict[str, Any]]], floor: int
) -> tuple[dict[str, float], set[str]]:
    """Estimate decode tokens per second for every size in the grid.

    Sizes with at least ``floor`` responses get their measured rate. The rest are filled in
    from a log-log fit of rate against parameter count, because decode speed falls with size
    as a power law on one device. With fewer than two measured sizes nothing is filled in.

    Returns
    -------
    tuple
        The rate per size, and the set of sizes whose rate was measured rather than fitted.
    """
    sizes = dict.fromkeys(size for size, _ in CELLS)
    speed: dict[str, float] = {}
    for size in sizes:
        rows = [row for (s, _), group in cells.items() if s == size for row in group]
        if len(rows) >= floor:
            wall = sum(row["wall_s"] for row in rows)
            if wall > 0:
                speed[size] = sum(row["eval_count"] for row in rows) / wall
    seen = set(speed)
    if len(speed) < 2:
        return speed, seen
    xs = [math.log(size_params(s)) for s in speed]
    ys = [math.log(v) for v in speed.values()]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    spread = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / spread
    for size in sizes:
        if size not in speed:
            speed[size] = math.exp(mean_y + slope * (math.log(size_params(size)) - mean_x))
    return speed, seen


def labelling(directory: Path, stage: str) -> tuple[list[str], float]:
    """Report the labelling grid, returning its lines and the seconds it still needs."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(directory / "generations.jsonl"):
        if record.get("subset") == stage:
            cells[(record["size"], record["mode"])].append(record)
    if not cells:
        return [f"LOCAL LABELLING GRID   no {stage}-stage records under {directory}"], 0.0

    runs = [r for r in read_jsonl(directory / "runs.jsonl") if r.get("stage") == stage]
    target = int(runs[-1]["n_prompts"]) if runs else max(len(v) for v in cells.values())

    sessions = [s for s in read_jsonl(directory / "sessions.jsonl") if s.get("stage") == stage]
    finished = [s for s in sessions if s.get("ended_utc")]
    generated = sum(
        row["wall_s"]
        for session in finished
        for (size, _), group in cells.items()
        if size == session["size"]
        for row in group
    )
    elapsed = sum(s["duration_min"] * 60 for s in finished)
    overhead = elapsed / generated if generated else 1.0

    speed, measured_sizes = throughput(cells, floor=target // 4 or 1)
    per_mode_tokens: dict[str, float] = {}
    for _, mode in CELLS:
        rows = [row for (_, m), group in cells.items() if m == mode for row in group]
        if rows:
            per_mode_tokens[mode] = sum(row["eval_count"] for row in rows) / len(rows)

    lines: list[str] = []
    remaining = 0.0
    for size, mode in CELLS:
        rows = cells[(size, mode)]
        count = len(rows)
        if count >= target:
            each = sum(row["wall_s"] for row in rows) / count
            capped = sum(row["done_reason"] == "length" for row in rows) / count * 100
            state = f"done, {each:5.1f} s each, {capped:4.1f}% capped"
        else:
            if count:
                each = sum(row["wall_s"] for row in rows) / count
                note = "RUNNING"
            elif mode in per_mode_tokens and size in speed:
                each = per_mode_tokens[mode] / speed[size]
                note = "queued "
            else:
                lines.append(f"  {size:5s} {mode:8s} {bar(0, target)} {0:4d}/{target}  not started")
                continue
            left = (target - count) * each * overhead
            remaining += left
            state = f"{note}, {each:5.1f} s each, {clock(left):>6s} left"
        lines.append(f"  {size:5s} {mode:8s} {bar(count, target)} {count:4d}/{target}  {state}")

    total, whole = sum(len(v) for v in cells.values()), target * len(CELLS)
    lines.insert(
        0,
        f"LOCAL LABELLING GRID   {bar(total, whole)} {total}/{whole} responses "
        f"({total / whole * 100:.0f}%)",
    )

    windows = [w for w in read_jsonl(directory / "windows.jsonl") if w.get("stage") == stage]
    if windows:
        closed = datetime.fromisoformat(windows[-1]["timestamp_utc"]).timestamp()
        rejected = sum(not w.get("accepted") for w in windows)
        lines.append(
            f"  last window closed {clock(time.time() - closed)} ago; "
            f"{rejected} of {len(windows)} rejected for other GPU work"
        )
    ordered = dict.fromkeys(size for size, _ in CELLS)
    rates = ", ".join(
        f"{s} {speed[s]:.0f} tok/s{'' if s in measured_sizes else ' (fitted)'}"
        for s in ordered
        if s in speed
    )
    lines.append(f"  decode {rates}; session overhead x{overhead:.2f}")
    return lines, remaining


def hosted(directory: Path, stage: str) -> list[str]:
    """Report the hosted arm's spend and collection, or nothing if it has not run here."""
    ledger = read_jsonl(directory / "ledger.jsonl")
    if not ledger:
        return []
    spent = sum(row.get("actual_usd") or 0.0 for row in ledger)
    cap = max(row.get("cap_usd") or 0.0 for row in ledger)
    waves = {row.get("label") for row in ledger if str(row.get("label", "")).startswith(stage)}
    lines = [f"HOSTED FRONTIER ARM    {bar(spent, cap)} ${spent:.4f} of ${cap:.2f} authorised"]
    summary_path = directory / stage / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lines.append(
            f"  {len(waves)} waves, {summary['responses']} responses over "
            f"{summary['prompts']} prompts, {summary['cap_hits']} cap hits, "
            f"{summary['prompts_dropped']} dropped"
        )
        pairing = "pending the local grid" if summary["pairing"] is None else "computed"
        lines.append(f"  ladder {' -> '.join(summary['ladder'])}; pairing {pairing}")
    return lines


def queue(config: Path, after: float) -> list[str]:
    """Date the stages that follow the grid, given the seconds the grid still needs."""
    if not config.exists():
        return []
    stages = yaml.safe_load(config.read_text(encoding="utf-8"))["queue"]
    lines = ["QUEUED AFTER THE GRID"]
    cursor = time.time() + after
    for stage in stages:
        cursor += float(stage["hours"]) * 3600
        ends = datetime.fromtimestamp(cursor, UTC).astimezone()
        lines.append(
            f"  {stage['name']:41s} {float(stage['hours']):4.1f} h  ends {ends:%a %d %H:%M}"
            f"  {stage['detail']}"
        )
    return lines


def paper(sections: Path) -> list[str]:
    """Report how much of each section still carries a placeholder."""
    if not sections.is_dir():
        return []
    lines, total = ["PAPER"], 0
    for path in sorted(sections.glob("*.tex")):
        text = path.read_text(encoding="utf-8")
        count = text.count("\\tbd{") + text.count("\\decision{")
        total += count
        state = "clear" if not count else f"{count} placeholder(s)"
        lines.append(f"  {path.name:26s} {len(text.splitlines()):4d} lines  {state}")
    lines.append(f"  {total} placeholder(s) left")
    return lines


def board(args: argparse.Namespace) -> str:
    """Assemble the whole board as one string."""
    grid, remaining = labelling(args.labelling, args.stage)
    parts = [f"ARBITER progress   {datetime.now().astimezone():%a %d %b %H:%M %Z}", "", *grid]
    if remaining:
        ends = datetime.fromtimestamp(time.time() + remaining, UTC).astimezone()
        parts.append(f"  GRID FINISHES IN ABOUT {clock(remaining)}, around {ends:%a %d %b %H:%M}")
    for block in (hosted(args.hosted, args.hosted_stage), queue(args.queue, remaining),
                  paper(args.sections)):  # fmt: skip
        if block:
            parts.extend(["", *block])
    return "\n".join(parts)


def main() -> None:
    """Print the board once, or refresh it until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labelling", type=Path, default=REPO_ROOT / "results/raw/labelling")
    parser.add_argument("--stage", default="main", help="labelling subset to report")
    parser.add_argument("--hosted", type=Path, default=REPO_ROOT / "results/processed/hosted_arm")
    parser.add_argument("--hosted-stage", default="main")
    parser.add_argument("--queue", type=Path, default=REPO_ROOT / "configs/progress.yaml")
    parser.add_argument("--sections", type=Path, default=REPO_ROOT / "paper/sections")
    parser.add_argument("--watch", action="store_true", help="refresh until interrupted")
    parser.add_argument("--every", type=int, default=120, help="seconds between refreshes")
    args = parser.parse_args()
    while True:
        print(board(args), flush=True)
        if not args.watch:
            return
        time.sleep(args.every)
        print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    main()
