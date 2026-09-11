"""Measure the open-ended judge's agreement and apply the reporting rule. CPU only.

Compares the local judge with the frontier model on the frontier sample, and both with
the first author's blind labels on the human sample (experimental protocol, section
5.4). The open-ended family's status follows the rule fixed in
:mod:`arbiter.eval.agreement`, and stays ``pending`` until every sheet row is scored.
Writes a text-free JSON summary.

Usage::

    python scripts/judge_agreement.py results/raw/labelling \
        results/processed/judge_agreement.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arbiter.eval.agreement import (  # noqa: E402
    CAVEAT_KAPPA,
    PRIMARY_KAPPA,
    agreement_summary,
    open_family_status,
)
from arbiter.eval.judge import KEY_FIELDS  # noqa: E402
from arbiter.utils.jsonl import read_jsonl  # noqa: E402

LOGGER = logging.getLogger("arbiter.judge_agreement")
Key = tuple[Any, ...]


def read_scores(path: Path) -> dict[Key, int]:
    """Read the readable scores in a judgement file, by generation."""
    if not path.exists():
        return {}
    records, _ = read_jsonl(path)
    return {tuple(r[k] for k in KEY_FIELDS): r["score"] for r in records if r["score"] is not None}


def read_human(out_dir: Path) -> tuple[dict[Key, int], int]:
    """Read the human scores and join them to generations through the key file.

    Returns
    -------
    scores : dict
        Score by generation, for the rows scored so far.
    rows : int
        Rows on the sheet.

    Raises
    ------
    SystemExit
        If a score is not an integer from 1 to 5.
    """
    sheet_path = out_dir / "human_sheet.csv"
    if not sheet_path.exists():
        return {}, 0
    with (out_dir / "human_key.csv").open(encoding="utf-8-sig", newline="") as fh:
        keys = {row["item"]: row for row in csv.DictReader(fh)}
    with sheet_path.open(encoding="utf-8-sig", newline="") as fh:
        sheet = list(csv.DictReader(fh))
    scores: dict[Key, int] = {}
    for row in sheet:
        value = (row["score"] or "").strip()
        if not value:
            continue
        if value not in {"1", "2", "3", "4", "5"}:
            raise SystemExit(f"{row['item']}: score {value!r} is not an integer from 1 to 5")
        k = keys[row["item"]]
        key = (k["prompt_id"], k["tag"], k["mode"], int(k["sample"]), k["variant"])
        scores[key] = int(value)
    return scores, len(sheet)


def paired(a: dict[Key, int], b: dict[Key, int]) -> tuple[list[int], list[int]]:
    """Return the two raters' scores on the generations both scored."""
    keys = sorted(a.keys() & b.keys())
    return [a[key] for key in keys], [b[key] for key in keys]


def finite(value: Any) -> Any:  # noqa: ANN401
    """Replace NaN and infinities with None, recursively, so the JSON is valid."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    return value


def main() -> None:
    """Compute agreement, apply the rule and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    out_dir = args.raw_dir / "judge_validation"
    local = read_scores(args.raw_dir / "judgements.jsonl")
    frontier = read_scores(out_dir / "frontier_judgements.jsonl")
    human, sheet_rows = read_human(out_dir)
    summary: dict[str, Any] = {
        "rule": {
            "primary_kappa": PRIMARY_KAPPA,
            "caveat_kappa": CAVEAT_KAPPA,
            "fixed": "2026-09-12",
        },
        "human_sheet_rows": sheet_rows,
        "human_scored": len(human),
    }
    pairs = {
        "local_vs_frontier": (local, frontier),
        "local_vs_human": (local, human),
        "frontier_vs_human": (frontier, human),
    }
    for name, (a, b) in pairs.items():
        xs, ys = paired(a, b)
        summary[name] = agreement_summary(xs, ys, args.n_boot, args.seed) if xs else None
    status = "pending"
    if sheet_rows and len(human) == sheet_rows and summary["local_vs_human"]:
        kappa = summary["local_vs_human"]["kappa_binary"]
        status = open_family_status(kappa) if math.isfinite(kappa) else "undefined"
    summary["open_family_status"] = status
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(finite(summary), indent=2), encoding="utf-8")
    LOGGER.info("%s\n-> %s", json.dumps(finite(summary), indent=2), args.out)


if __name__ == "__main__":
    main()
