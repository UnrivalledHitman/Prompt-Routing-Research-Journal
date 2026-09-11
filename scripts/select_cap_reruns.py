"""Choose capped generations to rerun with a longer cap (protocol section 13.1). CPU only.

Takes up to ``--per-cell`` capped generations for each (size, mode), spread evenly over
task families, and writes the (prompt_id, size, mode) list read by the ``cap`` labelling
stage. Logs an upper bound on GPU time, assuming every rerun uses the whole new cap.

Usage::

    python scripts/select_cap_reruns.py results/processed/grades_labelling.csv \
        results/processed/cap_reruns.csv --per-cell 40
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LOGGER = logging.getLogger("arbiter.select_cap_reruns")


def spread_over_families(group: pd.DataFrame, limit: int, rng: np.random.Generator) -> pd.DataFrame:
    """Take up to ``limit`` rows, dealing one family at a time in a seeded order."""
    queues = {
        family: list(rng.permutation(rows.index.to_numpy()))
        for family, rows in sorted(group.groupby("family"))
    }
    chosen: list[int] = []
    while len(chosen) < limit and any(queues.values()):
        for queue in queues.values():
            if queue and len(chosen) < limit:
                chosen.append(queue.pop())
    return group.loc[chosen]


def main() -> None:
    """Select, write and log the rerun list."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grades", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--per-cell", type=int, default=40)
    parser.add_argument("--new-cap", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    grades = pd.read_csv(args.grades)
    base = grades[(grades["sample"] == 0) & (grades["variant"] == "base")]
    capped = base[base["done_reason"] == "length"]
    rng = np.random.default_rng(args.seed)
    parts = [
        spread_over_families(group, args.per_cell, rng)
        for _, group in capped.groupby(["size", "mode"], sort=True)
    ]
    chosen = pd.concat(parts) if parts else capped.iloc[:0]
    chosen = chosen.sort_values(["size", "mode", "prompt_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    chosen[["prompt_id", "size", "mode", "family"]].to_csv(args.out, index=False)

    rate = (base["eval_count"] / base["wall_s"]).groupby(base["size"]).median()
    hours = sum(args.new_cap / rate[size] for size in chosen["size"]) / 3600
    summary = pd.DataFrame(
        {
            "capped": capped.groupby(["size", "mode"]).size(),
            "generations": base.groupby(["size", "mode"]).size(),
            "chosen": chosen.groupby(["size", "mode"]).size(),
        }
    ).fillna(0)
    summary["cap_rate"] = summary["capped"] / summary["generations"]
    LOGGER.info(
        "%d cells -> %s; at most %.1f GPU hours\n%s",
        len(chosen),
        args.out,
        hours,
        summary.round(3).to_string(),
    )


if __name__ == "__main__":
    main()
