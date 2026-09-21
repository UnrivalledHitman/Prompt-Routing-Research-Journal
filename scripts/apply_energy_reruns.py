"""Replace excluded energy rows with the same generations rerun on mains power. CPU only.

Reads the attributed energy table and the attributed ``energy_rerun`` stage, writes the
merged table (protocol section 13.7), and logs how many rerun answers match the original
text exactly: the seeds are the same, so a low share would point at a changed setup.

Usage::

    python scripts/apply_energy_reruns.py results/processed/energy_labelling.csv \
        results/processed/energy_rerun.csv results/processed/energy_labelling.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from arbiter.energy.attribution import KEY, apply_reruns  # noqa: E402
from arbiter.utils.jsonl import read_jsonl  # noqa: E402

LOGGER = logging.getLogger("arbiter.apply_energy_reruns")


def read_energy(path: Path) -> pd.DataFrame:
    """Read an energy table, keeping empty exclusion reasons as empty strings."""
    frame = pd.read_csv(path)
    for column in ("energy_excluded", "energy_replaced"):
        if column in frame.columns:
            frame[column] = frame[column].fillna("")
    return frame


def text_agreement(original_raw: Path, rerun_raw: Path) -> tuple[int, int]:
    """Return (identical, compared) answers between the rerun and the original generations."""
    originals = {
        tuple(r[k] for k in KEY): (r["thinking"], r["content"])
        for r in read_jsonl(original_raw / "generations.jsonl")[0]
    }
    reruns, _ = read_jsonl(rerun_raw / "generations.jsonl")
    pairs = [
        (originals[key], (r["thinking"], r["content"]))
        for r in reruns
        if (key := tuple(r[k] for k in KEY)) in originals
    ]
    return sum(first == again for first, again in pairs), len(pairs)


def main() -> None:
    """Merge, write and log."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("energy", type=Path)
    parser.add_argument("reruns", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--original-raw", type=Path, default=REPO_ROOT / "results/raw/labelling")
    parser.add_argument("--rerun-raw", type=Path, default=REPO_ROOT / "results/raw/energy_rerun")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    merged = apply_reruns(read_energy(args.energy), read_energy(args.reruns))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    if (args.rerun_raw / "generations.jsonl").exists():
        identical, compared = text_agreement(args.original_raw, args.rerun_raw)
        LOGGER.info("rerun answers identical to the original: %d of %d", identical, compared)
    LOGGER.info("%d rows -> %s", len(merged), args.out)


if __name__ == "__main__":
    main()
