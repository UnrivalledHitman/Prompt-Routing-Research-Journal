"""Attribute window energy to generations for one labelling stage. CPU only.

Writes a text-free CSV with one row per accepted generation and logs a summary by
model and mode.

Usage::

    python scripts/attribute_energy.py results/raw/labelling_smoke \
        results/processed/energy_smoke.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from arbiter.energy.attribution import attribute_stage  # noqa: E402

LOGGER = logging.getLogger("arbiter.attribute_energy")


def main() -> None:
    """Attribute, write and summarise."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    frame = pd.DataFrame(attribute_stage(args.raw_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby(["size", "mode"]).agg(
        n=("prompt_id", "size"),
        wall_s=("wall_s", "mean"),
        board_j=("energy_board_j", "mean"),
        dyn_j=("energy_dyn_j", "mean"),
        baseline_w=("baseline_w", "mean"),
        contaminated=("contaminated", "sum"),
    )
    LOGGER.info("%d generations -> %s\n%s", len(frame), args.out, summary.round(2).to_string())


if __name__ == "__main__":
    main()
