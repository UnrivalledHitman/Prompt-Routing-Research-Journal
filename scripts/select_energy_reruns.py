"""List the generations whose energy was excluded, for rerunning on mains power. CPU only.

Reads an attributed energy table and writes the (prompt_id, size, mode) cells read by the
``energy_rerun`` labelling stage (protocol section 13.7). The rerun keeps the original
sample index and variant, so every generation gets the seed it had the first time.

Usage::

    python scripts/select_energy_reruns.py results/processed/energy_labelling.csv \
        results/processed/energy_reruns.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

LOGGER = logging.getLogger("arbiter.select_energy_reruns")
COLUMNS = ["prompt_id", "size", "mode", "sample", "variant", "window_id", "energy_excluded"]


def excluded_cells(energy: pd.DataFrame) -> pd.DataFrame:
    """Return the excluded generations, one row per cell, sorted.

    Raises
    ------
    ValueError
        If an excluded generation is not sample 0 of the base variant, which the rerun
        stage could not reproduce.
    """
    excluded = energy[energy["energy_excluded"].fillna("") != ""]
    odd = excluded[(excluded["sample"] != 0) | (excluded["variant"] != "base")]
    if not odd.empty:
        raise ValueError(f"cannot rerun non-base generations: {odd['window_id'].unique()[:5]}")
    return excluded[COLUMNS].sort_values(["size", "mode", "prompt_id"]).reset_index(drop=True)


def main() -> None:
    """Select and write the rerun list."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("energy", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    cells = excluded_cells(pd.read_csv(args.energy, keep_default_na=False, na_values=[""]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.out, index=False)
    LOGGER.info(
        "%d cells over %d prompts -> %s\n%s",
        len(cells),
        cells["prompt_id"].nunique(),
        args.out,
        cells.groupby(["size", "mode", "energy_excluded"]).size().to_string(),
    )


if __name__ == "__main__":
    main()
