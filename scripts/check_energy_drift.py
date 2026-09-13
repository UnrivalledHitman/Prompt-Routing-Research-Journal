"""Compare energy per generated token before and after a driver update. CPU only.

The noise stage resamples prompts and cells that the main run measured, and it ran
after the GPU driver was updated, so the first sample and the later samples of the
same (prompt, cell) pairs sit on either side of the change.

Usage::

    python scripts/check_energy_drift.py results/processed/grades_labelling.csv \
        results/processed/energy_labelling.csv results/processed/energy_drift.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from arbiter.eval.robustness import energy_drift  # noqa: E402

LOGGER = logging.getLogger("arbiter.check_energy_drift")


def main() -> None:
    """Compute, write and log the per-cell comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grades", type=Path)
    parser.add_argument("energy", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--before", type=int, nargs="+", default=[0])
    parser.add_argument("--after", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    result = energy_drift(
        pd.read_csv(args.grades),
        pd.read_csv(args.energy),
        before=args.before,
        after=args.after,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    for row in result.itertuples():
        LOGGER.info(
            "%s (%s %s): %d prompts, J/token %.3f -> %.3f, ratio %.3f [%.3f, %.3f], "
            "baseline %.2f -> %.2f W",
            row.cell,
            row.size,
            row.mode,
            row.n_prompts,
            row.j_per_token_before,
            row.j_per_token_after,
            row.ratio,
            row.ratio_lo,
            row.ratio_hi,
            row.baseline_w_before,
            row.baseline_w_after,
        )


if __name__ == "__main__":
    main()
