"""Parse probe generations into a text-free table for R3 or the cascade. CPU only.

Each row holds the parsed answer (label probabilities or a verification confidence),
token counts and wall time, joined with the energy attributed by
``scripts/attribute_energy.py`` when ``--energy`` is given.

Usage::

    python scripts/collect_probes.py verify results/raw/verify results/processed/probes_verify.csv
    python scripts/collect_probes.py route results/raw/route results/processed/probes_route.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from arbiter.eval import probes  # noqa: E402
from arbiter.utils.jsonl import read_jsonl  # noqa: E402

LOGGER = logging.getLogger("arbiter.collect_probes")
KEYS = ["tag", "mode", "sample", "variant"]


def main() -> None:
    """Parse, join energy, write and summarise."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("verify", "route"))
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--energy", type=Path, help="attribute_energy.py output for raw_dir")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    records, truncated = read_jsonl(args.raw_dir / "generations.jsonl")
    if truncated:
        LOGGER.warning("ignored a truncated last line")
    frame = pd.DataFrame([probes.probe_row(record, args.kind) for record in records])
    if args.energy is not None:
        energy = pd.read_csv(args.energy).rename(columns={"prompt_id": "probe_id"})
        frame = frame.merge(
            energy[["probe_id", *KEYS, "energy_dyn_j", "energy_board_j"]],
            on=["probe_id", *KEYS],
            how="left",
            validate="one_to_one",
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby(["variant", "size"]).agg(
        n=("probe_id", "size"),
        parsed=("parsed", "mean"),
        candidate_mass=("candidate_mass", "mean"),
        wall_s=("wall_s", "median"),
        prompt_tokens=("prompt_eval_count", "median"),
    )
    LOGGER.info("%d probes -> %s\n%s", len(frame), args.out, summary.round(3).to_string())


if __name__ == "__main__":
    main()
