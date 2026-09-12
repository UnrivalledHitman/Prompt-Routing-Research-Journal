"""Recompute the escalation tax on the public benchmark. CPU only, no inference.

Reads the compact record table written by ``scripts/build_public_arm.py`` and writes the
pool summary, the minimum-sufficient-model distribution, the policy table and the
thinking/non-thinking comparison to the output directory, plus ``summary.json``.

Usage::

    python scripts/build_public_arm.py --config configs/public_arm.yaml
    python scripts/run_public_arm.py --config configs/public_arm.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from build_public_arm import low_priority  # noqa: E402

from arbiter.eval import public  # noqa: E402

LOGGER = logging.getLogger("arbiter.run_public_arm")
HEADLINE = ("P1", "P3", "random", "oracle")


def write(out: Path, name: str, frame: pd.DataFrame) -> None:
    """Write one results table and log its size."""
    frame.to_csv(out / f"{name}.csv", index=False)
    LOGGER.info("wrote %s.csv (%d rows)", name, len(frame))


def main() -> None:
    """Build the table, run the analyses and write them out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "public_arm.yaml")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=None, help="override the config")
    parser.add_argument("--low-priority", action="store_true", help="yield CPU to a GPU run")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    if args.low_priority:
        low_priority()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    boot = cfg["bootstrap"]
    n_boot = args.n_boot or boot["n_boot"]
    kwargs = {"n_boot": n_boot, "seed": boot["seed"], "level": boot["level"]}
    out = args.out or REPO_ROOT / cfg["paths"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(REPO_ROOT / cfg["paths"]["records"])
    units = tuple(cfg["units"])

    table = public.build_table(records, cfg["pool"], cfg["success"]["primary"])
    LOGGER.info("%d queries, %d models in cost order", len(table), len(table.models))

    pool = public.pool_summary(table, units)
    write(out, "pool", pool)
    ladder = pool[(pool["scope"] == "all") & pool["on_ladder"]]["model"].tolist()
    LOGGER.info("ladder: %s", " -> ".join(ladder))
    write(out, "cstar", public.cstar_distribution(table, units))
    taxes = public.tax_table(table, units, **kwargs)
    write(out, "tax_policies", taxes)

    pair = cfg["mode_pair"]
    write(
        out, "mode_pair", public.mode_pair(table, pair["thinking"], pair["plain"], units, **kwargs)
    )

    lenient = public.build_table(records, cfg["pool"], cfg["success"]["sensitivity"])
    write(out, "tax_policies_lenient", public.tax_table(lenient, units, **kwargs))

    summary: dict[str, Any] = {
        "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "queries": len(table),
        "datasets": sorted(set(table.queries["dataset"])),
        "models": list(table.models),
        "ladder": ladder,
        "success_threshold": cfg["success"]["primary"],
        "sensitivity_threshold": cfg["success"]["sensitivity"],
        "n_boot": n_boot,
        "units": list(units),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    headline = taxes[
        (taxes["scope"] == "all") & (taxes["unit"] == "dollars") & taxes["policy"].isin(HEADLINE)
    ]
    columns = ["policy", "mean_cost", "solved_rate", "tax", "relative_tax"]
    LOGGER.info("done -> %s\n%s", out, headline[columns].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
