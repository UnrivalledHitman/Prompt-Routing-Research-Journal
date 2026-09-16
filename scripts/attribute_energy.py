"""Attribute window energy to generations for one labelling stage. CPU only.

Writes a text-free CSV with one row per accepted generation and logs a summary by
model and mode. Windows whose energy cannot be used (a telemetry gap, or a period
without mains power listed in ``configs/power_source.yaml``) keep their rows with
NaN energy and the reason in ``energy_excluded``.

Usage::

    python scripts/attribute_energy.py results/raw/labelling_smoke \
        results/processed/energy_smoke.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from arbiter.energy.attribution import MAX_GAP_S, attribute_stage  # noqa: E402

LOGGER = logging.getLogger("arbiter.attribute_energy")
POWER_SOURCE = REPO_ROOT / "configs" / "power_source.yaml"


def load_on_battery(path: Path) -> list[tuple[float, float]]:
    """Read periods without mains power as Unix-time intervals; none if the file is absent.

    Raises
    ------
    ValueError
        If a period ends before it starts or a time has no timezone.
    """
    if not path.exists():
        return []
    periods = yaml.safe_load(path.read_text(encoding="utf-8")).get("on_battery_utc") or []
    out = []
    for period in periods:
        since, until = (datetime.fromisoformat(period[k]) for k in ("from", "to"))
        if since.tzinfo is None or until.tzinfo is None:
            raise ValueError(f"{path}: times need a timezone: {period}")
        if until < since:
            raise ValueError(f"{path}: period ends before it starts: {period}")
        out.append((since.timestamp(), until.timestamp()))
    return out


def main() -> None:
    """Attribute, write and summarise."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--power-source", type=Path, default=POWER_SOURCE)
    parser.add_argument("--max-gap-s", type=float, default=MAX_GAP_S)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    on_battery = load_on_battery(args.power_source)
    LOGGER.info("%d periods without mains power from %s", len(on_battery), args.power_source)
    frame = pd.DataFrame(attribute_stage(args.raw_dir, on_battery, args.max_gap_s))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby(["size", "mode"]).agg(
        n=("prompt_id", "size"),
        wall_s=("wall_s", "mean"),
        board_j=("energy_board_j", "mean"),
        dyn_j=("energy_dyn_j", "mean"),
        baseline_w=("baseline_w", "mean"),
        contaminated=("contaminated", "sum"),
        excluded=("energy_excluded", lambda s: int((s != "").sum())),
    )
    LOGGER.info("%d generations -> %s\n%s", len(frame), args.out, summary.round(2).to_string())


if __name__ == "__main__":
    main()
