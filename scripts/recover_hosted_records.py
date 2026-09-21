"""Rebuild the hosted arm's records from its settled batches. No spend, no ledger write.

The hosted arm's raw responses are gitignored, so a checkout without them cannot regrade
the arm (protocol section 13.8). Every batch in the ledger is already paid for and settled,
and downloading a finished batch's output file again costs nothing, so the records are
rebuilt from the batch ids the ledger records. The ledger itself is read only: each batch's
recomputed cost is compared with the settled figure and any difference is reported.

Usage::

    python scripts/recover_hosted_records.py --stage main --dry-run
    python scripts/recover_hosted_records.py --stage main
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from arbiter.hosted import batch as hb  # noqa: E402
from arbiter.hosted.pricing import price_table  # noqa: E402
from arbiter.utils.jsonl import append_jsonl  # noqa: E402

LOGGER = logging.getLogger("arbiter.recover_hosted")


def settled_batches(ledger: Path, stage: str) -> list[dict]:
    """Return the ledger's settled entries for a stage, in submission order."""
    entries = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return [e for e in entries if e["stage"] == stage and e.get("actual_usd") is not None]


def main() -> None:
    """Download each settled batch of a stage and rebuild its records file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", default="main")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "hosted_arm.yaml")
    parser.add_argument("--dry-run", action="store_true", help="list the batches and stop")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ledger = REPO_ROOT / cfg["budget"]["ledger"]
    settled = settled_batches(ledger, args.stage)
    LOGGER.info("%d settled batches for stage %s in %s", len(settled), args.stage, ledger.name)
    if args.dry_run:
        for entry in settled:
            LOGGER.info("  %s %s $%.4f", entry["batch_id"], entry.get("model"), entry["actual_usd"])
        return

    from openai import OpenAI

    raw_dir = REPO_ROOT / cfg["raw_dir"]
    path = raw_dir / f"{args.stage}_records.jsonl"
    if path.exists():
        raise SystemExit(f"{path} already exists; move it aside before rebuilding")
    client = OpenAI(api_key=hb.api_key(*hb.dotenv_paths(REPO_ROOT)))
    cells = {c.cell: c for c in hb.cells_from_config(cfg["cells"])}
    prices = price_table(cfg["pricing"], batch=True)
    written, mismatches = 0, 0
    for entry in settled:
        job = client.batches.retrieve(entry["batch_id"])
        if job.status != "completed":
            LOGGER.error("batch %s is %s; skipped", entry["batch_id"], job.status)
            mismatches += 1
            continue
        records, totals = hb.collect(
            client,
            job,
            cells,
            prices,
            raw_dir=raw_dir,
            label=f"{entry.get('label', args.stage)}_{entry.get('model', 'all')}",
        )
        append_jsonl(path, records)
        written += len(records)
        drift = totals["actual_usd"] - float(entry["actual_usd"])
        if abs(drift) > 1e-6:
            mismatches += 1
            LOGGER.warning(
                "%s: recomputed $%.6f against settled $%.6f (difference $%.6f)",
                entry["batch_id"], totals["actual_usd"], entry["actual_usd"], drift,
            )  # fmt: skip
        LOGGER.info(
            "%s %s: %d responses, %d errors, $%.4f",
            entry["batch_id"], entry.get("model"), totals["responses"], totals["errors"],
            totals["actual_usd"],
        )  # fmt: skip
    LOGGER.info("%d records -> %s; %d batches differed from the ledger", written, path, mismatches)
    LOGGER.info("the ledger was not modified")


if __name__ == "__main__":
    main()
