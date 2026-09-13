"""Build the probe corpora for the cascade verifier and the R3 router. CPU only.

Writes a gitignored JSONL corpus and a cell list under ``--out-dir`` for each probe
stage in ``configs/labelling.yaml``:

* ``verify``: ``verify_corpus.jsonl`` and ``verify_cells.csv``, from the recorded
  no-think answers on the P1 ladder below c8;
* ``route``: ``route_zero_corpus.jsonl`` and ``route_zero_cells.csv``; with ``--splits``
  also the few-shot pair, whose examples come from the training split.

Usage::

    python scripts/build_probes.py verify
    python scripts/build_probes.py route --splits results/processed/analysis/main/splits.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from arbiter.data.corpus import PromptItem, read_corpus  # noqa: E402
from arbiter.eval import probes  # noqa: E402
from arbiter.eval.labels import CELL_NAMES  # noqa: E402
from arbiter.utils.jsonl import read_jsonl  # noqa: E402

LOGGER = logging.getLogger("arbiter.build_probes")


def write(
    out_dir: Path, name: str, items: Sequence[PromptItem], cells: Sequence[probes.Cell]
) -> None:
    """Write one probe corpus and its cell list."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}_corpus.jsonl").open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    with (out_dir / f"{name}_cells.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["prompt_id", "size", "mode"])
        writer.writerows(cells)
    lengths = [len(item.text) for item in items]
    LOGGER.info(
        "%s: %d probes; prompt characters median %.0f, max %d",
        name,
        len(items),
        statistics.median(lengths) if lengths else 0.0,
        max(lengths, default=0),
    )


def main() -> None:
    """Build the requested probe corpora."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("verify", "route"))
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "data/processed/corpus.jsonl")
    parser.add_argument("--subsets", nargs="+", default=["pilot", "main"])
    parser.add_argument(
        "--generations",
        type=Path,
        default=REPO_ROOT / "results/raw/labelling/generations.jsonl",
    )
    parser.add_argument("--splits", type=Path, help="split assignment from run_analysis.py")
    parser.add_argument("--label-unit", default="energy_j")
    parser.add_argument("--per-cell", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results/raw/probes")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    corpus = read_corpus(args.corpus)
    items = [item for item in corpus if item.subset in args.subsets]
    if not items:
        raise SystemExit(f"{args.corpus} holds no prompts in subsets {args.subsets}")
    if args.kind == "verify":
        if not args.generations.exists():
            raise SystemExit(f"{args.generations} does not exist; run the labelling stage first")
        records, truncated = read_jsonl(args.generations)
        if truncated:
            LOGGER.warning("ignored a truncated last line in %s", args.generations)
        built, cells = probes.verify_items(items, records)
        # An empty corpus would be written without complaint and then labelled for nothing,
        # so say which input came up short rather than reporting "0 probes" as a result.
        if not built:
            raise SystemExit(
                f"no verification probes from {args.generations}: it holds no base "
                f"no-think answers for subsets {args.subsets}"
            )
        write(args.out_dir, "verify", built, cells)
        return
    write(args.out_dir, "route_zero", *probes.route_items(items))
    if args.splits is None:
        LOGGER.info("no --splits given; few-shot routing probes not built")
        return
    splits = pd.read_csv(args.splits)
    texts = {item.prompt_id: item.text for item in corpus}
    column = f"cstar_{args.label_unit}"
    examples = probes.pick_examples(splits, texts, column, args.per_cell, args.seed)
    LOGGER.info("few-shot examples for cells %s", [CELL_NAMES[cell] for _, cell in examples])
    analysed = set(splits["prompt_id"])
    few = [item for item in items if item.prompt_id in analysed]
    write(args.out_dir, "route_few", *probes.route_items(few, examples))


if __name__ == "__main__":
    main()
