"""Draw the judge-validation samples from judged open-ended answers. CPU only.

About 200 judged answers, stratified by Dolly category and the local judge's verdict,
go to the frontier-model check, and about 100 of those, stratified the same way, to
blind human labelling (experimental protocol, section 5.4). Samples are drawn once, and
a sample whose human sheet has any score is never redrawn.

Everything is written under ``RAW_DIR/judge_validation`` (gitignored):

* ``frontier_items.jsonl``: the answers for ``scripts/judge_openai.py``;
* ``human_sheet.csv``: request, reference and answer, in a shuffled order, with an empty
  ``score`` column and no model, mode or judge score;
* ``human_key.csv``: the generation behind each sheet row, kept apart from the sheet;
* ``rubric.txt``: the judge's rubric, for the human labeller;
* ``sample_info.json``: seed and stratum counts.

Usage::

    python scripts/sample_judge_validation.py results/raw/labelling
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from arbiter.data.corpus import read_corpus  # noqa: E402
from arbiter.eval.agreement import stratified_sample  # noqa: E402
from arbiter.eval.grading import JUDGE_SUCCESS_SCORE, final_text  # noqa: E402
from arbiter.eval.judge import KEY_FIELDS, RUBRIC  # noqa: E402
from arbiter.utils.jsonl import read_jsonl  # noqa: E402

LOGGER = logging.getLogger("arbiter.sample_judge_validation")
SEED = 20260912
HUMAN_NOTE = (
    "Score every row of human_sheet.csv from 1 to 5 in its score column, using the rubric "
    "above. The sheet does not show which model or mode wrote an answer, or the judge's "
    "score. Do not open human_key.csv until every row is scored."
)


def key_of(record: dict[str, Any]) -> tuple[Any, ...]:
    """Identify a generation by prompt, model, mode, sample and variant."""
    return tuple(record[key] for key in KEY_FIELDS)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV with a byte-order mark, so spreadsheet programs read UTF-8."""
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sheet_has_scores(path: Path) -> bool:
    """Return True if a human sheet exists and any row has a score."""
    if not path.exists():
        return False
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return any((row.get("score") or "").strip() for row in csv.DictReader(fh))


def main() -> None:
    """Draw both samples and write them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "data/processed/corpus.jsonl")
    parser.add_argument("--frontier", type=int, default=200)
    parser.add_argument("--human", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="redraw an unlabelled sample")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    out_dir = args.raw_dir / "judge_validation"
    if sheet_has_scores(out_dir / "human_sheet.csv"):
        raise SystemExit(f"{out_dir}: the human sheet already has scores; not redrawing")
    if (out_dir / "frontier_items.jsonl").exists() and not args.force:
        raise SystemExit(f"{out_dir} already holds a sample; pass --force to redraw it")

    items = {item.prompt_id: item for item in read_corpus(args.corpus)}
    generations = {key_of(r): r for r in read_jsonl(args.raw_dir / "generations.jsonl")[0]}
    judged = sorted(
        (r for r in read_jsonl(args.raw_dir / "judgements.jsonl")[0] if r["score"] is not None),
        key=key_of,
    )
    verdicts = ["pass" if r["score"] >= JUDGE_SUCCESS_SCORE else "fail" for r in judged]
    strata = [f"{r['category']}|{v}" for r, v in zip(judged, verdicts, strict=True)]
    frontier_idx = stratified_sample(strata, args.frontier, SEED, "frontier")
    human_pos = stratified_sample([strata[i] for i in frontier_idx], args.human, SEED, "human")

    rows = []
    for i in frontier_idx:
        record = judged[i]
        item = items[record["prompt_id"]]
        rows.append(
            {key: record[key] for key in KEY_FIELDS}
            | {
                "category": record["category"],
                "request": item.text,
                "reference": item.reference,
                "answer": final_text(generations[key_of(record)]),
            }
        )
    human = [rows[p] for p in human_pos]
    random.Random(f"{SEED}|human-order").shuffle(human)
    ids = [f"h{n:03d}" for n in range(1, len(human) + 1)]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frontier_items.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    write_csv(
        out_dir / "human_sheet.csv",
        ["item", "request", "reference", "answer", "score"],
        [
            {"item": n, "request": r["request"], "reference": r["reference"]}
            | {"answer": r["answer"], "score": ""}
            for n, r in zip(ids, human, strict=True)
        ],
    )
    write_csv(
        out_dir / "human_key.csv",
        ["item", *KEY_FIELDS],
        [
            {"item": n} | {key: r[key] for key in KEY_FIELDS}
            for n, r in zip(ids, human, strict=True)
        ],
    )
    (out_dir / "rubric.txt").write_text(RUBRIC + "\n\n" + HUMAN_NOTE + "\n", encoding="utf-8")
    info = {
        "seed": SEED,
        "judged": len(judged),
        "frontier": len(rows),
        "human": len(human),
        "strata_judged": dict(sorted(Counter(strata).items())),
        "strata_frontier": dict(sorted(Counter(strata[i] for i in frontier_idx).items())),
        "strata_human": dict(sorted(Counter(strata[frontier_idx[p]] for p in human_pos).items())),
    }
    (out_dir / "sample_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    LOGGER.info(
        "%d judged answers; %d drawn for the frontier check, %d of them for human labels -> %s",
        len(judged),
        len(rows),
        len(human),
        out_dir,
    )


if __name__ == "__main__":
    main()
