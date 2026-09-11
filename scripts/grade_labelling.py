"""Grade every generation of a labelling stage. CPU only.

Writes a text-free CSV of outcomes to ``results/processed`` and an audit JSONL with
the extracted answers next to the raw generations (gitignored). Code is executed in
worker threads, each running its own interpreter. Open-ended answers are graded from
the judge's scores (``scripts/run_judge.py``), which must exist for every answer that
needs one unless the open-ended family is left out with ``--skip-open``.

Usage::

    python scripts/grade_labelling.py results/raw/labelling_pilot \
        results/processed/grades_pilot.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from arbiter.data.corpus import PromptItem, read_corpus  # noqa: E402
from arbiter.eval.grading import grade, open_precheck  # noqa: E402
from arbiter.utils.jsonl import read_jsonl  # noqa: E402

LOGGER = logging.getLogger("arbiter.grade_labelling")
CARRIED = (
    "prompt_id", "family", "source", "subset", "size", "tag", "mode", "sample", "variant",
    "done_reason", "eval_count", "prompt_eval_count", "tokens_thinking", "tokens_content",
    "wall_s", "window_id", "contaminated",
)  # fmt: skip
KEY = ("prompt_id", "tag", "mode", "sample", "variant")


def record_id(record: dict[str, Any]) -> tuple[Any, ...]:
    """Identify a generation by prompt, model, mode, sample and variant."""
    return tuple(record.get(key) for key in KEY)


def grade_row(
    record: dict[str, Any], item: PromptItem, judgement: dict[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Grade one record; return the text-free row and the audit entry."""
    result = grade(record, item, judgement)
    row = {key: record.get(key) for key in CARRIED}
    row.update(
        correct=result.correct,
        reason=result.reason,
        judge_score=judgement.get("score") if judgement else None,
    )
    audit = {key: record.get(key) for key in KEY} | {
        "correct": result.correct,
        "reason": result.reason,
        "extracted": result.extracted,
        "reference": item.reference,
    }
    if judgement:
        audit["judge_reason"] = judgement.get("reason")
    return row, audit


def main() -> None:
    """Grade, write and summarise."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "data/processed/corpus.jsonl")
    parser.add_argument(
        "--judgements", type=Path, help="judge scores (default: RAW_DIR/judgements.jsonl)"
    )
    parser.add_argument(
        "--skip-open", action="store_true", help="leave the open-ended family out entirely"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    items = {item.prompt_id: item for item in read_corpus(args.corpus)}
    records, _ = read_jsonl(args.raw_dir / "generations.jsonl")
    if args.skip_open:
        records = [r for r in records if items[r["prompt_id"]].family != "open"]
    judgement_path = args.judgements or args.raw_dir / "judgements.jsonl"
    judged = read_jsonl(judgement_path)[0] if judgement_path.exists() else []
    judgements = {record_id(j): j for j in judged}
    unjudged = sum(
        items[r["prompt_id"]].family == "open"
        and open_precheck(r) is None
        and record_id(r) not in judgements
        for r in records
    )
    if unjudged:
        raise SystemExit(
            f"{unjudged} open-ended answers have no judgement in {judgement_path}; "
            "run scripts/run_judge.py first, or pass --skip-open"
        )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(
                lambda r: grade_row(r, items[r["prompt_id"]], judgements.get(record_id(r))),
                records,
            )
        )
    frame = pd.DataFrame([row for row, _ in results])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    with (args.raw_dir / "grades_audit.jsonl").open("w", encoding="utf-8") as fh:
        for _, audit in results:
            fh.write(json.dumps(audit, ensure_ascii=False) + "\n")
    if frame.empty:
        LOGGER.warning("no generations in %s", args.raw_dir)
        return
    accuracy = frame.pivot_table(
        index="family", columns=["size", "mode"], values="correct", aggfunc="mean"
    )
    reasons = frame.groupby(["family", "reason"]).size().unstack(fill_value=0)
    LOGGER.info(
        "%d generations -> %s\naccuracy\n%s\nreasons\n%s",
        len(frame),
        args.out,
        accuracy.round(2).to_string(),
        reasons.to_string(),
    )


if __name__ == "__main__":
    main()
