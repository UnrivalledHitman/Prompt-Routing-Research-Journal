"""Grade the hosted arm's responses and compute the escalation tax on its ladder.

Reads the records written by ``scripts/run_hosted_arm.py collect``, grades them with the
same graders the local grid uses, and reuses :mod:`arbiter.eval.public` so the pool
summary, the minimum-sufficient distribution, the policy table and the mode comparison are
computed by exactly the code that produced them for the public benchmark.

Usage::

    python scripts/run_hosted_analysis.py --stage pilot
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

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from arbiter.data.corpus import PromptItem, read_corpus  # noqa: E402
from arbiter.eval import public  # noqa: E402
from arbiter.eval.grading import grade  # noqa: E402
from arbiter.hosted.pricing import cost_of, price_table  # noqa: E402
from arbiter.utils.jsonl import read_jsonl  # noqa: E402

LOGGER = logging.getLogger("arbiter.hosted_analysis")


def graded_frame(
    records: list[dict[str, Any]],
    items: dict[str, PromptItem],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Grade every response and return the record frame :mod:`public` expects.

    The cap rule lives in the grader, not here: :func:`arbiter.eval.grading.grade` reads
    ``done_reason`` and fails a truncated generation outright, which is why the hosted record
    carries the local field names. A request that errored has no answer at all and is
    likewise not a success. Prompts with a failed or missing cell are dropped whole, so the
    success matrix has no holes and the pool stays fixed across queries.
    """
    prices = price_table(cfg["pricing"], batch=True)
    rows: list[dict[str, Any]] = []
    notes = {"graded": 0, "capped": 0, "errors": 0}
    for record in records:
        item = items.get(record["prompt_id"])
        if item is None:
            raise ValueError(f"prompt {record['prompt_id']} is not in the corpus")
        capped = record["done_reason"] == "length"
        failed = record["error"] is not None
        correct = False if failed else bool(grade(record, item).correct)
        notes["graded"] += 1
        notes["capped"] += int(capped)
        notes["errors"] += int(failed)
        rows.append(
            {
                "dataset": item.family,
                "index": record["prompt_id"],
                "model": record["cell"],
                "score": float(correct),
                "prompt_tokens": record["input_tokens"] + record["cached_tokens"],
                "completion_tokens": record["output_tokens"],
                "reasoning_tokens": record["reasoning_tokens"],
                "cost": cost_of(record, prices[record["model"]]),
                "capped": capped,
                "failed": failed,
            }
        )
    frame = pd.DataFrame(rows)
    cells = [row["cell"] for row in cfg["cells"]]
    complete = frame.groupby("index")["model"].nunique() == len(cells)
    broken = set(frame.loc[frame["failed"], "index"])
    keep = set(complete[complete].index) - broken
    notes["prompts_seen"] = int(frame["index"].nunique())
    notes["prompts_kept"] = len(keep)
    if dropped := notes["prompts_seen"] - notes["prompts_kept"]:
        LOGGER.warning("dropped %d prompt(s) with a missing or failed cell", dropped)
    return frame[frame["index"].isin(keep)].reset_index(drop=True), notes


def cell_report(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Summarise each cell: accuracy, tokens, reasoning tokens, cap-hit rate and cost."""
    spec = {row["cell"]: row for row in cfg["cells"]}
    grouped = frame.groupby("model", sort=False)
    out = pd.DataFrame(
        {
            "accuracy": grouped["score"].mean(),
            "mean_prompt_tokens": grouped["prompt_tokens"].mean(),
            "mean_completion_tokens": grouped["completion_tokens"].mean(),
            "mean_reasoning_tokens": grouped["reasoning_tokens"].mean(),
            "cap_hit_rate": grouped["capped"].mean(),
            "mean_cost_usd": grouped["cost"].mean(),
        }
    ).reset_index(names="cell")
    out["model"] = out["cell"].map(lambda c: spec[c]["model"])
    out["effort"] = out["cell"].map(lambda c: spec[c]["effort"])
    out["cap"] = out["cell"].map(lambda c: spec[c]["cap"])
    order = [row["cell"] for row in cfg["cells"]]
    return out.set_index("cell").loc[order].reset_index()


def write(out: Path, name: str, frame: pd.DataFrame) -> None:
    """Write one results table and log its size."""
    frame.to_csv(out / f"{name}.csv", index=False)
    LOGGER.info("wrote %s.csv (%d rows)", name, len(frame))


def main() -> None:
    """Grade a stage's responses and write the arm's tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "hosted_arm.yaml")
    parser.add_argument("--stage", default="pilot")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=None, help="override the config")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    boot = cfg["bootstrap"]
    kwargs = {
        "n_boot": args.n_boot or boot["n_boot"],
        "seed": boot["seed"],
        "level": boot["level"],
    }
    records, _ = read_jsonl(REPO_ROOT / cfg["raw_dir"] / f"{args.stage}_records.jsonl")
    items = {item.prompt_id: item for item in read_corpus(REPO_ROOT / cfg["corpus"])}
    frame, notes = graded_frame(records, items, cfg)
    LOGGER.info(
        "%d responses over %d prompts, %d cap hits, %d errors",
        notes["graded"],
        notes["prompts_kept"],
        notes["capped"],
        notes["errors"],
    )

    pool = [row["cell"] for row in cfg["cells"]]
    table = public.build_table(frame, pool, cfg["success"]["primary"])
    units = ("dollars", "tokens")
    out = args.out or REPO_ROOT / cfg["out_dir"] / args.stage
    out.mkdir(parents=True, exist_ok=True)

    write(out, "cells", cell_report(frame, cfg))
    summary = public.pool_summary(table, units)
    write(out, "pool", summary)
    ladder = summary[(summary["scope"] == "all") & summary["on_ladder"]]["model"].tolist()
    LOGGER.info("ladder: %s", " -> ".join(ladder))
    write(out, "cstar", public.cstar_distribution(table, units))
    taxes = public.tax_table(table, units, **kwargs)
    write(out, "tax_policies", taxes)

    # The mode axis, one pair per size: high effort against minimal on the same model.
    spec = {row["cell"]: row for row in cfg["cells"]}
    pairs = []
    for model in dict.fromkeys(row["model"] for row in cfg["cells"]):
        cells = [c for c, row in spec.items() if row["model"] == model]
        high = next((c for c in cells if spec[c]["effort"] == "high"), None)
        plain = next((c for c in cells if spec[c]["effort"] == "minimal"), None)
        if high and plain:
            pair = public.mode_pair(table, high, plain, units, **kwargs)
            pairs.append(pair.assign(model=model, thinking=high, plain=plain))
    if pairs:
        write(out, "mode_pairs", pd.concat(pairs, ignore_index=True))

    headline = taxes[
        (taxes["scope"] == "all") & (taxes["unit"] == "dollars") & (taxes["detection"] == "oracle")
    ].set_index("policy")
    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "stage": args.stage,
        "prompts": notes["prompts_kept"],
        "responses": notes["graded"],
        "cap_hits": notes["capped"],
        "errors": notes["errors"],
        "ladder": ladder,
        "n_boot": kwargs["n_boot"],
        "relative_tax": {
            policy: round(float(headline.loc[policy, "relative_tax"]), 4)
            for policy in headline.index
            if policy in {"P1", "P3", "P4-uniform", "random", "oracle"}
        },
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("summary: %s", json.dumps(payload["relative_tax"]))


if __name__ == "__main__":
    main()
