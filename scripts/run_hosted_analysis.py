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
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from arbiter.data.corpus import PromptItem, read_corpus  # noqa: E402
from arbiter.eval import public  # noqa: E402
from arbiter.eval.grading import grade  # noqa: E402
from arbiter.eval.labels import minimum_sufficient  # noqa: E402
from arbiter.hosted.pricing import cost_of, price_table  # noqa: E402
from arbiter.utils.jsonl import read_jsonl  # noqa: E402

LOGGER = logging.getLogger("arbiter.hosted_analysis")
MODES_PER_SIZE = 2  # both ladders pair every model size with a cheap and a costly mode


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
    """Summarise each cell: accuracy, tokens, reasoning tokens, cap-hit rate and cost.

    ``accuracy`` is the headline: a generation cut off at the cap is graded wrong, the same
    rule the local grid uses, because an answer the user never receives is no answer.
    ``accuracy_finished`` is that same accuracy over only the generations that stopped on
    their own. The two separate an effort setting that does not help from one whose
    reasoning does not fit in the cap, which look identical in the headline number.
    """
    spec = {row["cell"]: row for row in cfg["cells"]}
    grouped = frame.groupby("model", sort=False)
    finished = frame[~frame["capped"].astype(bool)].groupby("model", sort=False)
    out = pd.DataFrame(
        {
            "accuracy": grouped["score"].mean(),
            "accuracy_finished": finished["score"].mean(),
            "finished": finished["score"].size(),
            "mean_prompt_tokens": grouped["prompt_tokens"].mean(),
            "mean_completion_tokens": grouped["completion_tokens"].mean(),
            "mean_reasoning_tokens": grouped["reasoning_tokens"].mean(),
            "cap_hit_rate": grouped["capped"].mean(),
            "mean_cost_usd": grouped["cost"].mean(),
        }
    ).reset_index(names="cell")
    out["finished"] = out["finished"].fillna(0).astype(int)
    out["model"] = out["cell"].map(lambda c: spec[c]["model"])
    out["effort"] = out["cell"].map(lambda c: spec[c]["effort"])
    out["cap"] = out["cell"].map(lambda c: spec[c]["cap"])
    order = [row["cell"] for row in cfg["cells"]]
    return out.set_index("cell").loc[order].reset_index()


def write(out: Path, name: str, frame: pd.DataFrame) -> None:
    """Write one results table and log its size."""
    frame.to_csv(out / f"{name}.csv", index=False)
    LOGGER.info("wrote %s.csv (%d rows)", name, len(frame))


def cell_rank(name: str, per_size: int = MODES_PER_SIZE) -> tuple[int, int]:
    """Return ``(size rank, mode rank)`` for a size-major, mode-minor cell name.

    Both ladders are indexed the same way -- ``c1``..``c8`` over four sizes and ``h1``..``h6``
    over three -- so one function reads either, and the ranks are what compare across ladders
    whose absolute costs differ by orders of magnitude.
    """
    index = int(name[1:]) - 1
    return index // per_size, index % per_size


def cstar_per_prompt(table: public.PublicTable) -> pd.Series:
    """Return each prompt's minimum sufficient cell, ``unservable`` where none succeeds."""
    choice = minimum_sufficient(table.success, table.costs["dollars"])
    names = [table.models[c] if c >= 0 else "unservable" for c in choice]
    return pd.Series(names, index=table.queries["index"].to_numpy(), name="hosted_cstar")


def share(values: pd.Series) -> float | None:
    """Return a rounded share, or None when there is nothing to take a share of.

    ``json.dumps`` writes a float NaN as bare ``NaN``, which no strict JSON reader
    accepts, so an undefined statistic has to leave the payload as null.
    """
    if values.empty:
        return None
    return round(float(values.mean()), 4)


def pairing(
    hosted: pd.Series, splits: pd.DataFrame, unit: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Pair hosted and local ``c*`` on the prompts both arms labelled.

    The hosted prompts are drawn from the local main subset precisely so this comparison
    exists. What carries across two ladders of such different scale is not the cell but its
    rank: how far up its own ladder a prompt must be sent, and whether it needs the expensive
    mode. Prompts missing from either arm are dropped, and the unservable ones are reported
    rather than ranked, since they have no rung.
    """
    column = f"cstar_{unit}"
    if column not in splits.columns:
        raise ValueError(f"{column} is not in the local splits table")
    local = splits.set_index("prompt_id")
    shared = local.index.intersection(hosted.index)
    if shared.empty:
        raise ValueError(
            "no prompt is in both arms: the local analysis covers a different subset "
            "from the hosted stage"
        )
    frame = pd.DataFrame(
        {
            "family": local.loc[shared, "family"],
            "local_cstar": local.loc[shared, column],
            "hosted_cstar": hosted.loc[shared],
        }
    ).reset_index(names="prompt_id")
    for side in ("local", "hosted"):
        served = frame[f"{side}_cstar"] != "unservable"
        ranks = [cell_rank(cell) for cell in frame.loc[served, f"{side}_cstar"]]
        frame.loc[served, f"{side}_size"] = [rank[0] for rank in ranks]
        frame.loc[served, f"{side}_mode"] = [rank[1] for rank in ranks]

    counts = (
        frame.groupby(["local_cstar", "hosted_cstar"], sort=True).size().reset_index(name="count")
    )
    both = frame.dropna(subset=["local_size", "hosted_size"])
    rho, pvalue = (
        spearmanr(both["local_size"], both["hosted_size"])
        if len(both) > 2
        else (float("nan"), float("nan"))
    )
    rescued = frame[frame["local_cstar"] == "unservable"]["hosted_cstar"]
    summary = {
        "unit": unit,
        "paired": int(len(frame)),
        "servable_both": int(len(both)),
        "local_unservable": int((frame["local_cstar"] == "unservable").sum()),
        "hosted_unservable": int((frame["hosted_cstar"] == "unservable").sum()),
        "local_cheap_mode_share": share(both["local_mode"] == 0),
        "hosted_cheap_mode_share": share(both["hosted_mode"] == 0),
        "both_cheap_mode_share": share((both["local_mode"] == 0) & (both["hosted_mode"] == 0)),
        "size_rank_spearman": None if math.isnan(rho) else round(float(rho), 4),
        "size_rank_p": None if math.isnan(pvalue) else float(pvalue),
        "rescued_by_hosted": {
            str(cell): int(count) for cell, count in rescued.value_counts().items()
        },
    }
    return frame, counts, summary


def main() -> None:
    """Grade a stage's responses and write the arm's tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "hosted_arm.yaml")
    parser.add_argument("--stage", default="pilot")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=None, help="override the config")
    parser.add_argument(
        "--local", type=Path, default=None,
        help="local analysis directory holding splits.csv, to pair c* prompt by prompt",
    )  # fmt: skip
    parser.add_argument("--local-unit", default="energy_j", help="local c* unit to pair")
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
    paired: dict[str, Any] | None = None
    if args.local is not None:
        splits = pd.read_csv(args.local / "splits.csv")
        rows, counts, paired = pairing(cstar_per_prompt(table), splits, args.local_unit)
        write(out, "pairing", rows)
        write(out, "pairing_contingency", counts)
        LOGGER.info("pairing: %s", json.dumps(paired))

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
        "pairing": paired,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("summary: %s", json.dumps(payload["relative_tax"]))


if __name__ == "__main__":
    main()
