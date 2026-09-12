"""Run the hosted-frontier arm through the Batch API, under a hard spend cap.

Six hosted cells -- three model sizes at two reasoning efforts -- over prompts the local
grid has already labelled, so each hosted c* is paired with a local one. No GPU is used.

Subcommands, in the order they are meant to be run::

    python scripts/run_hosted_arm.py plan    --stage pilot        # free: costs nothing
    python scripts/run_hosted_arm.py submit  --stage pilot
    python scripts/run_hosted_arm.py collect --stage pilot
    python scripts/run_hosted_arm.py campaign --stage main --n 860
    python scripts/run_hosted_arm.py status

``plan`` makes no network call at all. ``submit`` refuses to send a batch whose worst case
-- every request emitting its full output cap -- would take cumulative spend past the cap
in the config. ``collect`` prices the responses from the provider's own token counts and
settles the ledger, which is the audit trail for what was actually spent.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from arbiter.data.corpus import PromptItem, read_corpus  # noqa: E402
from arbiter.hosted import batch as hb  # noqa: E402
from arbiter.hosted.pricing import (  # noqa: E402
    BudgetError,
    Meter,
    Price,
    price_table,
)
from arbiter.utils.jsonl import append_jsonl  # noqa: E402

LOGGER = logging.getLogger("arbiter.hosted")


def config(path: Path) -> dict[str, Any]:
    """Read the arm's config."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def used_prompt_ids(cfg: dict[str, Any], stages: list[str]) -> set[str]:
    """Return the prompt ids earlier stages already spent money on, so none is paid for twice.

    A stage submits one batch per model, so its inputs are spread over several files; every
    one of them is read.
    """
    raw = REPO_ROOT / cfg["raw_dir"]
    used: set[str] = set()
    for stage in stages:
        for path in sorted(raw.glob(f"{stage}*_input.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    used.add(hb.split_custom_id(json.loads(line)["custom_id"])[1])
    return used


def select(
    cfg: dict[str, Any],
    stage: str,
    n: int | None,
    *,
    exclude: frozenset[str] = frozenset(),
    strict: bool = True,
) -> list[PromptItem]:
    """Draw the stage's prompts, stratified by family and disjoint from earlier stages.

    Selection is deterministic given the config seed: the corpus order is already
    stratified, so taking a fixed stride through each family is reproducible and needs no
    random state of its own. ``exclude`` withholds prompts a caller has already spent on --
    earlier waves of the same stage, which the stage's own input files would not exclude.
    With ``strict`` false a family running out ends the draw instead of failing, which is
    what a final wave needs.
    """
    spec = cfg["stages"][stage]
    count = n if n is not None else spec.get("n_prompts")
    if not count:
        raise SystemExit(f"stage {stage}: no prompt count; pass --n")
    items = read_corpus(REPO_ROOT / cfg["corpus"])
    excluded = used_prompt_ids(
        cfg, [s for s in cfg["sampling"]["exclude_stages"] if s != stage]
    ) | set(exclude)
    families = cfg["sampling"]["families"]
    wanted = spec.get("select")
    pools = {
        family: [
            item
            for item in items
            if item.family == family and item.subset == wanted and item.prompt_id not in excluded
        ]
        for family in families
    }
    cells = spec.get("cells") or [row["cell"] for row in cfg["cells"]]
    chosen: list[PromptItem] = []
    for index in range(count):
        family = families[index % len(families)]
        pool = pools[family]
        position = index // len(families)
        if position >= len(pool):
            if strict:
                raise SystemExit(
                    f"stage {stage}: family {family} has only {len(pool)} prompts left"
                )
            break
        chosen.append(pool[position])
    LOGGER.info(
        "stage %s: %d prompts x %d cells = %d requests",
        stage,
        len(chosen),
        len(cells),
        len(chosen) * len(cells),
    )
    return chosen


def stage_cells(cfg: dict[str, Any], stage: str) -> tuple[hb.Cell, ...]:
    """Return the cells this stage runs, defaulting to the whole ladder."""
    ladder = hb.cells_from_config(cfg["cells"])
    names = cfg["stages"][stage].get("cells")
    return ladder if not names else tuple(c for c in ladder if c.cell in set(names))


def meter_of(cfg: dict[str, Any]) -> Meter:
    """Open the spend ledger."""
    budget = cfg["budget"]
    return Meter.load(
        cap_usd=float(budget["cap_usd"]),
        reserve_usd=float(budget.get("reserve_usd", 0.0)),
        path=REPO_ROOT / budget["ledger"],
    )


def client_of(cfg: dict[str, Any]) -> hb.Client:
    """Build an OpenAI client from the key in .env, which is never logged."""
    from openai import OpenAI

    return OpenAI(api_key=hb.api_key(*hb.dotenv_paths(REPO_ROOT)))


def report_plan(plan: dict[str, float], meter: Meter, requests: int) -> None:
    """Log what a batch would cost and what the budget allows."""
    LOGGER.info("%d requests, worst case $%.4f", requests, plan["worst_case_usd"])
    LOGGER.info(
        "budget: cap $%.2f, reserve $%.2f, committed $%.4f, available $%.4f",
        meter.cap_usd,
        meter.reserve_usd,
        meter.committed,
        meter.available,
    )


def fit_to_budget(
    items: Sequence[PromptItem],
    cells: Sequence[hb.Cell],
    prices: Mapping[str, Price],
    available: float,
) -> list[PromptItem]:
    """Return the longest prefix of ``items`` whose worst case fits ``available``.

    Per-prompt worst case varies with prompt length, so the prefix is grown one prompt at a
    time rather than divided out of a mean.
    """
    kept: list[PromptItem] = []
    total = 0.0
    for item in items:
        step = hb.plan_cost([item], cells, prices)["worst_case_usd"]
        if total + step > available:
            break
        kept.append(item)
        total += step
    return kept


def submit_stage(
    cfg: dict[str, Any],
    stage: str,
    items: Sequence[PromptItem],
    cells: Sequence[hb.Cell],
    *,
    meter: Meter,
    client: hb.Client,
    label: str,
) -> None:
    """Submit one stage, or one wave of it, as one batch per model once the meter approves.

    The provider refuses a batch naming more than one model, so the request set is split by
    model. The budget is checked once against the whole submission before anything is sent, so
    it either fits or none of it goes out.
    """
    prices = price_table(cfg["pricing"], batch=True)
    plan = hb.plan_cost(items, cells, prices)
    report_plan(plan, meter, len(items) * len(cells))
    meter.check(plan["worst_case_usd"], label)
    for model, group in hb.group_by_model(cells):
        requests = hb.build_requests(items, group)
        share = hb.plan_cost(items, group, prices)
        job = hb.submit(
            client,
            requests,
            endpoint=cfg["endpoint"],
            completion_window=cfg["completion_window"],
            raw_dir=REPO_ROOT / cfg["raw_dir"],
            label=f"{label}_{model}",
        )
        entry = hb.ledger_entry(job.id, stage, requests, share, meter)
        entry["model"] = model
        entry["label"] = label
        meter.record(entry)
        LOGGER.info(
            "submitted %s: batch %s (%s), %d requests, worst case $%.4f",
            model,
            job.id,
            job.status,
            len(requests),
            share["worst_case_usd"],
        )


def collect_stage(
    cfg: dict[str, Any],
    stage: str,
    *,
    meter: Meter,
    client: hb.Client,
    reset: bool,
) -> dict[str, float]:
    """Wait for every unsettled batch of a stage, price it from reported usage and settle it."""
    pending = [e for e in meter.entries if e["stage"] == stage and e.get("actual_usd") is None]
    if not pending:
        raise SystemExit(f"stage {stage} has no unsettled batch in the ledger")
    cells = {c.cell: c for c in hb.cells_from_config(cfg["cells"])}
    prices = price_table(cfg["pricing"], batch=True)
    path = REPO_ROOT / cfg["raw_dir"] / f"{stage}_records.jsonl"
    if reset:
        path.unlink(missing_ok=True)
    grand: dict[str, float] = {}
    for entry in pending:
        job = hb.wait(client, entry["batch_id"], poll_s=float(cfg["poll_s"]), log=LOGGER)
        if job.status != "completed":
            LOGGER.error("batch %s (%s) ended as %s", job.id, entry.get("model"), job.status)
            hb.log_failure(client, job, LOGGER)
        records, totals = hb.collect(
            client,
            job,
            cells,
            prices,
            raw_dir=REPO_ROOT / cfg["raw_dir"],
            label=f"{entry.get('label', stage)}_{entry.get('model', 'all')}",
        )
        append_jsonl(path, records)
        meter.settle(job.id, totals["actual_usd"], totals)
        LOGGER.info(
            "%s: %d responses, %d errors, %d cap hits, %d reasoning tokens, $%.4f "
            "against a worst case of $%.4f",
            entry.get("model"),
            totals["responses"],
            totals["errors"],
            totals["cap_hits"],
            totals["reasoning_tokens"],
            totals["actual_usd"],
            entry["worst_case_usd"],
        )
        for key, value in totals.items():
            grand[key] = grand.get(key, 0) + float(value)
    return grand


def report_stage(stage: str, grand: Mapping[str, float], cells: int, meter: Meter) -> None:
    """Log a stage's measured totals and what the budget has left."""
    prompts = max(grand.get("responses", 0) / max(cells, 1), 1)
    LOGGER.info(
        "stage %s: %d responses, %d errors, %d cap hits, $%.4f total, $%.5f per prompt",
        stage,
        int(grand.get("responses", 0)),
        int(grand.get("errors", 0)),
        int(grand.get("cap_hits", 0)),
        grand.get("actual_usd", 0.0),
        grand.get("actual_usd", 0.0) / prompts,
    )
    LOGGER.info("spent $%.4f of $%.2f; $%.4f still available", meter.spent, meter.cap_usd,
                meter.available)  # fmt: skip


def cmd_plan(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Price a stage and check it against the budget, without any network call."""
    items = select(cfg, args.stage, args.n)
    cells = stage_cells(cfg, args.stage)
    prices = price_table(cfg["pricing"], batch=True)
    plan = hb.plan_cost(items, cells, prices)
    meter = meter_of(cfg)
    report_plan(plan, meter, len(items) * len(cells))
    LOGGER.info("worst case per prompt $%.5f", plan["worst_case_per_prompt"])
    try:
        meter.check(plan["worst_case_usd"], args.stage)
    except BudgetError as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error
    LOGGER.info("within budget; submit with: run_hosted_arm.py submit --stage %s", args.stage)


def cmd_submit(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Submit a whole stage as one submission."""
    items = select(cfg, args.stage, args.n)
    cells = stage_cells(cfg, args.stage)
    submit_stage(
        cfg,
        args.stage,
        items,
        cells,
        meter=meter_of(cfg),
        client=client_of(cfg),
        label=args.stage,
    )
    LOGGER.info("collect with: run_hosted_arm.py collect --stage %s", args.stage)


def cmd_collect(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Wait for every batch of a stage, price them from reported usage and settle the ledger.

    The records file is rewritten only for a stage nothing has been collected for yet. Once a
    stage has a settled batch its records were paid for, and truncating the file to re-collect
    the remainder would throw them away for good: the provider keeps an output file, but the
    ledger will not spend again to rebuild what is already on disk.
    """
    meter = meter_of(cfg)
    settled = any(
        entry["stage"] == args.stage and entry.get("actual_usd") is not None
        for entry in meter.entries
    )
    if settled:
        LOGGER.info("stage %s already has settled batches; appending rather than rewriting",
                    args.stage)  # fmt: skip
    grand = collect_stage(cfg, args.stage, meter=meter, client=client_of(cfg), reset=not settled)
    report_stage(args.stage, grand, len(stage_cells(cfg, args.stage)), meter)


def cmd_campaign(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Run a stage in waves, each sized to the worst case the remaining budget then allows.

    The guard prices a wave at its worst case -- every request emitting its full output cap --
    which real generations fall well short of. Settling a wave at its measured cost frees the
    difference, so a sequence of waves covers far more prompts than one worst-case-sized batch
    could while every wave is still checked against the cap before it is sent. Waves share one
    records file and one ledger, so the campaign is resumable: an interrupted run picks up from
    the prompts its own input files show were already paid for.
    """
    cells = stage_cells(cfg, args.stage)
    prices = price_table(cfg["pricing"], batch=True)
    client = client_of(cfg)
    target = args.n if args.n is not None else cfg["stages"][args.stage].get("n_prompts")
    done = 0
    for wave in range(1, args.max_waves + 1):
        meter = meter_of(cfg)
        spent_ids = used_prompt_ids(cfg, [args.stage])
        want = args.pool if target is None else max(target - done, 0)
        if want <= 0:
            LOGGER.info("target of %d prompt(s) reached", target)
            break
        drawn = select(cfg, args.stage, min(want, args.pool), exclude=frozenset(spent_ids),
                       strict=False)  # fmt: skip
        items = fit_to_budget(drawn, cells, prices, meter.available)
        if len(items) < args.floor:
            LOGGER.info(
                "wave %d: budget allows %d of %d drawn prompt(s), below the floor of %d; stopping",
                wave,
                len(items),
                len(drawn),
                args.floor,
            )
            break
        label = f"{args.stage}_w{wave}"
        LOGGER.info("wave %d: %d prompts as %s", wave, len(items), label)
        submit_stage(cfg, args.stage, items, cells, meter=meter, client=client, label=label)
        grand = collect_stage(cfg, args.stage, meter=meter, client=client, reset=False)
        report_stage(label, grand, len(cells), meter)
        if not grand.get("responses"):
            LOGGER.error("wave %d returned no response; stopping", wave)
            break
        done += len(items)
        LOGGER.info("campaign: %d prompt(s) collected so far", done)
    LOGGER.info("analyse with: run_hosted_analysis.py --stage %s", args.stage)


def cmd_status(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Print the ledger and the remaining budget."""
    meter = meter_of(cfg)
    for entry in meter.entries:
        actual = entry.get("actual_usd")
        LOGGER.info(
            "%-6s %-11s %s  %4d requests  worst $%.4f  actual %s",
            entry["stage"],
            entry.get("model", "-"),
            entry["batch_id"],
            entry["requests"],
            entry["worst_case_usd"],
            "pending" if actual is None else f"${actual:.4f}",
        )
    LOGGER.info(
        "cap $%.2f, spent $%.4f, committed $%.4f, available $%.4f",
        meter.cap_usd,
        meter.spent,
        meter.committed,
        meter.available,
    )


COMMANDS = {
    "plan": cmd_plan,
    "submit": cmd_submit,
    "collect": cmd_collect,
    "campaign": cmd_campaign,
    "status": cmd_status,
}


def main() -> None:
    """Dispatch a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "hosted_arm.yaml")
    parser.add_argument("--stage", default="pilot")
    parser.add_argument("--n", type=int, default=None, help="override the stage prompt count")
    parser.add_argument("--max-waves", type=int, default=12, help="campaign: wave limit")
    parser.add_argument("--floor", type=int, default=20,
                        help="campaign: stop once a wave would be smaller than this")  # fmt: skip
    parser.add_argument("--pool", type=int, default=900,
                        help="campaign: most prompts to draw for one wave")  # fmt: skip
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    COMMANDS[args.command](config(args.config), args)


if __name__ == "__main__":
    main()
