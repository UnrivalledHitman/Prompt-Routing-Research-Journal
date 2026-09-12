"""Unit tests for the hosted-frontier arm: pricing, the spend meter and batch plumbing.

Nothing here makes a network call. The client is a stub, so the request bodies, the
worst-case arithmetic and above all the budget guard are checked without spending anything.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from arbiter.data.corpus import PromptItem
from arbiter.eval.grading import grade
from arbiter.hosted import batch as hb
from arbiter.hosted.pricing import (
    BudgetError,
    Meter,
    Price,
    cost_of,
    price_table,
    usage_tokens,
    worst_case,
)

PRICING = {
    "batch_multiplier": 0.5,
    "models": {
        "cheap": {"input": 1.0, "cached_input": 0.1, "output": 10.0},
        "dear": {"input": 10.0, "cached_input": 1.0, "output": 100.0},
    },
}
CELLS = ({"cell": "h1", "model": "cheap", "effort": "minimal", "cap": 100},
         {"cell": "h2", "model": "dear", "effort": "high", "cap": 200})  # fmt: skip


def _item(prompt_id: str = "gsm8k-1", tokens: int = 40) -> PromptItem:
    return PromptItem(
        prompt_id=prompt_id,
        family="math",
        source="gsm8k",
        text="What is 2 + 2?",
        reference="4",
        subset="main",
        noise=False,
        meta={"prompt_tokens": tokens},
    )


def _body(
    output_tokens: int = 30, reasoning: int = 20, status: str = "completed"
) -> dict[str, Any]:
    return {
        "status": status,
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": "The answer is 4."}]},
        ],
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


def _line(custom_id: str = "h1|gsm8k-1", output_tokens: int = 30) -> dict[str, Any]:
    body = _body(output_tokens=output_tokens)
    return {"custom_id": custom_id, "response": {"status_code": 200, "body": body},
            "error": None}  # fmt: skip


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def test_price_table_applies_the_batch_discount() -> None:
    """Batch rates are half of list, and list prices are still reachable for the paper."""
    half = Price(input=0.5, output=5.0, cached_input=0.05)
    assert price_table(PRICING, batch=True)["cheap"] == half
    assert price_table(PRICING, batch=False)["cheap"].output == 10.0


def test_usage_tokens_separates_cached_input_and_does_not_double_count_reasoning() -> None:
    """Reasoning tokens are already inside output_tokens, so they are reported, not added."""
    tokens = usage_tokens(_body()["usage"])
    assert tokens == {
        "input_tokens": 10,  # 12 total less 2 cached
        "cached_tokens": 2,
        "output_tokens": 30,
        "reasoning_tokens": 20,
    }
    price = Price(input=1.0, output=10.0, cached_input=0.1)
    assert cost_of(tokens, price) == pytest.approx((10 * 1.0 + 2 * 0.1 + 30 * 10.0) / 1e6)


def test_worst_case_charges_the_whole_output_cap() -> None:
    """A request that has not run can only be bounded by its cap."""
    assert worst_case(100, 1000, Price(input=1.0, output=10.0)) == pytest.approx(10100 / 1e6)


# ---------------------------------------------------------------------------
# The budget guard
# ---------------------------------------------------------------------------


def test_meter_refuses_a_batch_that_could_pass_the_cap() -> None:
    """The guard plans against the worst case, never the expected case."""
    meter = Meter(cap_usd=1.0, reserve_usd=0.25)
    meter.check(0.70, "pilot")
    with pytest.raises(BudgetError, match="exceeds"):
        meter.check(0.80, "main")  # 0.80 > 1.00 - 0.25


def test_meter_charges_an_unsettled_batch_at_its_worst_case(tmp_path: Path) -> None:
    """An in-flight batch blocks the budget it might use, and releases it once priced."""
    path = tmp_path / "ledger.jsonl"
    meter = Meter.load(cap_usd=8.0, reserve_usd=0.0, path=path)
    meter.record({"batch_id": "b1", "stage": "pilot", "requests": 6, "worst_case_usd": 5.0,
                  "actual_usd": None})  # fmt: skip
    assert meter.committed == 5.0
    assert meter.spent == 0.0
    with pytest.raises(BudgetError):
        meter.check(4.0, "main")
    meter.settle("b1", 0.80, {"output_tokens": 10})
    assert meter.committed == pytest.approx(0.80)
    assert meter.spent == pytest.approx(0.80)
    meter.check(4.0, "main")


def test_meter_survives_a_new_process(tmp_path: Path) -> None:
    """The cap has to hold across invocations, so the ledger is the state, not memory."""
    path = tmp_path / "ledger.jsonl"
    first = Meter.load(cap_usd=8.0, reserve_usd=0.25, path=path)
    first.record({"batch_id": "b1", "stage": "pilot", "requests": 6, "worst_case_usd": 2.0,
                  "actual_usd": 1.5})  # fmt: skip
    second = Meter.load(cap_usd=8.0, reserve_usd=0.25, path=path)
    assert second.spent == pytest.approx(1.5)
    assert second.available == pytest.approx(8.0 - 0.25 - 1.5)
    with pytest.raises(KeyError):
        second.settle("absent", 1.0, {})


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def test_request_body_mirrors_the_local_request() -> None:
    """One user message, no system prompt, and a cap that counts reasoning tokens."""
    cell = hb.cells_from_config(CELLS)[1]
    body = hb.request_body(cell, "What is 2 + 2?")
    assert body["input"] == [{"role": "user", "content": "What is 2 + 2?"}]
    assert body["reasoning"] == {"effort": "high"}
    assert body["max_output_tokens"] == 200
    assert "temperature" not in body  # reasoning models reject it; the paper says so


def test_custom_id_round_trips_and_rejects_ambiguity() -> None:
    """The id carries the cell and the prompt, so it must not contain the separator."""
    assert hb.split_custom_id(hb.custom_id("h1", "gsm8k-1")) == ("h1", "gsm8k-1")
    with pytest.raises(ValueError, match="unusable"):
        hb.custom_id("h1", "bad|id")
    with pytest.raises(ValueError, match="unusable"):
        hb.custom_id("h1", "x" * 70)
    with pytest.raises(ValueError, match="malformed"):
        hb.split_custom_id("h1-gsm8k-1")


def test_build_requests_covers_every_prompt_and_cell() -> None:
    """Six requests for three prompts on two cells, each addressable afterwards."""
    items = [_item("a-1"), _item("a-2"), _item("a-3")]
    requests = hb.build_requests(items, hb.cells_from_config(CELLS))
    assert len(requests) == 6
    assert {r["url"] for r in requests} == {"/v1/responses"}
    assert len({r["custom_id"] for r in requests}) == 6


def test_plan_cost_inflates_the_prompt_token_count() -> None:
    """A hosted tokenizer counts the prompt differently, so the bound must not trust ours."""
    prices = price_table(PRICING, batch=True)
    plan = hb.plan_cost([_item(tokens=40)], hb.cells_from_config(CELLS), prices)
    bound = 60  # 40 * INPUT_SLACK
    expected = (bound * 0.5 + 100 * 5.0) / 1e6 + (bound * 5.0 + 200 * 50.0) / 1e6
    assert plan["worst_case_usd"] == pytest.approx(expected)
    assert plan["worst_case_per_prompt"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def test_output_text_ignores_reasoning_items() -> None:
    """Only the visible message is the answer; a reasoning item carries no text."""
    assert hb.output_text(_body()) == "The answer is 4."
    assert hb.output_text({"output_text": "direct"}) == "direct"
    assert hb.output_text({"output": []}) == ""


def test_parse_line_produces_a_record_the_local_grader_can_read() -> None:
    """``content`` and ``done_reason`` carry the local names, so grading is unchanged."""
    cells = {c.cell: c for c in hb.cells_from_config(CELLS)}
    record = hb.parse_line(_line(), cells)
    assert record["prompt_id"] == "gsm8k-1"
    assert record["cell"] == "h1"
    assert record["content"] == "The answer is 4."
    assert record["done_reason"] == "stop"
    assert record["reasoning_tokens"] == 20
    assert record["thinking"] == ""  # the provider returns the count, never the trace


def test_parse_line_treats_a_capped_response_as_a_failure() -> None:
    """A truncated generation is a cap hit, exactly as it is on the local grid."""
    cells = {c.cell: c for c in hb.cells_from_config(CELLS)}
    body = _body(status="incomplete")
    body["incomplete_details"] = {"reason": "max_output_tokens"}
    line = {"custom_id": "h2|gsm8k-1", "response": {"status_code": 200, "body": body}}
    assert hb.parse_line(line, cells)["done_reason"] == "length"


def test_parse_line_records_an_error_without_pricing_a_missing_answer() -> None:
    """A failed request has no usage, so it contributes no cost and no success."""
    cells = {c.cell: c for c in hb.cells_from_config(CELLS)}
    line = {"custom_id": "h1|gsm8k-1", "error": {"message": "rate limited"},
            "response": {"status_code": 429, "body": {}}}  # fmt: skip
    record = hb.parse_line(line, cells)
    assert record["error"] == "rate limited"
    assert record["done_reason"] == "error"
    assert record["output_tokens"] == 0


class _Files:
    def __init__(self, text: str) -> None:
        self.text = text

    def content(self, file_id: str) -> Any:  # noqa: ANN401 - mimics the SDK's response
        return type("Body", (), {"text": self.text})()


class _Client:
    def __init__(self, text: str) -> None:
        self.files = _Files(text)


def test_collect_prices_every_response_from_reported_usage(tmp_path: Path) -> None:
    """The dollar figure comes from the provider's token counts, not from the estimate."""
    text = json.dumps(_line()) + "\n" + json.dumps(_line("h2|gsm8k-2")) + "\n"
    job = type("Job", (), {"output_file_id": "f1", "error_file_id": None})()
    cells = {c.cell: c for c in hb.cells_from_config(CELLS)}
    prices = price_table(PRICING, batch=True)
    records, totals = hb.collect(_Client(text), job, cells, prices, raw_dir=tmp_path, label="pilot")
    assert len(records) == 2
    assert totals["responses"] == 2
    assert totals["cap_hits"] == 0
    assert totals["reasoning_tokens"] == 40
    cheap = (10 * 0.5 + 2 * 0.05 + 30 * 5.0) / 1e6
    dear = (10 * 5.0 + 2 * 0.5 + 30 * 50.0) / 1e6
    assert totals["actual_usd"] == pytest.approx(round(cheap + dear, 6))
    assert (tmp_path / "pilot_output.jsonl").exists()


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def test_api_key_reads_dotenv_and_rejects_a_malformed_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key with an inserted space has to fail loudly, and never be echoed."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    good = tmp_path / "good.env"
    good.write_text("OTHER=1\nOPENAI_API_KEY=sk-" + "a" * 60 + "\n", encoding="utf-8")
    assert hb.api_key(good).startswith("sk-")
    bad = tmp_path / "bad.env"
    bad.write_text('OPENAI_API_KEY="sk= -proj-' + "a" * 60 + '"\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not look like an API key") as caught:
        hb.api_key(bad)
    assert "aaaa" not in str(caught.value)
    with pytest.raises(RuntimeError, match="not set"):
        hb.api_key(tmp_path / "absent.env")


def test_dotenv_paths_reaches_the_main_checkout_from_a_worktree(tmp_path: Path) -> None:
    """A worktree keeps no credential of its own, so the search follows the gitdir pointer."""
    main = tmp_path / "repo"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    tree = tmp_path / "wt"
    tree.mkdir()
    pointer = main / ".git" / "worktrees" / "wt"
    (tree / ".git").write_text(f"gitdir: {pointer}", encoding="utf-8")
    assert hb.dotenv_paths(tree) == (tree / ".env", main / ".env")
    plain = tmp_path / "plain"
    (plain / ".git").mkdir(parents=True)
    assert hb.dotenv_paths(plain) == (plain / ".env",)


def test_api_key_prefers_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A key exported for one run must win over a stale dotenv value."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "b" * 60)
    path = tmp_path / "env"
    path.write_text("OPENAI_API_KEY=sk-" + "c" * 60, encoding="utf-8")
    assert hb.api_key(path) == "sk-" + "b" * 60


def test_group_by_model_keeps_both_efforts_of_a_size_together() -> None:
    """A batch may name only one model, so a stage splits by model, not by cell."""
    cells = hb.cells_from_config(
        [
            {"cell": "h1", "model": "cheap", "effort": "minimal", "cap": 100},
            {"cell": "h2", "model": "cheap", "effort": "high", "cap": 200},
            {"cell": "h3", "model": "dear", "effort": "minimal", "cap": 100},
        ]
    )
    groups = hb.group_by_model(cells)
    assert [model for model, _ in groups] == ["cheap", "dear"]
    assert [c.cell for c in groups[0][1]] == ["h1", "h2"]
    for _, group in groups:
        bodies = hb.build_requests([_item()], group)
        assert len({line["body"]["model"] for line in bodies}) == 1


# ---------------------------------------------------------------------------
# Grading and the cell report
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_hosted_analysis import cell_report, graded_frame  # noqa: E402

CFG = {
    "pricing": PRICING,
    "cells": [
        {"cell": "h1", "model": "cheap", "effort": "minimal", "cap": 100},
        {"cell": "h2", "model": "cheap", "effort": "high", "cap": 200},
    ],
}


def _record(cell: str, prompt_id: str, content: str, **over: str | int | None) -> dict[str, Any]:
    record = {
        "prompt_id": prompt_id,
        "cell": cell,
        "model": "cheap",
        "effort": "minimal",
        "cap": 100,
        "content": content,
        "thinking": "",
        "done_reason": "stop",
        "status": "completed",
        "error": None,
        "input_tokens": 10,
        "cached_tokens": 0,
        "output_tokens": 20,
        "reasoning_tokens": 0,
    }
    return record | over


def test_graded_frame_scores_with_the_local_grader_and_prices_each_response() -> None:
    """A right answer scores 1, a wrong one 0, and cost comes from the reported tokens."""
    items = {"gsm8k-1": _item("gsm8k-1")}
    records = [
        _record("h1", "gsm8k-1", r"the answer is \boxed{4}"),
        _record("h2", "gsm8k-1", r"\boxed{7}"),
    ]
    frame, notes = graded_frame(records, items, CFG)
    assert notes["prompts_kept"] == 1
    scores = dict(zip(frame["model"], frame["score"], strict=True))
    assert scores == {"h1": 1.0, "h2": 0.0}
    assert frame["cost"].iloc[0] == pytest.approx((10 * 0.5 + 20 * 5.0) / 1e6)
    assert set(frame["dataset"]) == {"math"}


def test_a_capped_answer_fails_even_when_it_parses() -> None:
    """The cap rule lives in the grader: a truncated generation cannot score, however it reads.

    The two records differ only in ``done_reason``, so this pins the rule to that field and
    proves the hosted arm inherits it rather than reimplementing it.
    """
    items = {"gsm8k-1": _item("gsm8k-1")}
    records = [
        _record("h1", "gsm8k-1", r"\boxed{4}", done_reason="length"),
        _record("h2", "gsm8k-1", r"\boxed{4}"),
    ]
    assert grade(records[0], items["gsm8k-1"]).reason == "cap_hit"
    frame, notes = graded_frame(records, items, CFG)
    assert notes["capped"] == 1
    assert dict(zip(frame["model"], frame["score"], strict=True)) == {"h1": 0.0, "h2": 1.0}


def test_graded_frame_drops_a_prompt_with_a_failed_or_missing_cell() -> None:
    """A hole in the matrix would change the pool, so the whole prompt goes."""
    items = {"a-1": _item("a-1"), "a-2": _item("a-2"), "a-3": _item("a-3")}
    records = [
        _record("h1", "a-1", r"\boxed{4}"),
        _record("h2", "a-1", r"\boxed{4}"),
        _record("h1", "a-2", r"\boxed{4}"),  # h2 never came back
        _record("h1", "a-3", "", error="rate limited", done_reason="error"),
        _record("h2", "a-3", r"\boxed{4}"),
    ]
    frame, notes = graded_frame(records, items, CFG)
    assert notes["prompts_seen"] == 3
    assert notes["prompts_kept"] == 1
    assert set(frame["index"]) == {"a-1"}


def test_graded_frame_rejects_a_record_outside_the_corpus() -> None:
    """A prompt id that cannot be looked up cannot be graded, and must not pass silently."""
    with pytest.raises(ValueError, match="not in the corpus"):
        graded_frame([_record("h1", "absent-1", "x")], {}, CFG)


def test_cell_report_keeps_ladder_order_and_reports_the_cap_rate() -> None:
    """The report is read as a ladder, so it follows config order, not accuracy."""
    items = {"a-1": _item("a-1")}
    records = [
        _record("h1", "a-1", r"\boxed{4}", done_reason="length"),
        _record("h2", "a-1", r"\boxed{4}", reasoning_tokens=15),
    ]
    frame, _ = graded_frame(records, items, CFG)
    report = cell_report(frame, CFG)
    assert report["cell"].tolist() == ["h1", "h2"]
    assert report["effort"].tolist() == ["minimal", "high"]
    assert report.set_index("cell").loc["h1", "cap_hit_rate"] == 1.0
    assert report.set_index("cell").loc["h2", "mean_reasoning_tokens"] == 15.0


# ---------------------------------------------------------------------------
# The wave campaign
# ---------------------------------------------------------------------------

from run_hosted_arm import fit_to_budget, select, used_prompt_ids  # noqa: E402


def _corpus(path: Path, per_family: int) -> list[PromptItem]:
    """Write a small main-subset corpus, stratified the way the real one is."""
    items = [
        PromptItem(
            prompt_id=f"{family}-{index}",
            family=family,
            source=family,
            text="What is 2 + 2?",
            reference="4",
            subset="main",
            noise=False,
            meta={"prompt_tokens": 40},
        )
        for index in range(per_family)
        for family in ("math", "code", "mcq")
    ]
    path.write_text(
        "".join(json.dumps(asdict(item)) + "\n" for item in items), encoding="utf-8"
    )
    return items


def _campaign_cfg(tmp_path: Path, per_family: int = 4) -> dict[str, Any]:
    """Build a config whose corpus and raw directory live under ``tmp_path``."""
    corpus = tmp_path / "corpus.jsonl"
    _corpus(corpus, per_family)
    return {
        "corpus": str(corpus),
        "raw_dir": str(tmp_path),
        "cells": list(CELLS),
        "sampling": {"families": ["math", "code", "mcq"], "exclude_stages": ["pilot"]},
        "stages": {"main": {"n_prompts": None, "select": "main"}},
    }


def test_fit_to_budget_grows_a_wave_one_prompt_at_a_time() -> None:
    """A wave is the longest prefix whose worst case fits, never a mean divided out."""
    cells = hb.cells_from_config(CELLS)
    prices = price_table(PRICING, batch=True)
    items = [_item(f"gsm8k-{index}") for index in range(10)]
    each = hb.plan_cost([items[0]], cells, prices)["worst_case_usd"]
    assert len(fit_to_budget(items, cells, prices, each * 3.5)) == 3
    assert fit_to_budget(items, cells, prices, each * 0.9) == []
    assert len(fit_to_budget(items, cells, prices, each * 100)) == 10


def test_select_withholds_prompts_an_earlier_wave_already_paid_for(tmp_path: Path) -> None:
    """Waves of one stage must be disjoint, which the stage's own input files cannot enforce."""
    cfg = _campaign_cfg(tmp_path)
    first = select(cfg, "main", 6)
    assert len(first) == 6
    spent = frozenset(item.prompt_id for item in first)
    second = select(cfg, "main", 6, exclude=spent)
    assert len(second) == 6
    assert not spent & {item.prompt_id for item in second}


def test_select_stops_instead_of_failing_when_a_final_wave_runs_short(tmp_path: Path) -> None:
    """A final wave asks for more than the pool holds; the fixed stages still want a failure."""
    cfg = _campaign_cfg(tmp_path, per_family=2)
    assert len(select(cfg, "main", 99, strict=False)) == 6
    with pytest.raises(SystemExit):
        select(cfg, "main", 99)


def test_used_prompt_ids_reads_every_wave_of_a_stage(tmp_path: Path) -> None:
    """A stage spreads its requests over one input file per model and per wave."""
    cfg = _campaign_cfg(tmp_path)
    for name, prompt_id in (("main_w1_cheap", "math-0"), ("main_w2_dear", "code-1")):
        line = {"custom_id": hb.custom_id("h1", prompt_id), "method": "POST"}
        (tmp_path / f"{name}_input.jsonl").write_text(
            json.dumps(line) + "\n", encoding="utf-8"
        )
    assert used_prompt_ids(cfg, ["main"]) == {"math-0", "code-1"}
