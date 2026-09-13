"""Tests for the validation re-scoring script's pricing and its spend ceiling.

The ceiling is the only thing standing between an authorised figure and an API bill, so
the arithmetic behind it is tested directly: what a request costs, what the guard charges
a request before sending it, and that a ceiling cannot be asked for without prices.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "judge_openai", REPO_ROOT / "scripts" / "judge_openai.py"
)
judge_openai = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(judge_openai)

# gpt-5-mini list prices, USD per million tokens, as configs/hosted_arm.yaml records them.
MINI_IN, MINI_OUT = 0.25, 2.00


def priced(inr_per_usd: float | None = None) -> argparse.Namespace:
    """Return the pricing arguments the cost helpers read."""
    return argparse.Namespace(usd_in=MINI_IN, usd_out=MINI_OUT, inr_per_usd=inr_per_usd)


def test_cost_prices_input_and_output_separately() -> None:
    """Output tokens cost eight times input tokens at these prices, not the same."""
    args = priced()
    assert judge_openai.cost_usd(1e6, 0, args) == pytest.approx(MINI_IN)
    assert judge_openai.cost_usd(0, 1e6, args) == pytest.approx(MINI_OUT)
    assert judge_openai.cost_usd(1e6, 1e6, args) == pytest.approx(MINI_IN + MINI_OUT)


def test_cost_does_not_depend_on_a_conversion_rate() -> None:
    """Dollars are dollars whether or not a rupee rate was supplied."""
    without = judge_openai.cost_usd(1000, 500, priced())
    with_rate = judge_openai.cost_usd(1000, 500, priced(inr_per_usd=88.0))
    assert without == pytest.approx(with_rate)


def test_money_adds_rupees_only_when_a_rate_is_given() -> None:
    """The display carries the authorised unit first and the credit's unit alongside."""
    assert judge_openai.money(0.36, priced()) == "$0.3600"
    assert judge_openai.money(0.36, priced(inr_per_usd=88.0)) == "$0.3600 (INR 32)"


def test_the_guard_charges_a_request_its_full_budget_before_sending_it() -> None:
    """The pre-send bound is the worst case, so the ceiling cannot be crossed mid-answer."""
    args = priced()
    body = {"messages": [{"role": "user", "content": "x" * 3000}]}
    bound = judge_openai.cost_usd(judge_openai.prompt_tokens_bound(body), 2000, args)
    realistic = judge_openai.cost_usd(judge_openai.prompt_tokens_bound(body), 250, args)
    assert bound > realistic
    # 3,000 characters at 3 chars per token, plus the full 2,000-token reply budget.
    assert bound == pytest.approx((1000 * MINI_IN + 2000 * MINI_OUT) / 1e6)


def test_a_ceiling_of_036_stops_short_rather_than_overspending() -> None:
    """Items are refused once the next one could pass the ceiling, leaving it unbroken."""
    args = priced()
    body = {"messages": [{"role": "user", "content": "x" * 3600}]}
    each = judge_openai.cost_usd(judge_openai.prompt_tokens_bound(body), 2000, args)
    spent, scored = 0.0, 0
    for _ in range(200):
        if spent + each > 0.36:
            break
        spent += each
        scored += 1
    assert spent <= 0.36
    assert scored < 200  # this bound does not buy the whole sample, and must say so


@pytest.mark.parametrize("flag", ["--estimate", "--max-usd"])
def test_a_ceiling_or_an_estimate_cannot_be_asked_for_without_prices(
    flag: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both refuse to run unpriced rather than silently treating the cost as zero."""
    argv = ["judge_openai.py", str(tmp_path), "--model", "gpt-5-mini", flag]
    if flag == "--max-usd":
        argv += ["0.36", "--confirm-spend"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as caught:
        judge_openai.main()
    assert caught.value.code == 2
