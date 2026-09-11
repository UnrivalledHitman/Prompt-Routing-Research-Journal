"""Unit tests for the R3 and verifier prompts and their log-probability parsing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arbiter.eval.policies import select_cells
from arbiter.routers import llm


def test_answer_probabilities_skips_blank_and_thinking_tokens() -> None:
    """The first visible position gives the distribution, from its alternatives if present."""
    top = [
        {"token": "C", "logprob": math.log(0.5)},
        {"token": " A", "logprob": math.log(0.3)},
        {"token": "x", "logprob": math.log(0.1)},
    ]
    logprobs = [
        {"token": "\n\n", "logprob": 0.0, "top_logprobs": []},
        {"token": "</think>", "logprob": 0.0},
        {"token": "C", "logprob": math.log(0.5), "top_logprobs": top},
        {"token": ".", "logprob": 0.0},
    ]
    assert llm.answer_probabilities(logprobs) == pytest.approx({"C": 0.5, " A": 0.3, "x": 0.1})
    assert llm.answer_probabilities([{"token": "B", "logprob": 0.0}]) == {"B": 1.0}
    assert llm.answer_probabilities([{"token": " ", "logprob": 0.0}]) == {}
    assert llm.answer_probabilities(None) == {}


def test_label_distribution_renormalises_capital_letters() -> None:
    """Only the eight capital letters count, with surrounding space ignored."""
    labels = llm.label_distribution({"C": 0.5, " A": 0.3, "x": 0.1, "a": 0.1})
    expected = np.zeros(8)
    expected[[0, 2]] = [0.375, 0.625]
    assert labels == pytest.approx(expected)
    assert np.isnan(llm.label_distribution({"maybe": 1.0})).all()


def test_yes_confidence() -> None:
    """Confidence is P(Yes) over P(Yes) + P(No), case and space ignored."""
    assert llm.yes_confidence({"Yes": 0.6, " no": 0.2, "Maybe": 0.2}) == pytest.approx(0.75)
    assert math.isnan(llm.yes_confidence({}))


def test_success_from_labels_accumulates_in_cost_order() -> None:
    """Mass on c3 and c2 makes c3 half-likely and every cell dearer than c2 certain."""
    mean_cost = np.array([1.0, 4.0, 2.0, 8.0, 3.0, 16.0, 5.0, 32.0])
    labels = np.zeros((1, 8))
    labels[0, [1, 2]] = 0.5
    success = llm.success_from_labels(labels, mean_cost)
    assert success[0] == pytest.approx([0.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0])


def test_lookup_router_sends_unknown_prompts_to_the_costliest_cell() -> None:
    """A measured distribution routes by threshold; an unmeasured prompt goes to c8."""
    mean_cost = np.arange(1.0, 9.0)
    router = llm.LookupRouter({"known": np.eye(8)[2]}, mean_cost).fit(["known"], np.ones((1, 8)))
    p_hat = router.predict_proba(["known", "unknown"])
    assert p_hat[0] == pytest.approx([0, 0, 1, 1, 1, 1, 1, 1])
    assert p_hat[1] == pytest.approx([0, 0, 0, 0, 0, 0, 0, 1])
    assert select_cells(p_hat, mean_cost, 0.5).tolist() == [2, 7]


def test_prompts_carry_options_examples_and_the_request() -> None:
    """The router lists every configuration and truncates long examples."""
    prompt = llm.router_prompt("What is 2 + 2?", [("x" * 1000, 2)])
    assert "H. 8B model, thinks step by step first" in prompt
    assert "[...]\n</request>\nC" in prompt
    assert prompt.endswith("<request>\nWhat is 2 + 2?\n</request>")
    assert "Examples" not in llm.router_prompt("What is 2 + 2?")

    check = llm.verify_prompt("What is 2 + 2?", "4")
    assert "<question>\nWhat is 2 + 2?\n</question>\n<answer>\n4\n</answer>" in check
    assert check.endswith("Reply Yes or No.")
