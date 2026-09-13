"""Unit tests for the open-ended judge's prompt, reply parsing and selection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from arbiter.data.corpus import PromptItem
from arbiter.eval import judge

ITEM = PromptItem(
    "dolly-7",
    "open",
    "dolly",
    "Summarise the passage.\n\nThe passage.",
    "A summary.",
    "pilot",
    meta={"category": "summarization"},
)
CFG: dict[str, Any] = {
    "seed": 5,
    "keep_alive": "5m",
    "model": {"tag": "llama3.1:8b-instruct-q4_K_M"},
    "decoding": {
        "temperature": 0.0,
        "num_ctx": 10240,
        "num_predict": 384,
        "retry_num_predict": 1024,
    },
}


class FakeClient:
    """Stand-in for ``ollama.Client`` that replays canned replies and records calls."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401
        """Return the next canned reply."""
        self.calls.append(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content=self.replies.pop(0)),
            done_reason="stop",
            eval_count=20,
            prompt_eval_count=300,
        )


def _record(
    content: str = "An answer.",
    done_reason: str = "stop",
    **key: Any,  # noqa: ANN401
) -> dict[str, Any]:
    base = {
        "prompt_id": "dolly-7",
        "tag": "qwen3:0.6b-q4_K_M",
        "mode": "no_think",
        "sample": 0,
        "variant": "base",
    }
    return base | key | {"content": content, "done_reason": done_reason}


def test_messages_carry_request_reference_and_answer() -> None:
    """The rubric is the system message; the user message holds all three texts."""
    messages, truncated = judge.build_messages(
        ITEM.text, ITEM.reference, "An answer.", max_prompt_tokens=4096
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == judge.SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "Summarise the passage." in user and "A summary." in user and "An answer." in user
    assert not truncated


def test_human_rubric_is_the_judge_rubric_without_the_reply_format() -> None:
    """The human labeller reads the judge's rubric, minus the JSON instruction."""
    assert judge.SYSTEM_PROMPT.startswith(judge.RUBRIC)
    assert "JSON" not in judge.RUBRIC and "JSON" in judge.SYSTEM_PROMPT


def test_braces_in_texts_survive_formatting() -> None:
    """Braces in the request, reference or answer are not treated as placeholders."""
    messages, _ = judge.build_messages("Explain {x}.", "It is {y}.", "Use {z}.", 4096)
    user = messages[1]["content"]
    assert "Explain {x}." in user and "It is {y}." in user and "Use {z}." in user


def test_only_the_reference_is_shortened() -> None:
    """A long reference is cut to fit; the request and the answer stay whole."""
    messages, truncated = judge.build_messages("Summarise.", "word " * 5000, "An answer.", 2000)
    system, user = (m["content"] for m in messages)
    assert truncated
    assert user.startswith("[Request]\nSummarise.") and user.endswith("An answer.")
    assert judge.TRUNCATION_NOTE in user
    assert len(system) + len(user) <= 2000 * judge.CHARS_PER_TOKEN


def test_prompt_budget_leaves_room_for_the_longest_reply() -> None:
    """The prompt budget is the context less the retry reply budget and a margin."""
    assert judge.prompt_budget(CFG) == 10240 - 1024 - judge.PROMPT_MARGIN_TOKENS


@pytest.mark.parametrize(
    ("content", "score"),
    [
        ('{"reason": "Accurate.", "score": 4}', 4),
        ('{"reason": "Wrong.", "score": 1}', 1),
        ('{"reason": "Out of range.", "score": 7}', None),
        ('{"reason": "Boolean.", "score": true}', None),
        ('{"reason": "Fractional.", "score": 4.5}', None),
        ('{"reason": "String score.", "score": "3"}', 3),
        ('{"reason": "Cut off after the score", "score": 5', 5),
        ('{"reason": "Cut off before the score', None),
        ("", None),
    ],
)
def test_parse_judgement(content: str, score: int | None) -> None:
    """Scores are read from JSON, with a fallback for a reply cut off after the score."""
    assert judge.parse_judgement(content)[0] == score


def test_judgeable_skips_failures_and_other_families() -> None:
    """Cap hits and empty answers fail without the judge; gradable families are ignored."""
    math = PromptItem("gsm8k-1", "math", "gsm8k", "", "3", "pilot")
    items = {"dolly-7": ITEM, "gsm8k-1": math}
    records = [
        _record(),
        _record(done_reason="length", mode="think"),
        _record(content="<think>only thoughts</think>  ", sample=1),
        _record(prompt_id="gsm8k-1"),
    ]
    todo, skipped = judge.judgeable(records, items)
    assert list(todo) == [("dolly-7", "qwen3:0.6b-q4_K_M", "no_think", 0, "base")]
    assert skipped == {"cap_hit": 1, "no_answer": 1}


def test_order_is_fixed_and_independent_of_input_order() -> None:
    """The judging order depends only on the keys and the seed."""
    keys = [(f"dolly-{i}", "t", "think", 0, "base") for i in range(50)]
    order = judge.judge_order(keys, 1)
    assert order == judge.judge_order(list(reversed(keys)), 1)
    assert order != sorted(keys) and sorted(order) == sorted(keys)


def test_judge_one_scores_greedily_under_the_schema() -> None:
    """One request per answer, greedy, schema-constrained, keyed like the generation."""
    client = FakeClient(['{"reason": "Accurate summary.", "score": 5}'])
    out = judge.judge_one(client, CFG, ITEM, _record())
    call = client.calls[0]
    assert call["format"] == judge.RESPONSE_FORMAT
    assert call["options"]["temperature"] == 0.0 and call["options"]["num_predict"] == 384
    assert out["score"] == 5 and out["reason"] == "Accurate summary." and out["attempts"] == 1
    key = tuple(out[field] for field in judge.KEY_FIELDS)
    assert key == ("dolly-7", "qwen3:0.6b-q4_K_M", "no_think", 0, "base")
    assert out["category"] == "summarization"
    assert not out["context_full"] and not out["reference_truncated"]


def test_judge_one_retries_an_unreadable_reply_with_a_longer_budget() -> None:
    """A reply cut off before its score is retried once with more tokens."""
    client = FakeClient(['{"reason": "The answer is', '{"reason": "Right.", "score": 4}'])
    out = judge.judge_one(client, CFG, ITEM, _record())
    assert [c["options"]["num_predict"] for c in client.calls] == [384, 1024]
    assert out["score"] == 4 and out["attempts"] == 2


def test_judge_sees_only_the_visible_answer() -> None:
    """A think block in the content is removed before judging."""
    client = FakeClient(['{"reason": "Fine.", "score": 4}'])
    judge.judge_one(client, CFG, ITEM, _record(content="<think>private</think>Visible answer."))
    user = client.calls[0]["messages"][1]["content"]
    assert "Visible answer." in user and "private" not in user
