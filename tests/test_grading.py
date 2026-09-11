"""Unit tests for rule-based grading."""

from __future__ import annotations

import pytest

from arbiter.data.corpus import PromptItem
from arbiter.eval import grading

HUMANEVAL = PromptItem(
    prompt_id="humaneval-0",
    family="code",
    source="humaneval",
    text="",
    reference=None,
    subset="pilot",
    meta={
        "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        "entry_point": "add",
    },
)
MBPP = PromptItem(
    prompt_id="mbpp-0",
    family="code",
    source="mbpp",
    text="",
    reference=None,
    subset="pilot",
    meta={"test_list": ["assert double(2) == 4"], "test_setup_code": ""},
)


def _item(source: str, family: str, reference: str) -> PromptItem:
    return PromptItem(f"{source}-1", family, source, "", reference, "pilot")


def _record(content: str, done_reason: str = "stop") -> dict[str, str]:
    return {"content": content, "done_reason": done_reason}


def test_last_boxed_balances_braces() -> None:
    """Nested braces are kept and only the last box counts."""
    assert grading.last_boxed(r"\boxed{1} then \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"
    assert grading.last_boxed("no box") is None


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        (r"\dfrac12", r"\frac{1}{2}"),
        ("0.5", r"\frac{1}{2}"),
        ("x = 3", "3"),
        (r"10\%", "10"),
        ("1,000", "1000"),
        (r"90^\circ", "90"),
        ("(1, 2)", "(1,2)"),
        (r"\text{(C)}", "(C)"),
        ("3.0", "3"),
        (r"\sqrt2", r"\sqrt{2}"),
    ],
)
def test_math_equal_accepts_equivalent_forms(predicted: str, gold: str) -> None:
    """Common notational variants compare equal."""
    assert grading.math_equal(predicted, gold)


def test_math_equal_rejects_different_answers() -> None:
    """Different values and different expressions do not compare equal."""
    assert not grading.math_equal("3", "4")
    assert not grading.math_equal(r"\frac{1}{3}", "0.33")
    assert not grading.math_equal(r"\sqrt{2}", "2")


def test_grade_math_falls_back_to_last_number_for_gsm8k_only() -> None:
    """GSM8K accepts an unboxed final number; MATH-500 requires a box."""
    text = "Each costs 3, so the total is 1,234."
    assert grading.grade(_record(text), _item("gsm8k", "math", "1234")).correct
    result = grading.grade(_record(text), _item("math500", "math", "1234"))
    assert (result.correct, result.reason) == (False, "no_answer")


def test_cap_hit_always_fails() -> None:
    """A truncated response fails even if it contains the right answer."""
    result = grading.grade(_record(r"\boxed{4}", "length"), _item("gsm8k", "math", "4"))
    assert (result.correct, result.reason) == (False, "cap_hit")


@pytest.mark.parametrize(
    ("text", "letter"),
    [
        ("Reasoning...\nAnswer: C", "C"),
        ("Answer: (B)", "B"),
        ("Answer: **D**", "D"),
        (r"So \boxed{A}", "A"),
        ("B", "B"),
        ("The answer is E.", "E"),
    ],
)
def test_extract_choice(text: str, letter: str) -> None:
    """Formatted, boxed and bare answers are recognised."""
    assert grading.extract_choice(text) == letter


def test_extract_choice_ignores_lowercase_words() -> None:
    """A lowercase article after 'answer' is not an option letter."""
    assert grading.extract_choice("The answer is a bit subtle.") is None


def test_grade_mcq_strips_think_blocks() -> None:
    """A stray think block does not contribute an answer."""
    record = _record("<think>Answer: A</think>\nAnswer: B")
    assert grading.grade(record, _item("mmlu", "mcq", "B")).correct


def test_humaneval_full_function_and_body_only_completions() -> None:
    """Both a rewritten function and a bare body pass the benchmark tests."""
    full = "```python\ndef add(a, b):\n    return a + b\n```"
    assert grading.grade(_record(full), HUMANEVAL).correct
    body = "```python\n    return a + b\n```"
    assert grading.grade(_record(body), HUMANEVAL).correct
    wrong = "```python\ndef add(a, b):\n    return a - b\n```"
    result = grading.grade(_record(wrong), HUMANEVAL)
    assert (result.correct, result.reason) == (False, "AssertionError")


def test_mbpp_uses_the_last_code_block() -> None:
    """An earlier draft block is ignored in favour of the final one."""
    content = (
        "Draft:\n```python\ndef double(x):\n    return x\n```\n"
        "Final:\n```python\ndef double(x):\n    return 2 * x\n```"
    )
    assert grading.grade(_record(content), MBPP).correct
    assert grading.grade(_record("I cannot help."), MBPP).reason == "no_code"


def test_run_program_times_out_and_blocks_file_removal() -> None:
    """Infinite loops time out and destructive calls are disabled."""
    assert grading.run_program("while True:\n    pass\n", timeout_s=2.0) == (False, "timeout")
    passed, reason = grading.run_program("import os\nos.remove('anything')\n")
    assert (passed, reason) == (False, "PermissionError")


OPEN = PromptItem("dolly-1", "open", "dolly", "Name a colour.", "Blue.", "pilot")


@pytest.mark.parametrize(
    ("score", "correct", "reason"),
    [
        (5, True, "judge_5"),
        (4, True, "judge_4"),
        (3, False, "judge_3"),
        (1, False, "judge_1"),
        (None, False, "judge_unreadable"),
    ],
)
def test_open_answer_succeeds_at_a_judge_score_of_four(
    score: int | None, correct: bool, reason: str
) -> None:
    """A score of 4 or more succeeds, and an unreadable score fails."""
    result = grading.grade(_record("Red."), OPEN, {"score": score})
    assert (result.correct, result.reason) == (correct, reason)


def test_open_answer_can_fail_before_judging() -> None:
    """A cut-off or empty answer fails without a judgement."""
    assert grading.grade(_record("Red.", "length"), OPEN).reason == "cap_hit"
    assert grading.grade(_record("<think>hmm</think> "), OPEN).reason == "no_answer"


def test_unjudged_open_answer_is_an_error() -> None:
    """A judgeable answer with no judgement is not silently graded."""
    with pytest.raises(ValueError, match="not been judged"):
        grading.grade(_record("Red."), OPEN)
