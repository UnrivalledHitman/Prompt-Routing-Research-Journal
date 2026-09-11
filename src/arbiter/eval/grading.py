r"""Rule-based grading of labelling generations (experimental protocol, section 5.2).

A generation succeeds only if it finished within its token cap and its final answer
is correct. The rules per family:

* **Math.** The answer is the last ``\\boxed{}`` in the response; for GSM8K, the last
  number is used if nothing is boxed. Answers are compared after LaTeX normalisation
  (fractions, spacing, units, degree and percent signs, leading ``x =``), then
  numerically when both sides parse as numbers.
* **Multiple choice.** The answer is the last ``Answer: X`` line, then a boxed letter,
  then a response that is a single letter.
* **Code.** The last Python code block runs against the benchmark's own tests in a
  separate interpreter, in a temporary directory, with a timeout. Destructive file,
  process and network functions are disabled before the candidate code runs, as in
  the HumanEval harness. This is a guard against accidents, not a security sandbox.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from arbiter.data.corpus import PromptItem

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
CODE_BLOCK = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.DOTALL)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
MCQ_ANSWER = re.compile(r"(?i:answer)\s*(?i:is)?\s*[:：]?\s*\**\s*\(?([A-H])\)?(?![A-Za-z])")
ERROR_TYPE = re.compile(r"^(\w+(?:Error|Exception|Exit|Interrupt|Warning))\b")
BOXED_LETTER = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-H])\)?\s*\}?\s*\}")
CODE_TIMEOUT_S = 10.0

GUARD = textwrap.dedent(
    """
    import builtins as _b, os as _os, shutil as _sh, subprocess as _sp, socket as _so
    def _blocked(*args, **kwargs):
        raise PermissionError("disabled during grading")
    for _name in ("system", "remove", "unlink", "rmdir", "removedirs", "rename", "renames",
                  "replace", "kill", "fork", "forkpty", "killpg", "putenv", "unsetenv",
                  "chmod", "chown", "truncate", "execv", "execve", "execvp", "spawnv",
                  "startfile"):
        if hasattr(_os, _name):
            setattr(_os, _name, _blocked)
    for _name in ("rmtree", "move", "chown"):
        setattr(_sh, _name, _blocked)
    _sp.Popen = _blocked
    _sp.run = _blocked
    _sp.call = _blocked
    _sp.check_output = _blocked
    _so.socket = _blocked
    _so.create_connection = _blocked
    del _b, _os, _sh, _sp, _so, _name
    """
)


@dataclass(frozen=True)
class Grade:
    """Outcome of grading one generation."""

    correct: bool
    reason: str
    extracted: str | None


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def final_text(record: dict[str, Any]) -> str:
    """Return the visible answer with any stray think block removed."""
    return THINK_BLOCK.sub("", record.get("content") or "").strip()


def last_boxed(text: str) -> str | None:
    r"""Return the contents of the last ``\boxed{...}`` or ``\fbox{...}``, braces balanced."""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    open_brace = text.find("{", start)
    if open_brace < 0:
        return None
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index]
    return None


def _strip_wrapper(text: str, command: str) -> str:
    r"""Replace ``\command{inner}`` with ``inner``, braces balanced."""
    needle = f"\\{command}{{"
    while (start := text.find(needle)) >= 0:
        depth, index = 0, start + len(needle) - 1
        for index in range(start + len(needle) - 1, len(text)):  # noqa: B007
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    break
        else:
            return text
        text = text[:start] + text[start + len(needle) : index] + text[index + 1 :]
    return text


def _brace_fracs(text: str) -> str:
    r"""Rewrite ``\frac12`` and ``\frac1{2}`` as ``\frac{1}{2}``."""
    return re.sub(
        r"\\frac(\{[^{}]*\}|\w)(\{[^{}]*\}|\w)",
        lambda m: "\\frac{" + m.group(1).strip("{}") + "}{" + m.group(2).strip("{}") + "}",
        text,
    )


def normalise_math(answer: str) -> str:
    """Normalise a LaTeX answer so that equivalent forms usually compare equal."""
    text = answer.strip()
    for command in ("text", "textbf", "mathrm", "mbox", "boxed"):
        text = _strip_wrapper(text, command)
    replacements = {
        "\n": "",
        "\\!": "",
        "\\,": "",
        "\\;": "",
        "\\ ": "",
        "\\\\": "\\",
        "tfrac": "frac",
        "dfrac": "frac",
        "\\left": "",
        "\\right": "",
        "^{\\circ}": "",
        "^\\circ": "",
        "\\$": "",
        "$": "",
        "\\%": "",
        "%": "",
        "\\displaystyle": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", text)
    text = re.sub(r"(degrees?|units?|cm|inches|inch|meters?|dollars?|cents?)\b", "", text)
    text = text.replace(" ", "")
    text = re.sub(r"^[a-zA-Z]\w{0,1}=", "", text) if len(text.split("=")) == 2 else text
    text = _brace_fracs(text)
    text = re.sub(r"(?<![\d.])\.(\d)", r"0.\1", text)
    text = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", text)
    text = text.rstrip(".,")
    if re.fullmatch(r"-?\d+/\d+", text):
        numerator, denominator = text.split("/")
        text = f"\\frac{{{numerator}}}{{{denominator}}}"
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    return text


def to_number(text: str) -> Fraction | None:
    """Parse a normalised answer as an exact number, if it is one."""
    sign = -1 if text.startswith("-") else 1
    body = text[1:] if sign < 0 else text
    try:
        if match := re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", body):
            return sign * Fraction(match.group(1)) / Fraction(match.group(2))
        if re.fullmatch(r"\d+(?:\.\d+)?", body):
            return sign * Fraction(body)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def math_equal(predicted: str, gold: str) -> bool:
    """Compare two answers by normalised string, then by exact numeric value."""
    a, b = normalise_math(predicted), normalise_math(gold)
    if a == b:
        return True
    x, y = to_number(a), to_number(b)
    if x is None or y is None:
        return False
    return abs(x - y) <= Fraction(1, 10**6) * max(1, abs(y))


def grade_math(record: dict[str, Any], item: PromptItem) -> Grade:
    """Grade a math answer against the reference."""
    text = final_text(record)
    answer = last_boxed(text)
    if answer is None and item.source == "gsm8k":
        numbers = NUMBER.findall(text)
        answer = numbers[-1] if numbers else None
    if answer is None:
        return Grade(False, "no_answer", None)
    return Grade(math_equal(answer, str(item.reference)), "graded", answer)


# ---------------------------------------------------------------------------
# Multiple choice
# ---------------------------------------------------------------------------


def extract_choice(text: str) -> str | None:
    """Return the chosen option letter, or None."""
    matches = MCQ_ANSWER.findall(text)
    if matches:
        return matches[-1].upper()
    boxed = BOXED_LETTER.findall(text)
    if boxed:
        return boxed[-1].upper()
    bare = re.fullmatch(r"\W*([A-H])\W*", text.strip())
    return bare.group(1) if bare else None


def grade_mcq(record: dict[str, Any], item: PromptItem) -> Grade:
    """Grade a multiple-choice answer against the reference letter."""
    choice = extract_choice(final_text(record))
    if choice is None:
        return Grade(False, "no_answer", None)
    return Grade(choice == item.reference, "graded", choice)


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------


def extract_code(text: str) -> str | None:
    """Return the last fenced Python block, or unfenced text that looks like code."""
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        return blocks[-1]
    if re.search(r"^\s*(def|class|import|from)\s", text, re.MULTILINE) and "```" not in text:
        return text
    return None


def build_program(code: str, item: PromptItem) -> str:
    """Assemble the test program for a HumanEval or MBPP item."""
    meta = item.meta
    if item.source == "humaneval":
        check = f"check({meta['entry_point']})"
        return f"{meta['prompt']}\n    pass\n\n{code}\n\n{meta['test']}\n\n{check}\n"
    if item.source == "mbpp":
        tests = "\n".join(meta["test_list"])
        return f"{meta.get('test_setup_code', '')}\n{code}\n\n{tests}\n"
    raise ValueError(f"no code harness for source {item.source}")


def run_program(program: str, timeout_s: float = CODE_TIMEOUT_S) -> tuple[bool, str]:
    """Run a test program in a fresh interpreter; return (passed, reason)."""
    with tempfile.TemporaryDirectory(prefix="arbiter-grade-") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(GUARD + "\n" + program, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if result.returncode == 0:
        return True, "passed"
    lines = [line for line in result.stderr.strip().splitlines() if line.strip()]
    match = ERROR_TYPE.match(lines[-1]) if lines else None
    return False, match.group(1) if match else f"exit_{result.returncode}"


def grade_code(record: dict[str, Any], item: PromptItem) -> Grade:
    """Grade code by running the benchmark tests."""
    code = extract_code(final_text(record))
    if code is None:
        return Grade(False, "no_code", None)
    passed, reason = run_program(build_program(code, item))
    return Grade(passed, reason, None)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def grade(record: dict[str, Any], item: PromptItem) -> Grade:
    """Grade one generation; a response cut off by its token cap always fails.

    Raises
    ------
    ValueError
        If the item's family has no rule-based grader.
    """
    if record.get("done_reason") == "length":
        return Grade(False, "cap_hit", None)
    if item.family == "math":
        return grade_math(record, item)
    if item.family == "mcq":
        return grade_mcq(record, item)
    if item.family == "code":
        return grade_code(record, item)
    raise ValueError(f"family {item.family} needs the judge, not a rule-based grader")
