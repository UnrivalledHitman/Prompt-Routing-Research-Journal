"""Unit tests for the gate 2 parsing, grading, sampling and analysis helpers."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gate2_mode_separation.py"
_SPEC = importlib.util.spec_from_file_location("gate2_mode_separation", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load {_PATH}")
gate2 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gate2  # dataclasses resolve annotations through sys.modules
_SPEC.loader.exec_module(gate2)


def test_extract_boxed_handles_nesting_and_takes_last() -> None:
    """Nested braces survive and the last boxed answer wins."""
    assert gate2.extract_boxed(r"first \boxed{1} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert gate2.extract_boxed("no answer here") is None
    assert gate2.extract_boxed(r"\boxed{\frac{1}{2}") is None


def test_math_equal_normalises_formatting() -> None:
    """Thousands separators, frac variants and float forms compare equal."""
    assert gate2.math_equal("1,234", "1234")
    assert gate2.math_equal(r"\dfrac{14}{3}", r"\frac{14}{3}")
    assert gate2.math_equal("3.0", "3")
    assert gate2.math_equal(r"\text{5}", "5")
    assert not gate2.math_equal("2", "3")
    assert not gate2.math_equal(None, "3")


def test_extract_math_answer_falls_back_to_last_number() -> None:
    """Without a boxed answer, the last number in the text is used."""
    assert gate2.extract_math_answer("She has 3 bags, so 42 apples.") == "42"
    assert gate2.extract_math_answer(r"so \boxed{7} not 9") == "7"
    assert gate2.extract_math_answer("no digits") is None


def test_extract_choice_requires_capital_letter_answer_line() -> None:
    """Only an explicit capital-letter answer line counts."""
    assert gate2.extract_choice("Reasoning...\nAnswer: C") == "C"
    assert gate2.extract_choice("Answer: (B)") == "B"
    assert gate2.extract_choice("answer: D.") == "D"
    assert gate2.extract_choice("First answer: A\nAnswer: B") == "B"
    assert gate2.extract_choice("Answer: Blue") is None
    assert gate2.extract_choice("the answer a question needs") is None


def test_compile_markers_respects_word_boundaries() -> None:
    """Markers match whole words; comma-terminated markers need the comma."""
    pattern = gate2.compile_markers(["wait", "let me", "first,"])
    text = "Wait, let me check. Awaiting firstly. First, done."
    assert len(pattern.findall(text)) == 3


def test_stratified_sample_is_deterministic_and_balanced() -> None:
    """Same seed gives same draw, and strata are visited round-robin."""
    keys = ["a"] * 50 + ["b"] * 50 + ["c"] * 5
    first = gate2.stratified_sample(len(keys), 9, keys, random.Random(7))
    second = gate2.stratified_sample(len(keys), 9, keys, random.Random(7))
    assert first == second
    assert len(set(first)) == 9
    assert sorted(keys[i] for i in first) == ["a"] * 3 + ["b"] * 3 + ["c"] * 3
    with pytest.raises(ValueError):
        gate2.stratified_sample(3, 4, None, random.Random(0))


def _minimal_config() -> dict[str, Any]:
    return {
        "seed": 1,
        "ollama_host": "http://127.0.0.1:11434",
        "request_timeout_s": 10,
        "keep_alive": "1m",
        "num_ctx": 1024,
        "max_prompt_tokens": 512,
        "max_errors": 1,
        "bootstrap_resamples": 200,
        "tokenizer_id": "unused",
        "paths": {},
        "models": [{"size": "s", "tag": "m:s", "expected_context_length": 1}],
        "decoding": {"think": {"num_predict": 64}, "no_think": {"num_predict": 64}},
        "sources": {},
        "thresholds": {
            "auc_min": 0.9,
            "auc_partial_min": 0.75,
            "max_nothink_leak_rate": 0.05,
            "max_marker_density_ratio": 0.5,
        },
        "reasoning_markers": ["wait"],
        "gpu_guard": {"max_util_pct": 15, "util_window_s": 1},
    }


def test_load_config_rejects_asymmetric_generation_caps(tmp_path: Path) -> None:
    """Different num_predict per mode would manufacture separation, so it is refused."""
    cfg = _minimal_config()
    cfg["decoding"]["no_think"]["num_predict"] = 16
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="num_predict"):
        gate2.load_config(path)


def _record(prompt_id: str, mode: str, eval_count: int, *, thinking: str = "") -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "family": "math",
        "tag": "m:s",
        "mode": mode,
        "eval_count": eval_count,
        "eval_duration_s": 1.0,
        "thinking": thinking,
        "think_tag_in_content": False,
        "tokens_thinking": 100 if mode == "think" else 0,
        "tokens_content": 10,
        "markers_thinking": 5 if mode == "think" else 0,
        "markers_content": 0,
        "done_reason": "stop",
        "correct": True,
    }


def test_analyse_marks_clean_modes_as_separated() -> None:
    """Long think traces, short clean no-think answers: every check passes."""
    records = []
    for i in range(10):
        records.append(_record(f"p{i}", "think", 500 + i, thinking="wait, let me see"))
        records.append(_record(f"p{i}", "no_think", 20 + i))
    result = gate2.analyse(_minimal_config(), records)["sizes"]["s"]
    assert result["auc_think_longer"] == pytest.approx(1.0)
    assert result["nothink_leak_rate"] == 0.0
    assert result["status"] == "separated"


def test_analyse_flags_thinking_leakage_into_no_think() -> None:
    """Length still separates, but a leaked thinking field demotes the size to partial."""
    records = []
    for i in range(10):
        records.append(_record(f"p{i}", "think", 500 + i, thinking="wait"))
        records.append(_record(f"p{i}", "no_think", 20 + i, thinking="leaked reasoning"))
    result = gate2.analyse(_minimal_config(), records)["sizes"]["s"]
    assert result["nothink_leak_rate"] == 1.0
    assert result["checks"]["leak_rate"] is False
    assert result["status"] == "partial"


def test_analyse_reports_insufficient_data_without_pairs() -> None:
    """A size with no complete think/no-think pair is not given a verdict."""
    records = [_record("p0", "think", 100)]
    result = gate2.analyse(_minimal_config(), records)
    assert result["sizes"]["s"]["status"] == "insufficient_data"
    assert result["passed"] is False
