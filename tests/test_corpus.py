"""Unit tests for corpus selection."""

from __future__ import annotations

import random

import pytest

from arbiter.data import corpus


class FakeTokenizer:
    """Counts whitespace-separated words as tokens."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Return one token per word."""
        return [0] * len(text.split())


def _rows(n: int) -> list[dict[str, str]]:
    return [{"question": f"question number {i}", "answer": f"#### {i}"} for i in range(n)]


def test_stratified_order_is_a_balanced_permutation() -> None:
    """Every index appears once and a prefix alternates between strata."""
    keys = ["a"] * 6 + ["b"] * 3
    order = corpus.stratified_order(keys, random.Random(1))
    assert sorted(order) == list(range(9))
    prefix = [keys[i] for i in order[:6]]
    assert prefix.count("a") == 3
    assert prefix.count("b") == 3


def test_select_source_nests_pilot_and_noise() -> None:
    """Pilot items come first, and the noise subset is the head of the pilot."""
    spec = {"family": "math", "formatter": "gsm8k", "pilot": 4, "main": 6, "noise": 2}
    items, stats = corpus.select_source(
        "gsm8k", spec, _rows(20), FakeTokenizer(), random.Random(3), set(), set(), 100
    )
    assert [item.subset for item in items] == ["pilot"] * 4 + ["main"] * 6
    assert [item.noise for item in items] == [True] * 2 + [False] * 8
    assert len({item.prompt_id for item in items}) == 10
    assert stats["taken"] == 10


def test_select_source_skips_excluded_duplicate_and_long_rows() -> None:
    """Excluded ids, normalised duplicates and over-length prompts are never taken."""
    rows = _rows(6)
    rows[1] = {"question": "QUESTION   number 0", "answer": "#### 0"}
    rows[2] = {"question": "word " * 50, "answer": "#### 2"}
    spec = {"family": "math", "formatter": "gsm8k", "pilot": 1, "main": 2}
    items, stats = corpus.select_source(
        "gsm8k", spec, rows, FakeTokenizer(), random.Random(0), {"gsm8k-3"}, set(), 20
    )
    ids = {item.prompt_id for item in items}
    assert {"gsm8k-4", "gsm8k-5"} <= ids
    assert len(ids & {"gsm8k-0", "gsm8k-1"}) == 1
    assert not ids & {"gsm8k-2", "gsm8k-3"}
    assert stats["taken"] == 3


def test_select_source_raises_when_quota_cannot_be_filled() -> None:
    """A source with too few eligible rows fails loudly."""
    spec = {"family": "math", "formatter": "gsm8k", "pilot": 3, "main": 3}
    with pytest.raises(ValueError, match="quota"):
        corpus.select_source(
            "gsm8k", spec, _rows(5), FakeTokenizer(), random.Random(0), set(), set(), 100
        )


def test_select_source_rejects_noise_larger_than_pilot() -> None:
    """The noise subset must fit inside the pilot."""
    spec = {"family": "math", "formatter": "gsm8k", "pilot": 1, "main": 1, "noise": 2}
    with pytest.raises(ValueError, match="noise"):
        corpus.select_source(
            "gsm8k", spec, _rows(5), FakeTokenizer(), random.Random(0), set(), set(), 100
        )


def test_text_digest_ignores_case_and_whitespace() -> None:
    """Digests match across case and whitespace changes only."""
    assert corpus.text_digest("Hello  World\n") == corpus.text_digest("hello world")
    assert corpus.text_digest("hello world") != corpus.text_digest("hello worlds")


def test_strata_keys_bins_numeric_columns() -> None:
    """Integer binning groups nearby task ids; no column gives one stratum."""
    rows = [{"task_id": 11}, {"task_id": 150}, {"task_id": 199}]
    assert corpus.strata_keys(rows, {"stratify_by": "task_id", "stratify_bin": 100}) == [
        "0",
        "1",
        "1",
    ]
    assert corpus.strata_keys(rows, {}) == ["all", "all", "all"]
