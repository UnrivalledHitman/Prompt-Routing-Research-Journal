"""Unit tests for judge agreement statistics and the validation sample."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from arbiter.eval import agreement


def test_kappa_matches_sklearn() -> None:
    """Unweighted and quadratic kappa agree with scikit-learn."""
    rng = np.random.default_rng(0)
    a = rng.integers(1, 6, 200)
    b = np.clip(a + rng.integers(-1, 2, 200), 1, 5)
    labels = [1, 2, 3, 4, 5]
    expected = cohen_kappa_score(a, b, labels=labels)
    assert agreement.cohen_kappa(a, b, labels) == pytest.approx(expected)
    expected_q = cohen_kappa_score(a, b, labels=labels, weights="quadratic")
    assert agreement.cohen_kappa(a, b, labels, "quadratic") == pytest.approx(expected_q)


def test_kappa_is_undefined_when_both_raters_are_constant() -> None:
    """Perfect chance agreement leaves kappa undefined rather than raising."""
    assert np.isnan(agreement.cohen_kappa([1, 1, 1], [1, 1, 1], [0, 1]))


def test_kappa_rejects_unknown_labels() -> None:
    """A rating outside the label set is an error, not a silent miscount."""
    with pytest.raises(ValueError, match="not in"):
        agreement.cohen_kappa([1, 6], [1, 2], [1, 2, 3, 4, 5])


def test_summary_thresholds_scores_at_four() -> None:
    """The success decision is a score of 4 or more, as in grading."""
    local, human = [5, 4, 3, 2, 4, 1], [4, 5, 2, 3, 3, 1]
    summary = agreement.agreement_summary(local, human, n_boot=200, seed=1)
    assert summary["n"] == 6
    assert summary["percent_agreement"] == pytest.approx(5 / 6)
    assert summary["kappa_binary"] == pytest.approx(2 / 3)
    low, high = summary["kappa_binary_ci"]
    assert low <= high
    assert summary == agreement.agreement_summary(local, human, n_boot=200, seed=1)


@pytest.mark.parametrize(
    ("kappa", "status"),
    [
        (0.85, "primary"),
        (0.60, "primary"),
        (0.5999, "caveat"),
        (0.40, "caveat"),
        (0.3999, "secondary"),
        (-0.1, "secondary"),
    ],
)
def test_open_family_status(kappa: float, status: str) -> None:
    """The pre-registered bands map kappa to how the family is reported."""
    assert agreement.open_family_status(kappa) == status


def test_status_refuses_undefined_kappa() -> None:
    """An undefined kappa cannot decide the family's status."""
    with pytest.raises(ValueError, match="undefined"):
        agreement.open_family_status(float("nan"))


def test_allocation_is_proportional_with_one_per_stratum() -> None:
    """Counts sum to n, every stratum gets one, and larger strata get more."""
    alloc = agreement.allocate({"a": 900, "b": 90, "c": 10}, 100)
    assert sum(alloc.values()) == 100
    assert alloc["a"] > alloc["b"] > alloc["c"] >= 1


def test_sample_is_reproducible_and_covers_strata() -> None:
    """The same seed and salt draw the same distinct indices from every stratum."""
    strata = ["x"] * 50 + ["y"] * 30 + ["z"] * 5
    first = agreement.stratified_sample(strata, 20, seed=3, salt="frontier")
    assert first == agreement.stratified_sample(strata, 20, seed=3, salt="frontier")
    assert first != agreement.stratified_sample(strata, 20, seed=3, salt="human")
    assert len(set(first)) == 20
    assert {strata[i] for i in first} == {"x", "y", "z"}


def test_sample_takes_everything_when_asked_for_more() -> None:
    """Asking for more than the population returns all of it."""
    assert agreement.stratified_sample(["x", "y"], 5, seed=0, salt="s") == [0, 1]
