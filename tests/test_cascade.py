"""Unit tests for the reactive cascade baseline, on hand-computed outcomes."""

from __future__ import annotations

import numpy as np
import pytest

from arbiter.eval import cascade
from arbiter.eval.labels import cstar_cost, minimum_sufficient
from arbiter.eval.policies import Detection


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two prompts; cell k costs k + 1 and every verification costs 0.5.

    p0 succeeds from c3 upwards and its verifier doubts c1 but trusts c3. p1 succeeds
    only at c8, and its verifier wrongly trusts c1.
    """
    success = np.zeros((2, 8), dtype=bool)
    success[0, 2:] = True
    success[1, 7] = True
    cost = np.tile(np.arange(1.0, 9.0), (2, 1))
    verify_cost = np.full((2, 8), 0.5)
    confidence = np.full((2, 8), np.nan)
    confidence[0, [0, 2, 4, 6]] = [0.2, 0.9, 0.9, 0.9]
    confidence[1, [0, 2, 4, 6]] = [0.95, 0.1, 0.1, 0.1]
    return success, cost, verify_cost, confidence


def test_cascade_charges_passed_rungs_and_verification() -> None:
    """Passed rungs are escalation; the kept rung carries its own verification."""
    success, cost, verify_cost, confidence = _grid()
    outcome, run = cascade.cascade_outcome(success, cost, verify_cost, confidence, 0.8)
    assert run.stop.tolist() == [2, 0]
    assert outcome.cost == pytest.approx([1.5 + 3.5, 1.5 + 8.0])
    assert outcome.escalation == pytest.approx([1.5, 1.5])
    assert outcome.final_cost == pytest.approx([3.5, 8.0])
    assert outcome.solved == pytest.approx([1.0, 1.0])
    assert outcome.attempts == pytest.approx([2.0, 2.0])

    alone, _ = cascade.cascade_outcome(
        success, cost, verify_cost, confidence, 0.8, detection=Detection(p_accept_wrong=1.0)
    )
    assert alone.cost == pytest.approx([5.0, 1.5])
    assert alone.solved == pytest.approx([1.0, 0.0])


def test_cascade_threshold_extremes_and_missing_confidence() -> None:
    """Above every confidence the cascade climbs to c8; a NaN confidence never stops it."""
    success, cost, verify_cost, confidence = _grid()
    outcome, run = cascade.cascade_outcome(success, cost, verify_cost, confidence, 1.01)
    assert run.stop.tolist() == [7, 7]
    assert outcome.cost[0] == pytest.approx(1.5 + 3.5 + 5.5 + 7.5 + 8.0)
    assert outcome.attempts[0] == 5.0

    confidence[0, 2] = np.nan
    _, run = cascade.cascade_outcome(success, cost, verify_cost, confidence, 0.8)
    assert run.stop[0] == 4


def test_cascade_sweep_matches_outcomes() -> None:
    """Sweep rows reproduce the cascade's cost, tax and single-shot accuracy."""
    success, cost, verify_cost, confidence = _grid()
    cstar = minimum_sufficient(success, cost)
    best = cstar_cost(success, cost)
    frame = cascade.cascade_sweep(
        success, cost, verify_cost, confidence, cstar, best, [0.0, 0.8]
    ).set_index("tau")
    assert frame.loc[0.8, "mean_cost"] == pytest.approx((5.0 + 9.5) / 2)
    assert frame.loc[0.8, "tax"] == pytest.approx(((5.0 - 3.0) + (9.5 - 8.0)) / 2)
    assert frame.loc[0.8, "single_accuracy"] == pytest.approx(0.5)
    assert frame.loc[0.8, "single_mean_cost"] == pytest.approx((5.0 + 1.5) / 2)
    assert frame.loc[0.0, "rungs_passed"] == 0.0
    assert frame.loc[0.0, "router_cost"] == 0.0
