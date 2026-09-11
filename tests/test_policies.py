"""Unit tests for escalation policies and the escalation tax."""

from __future__ import annotations

import numpy as np
import pytest

from arbiter.eval import policies
from arbiter.eval.labels import cstar_cost

COST = np.arange(1.0, 9.0)[None, :]  # c1 costs 1, ..., c8 costs 8
SOLVED_FROM_C3 = np.array([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=bool)


def test_frugal_climb_pays_escalation_only() -> None:
    """P1 fails at c1, succeeds at c3 = c*: tax 1, all of it escalation."""
    outcome = policies.manual_policy(SOLVED_FROM_C3, COST, "P1")
    best = cstar_cost(SOLVED_FROM_C3, COST)
    assert outcome.cost.tolist() == [4.0]
    assert outcome.escalation.tolist() == [1.0]
    assert outcome.tax(best).tolist() == [1.0]
    assert outcome.stickiness(best).tolist() == [0.0]
    assert outcome.solved.tolist() == [1.0]


def test_think_first_overshoots_and_sticky_high_pays_stickiness_only() -> None:
    """P2 escalates past c3 to c4; P3 pays the full premium of c8 over c*."""
    best = cstar_cost(SOLVED_FROM_C3, COST)
    p2 = policies.manual_policy(SOLVED_FROM_C3, COST, "P2")
    assert (p2.cost[0], p2.escalation[0], p2.stickiness(best)[0]) == (7.0, 3.0, 1.0)
    p3 = policies.manual_policy(SOLVED_FROM_C3, COST, "P3")
    assert (p3.cost[0], p3.escalation[0], p3.stickiness(best)[0]) == (8.0, 0.0, 5.0)


def test_imperfect_detection_expectation_is_exact() -> None:
    """With every answer correct and half rejected, P1's expectation matches hand algebra."""
    success = np.ones((1, 8), dtype=bool)
    detection = policies.Detection(p_accept_wrong=0.0, p_reject_right=0.5)
    outcome = policies.manual_policy(success, COST, "P1", detection)
    assert outcome.cost[0] == pytest.approx(1 + 0.5 * 3 + 0.25 * 5 + 0.125 * 7 + 0.0625 * 8)
    assert outcome.escalation[0] + outcome.final_cost[0] == pytest.approx(outcome.cost[0])
    assert outcome.solved[0] == pytest.approx(1.0)


def test_accepting_wrong_answers_leaves_prompts_unsolved() -> None:
    """A user who always accepts stops at c1 and fails."""
    outcome = policies.manual_policy(
        SOLVED_FROM_C3, COST, "P1", policies.Detection(p_accept_wrong=1.0)
    )
    assert (outcome.cost[0], outcome.solved[0]) == (1.0, 0.0)


def test_p4_mixes_starts_and_climbs_p1() -> None:
    """P4 is the weighted mean of ladders that start at each cell and continue up P1."""
    assert policies.continuation(1) == (1, 2, 4, 6, 7)
    assert policies.continuation(7) == (7,)
    weights = [0.5, 0, 0, 0, 0, 0, 0, 0.5]
    outcome = policies.manual_policy(SOLVED_FROM_C3, COST, "P4", start_weights=weights)
    assert outcome.cost[0] == pytest.approx(0.5 * 4.0 + 0.5 * 8.0)
    with pytest.raises(ValueError):
        policies.manual_policy(SOLVED_FROM_C3, COST, "P4", start_weights=[1.0] * 8)


def test_routed_policy_recovery_rules() -> None:
    """A misroute to c1 recovers at c8 directly, or climbs P1 and stops at c3."""
    routed = np.array([0])
    top = policies.routed_policy(SOLVED_FROM_C3, COST, routed, "top")
    climb = policies.routed_policy(SOLVED_FROM_C3, COST, routed, "climb")
    assert (top.cost[0], top.escalation[0]) == (9.0, 1.0)
    assert (climb.cost[0], climb.escalation[0]) == (4.0, 1.0)
    exact = policies.routed_policy(SOLVED_FROM_C3, COST, np.array([2]), "top")
    assert (exact.cost[0], exact.attempts[0]) == (3.0, 1.0)


def test_per_prompt_ladders_match_row_by_row_evaluation() -> None:
    """Grouping prompts by ladder gives the same numbers as evaluating each alone."""
    rng = np.random.default_rng(0)
    success = rng.random((40, 8)) < 0.5
    cost = rng.random((40, 8)) * 10
    routed = rng.integers(0, 8, size=40)
    grouped = policies.routed_policy(success, cost, routed, "climb")
    for i in range(40):
        single = policies.ladder_outcome(
            success[i : i + 1], cost[i : i + 1], policies.continuation(int(routed[i]))
        )
        assert grouped.cost[i] == pytest.approx(single.cost[0])
        assert grouped.solved[i] == pytest.approx(single.solved[0])


def test_select_cells_threshold_and_fallback() -> None:
    """The cheapest cell above tau wins; with none above tau, the most confident cell wins."""
    p_hat = np.array([[0.2, 0.9, 0.95, 0.1, 0, 0, 0, 0.99], [0.1] * 8])
    p_hat[1, 5] = 0.3
    mean_cost = np.array([1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0])
    assert policies.select_cells(p_hat, mean_cost, 0.8).tolist() == [2, 5]
    assert policies.select_cells(p_hat, mean_cost, 0.0).tolist() == [0, 0]


def test_oracle_and_static_policies() -> None:
    """The oracle pays exactly c*; a static policy never retries."""
    oracle = policies.oracle_outcome(SOLVED_FROM_C3, COST)
    assert (oracle.cost[0], oracle.solved[0]) == (3.0, 1.0)
    static = policies.static_policy(SOLVED_FROM_C3, COST, 0)
    assert (static.cost[0], static.solved[0]) == (1.0, 0.0)


def test_detection_validates_probabilities() -> None:
    """Probabilities outside [0, 1] are rejected."""
    with pytest.raises(ValueError):
        policies.Detection(p_accept_wrong=1.5)
