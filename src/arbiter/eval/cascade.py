"""Reactive cascade baseline in the style of AutoMix (paper section III-F).

The cascade climbs the P1 ladder. After each attempt below the top rung, the model
that answered is asked whether its answer is correct, and the cascade stops at the
first attempt whose verification confidence reaches the threshold. The top rung is
kept without verification. The answer the cascade stops at then meets the user as a
routed answer does: it is accepted, or rejected and recovered.

Verification calls are charged like generations. The verification of the rung the
cascade stops at belongs to that attempt; the answers and verifications of the rungs
it passes are escalation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from arbiter.eval.policies import ORACLE, P1, Detection, Outcome, routed_policy
from arbiter.eval.routing import routing_errors, summarise


@dataclass(frozen=True)
class Verification:
    """Verifier results on one prompt set, cells in grid order.

    Attributes
    ----------
    confidence : ndarray, shape (n, 8)
        Confidence that each cell's answer is correct; NaN where no verification ran.
    costs : dict of str to ndarray, shape (n, 8)
        Cost of each verification call, per unit.
    """

    confidence: np.ndarray
    costs: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class Climb:
    """Where the cascade stopped for each prompt and what it spent getting there.

    Attributes
    ----------
    stop : ndarray of int, shape (n,)
        Cell whose answer the cascade returns.
    passed_cost : ndarray, shape (n,)
        Cost of the answers and verifications of the rungs passed on the way.
    passed : ndarray, shape (n,)
        Number of rungs passed.
    cost : ndarray, shape (n, 8)
        Cell costs with the stopping rung's verification added.
    """

    stop: np.ndarray
    passed_cost: np.ndarray
    passed: np.ndarray
    cost: np.ndarray


def climb(
    confidence: np.ndarray,
    cost: np.ndarray,
    verify_cost: np.ndarray,
    threshold: float,
    ladder: Sequence[int] = P1,
) -> Climb:
    """Run the cascade's automatic part: climb until a verification reaches ``threshold``.

    A missing confidence (NaN) never reaches the threshold.
    """
    n = confidence.shape[0]
    rows = np.arange(n)
    stop = np.full(n, ladder[-1], dtype=int)
    passed_cost = np.zeros(n)
    passed = np.zeros(n)
    climbing = np.ones(n, dtype=bool)
    for cell in ladder[:-1]:
        accept = climbing & (np.nan_to_num(confidence[:, cell], nan=-np.inf) >= threshold)
        stop[accept] = cell
        reject = climbing & ~accept
        passed_cost[reject] += cost[reject, cell] + verify_cost[reject, cell]
        passed[reject] += 1
        climbing = reject
    charged = np.array(cost, dtype=float)
    verified = stop != ladder[-1]
    charged[rows[verified], stop[verified]] += verify_cost[rows[verified], stop[verified]]
    return Climb(stop, passed_cost, passed, charged)


def cascade_outcome(
    success: np.ndarray,
    cost: np.ndarray,
    verify_cost: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    recovery: str = "top",
    detection: Detection = ORACLE,
    ladder: Sequence[int] = P1,
) -> tuple[Outcome, Climb]:
    """Evaluate the cascade followed by the user's check and recovery.

    ``Detection(p_accept_wrong=1.0)`` gives the cascade alone, with no user check.
    """
    run = climb(confidence, cost, verify_cost, threshold, ladder)
    routed = routed_policy(success, run.cost, run.stop, recovery, detection)
    outcome = Outcome(
        cost=routed.cost + run.passed_cost,
        escalation=routed.escalation + run.passed_cost,
        final_cost=routed.final_cost,
        solved=routed.solved,
        attempts=routed.attempts + run.passed,
    )
    return outcome, run


def cascade_sweep(
    success: np.ndarray,
    cost: np.ndarray,
    verify_cost: np.ndarray,
    confidence: np.ndarray,
    cstar: np.ndarray,
    best: np.ndarray,
    thresholds: Sequence[float],
    recovery: str = "top",
    detection: Detection = ORACLE,
) -> pd.DataFrame:
    """Evaluate the cascade at every threshold, in the columns of a router sweep.

    ``tau`` holds the threshold. ``single_accuracy`` and ``single_mean_cost`` describe the
    cascade alone, before the user's check; verification is part of the policy cost, so
    ``router_cost`` is zero.
    """
    rows = np.arange(len(success))
    records = []
    for threshold in thresholds:
        outcome, run = cascade_outcome(
            success, cost, verify_cost, confidence, threshold, recovery, detection
        )
        records.append(
            {
                "tau": float(threshold),
                "single_accuracy": float(success[rows, run.stop].mean()),
                "single_mean_cost": float((run.passed_cost + run.cost[rows, run.stop]).mean()),
                "rungs_passed": float(run.passed.mean()),
                **summarise(outcome, best),
                **routing_errors(run.stop, success, cost, cstar),
            }
        )
    return pd.DataFrame.from_records(records)
