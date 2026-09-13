"""Escalation policies, routed policies and the escalation tax (experimental protocol, section 3).

A policy is a ladder of cells tried in order. On each attempt, the user accepts or
rejects the answer:

* **Oracle detection.** Every correct answer is accepted and every wrong one rejected.
* **Imperfect detection.** A wrong answer is accepted with probability ``p_accept_wrong``
  and a correct one rejected with probability ``p_reject_right``.

The last rung is always kept, because the ladder has run out. Outcomes are computed
as exact expectations rather than by simulation. For each prompt:

* ``cost``: expected total cost.
* ``escalation``: expected cost of discarded attempts.
* ``final_cost``: expected cost of the kept attempt.
* ``solved``: probability that the kept answer is correct.

Tax, escalation and stickiness then follow paper equation (4). Cells are indexed
0-7 in grid order, c1-c8.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

P1: tuple[int, ...] = (0, 2, 4, 6, 7)
P2: tuple[int, ...] = (0, 1, 3, 5, 7)
P3: tuple[int, ...] = (7,)
LADDERS: dict[str, tuple[int, ...]] = {"P1": P1, "P2": P2, "P3": P3}
TOP = 7
RECOVERIES = ("top", "climb")


@dataclass(frozen=True)
class Detection:
    """How reliably the user judges an answer."""

    p_accept_wrong: float = 0.0
    p_reject_right: float = 0.0

    def __post_init__(self) -> None:
        """Validate probabilities.

        Raises
        ------
        ValueError
            If either probability lies outside [0, 1].
        """
        for value in (self.p_accept_wrong, self.p_reject_right):
            if not 0.0 <= value <= 1.0:
                raise ValueError("detection probabilities must lie in [0, 1]")


ORACLE = Detection()


@dataclass(frozen=True)
class Outcome:
    """Expected per-prompt outcome of a policy; every array has shape (n,)."""

    cost: np.ndarray
    escalation: np.ndarray
    final_cost: np.ndarray
    solved: np.ndarray
    attempts: np.ndarray

    def tax(self, cstar_cost: np.ndarray) -> np.ndarray:
        """Return the escalation tax; NaN for unservable prompts."""
        return self.cost - cstar_cost

    def stickiness(self, cstar_cost: np.ndarray) -> np.ndarray:
        """Return the stickiness component; NaN for unservable prompts."""
        return self.final_cost - cstar_cost


def continuation(start: int, ladder: Sequence[int] = P1) -> tuple[int, ...]:
    """Start at ``start`` and continue with the rungs of ``ladder`` above it."""
    return (start, *(cell for cell in ladder if cell > start))


def ladder_outcome(
    success: np.ndarray, cost: np.ndarray, ladder: Sequence[int], detection: Detection = ORACLE
) -> Outcome:
    """Compute the expected outcome of one ladder for every prompt."""
    n = success.shape[0]
    reach = np.ones(n)
    total, escalation, final, solved, attempts = (np.zeros(n) for _ in range(5))
    for position, cell in enumerate(ladder):
        s = success[:, cell].astype(float)
        c = cost[:, cell]
        total += reach * c
        attempts += reach
        if position == len(ladder) - 1:
            final += reach * c
            solved += reach * s
            break
        accept = s * (1.0 - detection.p_reject_right) + (1.0 - s) * detection.p_accept_wrong
        final += reach * accept * c
        escalation += reach * (1.0 - accept) * c
        solved += reach * s * (1.0 - detection.p_reject_right)
        reach = reach * (1.0 - accept)
    return Outcome(total, escalation, final, solved, attempts)


def per_prompt_ladders(
    success: np.ndarray,
    cost: np.ndarray,
    ladders: Sequence[tuple[int, ...]],
    detection: Detection = ORACLE,
) -> Outcome:
    """Compute outcomes when each prompt follows its own ladder."""
    n = success.shape[0]
    fields = {
        name: np.zeros(n) for name in ("cost", "escalation", "final_cost", "solved", "attempts")
    }
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, ladder in enumerate(ladders):
        groups.setdefault(tuple(ladder), []).append(index)
    for ladder, indices in groups.items():
        rows = np.asarray(indices)
        outcome = ladder_outcome(success[rows], cost[rows], ladder, detection)
        for name in fields:
            fields[name][rows] = getattr(outcome, name)
    return Outcome(**fields)


def mixture(outcomes: Sequence[Outcome], weights: Sequence[float]) -> Outcome:
    """Weight outcomes, as when a population of users starts from different cells.

    Raises
    ------
    ValueError
        If the weights are negative or do not sum to one.
    """
    w = np.asarray(weights, dtype=float)
    if (w < 0).any() or not np.isclose(w.sum(), 1.0):
        raise ValueError("mixture weights must be non-negative and sum to 1")
    names = ("cost", "escalation", "final_cost", "solved", "attempts")
    return Outcome(
        **{
            name: sum(wi * getattr(o, name) for wi, o in zip(w, outcomes, strict=True))
            for name in names
        }
    )


def manual_policy(
    success: np.ndarray,
    cost: np.ndarray,
    name: str,
    detection: Detection = ORACLE,
    start_weights: Sequence[float] | None = None,
) -> Outcome:
    """Evaluate P1, P2, P3, or P4, which starts from a weighted cell then climbs P1.

    Raises
    ------
    ValueError
        If the policy is unknown or P4 has no start weights.
    """
    if name in LADDERS:
        return ladder_outcome(success, cost, LADDERS[name], detection)
    if name == "P4":
        if start_weights is None:
            raise ValueError("P4 needs start weights over the eight cells")
        outcomes = [ladder_outcome(success, cost, continuation(i), detection) for i in range(8)]
        return mixture(outcomes, start_weights)
    raise ValueError(f"unknown policy {name}")


def static_policy(success: np.ndarray, cost: np.ndarray, cell: int) -> Outcome:
    """Evaluate a policy that always uses one cell and never retries."""
    s = success[:, cell].astype(float)
    c = cost[:, cell]
    return Outcome(c.copy(), np.zeros_like(c), c.copy(), s, np.ones_like(c))


def routed_policy(
    success: np.ndarray,
    cost: np.ndarray,
    routed: np.ndarray,
    recovery: str = "top",
    detection: Detection = ORACLE,
) -> Outcome:
    """Try the routed cell, then recover on rejection.

    ``recovery="top"`` resubmits once to c8. ``recovery="climb"`` continues up the P1
    ladder from the routed cell. The router's own cost is not included here.

    Raises
    ------
    ValueError
        If the recovery rule is unknown.
    """
    if recovery not in RECOVERIES:
        raise ValueError(f"recovery must be one of {RECOVERIES}")
    if recovery == "top":
        ladders = [(int(r),) if r == TOP else (int(r), TOP) for r in routed]
    else:
        ladders = [continuation(int(r)) for r in routed]
    return per_prompt_ladders(success, cost, ladders, detection)


def select_cells(p_hat: np.ndarray, mean_cost: np.ndarray, tau: float) -> np.ndarray:
    """Pick the cheapest cell with predicted success at least ``tau`` (paper eq. 8).

    When no cell clears ``tau``, the cell with the highest predicted success is taken.
    """
    eligible = p_hat >= tau
    masked = np.where(eligible, mean_cost[None, :], np.inf)
    choice = np.argmin(masked, axis=1)
    none = ~eligible.any(axis=1)
    choice[none] = np.argmax(p_hat[none], axis=1)
    return choice


def oracle_outcome(success: np.ndarray, cost: np.ndarray, top: int = TOP) -> Outcome:
    """Route every prompt to ``c*(q)``; unservable prompts try ``top`` and fail.

    ``top`` is the last rung of the ladder, c8 in the local grid. The public-benchmark
    arm passes the top of its own pool.
    """
    masked = np.where(success, cost, np.inf)
    choice = np.argmin(masked, axis=1)
    choice[~success.any(axis=1)] = top
    rows = np.arange(len(choice))
    c = cost[rows, choice]
    s = success[rows, choice].astype(float)
    return Outcome(c, np.zeros_like(c), c.copy(), s, np.ones_like(c))
