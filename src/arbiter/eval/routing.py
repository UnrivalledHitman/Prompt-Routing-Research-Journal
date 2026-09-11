"""Router and policy evaluation (experimental protocol, section 8).

This module provides:

* cost-quality sweeps over the threshold ``tau``;
* normalised frontier area;
* the regret decomposition, which keeps over-routing cost and under-routing rate
  separate;
* bootstrap confidence intervals and paired p-values over prompts;
* Holm-Bonferroni correction.

All costs are per prompt in one unit. Router overhead is passed in explicitly, so gross
and net figures come from the same call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from arbiter.eval.policies import ORACLE, Detection, Outcome, routed_policy, select_cells

N_BOOT = 10_000


def mean_cell_cost(cost: np.ndarray) -> np.ndarray:
    """Return each cell's mean cost, the router's only cost knowledge before generation."""
    return np.nanmean(cost, axis=0)


def summarise(
    outcome: Outcome, cstar_cost: np.ndarray, router_cost: np.ndarray | float = 0.0
) -> dict[str, float]:
    """Summarise a policy outcome.

    ``tax``, ``escalation`` and ``stickiness`` are means over servable prompts and exclude
    router overhead. ``net_mean_cost`` and ``net_cost_per_solved`` include it.
    """
    servable = ~np.isnan(cstar_cost)
    overhead = np.broadcast_to(np.asarray(router_cost, dtype=float), outcome.cost.shape)
    net = outcome.cost + overhead
    solved = outcome.solved.sum()
    return {
        "mean_cost": float(outcome.cost.mean()),
        "net_mean_cost": float(net.mean()),
        "router_cost": float(overhead.mean()),
        "solved_rate": float(outcome.solved.mean()),
        "cost_per_solved": float(outcome.cost.sum() / solved) if solved else float("inf"),
        "net_cost_per_solved": float(net.sum() / solved) if solved else float("inf"),
        "tax": float(np.mean(outcome.cost[servable] - cstar_cost[servable])),
        "escalation": float(np.mean(outcome.escalation[servable])),
        "stickiness": float(np.mean(outcome.final_cost[servable] - cstar_cost[servable])),
        "attempts": float(outcome.attempts.mean()),
    }


def routing_errors(
    routed: np.ndarray, success: np.ndarray, cost: np.ndarray, cstar: np.ndarray
) -> dict[str, float]:
    """Decompose routing regret into its two kinds, never combined.

    Returns
    -------
    dict
        ``under_rate``: share of servable prompts routed to a failing cell.
        ``over_cost``: mean premium over ``c*`` where the routed cell succeeded.
        ``over_rate``: share of servable prompts routed to a successful cell costlier
        than ``c*``. ``exact_rate``: share routed exactly to ``c*``.
    """
    rows = np.arange(len(routed))
    servable = cstar >= 0
    ok = success[rows, routed]
    premium = cost[rows, routed] - cost[rows, np.maximum(cstar, 0)]
    over = servable & ok & (premium > 0)
    return {
        "under_rate": float(np.mean(~ok[servable])) if servable.any() else float("nan"),
        "over_rate": float(np.mean(over[servable])) if servable.any() else float("nan"),
        "over_cost": float(np.mean(premium[servable & ok])) if (servable & ok).any() else 0.0,
        "exact_rate": float(np.mean(routed[servable] == cstar[servable])),
    }


def sweep(
    p_hat: np.ndarray,
    success: np.ndarray,
    cost: np.ndarray,
    mean_cost: np.ndarray,
    cstar: np.ndarray,
    cstar_cost: np.ndarray,
    taus: Sequence[float],
    recovery: str = "top",
    detection: Detection = ORACLE,
    router_cost: np.ndarray | float = 0.0,
) -> pd.DataFrame:
    """Evaluate a router at every threshold, single-shot and with recovery."""
    rows = np.arange(len(p_hat))
    records = []
    for tau in taus:
        routed = select_cells(p_hat, mean_cost, tau)
        recovered = routed_policy(success, cost, routed, recovery, detection)
        record = {
            "tau": float(tau),
            "single_accuracy": float(success[rows, routed].mean()),
            "single_mean_cost": float(cost[rows, routed].mean()),
            **summarise(recovered, cstar_cost, router_cost),
            **routing_errors(routed, success, cost, cstar),
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def choose_tau(frame: pd.DataFrame, objective: str = "net_cost_per_solved") -> float:
    """Return the threshold minimising ``objective``; ties go to the higher threshold."""
    best = frame[frame[objective] == frame[objective].min()]
    return float(best["tau"].max())


def frontier_area(
    costs: Sequence[float], accuracies: Sequence[float], lo: float, hi: float
) -> float:
    """Mean best achievable accuracy over cost budgets in ``[lo, hi]``.

    For a budget ``x`` the achievable accuracy is the best accuracy among points
    costing at most ``x``, or zero if none does. Averaging over the budget range
    normalises the area to [0, 1] regardless of the unit.

    Raises
    ------
    ValueError
        If ``hi <= lo``.
    """
    if hi <= lo:
        raise ValueError("frontier range must have hi > lo")
    order = np.argsort(costs)
    c = np.clip(np.asarray(costs, dtype=float)[order], lo, hi)
    a = np.maximum.accumulate(np.asarray(accuracies, dtype=float)[order])
    edges = np.append(c, hi)
    return float(np.sum(a * np.diff(edges)) / (hi - lo))


def bootstrap(
    arrays: Mapping[str, np.ndarray],
    statistic: Callable[[Mapping[str, np.ndarray]], float],
    n_boot: int = N_BOOT,
    seed: int = 0,
    chunk: int = 500,
) -> np.ndarray:
    """Resample prompts with replacement and return the bootstrap distribution."""
    n = len(next(iter(arrays.values())))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        for b in range(size):
            idx = rng.integers(0, n, size=n)
            out[done + b] = statistic({k: v[idx] for k, v in arrays.items()})
        done += size
    return out


def percentile_ci(
    arrays: Mapping[str, np.ndarray],
    statistic: Callable[[Mapping[str, np.ndarray]], float],
    n_boot: int = N_BOOT,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Return the point estimate and a percentile bootstrap interval."""
    point = statistic(arrays)
    dist = bootstrap(arrays, statistic, n_boot, seed)
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.nanpercentile(dist, [tail, 100.0 - tail])
    return float(point), float(lo), float(hi)


def paired_p_value(difference: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> float:
    """Two-sided bootstrap p-value that the mean per-prompt difference is zero.

    Obtained by inverting the percentile interval: twice the smaller share of bootstrap
    means on either side of zero, floored at one over ``n_boot``.
    """
    dist = bootstrap({"d": difference}, lambda a: float(np.mean(a["d"])), n_boot, seed)
    share = min(np.mean(dist <= 0.0), np.mean(dist >= 0.0))
    return float(min(1.0, max(2.0 * share, 1.0 / n_boot)))


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm-Bonferroni adjusted p-values in the input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted
