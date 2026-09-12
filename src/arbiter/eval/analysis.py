"""Analyses behind the results section (experimental protocol, sections 1, 3, 8 and 13).

The results section follows the hypotheses, and so does this module:

* corpus and oracle labels: :func:`label_summary`, :func:`cstar_distribution`;
* H1, the escalation tax: :func:`policy_table`, :func:`detection_sweep`;
* H2 and H3, routing gross and net of router cost: :func:`run_router`,
  :func:`run_cascade`, :func:`baseline_evaluations`, :func:`compare`,
  :func:`frontier_table`;
* H4, the cost of the reasoning-mode axis: :func:`mode_signature`;
* H5, asymmetric misallocation: :func:`asymmetry`.

Uncertainty comes from one matrix of bootstrap prompt indices per prompt set. Every
statistic on that set is computed on the same resamples, so differences between
policies are paired by construction.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from arbiter.energy.carbon import J_PER_KWH, carbon_g
from arbiter.eval.cascade import Verification, cascade_outcome, cascade_sweep
from arbiter.eval.labels import (
    CELL_NAMES,
    CELLS,
    LabelTable,
    cell_index,
    cstar_cost,
    minimum_sufficient,
)
from arbiter.eval.policies import (
    LADDERS,
    ORACLE,
    P1,
    TOP,
    Detection,
    Outcome,
    manual_policy,
    mixture,
    oracle_outcome,
    routed_policy,
    select_cells,
    static_policy,
)
from arbiter.eval.routing import (
    choose_tau,
    evaluate_routed,
    frontier_area,
    holm,
    mean_cell_cost,
    routing_errors,
    summarise,
    sweep,
)
from arbiter.routers.classical import cost_sensitive_weights

LOGGER = logging.getLogger("arbiter.analysis")
SIZES = ("0.6b", "1.7b", "4b", "8b")
STATISTICS = (
    "mean_cost",
    "net_mean_cost",
    "solved_rate",
    "cost_per_solved",
    "net_cost_per_solved",
    "tax",
    "relative_tax",
    "escalation",
    "stickiness",
    "attempts",
)
MANUAL = ("P1", "P2", "P3", "P4-uniform")
COMPARED_BASELINES = (*MANUAL, "random")
BLOCK = 4_000_000  # resampled values held at once, which bounds peak memory


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resamples:
    """Bootstrap resamples of one prompt set, shared by every statistic on it.

    Attributes
    ----------
    index : ndarray of int, shape (n_boot, n)
        Prompt indices drawn with replacement, one row per resample.
    level : float
        Confidence level of percentile intervals.
    """

    index: np.ndarray
    level: float = 0.95

    @classmethod
    def draw(cls, n: int, n_boot: int, seed: int, level: float = 0.95) -> Resamples:
        """Draw ``n_boot`` resamples of ``n`` prompts."""
        rng = np.random.default_rng(seed)
        return cls(rng.integers(0, n, size=(n_boot, n), dtype=np.int32), level)

    @property
    def n_boot(self) -> int:
        """Number of resamples."""
        return int(self.index.shape[0])

    def interval(self, dist: np.ndarray) -> tuple[float, float]:
        """Return the percentile interval of a bootstrap distribution, ignoring NaN."""
        finite = dist[~np.isnan(dist)]
        if finite.size == 0:
            return float("nan"), float("nan")
        tail = (1.0 - self.level) / 2.0 * 100.0
        lo, hi = np.percentile(finite, [tail, 100.0 - tail])
        return float(lo), float(hi)

    def p_value(self, dist: np.ndarray) -> float:
        """Return the two-sided p-value that a paired difference is zero.

        The percentile interval is inverted: twice the smaller share of resamples on
        either side of zero, floored at one over the number of resamples.
        """
        finite = dist[~np.isnan(dist)]
        if finite.size == 0:
            return float("nan")
        share = min(np.mean(finite <= 0.0), np.mean(finite >= 0.0))
        return float(min(1.0, max(2.0 * share, 1.0 / self.n_boot)))


def _full(n: int) -> np.ndarray:
    return np.arange(n)[None, :]


def _blocks(index: np.ndarray) -> Iterator[np.ndarray]:
    """Split the resample matrix into row blocks, so a gather stays inside cache and RAM.

    The blocks are contiguous and used in order, so every statistic is unchanged.
    """
    rows = max(1, BLOCK // max(int(index.shape[1]), 1))
    for start in range(0, index.shape[0], rows):
        yield index[start : start + rows]


def _mean(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.concatenate([np.nanmean(values[block], axis=1) for block in _blocks(index)])


def _ratio(numerator: np.ndarray, denominator: np.ndarray, index: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.concatenate(
            [
                np.nansum(numerator[block], axis=1) / np.nansum(denominator[block], axis=1)
                for block in _blocks(index)
            ]
        )


def _divide(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _statistics(
    outcome: Outcome, best: np.ndarray, router_cost: float | np.ndarray, index: np.ndarray
) -> dict[str, np.ndarray]:
    servable = ~np.isnan(best)
    overhead = np.broadcast_to(np.asarray(router_cost, dtype=float), outcome.cost.shape)
    net = outcome.cost + overhead
    return {
        "mean_cost": _mean(outcome.cost, index),
        "net_mean_cost": _mean(net, index),
        "solved_rate": _mean(outcome.solved, index),
        "cost_per_solved": _ratio(outcome.cost, outcome.solved, index),
        "net_cost_per_solved": _ratio(net, outcome.solved, index),
        "tax": _mean(np.where(servable, outcome.cost - best, np.nan), index),
        "relative_tax": _ratio(np.where(servable, outcome.cost, np.nan), best, index) - 1.0,
        "escalation": _mean(np.where(servable, outcome.escalation, np.nan), index),
        "stickiness": _mean(np.where(servable, outcome.final_cost - best, np.nan), index),
        "attempts": _mean(outcome.attempts, index),
    }


@dataclass(frozen=True)
class Evaluated:
    """A policy on one prompt set.

    ``row`` maps each statistic to its point estimate, with ``_lo`` and ``_hi`` interval
    bounds. ``boot`` holds each statistic's bootstrap distribution for paired
    comparisons on the same resamples.
    """

    row: dict[str, float]
    boot: dict[str, np.ndarray]


def evaluate(
    outcome: Outcome,
    best: np.ndarray,
    resamples: Resamples,
    router_cost: float | np.ndarray = 0.0,
) -> Evaluated:
    """Return point estimates, percentile intervals and bootstrap distributions.

    Parameters
    ----------
    outcome : Outcome
        Expected per-prompt outcome of the policy.
    best : ndarray
        Cost of ``c*(q)`` per prompt, NaN where unservable.
    resamples : Resamples
        Resamples of the same prompt set.
    router_cost : float or ndarray
        Router overhead per prompt, in the same unit.
    """
    point = _statistics(outcome, best, router_cost, _full(len(outcome.cost)))
    boot = _statistics(outcome, best, router_cost, resamples.index)
    row: dict[str, float] = {}
    for name in STATISTICS:
        lo, hi = resamples.interval(boot[name])
        row.update({name: float(point[name][0]), f"{name}_lo": lo, f"{name}_hi": hi})
    return Evaluated(row, boot)


# ---------------------------------------------------------------------------
# Corpus and oracle labels
# ---------------------------------------------------------------------------


def scopes(table: LabelTable) -> Iterator[tuple[str, LabelTable]]:
    """Yield the whole table as ``all``, then one sub-table per task family."""
    yield "all", table
    families = table.prompts["family"].to_numpy()
    for family in sorted(set(families)):
        yield str(family), table.take(families == family)


def label_summary(table: LabelTable, units: Sequence[str]) -> pd.DataFrame:
    """Tabulate accuracy, cap hits, contamination and cost per cell and scope."""
    rows = []
    for scope, sub in scopes(table):
        for cell, (size, mode) in enumerate(CELLS):
            solved = int(sub.success[:, cell].sum())
            row: dict[str, Any] = {
                "scope": scope,
                "cell": CELL_NAMES[cell],
                "size": size,
                "mode": mode,
                "n": len(sub),
                "accuracy": float(sub.success[:, cell].mean()),
                "cap_rate": float(sub.capped[:, cell].mean()),
                "contaminated_rate": float(sub.contaminated[:, cell].mean()),
            }
            for unit in units:
                total = float(sub.costs[unit][:, cell].sum())
                row[f"mean_{unit}"] = total / len(sub)
                row[f"{unit}_per_solved"] = total / solved if solved else float("inf")
            rows.append(row)
    return pd.DataFrame(rows)


def cstar_distribution(table: LabelTable, units: Sequence[str]) -> pd.DataFrame:
    """Count prompts by minimum sufficient cell, per scope and unit."""
    rows = []
    for scope, sub in scopes(table):
        for unit in units:
            choice = minimum_sufficient(sub.success, sub.costs[unit])
            for cell, label in [(-1, "unservable"), *enumerate(CELL_NAMES)]:
                count = int(np.sum(choice == cell))
                rows.append(
                    {
                        "scope": scope,
                        "unit": unit,
                        "cstar": label,
                        "count": count,
                        "share": count / len(sub),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H1: the escalation tax
# ---------------------------------------------------------------------------


def manual_outcomes(
    success: np.ndarray,
    cost: np.ndarray,
    detection: Detection,
    p4_weights: Mapping[str, Sequence[float]],
) -> dict[str, Outcome]:
    """Evaluate P1-P3, P4 under each start distribution, and random routing.

    The random router picks a cell uniformly and recovers a rejected answer at c8.
    """
    outcomes = {name: manual_policy(success, cost, name, detection) for name in LADDERS}
    for label, weights in p4_weights.items():
        outcomes[f"P4-{label}"] = manual_policy(success, cost, "P4", detection, weights)
    n = success.shape[0]
    uniform = [
        routed_policy(success, cost, np.full(n, cell), "top", detection)
        for cell in range(len(CELLS))
    ]
    outcomes["random"] = mixture(uniform, np.full(len(CELLS), 1.0 / len(CELLS)))
    return outcomes


def reference_outcomes(success: np.ndarray, cost: np.ndarray) -> dict[str, Outcome]:
    """Evaluate the eight static policies and the oracle, which need no detection model."""
    outcomes = {
        f"always-{CELL_NAMES[cell]}": static_policy(success, cost, cell)
        for cell in range(len(CELLS))
    }
    outcomes["oracle"] = oracle_outcome(success, cost)
    return outcomes


def policy_table(
    table: LabelTable,
    units: Sequence[str],
    detections: Mapping[str, Detection],
    p4_weights: Mapping[str, Sequence[float]],
    n_boot: int,
    seed: int,
    level: float = 0.95,
) -> pd.DataFrame:
    """Tabulate tax, escalation, stickiness and cost per solved prompt for every baseline.

    Rows cover each scope, unit and detection model. Static policies and the oracle
    appear once, under the oracle detection model.
    """
    rows = []
    for scope, sub in scopes(table):
        resamples = Resamples.draw(len(sub), n_boot, seed, level)
        for unit in units:
            cost = sub.costs[unit]
            best = cstar_cost(sub.success, cost)
            for label, detection in detections.items():
                outcomes = manual_outcomes(sub.success, cost, detection, p4_weights)
                if detection == ORACLE:
                    outcomes |= reference_outcomes(sub.success, cost)
                for policy, outcome in outcomes.items():
                    evaluated = evaluate(outcome, best, resamples)
                    rows.append(
                        {
                            "scope": scope,
                            "unit": unit,
                            "detection": label,
                            "policy": policy,
                            "n": len(sub),
                            **evaluated.row,
                        }
                    )
    return pd.DataFrame(rows)


def carbon_table(
    policies: pd.DataFrame,
    intensities: Mapping[str, float],
    unit: str = "energy_j",
    per: float = 1000.0,
) -> pd.DataFrame:
    """Convert the energy rows of a policy table to carbon at each grid intensity.

    One row per scope, detection model, policy and intensity. The conversion is a
    multiplication, so a figure's interval bounds carry over to carbon unchanged and no
    resampling is repeated here. ``per`` is the number of queries the carbon is reported
    for; 1,000 follows the convention of Luccioni et al.
    """
    rows = policies[policies["unit"] == unit]
    out = []
    for name, intensity in intensities.items():
        for _, row in rows.iterrows():
            record = {
                "scope": row["scope"],
                "unit": unit,
                "detection": row["detection"],
                "policy": row["policy"],
                "grid": name,
                "intensity_g_per_kwh": float(intensity),
                "kwh": row["mean_cost"] * per / J_PER_KWH,
            }
            for statistic in ("mean_cost", "tax", "escalation", "stickiness"):
                target = "carbon" if statistic == "mean_cost" else f"carbon_{statistic}"
                for suffix in ("", "_lo", "_hi"):
                    value = row.get(f"{statistic}{suffix}")
                    if value is not None:
                        record[f"{target}{suffix}"] = carbon_g(value * per, intensity)
            out.append(record)
    return pd.DataFrame(out)


def detection_sweep(
    table: LabelTable,
    unit: str,
    p_accept_wrong: Sequence[float],
    p_reject_right: Sequence[float],
    p4_weights: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """Recompute the manual policies over a grid of detection error rates (point estimates)."""
    cost = table.costs[unit]
    best = cstar_cost(table.success, cost)
    rows = []
    for accept in p_accept_wrong:
        for reject in p_reject_right:
            detection = Detection(accept, reject)
            outcomes = manual_outcomes(table.success, cost, detection, p4_weights)
            for policy, outcome in outcomes.items():
                rows.append(
                    {
                        "unit": unit,
                        "p_accept_wrong": accept,
                        "p_reject_right": reject,
                        "policy": policy,
                        **summarise(outcome, best),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H4: the reasoning-mode axis
# ---------------------------------------------------------------------------


def cell_ratio(
    cost: np.ndarray, solved: np.ndarray, a: int, b: int, resamples: Resamples
) -> tuple[float, float, float]:
    """Return cost per solved prompt of cell ``a`` over cell ``b``, with its interval."""
    full = _full(cost.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        point = _ratio(cost[:, a], solved[:, a], full) / _ratio(cost[:, b], solved[:, b], full)
        boot = _ratio(cost[:, a], solved[:, a], resamples.index) / _ratio(
            cost[:, b], solved[:, b], resamples.index
        )
    lo, hi = resamples.interval(boot)
    return float(point[0]), lo, hi


def mode_signature(
    table: LabelTable, unit: str, n_boot: int, seed: int, level: float = 0.95
) -> pd.DataFrame:
    """Compare thinking with not thinking, and with a size step, in cost per solved prompt.

    For each size, ``think_over_no_think`` is the thinking cell's cost per solved prompt
    over the non-thinking cell's. ``think_over_size_step`` divides it by the next size's
    non-thinking cell; above one, thinking costs more than moving up a size.
    """
    rows = []
    for scope, sub in scopes(table):
        resamples = Resamples.draw(len(sub), n_boot, seed, level)
        cost = sub.costs[unit]
        solved = sub.success.astype(float)
        full = _full(len(sub))
        for position, size in enumerate(SIZES):
            no_think, think = cell_index(size, "no_think"), cell_index(size, "think")
            row: dict[str, Any] = {
                "scope": scope,
                "unit": unit,
                "size": size,
                "n": len(sub),
                "acc_no_think": float(solved[:, no_think].mean()),
                "acc_think": float(solved[:, think].mean()),
                "mean_no_think": float(cost[:, no_think].mean()),
                "mean_think": float(cost[:, think].mean()),
            }
            with np.errstate(divide="ignore", invalid="ignore"):
                row["per_solved_no_think"] = float(
                    _ratio(cost[:, no_think], solved[:, no_think], full)[0]
                )
                row["per_solved_think"] = float(_ratio(cost[:, think], solved[:, think], full)[0])
            point, lo, hi = cell_ratio(cost, solved, think, no_think, resamples)
            row.update(think_over_no_think=point, think_over_no_think_lo=lo)
            row["think_over_no_think_hi"] = hi
            if position + 1 < len(SIZES):
                step = cell_index(SIZES[position + 1], "no_think")
                row["next_size"] = SIZES[position + 1]
                row["acc_next_no_think"] = float(solved[:, step].mean())
                point, lo, hi = cell_ratio(cost, solved, think, step, resamples)
                row.update(think_over_size_step=point, think_over_size_step_lo=lo)
                row["think_over_size_step_hi"] = hi
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H5: asymmetric misallocation
# ---------------------------------------------------------------------------


def _neighbour_premium(
    success: np.ndarray,
    cost: np.ndarray,
    cstar: np.ndarray,
    best: np.ndarray,
    order: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Price routing each servable prompt one tier away from ``c*``; NaN where no such tier."""
    tier = np.empty(len(order), dtype=int)
    tier[order] = np.arange(len(order))
    premium = np.full(len(cstar), np.nan)
    failed = np.full(len(cstar), np.nan)
    rows = np.flatnonzero(cstar >= 0)
    target = tier[cstar[rows]] + step
    valid = (target >= 0) & (target < len(order))
    rows, cells = rows[valid], order[target[valid]]
    ok = success[rows, cells]
    recovery = np.where(ok | (cells == TOP), 0.0, cost[rows, TOP])
    premium[rows] = cost[rows, cells] + recovery - best[rows]
    failed[rows] = (~ok).astype(float)
    return premium, failed


def asymmetry(
    table: LabelTable, unit: str, n_boot: int, seed: int, level: float = 0.95
) -> pd.DataFrame:
    """Price one tier too low against one tier too high.

    Tiers are cells ordered by mean cost. Each servable prompt is sent to the tier just
    below and just above its ``c*(q)``; a failed answer is recovered at c8 under oracle
    detection. The premium is the realised cost minus the cost of ``c*(q)``.
    """
    rows = []
    for scope, sub in scopes(table):
        resamples = Resamples.draw(len(sub), n_boot, seed, level)
        full = _full(len(sub))
        cost = sub.costs[unit]
        cstar = minimum_sufficient(sub.success, cost)
        best = cstar_cost(sub.success, cost)
        order = np.argsort(mean_cell_cost(cost), kind="stable")
        low, low_failed = _neighbour_premium(sub.success, cost, cstar, best, order, -1)
        high, high_failed = _neighbour_premium(sub.success, cost, cstar, best, order, +1)
        row: dict[str, Any] = {"scope": scope, "unit": unit, "n": len(sub)}
        for name, values in (("too_low", low), ("too_high", high)):
            lo, hi = resamples.interval(_mean(values, resamples.index))
            row.update(
                {
                    f"n_{name}": int(np.sum(~np.isnan(values))),
                    f"{name}_premium": float(_mean(values, full)[0]),
                    f"{name}_premium_lo": lo,
                    f"{name}_premium_hi": hi,
                }
            )
        row["too_low_failure_rate"] = float(_mean(low_failed, full)[0])
        row["too_high_failure_rate"] = float(_mean(high_failed, full)[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = _mean(low, resamples.index) / _mean(high, resamples.index)
        row["low_over_high"] = _divide(row["too_low_premium"], row["too_high_premium"])
        row["low_over_high_lo"], row["low_over_high_hi"] = resamples.interval(ratio)
        row["mean_cstar_cost"] = float(np.nanmean(best)) if (cstar >= 0).any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H2 and H3: routers against baselines, gross and net of router cost
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouterSpec:
    """A router and its validation grid.

    ``make(seed=..., **params)`` returns an unfitted router. Per-cell routers expose
    ``fit(texts, success, weights)`` and ``predict_proba(texts)``; direct routers expose
    ``fit(texts, cstar)`` and ``predict(texts)``.

    ``overhead(table, unit)``, when given, returns the router's measured cost for each
    prompt of a split, as for R3, whose decisions were metered on the GPU. Otherwise the
    router is timed on the CPU and charged by :func:`router_cost`.
    """

    name: str
    family: str
    make: Callable[..., Any]
    grid: tuple[dict[str, Any], ...] = ({},)
    cost_sensitive: bool = False
    direct: bool = False
    overhead: Callable[[LabelTable, str], np.ndarray] | None = None
    device: str = "cpu"


@dataclass(frozen=True)
class Split:
    """One split of the label table, with prompt texts in the same order."""

    table: LabelTable
    texts: list[str]


@dataclass(frozen=True)
class RouterSettings:
    """Evaluation settings shared by every router.

    ``label_unit`` defines ``c*`` for direct labels, cost-sensitive weights and the
    validation grid search. The first recovery rule is the primary one.
    """

    units: tuple[str, ...]
    label_unit: str
    taus: tuple[float, ...]
    seeds: tuple[int, ...]
    detections: Mapping[str, Detection]
    recoveries: tuple[str, ...] = ("top", "climb")
    objective: str = "net_cost_per_solved"
    cpu_power_w: float = 54.0


@dataclass
class RouterRun:
    """Everything one router produced.

    ``evaluated`` holds the first seed's test evaluations, keyed by unit, recovery rule
    and detection label, for paired comparisons.
    """

    name: str
    selection: list[dict[str, Any]] = field(default_factory=list)
    test: list[dict[str, Any]] = field(default_factory=list)
    sweeps: list[pd.DataFrame] = field(default_factory=list)
    overhead: list[dict[str, Any]] = field(default_factory=list)
    evaluated: dict[tuple[str, str, str], Evaluated] = field(default_factory=dict)


def fit_router(
    spec: RouterSpec, params: Mapping[str, Any], seed: int, split: Split, unit: str
) -> Any:  # noqa: ANN401
    """Fit a router on a split, with labels and weights defined in ``unit``."""
    router = spec.make(seed=seed, **params)
    success, cost = split.table.success, split.table.costs[unit]
    if spec.direct:
        return router.fit(split.texts, minimum_sufficient(success, cost))
    weights = cost_sensitive_weights(success, cost) if spec.cost_sensitive else None
    return router.fit(split.texts, success, weights)


def decide(
    router: Any,  # noqa: ANN401
    spec: RouterSpec,
    texts: Sequence[str],
    mean_cost: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Return the routed cell for each prompt."""
    if spec.direct:
        return np.asarray(router.predict(texts), dtype=int)
    return select_cells(router.predict_proba(texts), mean_cost, tau)


def decision_times(
    router: Any,  # noqa: ANN401
    spec: RouterSpec,
    texts: Sequence[str],
    mean_cost: np.ndarray,
) -> np.ndarray:
    """Time routing decisions one prompt at a time, as a deployed router makes them."""
    decide(router, spec, list(texts[:1]), mean_cost, 0.5)
    times = np.empty(len(texts))
    for i, text in enumerate(texts):
        start = time.perf_counter()
        decide(router, spec, [text], mean_cost, 0.5)
        times[i] = time.perf_counter() - start
    return times


def router_cost(unit: str, decision_s: float, cpu_power_w: float) -> float:
    """Charge a CPU router for one decision in ``unit``.

    A CPU router generates no tokens. Its energy is bounded above by decision time
    times the processor's power limit, charged in both energy units.
    """
    if unit.startswith("tokens"):
        return 0.0
    if unit == "latency_s":
        return decision_s
    return decision_s * cpu_power_w


def charged_cost(
    spec: RouterSpec, split: Split, seconds: np.ndarray | None, unit: str, cpu_power_w: float
) -> float | np.ndarray:
    """Return a router's cost per prompt on a split in ``unit``.

    A router with measured overhead is charged what was measured for each prompt; a CPU
    router is charged its mean decision time through :func:`router_cost`.
    """
    if spec.overhead is not None:
        return np.asarray(spec.overhead(split.table, unit), dtype=float)
    return router_cost(unit, float(np.mean(seconds)), cpu_power_w)


def overhead_record(
    spec: RouterSpec, seed: int, split: Split, seconds: np.ndarray | None, cpu_power_w: float
) -> dict[str, Any]:
    """Summarise a router's decision latency and energy on a split."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if spec.overhead is not None:
            seconds = np.asarray(spec.overhead(split.table, "latency_s"), dtype=float)
            joules = float(np.nanmean(spec.overhead(split.table, "energy_j")))
            vram = float("nan")
        else:
            joules = float(np.mean(seconds)) * cpu_power_w
            vram = 0.0
        return {
            "router": spec.name,
            "family": spec.family,
            "seed": seed,
            "device": spec.device,
            "p50_s": float(np.nanpercentile(seconds, 50)),
            "p95_s": float(np.nanpercentile(seconds, 95)),
            "mean_s": float(np.nanmean(seconds)),
            "peak_vram_mb": vram,
            "joules_per_decision": joules,
        }


def _view(table: LabelTable, unit: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cost = table.costs[unit]
    return (
        table.success,
        cost,
        minimum_sufficient(table.success, cost),
        cstar_cost(table.success, cost),
    )


def select_params(
    spec: RouterSpec, train: Split, val: Split, settings: RouterSettings
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pick the grid point with the best validation objective.

    The objective is its best value over ``tau`` in the label unit, with primary recovery,
    oracle detection and no router overhead. A one-point grid is returned unfitted.
    """
    if len(spec.grid) == 1:
        record = {"router": spec.name, "params": repr(spec.grid[0])}
        return dict(spec.grid[0]), [record | {"val_objective": float("nan"), "chosen": True}]
    success, cost, cstar, best = _view(val.table, settings.label_unit)
    mean_cost = mean_cell_cost(train.table.costs[settings.label_unit])
    recovery = settings.recoveries[0]
    records = []
    for params in spec.grid:
        router = fit_router(spec, params, settings.seeds[0], train, settings.label_unit)
        if spec.direct:
            routed = decide(router, spec, val.texts, mean_cost, float("nan"))
            score = evaluate_routed(routed, success, cost, cstar, best, recovery)[
                settings.objective
            ]
        else:
            p_hat = router.predict_proba(val.texts)
            frame = sweep(p_hat, success, cost, mean_cost, cstar, best, settings.taus, recovery)
            score = frame[settings.objective].min()
        records.append(
            {
                "router": spec.name,
                "params": repr(params),
                "val_objective": float(score),
                "chosen": False,
            }
        )
    chosen = int(np.argmin([record["val_objective"] for record in records]))
    records[chosen]["chosen"] = True
    return dict(spec.grid[chosen]), records


def run_router(
    spec: RouterSpec,
    train: Split,
    val: Split,
    test: Split,
    settings: RouterSettings,
    resamples: Resamples,
) -> RouterRun:
    """Select parameters, then fit, time and evaluate a router for every seed.

    For every unit, recovery rule and detection model, ``tau`` is chosen on validation
    with router overhead charged, and the test split is evaluated once at that ``tau``.
    Test sweeps over every ``tau`` are kept for frontiers, with primary recovery and
    oracle detection.
    """
    run = RouterRun(name=spec.name)
    params, run.selection = select_params(spec, train, val, settings)
    primary = settings.recoveries[0]
    for seed in settings.seeds:
        router = fit_router(spec, params, seed, train, settings.label_unit)
        mean_costs = {unit: mean_cell_cost(train.table.costs[unit]) for unit in settings.units}
        anchor = mean_costs[settings.units[0]]
        val_s = test_s = None
        if spec.overhead is None:
            val_s = decision_times(router, spec, val.texts, anchor)
            test_s = decision_times(router, spec, test.texts, anchor)
        run.overhead.append(overhead_record(spec, seed, test, test_s, settings.cpu_power_w))
        p_val = None if spec.direct else router.predict_proba(val.texts)
        p_test = None if spec.direct else router.predict_proba(test.texts)
        direct = decide(router, spec, test.texts, anchor, float("nan")) if spec.direct else None
        for unit in settings.units:
            v_success, v_cost, v_cstar, v_best = _view(val.table, unit)
            t_success, t_cost, t_cstar, t_best = _view(test.table, unit)
            v_overhead = charged_cost(spec, val, val_s, unit, settings.cpu_power_w)
            t_overhead = charged_cost(spec, test, test_s, unit, settings.cpu_power_w)
            for recovery in settings.recoveries:
                for label, detection in settings.detections.items():
                    if spec.direct:
                        tau, routed = float("nan"), direct
                    else:
                        frame = sweep(
                            p_val, v_success, v_cost, mean_costs[unit], v_cstar, v_best,
                            settings.taus, recovery, detection, v_overhead,
                        )  # fmt: skip
                        tau = choose_tau(frame, settings.objective)
                        routed = select_cells(p_test, mean_costs[unit], tau)
                    outcome = routed_policy(t_success, t_cost, routed, recovery, detection)
                    evaluated = evaluate(outcome, t_best, resamples, t_overhead)
                    rows = np.arange(len(routed))
                    run.test.append(
                        {
                            "router": spec.name,
                            "family": spec.family,
                            "seed": seed,
                            "unit": unit,
                            "recovery": recovery,
                            "detection": label,
                            "tau": tau,
                            "params": repr(params),
                            "router_cost": float(np.mean(t_overhead)),
                            "single_accuracy": float(t_success[rows, routed].mean()),
                            "single_mean_cost": float(t_cost[rows, routed].mean()),
                            **routing_errors(routed, t_success, t_cost, t_cstar),
                            **evaluated.row,
                        }
                    )
                    if seed == settings.seeds[0]:
                        run.evaluated[(unit, recovery, label)] = evaluated
            if spec.direct:
                record = evaluate_routed(
                    direct, t_success, t_cost, t_cstar, t_best, primary, ORACLE, t_overhead
                )
                frame = pd.DataFrame([{"tau": float("nan"), **record}])
            else:
                frame = sweep(
                    p_test, t_success, t_cost, mean_costs[unit], t_cstar, t_best,
                    settings.taus, primary, ORACLE, t_overhead,
                )  # fmt: skip
            run.sweeps.append(
                frame.assign(router=spec.name, family=spec.family, seed=seed, unit=unit)
            )
    return run


CASCADE = "cascade-automix"


def run_cascade(
    val: Split,
    test: Split,
    val_checks: Verification,
    test_checks: Verification,
    settings: RouterSettings,
    thresholds: Sequence[float],
    resamples: Resamples,
    name: str = CASCADE,
) -> RouterRun:
    """Choose the cascade's confidence threshold on validation and evaluate it on test.

    The cascade is run like a router: for every unit, recovery rule and detection model
    the threshold is chosen on validation, ties going to the higher threshold, and the
    test split is evaluated once at it. Test sweeps over every threshold use primary
    recovery and oracle detection. Verification is part of the policy cost, so
    ``router_cost`` is zero; the overhead record describes one verification call. The
    cascade is deterministic and runs under the first seed only.
    """
    run = RouterRun(name=name)
    run.selection = [
        {"router": name, "params": repr({}), "val_objective": float("nan"), "chosen": True}
    ]
    seed, primary = settings.seeds[0], settings.recoveries[0]
    rows = np.arange(len(test.table))
    for unit in settings.units:
        v_success, v_cost, v_cstar, v_best = _view(val.table, unit)
        t_success, t_cost, t_cstar, t_best = _view(test.table, unit)
        v_inputs = (v_success, v_cost, val_checks.costs[unit], val_checks.confidence)
        t_inputs = (t_success, t_cost, test_checks.costs[unit], test_checks.confidence)
        for recovery in settings.recoveries:
            for label, detection in settings.detections.items():
                frame = cascade_sweep(*v_inputs, v_cstar, v_best, thresholds, recovery, detection)
                threshold = choose_tau(frame, settings.objective)
                outcome, climbed = cascade_outcome(*t_inputs, threshold, recovery, detection)
                evaluated = evaluate(outcome, t_best, resamples)
                stop = climbed.stop
                run.test.append(
                    {
                        "router": name,
                        "family": "cascade",
                        "seed": seed,
                        "unit": unit,
                        "recovery": recovery,
                        "detection": label,
                        "tau": threshold,
                        "params": repr({}),
                        "router_cost": 0.0,
                        "single_accuracy": float(t_success[rows, stop].mean()),
                        "single_mean_cost": float(
                            (climbed.passed_cost + climbed.cost[rows, stop]).mean()
                        ),
                        "rungs_passed": float(climbed.passed.mean()),
                        **routing_errors(stop, t_success, t_cost, t_cstar),
                        **evaluated.row,
                    }
                )
                run.evaluated[(unit, recovery, label)] = evaluated
        frame = cascade_sweep(*t_inputs, t_cstar, t_best, thresholds, primary, ORACLE)
        run.sweeps.append(frame.assign(router=name, family="cascade", seed=seed, unit=unit))
    verified = list(P1[:-1])
    missing = np.array([np.nan])
    calls = {
        unit: test_checks.costs[unit][:, verified].ravel() if unit in test_checks.costs else missing
        for unit in ("latency_s", "energy_j")
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run.overhead.append(
            {
                "router": name,
                "family": "cascade",
                "seed": seed,
                "device": "gpu",
                "p50_s": float(np.nanpercentile(calls["latency_s"], 50)),
                "p95_s": float(np.nanpercentile(calls["latency_s"], 95)),
                "mean_s": float(np.nanmean(calls["latency_s"])),
                "peak_vram_mb": float("nan"),
                "joules_per_decision": float(np.nanmean(calls["energy_j"])),
            }
        )
    return run


def baseline_evaluations(
    table: LabelTable,
    units: Sequence[str],
    detections: Mapping[str, Detection],
    p4_weights: Mapping[str, Sequence[float]],
    resamples: Resamples,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], Evaluated]]:
    """Evaluate every baseline on one split, keyed by unit, detection label and policy."""
    rows, evaluated = [], {}
    for unit in units:
        cost = table.costs[unit]
        best = cstar_cost(table.success, cost)
        for label, detection in detections.items():
            outcomes = manual_outcomes(table.success, cost, detection, p4_weights)
            if detection == ORACLE:
                outcomes |= reference_outcomes(table.success, cost)
            for policy, outcome in outcomes.items():
                result = evaluate(outcome, best, resamples)
                evaluated[(unit, label, policy)] = result
                rows.append({"unit": unit, "detection": label, "policy": policy, **result.row})
    return pd.DataFrame(rows), evaluated


def compare(
    candidates: Mapping[str, Evaluated],
    baselines: Mapping[str, Evaluated],
    resamples: Resamples,
) -> pd.DataFrame:
    """Pair each candidate with each baseline on shared resamples.

    ``saving`` is the baseline's mean cost minus the candidate's, router cost included;
    ``gross_saving`` leaves router cost out; ``relative_saving`` divides ``saving`` by the
    baseline's mean cost. ``solved_difference`` is the candidate's solved rate minus the
    baseline's. ``cost_per_solved_ratio`` divides the candidate's net cost per solved
    prompt by the baseline's. P-values are Holm-corrected across all pairs in the call.
    """
    rows = []
    for cand_name, cand in candidates.items():
        for base_name, base in baselines.items():
            c, b = cand.row, base.row
            cb, bb = cand.boot, base.boot
            with np.errstate(divide="ignore", invalid="ignore"):
                dists = {
                    "saving": bb["net_mean_cost"] - cb["net_mean_cost"],
                    "gross_saving": bb["mean_cost"] - cb["mean_cost"],
                    "relative_saving": 1.0 - cb["net_mean_cost"] / bb["net_mean_cost"],
                    "solved_difference": cb["solved_rate"] - bb["solved_rate"],
                    "cost_per_solved_ratio": cb["net_cost_per_solved"] / bb["net_cost_per_solved"],
                }
            points = {
                "saving": b["net_mean_cost"] - c["net_mean_cost"],
                "gross_saving": b["mean_cost"] - c["mean_cost"],
                "relative_saving": 1.0 - _divide(c["net_mean_cost"], b["net_mean_cost"]),
                "solved_difference": c["solved_rate"] - b["solved_rate"],
                "cost_per_solved_ratio": _divide(
                    c["net_cost_per_solved"], b["net_cost_per_solved"]
                ),
            }
            row: dict[str, Any] = {"candidate": cand_name, "baseline": base_name}
            for name, dist in dists.items():
                lo, hi = resamples.interval(dist)
                row.update({name: float(points[name]), f"{name}_lo": lo, f"{name}_hi": hi})
            row["saving_p"] = resamples.p_value(dists["saving"])
            row["solved_difference_p"] = resamples.p_value(dists["solved_difference"])
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["saving_p_holm"] = holm(frame["saving_p"].to_numpy())
        frame["solved_difference_p_holm"] = holm(frame["solved_difference_p"].to_numpy())
    return frame


def frontier_table(sweeps: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """Normalised area under each cost-quality frontier on the test split.

    Budgets run over the mean-cost range of the eight static policies in each unit.
    Router curves use net mean cost against solved rate with primary recovery; the
    single-shot area uses the routed cell alone, router cost included. Static, manual
    and oracle rows give reference areas.
    """
    rows = []
    for unit, group in sweeps.groupby("unit"):
        in_unit = baselines[(baselines["unit"] == unit) & (baselines["detection"] == "oracle")]
        static = in_unit[in_unit["policy"].str.startswith("always-")]
        lo, hi = float(static["mean_cost"].min()), float(static["mean_cost"].max())
        if not hi > lo:
            LOGGER.warning("static policies span no cost range in %s; no frontier areas", unit)
            continue
        references = {
            "static": static,
            "manual": in_unit[in_unit["policy"].isin(MANUAL)],
            "oracle": in_unit[in_unit["policy"] == "oracle"],
        }
        bounds = {"unit": unit, "lo": lo, "hi": hi}
        for name, points in references.items():
            if points.empty:
                continue
            area = frontier_area(points["mean_cost"], points["solved_rate"], lo, hi)
            rows.append({"curve": name, "seed": np.nan, "area": area, **bounds})
        for (router, seed), frame in group.groupby(["router", "seed"]):
            rows.append(
                {
                    "curve": router,
                    "seed": seed,
                    "area": frontier_area(frame["net_mean_cost"], frame["solved_rate"], lo, hi),
                    "single_shot_area": frontier_area(
                        frame["single_mean_cost"] + frame["router_cost"],
                        frame["single_accuracy"],
                        lo,
                        hi,
                    ),
                    **bounds,
                }
            )
    return pd.DataFrame(rows)
