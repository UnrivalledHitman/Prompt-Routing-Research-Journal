"""The escalation tax on a published benchmark (experimental protocol, section 9, stage 5).

The local pool stops at 8B parameters. This module recomputes the same quantities on the
per-query outcomes published with LLMRouterBench, where the pool is thirteen flagship
models and the costs are the providers' own dollar charges. Nothing is generated here; the
records are read from the table written by ``scripts/build_public_arm.py``.

Two things differ from the local grid and are deliberate:

* **The ladder.** The grid has a natural size-and-mode order; a pool of thirteen models
  does not. A user escalating through it would not step onto a model that costs more than
  one already tried and answers no better, so the ladder is the cost-accuracy frontier of
  the pool, in cost order. Its top rung is both the costliest and the most accurate model
  on the frontier, which is the role c8 plays locally.
* **The policies.** P1, P3 and P4 carry over. P2 turns thinking on before moving up a
  size and has no analogue here, since the pool has no mode axis; the thinking and
  non-thinking releases of Qwen3-235B-A22B are compared separately by :func:`mode_pair`.

Model columns are ordered by mean cost throughout, so a column index is a rung.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from arbiter.eval.analysis import Resamples, cell_ratio, evaluate
from arbiter.eval.labels import cstar_cost
from arbiter.eval.policies import (
    ORACLE,
    Detection,
    Outcome,
    continuation,
    ladder_outcome,
    mixture,
    oracle_outcome,
    static_policy,
)

UNITS = ("dollars", "tokens")
KEYS = ("dataset", "index")


@dataclass(frozen=True)
class PublicTable:
    """Per-query outcomes of one model pool, with models ordered by mean cost.

    Attributes
    ----------
    queries : DataFrame
        One row per query, with ``dataset`` and ``index``.
    models : tuple of str
        Pool members in cost order; position is the rung.
    success : ndarray of bool, shape (n, k)
    costs : dict of str to ndarray, shape (n, k)
        One matrix per unit in :data:`UNITS`.
    """

    queries: pd.DataFrame
    models: tuple[str, ...]
    success: np.ndarray
    costs: dict[str, np.ndarray]

    def __len__(self) -> int:
        """Return the number of queries."""
        return len(self.queries)

    def take(self, mask: np.ndarray) -> PublicTable:
        """Return the queries selected by a boolean mask."""
        return PublicTable(
            queries=self.queries.iloc[np.flatnonzero(mask)].reset_index(drop=True),
            models=self.models,
            success=self.success[mask],
            costs={unit: matrix[mask] for unit, matrix in self.costs.items()},
        )


def build_table(records: pd.DataFrame, pool: Sequence[str], threshold: float) -> PublicTable:
    """Pivot the record table into success and cost matrices, in cost order.

    A query counts as answered when its score reaches ``threshold``. Token cost is the
    prompt and completion tokens of that query, so that a re-asked query pays its prompt
    tokens again, as in the local grid.

    Raises
    ------
    ValueError
        If a model of the pool is missing, or a (model, query) pair is repeated.
    """
    frame = records[records["model"].isin(list(pool))]
    missing = set(pool) - set(frame["model"])
    if missing:
        raise ValueError(f"records are missing {sorted(missing)}")
    if frame.duplicated([*KEYS, "model"]).any():
        raise ValueError("repeated (model, query) rows")
    frame = frame.assign(tokens=frame["prompt_tokens"] + frame["completion_tokens"])

    def matrix(column: str) -> pd.DataFrame:
        return frame.pivot(index=list(KEYS), columns="model", values=column).sort_index()

    dollars, tokens, score = matrix("cost"), matrix("tokens"), matrix("score")
    order = np.argsort(dollars.to_numpy().mean(axis=0), kind="stable")
    models = tuple(dollars.columns[order])
    queries = dollars.index.to_frame(index=False)
    return PublicTable(
        queries=queries,
        models=models,
        success=score.to_numpy()[:, order] >= threshold,
        costs={
            "dollars": dollars.to_numpy()[:, order],
            "tokens": tokens.to_numpy(dtype=float)[:, order],
        },
    )


def scopes(table: PublicTable) -> Iterator[tuple[str, PublicTable]]:
    """Yield the whole table as ``all``, then one sub-table per dataset."""
    yield "all", table
    datasets = table.queries["dataset"].to_numpy()
    for dataset in sorted(set(datasets)):
        yield str(dataset), table.take(datasets == dataset)


def frontier(mean_cost: np.ndarray, accuracy: np.ndarray) -> tuple[int, ...]:
    """Return the rungs a rational escalator would use, in cost order.

    A model joins the ladder when it answers more queries than every cheaper model on it.
    """
    rungs, best = [], -np.inf
    for index in np.argsort(mean_cost, kind="stable"):
        if accuracy[index] > best:
            rungs.append(int(index))
            best = accuracy[index]
    return tuple(rungs)


def pool_summary(table: PublicTable, units: Sequence[str] = UNITS) -> pd.DataFrame:
    """Tabulate accuracy and cost per model and scope, and mark the ladder."""
    rows = []
    for scope, sub in scopes(table):
        accuracy = sub.success.mean(axis=0)
        rungs = frontier(sub.costs["dollars"].mean(axis=0), accuracy)
        for index, model in enumerate(sub.models):
            solved = int(sub.success[:, index].sum())
            row: dict[str, Any] = {
                "scope": scope,
                "model": model,
                "rung": index,
                "n": len(sub),
                "accuracy": float(accuracy[index]),
                "on_ladder": index in rungs,
            }
            for unit in units:
                total = float(sub.costs[unit][:, index].sum())
                row[f"mean_{unit}"] = total / len(sub)
                row[f"{unit}_per_solved"] = total / solved if solved else float("inf")
            rows.append(row)
    return pd.DataFrame(rows)


def policy_outcomes(
    success: np.ndarray,
    cost: np.ndarray,
    ladder: Sequence[int],
    models: Sequence[str],
    detection: Detection = ORACLE,
    extra: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Outcome]:
    """Evaluate the escalation policies, the static ones and the oracle.

    ``P1`` climbs the ladder, ``P3`` uses its top rung once, ``P4-uniform`` starts
    uniformly over the pool and then climbs, and ``random`` picks a model uniformly and
    recovers at the top rung. A static ``always-<model>`` policy is reported for each rung
    of the ladder; :func:`pool_summary` covers the models off it.

    ``extra`` names further ladders to climb, for a pool whose models have a structure the
    cost frontier does not see. A pool of sizes crossed with a cheap and a costly mode has
    a mode-first ladder that the frontier will not contain whenever the costly mode is not
    worth its price on average, which is exactly the comparison such a pool exists to make.
    """
    k, top = success.shape[1], ladder[-1]
    weights = np.full(k, 1.0 / k)
    climbs = [ladder_outcome(success, cost, continuation(i, ladder), detection) for i in range(k)]
    retries = [
        ladder_outcome(success, cost, (i,) if i == top else (i, top), detection) for i in range(k)
    ]
    outcomes = {
        "P1": ladder_outcome(success, cost, ladder, detection),
        "P3": static_policy(success, cost, top),
        "P4-uniform": mixture(climbs, weights),
        "random": mixture(retries, weights),
    }
    for name, rungs in (extra or {}).items():
        outcomes[name] = ladder_outcome(success, cost, rungs, detection)
    if detection == ORACLE:
        outcomes |= {f"always-{models[i]}": static_policy(success, cost, i) for i in ladder}
        outcomes["oracle"] = oracle_outcome(success, cost, top)
    return outcomes


def tax_table(
    table: PublicTable,
    units: Sequence[str] = UNITS,
    detections: Mapping[str, Detection] | None = None,
    n_boot: int = 10000,
    seed: int = 0,
    level: float = 0.95,
    extra: Mapping[str, Sequence[int]] | None = None,
) -> pd.DataFrame:
    """Tabulate tax, escalation, stickiness and cost per solved query for every policy.

    ``extra`` passes further named ladders through to :func:`policy_outcomes`.
    """
    detections = detections or {"oracle": ORACLE}
    rows = []
    for scope, sub in scopes(table):
        resamples = Resamples.draw(len(sub), n_boot, seed, level)
        ladder = frontier(sub.costs["dollars"].mean(axis=0), sub.success.mean(axis=0))
        for unit in units:
            cost = sub.costs[unit]
            best = cstar_cost(sub.success, cost)
            for label, detection in detections.items():
                outcomes = policy_outcomes(sub.success, cost, ladder, sub.models, detection, extra)
                for policy, outcome in outcomes.items():
                    rows.append(
                        {
                            "scope": scope,
                            "unit": unit,
                            "detection": label,
                            "policy": policy,
                            "n": len(sub),
                            **evaluate(outcome, best, resamples).row,
                        }
                    )
    return pd.DataFrame(rows)


def cstar_distribution(table: PublicTable, units: Sequence[str] = UNITS) -> pd.DataFrame:
    """Count queries by minimum sufficient model, per scope and unit."""
    rows = []
    for scope, sub in scopes(table):
        for unit in units:
            masked = np.where(sub.success, sub.costs[unit], np.inf)
            choice = np.argmin(masked, axis=1)
            servable = sub.success.any(axis=1)
            counts = {"unservable": int((~servable).sum())}
            for index, model in enumerate(sub.models):
                counts[model] = int(np.sum(servable & (choice == index)))
            for label, count in counts.items():
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


def mode_pair(
    table: PublicTable,
    thinking: str,
    plain: str,
    units: Sequence[str] = UNITS,
    n_boot: int = 10000,
    seed: int = 0,
    level: float = 0.95,
) -> pd.DataFrame:
    """Compare a thinking release with its non-thinking sibling, as H4 does locally.

    ``needs_thinking`` is the share of queries the thinking release answers and the plain
    one does not, and ``plain_suffices`` the share the plain one already answers.

    Raises
    ------
    ValueError
        If either model is not in the pool.
    """
    for model in (thinking, plain):
        if model not in table.models:
            raise ValueError(f"{model} is not in the pool")
    a, b = table.models.index(thinking), table.models.index(plain)
    rows = []
    for scope, sub in scopes(table):
        resamples = Resamples.draw(len(sub), n_boot, seed, level)
        solved = sub.success.astype(float)
        for unit in units:
            cost = sub.costs[unit]
            point, lo, hi = cell_ratio(cost, solved, a, b, resamples)
            rows.append(
                {
                    "scope": scope,
                    "unit": unit,
                    "n": len(sub),
                    "thinking": thinking,
                    "plain": plain,
                    "acc_thinking": float(solved[:, a].mean()),
                    "acc_plain": float(solved[:, b].mean()),
                    "mean_thinking": float(cost[:, a].mean()),
                    "mean_plain": float(cost[:, b].mean()),
                    "plain_suffices": float(sub.success[:, b].mean()),
                    "needs_thinking": float((sub.success[:, a] & ~sub.success[:, b]).mean()),
                    "thinking_over_plain": point,
                    "thinking_over_plain_lo": lo,
                    "thinking_over_plain_hi": hi,
                }
            )
    return pd.DataFrame(rows)
