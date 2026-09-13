"""Robustness of the oracle labels (experimental protocol, sections 5.3, 6.2 and 13.1).

* :func:`label_noise`: how often repeated samples of a cell disagree on the noise
  subset, and how often ``c*(q)`` changes with the sample it is computed from;
* :func:`cap_sensitivity`: how often a capped failure succeeds under a longer cap, and
  whether that success would move ``c*(q)`` to a cheaper cell;
* :func:`apportioning_check`: whether energy labels depend on splitting window energy
  by wall time or by generated tokens;
* :func:`energy_drift`: whether energy per generated token moved between the first
  sample and later samples of the same prompts and cells, which were measured on
  either side of a driver update.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from arbiter.eval.labels import (
    CELL_NAMES,
    CELLS,
    KEY,
    LabelTable,
    build_label_table,
    cell_index,
    minimum_sufficient,
)


def _scopes(prompts: pd.DataFrame) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``("all", mask)`` and then one prompt mask per family."""
    yield "all", np.ones(len(prompts), dtype=bool)
    for family in sorted(prompts["family"].unique()):
        yield str(family), (prompts["family"] == family).to_numpy()


def aligned_samples(
    grades: pd.DataFrame,
    samples: Sequence[int],
    unit: str = "tokens",
    energy: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return the prompts answered in every cell by every sample, with stacked labels.

    Returns
    -------
    prompts : DataFrame
    success : ndarray of bool, shape (k, n, 8)
    cost : ndarray of float, shape (k, n, 8)

    Raises
    ------
    ValueError
        If no prompt has every sample in every cell.
    """
    tables = [build_label_table(grades, energy, sample=s) for s in samples]
    common = set.intersection(*(set(t.prompts["prompt_id"]) for t in tables))
    if not common:
        raise ValueError(f"no prompt has samples {list(samples)} in every cell")
    kept = [t.take(t.prompts["prompt_id"].isin(common).to_numpy()) for t in tables]
    prompts = kept[0].prompts
    success = np.stack([t.success for t in kept])
    cost = np.stack([t.costs[unit] for t in kept])
    return prompts, success, cost


def label_noise(
    grades: pd.DataFrame,
    samples: Sequence[int] = (0, 1, 2),
    unit: str = "tokens",
    energy: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure disagreement between repeated samples on the noise subset.

    The cell table gives, per scope and cell, the success rate over all samples, the
    share of prompts whose samples do not all agree, the mean disagreement over sample
    pairs, and how often the majority vote differs from the first sample, which is the
    label used everywhere else. The ``c*`` table gives how often ``c*(q)`` agrees
    between sample pairs, between the first sample and the majority vote (costed at
    the mean over samples), and how often servability itself varies.
    """
    prompts, success, cost = aligned_samples(grades, samples, unit, energy)
    k = len(samples)
    pairs = list(combinations(range(k), 2))
    disagree = np.mean([success[i] != success[j] for i, j in pairs], axis=0)
    varies = success.any(axis=0) & ~success.all(axis=0)
    majority = success.sum(axis=0) * 2 > k
    cstar = np.stack([minimum_sufficient(success[i], cost[i]) for i in range(k)])
    cstar_majority = minimum_sufficient(majority, cost.mean(axis=0))
    servable = cstar >= 0

    cell_rows, cstar_rows = [], []
    for scope, mask in _scopes(prompts):
        for index, (size, mode) in enumerate(CELLS):
            cell_rows.append(
                {
                    "scope": scope,
                    "cell": CELL_NAMES[index],
                    "size": size,
                    "mode": mode,
                    "n_prompts": int(mask.sum()),
                    "success_rate": float(success[:, mask, index].mean()),
                    "varies_rate": float(varies[mask, index].mean()),
                    "pairwise_disagreement": float(disagree[mask, index].mean()),
                    "majority_differs": float(
                        (majority[mask, index] != success[0, mask, index]).mean()
                    ),
                }
            )
        cstar_rows.append(
            {
                "scope": scope,
                "unit": unit,
                "n_prompts": int(mask.sum()),
                "n_samples": k,
                "cell_pairwise_disagreement": float(disagree[mask].mean()),
                "cstar_pairwise_agreement": float(
                    np.mean([cstar[i, mask] == cstar[j, mask] for i, j in pairs])
                ),
                "cstar_first_vs_majority": float((cstar[0, mask] == cstar_majority[mask]).mean()),
                "servability_varies": float(
                    (servable[:, mask].any(axis=0) & ~servable[:, mask].all(axis=0)).mean()
                ),
            }
        )
    return pd.DataFrame(cell_rows), pd.DataFrame(cstar_rows)


def _cell_cost(
    frame: pd.DataFrame, unit: str, suffix: str, price_in: float, price_out: float
) -> np.ndarray:
    """Return one generation cost per row in ``tokens`` or ``latency_s``.

    Raises
    ------
    ValueError
        If ``unit`` is neither.
    """
    if unit == "tokens":
        tokens = (
            price_in * frame[f"prompt_eval_count{suffix}"]
            + price_out * frame[f"eval_count{suffix}"]
        )
        return tokens.to_numpy(dtype=float)
    if unit == "latency_s":
        return frame[f"wall_s{suffix}"].to_numpy(dtype=float)
    raise ValueError(f"cap sensitivity supports tokens and latency_s, not {unit}")


def cap_sensitivity(
    grades: pd.DataFrame,
    variant: str = "cap8192",
    unit: str = "tokens",
    price_in: float = 1.0,
    price_out: float = 1.0,
) -> pd.DataFrame:
    """Compare capped base generations with their reruns under a longer cap.

    For every rerun of a generation that hit the base cap, ``rescued`` is whether the
    rerun succeeds and ``still_capped`` whether it hit the longer cap too. A rescued
    cell is given its rerun cost, and ``cstar_moved`` records whether ``c*(q)`` then
    changes, which can only be to a cheaper cell or from unservable to servable. Rows
    are per (size, mode), per family, and overall; ``cap_rate`` is the share of base
    generations in that group that hit the cap, so ``cap_rate * rescued_rate``
    estimates the share of all labels in the group that the cap turns into failures.

    Raises
    ------
    ValueError
        If there are no reruns of capped base generations.
    """
    base = grades[(grades["sample"] == 0) & (grades["variant"] == "base")]
    rerun = grades[(grades["sample"] == 0) & (grades["variant"] == variant)]
    keys = ["prompt_id", "size", "mode"]
    merged = rerun.merge(base, on=keys, suffixes=("_rerun", "_base"), validate="one_to_one")
    merged = merged[merged["done_reason_base"] == "length"].reset_index(drop=True)
    table = build_label_table(grades, None, price_in=price_in, price_out=price_out)
    position = {pid: i for i, pid in enumerate(table.prompts["prompt_id"])}
    merged = merged[merged["prompt_id"].isin(position)].reset_index(drop=True)
    if merged.empty:
        raise ValueError(f"no {variant} reruns of capped base generations")

    cost = table.costs[unit]
    before = minimum_sufficient(table.success, cost)
    rows = merged["prompt_id"].map(position).to_numpy()
    cells = [cell_index(s, m) for s, m in zip(merged["size"], merged["mode"], strict=True)]
    rescued = merged["correct_rerun"].astype(bool).to_numpy()
    rerun_cost = _cell_cost(merged, unit, "_rerun", price_in, price_out)
    moved = np.zeros(len(merged), dtype=bool)
    for j in np.flatnonzero(rescued):
        success, costs = table.success[rows[j]].copy(), cost[rows[j]].copy()
        success[cells[j]], costs[cells[j]] = True, rerun_cost[j]
        moved[j] = minimum_sufficient(success[None], costs[None])[0] != before[rows[j]]
    merged = merged.assign(
        rescued=rescued,
        still_capped=(merged["done_reason_rerun"] == "length").to_numpy(),
        cstar_moved=moved,
        newly_servable=rescued & (before[rows] == -1),
        family=merged["family_rerun"],
    )

    def summary(group: pd.DataFrame, base_group: pd.DataFrame, **labels: str) -> dict:
        rescued_tokens = group.loc[group["rescued"], "eval_count_rerun"]
        return {
            **labels,
            "n_base": len(base_group),
            "cap_rate": float((base_group["done_reason"] == "length").mean()),
            "n_rerun": len(group),
            "rescued_rate": float(group["rescued"].mean()),
            "still_capped_rate": float(group["still_capped"].mean()),
            "cstar_moved_rate": float(group["cstar_moved"].mean()),
            "newly_servable_rate": float(group["newly_servable"].mean()),
            "median_output_tokens_rescued": float(rescued_tokens.median())
            if len(rescued_tokens)
            else float("nan"),
        }

    out = [summary(merged, base, family="all", size="all", mode="all")]
    for (size, mode), group in merged.groupby(["size", "mode"], sort=True):
        in_base = base[(base["size"] == size) & (base["mode"] == mode)]
        out.append(summary(group, in_base, family="all", size=size, mode=mode))
    for family, group in merged.groupby("family", sort=True):
        out.append(
            summary(group, base[base["family"] == family], family=family, size="all", mode="all")
        )
    return pd.DataFrame(out)


def apportioning_check(
    table: LabelTable, wall_unit: str = "energy_j", token_unit: str = "energy_token_j"
) -> pd.DataFrame:
    """Compare energy labels under wall-time and token-share apportioning.

    Per scope: the share of prompts whose ``c*(q)`` is the same under both rules, the
    rank correlation of per-generation energy, and per cell the ratio of mean energy
    under token share to mean energy under wall-time share.
    """
    wall, tokens = table.costs[wall_unit], table.costs[token_unit]
    cstar_wall = minimum_sufficient(table.success, wall)
    cstar_tokens = minimum_sufficient(table.success, tokens)
    rows = []
    for scope, mask in _scopes(table.prompts):
        row = {
            "scope": scope,
            "n_prompts": int(mask.sum()),
            "cstar_agreement": float((cstar_wall[mask] == cstar_tokens[mask]).mean()),
            "spearman": float(
                pd.Series(wall[mask].ravel()).corr(
                    pd.Series(tokens[mask].ravel()), method="spearman"
                )
            ),
        }
        for index, name in enumerate(CELL_NAMES):
            row[f"ratio_{name}"] = float(
                np.nanmean(tokens[mask, index]) / np.nanmean(wall[mask, index])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def energy_drift(
    grades: pd.DataFrame,
    energy: pd.DataFrame,
    before: Sequence[int] = (0,),
    after: Sequence[int] = (1, 2),
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> pd.DataFrame:
    """Compare energy per generated token between two groups of samples.

    Only base-variant, uncontaminated generations of (prompt, cell) pairs measured in
    both groups are used, so the two sides cover the same prompts. Per cell, the ratio
    is joules per output token after over joules per output token before, each a
    ratio of sums, with a percentile interval from resampling prompts. Board baseline
    power and output tokens per second are reported beside it, since a driver change
    could move either without moving the other.

    Raises
    ------
    ValueError
        If no (prompt, cell) pair has generations in both groups.
    """
    frame = grades[(grades["variant"] == "base") & ~grades["contaminated"].astype(bool)]
    frame = frame.merge(
        energy[[*KEY, "energy_dyn_j", "baseline_w"]], on=KEY, how="inner", validate="one_to_one"
    )
    frame = frame[frame["eval_count"] > 0].copy()
    frame["group"] = np.where(
        frame["sample"].isin(before), "before", np.where(frame["sample"].isin(after), "after", "")
    )
    frame = frame[frame["group"] != ""]
    sides = frame.groupby(["prompt_id", "size", "mode"])["group"].nunique()
    paired = sides[sides == 2].index
    if paired.empty:
        raise ValueError("no (prompt, cell) pair has generations before and after")
    frame = frame.set_index(["prompt_id", "size", "mode"]).loc[paired].reset_index()

    rng = np.random.default_rng(seed)
    tail = (1.0 - level) / 2.0 * 100.0
    rows = []
    for (size, mode), cell in frame.groupby(["size", "mode"], sort=False):
        sums = (
            cell.groupby(["prompt_id", "group"])[["energy_dyn_j", "eval_count", "wall_s"]]
            .sum()
            .unstack("group")
        )
        joules_b, joules_a = sums[("energy_dyn_j", "before")], sums[("energy_dyn_j", "after")]
        tokens_b, tokens_a = sums[("eval_count", "before")], sums[("eval_count", "after")]
        wall_b, wall_a = sums[("wall_s", "before")], sums[("wall_s", "after")]
        jb, ja = joules_b.to_numpy(), joules_a.to_numpy()
        tb, ta = tokens_b.to_numpy(), tokens_a.to_numpy()
        index = rng.integers(0, len(jb), size=(n_boot, len(jb)))
        dist = (ja[index].sum(1) / ta[index].sum(1)) / (jb[index].sum(1) / tb[index].sum(1))
        lo, hi = np.percentile(dist, [tail, 100.0 - tail])
        by_group = cell.groupby("group")
        rows.append(
            {
                "cell": CELL_NAMES[cell_index(size, mode)],
                "size": size,
                "mode": mode,
                "n_prompts": len(jb),
                "j_per_token_before": float(jb.sum() / tb.sum()),
                "j_per_token_after": float(ja.sum() / ta.sum()),
                "ratio": float((ja.sum() / ta.sum()) / (jb.sum() / tb.sum())),
                "ratio_lo": float(lo),
                "ratio_hi": float(hi),
                "tokens_per_s_before": float(tb.sum() / wall_b.sum()),
                "tokens_per_s_after": float(ta.sum() / wall_a.sum()),
                "baseline_w_before": float(by_group["baseline_w"].mean()["before"]),
                "baseline_w_after": float(by_group["baseline_w"].mean()["after"]),
            }
        )
    return pd.DataFrame(rows).sort_values("cell").reset_index(drop=True)
