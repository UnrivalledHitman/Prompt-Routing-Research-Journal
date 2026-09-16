"""Oracle labels, per-cell costs and data splits (experimental protocol, sections 3, 4, 5.3).

The unit of analysis is a prompt with all eight cells answered. For each cell the
label table holds success and four costs:

* ``tokens``: ``price_in * n_in + price_out * n_out``. ``n_out`` includes the reasoning
  trace.
* ``latency_s``: wall-clock time of the request.
* ``energy_j``: dynamic GPU energy, above the loaded-idle baseline.
* ``board_j``: total GPU board energy.

``c*(q)`` is the cheapest successful cell in a chosen unit, or -1 when no cell
succeeds (the unservable fraction).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("arbiter.labels")
CELLS: tuple[tuple[str, str], ...] = (
    ("0.6b", "no_think"),
    ("0.6b", "think"),
    ("1.7b", "no_think"),
    ("1.7b", "think"),
    ("4b", "no_think"),
    ("4b", "think"),
    ("8b", "no_think"),
    ("8b", "think"),
)
CELL_NAMES = tuple(f"c{i + 1}" for i in range(len(CELLS)))
COST_UNITS = ("tokens", "latency_s", "energy_j", "board_j")
KEY = ["prompt_id", "tag", "mode", "sample", "variant"]
SPLIT_PATTERN = ("train", "train", "train", "val", "test")


@dataclass(frozen=True)
class LabelTable:
    """Per-prompt success and cost matrices, with cells in grid order.

    Attributes
    ----------
    prompts : DataFrame
        One row per prompt with ``prompt_id``, ``family``, ``source`` and ``subset``.
    success : ndarray of bool, shape (n, 8)
    costs : dict of str to ndarray of float, shape (n, 8)
        One matrix per unit in :data:`COST_UNITS`.
    capped : ndarray of bool, shape (n, 8)
        True where the generation hit its token cap.
    contaminated : ndarray of bool, shape (n, 8)
        True where the energy window was accepted despite another workload.
    """

    prompts: pd.DataFrame
    success: np.ndarray
    costs: dict[str, np.ndarray]
    capped: np.ndarray
    contaminated: np.ndarray

    def __len__(self) -> int:
        """Return the number of prompts."""
        return len(self.prompts)

    def take(self, mask: np.ndarray) -> LabelTable:
        """Return the rows selected by a boolean mask or an index array."""
        return LabelTable(
            prompts=self.prompts.iloc[
                np.flatnonzero(mask) if mask.dtype == bool else mask
            ].reset_index(drop=True),
            success=self.success[mask],
            costs={unit: matrix[mask] for unit, matrix in self.costs.items()},
            capped=self.capped[mask],
            contaminated=self.contaminated[mask],
        )


def cell_index(size: str, mode: str) -> int:
    """Return the grid index of a (size, mode) pair.

    Raises
    ------
    KeyError
        If the pair is not in the grid.
    """
    try:
        return CELLS.index((size, mode))
    except ValueError as exc:
        raise KeyError(f"({size}, {mode}) is not a grid cell") from exc


def build_label_table(
    grades: pd.DataFrame,
    energy: pd.DataFrame | None,
    sample: int = 0,
    variant: str = "base",
    price_in: float = 1.0,
    price_out: float = 1.0,
) -> LabelTable:
    """Assemble the label table from graded generations and attributed energy.

    Prompts missing any of the eight cells are dropped with a warning, so a partly
    finished stage can be analysed.

    Raises
    ------
    ValueError
        If a (prompt, cell) pair appears more than once.
    """
    frame = grades[(grades["sample"] == sample) & (grades["variant"] == variant)].copy()
    if energy is not None:
        extra = [c for c in ("energy_dyn_token_j",) if c in energy.columns]
        measured = ["wall_s"] if "wall_s" in energy.columns else []
        joules = energy[[*KEY, "energy_dyn_j", "energy_board_j", *extra, *measured]]
        frame = frame.merge(
            joules.rename(columns={"wall_s": "energy_wall_s"}),
            on=KEY,
            how="left",
            validate="one_to_one",
        )
        if measured:
            # Latency comes from the generation whose energy was measured. The two differ
            # only where a generation was rerun for its energy (protocol section 13.7).
            frame["wall_s"] = frame["energy_wall_s"].fillna(frame["wall_s"])
    else:
        frame["energy_dyn_j"] = np.nan
        frame["energy_board_j"] = np.nan
    frame["cell"] = [cell_index(s, m) for s, m in zip(frame["size"], frame["mode"], strict=True)]
    if frame.duplicated(["prompt_id", "cell"]).any():
        raise ValueError("duplicate (prompt, cell) generations")

    counts = frame.groupby("prompt_id")["cell"].nunique()
    complete = counts.index[counts == len(CELLS)]
    if len(complete) < len(counts):
        LOGGER.warning("dropping %d prompts without all cells", len(counts) - len(complete))
    frame = frame[frame["prompt_id"].isin(complete)]

    prompts = (
        frame.drop_duplicates("prompt_id")[["prompt_id", "family", "source", "subset"]]
        .sort_values("prompt_id")
        .reset_index(drop=True)
    )
    wide = frame.set_index(["prompt_id", "cell"]).sort_index()

    def matrix(column: str | pd.Series, dtype: type = float) -> np.ndarray:
        values = wide[column] if isinstance(column, str) else column
        table = values.unstack("cell").reindex(index=prompts["prompt_id"], columns=range(8))
        return table.to_numpy(dtype=dtype)

    tokens = price_in * wide["prompt_eval_count"] + price_out * wide["eval_count"]
    contaminated = wide["contaminated"].fillna(False).astype(bool)
    costs = {
        "tokens": matrix(tokens),
        "latency_s": matrix("wall_s"),
        "energy_j": matrix("energy_dyn_j"),
        "board_j": matrix("energy_board_j"),
    }
    if "energy_dyn_token_j" in wide.columns:
        # Dynamic energy split by generated tokens: a check on wall-time apportioning.
        costs["energy_token_j"] = matrix("energy_dyn_token_j")
    return LabelTable(
        prompts=prompts,
        success=matrix("correct", bool),
        costs=costs,
        capped=matrix(wide["done_reason"] == "length", bool),
        contaminated=matrix(contaminated, bool),
    )


def minimum_sufficient(success: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Return ``c*(q)`` per prompt, -1 where no cell succeeds; ties go to the lower cell."""
    masked = np.where(success, cost, np.inf)
    choice = np.argmin(masked, axis=1)
    choice[~success.any(axis=1)] = -1
    return choice


def cstar_cost(success: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Return the cost of ``c*(q)`` per prompt, NaN where unservable."""
    choice = minimum_sufficient(success, cost)
    rows = np.arange(len(choice))
    return np.where(choice >= 0, cost[rows, np.maximum(choice, 0)], np.nan)


def _unit_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def assign_splits(
    prompt_ids: Sequence[str],
    strata: Sequence[str],
    seed: int,
    pattern: Sequence[str] = SPLIT_PATTERN,
) -> np.ndarray:
    """Assign train/val/test deterministically, in proportion to ``pattern`` within each stratum.

    Prompts in a stratum are ordered by a seeded hash of their id and dealt out along
    the repeating pattern, starting at a seeded offset so that small strata do not
    always land in training.
    """
    splits = np.empty(len(prompt_ids), dtype=object)
    by_stratum: dict[str, list[int]] = {}
    for index, stratum in enumerate(strata):
        by_stratum.setdefault(stratum, []).append(index)
    for stratum, indices in by_stratum.items():
        indices.sort(key=lambda i: _unit_hash(f"{seed}|{prompt_ids[i]}"))
        offset = _unit_hash(f"{seed}|stratum|{stratum}") % len(pattern)
        for position, index in enumerate(indices):
            splits[index] = pattern[(position + offset) % len(pattern)]
    return splits
