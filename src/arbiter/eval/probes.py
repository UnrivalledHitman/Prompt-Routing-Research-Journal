"""Probe items and results for R3 and the cascade (paper sections III-D and III-F).

Probes are short questions put to a pool model and run by a probe stage of the
labelling harness, so their latency and energy are measured like the pool's:

* verification probes, one per no-think answer on the P1 ladder below c8, asked of the
  model that wrote the answer, with ids ``<prompt_id>@<size>``;
* routing probes, one per prompt, asked of Qwen3-0.6B zero-shot or few-shot.

This module builds the probe items, parses recorded generations into text-free rows,
and aligns those rows with a label table.
"""

from __future__ import annotations

import logging
import random
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from arbiter.data.corpus import PromptItem
from arbiter.eval.cascade import Verification
from arbiter.eval.labels import CELL_NAMES, CELLS, LabelTable, cell_index
from arbiter.eval.policies import P1
from arbiter.routers.llm import (
    EXAMPLE_CHARS,
    LETTERS,
    answer_probabilities,
    label_distribution,
    router_prompt,
    verify_prompt,
    yes_confidence,
)

LOGGER = logging.getLogger("arbiter.probes")
SEPARATOR = "@"
VERIFIED_CELLS = tuple(CELLS[cell] for cell in P1[:-1])
ROUTER_CELL = CELLS[0]
LABEL_COLUMNS = tuple(f"p_{letter}" for letter in LETTERS)
RECORD_COLUMNS = (
    "family",
    "size",
    "tag",
    "mode",
    "sample",
    "variant",
    "prompt_eval_count",
    "eval_count",
    "wall_s",
    "contaminated",
)
Cell = tuple[str, str, str]


def verify_items(
    items: Sequence[PromptItem], generations: Iterable[Mapping[str, Any]]
) -> tuple[list[PromptItem], list[Cell]]:
    """Build one verification probe per base answer in a verified cell.

    Returns the probe items and the ``(probe_id, size, mode)`` cells a probe stage runs.
    """
    by_id = {item.prompt_id: item for item in items}
    probes: list[PromptItem] = []
    cells: list[Cell] = []
    seen: set[str] = set()
    for record in generations:
        size, mode = record["size"], record["mode"]
        if (size, mode) not in VERIFIED_CELLS or record["prompt_id"] not in by_id:
            continue
        if int(record["sample"]) != 0 or record["variant"] != "base":
            continue
        item = by_id[record["prompt_id"]]
        probe_id = f"{item.prompt_id}{SEPARATOR}{size}"
        if probe_id in seen:
            continue
        seen.add(probe_id)
        text = verify_prompt(item.text, record.get("content") or "")
        meta = {"probe": "verify", "prompt_id": item.prompt_id, "size": size}
        probes.append(
            PromptItem(probe_id, item.family, item.source, text, None, item.subset, meta=meta)
        )
        cells.append((probe_id, size, mode))
    return probes, cells


def pick_examples(
    splits: pd.DataFrame,
    texts: Mapping[str, str],
    label_column: str,
    per_cell: int = 1,
    seed: int = 0,
    max_chars: int = EXAMPLE_CHARS,
) -> list[tuple[str, int]]:
    """Pick few-shot examples from the training split, ``per_cell`` for each cell.

    A cell's examples are drawn with a seeded generator from the training prompts whose
    cheapest sufficient cell it is, preferring prompts short enough to need no cut.
    Cells that are no prompt's cheapest sufficient cell get no example.
    """
    rng = random.Random(seed)
    train = splits[splits["split"] == "train"]
    examples = []
    for index, name in enumerate(CELL_NAMES):
        ids = sorted(train.loc[train[label_column] == name, "prompt_id"])
        short = [pid for pid in ids if len(texts[pid]) <= max_chars]
        pool = short or sorted(ids, key=lambda pid: len(texts[pid]))[:per_cell]
        examples += [(texts[pid], index) for pid in rng.sample(pool, min(per_cell, len(pool)))]
    return examples


def route_items(
    items: Sequence[PromptItem], examples: Sequence[tuple[str, int]] = ()
) -> tuple[list[PromptItem], list[Cell]]:
    """Build one routing probe per prompt, for Qwen3-0.6B without thinking."""
    size, mode = ROUTER_CELL
    kind = "route-few" if examples else "route-zero"
    probes = [
        PromptItem(
            item.prompt_id,
            item.family,
            item.source,
            router_prompt(item.text, examples),
            None,
            item.subset,
            meta={"probe": kind},
        )
        for item in items
    ]
    return probes, [(item.prompt_id, size, mode) for item in items]


def probe_row(record: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """Parse one recorded probe generation into a text-free row.

    Raises
    ------
    ValueError
        If ``kind`` is neither ``verify`` nor ``route``.
    """
    if kind not in ("verify", "route"):
        raise ValueError(f"unknown probe kind {kind}")
    probabilities = answer_probabilities(record.get("logprobs"))
    probe_id = str(record["prompt_id"])
    row: dict[str, Any] = {
        "probe_id": probe_id,
        "prompt_id": probe_id.rsplit(SEPARATOR, 1)[0] if kind == "verify" else probe_id,
        **{column: record.get(column) for column in RECORD_COLUMNS},
        "candidate_mass": float(sum(probabilities.values())),
    }
    if kind == "verify":
        row["confidence"] = yes_confidence(probabilities)
        row["parsed"] = bool(np.isfinite(row["confidence"]))
    else:
        labels = label_distribution(probabilities)
        row.update({column: float(p) for column, p in zip(LABEL_COLUMNS, labels, strict=True)})
        row["parsed"] = bool(np.isfinite(labels).all())
    return row


def unit_costs(frame: pd.DataFrame, prices: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Add one cost column per unit, named as in the label table.

    The primary price sheet gives ``tokens`` and every other sheet ``tokens_<label>``.
    Energy columns are NaN when no energy was attributed.
    """
    out = frame.copy()
    for label, sheet in prices.items():
        name = "tokens" if label == "primary" else f"tokens_{label}"
        out[name] = (
            sheet["price_in"] * out["prompt_eval_count"] + sheet["price_out"] * out["eval_count"]
        )
    out["latency_s"] = out["wall_s"]
    out["energy_j"] = out["energy_dyn_j"] if "energy_dyn_j" in out.columns else np.nan
    out["board_j"] = out["energy_board_j"] if "energy_board_j" in out.columns else np.nan
    return out


def per_prompt(frame: pd.DataFrame, table: LabelTable, column: str) -> np.ndarray:
    """Return one probe value per prompt of ``table``, NaN where it has none."""
    series = frame.drop_duplicates("prompt_id").set_index("prompt_id")[column]
    return series.reindex(table.prompts["prompt_id"]).to_numpy(dtype=float)


def per_cell(
    frame: pd.DataFrame, table: LabelTable, column: str, mode: str = "no_think"
) -> np.ndarray:
    """Return an (n, 8) matrix of probe values at ``(size, mode)``, NaN elsewhere."""
    out = np.full((len(table), len(CELLS)), np.nan)
    position = {pid: i for i, pid in enumerate(table.prompts["prompt_id"])}
    sub = frame[frame["prompt_id"].isin(position)]
    rows = sub["prompt_id"].map(position).to_numpy(dtype=int)
    cols = np.array([cell_index(size, mode) for size in sub["size"]], dtype=int)
    out[rows, cols] = sub[column].to_numpy(dtype=float)
    return out


def label_matrix(frame: pd.DataFrame, table: LabelTable) -> np.ndarray:
    """Return R3's (n, 8) label distributions, NaN rows where a prompt has none."""
    return np.column_stack([per_prompt(frame, table, column) for column in LABEL_COLUMNS])


def fill_missing(values: np.ndarray, what: str) -> np.ndarray:
    """Replace missing costs with the mean of the measured ones in the same column.

    Columns without any measurement stay NaN.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(values, axis=0)
    filled = np.where(np.isnan(values), means, values)
    count = int(np.sum(np.isnan(values) & ~np.isnan(filled)))
    if count:
        LOGGER.warning("%s: %d missing probe costs filled with the measured mean", what, count)
    return filled


def prompt_costs(frame: pd.DataFrame, table: LabelTable, unit: str) -> np.ndarray:
    """Return each prompt's measured probe cost in ``unit``, gaps filled with the mean."""
    return fill_missing(per_prompt(frame, table, unit), f"probe {unit}")


def verification(frame: pd.DataFrame, table: LabelTable, units: Sequence[str]) -> Verification:
    """Align verification probes with a split: confidences and per-unit call costs.

    An answer without a parsed verification keeps a NaN confidence, so the cascade never
    stops there, and its verification is charged the mean measured cost.
    """
    confidence = per_cell(frame, table, "confidence")
    verified = confidence[:, list(P1[:-1])]
    missing = int(np.isnan(verified).sum())
    if missing:
        LOGGER.warning("%d of %d verified answers have no confidence", missing, verified.size)
    costs = {
        unit: fill_missing(per_cell(frame, table, unit), f"verification {unit}") for unit in units
    }
    return Verification(confidence, costs)
