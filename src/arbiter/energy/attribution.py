"""Energy attribution for labelling windows (experimental protocol, sections 6.2 and 13.7).

For an accepted batch window ``w`` in a session whose idle power is ``P_base``:

    E_w     = integral of P(t) over [start_w, end_w]      (board energy)
    E_dyn_w = E_w - P_base * (end_w - start_w)            (energy above idle)

Each generation ``g`` in ``w`` receives the share ``wall_g / sum(wall)`` of both. The
tail after the last generation lies inside the window, so the decay of the smoothed
power reading is charged to that window's generations rather than lost.

``P_base`` is the energy of the session's accepted baseline windows divided by their
total duration, measured with the model loaded and idle. The dynamic figure is the
primary cost measure; board energy is reported alongside it.

A window's energy is left unmeasured (NaN, with the reason in ``energy_excluded``)
when it cannot be compared with the rest of the grid:

* ``sampling_gap``: telemetry has a gap longer than ``max_gap_s`` inside the window.
  A system suspend is the usual cause, and it also stretches the wall time of the
  generation it interrupts, so wall-time shares would be wrong as well.
* ``on_battery``: the window overlaps a period without mains power, during which the
  GPU runs under a lower power limit. The harness records this for new windows;
  for earlier ones the periods come from the operating system's event log.

The analysis drops prompts with an unmeasured cell, so an excluded window removes
its prompts from the analysis instead of pricing them wrongly.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from arbiter.energy.telemetry import (
    apportion,
    integrate_energy,
    load_clock,
    load_telemetry,
    max_gap_in_window,
)
from arbiter.utils.jsonl import read_jsonl

LOGGER = logging.getLogger("arbiter.attribution")
MAX_GAP_S = 1.0  # ten missed samples at the 100 ms interval


def baseline_power_w(
    t: np.ndarray, power_w: np.ndarray, windows: Sequence[dict[str, Any]]
) -> float:
    """Return mean idle power over a session's accepted baseline windows.

    Raises
    ------
    ValueError
        If there is no accepted baseline window.
    """
    chosen = [w for w in windows if w.get("accepted") and str(w["kind"]).startswith("baseline")]
    if not chosen:
        raise ValueError("no accepted baseline window")
    energy = duration = 0.0
    for window in chosen:
        # A closing baseline may end a fraction of a second after the last sample if
        # the recorder stopped promptly; an idle mean is unaffected by trimming it.
        start = window["start_perf"]
        end = min(window["end_perf"], float(t[-1])) if t.size else window["end_perf"]
        energy += integrate_energy(t, power_w, start, end)
        duration += end - start
    return energy / duration


def window_exclusion(
    t: np.ndarray,
    window: dict[str, Any],
    clock: tuple[np.ndarray, np.ndarray] | None = None,
    on_battery: Sequence[tuple[float, float]] = (),
    max_gap_s: float = MAX_GAP_S,
) -> str | None:
    """Return why a window's energy cannot be used, or None if it can.

    Parameters
    ----------
    t : ndarray
        Times of the session's valid power samples (``perf_s``).
    window : dict
        The window record; ``on_battery`` and ``stall_s`` are used when present.
    clock : tuple of ndarray, optional
        ``perf_s`` and ``epoch_s`` of every telemetry row, to place the window in
        wall-clock time. Required when ``on_battery`` is given.
    on_battery : sequence of (float, float)
        Unix-time intervals without mains power.
    max_gap_s : float
        Longest tolerated interval without a valid power sample.
    """
    start, end = window["start_perf"], window["end_perf"]
    if window.get("on_battery"):
        return "on_battery"
    if (window.get("stall_s") or 0.0) > max_gap_s or max_gap_in_window(t, start, end) > max_gap_s:
        return "sampling_gap"
    if on_battery:
        if clock is None:
            raise ValueError("on_battery intervals need the session clock")
        wall_start, wall_end = np.interp([start, end], *clock)
        if any(wall_start <= until and wall_end >= since for since, until in on_battery):
            return "on_battery"
    return None


def attribute_window(
    t: np.ndarray,
    power_w: np.ndarray,
    window: dict[str, Any],
    records: Sequence[dict[str, Any]],
    baseline_w: float,
    exclusion: str | None = None,
) -> list[dict[str, Any]]:
    """Split one window's board and dynamic energy across its generations by wall time.

    With an ``exclusion`` reason, every energy figure is NaN and the reason is recorded.
    """
    duration = window["end_perf"] - window["start_perf"]
    weights = [r["wall_s"] for r in records]
    if exclusion is None:
        board = integrate_energy(t, power_w, window["start_perf"], window["end_perf"])
        dynamic = board - baseline_w * duration
        tokens = [r.get("eval_count") or 0 for r in records]
        token_weights = tokens if sum(tokens) > 0 else weights
        shares = zip(
            apportion(board, weights),
            apportion(dynamic, weights),
            apportion(dynamic, token_weights),
            strict=True,
        )
    else:
        board = dynamic = float("nan")
        shares = iter([(board, board, board)] * len(records))
    rows = []
    for record, (e_board, e_dyn, e_dyn_tokens) in zip(records, shares, strict=True):
        rows.append(
            {
                "prompt_id": record["prompt_id"],
                "tag": record["tag"],
                "size": record["size"],
                "mode": record["mode"],
                "sample": record["sample"],
                "variant": record["variant"],
                "window_id": window["window_id"],
                "session_id": window["session_id"],
                "wall_s": record["wall_s"],
                "energy_board_j": e_board,
                "energy_dyn_j": e_dyn,
                "energy_dyn_token_j": e_dyn_tokens,
                "baseline_w": baseline_w,
                "window_s": duration,
                "window_board_j": board,
                "window_dyn_j": dynamic,
                "contaminated": bool(window.get("contaminated", False)),
                "energy_excluded": exclusion or "",
            }
        )
    return rows


def attribute_stage(
    raw_dir: Path,
    on_battery: Sequence[tuple[float, float]] = (),
    max_gap_s: float = MAX_GAP_S,
) -> list[dict[str, Any]]:
    """Attribute energy to every generation recorded in a labelling stage directory.

    Parameters
    ----------
    raw_dir : Path
        Stage directory with ``generations.jsonl``, ``windows.jsonl`` and telemetry.
    on_battery : sequence of (float, float)
        Unix-time intervals without mains power; overlapping windows are excluded.
    max_gap_s : float
        Longest tolerated interval without a power sample inside a batch window.

    Raises
    ------
    ValueError
        If a generation's window is missing, or a window's generations do not match
        its recorded count.
    """
    generations, _ = read_jsonl(raw_dir / "generations.jsonl")
    windows, _ = read_jsonl(raw_dir / "windows.jsonl")
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in generations:
        by_window[record["window_id"]].append(record)
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        by_session[window["session_id"]].append(window)

    rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for session_id, session_windows in by_session.items():
        batches = [w for w in session_windows if w["kind"] == "batch" and w.get("accepted")]
        if not batches:
            continue
        telemetry_path = raw_dir / session_windows[0]["telemetry_file"]
        t, power_w = load_telemetry(telemetry_path)
        clock = load_clock(telemetry_path) if on_battery else None
        baseline_w = baseline_power_w(t, power_w, session_windows)
        for window in batches:
            records = by_window.pop(window["window_id"], [])
            if len(records) != window["n_generations"]:
                raise ValueError(
                    f"{window['window_id']}: {len(records)} generations recorded, "
                    f"window says {window['n_generations']}"
                )
            reason = window_exclusion(t, window, clock, on_battery, max_gap_s)
            if reason is not None:
                excluded[reason] += 1
                LOGGER.warning(
                    "%s: energy excluded (%s), %d generations", window["window_id"], reason,
                    len(records),
                )  # fmt: skip
            rows.extend(attribute_window(t, power_w, window, records, baseline_w, reason))
        LOGGER.info("%s: %d windows, baseline %.2f W", session_id, len(batches), baseline_w)
    if by_window:
        raise ValueError(f"generations without an accepted window: {sorted(by_window)[:5]}")
    if excluded:
        LOGGER.warning("windows with energy excluded: %s", dict(excluded))
    return rows
