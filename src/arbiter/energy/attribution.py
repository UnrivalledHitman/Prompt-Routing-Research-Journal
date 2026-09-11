"""Energy attribution for labelling windows (experimental protocol, section 6.2).

For an accepted batch window ``w`` in a session whose idle power is ``P_base``:

    E_w     = integral of P(t) over [start_w, end_w]      (board energy)
    E_dyn_w = E_w - P_base * (end_w - start_w)            (energy above idle)

Each generation ``g`` in ``w`` receives the share ``wall_g / sum(wall)`` of both. The
tail after the last generation lies inside the window, so the decay of the smoothed
power reading is charged to that window's generations rather than lost.

``P_base`` is the energy of the session's accepted baseline windows divided by their
total duration, measured with the model loaded and idle. The dynamic figure is the
primary cost measure; board energy is reported alongside it.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from arbiter.energy.telemetry import apportion, integrate_energy, load_telemetry
from arbiter.utils.jsonl import read_jsonl

LOGGER = logging.getLogger("arbiter.attribution")


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


def attribute_window(
    t: np.ndarray,
    power_w: np.ndarray,
    window: dict[str, Any],
    records: Sequence[dict[str, Any]],
    baseline_w: float,
) -> list[dict[str, Any]]:
    """Split one window's board and dynamic energy across its generations by wall time."""
    duration = window["end_perf"] - window["start_perf"]
    board = integrate_energy(t, power_w, window["start_perf"], window["end_perf"])
    dynamic = board - baseline_w * duration
    weights = [r["wall_s"] for r in records]
    tokens = [r.get("eval_count") or 0 for r in records]
    token_weights = tokens if sum(tokens) > 0 else weights
    rows = []
    for record, e_board, e_dyn, e_dyn_tokens in zip(
        records,
        apportion(board, weights),
        apportion(dynamic, weights),
        apportion(dynamic, token_weights),
        strict=True,
    ):
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
            }
        )
    return rows


def attribute_stage(raw_dir: Path) -> list[dict[str, Any]]:
    """Attribute energy to every generation recorded in a labelling stage directory.

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
    for session_id, session_windows in by_session.items():
        batches = [w for w in session_windows if w["kind"] == "batch" and w.get("accepted")]
        if not batches:
            continue
        t, power_w = load_telemetry(raw_dir / session_windows[0]["telemetry_file"])
        baseline_w = baseline_power_w(t, power_w, session_windows)
        for window in batches:
            records = by_window.pop(window["window_id"], [])
            if len(records) != window["n_generations"]:
                raise ValueError(
                    f"{window['window_id']}: {len(records)} generations recorded, "
                    f"window says {window['n_generations']}"
                )
            rows.extend(attribute_window(t, power_w, window, records, baseline_w))
        LOGGER.info("%s: %d windows, baseline %.2f W", session_id, len(batches), baseline_w)
    if by_window:
        raise ValueError(f"generations without an accepted window: {sorted(by_window)[:5]}")
    return rows
