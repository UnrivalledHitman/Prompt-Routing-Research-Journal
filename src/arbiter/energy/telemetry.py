"""Continuous NVML telemetry and window energy (experimental protocol, sections 6 and 6.2).

A recorder thread samples board power, utilisation, temperature and memory use at a
fixed interval for the whole of a model session and streams every sample to CSV.
The labelling harness marks windows by ``time.perf_counter`` and energy is
integrated afterwards, so the attribution rule can be revised without rerunning any
generation.

Gate 1 established that instantaneous power is trustworthy on this GPU but refreshes
only every 0.5 s and is smoothed, and that the cumulative energy counter is not
physical. Energy here therefore always comes from sampled power, integrated with
the trapezoidal rule

    E = sum_i 0.5 * (P_i + P_{i+1}) * (t_{i+1} - t_i).
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import pynvml

LOGGER = logging.getLogger("arbiter.telemetry")
CSV_COLUMNS = ("perf_s", "epoch_s", "power_mw", "util_pct", "temp_c", "mem_used_mb")
MAX_VALID_W = 200.0  # gate 1 measured a 115 W enforced limit; above this is a read error
T = TypeVar("T")


@dataclass(frozen=True)
class TelemetrySample:
    """One NVML reading; a field is None when its read failed."""

    perf_s: float
    epoch_s: float
    power_mw: int | None
    util_pct: int | None
    temp_c: int | None
    mem_used_mb: float | None


def _try(fn: Callable[..., T], *args: object) -> T | None:
    try:
        return fn(*args)
    except pynvml.NVMLError:
        return None


def _cell(value: float | int | None, fmt: str) -> str:
    return "" if value is None else format(value, fmt)


class TelemetryRecorder(threading.Thread):
    """Sample NVML at a fixed interval, streaming to CSV and keeping recent samples.

    Parameters
    ----------
    path : Path
        CSV destination; parent directories are created.
    interval_s : float
        Sampling interval.
    keep_s : float
        History available to :meth:`window`.
    device_index : int
        NVML device index.
    """

    def __init__(
        self, path: Path, interval_s: float, keep_s: float = 900.0, device_index: int = 0
    ) -> None:
        super().__init__(daemon=True, name="nvml-telemetry")
        self.path = path
        self._interval_s = interval_s
        self._device_index = device_index
        self._recent: deque[TelemetrySample] = deque(maxlen=int(keep_s / interval_s) + 1)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._first_sample = threading.Event()
        self.error: BaseException | None = None

    def start(self) -> None:
        """Start sampling and wait for the first sample or a failure."""
        super().start()
        self._first_sample.wait(timeout=10.0)
        if self.error is not None:
            raise RuntimeError("telemetry recorder failed to start") from self.error

    def run(self) -> None:
        """Sample until stopped; a failure is kept in :attr:`error`."""
        initialised = False
        try:
            pynvml.nvmlInit()
            initialised = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(CSV_COLUMNS)
                next_t = time.perf_counter()
                last_flush = next_t
                while not self._stop_event.is_set():
                    sample = self._read(handle)
                    with self._lock:
                        self._recent.append(sample)
                    writer.writerow(
                        [
                            _cell(sample.perf_s, ".4f"),
                            _cell(sample.epoch_s, ".3f"),
                            _cell(sample.power_mw, "d"),
                            _cell(sample.util_pct, "d"),
                            _cell(sample.temp_c, "d"),
                            _cell(sample.mem_used_mb, ".0f"),
                        ]
                    )
                    self._first_sample.set()
                    if sample.perf_s - last_flush >= 1.0:
                        fh.flush()
                        last_flush = sample.perf_s
                    next_t += self._interval_s
                    delay = next_t - time.perf_counter()
                    if delay < -10 * self._interval_s:
                        next_t = time.perf_counter()  # resynchronise after a stall
                    elif delay > 0:
                        time.sleep(delay)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the harness via .error
            self.error = exc
            LOGGER.exception("telemetry recorder failed")
        finally:
            self._first_sample.set()
            if initialised:
                pynvml.nvmlShutdown()

    @staticmethod
    def _read(handle: object) -> TelemetrySample:
        perf, epoch = time.perf_counter(), time.time()
        util = _try(pynvml.nvmlDeviceGetUtilizationRates, handle)
        memory = _try(pynvml.nvmlDeviceGetMemoryInfo, handle)
        return TelemetrySample(
            perf_s=perf,
            epoch_s=epoch,
            power_mw=_try(pynvml.nvmlDeviceGetPowerUsage, handle),
            util_pct=None if util is None else int(util.gpu),
            temp_c=_try(pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU),
            mem_used_mb=None if memory is None else memory.used / 2**20,
        )

    def window(self, start: float, end: float) -> list[TelemetrySample]:
        """Return recent samples with ``start <= perf_s <= end``."""
        with self._lock:
            return [s for s in self._recent if start <= s.perf_s <= end]

    def stop(self) -> None:
        """Stop sampling and close the CSV file."""
        self._stop_event.set()
        self.join(timeout=10.0)


# ---------------------------------------------------------------------------
# Offline energy
# ---------------------------------------------------------------------------


def load_telemetry(path: Path, max_valid_w: float = MAX_VALID_W) -> tuple[np.ndarray, np.ndarray]:
    """Read ``perf_s`` and power in watts, dropping failed and physically impossible reads.

    The labelling smoke run recorded one 588 W sample, between 1.2 W neighbours, as
    telemetry started. The GPU's highest reading in gate 1 was 88 W, so anything above
    ``max_valid_w`` is treated as a read error and interpolated over.
    """
    times: list[float] = []
    watts: list[float] = []
    dropped = 0
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row["power_mw"]:
                continue
            power_w = int(row["power_mw"]) / 1000.0
            if power_w > max_valid_w:
                dropped += 1
                continue
            times.append(float(row["perf_s"]))
            watts.append(power_w)
    if dropped:
        LOGGER.warning("%s: dropped %d power reads above %.0f W", path.name, dropped, max_valid_w)
    return np.asarray(times, dtype=np.float64), np.asarray(watts, dtype=np.float64)


def integrate_energy(
    t: np.ndarray, power_w: np.ndarray, start: float, end: float, max_gap_s: float = 1.0
) -> float:
    """Integrate power over ``[start, end]``, interpolating the power at both edges.

    Returns
    -------
    float
        Energy in joules.

    Raises
    ------
    ValueError
        If the window is empty, the samples do not cover it, or two consecutive
        samples inside it are further apart than ``max_gap_s``.
    """
    if end <= start:
        raise ValueError(f"empty window [{start}, {end}]")
    if t.size < 2 or t[0] > start or t[-1] < end:
        raise ValueError(f"samples do not cover window [{start:.3f}, {end:.3f}]")
    inside = (t > start) & (t < end)
    ts = np.concatenate(([start], t[inside], [end]))
    ps = np.concatenate(
        ([np.interp(start, t, power_w)], power_w[inside], [np.interp(end, t, power_w)])
    )
    gap = float(np.max(np.diff(ts)))
    if gap > max_gap_s:
        raise ValueError(f"{gap:.2f} s gap between samples inside window [{start:.3f}, {end:.3f}]")
    return float(np.trapezoid(ps, ts))


def apportion(total_j: float, weights: Sequence[float]) -> list[float]:
    """Split ``total_j`` in proportion to non-negative ``weights``.

    Raises
    ------
    ValueError
        If a weight is negative or all weights are zero.
    """
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("weights sum to zero")
    return [total_j * w / total_weight for w in weights]
