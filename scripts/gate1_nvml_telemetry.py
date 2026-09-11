"""Gate 1: verify NVML power and energy telemetry on the local GPU.

The energy contribution of the paper rests on measured joules, so this gate
must pass before any labelling run. It checks four things:

1. Instantaneous board power (``nvmlDeviceGetPowerUsage``) is supported.
2. Power readings refresh fast enough to integrate short generations.
3. Readings respond to load, i.e. they are not a stale or clamped constant.
4. The cumulative energy counter (``nvmlDeviceGetTotalEnergyConsumption``),
   if present, agrees with integrated power samples and is physically
   plausible. Monotonicity alone is not evidence: on the RTX 4060 Laptop the
   counter is monotonic yet implies 88 W at idle and 257 W after load.

NVML reports whole-board power. A concurrent workload would contaminate the
measurement and would also be disturbed by our synthetic load, so the load
phase is skipped when the GPU is already busy.

Energy from sampled power is the trapezoidal integral

    E = sum_i 0.5 * (P_i + P_{i+1}) * (t_{i+1} - t_i),

and the counter estimate is simply E = counter(t_end) - counter(t_start).

Run with ``--reverdict REPORT_JSON`` to recompute the verdict of a saved report
without touching the GPU, for example after the verdict rules change.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pynvml

LOGGER = logging.getLogger("arbiter.gate1")
REPO_ROOT = Path(__file__).resolve().parents[1]
T = TypeVar("T")


@dataclass(frozen=True)
class GateConfig:
    """Parameters for one gate run; every field is written to the output."""

    idle_s: float = 10.0
    load_s: float = 8.0
    cooldown_s: float = 8.0
    settle_s: float = 3.0
    interval_ms: int = 100
    matrix_size: int = 2048
    seed: int = 20260911
    allow_load: bool = True
    busy_util_pct: float = 15.0
    responsive_delta_w: float = 5.0
    agreement_tolerance: float = 0.15
    plausibility_margin: float = 1.05
    power_refresh_note_s: float = 0.25
    out_dir: str = "results/processed"


@dataclass(frozen=True)
class Sample:
    """One telemetry reading.

    Attributes
    ----------
    t : float
        ``time.perf_counter()`` seconds at the read.
    power_mw : int or None
        Board power in milliwatts, None if the read failed.
    energy_mj : int or None
        Cumulative energy counter in millijoules, None if unsupported.
    util_pct : int or None
        GPU utilisation percentage, None if unsupported.
    """

    t: float
    power_mw: int | None
    energy_mj: int | None
    util_pct: int | None


def _read(fn: Callable[..., T], *args: object) -> tuple[T | None, str | None]:
    """Call an NVML getter, returning ``(value, None)`` or ``(None, error)``."""
    try:
        return fn(*args), None
    except pynvml.NVMLError as exc:
        return None, str(exc)


class NvmlSampler(threading.Thread):
    """Poll power, energy counter and utilisation at a fixed interval."""

    def __init__(self, handle: object, interval_s: float) -> None:
        super().__init__(daemon=True)
        self._handle = handle
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self.samples: list[Sample] = []

    def run(self) -> None:
        """Sample until stopped, on a drift-corrected schedule."""
        next_t = time.perf_counter()
        while not self._stop_event.is_set():
            now = time.perf_counter()
            power, _ = _read(pynvml.nvmlDeviceGetPowerUsage, self._handle)
            energy, _ = _read(pynvml.nvmlDeviceGetTotalEnergyConsumption, self._handle)
            util, _ = _read(pynvml.nvmlDeviceGetUtilizationRates, self._handle)
            self.samples.append(Sample(now, power, energy, None if util is None else int(util.gpu)))
            next_t += self._interval_s
            time.sleep(max(0.0, next_t - time.perf_counter()))

    def stop(self) -> list[Sample]:
        """Stop sampling and return the collected samples."""
        self._stop_event.set()
        self.join()
        return self.samples


def sample_window(handle: object, seconds: float, interval_s: float) -> list[Sample]:
    """Sample passively for ``seconds``."""
    sampler = NvmlSampler(handle, interval_s)
    sampler.start()
    time.sleep(seconds)
    return sampler.stop()


def _update_interval(t: np.ndarray, values: np.ndarray) -> float | None:
    """Return the median seconds between successive changes of a reading."""
    change_idx = np.flatnonzero(np.diff(values) != 0) + 1
    if change_idx.size < 2:
        return None
    return float(np.median(np.diff(t[change_idx])))


def summarise(samples: list[Sample], label: str) -> dict[str, Any]:
    """Reduce a sample window to the statistics the verdict needs.

    Raises
    ------
    ValueError
        If the window holds fewer than two samples.
    """
    if len(samples) < 2:
        raise ValueError(f"window '{label}' has {len(samples)} samples; need >= 2")
    t = np.array([s.t for s in samples], dtype=np.float64)
    out: dict[str, Any] = {
        "label": label,
        "n_samples": len(samples),
        "duration_s": float(t[-1] - t[0]),
    }

    if all(s.power_mw is not None for s in samples):
        p_w = np.array([s.power_mw for s in samples], dtype=np.float64) / 1000.0
        out.update(
            power_mean_w=float(p_w.mean()),
            power_std_w=float(p_w.std()),
            power_min_w=float(p_w.min()),
            power_max_w=float(p_w.max()),
            power_n_distinct=int(np.unique(p_w).size),
            power_update_interval_s=_update_interval(t, p_w),
            energy_trapz_j=float(np.trapezoid(p_w, t)),
        )

    if all(s.energy_mj is not None for s in samples):
        e_j = np.array([s.energy_mj for s in samples], dtype=np.float64) / 1000.0
        delta = float(e_j[-1] - e_j[0])
        out.update(
            counter_monotonic=bool(np.all(np.diff(e_j) >= 0)),
            counter_delta_j=delta,
            counter_mean_power_w=delta / out["duration_s"],
            counter_update_interval_s=_update_interval(t, e_j),
        )

    utils = [s.util_pct for s in samples if s.util_pct is not None]
    if utils:
        out["util_mean_pct"] = float(np.mean(utils))
    return out


def list_gpu_processes(handle: object) -> list[dict[str, Any]]:
    """List the processes NVML reports on the device.

    Per-process memory is usually None under Windows WDDM.
    """
    procs: list[dict[str, Any]] = []
    getters = (
        ("compute", pynvml.nvmlDeviceGetComputeRunningProcesses),
        ("graphics", pynvml.nvmlDeviceGetGraphicsRunningProcesses),
    )
    for kind, getter in getters:
        found, _ = _read(getter, handle)
        for proc in found or []:
            used = getattr(proc, "usedGpuMemory", None)
            procs.append(
                {
                    "kind": kind,
                    "pid": int(proc.pid),
                    "used_mb": None if used is None else round(used / 2**20, 1),
                }
            )
    return procs


def synthetic_load(cfg: GateConfig, handle: object) -> tuple[list[Sample], dict[str, Any]]:
    """Run a short fp16 matrix-multiply loop while sampling telemetry.

    CUDA and cuBLAS are initialised and warmed up before the sampled window, so
    that context creation and the first-kernel cost do not appear as load.
    """
    import torch

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    n = cfg.matrix_size
    a = torch.randn(n, n, device=device, dtype=torch.float16)
    b = torch.randn(n, n, device=device, dtype=torch.float16)
    _ = a @ b  # first matmul initialises cuBLAS; keep it out of the window
    torch.cuda.synchronize()
    time.sleep(cfg.settle_s)

    sampler = NvmlSampler(handle, cfg.interval_ms / 1000.0)
    sampler.start()
    iterations = 0
    deadline = time.perf_counter() + cfg.load_s
    while time.perf_counter() < deadline:
        _ = a @ b
        iterations += 1
        if iterations % 16 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    samples = sampler.stop()

    info = {
        "iterations": iterations,
        "matrix_size": n,
        "dtype": "float16",
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    del a, b
    torch.cuda.empty_cache()
    return samples, info


def verdict(
    cfg: GateConfig,
    caps: dict[str, Any],
    windows: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    """Decide which measurement method, if any, the study may rely on.

    A source is trusted only if it is supported, responds to load, and is
    physically plausible: its implied power never exceeds the enforced power
    limit. The energy counter must also agree with integrated power samples,
    because a monotonic counter that responds to load can still be measuring
    something other than board energy.
    """
    idle, load = windows["idle"], windows.get("load")
    measured = [w for w in windows.values() if w is not None]
    limit = caps["enforced_power_limit"]
    limit_w = limit["value"] / 1000.0 if limit["supported"] else None
    ceiling = None if limit_w is None else limit_w * cfg.plausibility_margin
    v: dict[str, Any] = {"notes": [], "enforced_power_limit_w": limit_w}

    power_ok = bool(caps["power_usage"]["supported"])
    counter_ok = bool(caps["energy_counter"]["supported"])
    v["power_supported"] = power_ok
    v["counter_supported"] = counter_ok

    if load is None:
        v.update(power_responds=None, counter_responds=None, counter_power_agreement=None)
        v["notes"].append("Load phase skipped; responsiveness untested.")
    else:
        delta = cfg.responsive_delta_w
        v["power_responds"] = power_ok and load["power_mean_w"] - idle["power_mean_w"] >= delta
        v["counter_responds"] = (
            counter_ok
            and load["counter_monotonic"]
            and load["counter_mean_power_w"] - idle["counter_mean_power_w"] >= delta
        )
        if power_ok and counter_ok and load["counter_delta_j"] > 0:
            rel = abs(load["energy_trapz_j"] - load["counter_delta_j"]) / load["counter_delta_j"]
            v["counter_power_rel_diff"] = round(rel, 4)
            v["counter_power_agreement"] = rel <= cfg.agreement_tolerance
        else:
            v["counter_power_agreement"] = None

    def plausible(key: str) -> bool | None:
        """Check every window's ``key`` against the enforced power limit."""
        if ceiling is None:
            return None
        values = [w[key] for w in measured if key in w]
        return bool(values) and max(values) <= ceiling

    v["power_plausible"] = plausible("power_max_w") if power_ok else False
    v["counter_plausible"] = plausible("counter_mean_power_w") if counter_ok else False
    v["power_trusted"] = bool(v["power_responds"]) and v["power_plausible"] is not False
    v["counter_trusted"] = (
        bool(v["counter_responds"])
        and v["counter_power_agreement"] is True
        and v["counter_plausible"] is not False
    )

    if counter_ok and not v["counter_trusted"]:
        reasons: list[str] = []
        if v["counter_power_agreement"] is False:
            reasons.append(
                f"disagrees with integrated power by {v['counter_power_rel_diff']:.0%} under load"
            )
        if v["counter_plausible"] is False:
            peak = max(w["counter_mean_power_w"] for w in measured if "counter_mean_power_w" in w)
            reasons.append(
                f"implies {peak:.0f} W mean power, above the {limit_w:.0f} W enforced limit"
            )
        if v["counter_responds"] is False:
            reasons.append("does not respond to load")
        v["counter_rejected_because"] = reasons

    if v["counter_trusted"]:
        v["recommended_method"] = "energy_counter"
    elif v["power_trusted"]:
        v["recommended_method"] = "power_sampling"
    elif load is None and (power_ok or counter_ok):
        v["recommended_method"] = "inconclusive_rerun_with_load"
    else:
        v["recommended_method"] = "unsupported_use_wall_meter"

    interval = (load or idle).get("power_update_interval_s")
    if interval is not None and interval >= cfg.power_refresh_note_s:
        v["power_update_interval_s"] = round(interval, 3)
        v["notes"].append(
            f"Power refreshes every ~{interval:.2f} s and is smoothed; integrate energy over "
            "batched windows with a post-load tail rather than per-query windows."
        )
    v["passed"] = v["recommended_method"] in {"energy_counter", "power_sampling"}
    return v


def git_commit() -> str | None:
    """Return the current commit hash, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _as_str(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def run(cfg: GateConfig) -> dict[str, Any]:
    """Execute the gate and return the full report."""
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        interval_s = cfg.interval_ms / 1000.0

        caps: dict[str, Any] = {}
        for key, fn in (
            ("power_usage", pynvml.nvmlDeviceGetPowerUsage),
            ("energy_counter", pynvml.nvmlDeviceGetTotalEnergyConsumption),
            ("power_limit", pynvml.nvmlDeviceGetPowerManagementLimit),
            ("enforced_power_limit", pynvml.nvmlDeviceGetEnforcedPowerLimit),
            ("performance_state", pynvml.nvmlDeviceGetPerformanceState),
        ):
            value, err = _read(fn, handle)
            caps[key] = {"supported": err is None, "value": value, "error": err}
        field_values, err = _read(
            pynvml.nvmlDeviceGetFieldValues,
            handle,
            [pynvml.NVML_FI_DEV_POWER_INSTANT, pynvml.NVML_FI_DEV_POWER_AVERAGE],
        )
        caps["field_power_instant_average"] = (
            {"supported": False, "error": err}
            if field_values is None
            else {
                "supported": [int(f.nvmlReturn) == 0 for f in field_values],
                "value_mw": [
                    int(f.value.uiVal) if int(f.nvmlReturn) == 0 else None for f in field_values
                ],
            }
        )
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        device = {
            "name": _as_str(pynvml.nvmlDeviceGetName(handle)),
            "driver": _as_str(pynvml.nvmlSystemGetDriverVersion()),
            "nvml": _as_str(pynvml.nvmlSystemGetNVMLVersion()),
            "vram_total_mb": round(memory.total / 2**20),
            "vram_used_mb_at_start": round(memory.used / 2**20),
        }
        LOGGER.info("device %s | driver %s", device["name"], device["driver"])
        LOGGER.info(
            "power_usage supported=%s | energy_counter supported=%s",
            caps["power_usage"]["supported"],
            caps["energy_counter"]["supported"],
        )

        LOGGER.info("idle window: %.0f s (no CUDA context yet)", cfg.idle_s)
        idle_samples = sample_window(handle, cfg.idle_s, interval_s)
        idle = summarise(idle_samples, "idle")
        processes = list_gpu_processes(handle)
        # Utilisation decides busy. On Windows WDDM every desktop process that
        # holds a GPU context (explorer, browsers, editors) is listed as a
        # compute app with unknown memory, so process presence says nothing.
        busy = idle.get("util_mean_pct", 0.0) > cfg.busy_util_pct
        LOGGER.info(
            "idle mean power %s W | util %s%% | busy=%s",
            idle.get("power_mean_w"),
            idle.get("util_mean_pct"),
            busy,
        )

        load = cooldown = load_info = None
        all_samples = [("idle", s) for s in idle_samples]
        if cfg.allow_load and not busy:
            LOGGER.info("load window: %.0f s fp16 matmul %d^2", cfg.load_s, cfg.matrix_size)
            load_samples, load_info = synthetic_load(cfg, handle)
            load = summarise(load_samples, "load")
            LOGGER.info("load mean power %s W", load.get("power_mean_w"))
            cooldown_samples = sample_window(handle, cfg.cooldown_s, interval_s)
            cooldown = summarise(cooldown_samples, "cooldown")
            all_samples += [("load", s) for s in load_samples]
            all_samples += [("cooldown", s) for s in cooldown_samples]
        elif busy:
            LOGGER.warning("GPU busy (utilisation above threshold); load phase skipped")

        windows = {"idle": idle, "load": load, "cooldown": cooldown}
        report = {
            "gate": "gate1_nvml_telemetry",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "nvidia_ml_py": getattr(pynvml, "__version__", None),
            "config": asdict(cfg),
            "device": device,
            "capabilities": caps,
            "gpu_processes_at_idle": processes,
            "gpu_busy_at_idle": busy,
            "windows": windows,
            "load_info": load_info,
            "verdict": verdict(cfg, caps, windows),
        }
    finally:
        pynvml.nvmlShutdown()

    out_dir = REPO_ROOT / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate1_nvml_telemetry.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    with (out_dir / "gate1_nvml_samples.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["window", "t_s", "power_mw", "energy_mj", "util_pct"])
        t0 = all_samples[0][1].t
        for window, s in all_samples:
            writer.writerow([window, f"{s.t - t0:.4f}", s.power_mw, s.energy_mj, s.util_pct])
    return report


def reverdict(report_path: Path) -> dict[str, Any]:
    """Recompute the verdict of a saved report offline, keeping the old verdict.

    The superseded verdict is appended to ``verdict_history`` so the file keeps
    a record of what the earlier rules concluded.

    Raises
    ------
    FileNotFoundError
        If ``report_path`` does not exist.
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(GateConfig)}
    cfg = GateConfig(**{k: v for k, v in report["config"].items() if k in known})
    now = datetime.now(UTC).isoformat()
    report.setdefault("verdict_history", []).append(
        {"superseded_at_utc": now, "verdict": report["verdict"]}
    )
    report["config"] = asdict(cfg)
    report["verdict"] = verdict(cfg, report["capabilities"], report["windows"])
    report["verdict_recomputed_at_utc"] = now
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def parse_args() -> tuple[GateConfig, Path | None]:
    """Build a :class:`GateConfig` and optional re-verdict path from the CLI."""
    defaults = GateConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--idle-s", type=float, default=defaults.idle_s)
    parser.add_argument("--load-s", type=float, default=defaults.load_s)
    parser.add_argument("--cooldown-s", type=float, default=defaults.cooldown_s)
    parser.add_argument("--interval-ms", type=int, default=defaults.interval_ms)
    parser.add_argument("--matrix-size", type=int, default=defaults.matrix_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--no-load", action="store_true", help="probe and idle-sample only")
    parser.add_argument("--out-dir", default=defaults.out_dir)
    parser.add_argument(
        "--reverdict",
        type=Path,
        default=None,
        metavar="REPORT_JSON",
        help="recompute the verdict of a saved report without using the GPU",
    )
    ns = parser.parse_args()
    cfg = GateConfig(
        idle_s=ns.idle_s,
        load_s=ns.load_s,
        cooldown_s=ns.cooldown_s,
        interval_ms=ns.interval_ms,
        matrix_size=ns.matrix_size,
        seed=ns.seed,
        allow_load=not ns.no_load,
        out_dir=ns.out_dir,
    )
    return cfg, ns.reverdict


def main() -> None:
    """Run the gate, or re-evaluate a saved report."""
    cfg, reverdict_path = parse_args()
    log_dir = REPO_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "gate1.log")],
    )
    report = reverdict(reverdict_path) if reverdict_path is not None else run(cfg)
    LOGGER.info("verdict: %s", json.dumps(report["verdict"]))


if __name__ == "__main__":
    main()
