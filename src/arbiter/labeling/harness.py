"""Oracle labelling harness (experimental protocol, sections 5 and 6.2).

Every selected prompt is answered by every cell of the size x mode grid. Runs are
model-major: one model is loaded, both modes are swept, and the model is unloaded
before the next is loaded. Within a model, prompts are processed in batches, and
each batch runs one mode at a time as a contiguous *window*, with the mode that goes
first alternating between batches (ABBA). Energy is attributed per window, never per
generation, because NVML power refreshes every 0.5 s and is smoothed (gate 1).

Per model session:

1. Wait until the GPU is idle and has enough free memory.
2. Load the model with one no-think and one think warm-up generation, which are not
   recorded (gate 2 found a one-off prefill stall per server session).
3. Check the model sits fully in VRAM.
4. Record an idle baseline window with the model loaded.
5. For each batch and mode: run the generations, hold a tail so the smoothed power
   reading decays inside the window, then inspect utilisation at the end of the
   tail. If another workload is using the GPU, or a generation had to reload the
   model, the window is discarded and rerun once the GPU is idle again.
6. Record a closing baseline window and unload.

Telemetry streams to one CSV per session. Generations and windows are appended to
JSONL after each accepted window, so an interrupted stage resumes at its first
incomplete window.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import statistics
import subprocess
import time
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import httpx
import ollama
import pynvml
import yaml

from arbiter.data.corpus import PromptItem, Tokenizer, count_tokens, read_corpus
from arbiter.energy.telemetry import TelemetryRecorder, TelemetrySample
from arbiter.utils.jsonl import append_jsonl, read_jsonl

LOGGER = logging.getLogger("arbiter.labelling")
T = TypeVar("T")
MODES: tuple[str, str] = ("no_think", "think")
SELECTIONS = ("pilot", "main", "noise", "all", "cells")
PLACEMENT_FULL = 0.999
RETRYABLE: tuple[type[Exception], ...] = (
    ollama.ResponseError,
    httpx.HTTPError,
    ConnectionError,
    TimeoutError,
)
REQUIRED_KEYS = (
    "seed",
    "ollama_host",
    "request_timeout_s",
    "keep_alive",
    "num_ctx",
    "tokenizer_id",
    "corpus",
    "raw_dir",
    "models",
    "decoding",
    "window",
    "session",
    "telemetry",
    "gpu_guard",
    "server",
    "warmup",
    "stages",
)
RecordKey = tuple[str, str, str, int, str]


# ---------------------------------------------------------------------------
# Configuration and selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    """A labelling stage with its overrides merged into the base config.

    ``corpus`` replaces the base corpus, as the probe stages for R3 and the cascade do.
    ``request`` holds extra chat parameters, such as ``logprobs``, sent with every
    generation.
    """

    name: str
    select: str
    samples: tuple[int, ...]
    variant: str
    sizes: tuple[str, ...] | None
    per_family: int | None
    raw_dir: Path
    batch_size: int
    baseline_s: float
    end_baseline_s: float
    decoding: dict[str, dict[str, Any]]
    num_ctx: int
    cells: frozenset[tuple[str, str, str]] | None = None
    corpus: Path | None = None
    request: dict[str, Any] = field(default_factory=dict)

    def allows(self, prompt_id: str, size: str, mode: str) -> bool:
        """Return whether the stage runs this cell; unrestricted stages run every cell."""
        return self.cells is None or (prompt_id, size, mode) in self.cells


def read_cells(path: Path) -> frozenset[tuple[str, str, str]]:
    """Read the (prompt_id, size, mode) cells that a restricted stage reruns.

    Raises
    ------
    ValueError
        If a column is missing or a row names an unknown mode.
    """
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"prompt_id", "size", "mode"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = [(row["prompt_id"], row["size"], row["mode"]) for row in reader]
    unknown = {mode for _, _, mode in rows} - set(MODES)
    if unknown:
        raise ValueError(f"{path}: unknown modes {sorted(unknown)}")
    return frozenset(rows)


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the labelling config.

    Raises
    ------
    ValueError
        If a required key or a mode's decoding settings are missing.
    """
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    for mode in MODES:
        if mode not in cfg["decoding"]:
            raise ValueError(f"{path}: decoding settings missing for mode '{mode}'")
    return cfg


def resolve_stage(cfg: dict[str, Any], name: str, repo_root: Path) -> Stage:
    """Merge a stage's overrides into the base config.

    Raises
    ------
    KeyError
        If the stage is not defined.
    ValueError
        If the stage's ``select`` is not a known selection.
    """
    if name not in cfg["stages"]:
        raise KeyError(f"unknown stage '{name}'; defined: {sorted(cfg['stages'])}")
    spec = cfg["stages"][name]
    if spec["select"] not in SELECTIONS:
        raise ValueError(f"stage {name}: select must be one of {SELECTIONS}")
    overrides = spec.get("decoding", {})
    sizes = spec.get("sizes")
    cells = None
    if spec["select"] == "cells":
        if "cells_file" not in spec:
            raise ValueError(f"stage {name}: select 'cells' needs a cells_file")
        cells = read_cells(repo_root / spec["cells_file"])
    return Stage(
        name=name,
        select=spec["select"],
        samples=tuple(int(s) for s in spec.get("samples", [0])),
        variant=str(spec.get("variant", "base")),
        sizes=tuple(str(s) for s in sizes) if sizes else None,
        per_family=spec.get("per_family"),
        raw_dir=repo_root / spec.get("raw_dir", cfg["raw_dir"]),
        batch_size=int(spec.get("batch_size", cfg["window"]["batch_size"])),
        baseline_s=float(spec.get("baseline_s", cfg["session"]["baseline_s"])),
        end_baseline_s=float(spec.get("end_baseline_s", cfg["session"]["end_baseline_s"])),
        decoding={mode: {**cfg["decoding"][mode], **overrides.get(mode, {})} for mode in MODES},
        num_ctx=int(spec.get("num_ctx", cfg["num_ctx"])),
        cells=cells,
        corpus=repo_root / spec["corpus"] if "corpus" in spec else None,
        request=dict(spec.get("request", {})),
    )


def select_items(items: Sequence[PromptItem], stage: Stage, seed: int) -> list[PromptItem]:
    """Pick a stage's prompts and shuffle them into a fixed stage order.

    The shuffle mixes families within every batch, so family is not confounded with
    elapsed time or thermal state, and the order is identical for every model.
    """
    if stage.select == "noise":
        chosen = [item for item in items if item.noise]
    elif stage.select == "all":
        chosen = list(items)
    elif stage.select == "cells":
        listed = {prompt_id for prompt_id, _, _ in stage.cells or ()}
        chosen = [item for item in items if item.prompt_id in listed]
    else:
        chosen = [item for item in items if item.subset == stage.select]
    if stage.per_family is not None:
        counts: Counter[str] = Counter()
        limited: list[PromptItem] = []
        for item in chosen:
            if counts[item.family] < stage.per_family:
                limited.append(item)
                counts[item.family] += 1
        chosen = limited
    random.Random(f"{seed}|{stage.name}").shuffle(chosen)
    return chosen


def cell_seed(base: int, prompt_id: str, tag: str, mode: str, sample: int, variant: str) -> int:
    """Derive a deterministic, decorrelated sampler seed for one generation."""
    key = f"{base}|{prompt_id}|{tag}|{mode}|{sample}|{variant}"
    return zlib.crc32(key.encode()) & 0x7FFFFFFF


def record_key(record: dict[str, Any]) -> RecordKey:
    """Identify a generation by prompt, model, mode, sample index and variant."""
    return (
        record["prompt_id"],
        record["tag"],
        record["mode"],
        int(record["sample"]),
        record["variant"],
    )


def chunks(seq: Sequence[T], size: int) -> list[list[T]]:
    """Split ``seq`` into consecutive lists of at most ``size`` items."""
    if size < 1:
        raise ValueError("chunk size must be positive")
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def tail_contaminated(
    samples: Sequence[TelemetrySample], max_util_pct: float
) -> tuple[bool, float | None]:
    """Decide whether another workload was using the GPU at the end of a window tail.

    Our generations have finished before the tail starts, so sustained utilisation at
    its end belongs to someone else. Without utilisation readings the check cannot be
    made and the window is treated as clean.
    """
    utils = [s.util_pct for s in samples if s.util_pct is not None]
    if not utils:
        return False, None
    mean = statistics.fmean(utils)
    return mean > max_util_pct, mean


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def git_commit(repo_root: Path) -> str | None:
    """Return the current commit hash, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def utc_now() -> str:
    """Return the current UTC time in ISO format."""
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Ollama and GPU
# ---------------------------------------------------------------------------


def load_tokenizer(tokenizer_id: str) -> Tokenizer:
    """Load the Qwen3 tokenizer, shared by every size in the family."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_id)


def ollama_version(host: str) -> str | None:
    """Return the Ollama server version, or None if it cannot be read."""
    try:
        return str(httpx.get(f"{host}/api/version", timeout=10).json().get("version"))
    except (httpx.HTTPError, ValueError):
        return None


def verify_models(
    cfg: dict[str, Any], client: ollama.Client
) -> tuple[list[dict[str, Any]], list[str]]:
    """Check every configured tag is a hybrid-thinking Qwen3 model, from metadata alone.

    A tag that resolves to a 2507 release still loads and answers, but it either
    cannot think or cannot stop thinking, which would silently break the grid.

    Returns
    -------
    manifest : list of dict
        Digest, size, quantisation, context length and capabilities per tag.
    problems : list of str
        Human-readable failures; empty when every tag passes.
    """
    listed = {model.model: model for model in client.list().models}
    manifest: list[dict[str, Any]] = []
    problems: list[str] = []
    for spec in cfg["models"]:
        tag = spec["tag"]
        if tag not in listed:
            problems.append(f"{tag}: not pulled")
            continue
        info = client.show(tag)
        modelinfo = dict(getattr(info, "modelinfo", None) or {})
        context = next((v for k, v in modelinfo.items() if k.endswith(".context_length")), None)
        capabilities = list(getattr(info, "capabilities", None) or [])
        details = getattr(info, "details", None)
        manifest.append(
            {
                "size": spec["size"],
                "tag": tag,
                "digest": listed[tag].digest,
                "size_bytes": listed[tag].size,
                "parameter_size": getattr(details, "parameter_size", None),
                "quantization": getattr(details, "quantization_level", None),
                "context_length": context,
                "capabilities": capabilities,
            }
        )
        if "thinking" not in capabilities:
            problems.append(f"{tag}: no 'thinking' capability (has {capabilities})")
        if context != spec["expected_context_length"]:
            problems.append(
                f"{tag}: context_length {context} != expected {spec['expected_context_length']}"
            )
    return manifest, problems


def wait_for_server(client: ollama.Client, server_cfg: dict[str, Any]) -> None:
    """Block until the Ollama server answers, for example after an app restart.

    Raises
    ------
    RuntimeError
        If the server stays unreachable for longer than ``server_cfg['wait_s']``.
    """
    deadline = time.perf_counter() + float(server_cfg["wait_s"])
    while True:
        try:
            client.list()
            return
        except RETRYABLE as exc:
            if time.perf_counter() > deadline:
                raise RuntimeError("Ollama server unreachable") from exc
            LOGGER.warning("Ollama unreachable (%s); retrying in %s s", exc, server_cfg["poll_s"])
            time.sleep(float(server_cfg["poll_s"]))


def gpu_placement(client: ollama.Client, tag: str) -> float | None:
    """Return the fraction of a loaded model's memory held in VRAM, None if not loaded."""
    for model in client.ps().models:
        if model.model == tag and model.size:
            return model.size_vram / model.size
    return None


def is_loaded(client: ollama.Client, tag: str) -> bool:
    """Return True if Ollama currently holds ``tag`` in memory."""
    return any(model.model == tag for model in client.ps().models)


def unload(client: ollama.Client, tag: str) -> None:
    """Ask Ollama to unload ``tag``, logging rather than raising on failure."""
    try:
        client.generate(model=tag, prompt="", keep_alive=0)
    except RETRYABLE as exc:
        LOGGER.warning("could not unload %s: %s", tag, exc)


class GpuGuard:
    """Hold the run while another workload uses the GPU, judged from recorder samples."""

    def __init__(self, recorder: TelemetryRecorder, cfg: dict[str, Any]) -> None:
        self._recorder = recorder
        self._max_util = float(cfg["max_util_pct"])
        self._poll_s = float(cfg["poll_s"])
        self._log_every_s = float(cfg["log_every_s"])

    def raise_if_failed(self) -> None:
        """Stop the run if telemetry has died, since energy would be missing.

        Raises
        ------
        RuntimeError
            If the recorder thread failed or exited.
        """
        if self._recorder.error is not None:
            raise RuntimeError("telemetry recorder failed") from self._recorder.error
        if not self._recorder.is_alive():
            raise RuntimeError("telemetry recorder stopped unexpectedly")

    def wait_until_idle(self, seconds: float, reason: str) -> float:
        """Return once mean utilisation stays at or below the ceiling for ``seconds``.

        Returns
        -------
        float
            Seconds spent waiting.
        """
        start = time.perf_counter()
        last_log = start
        idle_since: float | None = None
        while True:
            self.raise_if_failed()
            time.sleep(self._poll_s)
            now = time.perf_counter()
            utils = [
                s.util_pct
                for s in self._recorder.window(now - self._poll_s, now)
                if s.util_pct is not None
            ]
            if utils and statistics.fmean(utils) > self._max_util:
                idle_since = None
            elif idle_since is None:
                idle_since = now - self._poll_s
            if idle_since is not None and now - idle_since >= seconds:
                waited = now - start
                if waited > seconds + 4 * self._poll_s:
                    LOGGER.info("GPU idle again after %.1f min (%s)", waited / 60, reason)
                return waited
            if now - last_log >= self._log_every_s:
                LOGGER.warning(
                    "waiting for the GPU to be idle (%s): %.0f min so far",
                    reason,
                    (now - start) / 60,
                )
                last_log = now

    def wait_for_vram(self, min_free_mb: float, reason: str) -> None:
        """Return once at least ``min_free_mb`` of VRAM is free."""
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        start = time.perf_counter()
        last_log = start - self._log_every_s
        while True:
            free_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).free / 2**20
            if free_mb >= min_free_mb:
                return
            now = time.perf_counter()
            if now - last_log >= self._log_every_s:
                LOGGER.warning(
                    "waiting for VRAM (%s): %.0f MB free, need %.0f MB",
                    reason,
                    free_mb,
                    min_free_mb,
                )
                last_log = now
            time.sleep(self._poll_s * 6)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class Context:
    """State shared by every session of a stage."""

    cfg: dict[str, Any]
    stage: Stage
    client: ollama.Client
    tokenizer: Tokenizer
    done: set[RecordKey]
    generations_path: Path
    windows_path: Path
    sessions_path: Path


def warm_up(ctx: Context, tag: str) -> None:
    """Load ``tag`` and run one unrecorded generation per mode."""
    for mode in MODES:
        spec = ctx.cfg["warmup"][mode]
        ctx.client.chat(
            model=tag,
            messages=[{"role": "user", "content": spec["prompt"]}],
            think=mode == "think",
            options={
                "num_predict": spec["num_predict"],
                "num_ctx": ctx.stage.num_ctx,
                "seed": ctx.cfg["seed"],
            },
            keep_alive=ctx.cfg["keep_alive"],
        )


def load_model(ctx: Context, spec: dict[str, Any], guard: GpuGuard) -> float | None:
    """Load and warm up a model once the GPU is idle, insisting on full VRAM placement.

    Returns
    -------
    float or None
        Fraction of the model in VRAM, None if Ollama does not report it.

    Raises
    ------
    RuntimeError
        If the model cannot be placed fully in VRAM within the retry budget.
    """
    cfg, tag = ctx.cfg, spec["tag"]
    retries = int(cfg["gpu_guard"]["placement_retries"])
    for attempt in range(1, retries + 1):
        guard.wait_until_idle(float(cfg["gpu_guard"]["idle_required_s"]), f"before loading {tag}")
        guard.wait_for_vram(float(spec["min_free_vram_mb"]), f"before loading {tag}")
        try:
            warm_up(ctx, tag)
            placement = gpu_placement(ctx.client, tag)
        except RETRYABLE as exc:
            LOGGER.error("%s: load attempt %d failed: %s", tag, attempt, exc)
            wait_for_server(ctx.client, cfg["server"])
            continue
        if placement is None or placement >= PLACEMENT_FULL:
            share = "unknown share" if placement is None else f"{placement:.0%}"
            LOGGER.info("%s: loaded and warmed up, %s in VRAM", tag, share)
            return placement
        LOGGER.warning("%s: only %.0f%% in VRAM; unloading and retrying", tag, placement * 100)
        unload(ctx.client, tag)
    raise RuntimeError(f"{tag}: not fully in VRAM after {retries} attempts")


def serialise_logprobs(logprobs: Sequence[Any] | None) -> list[dict[str, Any]]:
    """Convert Ollama log probabilities into plain dicts for JSONL."""
    out = []
    for entry in logprobs or ():
        top = getattr(entry, "top_logprobs", None) or ()
        out.append(
            {
                "token": entry.token,
                "logprob": float(entry.logprob),
                "top_logprobs": [{"token": t.token, "logprob": float(t.logprob)} for t in top],
            }
        )
    return out


def generate_one(
    ctx: Context, spec: dict[str, Any], item: PromptItem, mode: str, sample: int
) -> dict[str, Any]:
    """Generate one response and record its text, token counts and timings."""
    stage = ctx.stage
    seed = cell_seed(ctx.cfg["seed"], item.prompt_id, spec["tag"], mode, sample, stage.variant)
    options = {**stage.decoding[mode], "seed": seed, "num_ctx": stage.num_ctx}
    t_start = time.perf_counter()
    response = ctx.client.chat(
        model=spec["tag"],
        messages=[{"role": "user", "content": item.text}],
        think=mode == "think",
        options=options,
        keep_alive=ctx.cfg["keep_alive"],
        **stage.request,
    )
    t_end = time.perf_counter()
    content = response.message.content or ""
    thinking = getattr(response.message, "thinking", None) or ""
    ns = 1e-9
    record: dict[str, Any] = {
        "prompt_id": item.prompt_id,
        "family": item.family,
        "source": item.source,
        "subset": item.subset,
        "size": spec["size"],
        "tag": spec["tag"],
        "mode": mode,
        "sample": sample,
        "variant": stage.variant,
        "seed": seed,
        "num_predict": options.get("num_predict"),
        "content": content,
        "thinking": thinking,
        "done_reason": response.done_reason,
        "eval_count": response.eval_count or 0,
        "prompt_eval_count": response.prompt_eval_count or 0,
        "eval_duration_s": (response.eval_duration or 0) * ns,
        "prompt_eval_duration_s": (response.prompt_eval_duration or 0) * ns,
        "load_duration_s": (response.load_duration or 0) * ns,
        "total_duration_s": (response.total_duration or 0) * ns,
        "t_start_perf": t_start,
        "t_end_perf": t_end,
        "wall_s": t_end - t_start,
        "tokens_thinking": count_tokens(ctx.tokenizer, thinking),
        "tokens_content": count_tokens(ctx.tokenizer, content),
        "timestamp_utc": utc_now(),
    }
    if stage.request.get("logprobs"):
        record["logprobs"] = serialise_logprobs(getattr(response, "logprobs", None))
    return record


def run_window(
    ctx: Context,
    spec: dict[str, Any],
    recorder: TelemetryRecorder,
    mode: str,
    todo: Sequence[tuple[PromptItem, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one contiguous single-mode window, including its tail and contamination check."""
    window_cfg = ctx.cfg["window"]
    start = time.perf_counter()
    records = [generate_one(ctx, spec, item, mode, sample) for item, sample in todo]
    gen_end = time.perf_counter()
    time.sleep(float(window_cfg["tail_s"]))
    end = time.perf_counter()
    tail = recorder.window(end - float(window_cfg["tail_check_s"]), end)
    busy, tail_util = tail_contaminated(tail, float(ctx.cfg["gpu_guard"]["max_util_pct"]))
    reloaded = any(r["load_duration_s"] > float(window_cfg["max_load_s"]) for r in records)
    window = {
        "start_perf": start,
        "gen_end_perf": gen_end,
        "end_perf": end,
        "n_generations": len(records),
        "wall_s_sum": sum(r["wall_s"] for r in records),
        "tail_util_mean": tail_util,
        "other_workload": busy,
        "reloaded": reloaded,
        "contaminated": busy or reloaded,
        "timestamp_utc": utc_now(),
    }
    return window, records


def measure_baseline(
    ctx: Context,
    spec: dict[str, Any],
    guard: GpuGuard,
    recorder: TelemetryRecorder,
    session_id: str,
    kind: str,
    seconds: float,
) -> None:
    """Record an idle window with the model loaded, retrying if the GPU was in use."""
    max_util = float(ctx.cfg["gpu_guard"]["max_util_pct"])
    attempts = 3
    for attempt in range(1, attempts + 1):
        guard.raise_if_failed()
        start = time.perf_counter()
        time.sleep(seconds)
        end = time.perf_counter()
        samples = recorder.window(start, end)
        utils = [s.util_pct for s in samples if s.util_pct is not None]
        powers = [s.power_mw for s in samples if s.power_mw is not None]
        util = statistics.fmean(utils) if utils else None
        contaminated = util is not None and util > max_util
        window = {
            "window_id": f"{session_id}-{kind}-a{attempt}",
            "session_id": session_id,
            "kind": kind,
            "stage": ctx.stage.name,
            "tag": spec["tag"],
            "size": spec["size"],
            "attempt": attempt,
            "telemetry_file": f"telemetry/{session_id}.csv",
            "start_perf": start,
            "end_perf": end,
            "util_mean": util,
            "power_mean_w": statistics.fmean(powers) / 1000.0 if powers else None,
            "contaminated": contaminated,
            "accepted": not contaminated or attempt == attempts,
            "timestamp_utc": utc_now(),
        }
        append_jsonl(ctx.windows_path, [window])
        LOGGER.info(
            "%s: %s %.2f W mean over %.0f s (util %s%%)",
            spec["tag"],
            kind,
            window["power_mean_w"] or float("nan"),
            seconds,
            "n/a" if util is None else f"{util:.0f}",
        )
        if window["accepted"]:
            return
        guard.wait_until_idle(float(ctx.cfg["gpu_guard"]["idle_required_s"]), kind)


def run_batch_mode(
    ctx: Context,
    spec: dict[str, Any],
    guard: GpuGuard,
    recorder: TelemetryRecorder,
    session_id: str,
    batch_index: int,
    mode: str,
    todo: Sequence[tuple[PromptItem, int]],
    placement: float | None,
) -> float | None:
    """Run one batch in one mode until a window is accepted.

    A window is rerun if a request fails, another workload was using the GPU, or the
    model had to be reloaded. On the final attempt a contaminated window is kept and
    its records are flagged.

    Returns
    -------
    float or None
        The model's VRAM placement, which changes if the model was reloaded.

    Raises
    ------
    RuntimeError
        If every attempt failed with a request error.
    """
    cfg, stage, tag = ctx.cfg, ctx.stage, spec["tag"]
    max_attempts = int(cfg["window"]["max_attempts"])
    idle_required = float(cfg["gpu_guard"]["idle_required_s"])
    for attempt in range(1, max_attempts + 1):
        window_id = f"{session_id}-b{batch_index:04d}-{mode}-a{attempt}"
        base = {
            "window_id": window_id,
            "session_id": session_id,
            "kind": "batch",
            "stage": stage.name,
            "tag": tag,
            "size": spec["size"],
            "mode": mode,
            "variant": stage.variant,
            "batch": batch_index,
            "attempt": attempt,
            "telemetry_file": f"telemetry/{session_id}.csv",
        }
        guard.raise_if_failed()
        try:
            if not is_loaded(ctx.client, tag):
                placement = load_model(ctx, spec, guard)
            window, records = run_window(ctx, spec, recorder, mode, todo)
        except RETRYABLE as exc:
            LOGGER.error("%s: attempt %d failed: %s", window_id, attempt, exc)
            append_jsonl(ctx.windows_path, [{**base, "accepted": False, "reason": str(exc)}])
            if attempt == max_attempts:
                raise RuntimeError(f"{window_id}: {max_attempts} failed attempts") from exc
            wait_for_server(ctx.client, cfg["server"])
            continue
        window = {**base, **window}
        if window["contaminated"] and attempt < max_attempts:
            LOGGER.warning(
                "%s: discarded (other workload %s, tail util %s, reloaded %s); rerunning",
                window_id,
                window["other_workload"],
                window["tail_util_mean"],
                window["reloaded"],
            )
            append_jsonl(
                ctx.windows_path, [{**window, "accepted": False, "reason": "contaminated"}]
            )
            guard.wait_until_idle(idle_required, f"rerun {window_id}")
            continue
        if window["contaminated"]:
            LOGGER.warning("%s: contaminated on the last attempt; records flagged", window_id)
        window["accepted"] = True
        for record in records:
            record.update(
                window_id=window_id,
                session_id=session_id,
                gpu_placement=placement,
                contaminated=window["contaminated"],
            )
        append_jsonl(ctx.generations_path, records)
        append_jsonl(ctx.windows_path, [window])
        ctx.done.update(record_key(record) for record in records)
        return placement
    raise AssertionError("unreachable")


def run_session(ctx: Context, spec: dict[str, Any], items: Sequence[PromptItem]) -> None:
    """Run every outstanding generation of one model for the current stage."""
    cfg, stage, tag, size = ctx.cfg, ctx.stage, spec["tag"], spec["size"]
    items = [item for item in items if any(stage.allows(item.prompt_id, size, m) for m in MODES)]
    keys = [
        (item.prompt_id, tag, mode, sample, stage.variant)
        for item in items
        for mode in MODES
        if stage.allows(item.prompt_id, size, mode)
        for sample in stage.samples
    ]
    remaining = sum(key not in ctx.done for key in keys)
    if remaining == 0:
        LOGGER.info("%s: stage %s already complete", tag, stage.name)
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    session_id = f"{stage.name}-{spec['size']}-{stamp}"
    recorder = TelemetryRecorder(
        stage.raw_dir / "telemetry" / f"{session_id}.csv",
        float(cfg["telemetry"]["interval_ms"]) / 1000.0,
    )
    recorder.start()
    guard = GpuGuard(recorder, cfg["gpu_guard"])
    started = time.perf_counter()
    session: dict[str, Any] = {
        "session_id": session_id,
        "stage": stage.name,
        "size": spec["size"],
        "tag": tag,
        "telemetry_file": f"telemetry/{session_id}.csv",
        "started_utc": utc_now(),
        "to_run": remaining,
        "of_total": len(keys),
    }
    LOGGER.info(
        "%s: session %s, %d of %d generations to run", tag, session_id, remaining, len(keys)
    )
    placement: float | None = None
    finished = 0
    try:
        for model in cfg["models"]:
            if model["tag"] != tag and is_loaded(ctx.client, model["tag"]):
                unload(ctx.client, model["tag"])
        placement = load_model(ctx, spec, guard)
        time.sleep(float(cfg["session"]["settle_s"]))
        measure_baseline(ctx, spec, guard, recorder, session_id, "baseline_start", stage.baseline_s)
        batches = chunks(items, stage.batch_size)
        gen_start = time.perf_counter()
        for batch_index, batch in enumerate(batches):
            order = MODES if batch_index % 2 == 0 else MODES[::-1]
            for mode in order:
                todo = [
                    (item, sample)
                    for item in batch
                    if stage.allows(item.prompt_id, size, mode)
                    for sample in stage.samples
                    if (item.prompt_id, tag, mode, sample, stage.variant) not in ctx.done
                ]
                if not todo:
                    continue
                placement = run_batch_mode(
                    ctx, spec, guard, recorder, session_id, batch_index, mode, todo, placement
                )
                finished += len(todo)
                minutes = (time.perf_counter() - gen_start) / 60
                rate = finished / minutes if minutes > 0 else 0.0
                LOGGER.info(
                    "%s: batch %d/%d %s | %d/%d | %.1f gen/min | ~%.0f min left",
                    tag,
                    batch_index + 1,
                    len(batches),
                    mode,
                    finished,
                    remaining,
                    rate,
                    (remaining - finished) / rate if rate > 0 else float("nan"),
                )
        measure_baseline(
            ctx, spec, guard, recorder, session_id, "baseline_end", stage.end_baseline_s
        )
    finally:
        unload(ctx.client, tag)
        time.sleep(2.0)  # keep sampling past the end of the closing baseline window
        recorder.stop()
        session.update(
            ended_utc=utc_now(),
            duration_min=(time.perf_counter() - started) / 60,
            finished=finished,
            gpu_placement=placement,
            telemetry_error=None if recorder.error is None else repr(recorder.error),
        )
        append_jsonl(ctx.sessions_path, [session])
        LOGGER.info("%s: session ended after %.1f min", tag, session["duration_min"])


def run_stage(cfg: dict[str, Any], stage_name: str, repo_root: Path, config_path: Path) -> None:
    """Run one labelling stage across every configured model.

    Raises
    ------
    RuntimeError
        If model verification fails.
    ValueError
        If the stage selects no prompts.
    """
    stage = resolve_stage(cfg, stage_name, repo_root)
    corpus = stage.corpus or repo_root / cfg["corpus"]
    items = select_items(read_corpus(corpus), stage, int(cfg["seed"]))
    if not items:
        raise ValueError(f"stage {stage.name} selects no prompts")
    stage.raw_dir.mkdir(parents=True, exist_ok=True)
    generations_path = stage.raw_dir / "generations.jsonl"
    records, truncated = read_jsonl(generations_path)
    if truncated:
        generations_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
    client = ollama.Client(host=cfg["ollama_host"], timeout=cfg["request_timeout_s"])
    wait_for_server(client, cfg["server"])
    manifest, problems = verify_models(cfg, client)
    (stage.raw_dir / "models.json").write_text(json.dumps(manifest, indent=2, default=str))
    if problems:
        raise RuntimeError("model verification failed:\n  " + "\n  ".join(problems))

    pynvml.nvmlInit()
    try:
        run_info = {
            "stage": stage.name,
            "started_utc": utc_now(),
            "git_commit": git_commit(repo_root),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "ollama_version": ollama_version(cfg["ollama_host"]),
            "driver": str(pynvml.nvmlSystemGetDriverVersion()),
            "n_prompts": len(items),
            "families": dict(Counter(item.family for item in items)),
            "samples": list(stage.samples),
            "variant": stage.variant,
            "decoding": stage.decoding,
            "num_ctx": stage.num_ctx,
            "restricted_cells": None if stage.cells is None else len(stage.cells),
            "corpus": corpus.relative_to(repo_root).as_posix()
            if corpus.is_relative_to(repo_root)
            else str(corpus),
            "request": stage.request,
            "recorded_before": len(records),
        }
        append_jsonl(stage.raw_dir / "runs.jsonl", [run_info])
        LOGGER.info("stage %s: %d prompts %s", stage.name, len(items), run_info["families"])
        ctx = Context(
            cfg=cfg,
            stage=stage,
            client=client,
            tokenizer=load_tokenizer(cfg["tokenizer_id"]),
            done={record_key(r) for r in records},
            generations_path=generations_path,
            windows_path=stage.raw_dir / "windows.jsonl",
            sessions_path=stage.raw_dir / "sessions.jsonl",
        )
        for spec in cfg["models"]:
            if stage.sizes is None or spec["size"] in stage.sizes:
                run_session(ctx, spec, items)
    finally:
        pynvml.nvmlShutdown()
    LOGGER.info("stage %s complete", stage.name)
