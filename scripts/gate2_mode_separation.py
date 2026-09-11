"""Gate 2: test whether Qwen3 think and no-think modes are distinct operating points.

Experimental protocol, section 10, phase 0b. The model-size x reasoning-mode grid
assumes the ``think`` flag produces two genuinely different behaviours at every
size. Wang et al. (arXiv:2510.12680) report that hybrid-thinking models separate
the modes only partially, with reasoning leaking into no-think output. This gate
measures separation on our own checkpoints before any labelling run.

Every prompt is answered by every size in both modes, alternating mode order
across prompts to balance thermal state. Three pre-registered criteria are then
evaluated per size:

1. Length separability, as the AUC of generated-token count for telling the modes
   apart, ``AUC = U / (n_think * n_no_think)`` from the Mann-Whitney U statistic,
   read as P(a think response is longer than a no-think response).
2. Thinking leakage: the fraction of no-think responses that return a non-empty
   thinking field or a literal ``<think>`` tag in the answer.
3. Reasoning-style leakage: marker density (markers per 100 tokens) in no-think
   answers divided by marker density in think-mode reasoning traces.

Usage::

    python scripts/gate2_mode_separation.py --prepare-only    # datasets only, no GPU
    python scripts/gate2_mode_separation.py --verify-models   # Ollama metadata, no GPU
    python scripts/gate2_mode_separation.py                   # full run, uses the GPU
    python scripts/gate2_mode_separation.py --analyse-only    # recompute from records

The full run resumes from existing records if interrupted.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import subprocess
import time
import zlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import yaml

if TYPE_CHECKING:
    import ollama

LOGGER = logging.getLogger("arbiter.gate2")
REPO_ROOT = Path(__file__).resolve().parents[1]
MODES: tuple[str, str] = ("no_think", "think")
LETTERS = "ABCDEFGH"
REQUIRED_KEYS = (
    "seed",
    "ollama_host",
    "request_timeout_s",
    "keep_alive",
    "num_ctx",
    "max_prompt_tokens",
    "max_errors",
    "bootstrap_resamples",
    "tokenizer_id",
    "paths",
    "models",
    "decoding",
    "sources",
    "thresholds",
    "reasoning_markers",
    "gpu_guard",
)


class Tokenizer(Protocol):
    """The subset of a Hugging Face tokenizer this script uses."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        """Encode text to token ids."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the YAML config.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a required key is missing, a mode lacks decoding settings, or the
        generation cap differs between modes.
    """
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    for mode in MODES:
        if mode not in cfg["decoding"]:
            raise ValueError(f"{path}: decoding settings missing for mode '{mode}'")
    caps = {cfg["decoding"][mode].get("num_predict") for mode in MODES}
    if len(caps) != 1:
        raise ValueError(
            f"{path}: num_predict must be identical across modes for this gate; an "
            "asymmetric cap would manufacture length separation"
        )
    return cfg


def repo_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a configured path relative to the repository root."""
    return REPO_ROOT / cfg["paths"][key]


# ---------------------------------------------------------------------------
# Prompt set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptItem:
    """A formatted prompt drawn from a source benchmark.

    Attributes
    ----------
    prompt_id : str
        ``<source>-<row index>``, stable for a given dataset revision.
    family : str
        ``math``, ``code`` or ``mcq``.
    source : str
        Source key from the config.
    text : str
        The user message sent to the model.
    reference : str or None
        Gold answer for math and MCQ; None for code, which this gate does not grade.
    meta : dict
        Source-specific fields kept for traceability.
    """

    prompt_id: str
    family: str
    source: str
    text: str
    reference: str | None
    meta: dict[str, Any] = field(default_factory=dict)


FormatResult = tuple[str, str | None, dict[str, Any]]

MCQ_INSTRUCTION = (
    "Respond with the letter of the correct option on the last line, formatted as 'Answer: X'."
)


def _mcq_text(question: str, options: list[str]) -> str:
    lines = [f"{LETTERS[i]}. {option}" for i, option in enumerate(options)]
    return f"{question}\n\n" + "\n".join(lines) + f"\n\n{MCQ_INSTRUCTION}"


def format_gsm8k(row: dict[str, Any]) -> FormatResult:
    """Format a GSM8K row; the gold answer follows ``####`` in the solution."""
    gold = row["answer"].split("####")[-1].strip().replace(",", "")
    text = f"{row['question']}\n\nGive your final answer as a number in \\boxed{{}}."
    return text, gold, {}


def format_math500(row: dict[str, Any]) -> FormatResult:
    """Format a MATH-500 row."""
    text = f"{row['problem']}\n\nPut your final answer in \\boxed{{}}."
    return text, str(row["answer"]), {"level": row.get("level"), "subject": row.get("subject")}


def format_humaneval(row: dict[str, Any]) -> FormatResult:
    """Format a HumanEval row as a function-completion request."""
    text = (
        "Complete the following Python function. Reply with the full function in a single "
        "Python code block.\n\n```python\n" + row["prompt"] + "```"
    )
    return text, None, {"task_id": row["task_id"], "entry_point": row["entry_point"]}


def format_mbpp(row: dict[str, Any]) -> FormatResult:
    """Format an MBPP row with its assert-style tests."""
    tests = "\n".join(row["test_list"])
    text = (
        f"{row['text']}\nYour code must pass these tests:\n{tests}\n"
        "Reply with the code in a single Python code block."
    )
    return text, None, {"task_id": row["task_id"]}


def format_mmlu(row: dict[str, Any]) -> FormatResult:
    """Format an MMLU row; ``answer`` is the index of the correct choice."""
    text = _mcq_text(row["question"], list(row["choices"]))
    return text, LETTERS[int(row["answer"])], {"subject": row["subject"]}


def format_arc(row: dict[str, Any]) -> FormatResult:
    """Format an ARC row, re-lettering choices whose labels are digits.

    Raises
    ------
    ValueError
        If the answer key is not among the choice labels.
    """
    labels = list(row["choices"]["label"])
    if row["answerKey"] not in labels:
        raise ValueError(f"ARC {row.get('id')}: answerKey {row['answerKey']!r} not in {labels}")
    text = _mcq_text(row["question"], list(row["choices"]["text"]))
    return text, LETTERS[labels.index(row["answerKey"])], {"id": row["id"]}


FORMATTERS = {
    "gsm8k": format_gsm8k,
    "math500": format_math500,
    "humaneval": format_humaneval,
    "mbpp": format_mbpp,
    "mmlu": format_mmlu,
    "arc": format_arc,
}


def stratified_sample(n_rows: int, n: int, keys: list[str] | None, rng: random.Random) -> list[int]:
    """Draw ``n`` row indices round-robin across strata visited in shuffled order.

    Raises
    ------
    ValueError
        If ``n`` exceeds ``n_rows`` or ``keys`` has the wrong length.
    """
    if n > n_rows:
        raise ValueError(f"cannot sample {n} of {n_rows} rows")
    if keys is not None and len(keys) != n_rows:
        raise ValueError(f"{len(keys)} strata keys for {n_rows} rows")
    groups: dict[str, list[int]] = {}
    for i in range(n_rows):
        groups.setdefault("all" if keys is None else keys[i], []).append(i)
    order = sorted(groups)
    rng.shuffle(order)
    for key in order:
        rng.shuffle(groups[key])
    chosen: list[int] = []
    depth = 0
    while len(chosen) < n:
        for key in order:
            if depth < len(groups[key]) and len(chosen) < n:
                chosen.append(groups[key][depth])
        depth += 1
    return sorted(chosen)


def load_tokenizer(tokenizer_id: str) -> Tokenizer:
    """Load the Qwen3 tokenizer, shared by every size in the family."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_id)


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    """Count tokens without special tokens; empty text counts zero."""
    return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0


def build_prompt_set(
    cfg: dict[str, Any], tokenizer: Tokenizer
) -> tuple[list[PromptItem], dict[str, Any]]:
    """Download the sources and draw the stratified, seeded prompt set.

    Returns
    -------
    items : list of PromptItem
        The prompt set, with over-length prompts dropped and logged.
    sources : dict
        Per-source dataset identity (id, config, split, row count, fingerprint).
    """
    from datasets import load_dataset

    rng = random.Random(cfg["seed"])
    items: list[PromptItem] = []
    sources: dict[str, Any] = {}
    for name, spec in cfg["sources"].items():
        formatter = FORMATTERS[spec["formatter"]]
        rows = load_dataset(spec["hf_id"], spec.get("hf_config"), split=spec["split"])
        strat = spec.get("stratify_by")
        keys = [str(value) for value in rows[strat]] if strat else None
        sources[name] = {
            "hf_id": spec["hf_id"],
            "hf_config": spec.get("hf_config"),
            "split": spec["split"],
            "n_rows": len(rows),
            "fingerprint": getattr(rows, "_fingerprint", None),
        }
        LOGGER.info("source %s: %d rows from %s", name, len(rows), spec["hf_id"])
        for idx in stratified_sample(len(rows), spec["n"], keys, rng):
            text, reference, meta = formatter(rows[idx])
            n_tok = count_tokens(tokenizer, text)
            if n_tok > cfg["max_prompt_tokens"]:
                LOGGER.warning("dropping %s-%d: %d prompt tokens", name, idx, n_tok)
                continue
            meta = {**meta, "prompt_tokens": n_tok}
            items.append(PromptItem(f"{name}-{idx}", spec["family"], name, text, reference, meta))
    return items, sources


def write_prompts(cfg: dict[str, Any], items: list[PromptItem], sources: dict[str, Any]) -> None:
    """Write full prompts (gitignored) plus a text-free manifest and source record."""
    prompts_path = repo_path(cfg, "prompts")
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    with prompts_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    manifest_path = repo_path(cfg, "prompt_manifest")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["prompt_id", "family", "source", "prompt_tokens", "has_reference"])
        for item in items:
            writer.writerow(
                [
                    item.prompt_id,
                    item.family,
                    item.source,
                    item.meta["prompt_tokens"],
                    item.reference is not None,
                ]
            )
    repo_path(cfg, "prompt_sources").write_text(json.dumps(sources, indent=2), encoding="utf-8")


def read_prompts(path: Path) -> list[PromptItem]:
    """Read a prompt set written by :func:`write_prompts`."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [PromptItem(**json.loads(line)) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Grading (approximate; the labelling pipeline uses the full graders)
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ANSWER_LINE = re.compile(r"(?i:answer)\s*[:\-]?\s*\(?([A-H])\)?(?![A-Za-z])")
_LATEX_NOISE = (
    "\\left",
    "\\right",
    "\\!",
    "\\,",
    "\\;",
    "\\:",
    "^\\circ",
    "^{\\circ}",
    "\\%",
    "%",
    "$",
)


def extract_boxed(text: str) -> str | None:
    r"""Return the contents of the last ``\boxed{...}``, honouring nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 1
    chars: list[str] = []
    for ch in text[start + len("\\boxed{") :]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(ch)
    return None


def normalise_math(answer: str) -> str:
    """Normalise a math answer for string comparison."""
    s = answer.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    for token in _LATEX_NOISE:
        s = s.replace(token, "")
    s = s.replace(" ", "").rstrip(".")
    if re.fullmatch(r"-?[\d,]+(?:\.\d+)?", s):
        s = s.replace(",", "")
    return s


def math_equal(predicted: str | None, gold: str) -> bool:
    """Compare math answers after normalisation, numerically when both parse."""
    if predicted is None:
        return False
    p, g = normalise_math(predicted), normalise_math(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) <= 1e-6 * max(1.0, abs(float(g)))
    except ValueError:
        return False


def extract_math_answer(content: str) -> str | None:
    """Prefer the last boxed answer; fall back to the last number in the text."""
    boxed = extract_boxed(content)
    if boxed is not None:
        return boxed
    numbers = _NUMBER.findall(content)
    return numbers[-1] if numbers else None


def extract_choice(content: str) -> str | None:
    """Return the capital option letter from the last ``Answer: X`` in a response."""
    matches = _ANSWER_LINE.findall(content)
    return matches[-1] if matches else None


def grade(item: PromptItem, content: str) -> bool | None:
    """Grade math and MCQ answers approximately; code is left ungraded (None).

    Raises
    ------
    ValueError
        If a referenced item belongs to a family without a grader.
    """
    if item.reference is None:
        return None
    if item.family == "math":
        return math_equal(extract_math_answer(content), item.reference)
    if item.family == "mcq":
        return extract_choice(content) == item.reference
    raise ValueError(f"no grader for family '{item.family}'")


def compile_markers(phrases: list[str]) -> re.Pattern[str]:
    """Compile reasoning-marker phrases, anchoring word boundaries only at word ends."""
    parts = []
    for phrase in phrases:
        prefix = r"\b" if phrase[0].isalnum() else ""
        suffix = r"\b" if phrase[-1].isalnum() else ""
        parts.append(f"{prefix}{re.escape(phrase)}{suffix}")
    return re.compile("|".join(parts), flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Model verification and GPU guard
# ---------------------------------------------------------------------------


def verify_models(
    cfg: dict[str, Any], client: ollama.Client
) -> tuple[list[dict[str, Any]], list[str]]:
    """Check every configured tag is a hybrid Qwen3 model, from metadata alone.

    A tag that resolves to a 2507 release still loads and answers, but it either
    cannot think or cannot stop thinking, which would silently break the grid.
    No weights are loaded.

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
                "family": getattr(details, "family", None),
                "context_length": context,
                "capabilities": capabilities,
                "template_mentions_think": "think" in (getattr(info, "template", "") or "").lower(),
            }
        )
        if "thinking" not in capabilities:
            problems.append(f"{tag}: no 'thinking' capability (has {capabilities})")
        if context != spec["expected_context_length"]:
            problems.append(
                f"{tag}: context_length {context} != expected "
                f"{spec['expected_context_length']} (possibly a 2507 release)"
            )
    return manifest, problems


def check_gpu_ready(cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, float]:
    """Refuse to start a model sweep on a GPU that is busy or short of memory.

    Raises
    ------
    RuntimeError
        If mean utilisation exceeds the configured ceiling, or free VRAM is below
        the model's requirement. Rerunning resumes from recorded generations.
    """
    import pynvml

    guard = cfg["gpu_guard"]
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        readings: list[int] = []
        deadline = time.perf_counter() + guard["util_window_s"]
        while time.perf_counter() < deadline:
            readings.append(int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu))
            time.sleep(0.25)
        free_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).free / 2**20
    finally:
        pynvml.nvmlShutdown()
    util = float(np.mean(readings))
    if util > guard["max_util_pct"]:
        raise RuntimeError(
            f"GPU busy before {spec['tag']}: {util:.0f}% utilisation; rerun later to resume"
        )
    if free_mb < spec["min_free_vram_mb"]:
        raise RuntimeError(
            f"only {free_mb:.0f} MB VRAM free before {spec['tag']}; "
            f"need {spec['min_free_vram_mb']} MB"
        )
    return {"util_pct": util, "free_vram_mb": free_mb}


def gpu_placement(client: ollama.Client, tag: str) -> float | None:
    """Return the fraction of a loaded model's memory held in VRAM."""
    for model in client.ps().models:
        if model.model == tag and model.size:
            return model.size_vram / model.size
    return None


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def cell_seed(base: int, prompt_id: str, tag: str, mode: str) -> int:
    """Derive a deterministic, decorrelated seed for one generation."""
    return zlib.crc32(f"{base}|{prompt_id}|{tag}|{mode}".encode()) & 0x7FFFFFFF


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read JSONL records, tolerating a truncated final line from an interrupted run.

    Returns
    -------
    records : list of dict
    truncated : bool
        True if a malformed final line was skipped.

    Raises
    ------
    json.JSONDecodeError
        If a line other than the last is malformed.
    """
    if not path.exists():
        return [], False
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if number == len(lines):
                LOGGER.warning("skipping truncated final line %d of %s", number, path)
                return records, True
            raise
    return records, False


def generate_one(
    client: ollama.Client,
    cfg: dict[str, Any],
    spec: dict[str, Any],
    item: PromptItem,
    mode: str,
    tokenizer: Tokenizer,
    markers: re.Pattern[str],
) -> dict[str, Any]:
    """Generate one response and derive every per-record metric."""
    seed = cell_seed(cfg["seed"], item.prompt_id, spec["tag"], mode)
    options = {**cfg["decoding"][mode], "seed": seed, "num_ctx": cfg["num_ctx"]}
    started = time.perf_counter()
    response = client.chat(
        model=spec["tag"],
        messages=[{"role": "user", "content": item.text}],
        think=mode == "think",
        options=options,
        keep_alive=cfg["keep_alive"],
    )
    wall_s = time.perf_counter() - started
    content = response.message.content or ""
    thinking = getattr(response.message, "thinking", None) or ""
    ns = 1e-9
    return {
        "prompt_id": item.prompt_id,
        "family": item.family,
        "source": item.source,
        "size": spec["size"],
        "tag": spec["tag"],
        "mode": mode,
        "seed": seed,
        "content": content,
        "thinking": thinking,
        "done_reason": response.done_reason,
        "eval_count": response.eval_count or 0,
        "prompt_eval_count": response.prompt_eval_count or 0,
        "eval_duration_s": (response.eval_duration or 0) * ns,
        "prompt_eval_duration_s": (response.prompt_eval_duration or 0) * ns,
        "load_duration_s": (response.load_duration or 0) * ns,
        "total_duration_s": (response.total_duration or 0) * ns,
        "wall_s": wall_s,
        "tokens_thinking": count_tokens(tokenizer, thinking),
        "tokens_content": count_tokens(tokenizer, content),
        "think_tag_in_content": "<think>" in content or "</think>" in content,
        "markers_thinking": len(markers.findall(thinking)),
        "markers_content": len(markers.findall(content)),
        "correct": grade(item, content),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def collect(
    cfg: dict[str, Any],
    client: ollama.Client,
    items: list[PromptItem],
    tokenizer: Tokenizer,
    markers: re.Pattern[str],
) -> None:
    """Run every (prompt, size, mode) not yet recorded, model-major, resumable.

    Raises
    ------
    RuntimeError
        If the GPU guard refuses a sweep or generation errors exceed ``max_errors``.
    """
    import httpx
    import ollama as ollama_module

    raw_path = repo_path(cfg, "raw_generations")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    records, truncated = read_jsonl(raw_path)
    if truncated:
        raw_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
    done = {(r["prompt_id"], r["tag"], r["mode"]) for r in records}
    total = len(items) * len(MODES) * len(cfg["models"])
    LOGGER.info("%d of %d generations already recorded", len(done), total)
    retryable = (ollama_module.ResponseError, httpx.HTTPError, ConnectionError)
    errors = 0

    for spec in cfg["models"]:
        tag = spec["tag"]
        remaining = sum((item.prompt_id, tag, m) not in done for item in items for m in MODES)
        if remaining == 0:
            LOGGER.info("%s: already complete", tag)
            continue
        state = check_gpu_ready(cfg, spec)
        LOGGER.info(
            "%s: %d to run | GPU %.0f%% util, %.0f MB free | loading",
            tag,
            remaining,
            state["util_pct"],
            state["free_vram_mb"],
        )
        client.chat(  # warm-up load, not recorded
            model=tag,
            messages=[{"role": "user", "content": "Reply with OK."}],
            think=False,
            options={"num_predict": 8, "num_ctx": cfg["num_ctx"], "seed": cfg["seed"]},
            keep_alive=cfg["keep_alive"],
        )
        placement = gpu_placement(client, tag)
        if placement is not None and placement < 0.999:
            LOGGER.warning(
                "%s: only %.0f%% of the model is in VRAM; latency includes CPU offload",
                tag,
                placement * 100,
            )
        sweep_start = time.perf_counter()
        finished = 0
        with raw_path.open("a", encoding="utf-8") as fh:
            for index, item in enumerate(items):
                order = MODES if index % 2 == 0 else MODES[::-1]
                for mode in order:
                    key = (item.prompt_id, tag, mode)
                    if key in done:
                        continue
                    try:
                        record = generate_one(client, cfg, spec, item, mode, tokenizer, markers)
                    except retryable as exc:
                        errors += 1
                        LOGGER.error("%s %s %s failed: %s", tag, item.prompt_id, mode, exc)
                        if errors > cfg["max_errors"]:
                            raise RuntimeError(f"aborting after {errors} errors") from exc
                        continue
                    record["gpu_placement"] = placement
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    done.add(key)
                    finished += 1
                    if finished % 10 == 0 or finished == remaining:
                        minutes = (time.perf_counter() - sweep_start) / 60
                        LOGGER.info(
                            "%s: %d/%d in %.1f min (%.1f per min)",
                            tag,
                            finished,
                            remaining,
                            minutes,
                            finished / minutes if minutes > 0 else 0.0,
                        )
        client.generate(model=tag, prompt="", keep_alive=0)
        LOGGER.info("%s: unloaded after %.1f min", tag, (time.perf_counter() - sweep_start) / 60)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _accuracy(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for family in sorted({r["family"] for r in records if r["correct"] is not None}):
        graded = [
            r["correct"] for r in records if r["family"] == family and r["correct"] is not None
        ]
        out[family] = {"n": len(graded), "accuracy": float(np.mean(graded))}
    return out


def _quartiles(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 25)), float(np.percentile(values, 75))]


def analyse(cfg: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-size separation statistics and the pre-registered verdict."""
    from scipy import stats

    thresholds = cfg["thresholds"]
    rng = np.random.default_rng(cfg["seed"])
    sizes: dict[str, Any] = {}
    for spec in cfg["models"]:
        tag = spec["tag"]
        pairs: dict[str, dict[str, dict[str, Any]]] = {}
        for record in records:
            if record["tag"] == tag:
                pairs.setdefault(record["prompt_id"], {})[record["mode"]] = record
        complete = sorted(pid for pid, modes in pairs.items() if all(m in modes for m in MODES))
        if len(complete) < 2:
            sizes[spec["size"]] = {
                "tag": tag,
                "n_pairs": len(complete),
                "status": "insufficient_data",
            }
            continue

        think = [pairs[pid]["think"] for pid in complete]
        nothink = [pairs[pid]["no_think"] for pid in complete]
        t_tok = np.array([r["eval_count"] for r in think], dtype=np.float64)
        n_tok = np.array([r["eval_count"] for r in nothink], dtype=np.float64)

        u_stat = stats.mannwhitneyu(t_tok, n_tok, alternative="two-sided").statistic
        auc = float(u_stat / (t_tok.size * n_tok.size))
        diff = t_tok - n_tok
        dominance = float(np.mean(np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))))
        wilcoxon_p = float(stats.wilcoxon(t_tok, n_tok).pvalue) if np.any(diff != 0) else None
        ratio = (t_tok + 1.0) / (n_tok + 1.0)
        boot = np.array(
            [
                np.median(ratio[rng.integers(0, ratio.size, ratio.size)])
                for _ in range(cfg["bootstrap_resamples"])
            ]
        )

        nonempty_thinking = [bool(r["thinking"].strip()) for r in nothink]
        tag_leak = [bool(r["think_tag_in_content"]) for r in nothink]
        leak_rate = float(
            np.mean([a or b for a, b in zip(nonempty_thinking, tag_leak, strict=True)])
        )
        think_tokens = sum(r["tokens_thinking"] for r in think)
        nothink_tokens = sum(r["tokens_content"] for r in nothink)
        think_density = 100.0 * sum(r["markers_thinking"] for r in think) / max(1, think_tokens)
        nothink_density = (
            100.0 * sum(r["markers_content"] for r in nothink) / max(1, nothink_tokens)
        )
        density_ratio = nothink_density / think_density if think_density > 0 else None

        checks = {
            "auc": auc >= thresholds["auc_min"],
            "leak_rate": leak_rate <= thresholds["max_nothink_leak_rate"],
            "marker_density_ratio": density_ratio is not None
            and density_ratio <= thresholds["max_marker_density_ratio"],
        }
        if all(checks.values()):
            status = "separated"
        elif auc >= thresholds["auc_partial_min"]:
            status = "partial"
        else:
            status = "not_separated"

        by_mode = {"think": think, "no_think": nothink}
        families = sorted({r["family"] for r in think})
        sizes[spec["size"]] = {
            "tag": tag,
            "n_pairs": len(complete),
            "tokens_median": {
                "think": float(np.median(t_tok)),
                "no_think": float(np.median(n_tok)),
            },
            "tokens_iqr": {"think": _quartiles(t_tok), "no_think": _quartiles(n_tok)},
            "tokens_median_by_family": {
                fam: {
                    mode: float(np.median([r["eval_count"] for r in recs if r["family"] == fam]))
                    for mode, recs in by_mode.items()
                }
                for fam in families
            },
            "auc_think_longer": auc,
            "paired_dominance": dominance,
            "wilcoxon_p": wilcoxon_p,
            "median_token_ratio": float(np.median(ratio)),
            "median_token_ratio_ci95": [
                float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5)),
            ],
            "nothink_leak_rate": leak_rate,
            "nothink_nonempty_thinking_rate": float(np.mean(nonempty_thinking)),
            "nothink_think_tag_rate": float(np.mean(tag_leak)),
            "marker_density_per_100_tokens": {
                "think_trace": think_density,
                "no_think_answer": nothink_density,
            },
            "marker_density_ratio": density_ratio,
            "cap_hit_rate": {
                mode: float(np.mean([r["done_reason"] == "length" for r in recs]))
                for mode, recs in by_mode.items()
            },
            "throughput_tok_per_s": {
                mode: sum(r["eval_count"] for r in recs)
                / max(1e-9, sum(r["eval_duration_s"] for r in recs))
                for mode, recs in by_mode.items()
            },
            "accuracy": {mode: _accuracy(recs) for mode, recs in by_mode.items()},
            "checks": checks,
            "status": status,
        }

    return {
        "gate": "gate2_mode_separation",
        "analysed_at_utc": datetime.now(UTC).isoformat(),
        "thresholds": thresholds,
        "reasoning_markers": cfg["reasoning_markers"],
        "decoding": cfg["decoding"],
        "sizes": sizes,
        "passed": len(sizes) == len(cfg["models"])
        and all(s["status"] == "separated" for s in sizes.values()),
    }


RECORD_COLUMNS = (
    "prompt_id",
    "family",
    "source",
    "size",
    "tag",
    "mode",
    "seed",
    "done_reason",
    "eval_count",
    "prompt_eval_count",
    "eval_duration_s",
    "prompt_eval_duration_s",
    "load_duration_s",
    "total_duration_s",
    "wall_s",
    "tokens_thinking",
    "tokens_content",
    "think_tag_in_content",
    "markers_thinking",
    "markers_content",
    "correct",
    "gpu_placement",
    "timestamp_utc",
)


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write per-record metrics without response text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RECORD_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "gate2_mode_separation.yaml"
    )
    parser.add_argument(
        "--rebuild-prompts", action="store_true", help="redraw prompts even if a set exists"
    )
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true", help="datasets only; no GPU")
    stage.add_argument("--verify-models", action="store_true", help="metadata only; no GPU")
    stage.add_argument("--analyse-only", action="store_true", help="recompute from records")
    return parser.parse_args()


def main() -> None:
    """Run the requested stage of gate 2."""
    args = parse_args()
    cfg = load_config(args.config)
    log_dir = REPO_ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "gate2.log")],
    )

    tokenizer = load_tokenizer(cfg["tokenizer_id"])
    prompts_path = repo_path(cfg, "prompts")
    if prompts_path.exists() and not args.rebuild_prompts:
        items = read_prompts(prompts_path)
    else:
        items, sources = build_prompt_set(cfg, tokenizer)
        write_prompts(cfg, items, sources)
    counts = {
        fam: sum(item.family == fam for item in items) for fam in sorted({i.family for i in items})
    }
    LOGGER.info("%d prompts: %s", len(items), counts)
    if args.prepare_only:
        return

    import ollama

    client = ollama.Client(host=cfg["ollama_host"], timeout=cfg["request_timeout_s"])
    manifest, problems = verify_models(cfg, client)
    manifest_path = repo_path(cfg, "model_manifest")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if problems:
        raise RuntimeError("model verification failed:\n  " + "\n  ".join(problems))
    LOGGER.info("all %d models verified as hybrid-thinking Qwen3", len(manifest))
    if args.verify_models:
        return

    markers = compile_markers(cfg["reasoning_markers"])
    if not args.analyse_only:
        collect(cfg, client, items, tokenizer, markers)

    records, _ = read_jsonl(repo_path(cfg, "raw_generations"))
    summary = analyse(cfg, records)
    summary.update(git_commit=git_commit(), models=manifest, n_prompts=len(items))
    repo_path(cfg, "summary").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_records_csv(repo_path(cfg, "records"), records)
    for size, result in summary["sizes"].items():
        LOGGER.info("%s: %s | checks %s", size, result["status"], result.get("checks"))
    LOGGER.info("gate 2 passed: %s", summary["passed"])


if __name__ == "__main__":
    main()
