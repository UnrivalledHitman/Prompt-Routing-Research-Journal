"""Corpus construction for oracle labelling (experimental protocol, section 4).

Each source is walked in a seeded, stratified round-robin order, and rows are taken
in that order until the source's quota is filled. A row is skipped if gate 2 already
used it, if its formatted prompt duplicates an earlier prompt after normalisation,
or if it exceeds the prompt-token limit. The first ``pilot`` rows taken form the
pilot subset and the next ``main`` rows the rest of the corpus, so the pilot is
nested in the full corpus and stratified by construction. The first ``noise`` pilot
rows of each source are also sampled repeatedly to estimate label noise.

Every source draws from its own random stream, seeded from the corpus seed and the
source name, so enabling a source later does not change what the others select.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

LOGGER = logging.getLogger("arbiter.corpus")
LETTERS = "ABCDEFGH"
SUBSETS = ("pilot", "main")
REQUIRED_KEYS = ("seed", "tokenizer_id", "max_prompt_tokens", "paths", "sources")
MCQ_INSTRUCTION = (
    "Respond with the letter of the correct option on the last line, formatted as 'Answer: X'."
)
FormatResult = tuple[str, str | None, dict[str, Any]]


class Tokenizer(Protocol):
    """The subset of a Hugging Face tokenizer used here."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        """Encode text to token ids."""
        ...


@dataclass(frozen=True)
class PromptItem:
    """A formatted prompt with everything needed to grade answers to it.

    Attributes
    ----------
    prompt_id : str
        ``<source>-<row index>``, stable for a given dataset revision and comparable
        with gate 2 identifiers.
    family : str
        ``math``, ``code``, ``mcq`` or ``open``.
    source : str
        Source key from the corpus config.
    text : str
        The user message sent to every cell.
    reference : str or None
        Gold answer (math, MCQ), reference response (open-ended), or None (code).
    subset : str
        ``pilot`` or ``main``.
    noise : bool
        True if the prompt receives extra samples for the label-noise estimate.
    meta : dict
        Grading fields (unit tests, entry point) and traceability fields.
    """

    prompt_id: str
    family: str
    source: str
    text: str
    reference: str | None
    subset: str
    noise: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Source formatters. Prompt wording matches gate 2 so the two runs are comparable.
# ---------------------------------------------------------------------------


def _mcq_text(question: str, options: Sequence[str]) -> str:
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
    meta = {
        "level": row.get("level"),
        "subject": row.get("subject"),
        "unique_id": row.get("unique_id"),
    }
    return text, str(row["answer"]), meta


def format_humaneval(row: dict[str, Any]) -> FormatResult:
    """Format a HumanEval row as a function-completion request, keeping its tests."""
    text = (
        "Complete the following Python function. Reply with the full function in a single "
        "Python code block.\n\n```python\n" + row["prompt"] + "```"
    )
    meta = {
        "task_id": row["task_id"],
        "entry_point": row["entry_point"],
        "prompt": row["prompt"],
        "test": row["test"],
    }
    return text, None, meta


def format_mbpp(row: dict[str, Any]) -> FormatResult:
    """Format an MBPP row with its assert-style tests."""
    tests = "\n".join(row["test_list"])
    text = (
        f"{row['text']}\nYour code must pass these tests:\n{tests}\n"
        "Reply with the code in a single Python code block."
    )
    meta = {
        "task_id": row["task_id"],
        "test_list": list(row["test_list"]),
        "test_setup_code": row.get("test_setup_code") or "",
    }
    return text, None, meta


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


def format_dolly(row: dict[str, Any]) -> FormatResult:
    """Format a Dolly row; the human-written response is the judge's reference."""
    text = row["instruction"].strip()
    context = (row.get("context") or "").strip()
    if context:
        text = f"{text}\n\n{context}"
    return text, row["response"].strip(), {"category": row["category"]}


FORMATTERS: dict[str, Callable[[dict[str, Any]], FormatResult]] = {
    "gsm8k": format_gsm8k,
    "math500": format_math500,
    "humaneval": format_humaneval,
    "mbpp": format_mbpp,
    "mmlu": format_mmlu,
    "arc": format_arc,
    "dolly": format_dolly,
}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def stratified_order(keys: Sequence[str], rng: random.Random) -> list[int]:
    """Order row indices round-robin across strata visited in shuffled order.

    Every stratum contributes one row per round until it runs out, so any prefix
    of the order is balanced across the strata that still have rows.
    """
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    order = sorted(groups)
    rng.shuffle(order)
    for key in order:
        rng.shuffle(groups[key])
    out: list[int] = []
    depth = 0
    while len(out) < len(keys):
        for key in order:
            if depth < len(groups[key]):
                out.append(groups[key][depth])
        depth += 1
    return out


def strata_keys(rows: Any, spec: dict[str, Any]) -> list[str]:  # noqa: ANN401
    """Return every row's stratum, integer-binned when ``stratify_bin`` is set."""
    column = spec.get("stratify_by")
    if column is None:
        return ["all"] * len(rows)
    values = rows[column] if hasattr(rows, "column_names") else [row[column] for row in rows]
    width = spec.get("stratify_bin")
    return [str(int(value) // int(width)) if width else str(value) for value in values]


def text_digest(text: str) -> str:
    """Hash text after lower-casing and collapsing whitespace."""
    normalised = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    """Count tokens without special tokens; empty text counts zero."""
    return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0


def select_source(
    name: str,
    spec: dict[str, Any],
    rows: Any,  # noqa: ANN401
    tokenizer: Tokenizer,
    rng: random.Random,
    excluded_ids: set[str],
    seen_digests: set[str],
    max_prompt_tokens: int,
) -> tuple[list[PromptItem], dict[str, int]]:
    """Fill one source's pilot and main quotas in stratified order.

    ``seen_digests`` is updated in place so duplicates are caught across sources.

    Returns
    -------
    items : list of PromptItem
        Pilot items first, then main, in selection order.
    stats : dict
        Rows taken and rows skipped by reason.

    Raises
    ------
    ValueError
        If ``noise`` exceeds ``pilot``, or the source runs out of eligible rows
        before its quota is filled.
    """
    n_pilot, n_main = int(spec["pilot"]), int(spec["main"])
    n_noise = int(spec.get("noise", 0))
    if n_noise > n_pilot:
        raise ValueError(f"{name}: noise ({n_noise}) exceeds pilot ({n_pilot})")
    formatter = FORMATTERS[spec["formatter"]]
    keys = strata_keys(rows, spec)
    quota = n_pilot + n_main
    skipped: Counter[str] = Counter()
    items: list[PromptItem] = []
    for index in stratified_order(keys, rng):
        if len(items) == quota:
            break
        prompt_id = f"{name}-{index}"
        if prompt_id in excluded_ids:
            skipped["excluded"] += 1
            continue
        text, reference, meta = formatter(rows[index])
        digest = text_digest(text)
        if digest in seen_digests:
            skipped["duplicate"] += 1
            continue
        n_tokens = count_tokens(tokenizer, text)
        if n_tokens > max_prompt_tokens:
            skipped["too_long"] += 1
            continue
        seen_digests.add(digest)
        rank = len(items)
        items.append(
            PromptItem(
                prompt_id=prompt_id,
                family=spec["family"],
                source=name,
                text=text,
                reference=reference,
                subset="pilot" if rank < n_pilot else "main",
                noise=rank < n_noise,
                meta={**meta, "prompt_tokens": n_tokens, "stratum": keys[index], "rank": rank},
            )
        )
    if len(items) < quota:
        raise ValueError(
            f"{name}: only {len(items)} eligible rows for a quota of {quota} "
            f"(skipped {dict(skipped)})"
        )
    return items, {"taken": len(items), **skipped}


# ---------------------------------------------------------------------------
# Build, write, read
# ---------------------------------------------------------------------------


def load_corpus_config(path: Path) -> dict[str, Any]:
    """Load and validate the corpus config.

    Raises
    ------
    ValueError
        If a required key is missing or a source names an unknown formatter.
    """
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    for name, spec in cfg["sources"].items():
        if spec["formatter"] not in FORMATTERS:
            raise ValueError(f"{path}: source {name} has unknown formatter {spec['formatter']}")
    return cfg


def read_excluded_ids(paths: Sequence[Path]) -> set[str]:
    """Collect ``prompt_id`` values from earlier prompt manifests."""
    excluded: set[str] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as fh:
            excluded.update(row["prompt_id"] for row in csv.DictReader(fh))
    return excluded


def build_corpus(
    cfg: dict[str, Any], repo_root: Path, tokenizer: Tokenizer
) -> tuple[list[PromptItem], dict[str, Any]]:
    """Select every enabled source from cached datasets.

    Returns
    -------
    items : list of PromptItem
        All pilot items, then all main items, each in config source order.
    sources : dict
        Per-source dataset identity, licence and selection statistics.
    """
    from datasets import load_dataset

    excluded = read_excluded_ids([repo_root / p for p in cfg.get("exclude_manifests", [])])
    seen: set[str] = set()
    items: list[PromptItem] = []
    sources: dict[str, Any] = {}
    for name, spec in cfg["sources"].items():
        if not spec.get("enabled", True):
            LOGGER.info("source %s disabled", name)
            continue
        rows = load_dataset(spec["hf_id"], spec.get("hf_config"), split=spec["split"])
        rng = random.Random(f"{cfg['seed']}|{name}")
        chosen, stats = select_source(
            name, spec, rows, tokenizer, rng, excluded, seen, int(cfg["max_prompt_tokens"])
        )
        items.extend(chosen)
        sources[name] = {
            "hf_id": spec["hf_id"],
            "hf_config": spec.get("hf_config"),
            "split": spec["split"],
            "licence": spec.get("licence"),
            "n_rows": len(rows),
            "fingerprint": getattr(rows, "_fingerprint", None),
            **stats,
        }
        LOGGER.info("source %s: %s", name, stats)
    items.sort(key=lambda item: SUBSETS.index(item.subset))
    return items, sources


def write_corpus(
    cfg: dict[str, Any], repo_root: Path, items: Sequence[PromptItem], sources: dict[str, Any]
) -> None:
    """Write full prompts (gitignored) plus a text-free manifest and source record."""
    paths = {key: repo_root / value for key, value in cfg["paths"].items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    with paths["corpus"].open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    with paths["manifest"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["prompt_id", "family", "source", "subset", "noise", "prompt_tokens", "stratum"]
        )
        for item in items:
            writer.writerow(
                [
                    item.prompt_id,
                    item.family,
                    item.source,
                    item.subset,
                    item.noise,
                    item.meta["prompt_tokens"],
                    item.meta["stratum"],
                ]
            )
    paths["sources"].write_text(json.dumps(sources, indent=2), encoding="utf-8")


def read_corpus(path: Path) -> list[PromptItem]:
    """Read a corpus written by :func:`write_corpus`."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [PromptItem(**json.loads(line)) for line in lines if line.strip()]
