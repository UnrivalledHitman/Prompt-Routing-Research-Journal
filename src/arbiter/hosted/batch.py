"""Build, submit and collect Batch API jobs for the hosted-frontier arm.

The request sent to each hosted cell mirrors the local one as closely as the two APIs
allow: a single user message carrying exactly the prompt text the local pool saw, no
system message, and an output cap that counts reasoning tokens the way the local cap
counts the think trace. A truncated response is a failure, as a local cap hit is.

Everything paid passes the meter in :mod:`arbiter.hosted.pricing` first. Nothing here
prints the API key, and the key is never written to a record, a log or the ledger.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol

from arbiter.hosted.pricing import Meter, Price, cost_of, usage_tokens, worst_case

if TYPE_CHECKING:  # pragma: no cover - import cost only
    from arbiter.data.corpus import PromptItem


# The SDK's own types are not imported, so that the tests can drive this module with a stub
# and so that importing it costs nothing. These describe only the surface actually used.
class Handle(Protocol):
    """An uploaded file or a batch job, identified by its id."""

    id: str


class Job(Protocol):
    """A batch job as the provider reports it."""

    id: str
    status: str
    output_file_id: str | None
    error_file_id: str | None
    request_counts: Any


class Body(Protocol):
    """A downloaded file, whose ``text`` is its whole body."""

    text: str


class Files(Protocol):
    """The file endpoints: upload a batch input, download a batch output."""

    def create(self, *, file: BinaryIO, purpose: str) -> Handle:
        """Upload a file and return its handle."""

    def content(self, file_id: str) -> Body:
        """Return the file body."""


class Batches(Protocol):
    """The batch endpoints."""

    def create(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: Mapping[str, str],
    ) -> Job:
        """Create a batch job over an uploaded input file."""

    def retrieve(self, batch_id: str) -> Job:
        """Return a batch job's current state."""


class Client(Protocol):
    """The part of the OpenAI client this module touches."""

    files: Files
    batches: Batches


CUSTOM_ID_MAX = 64
SEPARATOR = "|"
# The corpus prompt-token counts come from the Qwen tokenizer. A hosted tokenizer counts
# the same text differently, so the worst-case bound inflates them rather than trusting them.
INPUT_SLACK = 1.5
TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})


def dotenv_paths(root: Path) -> tuple[Path, ...]:
    """Return the places a dotenv file may sit, the working tree first.

    A git worktree has no dotenv file of its own: the credential stays in the main
    checkout, which is reached through the ``gitdir:`` pointer in the worktree's ``.git``
    file, so it is read where it lives rather than copied per branch.
    """
    paths = [root / ".env"]
    marker = root / ".git"
    if marker.is_file():
        pointer = marker.read_text(encoding="utf-8").partition("gitdir:")[2].strip()
        if pointer:
            # <main>/.git/worktrees/<name> -> <main>
            common = Path(pointer).resolve()
            for parent in common.parents:
                if parent.name == ".git":
                    paths.append(parent.parent / ".env")
                    break
    return tuple(paths)


def api_key(*env_paths: Path, name: str = "OPENAI_API_KEY") -> str:
    """Return the API key from the process environment or a dotenv file, without logging it.

    Raises
    ------
    RuntimeError
        If the key is absent, or present but obviously malformed, which is worth catching
        before a batch is submitted rather than after.
    """
    key = os.environ.get(name, "")
    for env_path in env_paths:
        if key or not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            field, separator, value = line.partition("=")
            if separator and field.strip() == name:
                key = value.strip().strip("\"'")
                break
    if not key:
        listed = ", ".join(str(p) for p in env_paths) or "(no path given)"
        raise RuntimeError(f"{name} is not set and was not found in {listed}")
    if not key.startswith("sk-") or len(key) < 40 or any(c.isspace() for c in key):
        raise RuntimeError(
            f"{name} does not look like an API key (length {len(key)}); "
            "check the value in .env. Its content is deliberately not printed."
        )
    return key


@dataclass(frozen=True)
class Cell:
    """One hosted configuration: a model at a reasoning effort, with its output cap."""

    cell: str
    model: str
    effort: str
    cap: int


def cells_from_config(rows: Iterable[Mapping[str, Any]]) -> tuple[Cell, ...]:
    """Build the cell ladder from the config, preserving its cheapest-first order."""
    return tuple(
        Cell(cell=str(r["cell"]), model=str(r["model"]), effort=str(r["effort"]), cap=int(r["cap"]))
        for r in rows
    )


def custom_id(cell: str, prompt_id: str) -> str:
    """Return the batch line identifier, which must round-trip to cell and prompt."""
    value = f"{cell}{SEPARATOR}{prompt_id}"
    if len(value) > CUSTOM_ID_MAX or SEPARATOR in prompt_id:
        raise ValueError(f"custom id {value!r} is unusable: too long, or the prompt id has a pipe")
    return value


def split_custom_id(value: str) -> tuple[str, str]:
    """Split a batch line identifier back into its cell and prompt id."""
    cell, separator, prompt_id = value.partition(SEPARATOR)
    if not separator:
        raise ValueError(f"malformed custom id {value!r}")
    return cell, prompt_id


def request_body(cell: Cell, text: str) -> dict[str, Any]:
    """Build the Responses API body for one prompt in one cell.

    Sampling is left at the provider's default: the reasoning models do not accept the
    temperature and top-p the local protocol sets per mode, so the arm cannot and does not
    claim to match the local decoding parameters.
    """
    return {
        "model": cell.model,
        "input": [{"role": "user", "content": text}],
        "reasoning": {"effort": cell.effort},
        "max_output_tokens": cell.cap,
        "store": False,
    }


def build_requests(items: Sequence[PromptItem], cells: Sequence[Cell]) -> list[dict[str, Any]]:
    """Build one batch line per prompt and cell, prompt-major so a partial run is balanced."""
    return [
        {
            "custom_id": custom_id(cell.cell, item.prompt_id),
            "method": "POST",
            "url": "/v1/responses",
            "body": request_body(cell, item.text),
        }
        for item in items
        for cell in cells
    ]


def group_by_model(cells: Sequence[Cell]) -> list[tuple[str, tuple[Cell, ...]]]:
    """Split the ladder into one group per model, keeping ladder order.

    The Batch API refuses a job whose requests name more than one model, so a stage runs as
    one batch per model, each carrying that model's two reasoning efforts. Grouping this way
    rather than one batch per cell keeps both efforts of a size inside a single job, so a
    size is never half-run.
    """
    groups: dict[str, list[Cell]] = {}
    for cell in cells:
        groups.setdefault(cell.model, []).append(cell)
    return [(model, tuple(members)) for model, members in groups.items()]


def plan_cost(
    items: Sequence[PromptItem],
    cells: Sequence[Cell],
    prices: Mapping[str, Price],
) -> dict[str, float]:
    """Return the worst-case dollar cost of a batch, and its per-prompt rate.

    The worst case assumes every request emits its full cap. It is what the meter approves,
    because it is the only figure knowable before the batch runs.
    """
    total = 0.0
    for item in items:
        prompt_tokens = int(item.meta.get("prompt_tokens", 0) or 0)
        bound = max(int(prompt_tokens * INPUT_SLACK), 1)
        for cell in cells:
            total += worst_case(bound, cell.cap, prices[cell.model])
    n = max(len(items), 1)
    return {"worst_case_usd": total, "worst_case_per_prompt": total / n}


def submit(
    client: Client,
    requests: Sequence[Mapping[str, Any]],
    *,
    endpoint: str,
    completion_window: str,
    raw_dir: Path,
    label: str,
) -> Job:
    """Upload the batch input and create the job, keeping the input file for the record."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{label}_input.jsonl"
    path.write_text(
        "".join(json.dumps(line) + "\n" for line in requests),
        encoding="utf-8",
    )
    with path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    return client.batches.create(
        input_file_id=uploaded.id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata={"arm": "hosted", "stage": label},
    )


def wait(client: Client, batch_id: str, *, poll_s: float, log: logging.Logger | None = None) -> Job:
    """Poll a batch until it reaches a terminal state, logging progress as it goes."""
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = getattr(batch, "request_counts", None)
        if log is not None:
            done = getattr(counts, "completed", 0) or 0
            failed = getattr(counts, "failed", 0) or 0
            total = getattr(counts, "total", 0) or 0
            log.info("batch %s: %s, %d/%d done, %d failed", batch_id, batch.status, done, total,
                     failed)  # fmt: skip
        if batch.status in TERMINAL:
            return batch
        time.sleep(poll_s)


def log_failure(client: Client, job: Job, log: logging.Logger, limit: int = 5) -> None:
    """Log why a batch was rejected, which the job carries rather than an output file.

    A batch that fails validation produces no output and no error file, so without this the
    only symptom is zero responses. The provider's own message is logged verbatim.
    """
    del client  # the reason travels on the job itself
    errors = getattr(job, "errors", None)
    data = list(getattr(errors, "data", None) or ()) if errors is not None else []
    if not data:
        log.error("batch %s failed with no reason reported", job.id)
        return
    for item in data[:limit]:
        log.error(
            "batch %s rejected: %s | %s | line %s",
            job.id,
            getattr(item, "code", None),
            getattr(item, "message", None),
            getattr(item, "line", None),
        )
    if len(data) > limit:
        log.error("batch %s: %d further errors not shown", job.id, len(data) - limit)


def output_text(body: Mapping[str, Any]) -> str:
    """Return the visible answer of a Responses API body, ignoring reasoning items."""
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    parts: list[str] = []
    for item in body.get("output") or ():
        if item.get("type") != "message":
            continue
        for block in item.get("content") or ():
            if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "".join(parts)


def parse_line(line: Mapping[str, Any], cells: Mapping[str, Cell]) -> dict[str, Any]:
    """Turn one batch output line into a record the local graders can read.

    ``content`` and ``done_reason`` carry the names the local harness uses, so
    :func:`arbiter.eval.grading.grade` and the cap-hit rule apply unchanged.
    """
    cell_name, prompt_id = split_custom_id(str(line["custom_id"]))
    cell = cells[cell_name]
    record: dict[str, Any] = {
        "prompt_id": prompt_id,
        "cell": cell_name,
        "model": cell.model,
        "effort": cell.effort,
        "cap": cell.cap,
        "content": "",
        "thinking": "",  # the provider does not return the trace, only its token count
        "done_reason": "error",
        "status": None,
        "error": None,
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }
    error = line.get("error")
    response = line.get("response") or {}
    body = response.get("body") or {}
    if error or response.get("status_code") != 200:
        detail = error or body.get("error") or {"message": f"status {response.get('status_code')}"}
        record["error"] = str(detail.get("message", detail))[:500]
        return record
    status = body.get("status")
    incomplete = (body.get("incomplete_details") or {}).get("reason")
    record["status"] = status
    record["content"] = output_text(body)
    # A response truncated at the cap is a failure, as a local cap hit is; the grader
    # never sees it as a partial answer.
    record["done_reason"] = "length" if incomplete == "max_output_tokens" else "stop"
    if status == "incomplete" and incomplete != "max_output_tokens":
        record["done_reason"] = "error"
        record["error"] = f"incomplete: {incomplete}"
    record.update(usage_tokens(body.get("usage") or {}))
    return record


def collect(
    client: Client,
    batch: Job,
    cells: Mapping[str, Cell],
    prices: Mapping[str, Price],
    *,
    raw_dir: Path,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Download a finished batch, price every response and return the records and totals."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for name, file_id in (("output", batch.output_file_id), ("error", batch.error_file_id)):
        if not file_id:
            continue
        text = client.files.content(file_id).text
        (raw_dir / f"{label}_{name}.jsonl").write_text(text, encoding="utf-8")
        if name != "output":
            continue
        for line in text.splitlines():
            if line.strip():
                records.append(parse_line(json.loads(line), cells))
    totals = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "responses": 0,
        "errors": 0,
        "cap_hits": 0,
    }
    actual = 0.0
    for record in records:
        actual += cost_of(record, prices[record["model"]])
        totals["responses"] += 1
        totals["errors"] += int(record["error"] is not None)
        totals["cap_hits"] += int(record["done_reason"] == "length")
        for key in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"):
            totals[key] += int(record[key])
    totals["actual_usd"] = round(actual, 6)
    return records, totals


def ledger_entry(
    batch_id: str,
    label: str,
    requests: Sequence[Mapping[str, Any]],
    plan: Mapping[str, float],
    meter: Meter,
) -> dict[str, Any]:
    """Build the ledger line for a batch about to be submitted."""
    return {
        "batch_id": batch_id,
        "stage": label,
        "requests": len(requests),
        "worst_case_usd": round(float(plan["worst_case_usd"]), 6),
        "actual_usd": None,
        "cap_usd": meter.cap_usd,
        "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
