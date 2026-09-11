"""Re-score the frontier validation sample with an OpenAI model. Costs money.

The model sees the local judge's rubric, messages and reply schema (experimental
protocol, section 5.4). Nothing is sent without ``--confirm-spend``. ``--estimate``
sends nothing and prints an upper bound on the cost in rupees, from the prompt lengths,
the output budget and the prices given on the command line. The API key is read from
``OPENAI_API_KEY``, loading ``.env`` if present, and is never printed or logged. Scores
are appended to ``frontier_judgements.jsonl`` after every answer, so a stopped run
resumes.

Usage::

    python scripts/judge_openai.py results/raw/labelling --model MODEL --estimate \
        --usd-in PRICE --usd-out PRICE --inr-per-usd RATE
    python scripts/judge_openai.py results/raw/labelling --model MODEL --confirm-spend \
        --usd-in PRICE --usd-out PRICE --inr-per-usd RATE --max-inr 500
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from arbiter.eval.judge import (  # noqa: E402
    CHARS_PER_TOKEN,
    KEY_FIELDS,
    RESPONSE_FORMAT,
    build_messages,
    load_config,
    parse_judgement,
    prompt_budget,
)
from arbiter.utils.jsonl import append_jsonl, read_jsonl  # noqa: E402

LOGGER = logging.getLogger("arbiter.judge_openai")
API_BASE = "https://api.openai.com/v1"
RETRY_STATUS = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 5


def request_body(
    model: str, messages: list[dict[str, str]], max_output_tokens: int
) -> dict[str, Any]:
    """Build a chat-completions request whose reply must follow the judge's schema."""
    schema = RESPONSE_FORMAT | {"additionalProperties": False}
    return {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "judgement", "strict": True, "schema": schema},
        },
        "max_completion_tokens": max_output_tokens,
    }


def prompt_tokens_bound(body: dict[str, Any]) -> float:
    """Overestimate a request's prompt tokens from its characters."""
    return sum(len(message["content"]) for message in body["messages"]) / CHARS_PER_TOKEN


def cost_inr(tokens_in: float, tokens_out: float, args: argparse.Namespace) -> float:
    """Convert token counts to rupees at the prices given on the command line."""
    return (tokens_in * args.usd_in + tokens_out * args.usd_out) / 1e6 * args.inr_per_usd


def post(http: httpx.Client, body: dict[str, Any]) -> dict[str, Any]:
    """Send one request, backing off on rate limits, server errors and dropped connections.

    Raises
    ------
    httpx.HTTPError
        On any other error status, or when a retryable failure persists.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = http.post("/chat/completions", json=body)
        except httpx.TransportError as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            LOGGER.warning("request failed (%s); retrying", type(exc).__name__)
        else:
            if response.status_code not in RETRY_STATUS or attempt == MAX_ATTEMPTS:
                response.raise_for_status()
                return response.json()
            LOGGER.warning("HTTP %d; retrying", response.status_code)
        time.sleep(min(2**attempt, 60))
    raise RuntimeError("unreachable")


def main() -> None:
    """Estimate the cost, or score the frontier sample."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--model", required=True, help="OpenAI model id")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "judge.yaml")
    parser.add_argument(
        "--max-output-tokens", type=int, default=2000, help="reply budget, reasoning included"
    )
    parser.add_argument("--usd-in", type=float, help="input price, USD per million tokens")
    parser.add_argument("--usd-out", type=float, help="output price, USD per million tokens")
    parser.add_argument("--inr-per-usd", type=float)
    parser.add_argument("--max-inr", type=float, help="stop before the spend could pass this")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--estimate", action="store_true", help="print the cost bound only")
    mode.add_argument("--confirm-spend", action="store_true", help="send the requests")
    args = parser.parse_args()
    priced = None not in (args.usd_in, args.usd_out, args.inr_per_usd)
    if (args.estimate or args.max_inr is not None) and not priced:
        parser.error("--estimate and --max-inr need --usd-in, --usd-out and --inr-per-usd")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    budget = prompt_budget(load_config(args.config))
    out_dir = args.raw_dir / "judge_validation"
    items, _ = read_jsonl(out_dir / "frontier_items.jsonl")
    out_path = out_dir / "frontier_judgements.jsonl"
    scored = read_jsonl(out_path)[0] if out_path.exists() else []
    done = {tuple(r[key] for key in KEY_FIELDS) for r in scored}
    requests = []
    for item in items:
        if tuple(item[key] for key in KEY_FIELDS) in done:
            continue
        messages, _ = build_messages(
            item["request"], item["reference"] or "", item["answer"], budget
        )
        requests.append((item, request_body(args.model, messages, args.max_output_tokens)))
    tokens_in = sum(prompt_tokens_bound(body) for _, body in requests)
    tokens_out = len(requests) * args.max_output_tokens
    LOGGER.info(
        "%d of %d answers left to score with %s: at most ~%.0f prompt and %d reply tokens",
        len(requests),
        len(items),
        args.model,
        tokens_in,
        tokens_out,
    )
    if args.estimate:
        LOGGER.info("upper bound on cost: INR %.0f", cost_inr(tokens_in, tokens_out, args))
        return

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set; add it to .env")
    spent = 0.0
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(base_url=API_BASE, headers=headers, timeout=180.0) as http:
        for n, (item, body) in enumerate(requests, start=1):
            if args.max_inr is not None:
                bound = cost_inr(prompt_tokens_bound(body), args.max_output_tokens, args)
                if spent + bound > args.max_inr:
                    LOGGER.warning("stopping: the next answer could pass INR %.0f", args.max_inr)
                    break
            data = post(http, body)
            choice = data["choices"][0]
            score, reason = parse_judgement(choice["message"].get("content") or "")
            usage = data.get("usage") or {}
            used_in = usage.get("prompt_tokens", 0)
            used_out = usage.get("completion_tokens", 0)
            if priced:
                spent += cost_inr(used_in, used_out, args)
            row = {key: item[key] for key in KEY_FIELDS} | {
                "category": item["category"],
                "judge_model": data.get("model", args.model),
                "score": score,
                "reason": reason,
                "refusal": choice["message"].get("refusal"),
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": used_in,
                "completion_tokens": used_out,
                "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            append_jsonl(out_path, [row])
            if n % 25 == 0 or n == len(requests):
                LOGGER.info("%d/%d scored; about INR %.0f spent", n, len(requests), spent)


if __name__ == "__main__":
    main()
