"""R3, a language model as router, and the cascade's verifier (paper sections III-D, III-F).

Both ask a pool model for a short reply and read the answer from the next-token
distribution at the reply's first visible token, returned by Ollama as log
probabilities, rather than from the sampled text.

* **R3.** Qwen3-0.6B names one of the eight configurations by letter. The letters'
  probabilities, renormalised, are a distribution over the minimum sufficient cell.
  If a cell succeeds whenever a cheaper one does, the probability that cell *k*
  succeeds is the mass on cells no costlier than *k*, which feeds the same threshold
  rule as every other router.
* **Verifier.** The model that wrote an answer is asked whether it is correct. Its
  confidence is P(Yes) / (P(Yes) + P(No)).

Prompts are built here and run by the labelling harness as ordinary short
generations, so their latency and energy are measured like the pool's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

LETTERS = "ABCDEFGH"
CONFIGURATIONS = (
    "0.6B model, answers directly",
    "0.6B model, thinks step by step first",
    "1.7B model, answers directly",
    "1.7B model, thinks step by step first",
    "4B model, answers directly",
    "4B model, thinks step by step first",
    "8B model, answers directly",
    "8B model, thinks step by step first",
)
EXAMPLE_CHARS = 400
SKIPPED_TOKENS = frozenset({"<think>", "</think>"})
ROUTER_INSTRUCTION = (
    "You route requests to language models. There are eight configurations, listed from "
    "the smallest model to the largest. Larger models cost more than smaller ones, and "
    "thinking first costs several times more than answering directly."
)
ROUTER_QUESTION = (
    "Which is the cheapest configuration that will answer the request below correctly? "
    "Reply with its letter only."
)
VERIFY_INSTRUCTION = (
    "You check answers. For each question and answer, reply Yes if the answer is correct "
    "and No if it is not."
)
VERIFY_EXAMPLES = (
    ("What is 15% of 80?", "15% of 80 is 0.15 x 80 = 12. The answer is 12.", "Yes"),
    (
        "Pens cost $2 for a pack of 3. How much do 12 pens cost?",
        "12 pens is 4 packs, and 4 x $3 = $12. The answer is $12.",
        "No",
    ),
    (
        "Write a Python function is_even(n) that returns True when n is even.",
        "def is_even(n):\n    return n % 2 == 1",
        "No",
    ),
    (
        "Which gas do plants take in for photosynthesis?\n\n"
        "A. Oxygen\nB. Nitrogen\nC. Carbon dioxide\nD. Helium",
        "Plants take in carbon dioxide. The answer is C.",
        "Yes",
    ),
)


def truncate(text: str, limit: int = EXAMPLE_CHARS) -> str:
    """Shorten an example prompt, marking the cut."""
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def router_prompt(text: str, examples: Sequence[tuple[str, int]] = ()) -> str:
    """Build the R3 prompt, zero-shot or with ``(prompt, cheapest sufficient cell)`` examples."""
    options = "\n".join(f"{LETTERS[i]}. {name}" for i, name in enumerate(CONFIGURATIONS))
    parts = [ROUTER_INSTRUCTION, options]
    if examples:
        shots = "\n\n".join(
            f"<request>\n{truncate(example)}\n</request>\n{LETTERS[cell]}"
            for example, cell in examples
        )
        parts.append(
            f"Examples, each followed by its cheapest sufficient configuration:\n\n{shots}"
        )
    parts.append(f"{ROUTER_QUESTION}\n\n<request>\n{text}\n</request>")
    return "\n\n".join(parts)


def verify_prompt(question: str, answer: str) -> str:
    """Build the few-shot self-verification prompt for one answer."""
    shots = "\n\n".join(
        f"<question>\n{q}\n</question>\n<answer>\n{a}\n</answer>\nCorrect: {verdict}"
        for q, a, verdict in VERIFY_EXAMPLES
    )
    return (
        f"{VERIFY_INSTRUCTION}\n\n{shots}\n\n"
        f"<question>\n{question}\n</question>\n<answer>\n{answer}\n</answer>\n"
        "Is this answer correct? Reply Yes or No."
    )


def answer_probabilities(logprobs: Sequence[Mapping[str, Any]] | None) -> dict[str, float]:
    """Return candidate probabilities at the first position holding visible text.

    Positions whose generated token is blank or a thinking tag are skipped. The
    position's top alternatives are used when present, otherwise its generated token
    alone.
    """
    for position in logprobs or ():
        token = str(position["token"]).strip()
        if not token or token in SKIPPED_TOKENS:
            continue
        probabilities: dict[str, float] = {}
        for entry in position.get("top_logprobs") or [position]:
            candidate = str(entry["token"])
            probabilities[candidate] = probabilities.get(candidate, 0.0) + math.exp(
                float(entry["logprob"])
            )
        return probabilities
    return {}


def label_distribution(probabilities: Mapping[str, float]) -> np.ndarray:
    """Renormalise the probabilities of the eight letters; NaN if none was offered."""
    mass = np.zeros(len(LETTERS))
    for token, p in probabilities.items():
        key = token.strip()
        if len(key) == 1 and key in LETTERS:
            mass[LETTERS.index(key)] += p
    total = mass.sum()
    return mass / total if total > 0 else np.full(len(LETTERS), np.nan)


def yes_confidence(probabilities: Mapping[str, float]) -> float:
    """Return P(Yes) / (P(Yes) + P(No)), NaN if neither was offered."""
    yes = sum(p for token, p in probabilities.items() if token.strip().lower() == "yes")
    no = sum(p for token, p in probabilities.items() if token.strip().lower() == "no")
    return yes / (yes + no) if yes + no > 0 else float("nan")


def success_from_labels(labels: np.ndarray, mean_cost: np.ndarray) -> np.ndarray:
    """Turn distributions over the cheapest sufficient cell into per-cell success.

    ``labels`` has shape (n, 8). Cell *k* is taken to succeed when the cheapest
    sufficient cell costs no more than *k* on average, so its success probability is
    the mass on those cells.
    """
    order = np.argsort(mean_cost, kind="stable")
    cumulative = np.cumsum(labels[:, order], axis=1)
    success = np.empty_like(cumulative)
    success[:, order] = cumulative
    return np.clip(success, 0.0, 1.0)


@dataclass
class LookupRouter:
    """R3 at evaluation time: distributions measured on the GPU, looked up by prompt text.

    There is nothing to train. A prompt without a usable distribution goes to the
    costliest cell, the conservative choice.
    """

    distributions: Mapping[str, np.ndarray]
    mean_cost: np.ndarray
    seed: int = 0
    name: str = "R3-qwen3-0.6b"

    def fit(
        self, texts: Sequence[str], success: np.ndarray, weights: np.ndarray | None = None
    ) -> LookupRouter:
        """Return the router unchanged; its distributions were measured beforehand."""
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return per-cell success probabilities for each prompt."""
        unknown = np.full(len(LETTERS), np.nan)
        rows = np.array([self.distributions.get(text, unknown) for text in texts], dtype=float)
        rows = rows.reshape(len(texts), len(LETTERS))
        missing = np.isnan(rows).any(axis=1)
        rows[missing] = 0.0
        rows[missing, int(np.argmax(self.mean_cost))] = 1.0
        return success_from_labels(rows, self.mean_cost)
