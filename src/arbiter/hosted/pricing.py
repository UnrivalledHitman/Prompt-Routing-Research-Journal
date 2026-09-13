"""Token prices, the cost of a response and the spend meter that bounds this arm.

Every dollar figure in the hosted arm comes from here, computed from the token counts the
provider itself reports rather than from an estimate. Two quantities matter:

* **Actual cost**, from a completed response's ``usage``. Reasoning tokens are billed as
  output tokens, so they are counted as output.
* **Worst-case cost**, from a request that has not run yet, assuming it emits its full
  ``max_output_tokens``. The meter plans against this, never against the expected case, so
  a batch that could carry the total past the authorised cap is refused before submission.

The cap is a hard limit set in ``configs/hosted_arm.yaml`` and authorised by the
repository owner. :class:`Meter` reads and appends a ledger so the limit holds across
separate invocations, not merely within one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PER_MILLION = 1_000_000.0


@dataclass(frozen=True)
class Price:
    """Per-million-token prices for one model, already net of any batch discount."""

    input: float
    output: float
    cached_input: float = 0.0

    def scaled(self, multiplier: float) -> Price:
        """Return the same prices multiplied by ``multiplier``, as the Batch API discount."""
        return Price(
            input=self.input * multiplier,
            output=self.output * multiplier,
            cached_input=self.cached_input * multiplier,
        )


def price_table(pricing: Mapping[str, Any], *, batch: bool) -> dict[str, Price]:
    """Build the price table from the config block, optionally at batch rates.

    Parameters
    ----------
    pricing : Mapping
        The ``pricing`` block of the config: a ``models`` mapping plus
        ``batch_multiplier``.
    batch : bool
        Apply the batch multiplier. Passing False gives list prices, which the paper
        quotes alongside what was actually paid.
    """
    multiplier = float(pricing["batch_multiplier"]) if batch else 1.0
    table: dict[str, Price] = {}
    for model, row in pricing["models"].items():
        price = Price(
            input=float(row["input"]),
            output=float(row["output"]),
            cached_input=float(row.get("cached_input", 0.0)),
        )
        table[model] = price.scaled(multiplier)
    return table


def usage_tokens(usage: Mapping[str, Any]) -> dict[str, int]:
    """Pull billable token counts out of a Responses API ``usage`` object.

    Reasoning tokens are already included in ``output_tokens``; they are returned
    separately as ``reasoning_tokens`` for the paper's accounting, not added again.
    """
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    cached = int(input_details.get("cached_tokens", 0) or 0)
    total_input = int(usage.get("input_tokens", 0) or 0)
    return {
        "input_tokens": total_input - cached,
        "cached_tokens": cached,
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "reasoning_tokens": int(output_details.get("reasoning_tokens", 0) or 0),
    }


def cost_of(tokens: Mapping[str, int], price: Price) -> float:
    """Return the dollar cost of one response from its token counts."""
    return (
        tokens.get("input_tokens", 0) * price.input
        + tokens.get("cached_tokens", 0) * price.cached_input
        + tokens.get("output_tokens", 0) * price.output
    ) / PER_MILLION


def worst_case(input_tokens: int, cap: int, price: Price) -> float:
    """Return the most one request can cost: its prompt, and output filling the cap."""
    return (input_tokens * price.input + cap * price.output) / PER_MILLION


class BudgetError(RuntimeError):
    """Raised when a batch would take cumulative spend past the authorised cap."""


@dataclass
class Meter:
    """Cumulative spend against a hard cap, persisted to a JSONL ledger.

    The ledger is the audit trail: one line per batch, carrying the worst case that was
    approved and the actual cost once the results are in. ``committed`` is the sum of
    every batch's worst case until its actual cost is known, at which point the actual
    replaces it -- so an in-flight batch is always charged at its worst case.
    """

    cap_usd: float
    reserve_usd: float = 0.0
    path: Path | None = None
    entries: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, cap_usd: float, reserve_usd: float, path: Path) -> Meter:
        """Read the ledger at ``path``, or start an empty one if it does not exist."""
        entries: list[dict[str, Any]] = []
        if path.exists():
            entries = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
        return cls(cap_usd=cap_usd, reserve_usd=reserve_usd, path=path, entries=entries)

    @property
    def committed(self) -> float:
        """Return spend charged so far: actual where known, worst case where not."""
        total = 0.0
        for entry in self.entries:
            actual = entry.get("actual_usd")
            total += float(entry["worst_case_usd"] if actual is None else actual)
        return total

    @property
    def spent(self) -> float:
        """Return the actual spend of every batch whose results have been collected."""
        return sum(float(e["actual_usd"]) for e in self.entries if e.get("actual_usd") is not None)

    @property
    def available(self) -> float:
        """Return what may still be committed, holding the reserve back."""
        return self.cap_usd - self.reserve_usd - self.committed

    def check(self, worst_case_usd: float, label: str) -> None:
        """Raise :class:`BudgetError` unless ``worst_case_usd`` fits the remaining budget."""
        if worst_case_usd > self.available:
            raise BudgetError(
                f"{label}: worst case ${worst_case_usd:.4f} exceeds the ${self.available:.4f} "
                f"still available (cap ${self.cap_usd:.2f}, reserve ${self.reserve_usd:.2f}, "
                f"already committed ${self.committed:.4f}). "
                f"Reduce the prompt count or ask the owner to raise the cap."
            )

    def record(self, entry: dict[str, Any]) -> None:
        """Append a batch to the ledger and flush it to disk."""
        self.entries.append(entry)
        self._flush()

    def settle(self, batch_id: str, actual_usd: float, usage: Mapping[str, Any]) -> None:
        """Replace a batch's worst case with what it actually cost."""
        for entry in self.entries:
            if entry["batch_id"] == batch_id:
                entry["actual_usd"] = round(float(actual_usd), 6)
                entry["usage"] = dict(usage)
                self._flush()
                return
        raise KeyError(f"no ledger entry for batch {batch_id}")

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in self.entries)
        self.path.write_text(body, encoding="utf-8")
