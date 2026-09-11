"""Append-only JSONL files that survive interrupted runs."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("arbiter.jsonl")


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


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Append records and force them to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
