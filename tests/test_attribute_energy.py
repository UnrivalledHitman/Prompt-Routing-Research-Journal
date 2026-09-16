"""Unit tests for reading periods without mains power in scripts/attribute_energy.py."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "attribute_energy", REPO_ROOT / "scripts" / "attribute_energy.py"
)
assert SPEC is not None and SPEC.loader is not None
attribute_energy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attribute_energy)


def test_load_on_battery_reads_utc_periods(tmp_path: Path) -> None:
    """Periods become Unix-time intervals; a missing file means no periods."""
    path = tmp_path / "power.yaml"
    path.write_text(
        'on_battery_utc:\n  - {from: "2026-09-14T11:41:19Z", to: "2026-09-14T12:25:48Z"}\n',
        encoding="utf-8",
    )
    ((since, until),) = attribute_energy.load_on_battery(path)
    assert since == datetime(2026, 9, 14, 11, 41, 19, tzinfo=UTC).timestamp()
    assert until - since == pytest.approx(44 * 60 + 29)
    assert attribute_energy.load_on_battery(tmp_path / "absent.yaml") == []


def test_load_on_battery_rejects_naive_and_inverted_periods(tmp_path: Path) -> None:
    """Times without a timezone, and periods ending before they start, are errors."""
    path = tmp_path / "power.yaml"
    path.write_text(
        'on_battery_utc:\n  - {from: "2026-09-14T11:41:19", to: "2026-09-14T12:25:48"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="timezone"):
        attribute_energy.load_on_battery(path)
    path.write_text(
        'on_battery_utc:\n  - {from: "2026-09-14T12:00:00Z", to: "2026-09-14T11:00:00Z"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="before"):
        attribute_energy.load_on_battery(path)


def test_repository_power_source_file_parses() -> None:
    """The committed record of battery periods is well formed."""
    periods = attribute_energy.load_on_battery(REPO_ROOT / "configs" / "power_source.yaml")
    assert len(periods) >= 1
    assert all(until >= since for since, until in periods)
