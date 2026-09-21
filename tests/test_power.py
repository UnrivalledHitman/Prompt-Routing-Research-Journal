"""Unit tests for the host power helpers."""

from __future__ import annotations

import sys

import pytest

from arbiter.utils import power


def test_ac_online_is_a_bool_or_unknown() -> None:
    """The reading is True, False or None, never an error."""
    assert power.ac_online() in (True, False, None)


def test_ac_online_is_unknown_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Other platforms report an unknown power state rather than guessing."""
    monkeypatch.setattr(sys, "platform", "linux")
    assert power.ac_online() is None


def test_keep_awake_enters_and_releases() -> None:
    """The context manager can be held and released, including after an exception."""
    with power.KeepAwake() as held:
        assert isinstance(held, power.KeepAwake)
    with pytest.raises(RuntimeError), power.KeepAwake():
        raise RuntimeError("stage failed")
