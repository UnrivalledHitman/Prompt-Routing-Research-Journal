"""Unit tests for offline energy integration and apportionment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arbiter.energy import telemetry


def test_integrate_energy_constant_power_with_interpolated_edges() -> None:
    """A constant 10 W over 0.9 s is 9 J, independent of where samples fall."""
    t = np.arange(0.0, 1.01, 0.1)
    p = np.full_like(t, 10.0)
    assert telemetry.integrate_energy(t, p, 0.05, 0.95) == pytest.approx(9.0)


def test_integrate_energy_is_exact_for_a_linear_ramp() -> None:
    """Trapezoids with interpolated edges integrate a linear ramp exactly."""
    t = np.arange(0.0, 1.01, 0.1)
    p = 100.0 * t
    assert telemetry.integrate_energy(t, p, 0.0, 1.0) == pytest.approx(50.0)
    assert telemetry.integrate_energy(t, p, 0.25, 0.75) == pytest.approx(25.0)


def test_integrate_energy_rejects_uncovered_windows_and_gaps() -> None:
    """Windows outside the samples, inverted windows and sampling gaps raise."""
    t = np.array([0.0, 0.1, 2.0, 2.1])
    p = np.ones_like(t)
    with pytest.raises(ValueError, match="cover"):
        telemetry.integrate_energy(t, p, -0.5, 0.1)
    with pytest.raises(ValueError, match="empty"):
        telemetry.integrate_energy(t, p, 1.0, 1.0)
    with pytest.raises(ValueError, match="gap"):
        telemetry.integrate_energy(t, p, 0.0, 2.1, max_gap_s=1.0)


def test_apportion_is_proportional_and_validates_weights() -> None:
    """Shares follow the weights and preserve the total."""
    assert telemetry.apportion(10.0, [1.0, 3.0]) == pytest.approx([2.5, 7.5])
    with pytest.raises(ValueError):
        telemetry.apportion(10.0, [0.0, 0.0])
    with pytest.raises(ValueError):
        telemetry.apportion(10.0, [1.0, -1.0])


def test_load_telemetry_drops_failed_power_reads(tmp_path: Path) -> None:
    """Rows without a power reading are skipped."""
    path = tmp_path / "t.csv"
    path.write_text(
        "perf_s,epoch_s,power_mw,util_pct,temp_c,mem_used_mb\n"
        "1.0,100.0,5000,0,40,900\n"
        "1.1,100.1,,0,40,900\n"
        "1.2,100.2,7500,3,41,900\n",
        encoding="utf-8",
    )
    t, p = telemetry.load_telemetry(path)
    assert t.tolist() == [1.0, 1.2]
    assert p.tolist() == [5.0, 7.5]


def test_load_telemetry_drops_impossible_power_reads(tmp_path: Path) -> None:
    """A single spike above the plausibility ceiling is treated as a read error."""
    path = tmp_path / "t.csv"
    path.write_text(
        "perf_s,epoch_s,power_mw,util_pct,temp_c,mem_used_mb\n"
        "1.0,100.0,1200,0,40,900\n"
        "1.1,100.1,588200,0,40,900\n"
        "1.2,100.2,2800,0,40,900\n",
        encoding="utf-8",
    )
    t, p = telemetry.load_telemetry(path)
    assert t.tolist() == [1.0, 1.2]
    assert p.tolist() == [1.2, 2.8]


def test_load_clock_keeps_rows_without_power(tmp_path: Path) -> None:
    """The perf-to-wall-clock mapping uses every row, including failed power reads."""
    path = tmp_path / "t.csv"
    path.write_text(
        "perf_s,epoch_s,power_mw,util_pct,temp_c,mem_used_mb\n"
        "1.0,100.0,5000,0,40,900\n"
        "1.1,100.1,,0,40,900\n",
        encoding="utf-8",
    )
    perf, epoch = telemetry.load_clock(path)
    assert perf.tolist() == [1.0, 1.1]
    assert epoch.tolist() == [100.0, 100.1]


def test_max_gap_in_window_counts_the_edges() -> None:
    """A window starting well before its first sample has a gap at the edge."""
    t = np.array([1.0, 1.1, 1.2, 3.0, 3.1])
    assert telemetry.max_gap_in_window(t, 1.0, 1.2) == pytest.approx(0.1)
    assert telemetry.max_gap_in_window(t, 1.0, 3.1) == pytest.approx(1.8)
    assert telemetry.max_gap_in_window(t, 0.0, 1.1) == pytest.approx(1.0)


def test_longest_gap_s_is_the_overlap_with_the_window() -> None:
    """Only the part of a stall inside the window counts; stalls outside count as zero."""
    stalls = [(10.0, 20.0), (40.0, 41.5)]
    assert telemetry.longest_gap_s(stalls, 15.0, 45.0) == pytest.approx(5.0)
    assert telemetry.longest_gap_s(stalls, 30.0, 45.0) == pytest.approx(1.5)
    assert telemetry.longest_gap_s(stalls, 21.0, 39.0) == 0.0
    assert telemetry.longest_gap_s([], 0.0, 1.0) == 0.0
