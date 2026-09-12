"""Energy to carbon (experimental protocol, section 6.2; paper section III-G).

The accounting is that of Lacoste et al. and Luccioni et al.: energy in kWh times the
carbon intensity of the grid that supplied it. It is a multiplication, so a figure at
one intensity rescales to any other, and a reported interval carries over unchanged.

The intensity is not a property of this experiment. Grids differ by more than an order
of magnitude, and the same run therefore has no single carbon number, which is why
``configs/analysis.yaml`` lists a range of intensities and the results are reported
across it rather than at one value.

This module deliberately imports nothing from the telemetry stack, so the analysis can
convert measured energy on a machine with no GPU.
"""

from __future__ import annotations

J_PER_KWH = 3.6e6  # 1 kWh = 3.6 MJ


def carbon_g(joules: float, intensity_g_per_kwh: float) -> float:
    """Return grams of CO2 equivalent for ``joules`` drawn from a grid of that intensity."""
    return joules / J_PER_KWH * intensity_g_per_kwh
