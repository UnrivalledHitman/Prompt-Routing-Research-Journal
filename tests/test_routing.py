"""Unit tests for router evaluation and statistics."""

from __future__ import annotations

import numpy as np
import pytest

from arbiter.eval import routing
from arbiter.eval.labels import cstar_cost, minimum_sufficient

COST = np.tile(np.arange(1.0, 9.0), (4, 1))
SUCCESS = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],  # c* = c1
        [0, 0, 1, 1, 1, 1, 1, 1],  # c* = c3
        [0, 0, 0, 0, 0, 0, 0, 1],  # c* = c8
        [0, 0, 0, 0, 0, 0, 0, 0],  # unservable
    ],
    dtype=bool,
)


def test_routing_errors_separate_under_and_over_routing() -> None:
    """Routing to c3 everywhere under-routes one servable prompt and over-routes one."""
    cstar = minimum_sufficient(SUCCESS, COST)
    errors = routing.routing_errors(np.full(4, 2), SUCCESS, COST, cstar)
    assert errors["under_rate"] == pytest.approx(1 / 3)
    assert errors["over_rate"] == pytest.approx(1 / 3)
    assert errors["over_cost"] == pytest.approx((2.0 + 0.0) / 2)
    assert errors["exact_rate"] == pytest.approx(1 / 3)


def test_perfect_predictor_reaches_the_oracle_at_high_tau() -> None:
    """With exact success probabilities the router pays exactly c* on servable prompts."""
    cstar = minimum_sufficient(SUCCESS, COST)
    best = cstar_cost(SUCCESS, COST)
    frame = routing.sweep(
        SUCCESS.astype(float), SUCCESS, COST, routing.mean_cell_cost(COST), cstar, best, [0.5]
    )
    row = frame.iloc[0]
    assert row["tax"] == pytest.approx(0.0)
    assert row["under_rate"] == 0.0
    assert row["solved_rate"] == pytest.approx(0.75)


def test_summarise_adds_router_cost_to_net_figures_only() -> None:
    """Router overhead changes net cost, not the policy tax."""
    from arbiter.eval.policies import manual_policy

    best = cstar_cost(SUCCESS, COST)
    outcome = manual_policy(SUCCESS, COST, "P3")
    gross = routing.summarise(outcome, best)
    net = routing.summarise(outcome, best, router_cost=0.5)
    assert gross["tax"] == net["tax"]
    assert net["net_mean_cost"] == pytest.approx(gross["mean_cost"] + 0.5)


def test_frontier_area_is_normalised_step_integral() -> None:
    """Accuracy 0.5 from cost 2 and 1.0 from cost 6, over [0, 10], averages to 0.5."""
    area = routing.frontier_area([2.0, 6.0], [0.5, 1.0], 0.0, 10.0)
    assert area == pytest.approx((0.5 * 4 + 1.0 * 4) / 10)
    with pytest.raises(ValueError):
        routing.frontier_area([1.0], [1.0], 1.0, 1.0)


def test_choose_tau_prefers_higher_threshold_on_ties() -> None:
    """Equal objectives resolve to the more conservative threshold."""
    import pandas as pd

    frame = pd.DataFrame({"tau": [0.1, 0.5, 0.9], "net_cost_per_solved": [3.0, 2.0, 2.0]})
    assert routing.choose_tau(frame) == 0.9


def test_bootstrap_interval_and_paired_p_value() -> None:
    """Intervals bracket the mean; a clear shift gives a tiny p, noise does not."""
    rng = np.random.default_rng(0)
    values = rng.normal(1.0, 1.0, size=300)
    point, lo, hi = routing.percentile_ci(
        {"x": values}, lambda a: float(np.mean(a["x"])), n_boot=2000
    )
    assert lo < point < hi
    assert routing.paired_p_value(values, n_boot=2000) <= 1 / 2000 * 2
    assert routing.paired_p_value(rng.normal(0.0, 1.0, size=300), n_boot=2000) > 0.05


def test_holm_adjustment() -> None:
    """Holm multiplies sorted p-values by the remaining count and keeps them monotone."""
    adjusted = routing.holm([0.01, 0.04, 0.03, 0.5])
    assert adjusted.tolist() == pytest.approx([0.04, 0.09, 0.09, 0.5])
