"""Unit tests for the study analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arbiter.eval import analysis
from arbiter.eval.labels import LabelTable, assign_splits, cstar_cost
from arbiter.eval.policies import ORACLE, Detection, manual_policy
from arbiter.eval.routing import summarise
from arbiter.routers.classical import DirectClassifier, TfidfRouter

BASE = np.array([1.0, 3.0, 2.0, 6.0, 4.0, 12.0, 8.0, 24.0])
SCALE = {"tokens": 100.0, "latency_s": 0.5, "energy_j": 10.0, "board_j": 15.0}
PATTERNS = {
    "easy": [1, 1, 1, 1, 1, 1, 1, 1],
    "medium": [0, 0, 0, 1, 1, 1, 1, 1],
    "hard": [0, 0, 0, 0, 0, 0, 0, 1],
    "impossible": [0, 0, 0, 0, 0, 0, 0, 0],
}
TEMPLATES = {
    "easy": "What is {i} plus two?",
    "medium": "Prove the lemma about {i} consecutive primes and bound the gap.",
    "hard": "Compute the contour integral of the residue series at pole {i} exactly.",
    "impossible": "Resolve the open conjecture number {i} with full rigour.",
}
KINDS = ("easy", "medium", "hard", "easy", "medium", "hard", "easy", "impossible")


def _table(success: np.ndarray, cost: np.ndarray, families: list[str] | None = None) -> LabelTable:
    """Wrap matrices in a label table with the same cost in every unit."""
    n = len(success)
    families = families or ["math"] * n
    prompts = pd.DataFrame(
        {
            "prompt_id": [f"p{i:03d}" for i in range(n)],
            "family": families,
            "source": families,
            "subset": "x",
        }
    )
    zeros = np.zeros_like(success, dtype=bool)
    costs = {unit: cost.astype(float) for unit in SCALE}
    return LabelTable(prompts, success.astype(bool), costs, zeros, zeros)


def make_table(per_family: int = 40, seed: int = 0) -> tuple[LabelTable, list[str]]:
    """Three families whose difficulty is readable from the prompt text."""
    rng = np.random.default_rng(seed)
    ids, families, texts, rows = [], [], [], []
    for family in ("code", "math", "mcq"):
        for i in range(per_family):
            kind = KINDS[i % len(KINDS)]
            ids.append(f"{family}-{i:03d}")
            families.append(family)
            texts.append(TEMPLATES[kind].format(i=i))
            rows.append(PATTERNS[kind])
    n = len(ids)
    noise = rng.uniform(0.9, 1.1, size=(n, 8))
    prompts = pd.DataFrame({"prompt_id": ids, "family": families, "source": families})
    prompts["subset"] = "x"
    zeros = np.zeros((n, 8), dtype=bool)
    costs = {unit: BASE * noise * scale for unit, scale in SCALE.items()}
    return LabelTable(prompts, np.asarray(rows, dtype=bool), costs, zeros, zeros), texts


def test_evaluate_matches_summarise_and_intervals_bracket() -> None:
    """Point estimates agree with the plain summary; tax splits into its two parts."""
    table, _ = make_table()
    cost = table.costs["energy_j"]
    best = cstar_cost(table.success, cost)
    outcome = manual_policy(table.success, cost, "P1")
    resamples = analysis.Resamples.draw(len(table), 400, seed=0)
    row = analysis.evaluate(outcome, best, resamples, router_cost=0.5).row
    reference = summarise(outcome, best, 0.5)
    for name in ("mean_cost", "net_mean_cost", "solved_rate", "net_cost_per_solved", "tax"):
        assert row[name] == pytest.approx(reference[name])
        assert row[f"{name}_lo"] <= row[name] <= row[f"{name}_hi"]
    assert row["tax"] == pytest.approx(row["escalation"] + row["stickiness"])


def test_policy_table_scopes_policies_and_oracle_tax() -> None:
    """Static policies appear once; the oracle pays no tax; P3 is all stickiness."""
    table, _ = make_table(per_family=16)
    detections = {"oracle": ORACLE, "illustrative": Detection(0.15, 0.05)}
    frame = analysis.policy_table(table, ["tokens"], detections, {"uniform": [0.125] * 8}, 50, 0)
    assert set(frame["scope"]) == {"all", "code", "math", "mcq"}
    oracle_rows = frame[frame["detection"] == "oracle"]
    expected = {"P1", "P2", "P3", "P4-uniform", "random", "oracle", "always-c1", "always-c8"}
    assert expected <= set(oracle_rows["policy"])
    assert not frame[frame["detection"] == "illustrative"]["policy"].str.startswith("always").any()
    assert np.allclose(oracle_rows[oracle_rows["policy"] == "oracle"]["tax"], 0.0)
    p3 = oracle_rows[(oracle_rows["policy"] == "P3") & (oracle_rows["scope"] == "all")].iloc[0]
    assert p3["escalation"] == 0.0
    assert p3["stickiness"] == pytest.approx(p3["tax"])


def test_asymmetry_prices_one_tier_low_and_high() -> None:
    """With c* = c3, the tier below fails and pays c8; the tier above pays its premium."""
    cost = np.arange(1.0, 9.0)[None, :]
    success = np.array([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=bool)
    frame = analysis.asymmetry(_table(success, cost), "tokens", n_boot=20, seed=0)
    row = frame[frame["scope"] == "all"].iloc[0]
    assert row["too_low_premium"] == pytest.approx(2.0 + 8.0 - 3.0)
    assert row["too_high_premium"] == pytest.approx(4.0 - 3.0)
    assert row["low_over_high"] == pytest.approx(7.0)
    assert row["too_low_failure_rate"] == 1.0


def test_mode_signature_ratios() -> None:
    """Cost per solved prompt of thinking over not thinking, and over a size step."""
    cost = np.tile(BASE, (4, 1))
    success = np.ones((4, 8), dtype=bool)
    success[:2, 0] = False
    frame = analysis.mode_signature(_table(success, cost), "energy_j", n_boot=20, seed=0)
    row = frame[(frame["scope"] == "all") & (frame["size"] == "0.6b")].iloc[0]
    assert row["per_solved_no_think"] == pytest.approx(4 * 1.0 / 2)
    assert row["per_solved_think"] == pytest.approx(3.0)
    assert row["think_over_no_think"] == pytest.approx(1.5)
    assert row["think_over_size_step"] == pytest.approx(3.0 / 2.0)


def _splits(table: LabelTable, texts: list[str]) -> dict[str, analysis.Split]:
    splits = assign_splits(table.prompts["prompt_id"].tolist(), table.prompts["family"].tolist(), 1)
    return {
        name: analysis.Split(
            table.take(splits == name),
            [text for text, split in zip(texts, splits, strict=True) if split == name],
        )
        for name in ("train", "val", "test")
    }


def test_run_router_selects_on_validation_and_evaluates_test() -> None:
    """Every seed, unit and recovery gets one test row at a threshold from the grid."""
    table, texts = make_table(per_family=40)
    parts = _splits(table, texts)
    settings = analysis.RouterSettings(
        units=("tokens", "energy_j"),
        label_unit="energy_j",
        taus=(0.3, 0.6, 0.9),
        seeds=(0, 1),
        detections={"oracle": ORACLE},
    )
    resamples = analysis.Resamples.draw(len(parts["test"].table), 50, seed=0)
    spec = analysis.RouterSpec("R1a", "R1", TfidfRouter, ({"c": 0.1}, {"c": 10.0}))
    run = analysis.run_router(
        spec, parts["train"], parts["val"], parts["test"], settings, resamples
    )
    test = pd.DataFrame(run.test)
    assert len(test) == 2 * 2 * 2
    assert set(test["tau"]) <= set(settings.taus)
    assert sum(record["chosen"] for record in run.selection) == 1
    assert len(run.overhead) == 2 and run.overhead[0]["p50_s"] > 0
    assert set(run.evaluated) == {
        (u, r, "oracle") for u in settings.units for r in ("top", "climb")
    }
    assert len(pd.concat(run.sweeps)) == 2 * 2 * 3
    top = test[(test["unit"] == "energy_j") & (test["recovery"] == "top")].iloc[0]
    assert top["mean_cost"] < parts["test"].table.costs["energy_j"][:, 7].mean()

    direct = analysis.RouterSpec("direct", "R1-direct", DirectClassifier, direct=True)
    run = analysis.run_router(
        direct, parts["train"], parts["val"], parts["test"], settings, resamples
    )
    assert pd.DataFrame(run.test)["tau"].isna().all()


def test_compare_pairs_candidates_with_baselines() -> None:
    """The oracle saves against P1 and P3, significantly after Holm correction."""
    table, _ = make_table(per_family=20)
    resamples = analysis.Resamples.draw(len(table), 500, seed=0)
    _, evaluated = analysis.baseline_evaluations(
        table, ["energy_j"], {"oracle": ORACLE}, {"uniform": [0.125] * 8}, resamples
    )
    comparisons = analysis.compare(
        {"oracle": evaluated[("energy_j", "oracle", "oracle")]},
        {name: evaluated[("energy_j", "oracle", name)] for name in ("P1", "P3")},
        resamples,
    )
    assert (comparisons["saving"] > 0).all()
    assert (comparisons["saving_lo"] > 0).all()
    assert (comparisons["saving_p_holm"] <= 0.01).all()
    assert (comparisons["saving_p_holm"] >= comparisons["saving_p"]).all()


def test_frontier_table_normalises_over_static_costs() -> None:
    """Areas are step integrals over the static cost range."""
    baselines = pd.DataFrame(
        {
            "unit": "tokens",
            "detection": "oracle",
            "policy": ["always-c1", "always-c8", "oracle"],
            "mean_cost": [1.0, 9.0, 2.0],
            "solved_rate": [0.5, 1.0, 1.0],
        }
    )
    sweeps = pd.DataFrame(
        {
            "router": "R",
            "seed": 0,
            "unit": "tokens",
            "tau": [0.1, 0.9],
            "net_mean_cost": [1.0, 5.0],
            "solved_rate": [0.5, 1.0],
            "single_mean_cost": [1.0, 5.0],
            "single_accuracy": [0.5, 0.9],
            "router_cost": 0.0,
        }
    )
    frame = analysis.frontier_table(sweeps, baselines).set_index("curve")
    assert frame.loc["R", "area"] == pytest.approx((0.5 * 4 + 1.0 * 4) / 8)
    assert frame.loc["static", "area"] == pytest.approx(0.5)
    assert frame.loc["oracle", "area"] == pytest.approx(7 / 8)
