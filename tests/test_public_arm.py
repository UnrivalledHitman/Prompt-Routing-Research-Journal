"""Unit tests for the public-benchmark arm: the release builder and its analyses."""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from arbiter.eval import public
from arbiter.eval.analysis import Resamples, _blocks, _mean, _ratio
from arbiter.eval.policies import ORACLE, Detection, oracle_outcome

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_public_arm import build_frame, choose_subsets, read_release  # noqa: E402

POOL = ("cheap", "dear")
# Three queries: the first needs nothing, the second only the dear model, the third neither.
SCORES = {"cheap": [1.0, 0.5, 0.0], "dear": [1.0, 1.0, 0.0]}
COSTS = {"cheap": [1.0, 1.0, 1.0], "dear": [4.0, 4.0, 4.0]}


def _records(model: str, n: int = 3) -> list[dict[str, Any]]:
    scores, costs = SCORES.get(model, SCORES["cheap"]), COSTS.get(model, COSTS["cheap"])
    return [
        {
            "index": i,
            "score": scores[i % 3],
            "prompt_tokens": 10 * (i + 1),
            "completion_tokens": 5,
            "cost": costs[i % 3],
            "response": "x" * 8192,  # the release carries full generations; they are dropped
        }
        for i in range(n)
    ]


def _archive(tmp_path: Path, files: list[dict[str, Any]]) -> Path:
    """Write a miniature release archive, one JSON file per model and subset."""
    path = tmp_path / "release.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for spec in files:
            body = json.dumps(spec).encode("utf-8")
            name = f"bench/{spec['dataset_name']}/{spec['split']}/{spec['model_name']}.json"
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return path


def _file(model: str, dataset: str, split: str, n: int = 3) -> dict[str, Any]:
    return {
        "model_name": model,
        "dataset_name": dataset,
        "split": split,
        "data_fingerprint": f"{dataset}-{split}",
        "records": _records(model, n),
    }


def _records_frame() -> pd.DataFrame:
    rows = [
        {
            "dataset": "toy",
            "index": i,
            "model": model,
            "score": SCORES[model][i],
            "prompt_tokens": 10,
            "completion_tokens": 5 if model == "cheap" else 20,
            "cost": COSTS[model][i],
        }
        for model in POOL
        for i in range(3)
    ]
    return pd.DataFrame(rows)


def _table(threshold: float = 1.0) -> public.PublicTable:
    return public.build_table(_records_frame(), POOL, threshold)


def test_read_release_keeps_the_pool_and_drops_generations(tmp_path: Path) -> None:
    """Files outside the pool or the datasets are skipped, and only outcome fields survive."""
    files = [
        _file("cheap", "toy", "test"),
        _file("dear", "toy", "test"),
        _file("cheap", "other", "test"),
        _file("free", "toy", "test"),
    ]
    entries = read_release(_archive(tmp_path, files), set(POOL), {"toy"})
    assert sorted(entry["model"] for entry in entries) == ["cheap", "dear"]
    assert set(entries[0]["records"][0]) == {
        "index",
        "score",
        "prompt_tokens",
        "completion_tokens",
        "cost",
    }


def test_choose_subsets_takes_the_largest_shared_one(tmp_path: Path) -> None:
    """A subset only one model was run on loses to the smaller one the whole pool shares."""
    files = [
        _file("cheap", "toy", "big", n=9),
        _file("cheap", "toy", "test"),
        _file("dear", "toy", "test"),
    ]
    entries = read_release(_archive(tmp_path, files), set(POOL), {"toy"})
    chosen = choose_subsets(entries, set(POOL), ["toy"])
    assert chosen["toy"] == ("test", "toy-test")
    frame = build_frame(entries, chosen, set(POOL))
    assert len(frame) == 6
    assert set(frame.columns) == {
        "dataset",
        "split",
        "index",
        "model",
        "score",
        "prompt_tokens",
        "completion_tokens",
        "cost",
    }


def test_choose_subsets_rejects_a_dataset_the_pool_does_not_share(tmp_path: Path) -> None:
    """A dataset one pool model never ran cannot be used, since the pool must stay fixed."""
    entries = read_release(_archive(tmp_path, [_file("cheap", "toy", "test")]), set(POOL), {"toy"})
    with pytest.raises(ValueError, match="no subset covers"):
        choose_subsets(entries, set(POOL), ["toy"])


def test_build_frame_rejects_unequal_query_sets(tmp_path: Path) -> None:
    """Models must answer the same queries, or a column of the matrix would be ragged."""
    files = [_file("cheap", "toy", "test", n=3), _file("dear", "toy", "test", n=2)]
    entries = read_release(_archive(tmp_path, files), set(POOL), {"toy"})
    with pytest.raises(ValueError, match="different numbers"):
        build_frame(entries, {"toy": ("test", "toy-test")}, set(POOL))


def test_build_table_orders_by_cost_and_applies_the_threshold() -> None:
    """Columns run cheapest first, and a judged tie is not a correct answer."""
    table = _table()
    assert table.models == ("cheap", "dear")
    assert table.success.tolist() == [[True, True], [False, True], [False, False]]
    assert table.costs["dollars"][:, 0].tolist() == [1.0, 1.0, 1.0]
    assert table.costs["tokens"][0].tolist() == [15.0, 30.0]
    lenient = _table(threshold=0.5)
    assert lenient.success[1].tolist() == [True, True]


def test_build_table_rejects_a_missing_model_or_a_repeated_query() -> None:
    """Both faults would silently change the pool the tax is computed on."""
    with pytest.raises(ValueError, match="missing"):
        public.build_table(_records_frame(), [*POOL, "absent"], 1.0)
    doubled = pd.concat([_records_frame(), _records_frame().head(1)])
    with pytest.raises(ValueError, match="repeated"):
        public.build_table(doubled, POOL, 1.0)


def test_frontier_keeps_only_models_worth_escalating_to() -> None:
    """A model that costs more and answers no better is not a rung."""
    cost = np.array([1.0, 2.0, 3.0, 4.0])
    accuracy = np.array([0.4, 0.3, 0.6, 0.6])
    assert public.frontier(cost, accuracy) == (0, 2)


def test_pool_summary_marks_the_ladder_and_costs_per_solved() -> None:
    """The dear model answers one query more, so both models are rungs."""
    summary = public.pool_summary(_table(), ("dollars",))
    rows = summary[summary["scope"] == "all"].set_index("model")
    assert rows.loc["cheap", "accuracy"] == pytest.approx(1 / 3)
    assert rows.loc["dear", "dollars_per_solved"] == pytest.approx(12 / 2)
    assert rows["on_ladder"].tolist() == [True, True]
    assert set(summary["scope"]) == {"all", "toy"}


def test_policies_pay_the_tax_the_grid_does() -> None:
    """P1 escalates once, P3 overpays the cheap query, and the oracle pays neither."""
    table = _table()
    cost = table.costs["dollars"]
    ladder = public.frontier(cost.mean(axis=0), table.success.mean(axis=0))
    outcomes = public.policy_outcomes(table.success, cost, ladder, table.models)
    assert outcomes["P1"].cost.tolist() == [1.0, 5.0, 5.0]
    assert outcomes["P3"].cost.tolist() == [4.0, 4.0, 4.0]
    assert outcomes["oracle"].cost.tolist() == [1.0, 4.0, 4.0]
    assert set(outcomes) == {"P1", "P3", "P4-uniform", "random", "always-cheap", "always-dear",
                             "oracle"}  # fmt: skip


def test_static_policies_are_reported_for_ladder_rungs_only() -> None:
    """Models off the frontier are covered by the pool summary, not by the policy table."""
    success = np.array([[True, False, True]])
    cost = np.array([[1.0, 2.0, 3.0]])
    outcomes = public.policy_outcomes(success, cost, (0,), ("a", "b", "c"))
    assert "always-a" in outcomes
    assert "always-b" not in outcomes


def test_imperfect_detection_drops_the_oracle_and_the_static_policies() -> None:
    """Without oracle detection an oracle row would be meaningless, so it is not written."""
    table = _table()
    outcomes = public.policy_outcomes(
        table.success,
        table.costs["dollars"],
        (0, 1),
        table.models,
        Detection(p_accept_wrong=0.5),
    )
    assert set(outcomes) == {"P1", "P3", "P4-uniform", "random"}


def test_oracle_outcome_recovers_at_the_pool_top() -> None:
    """An unservable query is charged the top rung of whatever pool it was run on."""
    success = np.array([[False, False, False]])
    cost = np.array([[1.0, 2.0, 3.0]])
    assert oracle_outcome(success, cost, top=1).cost.tolist() == [2.0]
    assert oracle_outcome(success, cost, top=2).solved.tolist() == [0.0]


def test_tax_table_reports_every_scope_and_unit() -> None:
    """The oracle pays no tax; P3's tax is what sticking at the top costs the cheap query."""
    frame = public.tax_table(_table(), ("dollars",), n_boot=64, seed=1)
    rows = frame[frame["scope"] == "all"].set_index("policy")
    assert rows.loc["oracle", "tax"] == pytest.approx(0.0)
    assert rows.loc["P3", "tax"] == pytest.approx(1.5)  # (4-1) and (4-4), unservable dropped
    assert rows.loc["P1", "relative_tax"] == pytest.approx((1 + 5) / (1 + 4) - 1)
    assert set(frame["scope"]) == {"all", "toy"}
    assert (frame["detection"] == "oracle").all()


def test_tax_table_accepts_an_imperfect_detection_regime() -> None:
    """A user who sometimes keeps a wrong answer stops early and solves fewer queries."""
    frame = public.tax_table(
        _table(),
        ("dollars",),
        detections={"oracle": ORACLE, "illustrative": Detection(p_accept_wrong=0.5)},
        n_boot=64,
        seed=1,
    )
    rows = frame[(frame["scope"] == "all") & (frame["policy"] == "P1")].set_index("detection")
    assert rows.loc["illustrative", "solved_rate"] < rows.loc["oracle", "solved_rate"]


def test_cstar_distribution_counts_every_query_once() -> None:
    """Each query is charged to its cheapest sufficient model, or to nothing at all."""
    frame = public.cstar_distribution(_table(), ("dollars",))
    rows = frame[frame["scope"] == "all"].set_index("cstar")
    assert rows["count"].sum() == 3
    assert rows.loc["cheap", "count"] == 1
    assert rows.loc["dear", "count"] == 1
    assert rows.loc["unservable", "count"] == 1


def test_mode_pair_compares_the_thinking_release_with_its_sibling() -> None:
    """The plain model answers one query; thinking adds the second at four times the cost."""
    frame = public.mode_pair(_table(), "dear", "cheap", ("dollars",), n_boot=64, seed=1)
    row = frame[frame["scope"] == "all"].iloc[0]
    assert row["plain_suffices"] == pytest.approx(1 / 3)
    assert row["needs_thinking"] == pytest.approx(1 / 3)
    assert row["thinking_over_plain"] == pytest.approx((12 / 2) / (3 / 1))
    with pytest.raises(ValueError, match="not in the pool"):
        public.mode_pair(_table(), "absent", "cheap", ("dollars",), n_boot=8)


def test_blocked_bootstrap_matches_a_single_block() -> None:
    """Chunking the resample matrix bounds memory without moving any statistic."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    weights = rng.random(200)
    index = Resamples.draw(200, 50, seed=3).index
    blocks = [len(block) for block in _blocks(index)]
    assert sum(blocks) == 50
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("arbiter.eval.analysis.BLOCK", 400)
        assert len([*_blocks(index)]) > 1
        chunked = _mean(values, index), _ratio(values, weights, index)
    assert np.allclose(chunked[0], _mean(values, index))
    assert np.allclose(chunked[1], _ratio(values, weights, index))
