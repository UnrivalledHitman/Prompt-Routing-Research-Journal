"""Unit tests for probe items, parsing and alignment, and the probe stage overrides."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from arbiter.data.corpus import PromptItem
from arbiter.eval import probes
from arbiter.eval.labels import LabelTable
from arbiter.labeling import harness

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICES = {
    "primary": {"price_in": 1.0, "price_out": 1.0},
    "output_heavy": {"price_in": 1.0, "price_out": 4.0},
}


def _item(prompt_id: str, text: str = "What is 2 + 2?") -> PromptItem:
    return PromptItem(prompt_id, "math", "src", text, "4", "pilot")


def _answer(prompt_id: str, size: str, **extra: Any) -> dict[str, Any]:  # noqa: ANN401
    record = {"prompt_id": prompt_id, "size": size, "mode": "no_think", "sample": 0}
    return record | {"variant": "base", "content": f"answer from {size}"} | extra


def _table(ids: list[str]) -> LabelTable:
    n = len(ids)
    prompts = pd.DataFrame({"prompt_id": ids, "family": "math", "source": "src", "subset": "x"})
    zeros = np.zeros((n, 8), dtype=bool)
    return LabelTable(prompts, zeros, {"tokens": np.ones((n, 8))}, zeros, zeros)


def test_verify_items_cover_the_p1_rungs_below_c8() -> None:
    """Think answers, extra samples, reruns, repeats and unknown prompts get no probe."""
    generations = [
        _answer("p0", "0.6b"),
        _answer("p0", "8b"),
        _answer("p0", "8b", mode="think"),
        _answer("p0", "4b", sample=1),
        _answer("p0", "1.7b", variant="cap8192"),
        _answer("zz", "0.6b"),
        _answer("p0", "0.6b"),
    ]
    items, cells = probes.verify_items([_item("p0")], generations)
    assert cells == [("p0@0.6b", "0.6b", "no_think"), ("p0@8b", "8b", "no_think")]
    assert "<answer>\nanswer from 8b\n</answer>" in items[1].text
    assert items[1].meta == {"probe": "verify", "prompt_id": "p0", "size": "8b"}


def test_pick_examples_and_route_items() -> None:
    """Examples come from training prompts, one per cell that is some prompt's c*."""
    splits = pd.DataFrame(
        {
            "prompt_id": ["a", "b", "c", "d"],
            "split": ["train", "train", "test", "train"],
            "cstar_energy_j": ["c1", "c3", "c3", "unservable"],
        }
    )
    texts = {"a": "short a", "b": "x" * 500, "c": "short c", "d": "short d"}
    examples = probes.pick_examples(splits, texts, "cstar_energy_j", seed=1)
    assert examples == [("short a", 0), ("x" * 500, 2)]
    items, cells = probes.route_items([_item("a")], examples)
    assert cells == [("a", "0.6b", "no_think")]
    assert items[0].meta == {"probe": "route-few"}
    assert "Examples" in items[0].text


def test_probe_row_parses_verification_and_routing() -> None:
    """Verification rows carry a confidence and routing rows a label distribution."""
    top = [{"token": "Yes", "logprob": math.log(0.6)}, {"token": "No", "logprob": math.log(0.2)}]
    base = {"family": "math", "tag": "t", "mode": "no_think", "sample": 0, "wall_s": 0.3}
    verify = probes.probe_row(
        base
        | {"prompt_id": "p0@4b", "size": "4b", "variant": "verify"}
        | {"logprobs": [{"token": "Yes", "logprob": math.log(0.6), "top_logprobs": top}]},
        "verify",
    )
    assert verify["prompt_id"] == "p0"
    assert verify["confidence"] == pytest.approx(0.75)
    assert verify["parsed"]
    route = probes.probe_row(
        base | {"prompt_id": "p0", "size": "0.6b", "variant": "route-zero", "logprobs": []},
        "route",
    )
    assert not route["parsed"]
    assert math.isnan(route["p_A"])
    with pytest.raises(ValueError):
        probes.probe_row(route, "judge")


def test_alignment_unit_costs_and_filling() -> None:
    """Rows land on their prompt and cell; missing costs take the measured mean."""
    frame = pd.DataFrame(
        {
            "prompt_id": ["p0", "p0", "p1"],
            "size": ["0.6b", "4b", "0.6b"],
            "prompt_eval_count": [10, 30, 20],
            "eval_count": [1, 1, 1],
            "wall_s": [0.1, 0.3, 0.2],
            "confidence": [0.9, 0.4, 0.1],
        }
    )
    frame = probes.unit_costs(frame, PRICES)
    assert frame["tokens_output_heavy"].tolist() == [14.0, 34.0, 24.0]
    assert frame["energy_j"].isna().all()

    table = _table(["p1", "p0", "p2"])
    confidence = probes.per_cell(frame, table, "confidence")
    assert (confidence[0, 0], confidence[1, 0], confidence[1, 4]) == (0.1, 0.9, 0.4)
    assert np.isnan(confidence[2]).all()
    tokens = probes.fill_missing(probes.per_cell(frame, table, "tokens"), "tokens")
    assert tokens[:, 0].tolist() == [21.0, 11.0, 16.0]
    assert tokens[:, 4].tolist() == [31.0, 31.0, 31.0]
    assert np.isnan(tokens[:, 7]).all()
    assert probes.per_prompt(frame, table, "wall_s")[:2].tolist() == [0.2, 0.1]

    routes = pd.DataFrame({"prompt_id": ["p0"]} | {c: [0.0] for c in probes.LABEL_COLUMNS})
    routes["p_C"] = 1.0
    labels = probes.label_matrix(routes, table)
    assert labels[1, 2] == 1.0
    assert np.isnan(labels[0]).all()


def test_probe_stage_overrides_corpus_and_request(tmp_path: Path) -> None:
    """A probe stage reads its own corpus and sends extra chat parameters."""
    config = harness.load_config(REPO_ROOT / "configs" / "labelling.yaml")
    cells = tmp_path / "cells.csv"
    cells.write_text("prompt_id,size,mode\np0,0.6b,no_think\n", encoding="utf-8")
    config["stages"]["probe"] = {
        "select": "cells",
        "cells_file": str(cells),
        "corpus": str(tmp_path / "probe.jsonl"),
        "request": {"logprobs": True, "top_logprobs": 20},
        "decoding": {"no_think": {"num_predict": 8}},
    }
    stage = harness.resolve_stage(config, "probe", REPO_ROOT)
    assert stage.corpus == tmp_path / "probe.jsonl"
    assert stage.request == {"logprobs": True, "top_logprobs": 20}
    assert stage.decoding["no_think"]["num_predict"] == 8
    pilot = harness.resolve_stage(config, "pilot", REPO_ROOT)
    assert pilot.corpus is None
    assert pilot.request == {}


class _Client:
    """Records chat arguments and answers with fixed log probabilities."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def chat(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401
        """Return a one-token reply."""
        self.kwargs = kwargs
        top = [SimpleNamespace(token="B", logprob=-0.1), SimpleNamespace(token="A", logprob=-2.5)]
        return SimpleNamespace(
            message=SimpleNamespace(content="B", thinking=None),
            done_reason="length",
            eval_count=1,
            prompt_eval_count=50,
            eval_duration=1,
            prompt_eval_duration=1,
            load_duration=1,
            total_duration=1,
            logprobs=[SimpleNamespace(token="B", logprob=-0.1, top_logprobs=top)],
        )


class _Tokenizer:
    """Counts characters as tokens."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Return one id per character."""
        return list(range(len(text)))


def test_generate_one_sends_request_and_records_logprobs(tmp_path: Path) -> None:
    """Probe stages pass log-probability parameters and keep the returned distribution."""
    config = harness.load_config(REPO_ROOT / "configs" / "labelling.yaml")
    request = {"logprobs": True, "top_logprobs": 20}
    config["stages"]["probe"] = {"select": "pilot", "request": request}
    paths = [tmp_path / name for name in ("g.jsonl", "w.jsonl", "s.jsonl")]
    for name in ("probe", "pilot"):
        client = _Client()
        stage = harness.resolve_stage(config, name, REPO_ROOT)
        ctx = harness.Context(config, stage, client, _Tokenizer(), set(), *paths)
        record = harness.generate_one(ctx, config["models"][0], _item("p0"), "no_think", 0)
        if name == "probe":
            assert {key: client.kwargs[key] for key in request} == request
            assert record["logprobs"][0]["top_logprobs"][1] == {"token": "A", "logprob": -2.5}
        else:
            assert "logprobs" not in client.kwargs
            assert "logprobs" not in record


def test_build_probes_refuses_to_write_an_empty_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing or unhelpful generations file must stop the run, not label nothing."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import build_probes

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps(asdict(_item("q1"))), encoding="utf-8")
    argv = ["build_probes.py", "verify", "--corpus", str(corpus), "--out-dir", str(tmp_path)]

    monkeypatch.setattr(sys, "argv", [*argv, "--generations", str(tmp_path / "missing.jsonl")])
    with pytest.raises(SystemExit, match="does not exist"):
        build_probes.main()

    empty = tmp_path / "generations.jsonl"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [*argv, "--generations", str(empty)])
    with pytest.raises(SystemExit, match="no verification probes"):
        build_probes.main()

    monkeypatch.setattr(
        sys, "argv", [*argv, "--subsets", "nosuchsubset", "--generations", str(empty)]
    )
    with pytest.raises(SystemExit, match="no prompts in subsets"):
        build_probes.main()
    assert not list(tmp_path.glob("verify_*"))
