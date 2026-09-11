"""Unit tests for the routers built on pretrained networks, run with tiny random models.

Tokenizers are read from the local Hugging Face cache when present; tests that need one
are skipped otherwise. No weights are loaded.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from arbiter.eval import analysis
from arbiter.eval.labels import LabelTable
from arbiter.eval.policies import ORACLE
from arbiter.routers import neural
from arbiter.routers.llm import success_from_labels

NEURAL = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "configs/analysis.yaml").read_text(encoding="utf-8")
)["routers"]["neural"]
MEAN_COST = np.array([1.0, 3.0, 2.0, 6.0, 4.0, 12.0, 8.0, 24.0])


def _toy(n: int) -> tuple[list[str], np.ndarray]:
    """Hard prompts (integrals) fail on the four smallest cells; easy ones succeed everywhere."""
    texts, rows = [], []
    for i in range(n):
        hard = i % 2 == 0
        if hard:
            texts.append(f"Compute the integral of x^{i % 5} dx over [0, {i}].")
        else:
            texts.append(f"What colour is the sky on day {i}?")
        rows.append([not hard] * 4 + [True] * 4)
    return texts, np.asarray(rows, dtype=bool)


class WordTokenizer:
    """Split on words and symbols over a fixed vocabulary, with a leading [CLS] id."""

    pad_id = 0

    def __init__(self, texts: Sequence[str]) -> None:
        words = sorted({w for text in texts for w in self._split(text)})
        self.vocab = {word: i + 2 for i, word in enumerate(words)}

    @staticmethod
    def _split(text: str) -> list[str]:
        return re.findall(r"\w+|\S", text.lower())

    def encode(self, texts: Sequence[str], max_length: int) -> list[list[int]]:
        """Return ``[CLS] ids`` per text, truncated to ``max_length``."""
        return [[1, *(self.vocab[w] for w in self._split(t))][:max_length] for t in texts]


def _cached(load: Callable[[], Any], what: str) -> Any:  # noqa: ANN401
    try:
        return load()
    except Exception:  # noqa: BLE001 - any failure to read the cache means skip
        pytest.skip(f"{what} is not in the local Hugging Face cache")


def test_text_encoder_caches_batches_but_encodes_single_texts_afresh() -> None:
    """Repeated batches hit the cache; a single text, as when timing decisions, never does."""
    calls = []

    def encode(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        return np.array([[len(t), 1.0] for t in texts])

    encoder = neural.TextEncoder(encode)
    assert encoder(["bb", "a", "bb"])[:, 0].tolist() == [2, 1, 2]
    encoder(["a", "bb"])
    encoder(["a"])
    assert calls == [["a", "bb"], ["a"]]


def test_embedding_router_learns_from_embeddings() -> None:
    """Held-out hard prompts get lower small-cell success; constant cells stay constant."""
    texts, success = _toy(80)

    def encode(batch: Sequence[str]) -> np.ndarray:
        return np.array(
            [[float("integral" in t), float("colour" in t), len(t) / 50] for t in batch]
        )

    router = neural.EmbeddingRouter(encoder=encode).fit(texts[:60], success[:60])
    p_hat = router.predict_proba(texts[60:])
    hard = ~success[60:, 0]
    assert p_hat.shape == (20, 8)
    assert p_hat[hard, 0].mean() < p_hat[~hard, 0].mean() - 0.3
    assert np.all(p_hat[:, 4:] == 1.0)


def test_balanced_weights_equalise_outcomes_and_skip_constant_cells() -> None:
    """Both outcomes of a cell carry equal total weight, averaging one."""
    success = np.array([[1, 1], [0, 1], [0, 1], [0, 1]], dtype=bool)
    weights = neural.balanced_weights(success)
    column = success[:, 0]
    assert weights[column, 0].sum() == pytest.approx(weights[~column, 0].sum())
    assert weights[:, 0].mean() == pytest.approx(1.0)
    assert np.all(weights[:, 1] == 0.0)


def _distilbert(vocab: int) -> Any:  # noqa: ANN401
    from transformers import DistilBertConfig

    return DistilBertConfig(
        vocab_size=vocab, dim=32, hidden_dim=64, n_layers=1, n_heads=2,
        max_position_embeddings=64, dropout=0.0, attention_dropout=0.0,
    )  # fmt: skip


def _deberta(vocab: int) -> Any:  # noqa: ANN401
    from transformers import DebertaV2Config

    return DebertaV2Config(
        vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=64, relative_attention=True,
        position_buckets=16, pos_att_type=["p2c", "c2p"], position_biased_input=False,
        type_vocab_size=0, norm_rel_ebd="layer_norm", share_att_key=True,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )  # fmt: skip


@pytest.mark.parametrize("make_config", [_distilbert, _deberta], ids=["distilbert", "deberta-v2"])
def test_encoder_router_fits_a_tiny_random_encoder(make_config: Callable[[int], Any]) -> None:
    """Training separates hard from easy prompts, repeats exactly, and keeps constants."""
    texts, success = _toy(32)
    tokenizer = WordTokenizer(texts)
    settings = {
        "tokenizer": tokenizer,
        "config": make_config(len(tokenizer.vocab) + 2),
        "learning_rate": 3e-3,
        "epochs": 25,
        "batch_size": 8,
        "max_length": 32,
        "dropout": 0.0,
        "device": "cpu",
    }
    router = neural.EncoderRouter(**settings).fit(texts, success)
    p_hat = router.predict_proba(texts)
    hard = ~success[:, 0]
    assert p_hat.shape == (32, 8)
    assert router.losses[-1] < 0.5 * router.losses[0]
    assert p_hat[hard, :4].mean() < p_hat[~hard, :4].mean() - 0.4
    assert np.all(p_hat[:, 4:] == 1.0)
    unit_weights = np.ones(success.shape)
    again = neural.EncoderRouter(**settings).fit(texts, success, unit_weights)
    assert np.allclose(again.predict_proba(texts), p_hat, atol=1e-5)


def test_sentencepiece_tokenizer_reads_deberta_v3_ids() -> None:
    """The vocabulary's own ids, wrapped in [CLS] and [SEP], and truncated inside them."""
    from huggingface_hub import hf_hub_download

    spec = NEURAL["encoders"]["deberta-v3-small"]
    path = _cached(
        lambda: hf_hub_download(
            spec["model_id"], "spm.model", revision=spec["revision"], local_files_only=True
        ),
        "DeBERTa-v3-small's vocabulary",
    )
    tokenizer = neural.SentencePieceTokenizer(path)
    assert (tokenizer.pad_id, tokenizer.cls_id, tokenizer.sep_id) == (0, 1, 2)
    text = ["Route this prompt, please."]
    assert tokenizer.encode(text, 512) == [[1, 9343, 291, 7865, 261, 811, 260, 2]]
    assert tokenizer.encode(text, 4) == [[1, 9343, 291, 2]]


def test_cheapest_cell_is_inverted_by_success_from_labels() -> None:
    """Monotone success rows survive the round trip; unservable prompts go to the top."""
    order = np.argsort(MEAN_COST)
    starts = np.random.default_rng(0).integers(0, 9, size=50)
    success = np.zeros((50, 8), dtype=bool)
    for row, start in zip(success, starts, strict=True):
        row[order[start:]] = True
    cells = neural.cheapest_cell(success, MEAN_COST)
    servable = success.any(axis=1)
    recovered = success_from_labels(np.eye(8)[cells], MEAN_COST) == 1.0
    assert np.array_equal(recovered[servable], success[servable])
    assert np.all(cells[~servable] == np.argmax(MEAN_COST))


def _qwen_tokenizer() -> Any:  # noqa: ANN401
    from transformers import AutoTokenizer

    spec = NEURAL["lora"]
    return _cached(
        lambda: AutoTokenizer.from_pretrained(
            spec["model_id"], revision=spec["revision"], local_files_only=True
        ),
        "Qwen3-0.6B's tokenizer",
    )


def test_router_ids_build_the_zero_shot_prompt_without_thinking() -> None:
    """The request sits inside the R3 prompt, the reply starts after an empty think block."""
    tokenizer = _qwen_tokenizer()
    ids = neural.router_ids(tokenizer, "What is 2 + 2?", 768)
    text = tokenizer.decode(ids)
    assert "<request>\nWhat is 2 + 2?\n</request>" in text
    assert text.endswith("<think>\n\n</think>\n\n")
    assert neural.letter_ids(tokenizer) == tokenizer.convert_tokens_to_ids(list("ABCDEFGH"))
    cut = neural.router_ids(tokenizer, "word " * 2000, 50)
    assert "[...]" in tokenizer.decode(cut)
    assert len(cut) < len(ids) + 60


def test_lora_router_trains_only_adapters_on_a_tiny_random_qwen3() -> None:
    """Loss falls, only LoRA weights train, and success is cumulative in cost order."""
    pytest.importorskip("peft")
    from transformers import Qwen3Config

    tokenizer = _qwen_tokenizer()
    config = Qwen3Config(
        vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=1024, tie_word_embeddings=True,
    )  # fmt: skip
    texts, success = _toy(16)
    router = neural.LoraRouter(
        MEAN_COST, tokenizer=tokenizer, config=config, rank=4, alpha=8, lora_dropout=0.0,
        learning_rate=1e-2, epochs=8, batch_size=8, gradient_checkpointing=False, device="cpu",
    )  # fmt: skip
    with pytest.raises(ValueError, match="cost-sensitive"):
        router.fit(texts, success, np.ones(success.shape))
    router.fit(texts, success)
    trainable = [name for name, p in router._net.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)
    assert router.losses[-1] < router.losses[0]
    p_hat = router.predict_proba(texts)
    ordered = p_hat[:, np.argsort(MEAN_COST)]
    assert p_hat.shape == (16, 8)
    assert np.all(np.diff(ordered, axis=1) >= -1e-9)
    assert np.allclose(ordered[:, -1], 1.0)


SETTINGS = analysis.RouterSettings(
    units=("tokens", "energy_j"),
    label_unit="energy_j",
    taus=(0.3, 0.6, 0.9),
    seeds=(0,),
    detections={"oracle": ORACLE},
    recoveries=("top",),
)


def _splits(texts: list[str], success: np.ndarray) -> dict[str, analysis.Split]:
    """Split a toy table 60/20/20 by position, with the mean costs in every unit."""
    n = len(texts)
    prompts = pd.DataFrame({"prompt_id": [f"p{i:03d}" for i in range(n)], "family": "math"})
    prompts["source"], prompts["subset"] = "math", "x"
    cost = np.tile(MEAN_COST, (n, 1))
    zeros = np.zeros((n, 8), dtype=bool)
    table = LabelTable(prompts, success, {"tokens": cost, "energy_j": cost}, zeros, zeros)
    part = np.arange(n) % 5
    masks = {"train": part < 3, "val": part == 3, "test": part == 4}
    return {
        name: analysis.Split(
            table.take(mask), [t for t, keep in zip(texts, mask, strict=True) if keep]
        )
        for name, mask in masks.items()
    }


def _flat_overhead(table: LabelTable, unit: str) -> np.ndarray:
    return np.full(len(table), 2.0)


def test_encoder_router_runs_through_router_evaluation() -> None:
    """R2 fits with cost-sensitive weights, is selected on validation and timed on the CPU."""
    texts, success = _toy(40)
    parts = _splits(texts, success)
    tokenizer = WordTokenizer(texts)
    make = partial(
        neural.EncoderRouter, tokenizer=tokenizer, config=_distilbert(len(tokenizer.vocab) + 2),
        epochs=2, batch_size=8, max_length=32, device="cpu",
    )  # fmt: skip
    grid = ({"learning_rate": 1e-3}, {"learning_rate": 3e-3})
    spec = analysis.RouterSpec("R2", "R2", make, grid, cost_sensitive=True)
    resamples = analysis.Resamples.draw(len(parts["test"].table), 50, seed=0)
    run = analysis.run_router(
        spec, parts["train"], parts["val"], parts["test"], SETTINGS, resamples
    )
    assert sum(record["chosen"] for record in run.selection) == 1
    assert len(run.test) == 2
    assert run.overhead[0]["device"] == "cpu" and run.overhead[0]["p50_s"] > 0


def test_lora_router_runs_through_router_evaluation_with_metered_overhead() -> None:
    """R3-LoRA is charged its measured per-prompt cost instead of being timed."""
    pytest.importorskip("peft")
    from transformers import Qwen3Config

    tokenizer = _qwen_tokenizer()
    config = Qwen3Config(
        vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=1024, tie_word_embeddings=True,
    )  # fmt: skip
    texts, success = _toy(20)
    parts = _splits(texts, success)
    make = partial(
        neural.LoraRouter, MEAN_COST, tokenizer=tokenizer, config=config, rank=4, alpha=8,
        epochs=1, batch_size=8, gradient_checkpointing=False, device="cpu",
    )  # fmt: skip
    spec = analysis.RouterSpec("R3-LoRA", "R3", make, overhead=_flat_overhead, device="gpu")
    resamples = analysis.Resamples.draw(len(parts["test"].table), 50, seed=0)
    run = analysis.run_router(
        spec, parts["train"], parts["val"], parts["test"], SETTINGS, resamples
    )
    test = pd.DataFrame(run.test).set_index("unit")
    assert test.loc["energy_j", "router_cost"] == pytest.approx(2.0)
    assert run.overhead[0]["joules_per_decision"] == pytest.approx(2.0)
