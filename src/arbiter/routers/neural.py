"""Routers built on pretrained networks (experimental protocol, section 7).

* **R1c.** Frozen all-MiniLM-L6-v2 sentence embeddings into per-cell logistic
  regression. The encoder is not trained, so R1c is reported with the classical routers.
* **R2.** DistilBERT or DeBERTa-v3-small, fine-tuned with eight sigmoid outputs on the
  first token's final hidden state. The binary cross-entropy is class-weighted so that
  both outcomes of every cell carry equal weight. The cost-sensitive variant multiplies
  in :func:`~arbiter.routers.classical.cost_sensitive_weights`.
* **R3-LoRA.** Qwen3-0.6B with LoRA adapters, trained to name the cheapest sufficient
  configuration by letter from the zero-shot R3 prompt. As for the prompted R3, only the
  eight letters are read at the first answer position, and their distribution becomes
  per-cell success through :func:`~arbiter.routers.llm.success_from_labels`.

Weights are read from the local Hugging Face cache at pinned revisions and are never
downloaded. R2 and R3-LoRA train on the GPU when one is available. R1c and R2 decide on
the CPU, so they are timed and charged like R1. R3-LoRA decides on the GPU, like the
prompted R3.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.nn import functional

from arbiter.routers.classical import N_CELLS, PerCellModel
from arbiter.routers.llm import LETTERS, router_prompt, success_from_labels

LOGGER = logging.getLogger("arbiter.routers")
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
DISTILBERT = "distilbert/distilbert-base-uncased"
QWEN = "Qwen/Qwen3-0.6B"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EVAL_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Shared training helpers
# ---------------------------------------------------------------------------


def resolve_device(name: str) -> torch.device:
    """Return the device called ``name``; ``auto`` is CUDA when available, else the CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def autocast(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    """Return bfloat16 autocast on CUDA and a no-op elsewhere."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def batches(indices: np.ndarray, size: int) -> Iterator[np.ndarray]:
    """Yield consecutive slices of ``indices`` holding at most ``size`` entries."""
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


def length_order(sequences: Sequence[Sequence[int]]) -> np.ndarray:
    """Return indices that sort sequences by length, so batches carry little padding."""
    return np.argsort([len(s) for s in sequences], kind="stable")


def pad(
    sequences: Sequence[Sequence[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad token ids into a batch; return the ids and the attention mask."""
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.as_tensor(sequence, dtype=torch.long)
        mask[row, : len(sequence)] = 1
    return ids.to(device), mask.to(device)


def linear_warmup(total: int, warmup_ratio: float) -> Callable[[int], float]:
    """Return a learning-rate factor: linear warm-up, then linear decay to zero."""
    warmup = max(1, round(warmup_ratio * total))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / max(1, total - warmup))

    return factor


def train(
    parameters: Iterator[torch.nn.Parameter],
    loss_of: Callable[[np.ndarray], torch.Tensor],
    n: int,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
) -> list[float]:
    """Run AdamW over shuffled mini-batches of ``n`` examples; return each epoch's mean loss."""
    trainable = [p for p in parameters if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    total = epochs * math.ceil(n / batch_size)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_warmup(total, warmup_ratio))
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        running = 0.0
        for batch in batches(rng.permutation(n), batch_size):
            loss = loss_of(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            schedule.step()
            running += float(loss.detach()) * len(batch)
        losses.append(running / n)
    return losses


def log_fit(
    name: str, seed: int, device: torch.device, losses: list[float], started: float
) -> None:
    """Log one training run."""
    if losses:
        LOGGER.info(
            "%s seed %d: %d epochs on %s, loss %.3f -> %.3f, %.0f s",
            name, seed, len(losses), device.type, losses[0], losses[-1],
            time.perf_counter() - started,
        )  # fmt: skip


# ---------------------------------------------------------------------------
# R1c: frozen sentence embeddings + logistic regression
# ---------------------------------------------------------------------------


class TextEncoder:
    """A frozen sentence encoder that caches the embeddings of batches.

    A single text is always encoded afresh, as a deployed router would encode it, so
    decision timing includes the encoder. Batches reuse cached embeddings, which only
    avoids repeating the same work across the validation grid and the seeds.
    """

    def __init__(self, encode: Callable[[list[str]], np.ndarray]) -> None:
        self._encode = encode
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Return one embedding row per text."""
        if len(texts) == 1:
            return np.asarray(self._encode(list(texts)))
        missing = sorted(set(texts) - self._cache.keys())
        if missing:
            self._cache.update(zip(missing, np.asarray(self._encode(missing)), strict=True))
        return np.stack([self._cache[text] for text in texts])


@functools.cache
def sentence_encoder(model_id: str, revision: str | None) -> TextEncoder:
    """Load a sentence-transformers model from the local cache, on the CPU."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, revision=revision, device="cpu", local_files_only=True)
    return TextEncoder(
        lambda texts: model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )


@dataclass
class EmbeddingRouter:
    """R1c: frozen sentence embeddings into per-cell logistic regression.

    ``encoder`` replaces the pretrained sentence encoder, for tests.
    """

    c: float = 1.0
    model_id: str = MINILM
    revision: str | None = None
    seed: int = 0
    name: str = "R1c-minilm-lr"
    encoder: Callable[[Sequence[str]], np.ndarray] | None = None
    _model: PerCellModel | None = None

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        encoder = self.encoder or sentence_encoder(self.model_id, self.revision)
        return encoder(texts)

    def fit(
        self, texts: Sequence[str], success: np.ndarray, weights: np.ndarray | None = None
    ) -> EmbeddingRouter:
        """Embed the prompts and fit the eight classifiers."""
        self._model = PerCellModel(
            lambda: LogisticRegression(C=self.c, max_iter=2000, random_state=self.seed)
        ).fit(self._embed(texts), success, weights)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return per-cell success probabilities."""
        assert self._model is not None
        return self._model.predict_proba(self._embed(texts))


# ---------------------------------------------------------------------------
# R2: fine-tuned encoders with eight sigmoid outputs
# ---------------------------------------------------------------------------


class SentencePieceTokenizer:
    """DeBERTa-v3's vocabulary, read with ``sentencepiece`` alone.

    Transformers converts this vocabulary with protobuf or tiktoken, neither of which the
    project depends on. DeBERTa-v3's token ids are the SentencePiece ids, and a single
    text is wrapped as ``[CLS] text [SEP]``.
    """

    def __init__(self, path: str | Path) -> None:
        import sentencepiece

        self._model = sentencepiece.SentencePieceProcessor(model_file=str(path))
        self.cls_id = self._model.piece_to_id("[CLS]")
        self.sep_id = self._model.piece_to_id("[SEP]")
        self.pad_id = self._model.piece_to_id("[PAD]")

    def encode(self, texts: Sequence[str], max_length: int) -> list[list[int]]:
        """Return ``[CLS] ids [SEP]`` for each text, truncated to ``max_length`` tokens."""
        return [
            [self.cls_id, *ids[: max_length - 2], self.sep_id]
            for ids in self._model.encode(list(texts))
        ]


class HubTokenizer:
    """A Transformers tokenizer with the interface of :class:`SentencePieceTokenizer`."""

    def __init__(self, tokenizer: Any) -> None:  # noqa: ANN401
        self._tokenizer = tokenizer
        self.pad_id = int(tokenizer.pad_token_id)

    def encode(self, texts: Sequence[str], max_length: int) -> list[list[int]]:
        """Return token ids with special tokens, truncated to ``max_length``."""
        return self._tokenizer(list(texts), truncation=True, max_length=max_length)["input_ids"]


@functools.cache
def encoder_tokenizer(
    model_id: str, revision: str | None, kind: str
) -> SentencePieceTokenizer | HubTokenizer:
    """Load an encoder's tokenizer from the local cache.

    Raises
    ------
    ValueError
        If ``kind`` is neither ``auto`` nor ``sentencepiece``.
    """
    if kind == "sentencepiece":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_id, "spm.model", revision=revision, local_files_only=True)
        return SentencePieceTokenizer(path)
    if kind == "auto":
        from transformers import AutoTokenizer

        return HubTokenizer(
            AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
        )
    raise ValueError(f"unknown tokenizer kind {kind}")


def balanced_weights(success: np.ndarray) -> np.ndarray:
    """Weight labels so that both outcomes of each cell carry equal total weight.

    Weights average one within each cell. A cell with a single outcome gets zero weight,
    because it is predicted as a constant.
    """
    labels = np.asarray(success, dtype=bool)
    n = len(labels)
    positives = labels.sum(axis=0)
    negatives = n - positives
    varied = (positives > 0) & (negatives > 0)
    w_pos = np.divide(n, 2 * positives, out=np.zeros(labels.shape[1]), where=varied)
    w_neg = np.divide(n, 2 * negatives, out=np.zeros(labels.shape[1]), where=varied)
    return np.where(labels, w_pos, w_neg)


class SigmoidHeads(torch.nn.Module):
    """An encoder with one logit per cell, read from the first token's final hidden state."""

    def __init__(self, backbone: torch.nn.Module, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(backbone.config.hidden_size, N_CELLS)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, 8)."""
        hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.head(self.dropout(hidden[:, 0]))


@dataclass
class EncoderRouter:
    """R2: a pretrained encoder fine-tuned with eight sigmoid outputs.

    ``tokenizer`` is ``auto`` or ``sentencepiece`` to load the model's own, or an object
    with ``encode(texts, max_length)`` and ``pad_id``. ``config``, when given, builds a
    randomly initialised backbone instead of loading the weights, for tests. Training runs
    on ``device``; decisions always run on the CPU.
    """

    model_id: str = DISTILBERT
    revision: str | None = None
    tokenizer: Any = "auto"
    learning_rate: float = 5e-5
    epochs: int = 4
    batch_size: int = 16
    max_length: int = 512
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    device: str = "auto"
    seed: int = 0
    name: str = "R2-distilbert"
    config: Any = None
    losses: list[float] = field(default_factory=list)
    _net: SigmoidHeads | None = None
    _constants: dict[int, float] = field(default_factory=dict)

    def _encode(self, texts: Sequence[str]) -> tuple[Any, list[list[int]]]:
        tokenizer = self.tokenizer
        if isinstance(tokenizer, str):
            tokenizer = encoder_tokenizer(self.model_id, self.revision, tokenizer)
        return tokenizer, tokenizer.encode(texts, self.max_length)

    def _backbone(self) -> torch.nn.Module:
        from transformers import AutoModel

        if self.config is not None:
            return AutoModel.from_config(self.config)
        return AutoModel.from_pretrained(
            self.model_id, revision=self.revision, local_files_only=True, dtype=torch.float32
        )

    def fit(
        self, texts: Sequence[str], success: np.ndarray, weights: np.ndarray | None = None
    ) -> EncoderRouter:
        """Fine-tune the encoder and its heads on per-cell success."""
        started = time.perf_counter()
        torch.manual_seed(self.seed)
        device = resolve_device(self.device)
        tokenizer, tokens = self._encode(texts)
        varied = success.min(axis=0) != success.max(axis=0)
        self._constants = {int(c): float(success[0, c]) for c in np.flatnonzero(~varied)}
        weight = balanced_weights(success)
        if weights is not None:
            weight = weight * weights
        target = torch.as_tensor(success, dtype=torch.float32)
        weight_t = torch.as_tensor(weight, dtype=torch.float32)
        net = SigmoidHeads(self._backbone(), self.dropout).to(device)
        net.train()

        def loss_of(batch: np.ndarray) -> torch.Tensor:
            ids, mask = pad([tokens[i] for i in batch], tokenizer.pad_id, device)
            index = torch.as_tensor(batch)
            with autocast(device):
                logits = net(ids, mask)
            losses = functional.binary_cross_entropy_with_logits(
                logits.float(), target[index].to(device), reduction="none"
            )
            return (losses * weight_t[index].to(device)).sum() / (len(batch) * int(varied.sum()))

        self.losses = []
        if varied.any():
            self.losses = train(
                net.parameters(), loss_of, len(tokens),
                epochs=self.epochs, batch_size=self.batch_size,
                learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                warmup_ratio=self.warmup_ratio, seed=self.seed,
            )  # fmt: skip
        self._net = net.to("cpu").eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log_fit(self.name, self.seed, device, self.losses, started)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return per-cell success probabilities, computed on the CPU."""
        assert self._net is not None
        tokenizer, tokens = self._encode(texts)
        cpu = torch.device("cpu")
        out = np.empty((len(tokens), N_CELLS))
        with torch.inference_mode():
            for batch in batches(length_order(tokens), EVAL_BATCH_SIZE):
                ids, mask = pad([tokens[i] for i in batch], tokenizer.pad_id, cpu)
                out[batch] = torch.sigmoid(self._net(ids, mask)).numpy()
        for cell, value in self._constants.items():
            out[:, cell] = value
        return out


# ---------------------------------------------------------------------------
# R3-LoRA: Qwen3-0.6B trained to name the cheapest sufficient configuration
# ---------------------------------------------------------------------------


def cheapest_cell(success: np.ndarray, mean_cost: np.ndarray) -> np.ndarray:
    """Return each prompt's cheapest successful cell by mean cost, else the costliest cell.

    This is the configuration R3 is asked to name; :func:`success_from_labels` inverts it
    under the same order.
    """
    order = np.argsort(mean_cost, kind="stable")
    ranked = np.asarray(success, dtype=bool)[:, order]
    cheapest = order[np.argmax(ranked, axis=1)]
    return np.where(ranked.any(axis=1), cheapest, int(order[-1]))


@functools.cache
def causal_tokenizer(model_id: str, revision: str | None) -> Any:  # noqa: ANN401
    """Load a causal language model's tokenizer from the local cache."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)


def router_ids(tokenizer: Any, text: str, max_request_tokens: int) -> list[int]:  # noqa: ANN401
    """Return the token ids of the zero-shot R3 prompt for one request, as a chat turn.

    A request longer than ``max_request_tokens`` is cut and marked. Thinking is switched
    off, as in the R3 probes, so the reply opens with the letter.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) > max_request_tokens:
        text = tokenizer.decode(ids[:max_request_tokens]) + " [...]"
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": router_prompt(text)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(chat, add_special_tokens=False)["input_ids"]


def letter_ids(tokenizer: Any) -> list[int]:  # noqa: ANN401
    """Return the token id of each configuration letter.

    Raises
    ------
    ValueError
        If a letter does not encode as a single token of its own.
    """
    ids = [tokenizer.convert_tokens_to_ids(letter) for letter in LETTERS]
    for letter, token in zip(LETTERS, ids, strict=True):
        if token is None or tokenizer(letter, add_special_tokens=False)["input_ids"] != [token]:
            raise ValueError(f"configuration letter {letter} is not a single token")
    return ids


def letter_logits(
    net: Any,  # noqa: ANN401
    ids: torch.Tensor,
    mask: torch.Tensor,
    letters: torch.Tensor,
) -> torch.Tensor:
    """Return the eight letters' logits at each prompt's last position.

    Only those eight rows of the output projection are applied, which avoids computing
    the full vocabulary's logits at every position.
    """
    causal = net.get_base_model()
    hidden = causal.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    last = hidden[torch.arange(len(ids), device=ids.device), mask.sum(dim=1) - 1]
    logits = last.float() @ causal.lm_head.weight[letters].float().T
    bias = causal.lm_head.bias
    return logits if bias is None else logits + bias[letters].float()


@dataclass
class LoraRouter:
    """R3-LoRA: Qwen3-0.6B with LoRA adapters, trained to name the cheapest sufficient cell.

    ``mean_cost`` orders the cells, in the label unit on the training split, both for the
    training labels and for turning predicted letters into per-cell success. ``config``
    and ``tokenizer`` replace the pretrained model and its tokenizer, for tests. Training
    and decisions run on ``device``.
    """

    mean_cost: np.ndarray
    model_id: str = QWEN
    revision: str | None = None
    rank: int = 16
    alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 2
    batch_size: int = 8
    max_request_tokens: int = 768
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    gradient_checkpointing: bool = True
    device: str = "auto"
    seed: int = 0
    name: str = "R3-qwen3-0.6b-lora"
    config: Any = None
    tokenizer: Any = None
    losses: list[float] = field(default_factory=list)
    _net: Any = None
    _device: torch.device | None = None

    def _tokenizer(self) -> Any:  # noqa: ANN401
        if self.tokenizer is not None:
            return self.tokenizer
        return causal_tokenizer(self.model_id, self.revision)

    def _load(self, device: torch.device) -> Any:  # noqa: ANN401
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        if self.config is not None:
            base = AutoModelForCausalLM.from_config(self.config)
        else:
            base = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                local_files_only=True,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            )
        base.config.use_cache = False
        if self.gradient_checkpointing:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        adapters = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.lora_dropout,
            target_modules=list(LORA_TARGETS),
            bias="none",
        )
        return get_peft_model(base, adapters).to(device)

    def fit(
        self, texts: Sequence[str], success: np.ndarray, weights: np.ndarray | None = None
    ) -> LoraRouter:
        """Train the adapters to name each prompt's cheapest sufficient configuration.

        Raises
        ------
        ValueError
            If weights are passed: the cost-sensitive weights are defined per cell, and
            this router has no cost-sensitive variant.
        """
        if weights is not None:
            raise ValueError("R3-LoRA has no cost-sensitive variant")
        started = time.perf_counter()
        torch.manual_seed(self.seed)
        device = resolve_device(self.device)
        tokenizer = self._tokenizer()
        letters = torch.as_tensor(letter_ids(tokenizer), device=device)
        prompts = [router_ids(tokenizer, text, self.max_request_tokens) for text in texts]
        labels = torch.as_tensor(cheapest_cell(success, self.mean_cost))
        net = self._load(device)
        net.train()

        def loss_of(batch: np.ndarray) -> torch.Tensor:
            ids, mask = pad([prompts[i] for i in batch], tokenizer.pad_token_id, device)
            with autocast(device):
                logits = letter_logits(net, ids, mask, letters)
            return functional.cross_entropy(
                logits.float(), labels[torch.as_tensor(batch)].to(device)
            )

        self.losses = train(
            net.parameters(), loss_of, len(prompts),
            epochs=self.epochs, batch_size=self.batch_size,
            learning_rate=self.learning_rate, weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio, seed=self.seed,
        )  # fmt: skip
        self._net, self._device = net.eval(), device
        log_fit(self.name, self.seed, device, self.losses, started)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return per-cell success probabilities from the predicted letter distribution."""
        assert self._net is not None and self._device is not None
        tokenizer = self._tokenizer()
        letters = torch.as_tensor(letter_ids(tokenizer), device=self._device)
        prompts = [router_ids(tokenizer, text, self.max_request_tokens) for text in texts]
        labels = np.empty((len(prompts), len(LETTERS)))
        with torch.inference_mode():
            for batch in batches(length_order(prompts), self.batch_size):
                ids, mask = pad([prompts[i] for i in batch], tokenizer.pad_token_id, self._device)
                with autocast(self._device):
                    logits = letter_logits(self._net, ids, mask, letters)
                labels[batch] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return success_from_labels(labels, self.mean_cost)
