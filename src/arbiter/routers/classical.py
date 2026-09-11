"""R1 routers: classical per-cell success predictors (experimental protocol, section 7).

Every router predicts, for each prompt, the probability that each of the eight cells
succeeds (paper eq. 7). The decision rule is applied separately. Cells whose training
labels are all one class get a constant prediction instead of a fitted model.

* R1a: logistic regression on TF-IDF word and character n-grams.
* R1b: gradient-boosted trees (LightGBM or XGBoost) on hand-built prompt features.
* R1-direct: eight-way classification of ``c*(q)`` on the R1a features. This is the
  ablation for the decoupled form.

The cost-sensitive variant weights each training example by the cost its error would
cause. A missed failure costs a resubmission to c8. A missed success costs the premium
of c8 over the cell.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger("arbiter.routers")
N_CELLS = 8
TOP = 7
IMPERATIVE_VERBS = frozenset(
    "write implement create find compute calculate return given solve determine explain list "
    "describe summarize summarise classify give name identify extract generate convert".split()
)


def cost_sensitive_weights(success: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Weight each (prompt, cell) example by the realised cost of misjudging it.

    A cell that fails is weighted by the cost of the resubmission to c8 that a wrong
    success prediction would trigger. A cell that succeeds is weighted by the premium
    of c8 over it, which a wrong failure prediction would waste. Weights are scaled to
    mean one per cell.
    """
    top = cost[:, [TOP]]
    weights = np.where(success, np.maximum(top - cost, 0.0), top)
    weights = weights + 1e-6
    return weights / weights.mean(axis=0, keepdims=True)


@dataclass
class PerCellModel:
    """Eight independent binary models sharing one feature matrix."""

    make_model: Any
    models: list[Any] = field(default_factory=list)
    constants: dict[int, float] = field(default_factory=dict)

    def fit(self, x: Any, success: np.ndarray, weights: np.ndarray | None = None) -> PerCellModel:  # noqa: ANN401
        """Fit one model per cell."""
        self.models, self.constants = [], {}
        for cell in range(N_CELLS):
            y = success[:, cell].astype(int)
            if y.min() == y.max():
                self.constants[cell] = float(y[0])
                self.models.append(None)
                continue
            model = self.make_model()
            kwargs = {} if weights is None else {"sample_weight": weights[:, cell]}
            model.fit(x, y, **kwargs)
            self.models.append(model)
        return self

    def predict_proba(self, x: Any) -> np.ndarray:  # noqa: ANN401
        """Return predicted success probabilities, shape (n, 8)."""
        n = x.shape[0]
        out = np.empty((n, N_CELLS))
        for cell, model in enumerate(self.models):
            if model is None:
                out[:, cell] = self.constants[cell]
            else:
                out[:, cell] = model.predict_proba(x)[:, 1]
        return out


# ---------------------------------------------------------------------------
# R1a: TF-IDF + logistic regression
# ---------------------------------------------------------------------------


def make_tfidf() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    """Return the word and character vectorizers used by R1a."""
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=30000)
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=30000
    )
    return word, char


@dataclass
class TfidfRouter:
    """R1a: TF-IDF word and character n-grams into per-cell logistic regression."""

    c: float = 1.0
    seed: int = 0
    name: str = "R1a-tfidf-lr"
    _vectorizers: tuple[TfidfVectorizer, TfidfVectorizer] | None = None
    _model: PerCellModel | None = None

    def _features(self, texts: Sequence[str], fit: bool) -> sparse.csr_matrix:
        if fit:
            self._vectorizers = make_tfidf()
            parts = [v.fit_transform(texts) for v in self._vectorizers]
        else:
            assert self._vectorizers is not None
            parts = [v.transform(texts) for v in self._vectorizers]
        return sparse.hstack(parts).tocsr()

    def fit(
        self, texts: Sequence[str], success: np.ndarray, weights: np.ndarray | None = None
    ) -> TfidfRouter:
        """Fit vectorizers and the eight classifiers."""
        x = self._features(texts, fit=True)
        self._model = PerCellModel(
            lambda: LogisticRegression(C=self.c, max_iter=2000, random_state=self.seed)
        ).fit(x, success, weights)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return per-cell success probabilities."""
        assert self._model is not None
        return self._model.predict_proba(self._features(texts, fit=False))


# ---------------------------------------------------------------------------
# R1b: hand features + gradient boosting
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    "chars",
    "words",
    "lines",
    "mean_word_len",
    "digit_ratio",
    "operator_density",
    "latex_commands",
    "code_fences",
    "code_keywords",
    "question_marks",
    "option_lines",
    "uppercase_ratio",
    "imperative_start",
    "flesch_reading_ease",
)


def _syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    return max(1, len(groups) - (1 if word.lower().endswith("e") and len(groups) > 1 else 0))


def prompt_features(text: str) -> list[float]:
    """Compute the hand-built R1b features for one prompt."""
    words = re.findall(r"[A-Za-z]+", text)
    n_chars = max(len(text), 1)
    n_words = max(len(words), 1)
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    syllables = sum(_syllables(w) for w in words) if words else 0
    first = words[0].lower() if words else ""
    return [
        float(len(text)),
        float(len(words)),
        float(text.count("\n") + 1),
        sum(len(w) for w in words) / n_words,
        sum(ch.isdigit() for ch in text) / n_chars,
        len(re.findall(r"[+\-*/=^<>]", text)) / n_chars,
        float(len(re.findall(r"\\[A-Za-z]+", text))),
        float(text.count("```")),
        float(len(re.findall(r"\b(def|return|class|import|assert|lambda|for|while)\b", text))),
        float(text.count("?")),
        float(len(re.findall(r"^[A-H]\. ", text, flags=re.MULTILINE))),
        sum(ch.isupper() for ch in text) / n_chars,
        float(first in IMPERATIVE_VERBS),
        206.835 - 1.015 * (n_words / sentences) - 84.6 * (syllables / n_words),
    ]


def feature_matrix(texts: Sequence[str]) -> np.ndarray:
    """Stack R1b features for many prompts."""
    return np.asarray([prompt_features(t) for t in texts], dtype=np.float64)


@dataclass
class FeatureBoostRouter:
    """R1b: hand-built features into per-cell gradient-boosted trees."""

    backend: str = "lightgbm"
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 4
    seed: int = 0
    name: str = ""
    _model: PerCellModel | None = None

    def __post_init__(self) -> None:
        """Name the router after its backend."""
        self.name = self.name or f"R1b-features-{self.backend}"

    def _make(self) -> Any:  # noqa: ANN401
        if self.backend == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=min(2**self.max_depth - 1, 31),
                min_child_samples=10,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=self.seed,
                n_jobs=2,
                verbose=-1,
            )
        if self.backend == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                n_jobs=2,
                eval_metric="logloss",
            )
        raise ValueError(f"unknown backend {self.backend}")

    def fit(
        self, texts: Sequence[str], success: np.ndarray, weights: np.ndarray | None = None
    ) -> FeatureBoostRouter:
        """Fit the eight boosted classifiers."""
        self._model = PerCellModel(self._make).fit(feature_matrix(texts), success, weights)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return per-cell success probabilities."""
        assert self._model is not None
        return self._model.predict_proba(feature_matrix(texts))


# ---------------------------------------------------------------------------
# R1-direct: eight-way classification of c*
# ---------------------------------------------------------------------------


@dataclass
class DirectClassifier:
    """Ablation: predict ``c*(q)`` directly as one of eight classes, or c8 if unservable."""

    c: float = 1.0
    seed: int = 0
    name: str = "R1-direct-tfidf-lr"
    _vectorizers: tuple[TfidfVectorizer, TfidfVectorizer] | None = None
    _model: LogisticRegression | None = None
    _classes: np.ndarray | None = None

    def fit(self, texts: Sequence[str], cstar: np.ndarray) -> DirectClassifier:
        """Fit on ``c*`` labels; unservable prompts (-1) are labelled c8."""
        self._vectorizers = make_tfidf()
        x = sparse.hstack([v.fit_transform(texts) for v in self._vectorizers]).tocsr()
        y = np.where(cstar < 0, TOP, cstar)
        self._model = LogisticRegression(C=self.c, max_iter=2000, random_state=self.seed)
        self._model.fit(x, y)
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Return the predicted cell per prompt."""
        assert self._model is not None and self._vectorizers is not None
        x = sparse.hstack([v.transform(texts) for v in self._vectorizers]).tocsr()
        return self._model.predict(x).astype(int)
