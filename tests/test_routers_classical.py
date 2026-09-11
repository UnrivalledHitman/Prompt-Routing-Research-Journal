"""Unit tests for the R1 classical routers."""

from __future__ import annotations

import numpy as np
import pytest

from arbiter.routers import classical


def _toy(n: int = 120) -> tuple[list[str], np.ndarray]:
    """Hard prompts (with integrals) fail on the four smallest cells; easy ones succeed."""
    rng = np.random.default_rng(0)
    texts, rows = [], []
    for i in range(n):
        hard = i % 2 == 0
        word = rng.choice(["alpha", "beta", "gamma", "delta"])
        if hard:
            texts.append(f"Compute the integral of {word} x^{i % 7} dx over [0, {i}].")
        else:
            texts.append(f"What colour is the {word} sky number {i}?")
        rows.append([not hard] * 4 + [True] * 3 + [True])
    return texts, np.asarray(rows, dtype=bool)


def test_cost_sensitive_weights() -> None:
    """Failures weigh the c8 resubmission; successes weigh the premium of c8."""
    success = np.array([[0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=bool)
    cost = np.tile(np.arange(1.0, 9.0), (2, 1))
    weights = classical.cost_sensitive_weights(success, cost)
    assert weights.shape == (2, 8)
    assert weights.mean(axis=0) == pytest.approx(np.ones(8))
    assert weights[0, 0] > weights[1, 0]


@pytest.mark.parametrize(
    "router",
    [
        classical.TfidfRouter(),
        classical.FeatureBoostRouter(backend="lightgbm", n_estimators=50),
        classical.FeatureBoostRouter(backend="xgboost", n_estimators=50),
    ],
    ids=lambda r: r.name,
)
def test_routers_learn_the_hard_easy_split(router: object) -> None:
    """Held-out hard prompts get lower small-cell success than easy ones."""
    texts, success = _toy()
    router.fit(texts[:80], success[:80])
    p_hat = router.predict_proba(texts[80:])
    assert p_hat.shape == (40, 8)
    hard = ~success[80:, 0]
    assert p_hat[hard, 0].mean() < p_hat[~hard, 0].mean() - 0.2
    assert np.allclose(p_hat[:, 7], 1.0)  # c8 always succeeds in training: constant


def test_feature_vector_matches_names() -> None:
    """Every feature has a name, and code fences and options are counted."""
    text = "Write code.\n```python\ndef f():\n    return 1\n```\nA. one\nB. two"
    values = dict(zip(classical.FEATURE_NAMES, classical.prompt_features(text), strict=True))
    assert values["code_fences"] == 2
    assert values["option_lines"] == 2
    assert values["imperative_start"] == 1.0


def test_direct_classifier_maps_unservable_to_c8() -> None:
    """The ablation predicts a cell index in the grid."""
    texts, _ = _toy()
    cstar = np.where(np.arange(len(texts)) % 2 == 0, 4, 0)
    cstar[0] = -1
    model = classical.DirectClassifier().fit(texts, cstar)
    predicted = model.predict(texts[80:])
    assert set(predicted.tolist()) <= set(range(8))
    assert (predicted[::2] == 4).mean() > 0.8
