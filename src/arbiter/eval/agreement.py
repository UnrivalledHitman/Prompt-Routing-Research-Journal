"""Agreement between the open-ended judge and its validators (protocol, section 5.4).

The local judge is checked against a frontier model on a stratified sample of about 200
judged answers, and against the first author's blind labels on about 100 of them.
Agreement is Cohen's kappa on the success decision (a score of 4 or more) and
quadratic-weighted kappa on the 1-5 scores, each with a percentile bootstrap interval
over answers.

How the open-ended family is reported was fixed on 2026-09-12, before any judge output
existed. The rule uses the success-decision kappa between the local judge and the human
labels, with bands at the moderate and substantial boundaries of Landis and Koch:

* kappa >= 0.60: reported with the gradable families (``primary``);
* 0.40 <= kappa < 0.60: reported with an explicit caveat (``caveat``);
* kappa < 0.40: moved to a secondary analysis (``secondary``).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from arbiter.eval.grading import JUDGE_SUCCESS_SCORE

SCORES = (1, 2, 3, 4, 5)
PRIMARY_KAPPA = 0.60
CAVEAT_KAPPA = 0.40


# ---------------------------------------------------------------------------
# Kappa
# ---------------------------------------------------------------------------


def _codes(a: Sequence[int], b: Sequence[int], labels: Sequence[int]) -> np.ndarray:
    if len(a) != len(b):
        raise ValueError(f"ratings differ in length: {len(a)} and {len(b)}")
    lookup = {label: index for index, label in enumerate(labels)}
    try:
        return np.array(
            [lookup[int(x)] * len(labels) + lookup[int(y)] for x, y in zip(a, b, strict=True)],
            dtype=int,
        )
    except KeyError as exc:
        raise ValueError(f"rating {exc.args[0]!r} is not in {list(labels)}") from exc


def _weights(k: int, weights: str | None) -> np.ndarray:
    i, j = np.indices((k, k))
    if weights is None:
        return (i != j).astype(float)
    if weights == "linear":
        return np.abs(i - j) / (k - 1)
    if weights == "quadratic":
        return ((i - j) / (k - 1)) ** 2
    raise ValueError(f"unknown weights {weights!r}")


def _kappa(codes: np.ndarray, k: int, w: np.ndarray) -> float:
    if codes.size == 0:
        return float("nan")
    observed = np.bincount(codes, minlength=k * k).reshape(k, k) / codes.size
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    chance = float((w * expected).sum())
    if chance == 0.0:
        return float("nan")
    return 1.0 - float((w * observed).sum()) / chance


def cohen_kappa(
    a: Sequence[int], b: Sequence[int], labels: Sequence[int], weights: str | None = None
) -> float:
    """Compute Cohen's kappa between two raters, unweighted or linear or quadratic.

    Returns NaN when chance agreement is perfect, for example when both raters give
    every answer the same label.

    Raises
    ------
    ValueError
        If the ratings differ in length, use a label outside ``labels``, or ``weights``
        is unknown.
    """
    k = len(labels)
    return _kappa(_codes(a, b, labels), k, _weights(k, weights))


def bootstrap_kappa(
    a: Sequence[int],
    b: Sequence[int],
    labels: Sequence[int],
    weights: str | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute kappa with a percentile bootstrap interval over rated answers.

    Resamples in which kappa is undefined are left out of the interval.

    Returns
    -------
    kappa, low, high : float
    """
    k = len(labels)
    codes, w = _codes(a, b, labels), _weights(k, weights)
    point = _kappa(codes, k, w)
    if codes.size == 0:
        return point, float("nan"), float("nan")
    draws = np.random.default_rng(seed).integers(0, codes.size, size=(n_boot, codes.size))
    stats = np.array([_kappa(codes[row], k, w) for row in draws])
    if not np.isfinite(stats).any():
        return point, float("nan"), float("nan")
    alpha = 100 * (1.0 - level) / 2
    low, high = np.nanpercentile(stats, [alpha, 100 - alpha])
    return point, float(low), float(high)


def agreement_summary(
    scores_a: Sequence[int], scores_b: Sequence[int], n_boot: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    """Summarise agreement between two raters' 1-5 scores on the same answers.

    Raises
    ------
    ValueError
        If there are no paired scores.
    """
    a, b = np.asarray(scores_a, dtype=int), np.asarray(scores_b, dtype=int)
    if a.size == 0:
        raise ValueError("no paired scores")
    pass_a = (a >= JUDGE_SUCCESS_SCORE).astype(int)
    pass_b = (b >= JUDGE_SUCCESS_SCORE).astype(int)
    kappa_b, low_b, high_b = bootstrap_kappa(pass_a, pass_b, (0, 1), None, n_boot, seed)
    kappa_q, low_q, high_q = bootstrap_kappa(a, b, SCORES, "quadratic", n_boot, seed)
    return {
        "n": int(a.size),
        "success_rate_a": float(pass_a.mean()),
        "success_rate_b": float(pass_b.mean()),
        "percent_agreement": float((pass_a == pass_b).mean()),
        "kappa_binary": kappa_b,
        "kappa_binary_ci": [low_b, high_b],
        "kappa_quadratic": kappa_q,
        "kappa_quadratic_ci": [low_q, high_q],
    }


def open_family_status(kappa_binary_human: float) -> str:
    """Apply the pre-registered rule for how the open-ended family is reported.

    Raises
    ------
    ValueError
        If kappa is undefined.
    """
    if not np.isfinite(kappa_binary_human):
        raise ValueError("judge-human kappa is undefined")
    if kappa_binary_human >= PRIMARY_KAPPA:
        return "primary"
    if kappa_binary_human >= CAVEAT_KAPPA:
        return "caveat"
    return "secondary"


# ---------------------------------------------------------------------------
# Validation sample
# ---------------------------------------------------------------------------


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Split ``n`` draws across strata in proportion to size, at least one each if possible.

    Uses largest remainders, so the counts sum to ``n`` and never exceed a stratum.
    """
    keys = sorted(sizes)
    if n >= sum(sizes.values()):
        return {key: sizes[key] for key in keys}
    alloc = {key: int(n >= len(keys) and sizes[key] > 0) for key in keys}
    capacity = {key: sizes[key] - alloc[key] for key in keys}
    remaining, total = n - sum(alloc.values()), sum(capacity.values())
    quotas = {key: remaining * capacity[key] / total if total else 0.0 for key in keys}
    for key in keys:
        alloc[key] += int(quotas[key])
    leftover = n - sum(alloc.values())
    by_remainder = sorted(keys, key=lambda key: (-(quotas[key] - int(quotas[key])), key))
    for key in by_remainder[:leftover]:
        alloc[key] += 1
    return alloc


def stratified_sample(strata: Sequence[str], n: int, seed: int, salt: str) -> list[int]:
    """Draw ``n`` indices spread across strata by :func:`allocate`, reproducibly.

    Returns
    -------
    list of int
        Sorted indices into ``strata``.
    """
    groups: dict[str, list[int]] = {}
    for index, stratum in enumerate(strata):
        groups.setdefault(stratum, []).append(index)
    counts = allocate({key: len(members) for key, members in groups.items()}, n)
    rng = random.Random(f"{seed}|{salt}")
    chosen: list[int] = []
    for key in sorted(groups):
        chosen.extend(rng.sample(groups[key], counts[key]))
    return sorted(chosen)
