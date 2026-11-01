"""Calibration, accuracy, selective-prediction and uncertainty metrics.

Conventions
- `probs` is an (n, K) array of predicted probabilities; rows sum to one.
- `y` is an (n,) integer array of gold class indices.
- Calibration is measured on the top label: confidence = max_k p_k, correct = argmax == y.
- Brier score is the multi-class form, mean over items of sum_k (p_k - 1[k == y])^2.
  For a binary task this is twice the "probability of the positive class" Brier score.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_probs(probs: ArrayLike) -> FloatArray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2:
        raise ValueError("probs must be 2-D (n, K)")
    return p


def top_label(probs: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return (confidence, correct) arrays for the argmax prediction."""
    p = _as_probs(probs)
    yy = np.asarray(y, dtype=int)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == yy).astype(float)
    return conf, correct


def accuracy(probs: ArrayLike, y: ArrayLike) -> float:
    _, correct = top_label(probs, y)
    return float(correct.mean()) if correct.size else float("nan")


def _bin_index_equal_width(conf: FloatArray, n_bins: int) -> NDArray[np.int64]:
    # Bins are (lo, hi]; confidence exactly 0 falls in the first bin.
    idx = np.ceil(conf * n_bins - 1e-9).astype(int) - 1  # guard float noise at edges
    return np.clip(idx, 0, n_bins - 1)


def reliability_bins(
    conf: ArrayLike, correct: ArrayLike, n_bins: int = 15, scheme: str = "equal_width"
) -> list[dict[str, float]]:
    """Per-bin counts, mean confidence and accuracy. Empty bins are omitted."""
    c = np.asarray(conf, dtype=float)
    k = np.asarray(correct, dtype=float)
    out: list[dict[str, float]] = []
    if c.size == 0:
        return out
    if scheme == "equal_width":
        idx = _bin_index_equal_width(c, n_bins)
        for b in range(n_bins):
            m = idx == b
            if m.any():
                out.append(
                    {
                        "lo": b / n_bins,
                        "hi": (b + 1) / n_bins,
                        "count": float(m.sum()),
                        "conf": float(c[m].mean()),
                        "acc": float(k[m].mean()),
                    }
                )
    elif scheme == "adaptive":
        order = np.argsort(c, kind="stable")
        for chunk in np.array_split(order, min(n_bins, c.size)):
            if chunk.size:
                out.append(
                    {
                        "lo": float(c[chunk].min()),
                        "hi": float(c[chunk].max()),
                        "count": float(chunk.size),
                        "conf": float(c[chunk].mean()),
                        "acc": float(k[chunk].mean()),
                    }
                )
    else:
        raise ValueError(f"unknown binning scheme {scheme!r}")
    return out


def ece_from_conf(
    conf: ArrayLike, correct: ArrayLike, n_bins: int = 15, scheme: str = "equal_width"
) -> float:
    c = np.asarray(conf, dtype=float)
    if c.size == 0:
        return float("nan")
    bins = reliability_bins(c, correct, n_bins=n_bins, scheme=scheme)
    n = float(c.size)
    return float(sum(b["count"] / n * abs(b["acc"] - b["conf"]) for b in bins))


def ece(probs: ArrayLike, y: ArrayLike, n_bins: int = 15, scheme: str = "equal_width") -> float:
    """Expected calibration error on the top label."""
    conf, correct = top_label(probs, y)
    return ece_from_conf(conf, correct, n_bins=n_bins, scheme=scheme)


def brier(probs: ArrayLike, y: ArrayLike) -> float:
    p = _as_probs(probs)
    yy = np.asarray(y, dtype=int)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(yy)), yy] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def nll(probs: ArrayLike, y: ArrayLike, eps: float = 1e-12) -> float:
    p = _as_probs(probs)
    yy = np.asarray(y, dtype=int)
    py = np.clip(p[np.arange(len(yy)), yy], eps, 1.0)
    return float(-np.log(py).mean())


def percentile(values: ArrayLike, q: float) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.percentile(v, q)) if v.size else float("nan")


def cost_per_1k(costs: ArrayLike) -> float:
    c = np.asarray(costs, dtype=float)
    return float(c.mean() * 1000.0) if c.size else float("nan")


# --------------------------------------------------------------------------- selective prediction


def risk_coverage(conf: ArrayLike, correct: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Risk (error rate) of the k most confident predictions for k = 1..n.

    Returns (coverage, risk), both of length n. Ties are broken by input order.
    """
    c = np.asarray(conf, dtype=float)
    k = np.asarray(correct, dtype=float)
    order = np.argsort(-c, kind="stable")
    errors = np.cumsum(1.0 - k[order])
    counts = np.arange(1, c.size + 1, dtype=float)
    return counts / c.size, errors / counts


def aurc(conf: ArrayLike, correct: ArrayLike) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, risk = risk_coverage(conf, correct)
    return float(risk.mean()) if risk.size else float("nan")


def accuracy_at_coverage(conf: ArrayLike, correct: ArrayLike, coverage: float) -> float:
    cov, risk = risk_coverage(conf, correct)
    if cov.size == 0:
        return float("nan")
    i = int(np.searchsorted(cov, coverage - 1e-12))
    i = min(i, cov.size - 1)
    return float(1.0 - risk[i])


# --------------------------------------------------------------------------- summaries


@dataclass
class MetricSet:
    n: int
    accuracy: float
    ece: float
    ece_adaptive: float
    brier: float
    nll: float
    aurc: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "ece": self.ece,
            "ece_adaptive": self.ece_adaptive,
            "brier": self.brier,
            "nll": self.nll,
            "aurc": self.aurc,
        }


CORE_METRICS = ("accuracy", "ece", "ece_adaptive", "brier", "nll", "aurc")


def compute(probs: ArrayLike, y: ArrayLike, n_bins: int = 15) -> MetricSet:
    p = _as_probs(probs)
    yy = np.asarray(y, dtype=int)
    conf, correct = top_label(p, yy)
    return MetricSet(
        n=int(len(yy)),
        accuracy=float(correct.mean()),
        ece=ece_from_conf(conf, correct, n_bins, "equal_width"),
        ece_adaptive=ece_from_conf(conf, correct, n_bins, "adaptive"),
        brier=brier(p, yy),
        nll=nll(p, yy),
        aurc=aurc(conf, correct),
    )


# --------------------------------------------------------------------------- uncertainty


def bootstrap_ci(
    probs: ArrayLike,
    y: ArrayLike,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    n_bins: int = 15,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs for every core metric (items resampled with replacement)."""
    p = _as_probs(probs)
    yy = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(yy)
    draws: dict[str, list[float]] = {m: [] for m in CORE_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ms = compute(p[idx], yy[idx], n_bins).as_dict()
        for m in CORE_METRICS:
            draws[m].append(ms[m])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {m: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for m, v in draws.items()}


def paired_bootstrap(
    metric: Callable[[FloatArray, NDArray[np.int64]], float],
    probs_a: ArrayLike,
    probs_b: ArrayLike,
    y: ArrayLike,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap of metric(a) - metric(b) on the same items.

    The p-value is the two-sided bootstrap p: twice the smaller tail mass of the
    resampled differences around zero, capped at one.
    """
    pa, pb = _as_probs(probs_a), _as_probs(probs_b)
    yy = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = len(yy)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(pa[idx], yy[idx]) - metric(pb[idx], yy[idx])
    observed = metric(pa, yy) - metric(pb, yy)
    p_value = min(1.0, 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {
        "diff": float(observed),
        "lo": float(np.percentile(diffs, 100 * alpha / 2)),
        "hi": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
        "p_value": float(p_value),
    }


def mcnemar(correct_a: ArrayLike, correct_b: ArrayLike) -> dict[str, float]:
    """Exact McNemar test on paired correctness. b = a right & b wrong, c = a wrong & b right."""
    from scipy.stats import binomtest

    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    p = 1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return {"a_only": n10, "b_only": n01, "p_value": p}


def run_to_run(repeat_probs: Sequence[ArrayLike], y: ArrayLike) -> dict[str, Any]:
    """Variance across repeated runs of the same arm on the same items.

    Returns the std of accuracy and ECE across repeats, the mean per-item std of the
    gold-class probability, and the fraction of items whose argmax changes in any repeat.
    """
    yy = np.asarray(y, dtype=int)
    arrs = [_as_probs(p) for p in repeat_probs]
    if len(arrs) < 2:
        return {"repeats": len(arrs)}
    accs = [accuracy(p, yy) for p in arrs]
    eces = [ece(p, yy, n_bins=10) for p in arrs]
    stack = np.stack(arrs)  # (r, n, K)
    gold_p = stack[:, np.arange(len(yy)), yy]
    argmax = stack.argmax(axis=2)
    flips = (argmax != argmax[0]).any(axis=0)
    return {
        "repeats": len(arrs),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)),
        "ece_std": float(np.std(eces, ddof=1)),
        "gold_prob_std": float(gold_p.std(axis=0, ddof=1).mean()),
        "flip_rate": float(flips.mean()),
    }
