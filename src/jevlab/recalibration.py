"""Cheap post-hoc recalibration baselines: temperature scaling, Platt scaling, isotonic regression.

The question these answer: does a model trained for calibration (Jev's RLCD) beat
what anyone can get by fitting one or two parameters on a few hundred held-out labels?

All methods are evaluated with K-fold cross-fitting on the evaluation sample: each
fold is recalibrated by a map fitted on the other folds, so no item is scored by a
map that saw its own label.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
EPS = 1e-6


def _log(p: FloatArray) -> FloatArray:
    return np.log(np.clip(p, EPS, 1.0))


def softmax(z: FloatArray) -> FloatArray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: FloatArray = e / e.sum(axis=1, keepdims=True)
    return out


def fit_temperature(probs: FloatArray, y: NDArray[np.int64]) -> float:
    """Single temperature T minimising NLL of softmax(log p / T)."""
    from scipy.optimize import minimize_scalar

    logits = _log(probs)
    idx = np.arange(len(y))

    def loss(t: float) -> float:
        q = softmax(logits / t)
        return float(-np.log(np.clip(q[idx, y], EPS, 1.0)).mean())

    res = minimize_scalar(loss, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def apply_temperature(probs: FloatArray, t: float) -> FloatArray:
    return softmax(_log(probs) / t)


def _replace_top(probs: FloatArray, new_conf: FloatArray) -> FloatArray:
    """Set the top-label probability to `new_conf`, rescale the rest, keep the argmax."""
    p = probs.copy()
    n = len(p)
    top = p.argmax(axis=1)
    old = p[np.arange(n), top]
    others = p.copy()
    others[np.arange(n), top] = 0.0
    second = others.max(axis=1)
    rest = np.clip(1.0 - old, EPS, None)
    # keep the argmax: new >= second * (1 - new) / (1 - old)  <=>  new >= second / (rest + second)
    lower = np.maximum(second / (rest + second) + 1e-6, 1.0 / p.shape[1])
    new = np.clip(new_conf, lower, 1.0 - EPS)
    scaled = others * ((1.0 - new) / rest)[:, None]
    scaled[np.arange(n), top] = new
    out: FloatArray = scaled / scaled.sum(axis=1, keepdims=True)
    return out


def fit_platt(probs: FloatArray, y: NDArray[np.int64]) -> tuple[float, float]:
    """Logistic regression of correctness on logit(top-label confidence)."""
    from sklearn.linear_model import LogisticRegression

    conf = np.clip(probs.max(axis=1), EPS, 1 - EPS)
    correct = (probs.argmax(axis=1) == y).astype(int)
    x = np.log(conf / (1 - conf)).reshape(-1, 1)
    if correct.min() == correct.max():  # degenerate fold: all right or all wrong
        return 0.0, float(np.log((correct.mean() + EPS) / (1 - correct.mean() + EPS)))
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(x, correct)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(probs: FloatArray, params: tuple[float, float]) -> FloatArray:
    a, b = params
    conf = np.clip(probs.max(axis=1), EPS, 1 - EPS)
    new = 1.0 / (1.0 + np.exp(-(a * np.log(conf / (1 - conf)) + b)))
    return _replace_top(probs, new)


def fit_isotonic(probs: FloatArray, y: NDArray[np.int64]) -> object:
    from sklearn.isotonic import IsotonicRegression

    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(conf, correct)


def apply_isotonic(probs: FloatArray, model: object) -> FloatArray:
    new = model.predict(probs.max(axis=1))  # type: ignore[attr-defined]
    return _replace_top(probs, np.asarray(new, dtype=float))


METHODS: dict[str, tuple[Callable[..., object], Callable[..., FloatArray]]] = {
    "temperature": (fit_temperature, apply_temperature),
    "platt": (fit_platt, apply_platt),
    "isotonic": (fit_isotonic, apply_isotonic),
}


def cross_fit(
    method: str, probs: FloatArray, y: NDArray[np.int64], folds: int = 2, seed: int = 0
) -> FloatArray:
    fit, apply = METHODS[method]
    p = np.asarray(probs, dtype=float)
    yy = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(len(yy)) % folds
    out = np.empty_like(p)
    for k in range(folds):
        test = fold_of == k
        train = ~test
        if train.sum() < 2:
            out[test] = p[test]
            continue
        params = fit(p[train], yy[train])
        out[test] = apply(p[test], params)
    return out
