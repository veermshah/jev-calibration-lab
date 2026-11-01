import numpy as np
import pytest

from jevlab import metrics as M
from jevlab.cascade import sweep
from jevlab.recalibration import apply_temperature, cross_fit, fit_temperature


def test_cascade_hand_computed():
    p1 = np.array([[0.9, 0.1], [0.55, 0.45], [0.6, 0.4], [0.2, 0.8]])  # conf .9 .55 .6 .8
    p2 = np.array([[0.2, 0.8], [0.1, 0.9], [0.95, 0.05], [0.3, 0.7]])
    y = np.array([0, 1, 0, 1])
    rows = sweep(p1, p2, y, [1, 1, 1, 1], [10, 10, 10, 10], [100] * 4, [1000] * 4, taus=[0.0, 0.7])
    by_tau = {(r["tau"], r["always_escalate"]): r for r in rows}

    never = by_tau[(0.0, False)]
    assert never["escalation_rate"] == 0
    assert never["accuracy"] == pytest.approx(0.75)  # item 2 wrong
    assert never["cost_per_1k"] == pytest.approx(1000)
    assert never["mean_latency_ms"] == pytest.approx(100)

    mid = by_tau[(0.7, False)]  # escalate items with conf .55 and .6
    assert mid["escalation_rate"] == pytest.approx(0.5)
    assert mid["accuracy"] == pytest.approx(1.0)  # second arm fixes item 2, keeps item 3 right
    assert mid["cost_per_1k"] == pytest.approx((1 + 0.5 * 10) * 1000)
    assert mid["mean_latency_ms"] == pytest.approx(100 + 0.5 * 1000)

    always = by_tau[(1.0, True)]
    assert always["escalation_rate"] == 1.0
    assert always["accuracy"] == pytest.approx(0.75)  # second arm alone gets item 1 wrong


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    true_p = rng.uniform(0.5, 0.95, n)
    correct = rng.random(n) < true_p
    pred = np.where(correct, y, 1 - y)
    logit = np.log(true_p / (1 - true_p)) * 3.0  # sharpen: overconfident
    conf = 1 / (1 + np.exp(-logit))
    probs = np.zeros((n, 2))
    probs[np.arange(n), pred] = conf
    probs[np.arange(n), 1 - pred] = 1 - conf
    t = fit_temperature(probs, y)
    assert t == pytest.approx(3.0, rel=0.2)
    assert M.ece(apply_temperature(probs, t), y) < M.ece(probs, y) / 2


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic"])
def test_cross_fit_keeps_shape_and_argmax(method):
    rng = np.random.default_rng(1)
    probs = rng.dirichlet([1, 1, 1], size=300)
    y = rng.integers(0, 3, 300)
    out = cross_fit(method, probs, y, folds=2, seed=0)
    assert out.shape == probs.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert (out.argmax(axis=1) == probs.argmax(axis=1)).all()
