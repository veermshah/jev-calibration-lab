import math

import numpy as np
import pytest

from jevlab import metrics as M

P = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]])
Y = np.array([0, 1, 1, 1])
# top-label: conf = .9 .6 .7 .8, correct = 1 0 1 1


def test_accuracy():
    assert M.accuracy(P, Y) == 0.75


def test_ece_equal_width_hand_computed():
    # every item lands in its own bin: (0.1 + 0.6 + 0.3 + 0.2) / 4
    assert M.ece(P, Y, n_bins=10) == pytest.approx(0.3)


def test_ece_adaptive_hand_computed():
    # bins {.6,.7}: acc .5 conf .65 ; {.8,.9}: acc 1 conf .85 -> 0.5*.15 + 0.5*.15
    assert M.ece(P, Y, n_bins=2, scheme="adaptive") == pytest.approx(0.15)


def test_ece_single_bin_is_gap_between_mean_conf_and_acc():
    assert M.ece(P, Y, n_bins=1) == pytest.approx(abs(0.75 - 0.75))


def test_ece_perfectly_calibrated_is_zero():
    p = np.array([[0.5, 0.5]] * 4)
    y = np.array([0, 1, 0, 1])
    # argmax ties go to class 0, so 2/4 correct at confidence 0.5
    assert M.ece(p, y) == pytest.approx(0.0)


def test_bin_edges_are_right_closed():
    bins = M.reliability_bins([0.0, 0.1, 0.2, 1.0], [1, 1, 0, 1], n_bins=10)
    counts = {round(b["lo"], 1): b["count"] for b in bins}
    assert counts == {0.0: 2.0, 0.1: 1.0, 0.9: 1.0}


def test_brier_hand_computed():
    # .02 + .72 + .18 + .08 = 1.0 -> mean .25
    assert M.brier(P, Y) == pytest.approx(0.25)


def test_nll_hand_computed():
    expected = -(math.log(0.9) + math.log(0.4) + math.log(0.7) + math.log(0.8)) / 4
    assert M.nll(P, Y) == pytest.approx(expected)


def test_nll_clips_zero_probability():
    assert math.isfinite(M.nll(np.array([[1.0, 0.0]]), np.array([1])))


def test_multiclass_brier():
    p = np.array([[0.5, 0.3, 0.2]])
    assert M.brier(p, np.array([1])) == pytest.approx(0.25 + 0.49 + 0.04)


def test_risk_coverage_and_aurc():
    conf, correct = M.top_label(P, Y)
    cov, risk = M.risk_coverage(conf, correct)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0, 0, 0, 0.25])
    assert M.aurc(conf, correct) == pytest.approx(0.0625)
    assert M.accuracy_at_coverage(conf, correct, 0.75) == 1.0


def test_percentile_and_cost():
    assert M.percentile([10, 20, 30, 40, 50], 50) == 30
    assert M.percentile(list(range(101)), 95) == pytest.approx(95)
    assert M.cost_per_1k([0.001, 0.003]) == pytest.approx(2.0)


def test_mcnemar_exact():
    r = M.mcnemar([1, 1, 0, 0, 1], [0, 0, 0, 1, 1])
    assert (r["a_only"], r["b_only"]) == (2, 1)
    assert r["p_value"] == pytest.approx(1.0)
    r = M.mcnemar([1] * 10, [0] * 10)
    assert r["p_value"] == pytest.approx(2 * 0.5**10)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], size=200)
    y = rng.integers(0, 3, 200)
    ci = M.bootstrap_ci(p, y, n_boot=200, seed=1)
    point = M.compute(p, y).as_dict()
    for m in M.CORE_METRICS:
        lo, hi = ci[m]
        assert lo <= point[m] <= hi or m == "ece"  # ECE is biased upward under resampling


def test_paired_bootstrap_detects_clear_difference():
    y = np.array([0, 1] * 100)
    good = np.where(y[:, None] == np.arange(2), 0.9, 0.1)
    bad = np.full((200, 2), 0.5)
    r = M.paired_bootstrap(M.brier, bad, good, y, n_boot=300)
    assert r["diff"] > 0 and r["lo"] > 0 and r["p_value"] < 0.01


def test_run_to_run():
    y = np.array([0, 1, 0])
    a = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    b = np.array([[0.7, 0.3], [0.4, 0.6], [0.4, 0.6]])
    r = M.run_to_run([a, b], y)
    assert r["repeats"] == 2
    assert r["flip_rate"] == pytest.approx(1 / 3)
    assert r["gold_prob_std"] > 0
