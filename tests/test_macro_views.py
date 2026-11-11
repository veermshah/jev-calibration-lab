"""The "All datasets" (macro) view must aggregate across datasets, not show the first one."""

import copy
import json

import numpy as np

from jevlab import metrics as M
from jevlab.report import (
    add_macro_views,
    average_cascade,
    average_risk_coverage,
    pool_reliability,
)
from jevlab.site import render_site


def _curve(conf, correct):
    cov, risk = M.risk_coverage(conf, correct)
    return [[float(c), float(r)] for c, r in zip(cov, risk, strict=True)]


def _two_dataset_summary():
    rng = np.random.default_rng(0)
    # Dataset A: confident and right. Dataset B: unsure and often wrong.
    conf_a, ok_a = rng.uniform(0.85, 1.0, 200), rng.random(200) < 0.95
    conf_b, ok_b = rng.uniform(0.3, 0.6, 100), rng.random(100) < 0.4

    def cascade(acc):
        pts = [
            {
                "tau": t,
                "always_escalate": False,
                "escalation_rate": t,
                "accuracy": acc + 0.1 * t,
                "ece": 0.05,
                "cost_per_1k": 1.0 + t,
                "mean_latency_ms": 100.0,
            }
            for t in (0.0, 0.5, 1.0)
        ]
        return {"first": "jev", "second": "sonnet", "synthetic": False, "points": pts}

    summary = {
        "arms": {"jev": {"model": "m", "synthetic": False}},
        "datasets": {"a": {"title": "A", "n": 200, "k": 2}, "b": {"title": "B", "n": 100, "k": 2}},
        "metrics": [],
        "reliability": {
            "jev": {
                "a": M.reliability_bins(conf_a, ok_a, n_bins=10),
                "b": M.reliability_bins(conf_b, ok_b, n_bins=10),
            }
        },
        "risk_coverage": {"jev": {"a": _curve(conf_a, ok_a), "b": _curve(conf_b, ok_b)}},
        "recalibration": [],
        "cascade": {"a": cascade(0.9), "b": cascade(0.5)},
    }
    pooled_truth = M.reliability_bins(
        np.concatenate([conf_a, conf_b]), np.concatenate([ok_a, ok_b]), n_bins=10
    )
    return summary, pooled_truth


def test_pooled_reliability_equals_binning_all_items_together():
    summary, truth = _two_dataset_summary()
    pooled = pool_reliability(list(summary["reliability"]["jev"].values()))
    assert len(pooled) == len(truth)
    for p, t in zip(pooled, truth, strict=True):
        assert p["count"] == t["count"]
        assert np.isclose(p["conf"], t["conf"]) and np.isclose(p["acc"], t["acc"])


def test_macro_risk_coverage_is_mean_of_dataset_curves():
    summary, _ = _two_dataset_summary()
    a, b = summary["risk_coverage"]["jev"]["a"], summary["risk_coverage"]["jev"]["b"]
    avg = average_risk_coverage([a, b])
    assert avg[-1][0] == 1.0
    assert np.isclose(avg[-1][1], (a[-1][1] + b[-1][1]) / 2)
    # differs from either single dataset, i.e. it is not dataset A shown as "macro"
    assert not np.isclose(avg[-1][1], a[-1][1]) and not np.isclose(avg[-1][1], b[-1][1])


def test_macro_cascade_averages_per_tau_and_rejects_mismatched_grids():
    summary, _ = _two_dataset_summary()
    c = average_cascade([summary["cascade"]["a"], summary["cascade"]["b"]])
    assert c is not None
    assert [p["accuracy"] for p in c["points"]] == [0.7, 0.75, 0.8]
    bad = copy.deepcopy(summary["cascade"]["b"])
    bad["points"] = bad["points"][:2]
    assert average_cascade([summary["cascade"]["a"], bad]) is None


def test_site_macro_view_uses_aggregated_data_not_first_dataset():
    summary, truth = _two_dataset_summary()
    before = copy.deepcopy(summary)
    html = render_site(summary)
    assert summary == before  # input untouched
    data = json.loads(
        html.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    )
    macro_bins = data["reliability"]["jev"]["macro"]
    assert macro_bins != data["reliability"]["jev"]["a"]
    assert [b["count"] for b in macro_bins] == [t["count"] for t in truth]
    assert "macro" in data["risk_coverage"]["jev"] and "macro" in data["cascade"]
    # the old fallback that silently picked the first dataset is gone
    assert "Object.keys(S.datasets)[0]" not in html and "Object.keys(S.cascade)[0]" not in html


def test_add_macro_views_is_idempotent():
    summary, _ = _two_dataset_summary()
    once = add_macro_views(copy.deepcopy(summary))
    twice = add_macro_views(copy.deepcopy(once))
    assert once == twice
