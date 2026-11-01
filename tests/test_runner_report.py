import json

from jevlab.arms.jev import JevModel
from jevlab.arms.synthetic import PROFILES, jev_transport
from jevlab.budget import Budget
from jevlab.report import render_markdown, summarise
from jevlab.runner import load_predictions, plan_tasks, run
from jevlab.schema import Decision


def items(n=30):
    return [
        Decision(
            f"sst2-{i}",
            "sst2",
            f"text {i}",
            "q?",
            {"negative": "", "positive": ""},
            "positive" if i % 2 else "negative",
        )
        for i in range(n)
    ]


def test_plan_tasks_repeats_only_the_subset():
    tasks = plan_tasks(items(10), repeats=3, repeat_subset=4)
    assert len(tasks) == 10 + 2 * 4
    assert sum(1 for _, r in tasks if r == 0) == 10


def test_run_writes_rows_and_meta(tmp_path, prices, cache):
    budget = Budget(15)
    arm = JevModel(
        "jev", "jev-latest", jev_transport(PROFILES["jev"]), prices, cache, budget, synthetic=True
    )
    s = run(
        [arm], {"sst2": items()}, tmp_path, budget, repeats=2, repeat_subset=5, log=lambda m: None
    )
    rows = load_predictions(tmp_path / "predictions.jsonl")
    assert s.rows == len(rows) == 35
    assert all(r["synthetic"] for r in rows)
    meta = json.loads((tmp_path / "run.json").read_text())
    assert meta["spent_is_projected"] is True and meta["rows"] == 35


def test_run_stops_at_budget_cap(tmp_path, prices, cache):
    per_call = []

    def transport(request, decision, repeat):
        per_call.append(1)
        return {
            "answers": {"answer": {"probabilities": {"negative": 0.5, "positive": 0.5}}},
            "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
        }, 10.0

    budget = Budget(0.2)  # each call really costs $0.042, estimates are tiny
    arm = JevModel("jev", "jev-latest", transport, prices, cache, budget, concurrency=1)
    s = run([arm], {"sst2": items(20)}, tmp_path, budget, repeats=1, log=lambda m: None)
    assert s.budget_stopped
    assert len(per_call) < 20
    assert budget.spent_usd <= 0.2 + 0.042


def test_summary_and_markdown_label_synthetic_arms(tmp_path, prices, cache):
    budget = Budget(15)
    arms = [
        JevModel(
            name, "jev-latest", jev_transport(PROFILES[name]), prices, cache, budget, synthetic=True
        )
        for name in ("jev", "sonnet")
    ]
    run(arms, {"sst2": items()}, tmp_path, budget, repeats=2, repeat_subset=10, log=lambda m: None)
    rows = load_predictions(tmp_path / "predictions.jsonl")
    summary = summarise(rows, n_boot=50)
    assert {m["dataset"] for m in summary["metrics"]} == {"sst2", "macro"}
    assert summary["cascade"]["sst2"]["synthetic"] is True
    assert len(summary["recalibration"]) == 2 * 3
    assert summary["variance"][0]["repeats"] == 2
    md = render_markdown(summary, [])
    assert "Synthetic demo output" in md
    assert "jev (synthetic)" in md


def test_site_is_self_contained_and_badges_synthetic(tmp_path, prices, cache):
    from jevlab.site import render_site

    budget = Budget(15)
    arm = JevModel(
        "jev", "jev-latest", jev_transport(PROFILES["jev"]), prices, cache, budget, synthetic=True
    )
    run([arm], {"sst2": items()}, tmp_path, budget, repeats=1, log=lambda m: None)
    summary = summarise(load_predictions(tmp_path / "predictions.jsonl"), n_boot=20)
    summary["metrics"][0]["nll"] = float("nan")
    html = render_site(summary)
    assert "Synthetic demo output" in html and "badge" in html
    assert "NaN" not in html.split('<script id="data"')[1].split("</script>")[0]
    assert "<script src=" not in html and "<link" not in html  # no external assets
