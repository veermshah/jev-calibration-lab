"""Runs arms over datasets with bounded concurrency and writes one JSONL row per prediction."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .arms.base import DecisionModel
from .budget import Budget, BudgetExceeded
from .schema import Decision, Prediction


@dataclass
class RunSummary:
    rows: int = 0
    errors: int = 0
    budget_stopped: bool = False
    skipped: dict[str, int] = field(default_factory=dict)


def plan_tasks(items: list[Decision], repeats: int, repeat_subset: int) -> list[tuple[Decision, int]]:
    """Repeat 0 covers every item; repeats 1..R-1 cover the first `repeat_subset` items."""
    tasks = [(d, 0) for d in items]
    for r in range(1, repeats):
        tasks.extend((d, r) for d in items[:repeat_subset])
    return tasks


def record(arm: DecisionModel, d: Decision, repeat: int, pred: Prediction) -> dict[str, Any]:
    return {
        "arm": arm.name,
        "model": arm.model_id,
        "dataset": d.dataset,
        "item_id": d.id,
        "repeat": repeat,
        "choices": d.choice_ids,
        "gold": d.gold,
        "probs": pred.vector(d.choice_ids),
        "latency_ms": pred.latency_ms,
        "cost_usd": pred.cost_usd,
        "input_tokens": pred.input_tokens,
        "output_tokens": pred.output_tokens,
        "cached": pred.cached,
        "synthetic": pred.synthetic,
        "error": pred.error,
    }


def run_arm_on_dataset(
    arm: DecisionModel,
    items: list[Decision],
    out: Path,
    budget: Budget,
    repeats: int = 5,
    repeat_subset: int = 50,
    log: Callable[[str], None] = print,
) -> RunSummary:
    summary = RunSummary()
    if not items:
        return summary
    arm.prepare(items[0].dataset)
    tasks = plan_tasks(items, repeats, repeat_subset)
    lock = threading.Lock()
    stop = threading.Event()
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("a", encoding="utf-8") as f:

        def work(d: Decision, r: int) -> None:
            if stop.is_set():
                return
            try:
                pred = arm.predict(d, r)
            except BudgetExceeded:
                stop.set()
                return
            row = record(arm, d, r, pred)
            with lock:
                f.write(json.dumps(row) + "\n")
                summary.rows += 1
                summary.errors += int(pred.error is not None)

        with ThreadPoolExecutor(max_workers=max(1, arm.concurrency)) as pool:
            futures = [pool.submit(work, d, r) for d, r in tasks]
            done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            for fut in done:
                exc = fut.exception()
                if exc is not None:
                    stop.set()
                    raise exc
    if stop.is_set():
        summary.budget_stopped = True
        summary.skipped[items[0].dataset] = len(tasks) - summary.rows
        log(f"  budget cap hit: {len(tasks) - summary.rows} calls skipped")
    return summary


def run(
    arms: list[DecisionModel],
    datasets: dict[str, list[Decision]],
    out_dir: Path,
    budget: Budget,
    repeats: int = 5,
    repeat_subset: int = 50,
    log: Callable[[str], None] = print,
) -> RunSummary:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    if pred_path.exists():
        pred_path.unlink()  # a run rewrites its predictions; the response cache keeps reruns free
    total = RunSummary()
    for arm in arms:
        for name, items in datasets.items():
            tag = " [synthetic]" if arm.synthetic else ""
            log(f"{arm.name:8s} {name:10s} {len(items)} items{tag}")
            s = run_arm_on_dataset(arm, items, pred_path, budget, repeats, repeat_subset, log)
            total.rows += s.rows
            total.errors += s.errors
            total.skipped.update({f"{arm.name}/{k}": v for k, v in s.skipped.items()})
            total.budget_stopped |= s.budget_stopped
    meta = {
        "arms": [
            {"name": a.name, "model": a.model_id, "synthetic": a.synthetic} for a in arms
        ],
        "datasets": {k: len(v) for k, v in datasets.items()},
        "repeats": repeats,
        "repeat_subset": repeat_subset,
        "budget_cap_usd": budget.cap_usd,
        "spent_usd": budget.spent_usd,
        "spent_is_projected": any(a.synthetic for a in arms),
        "budget_stopped": total.budget_stopped,
        "rows": total.rows,
        "errors": total.errors,
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2))
    return total


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
