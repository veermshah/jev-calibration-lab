"""Turns a run's predictions into summary.json, report.md and figures/."""

from __future__ import annotations

import datetime as dt
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import metrics as M
from .cascade import sweep
from .datasets import DATASETS
from .recalibration import METHODS, cross_fit

SYNTHETIC_BANNER = (
    "> **Synthetic demo output.** Arms marked *synthetic* were produced by `--dry-run`: "
    "invented generators that mimic each provider's reply format. They are **not** "
    "measurements of TypeSafe Jev or Claude and say nothing about how those models "
    "behave. Only arms without the marker are real measurements."
)


def label(arm: str, synthetic: bool) -> str:
    return f"{arm} (synthetic)" if synthetic else arm


# --------------------------------------------------------------------------- summarise


def _group(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    g: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[(r["arm"], r["dataset"])].append(r)
    return g


def _arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    probs = np.array([r["probs"] for r in rows], dtype=float)
    y = np.array([r["choices"].index(r["gold"]) for r in rows], dtype=int)
    return probs, y


def summarise(
    rows: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    cascade_pair: tuple[str, str] = ("jev", "sonnet"),
) -> dict[str, Any]:
    meta = meta or {}
    groups = _group(rows)
    arms: dict[str, dict[str, Any]] = {}
    for r in rows:
        arms.setdefault(r["arm"], {"model": r["model"], "synthetic": bool(r["synthetic"])})
    datasets = sorted(
        {r["dataset"] for r in rows}, key=lambda d: list(DATASETS).index(d) if d in DATASETS else 99
    )

    out: dict[str, Any] = {
        "generated": dt.date.today().isoformat(),
        "run": meta,
        "arms": arms,
        "datasets": {},
        "metrics": [],
        "reliability": defaultdict(dict),
        "risk_coverage": defaultdict(dict),
        "recalibration": [],
        "variance": [],
        "paired": [],
        "cascade": {},
    }
    main: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (arm, ds), rs in groups.items():
        r0 = sorted((r for r in rs if r["repeat"] == 0), key=lambda r: r["item_id"])
        if not r0:
            continue
        main[(arm, ds)] = r0
        out["datasets"].setdefault(
            ds,
            {
                "title": DATASETS[ds].title if ds in DATASETS else ds,
                "n": len(r0),
                "k": len(r0[0]["choices"]),
            },
        )

    for (arm, ds), r0 in sorted(main.items()):
        probs, y = _arrays(r0)
        conf, correct = M.top_label(probs, y)
        ms = M.compute(probs, y).as_dict()
        ci = M.bootstrap_ci(probs, y, n_boot=n_boot, seed=seed) if n_boot else {}
        lat = [r["latency_ms"] for r in r0]
        out["metrics"].append(
            {
                "arm": arm,
                "dataset": ds,
                "synthetic": arms[arm]["synthetic"],
                **ms,
                "acc_at_80": M.accuracy_at_coverage(conf, correct, 0.8),
                "p50_ms": M.percentile(lat, 50),
                "p95_ms": M.percentile(lat, 95),
                "cost_per_1k": M.cost_per_1k([r["cost_usd"] for r in r0]),
                "errors": sum(1 for r in r0 if r["error"]),
                "ci": {k: list(v) for k, v in ci.items()},
            }
        )
        out["reliability"][arm][ds] = M.reliability_bins(conf, correct, n_bins=10)
        cov, risk = M.risk_coverage(conf, correct)
        step = max(1, len(cov) // 60)
        pts = list(zip(cov[::step].tolist(), risk[::step].tolist(), strict=True))
        if pts[-1][0] != 1.0:
            pts.append((1.0, float(risk[-1])))
        out["risk_coverage"][arm][ds] = pts

        for method in METHODS:
            rp = cross_fit(method, probs, y, folds=2, seed=seed)
            out["recalibration"].append(
                {
                    "arm": arm,
                    "dataset": ds,
                    "synthetic": arms[arm]["synthetic"],
                    "method": method,
                    "ece_before": ms["ece"],
                    "ece_after": M.ece(rp, y),
                    "nll_before": ms["nll"],
                    "nll_after": M.nll(rp, y),
                    "brier_before": ms["brier"],
                    "brier_after": M.brier(rp, y),
                }
            )

        reps: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        for r in groups[(arm, ds)]:
            reps[r["repeat"]][r["item_id"]] = r
        if len(reps) > 1:
            shared = sorted(set.intersection(*(set(v) for v in reps.values())))
            if shared:
                rep_probs = [np.array([reps[k][i]["probs"] for i in shared]) for k in sorted(reps)]
                y_sub = np.array([reps[0][i]["choices"].index(reps[0][i]["gold"]) for i in shared])
                out["variance"].append(
                    {
                        "arm": arm,
                        "dataset": ds,
                        "synthetic": arms[arm]["synthetic"],
                        "items": len(shared),
                        **M.run_to_run(rep_probs, y_sub),
                    }
                )

    # macro averages across datasets
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in out["metrics"]:
        by_arm[m["arm"]].append(m)
    for arm, ms_list in by_arm.items():
        keys = [
            "accuracy",
            "ece",
            "ece_adaptive",
            "brier",
            "nll",
            "aurc",
            "acc_at_80",
            "p50_ms",
            "p95_ms",
            "cost_per_1k",
        ]
        out["metrics"].append(
            {
                "arm": arm,
                "dataset": "macro",
                "synthetic": arms[arm]["synthetic"],
                "n": sum(m["n"] for m in ms_list),
                **{k: float(np.mean([m[k] for m in ms_list])) for k in keys},
                "errors": sum(m["errors"] for m in ms_list),
                "ci": {},
            }
        )

    # paired comparisons on shared items
    for ds in datasets:
        present = sorted(a for a in arms if (a, ds) in main)
        for a, b in itertools.combinations(present, 2):
            ra = {r["item_id"]: r for r in main[(a, ds)]}
            rb = {r["item_id"]: r for r in main[(b, ds)]}
            ids = sorted(set(ra) & set(rb))
            if len(ids) < 5:
                continue
            pa, y = _arrays([ra[i] for i in ids])
            pb, _ = _arrays([rb[i] for i in ids])
            _, ca = M.top_label(pa, y)
            _, cb = M.top_label(pb, y)
            nb = max(200, n_boot)
            out["paired"].append(
                {
                    "dataset": ds,
                    "a": a,
                    "b": b,
                    "synthetic": arms[a]["synthetic"] or arms[b]["synthetic"],
                    "n": len(ids),
                    "accuracy": M.paired_bootstrap(M.accuracy, pa, pb, y, n_boot=nb, seed=seed),
                    "brier": M.paired_bootstrap(M.brier, pa, pb, y, n_boot=nb, seed=seed),
                    "ece": M.paired_bootstrap(M.ece, pa, pb, y, n_boot=nb, seed=seed),
                    "mcnemar": M.mcnemar(ca, cb),
                }
            )

    first, second = cascade_pair
    for ds in datasets:
        if (first, ds) in main and (second, ds) in main:
            r1 = {r["item_id"]: r for r in main[(first, ds)]}
            r2 = {r["item_id"]: r for r in main[(second, ds)]}
            ids = sorted(set(r1) & set(r2))
            p1, y = _arrays([r1[i] for i in ids])
            p2, _ = _arrays([r2[i] for i in ids])
            out["cascade"][ds] = {
                "first": first,
                "second": second,
                "synthetic": arms[first]["synthetic"] or arms[second]["synthetic"],
                "points": sweep(
                    p1,
                    p2,
                    y,
                    [r1[i]["cost_usd"] for i in ids],
                    [r2[i]["cost_usd"] for i in ids],
                    [r1[i]["latency_ms"] for i in ids],
                    [r2[i]["latency_ms"] for i in ids],
                ),
            }
    out["reliability"] = dict(out["reliability"])
    out["risk_coverage"] = dict(out["risk_coverage"])
    return out


# --------------------------------------------------------------------------- macro views

MACRO = "macro"
_CASCADE_AVG_KEYS = ("escalation_rate", "accuracy", "ece", "cost_per_1k", "mean_latency_ms")


def pool_reliability(per_dataset: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    """Pool equal-width reliability bins across datasets, weighting each bin by its count.

    Because every dataset uses the same bin edges, this equals binning all items from all
    datasets together: the pooled diagram is the reliability of the union of the datasets.
    """
    acc: dict[tuple[float, float], list[float]] = {}
    for bins in per_dataset:
        for b in bins:
            key = (round(float(b["lo"]), 9), round(float(b["hi"]), 9))
            s = acc.setdefault(key, [0.0, 0.0, 0.0])
            n = float(b["count"])
            s[0] += n
            s[1] += n * float(b["conf"])
            s[2] += n * float(b["acc"])
    return [
        {"lo": lo, "hi": hi, "count": n, "conf": c / n, "acc": a / n}
        for (lo, hi), (n, c, a) in sorted(acc.items())
        if n > 0
    ]


def average_risk_coverage(curves: list[list[list[float]]], n_points: int = 60) -> list[list[float]]:
    """Macro-average risk-coverage curves: interpolate each onto a shared coverage grid
    and take the unweighted mean risk, so every dataset counts equally (like macro AURC).
    """
    curves = [c for c in curves if c]
    if not curves:
        return []
    start = max(float(c[0][0]) for c in curves)
    grid = np.linspace(start, 1.0, n_points)
    risks = [np.interp(grid, [float(p[0]) for p in c], [float(p[1]) for p in c]) for c in curves]
    mean = np.mean(risks, axis=0)
    return [[float(x), float(r)] for x, r in zip(grid, mean, strict=True)]


def average_cascade(per_dataset: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Average cascade sweeps across datasets point by point (same tau grid required)."""
    if not per_dataset:
        return None
    first = per_dataset[0]
    if any(c["first"] != first["first"] or c["second"] != first["second"] for c in per_dataset):
        return None
    taus = [[(p["tau"], p.get("always_escalate", False)) for p in c["points"]] for c in per_dataset]
    if any(t != taus[0] for t in taus):
        return None
    points = []
    for i, (tau, always) in enumerate(taus[0]):
        pt: dict[str, Any] = {"tau": tau, "always_escalate": always}
        for k in _CASCADE_AVG_KEYS:
            vals = [c["points"][i].get(k) for c in per_dataset]
            nums = [float(v) for v in vals if v is not None]
            pt[k] = float(np.mean(nums)) if nums else None
        points.append(pt)
    return {
        "first": first["first"],
        "second": first["second"],
        "synthetic": any(c["synthetic"] for c in per_dataset),
        "points": points,
    }


def add_macro_views(summary: dict[str, Any]) -> dict[str, Any]:
    """Add "macro" entries for the all-datasets view of the charts (in place; idempotent).

    - reliability: bins pooled over every item in every dataset (count-weighted).
    - risk_coverage: per-dataset curves averaged on a shared coverage grid (macro).
    - cascade: per-dataset sweeps averaged at each threshold tau (macro).
    """
    datasets = [d for d in summary.get("datasets", {}) if d != MACRO]
    for per_arm in summary.get("reliability", {}).values():
        if MACRO not in per_arm:
            pooled = pool_reliability([per_arm[d] for d in datasets if d in per_arm])
            if pooled:
                per_arm[MACRO] = pooled
    for per_arm in summary.get("risk_coverage", {}).values():
        if MACRO not in per_arm:
            avg = average_risk_coverage([per_arm[d] for d in datasets if d in per_arm])
            if avg:
                per_arm[MACRO] = avg
    cascade = summary.get("cascade")
    if cascade is not None and MACRO not in cascade:
        avg_c = average_cascade([cascade[d] for d in datasets if d in cascade])
        if avg_c is not None:
            cascade[MACRO] = avg_c
    return summary


def json_safe(obj: Any) -> Any:
    """Replace NaN/inf with None so the file is strict JSON (browsers reject NaN)."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, np.floating | np.integer):
        return json_safe(obj.item())
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- figures


def _colors(arms: list[str]) -> dict[str, str]:
    palette = ["#2563eb", "#d97706", "#7c3aed", "#059669", "#dc2626", "#0891b2", "#4b5563"]
    return {a: palette[i % len(palette)] for i, a in enumerate(arms)}


def make_figures(summary: dict[str, Any], fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    arms = list(summary["arms"])
    datasets = list(summary["datasets"])
    colors = _colors(arms)
    syn = {a: summary["arms"][a]["synthetic"] for a in arms}
    any_syn = any(syn.values())
    note = (
        "Arms marked (synthetic) are dry-run demo output, not real measurements." if any_syn else ""
    )
    written: list[str] = []

    def finish(fig: Any, name: str, title: str) -> None:
        fig.suptitle(title + (f"\n{note}" if note else ""), fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=130)
        plt.close(fig)
        written.append(name)

    # reliability diagrams
    fig, axes = plt.subplots(
        len(datasets), len(arms), figsize=(2.6 * len(arms), 2.5 * len(datasets)), squeeze=False
    )
    for i, ds in enumerate(datasets):
        for j, arm in enumerate(arms):
            ax = axes[i][j]
            ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#9ca3af")
            bins = summary["reliability"].get(arm, {}).get(ds, [])
            for b in bins:
                ax.bar(
                    (b["lo"] + b["hi"]) / 2,
                    b["acc"],
                    width=b["hi"] - b["lo"],
                    color=colors[arm],
                    alpha=0.75,
                    edgecolor="white",
                )
                ax.plot((b["lo"] + b["hi"]) / 2, b["conf"], marker="_", color="black", ms=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(label(arm, syn[arm]), fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{ds}\naccuracy", fontsize=8)
            if i == len(datasets) - 1:
                ax.set_xlabel("confidence", fontsize=8)
            if syn[arm]:
                ax.set_facecolor("#fff7ed")
    finish(fig, "reliability.png", "Reliability diagrams (top label, 10 equal-width bins)")

    # risk-coverage
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 3.0), squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[0][i]
        for arm in arms:
            pts = summary["risk_coverage"].get(arm, {}).get(ds)
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.plot(
                    xs,
                    ys,
                    color=colors[arm],
                    ls="--" if syn[arm] else "-",
                    label=label(arm, syn[arm]),
                )
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("coverage", fontsize=8)
        ax.set_ylabel("risk (error rate)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7)
    finish(fig, "risk_coverage.png", "Selective prediction: risk vs coverage (lower is better)")

    # cascade
    if summary["cascade"]:
        cds = list(summary["cascade"])
        fig, axes = plt.subplots(1, len(cds), figsize=(3.2 * len(cds), 3.0), squeeze=False)
        for i, ds in enumerate(cds):
            c = summary["cascade"][ds]
            pts = c["points"]
            ax = axes[0][i]
            ax.plot(
                [p["cost_per_1k"] for p in pts],
                [p["accuracy"] for p in pts],
                marker="o",
                ms=3,
                color="#111827",
            )
            ax.scatter(
                [pts[0]["cost_per_1k"]],
                [pts[0]["accuracy"]],
                color=colors.get(c["first"], "C0"),
                zorder=3,
                label=f"{label(c['first'], syn.get(c['first'], False))} only",
            )
            ax.scatter(
                [pts[-1]["cost_per_1k"]],
                [pts[-1]["accuracy"]],
                color=colors.get(c["second"], "C1"),
                zorder=3,
                label=f"{label(c['second'], syn.get(c['second'], False))} only",
            )
            ax.set_xscale("symlog", linthresh=0.01)
            ax.set_title(ds, fontsize=9)
            ax.set_xlabel("$ per 1k decisions", fontsize=8)
            ax.set_ylabel("accuracy", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0][0].legend(fontsize=7)
        first = summary["cascade"][cds[0]]
        finish(
            fig,
            "cascade.png",
            f"Cascade: {first['first']} first, escalate to {first['second']} below tau",
        )

    # recalibration
    rec = summary["recalibration"]
    if rec:
        methods = list(METHODS)
        fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(arms), 3.2))
        width = 0.8 / (len(methods) + 1)
        for j, arm in enumerate(arms):
            rs = [r for r in rec if r["arm"] == arm]
            if not rs:
                continue
            before = np.mean([r["ece_before"] for r in rs if r["method"] == methods[0]])
            ax.bar(j, before, width, color="#9ca3af", label="raw" if j == 0 else None)
            for k, m in enumerate(methods):
                after = np.mean([r["ece_after"] for r in rs if r["method"] == m])
                ax.bar(
                    j + (k + 1) * width, after, width, color=f"C{k}", label=m if j == 0 else None
                )
        ax.set_xticks([j + width * len(methods) / 2 for j in range(len(arms))])
        ax.set_xticklabels([label(a, syn[a]) for a in arms], fontsize=8)
        ax.set_ylabel("ECE (macro over datasets)", fontsize=8)
        ax.legend(fontsize=7)
        finish(fig, "recalibration.png", "ECE before and after cross-fitted post-hoc recalibration")

    # cost vs latency
    macro = [m for m in summary["metrics"] if m["dataset"] == "macro"]
    if macro:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for m in macro:
            ax.scatter(
                m["p50_ms"],
                m["cost_per_1k"],
                s=60,
                color=colors[m["arm"]],
                marker="s" if syn[m["arm"]] else "o",
            )
            ax.annotate(
                f"{label(m['arm'], syn[m['arm']])}\nacc {m['accuracy']:.2f}, ECE {m['ece']:.3f}",
                (m["p50_ms"], m["cost_per_1k"]),
                fontsize=7,
                xytext=(5, 5),
                textcoords="offset points",
            )
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=0.001)
        ax.set_xlabel("p50 latency (ms)", fontsize=8)
        ax.set_ylabel("$ per 1k decisions", fontsize=8)
        finish(fig, "cost_latency.png", "Cost vs latency (macro over datasets)")
    return written


# --------------------------------------------------------------------------- markdown


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _ci(m: dict[str, Any], key: str, nd: int = 3) -> str:
    ci = m.get("ci", {}).get(key)
    if not ci:
        return _f(m[key], nd)
    return f"{_f(m[key], nd)} [{_f(ci[0], nd)}, {_f(ci[1], nd)}]"


def render_markdown(summary: dict[str, Any], figures: list[str]) -> str:
    arms = summary["arms"]
    syn_arms = [a for a, v in arms.items() if v["synthetic"]]
    real_arms = [a for a, v in arms.items() if not v["synthetic"]]
    run = summary.get("run", {})
    lines = ["# jev-calibration-lab report", ""]
    lines.append(f"Generated {summary['generated']}.")
    lines.append("")
    if syn_arms:
        lines += [SYNTHETIC_BANNER, ""]
        lines.append(f"- Real measurements: {', '.join(real_arms) or 'none'}")
        lines.append(f"- Synthetic (dry-run) arms: {', '.join(syn_arms)}")
        lines.append("")
    if run:
        spent = run.get("spent_usd", 0.0)
        kind = (
            "projected cost of the same calls made for real"
            if run.get("spent_is_projected")
            else "actual spend"
        )
        lines.append(
            f"Run: {run.get('rows', '?')} predictions, {run.get('repeats', '?')} repeats on a "
            f"{run.get('repeat_subset', '?')}-item subset, budget cap ${run.get('budget_cap_usd', 0):.2f}, "
            f"${spent:.4f} ({kind})."
        )
        lines.append("")

    lines += ["## Datasets", "", "| dataset | items | choices |", "|---|---:|---:|"]
    for info in summary["datasets"].values():
        lines.append(f"| {info['title']} | {info['n']} | {info['k']} |")
    lines.append("")

    lines += [
        "## Headline (macro average over datasets)",
        "",
        "| arm | accuracy | ECE | adaptive ECE | Brier | NLL | AURC | acc@80% cov | p50 ms | p95 ms | $/1k |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in [m for m in summary["metrics"] if m["dataset"] == "macro"]:
        lines.append(
            f"| {label(m['arm'], m['synthetic'])} | {_f(m['accuracy'])} | {_f(m['ece'])} | {_f(m['ece_adaptive'])} | "
            f"{_f(m['brier'])} | {_f(m['nll'])} | {_f(m['aurc'])} | {_f(m['acc_at_80'])} | {_f(m['p50_ms'], 1)} | "
            f"{_f(m['p95_ms'], 1)} | {_f(m['cost_per_1k'], 4)} |"
        )
    lines.append("")

    lines += ["## Per dataset (95% bootstrap CIs)", ""]
    for ds, info in summary["datasets"].items():
        lines += [
            f"### {info['title']}",
            "",
            "| arm | accuracy | ECE (15 bins) | Brier | NLL | AURC | p50 ms | $/1k | errors |",
            "|---|---|---|---|---|---|---:|---:|---:|",
        ]
        for m in [m for m in summary["metrics"] if m["dataset"] == ds]:
            lines.append(
                f"| {label(m['arm'], m['synthetic'])} | {_ci(m, 'accuracy')} | {_ci(m, 'ece')} | {_ci(m, 'brier')} | "
                f"{_ci(m, 'nll')} | {_ci(m, 'aurc')} | {_f(m['p50_ms'], 1)} | {_f(m['cost_per_1k'], 4)} | {m['errors']} |"
            )
        lines.append("")

    if summary["recalibration"]:
        lines += [
            "## Post-hoc recalibration",
            "",
            "ECE before and after each method, 2-fold cross-fitted on the evaluation sample "
            "(macro over datasets). If a cheap fix closes most of the gap, native calibration "
            "matters less than it sounds.",
            "",
            "| arm | raw ECE | temperature | Platt | isotonic | best NLL after |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for arm in arms:
            rs = [r for r in summary["recalibration"] if r["arm"] == arm]
            if not rs:
                continue

            def avg(method: str, key: str, rs: list[dict[str, Any]] = rs) -> float:
                return float(np.mean([r[key] for r in rs if r["method"] == method]))

            raw = avg("temperature", "ece_before")
            best_nll = min(avg(m, "nll_after") for m in METHODS)
            lines.append(
                f"| {label(arm, arms[arm]['synthetic'])} | {_f(raw)} | {_f(avg('temperature', 'ece_after'))} | "
                f"{_f(avg('platt', 'ece_after'))} | {_f(avg('isotonic', 'ece_after'))} | {_f(best_nll)} |"
            )
        lines.append("")

    if summary["variance"]:
        lines += [
            "## Run-to-run variance",
            "",
            "Same items sent repeatedly. Flip rate is the share of items whose top answer changes in any repeat.",
            "",
            "| arm | dataset | items | repeats | accuracy mean | accuracy std | ECE std | gold-prob std | flip rate |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for v in summary["variance"]:
            if v.get("repeats", 0) < 2:
                continue
            lines.append(
                f"| {label(v['arm'], v['synthetic'])} | {v['dataset']} | {v['items']} | {v['repeats']} | "
                f"{_f(v['accuracy_mean'])} | {_f(v['accuracy_std'])} | {_f(v['ece_std'])} | "
                f"{_f(v['gold_prob_std'])} | {_f(v['flip_rate'])} |"
            )
        lines.append("")

    if summary["cascade"]:
        lines += [
            "## Cascade",
            "",
            "First arm answers; items whose top probability is below tau are escalated. "
            "Each row is the cheapest tau whose accuracy is within one point of always escalating.",
            "",
            "| dataset | cascade | first-only acc | always-escalate acc | chosen tau | escalated | acc | $/1k | cost vs always-escalate |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for ds, c in summary["cascade"].items():
            pts = c["points"]
            target = pts[-1]["accuracy"] - 0.01
            ok = [p for p in pts if p["accuracy"] >= target]
            best = min(ok, key=lambda p: p["cost_per_1k"])
            ratio = (
                best["cost_per_1k"] / pts[-1]["cost_per_1k"]
                if pts[-1]["cost_per_1k"]
                else float("nan")
            )
            tag = " (synthetic)" if c["synthetic"] else ""
            lines.append(
                f"| {ds} | {c['first']} -> {c['second']}{tag} | {_f(pts[0]['accuracy'])} | {_f(pts[-1]['accuracy'])} | "
                f"{_f(best['tau'], 2)} | {_f(best['escalation_rate'], 2)} | {_f(best['accuracy'])} | "
                f"{_f(best['cost_per_1k'], 4)} | {ratio:.1%} |"
            )
        lines.append("")

    if summary["paired"]:
        lines += [
            "## Paired comparisons",
            "",
            "Differences are a minus b on the same items; CIs and p-values from a paired bootstrap, "
            "plus an exact McNemar test on correctness.",
            "",
            "| dataset | a vs b | accuracy diff [95% CI] | Brier diff [95% CI] | McNemar p |",
            "|---|---|---|---|---:|",
        ]
        for p in summary["paired"]:
            acc, br = p["accuracy"], p["brier"]
            tag = " (synthetic)" if p["synthetic"] else ""
            lines.append(
                f"| {p['dataset']} | {p['a']} vs {p['b']}{tag} | {acc['diff']:+.3f} [{acc['lo']:+.3f}, {acc['hi']:+.3f}] | "
                f"{br['diff']:+.3f} [{br['lo']:+.3f}, {br['hi']:+.3f}] | {_f(p['mcnemar']['p_value'], 4)} |"
            )
        lines.append("")

    if figures:
        lines += ["## Figures", ""]
        for name in figures:
            lines += [f"![{name}](figures/{name})", ""]
    return "\n".join(lines)


def write_report(
    pred_path: Path,
    out_dir: Path,
    n_boot: int = 1000,
    cascade_pair: tuple[str, str] = ("jev", "sonnet"),
) -> dict[str, Any]:
    from .runner import load_predictions

    rows = load_predictions(pred_path)
    meta_path = pred_path.parent / "run.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    summary = summarise(rows, meta, n_boot=n_boot, cascade_pair=cascade_pair)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = make_figures(summary, out_dir / "figures")
    (out_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=1, allow_nan=False))
    (out_dir / "report.md").write_text(render_markdown(summary, figures))
    return summary
