"""Command line: `jevlab fetch | run | report | site`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import datasets as ds

ALL_ARMS = "jev,haiku,sonnet,logreg"


def _csv(value: str, universe: list[str]) -> list[str]:
    if value == "all":
        return list(universe)
    names = [v.strip() for v in value.split(",") if v.strip()]
    bad = [n for n in names if n not in universe]
    if bad:
        raise SystemExit(f"unknown name(s) {bad}; choose from {universe}")
    return names


def cmd_fetch(args: argparse.Namespace) -> int:
    names = _csv(args.datasets, list(ds.DATASETS))
    ds.fetch(names, Path(args.data_dir), n=args.n, seed=args.seed)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    from .arms import ConfigError, build_arms, load_arm_config
    from .budget import Budget
    from .cache import ResponseCache
    from .pricing import PriceTable
    from .runner import run

    load_dotenv()
    config = load_arm_config(Path(args.config))
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    names = _csv(args.datasets, list(ds.DATASETS))
    data_dir = Path(args.data_dir)
    budget = Budget(args.budget_usd)
    try:
        arms = build_arms(
            arm_names,
            config,
            dry_run=args.dry_run,
            prices=PriceTable.load(Path(args.pricing)),
            cache=ResponseCache(None if args.no_cache else Path(args.cache_dir)),
            budget=budget,
            data_dir=data_dir,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        items = {n: ds.load_eval(n, data_dir)[: args.limit or None] for n in names}
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(
            "DRY RUN: API arms use synthetic replies (no network, no keys). Results are demo output."
        )
    out = Path(args.results_dir) / args.run_name
    summary = run(arms, items, out, budget, repeats=args.repeats, repeat_subset=args.repeat_subset)
    spent = "projected" if args.dry_run else "spent"
    print(
        f"wrote {summary.rows} predictions to {out / 'predictions.jsonl'} "
        f"({summary.errors} parse errors); ${budget.spent_usd:.4f} {spent} of ${budget.cap_usd:.2f} cap"
    )
    if summary.budget_stopped:
        print("budget cap reached: some calls were skipped (see run.json)")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    pred = Path(args.results_dir) / args.run_name / "predictions.jsonl"
    if not pred.exists():
        print(f"error: {pred} not found; run `jevlab run` first", file=sys.stderr)
        return 2
    first, _, second = args.cascade.partition(":")
    write_report(pred, Path(args.out), n_boot=args.n_boot, cascade_pair=(first, second))
    print(f"wrote {Path(args.out) / 'report.md'} and figures/")
    return 0


def cmd_site(args: argparse.Namespace) -> int:
    from .site import build_site

    path = build_site(Path(args.summary), Path(args.out))
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jevlab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download datasets and write fixed-seed samples")
    f.add_argument("--datasets", default="all")
    f.add_argument("--data-dir", default="data")
    f.add_argument("--n", type=int, default=ds.DEFAULT_N)
    f.add_argument("--seed", type=int, default=ds.DEFAULT_SEED)
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="run arms over the datasets")
    r.add_argument("--arms", default=ALL_ARMS)
    r.add_argument("--datasets", default="all")
    r.add_argument("--budget-usd", type=float, default=15.0)
    r.add_argument(
        "--dry-run", action="store_true", help="synthetic replies for API arms; no network"
    )
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--repeat-subset", type=int, default=50)
    r.add_argument("--limit", type=int, default=0, help="only the first N items per dataset")
    r.add_argument("--run-name", default="default")
    r.add_argument("--config", default="configs/arms.toml")
    r.add_argument("--pricing", default="configs/pricing.toml")
    r.add_argument("--data-dir", default="data")
    r.add_argument("--cache-dir", default="cache")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--results-dir", default="results")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="metrics, figures and report.md from a run")
    rep.add_argument("--run-name", default="default")
    rep.add_argument("--results-dir", default="results")
    rep.add_argument("--out", default="report")
    rep.add_argument("--n-boot", type=int, default=1000)
    rep.add_argument("--cascade", default="jev:sonnet", help="first:second arm for the cascade")
    rep.set_defaults(func=cmd_report)

    s = sub.add_parser("site", help="self-contained HTML leaderboard from summary.json")
    s.add_argument("--summary", default="report/summary.json")
    s.add_argument("--out", default="site")
    s.set_defaults(func=cmd_site)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
