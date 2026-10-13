"""Command line: ``tierlift run-all [--fast] [--config PATH] [--seed N]``."""

from __future__ import annotations

import argparse
import logging
import sys

from tierlift.config import load_config
from tierlift.pipeline import run_all
from tierlift.simulate import simulate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tierlift", description="Fee-tier incentive experiment: simulate and analyse."
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_ in [
        ("run-all", "simulate, validate, analyse, write results/ and figures"),
        ("simulate", "simulate only, write data/"),
        (
            "montecarlo",
            "repeat simulate + estimate over many seeds; write results/montecarlo_*.csv",
        ),
    ]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
        sp.add_argument("--seed", type=int, default=None, help="override the config seed")
        sp.add_argument("--out", default=None, help="results directory")
        sp.add_argument("--fast", action="store_true", help="fewer permutations / trees, for CI")
        sp.add_argument(
            "--n-seeds", type=int, default=40, help="montecarlo: number of replications"
        )
        sp.add_argument("--workers", type=int, default=4, help="montecarlo: parallel processes")
        sp.add_argument(
            "--benchmark",
            action="store_true",
            help="montecarlo: replicate the uplift benchmark instead",
        )
    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    overrides = {"seed": a.seed} if a.seed is not None else {}
    cfg = load_config(a.config, **overrides)
    if a.command == "simulate":
        data = simulate(cfg)
        from tierlift.config import REPO_ROOT

        data.save(a.out or REPO_ROOT / "data")
        print(f"simulated {len(data.experiment)} clients")
        return 0
    if a.command == "montecarlo":
        from tierlift.montecarlo import run_benchmark_montecarlo, run_montecarlo

        if a.benchmark:
            summary = run_benchmark_montecarlo(
                cfg, n_seeds=a.n_seeds, workers=a.workers, out_dir=a.out
            )
        else:
            summary = run_montecarlo(cfg, n_seeds=a.n_seeds, workers=a.workers, out_dir=a.out)
        print(summary.to_string())
        return 0
    head = run_all(cfg, out_dir=a.out, fast=a.fast)
    print("\nHEADLINE")
    print(f"  CUPED variance reduction: {head['cuped_variance_reduction']:.0%}")
    for k, v in head["rollout"].items():
        print(f"  {k}: {v['est_net_revenue_pct']:+.1%} (truth {v['true_net_revenue_pct']:+.1%})")
    print(f"  weekly lapse hazard: {head['durability']['weekly_hazard']:.2%}")
    print(
        f"  naive annualisation overstates the benefit by {head['durability']['benefit_overstatement_pct']:.0%}"
    )
    print(f"  best uplift model by AUUC: {head['uplift']['best_by_auuc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
