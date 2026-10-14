"""Monte Carlo study: repeat the simulation and the key estimates over many seeds.

One simulated experiment is one draw. Showing that a method "works" on one
draw proves little: the estimate could be close to the truth by luck. The
honest check is to repeat the whole thing N times and ask

* **bias**: is the average estimate equal to the average truth?
* **calibration**: does the reported standard error match the spread of the
  estimates across seeds, and does the 95% CI cover the truth ~95% of the time?

This is also how the showcase seed in ``configs/default.yaml`` is chosen (see
docs/DECISIONS.md A3): the draw whose targeted-rollout estimate is closest to the
median of the estimates. The rule never looks at the planted truth, so it picks
the middle run, and the figures in the README show a *typical* experiment.

Runtime: ~5 s per seed single-threaded; 40 seeds on 4 workers ≈ 1 minute.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tierlift.config import REPO_ROOT, Config, load_config

log = logging.getLogger("tierlift")


def _one_seed(args: tuple[str, dict, int]) -> dict:
    """Worker: simulate one seed and compute the headline estimates."""
    from threadpoolctl import threadpool_limits

    config_path, overrides, seed = args
    with threadpool_limits(limits=1):  # one thread per worker, so workers do not fight
        from tierlift.durability.survival import weekly_hazard
        from tierlift.economics.rollout import ContrastScores
        from tierlift.simulate import simulate
        from tierlift.uplift.policy import rule_policy

        cfg = load_config(config_path, seed=seed, **overrides)
        data = simulate(cfg)
        e, truth = data.experiment, data.truth.set_index("client_id")
        primary, control = cfg.analysis.primary_contrast
        row: dict = {"seed": seed}
        scores = ContrastScores(e, primary, control, seed)
        targets = {
            "blanket": np.ones(len(e), bool),
            "band": rule_policy(e, cfg.analysis.targeting_band),
        }
        for name, target in targets.items():
            s = e[target & e["cell"].isin([primary, control])]
            tt = truth.loc[s["client_id"]]
            row[f"{name}_truth"] = float(
                tt[f"revenue_{primary}"].sum() / tt["revenue_holdout"].sum() - 1
            )
            c, a = scores.cuped(target), scores.estimate(target)
            row.update(
                {
                    f"{name}_cuped": c.rel_effect,
                    f"{name}_cuped_se": c.rel_se,
                    f"{name}_aipw": a.rel_effect,
                    f"{name}_aipw_se": a.rel_se,
                }
            )
            if name == "blanket":
                row["cuped_variance_reduction"] = c.extra["variance_reduction"]
        t = e[e["cell"] != control]
        row["weekly_hazard"] = weekly_hazard(
            t["lapse_week"].to_numpy(), t["lapsed"].to_numpy()
        ).attrs["pooled_hazard"]
        return row


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["blanket", "band"]:
        for est in ["cuped", "aipw"]:
            err = df[f"{target}_{est}"] - df[f"{target}_truth"]
            se = df[f"{target}_{est}_se"]
            rows.append(
                {
                    "target": target,
                    "estimator": est,
                    "n_seeds": len(df),
                    "mean_truth": df[f"{target}_truth"].mean(),
                    "mean_estimate": df[f"{target}_{est}"].mean(),
                    "bias": err.mean(),
                    "sd_of_estimates": df[f"{target}_{est}"].std(ddof=1),
                    "mean_reported_se": se.mean(),
                    "ci95_coverage": float((err.abs() <= 1.96 * se).mean()),
                }
            )
    out = pd.DataFrame(rows).set_index(["target", "estimator"])
    out.attrs["cuped_variance_reduction"] = (
        df["cuped_variance_reduction"].mean(),
        df["cuped_variance_reduction"].std(ddof=1),
    )
    out.attrs["weekly_hazard"] = (df["weekly_hazard"].mean(), df["weekly_hazard"].std(ddof=1))
    return out


def pick_showcase_seed(df: pd.DataFrame, column: str = "band_aipw") -> int:
    """The seed whose band estimate is closest to the median estimate; never uses the truth."""
    med = df[column].median()
    return int(df.loc[(df[column] - med).abs().idxmin(), "seed"])


def run_montecarlo(
    cfg: Config, n_seeds: int = 40, workers: int = 4, out_dir: str | Path | None = None, **overrides
) -> pd.DataFrame:
    out_dir = Path(out_dir or REPO_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(cfg.source, overrides, seed) for seed in range(n_seeds)]
    log.info("Monte Carlo: %d seeds on %d workers", n_seeds, workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_one_seed, jobs))
    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    df.to_csv(out_dir / "montecarlo_seeds.csv", index=False)
    summary = summarise(df)
    summary.to_csv(out_dir / "montecarlo_summary.csv")
    vr, hz = summary.attrs["cuped_variance_reduction"], summary.attrs["weekly_hazard"]
    extra = pd.Series(
        {
            "cuped_variance_reduction_mean": vr[0],
            "cuped_variance_reduction_sd": vr[1],
            "weekly_hazard_mean": hz[0],
            "weekly_hazard_sd": hz[1],
            "showcase_seed": pick_showcase_seed(df),
        }
    )
    extra.to_csv(out_dir / "montecarlo_extra.csv", header=False)
    log.info(
        "CUPED variance reduction %.3f ± %.3f; hazard %.4f ± %.4f; showcase seed %d",
        vr[0],
        vr[1],
        hz[0],
        hz[1],
        extra["showcase_seed"],
    )
    return summary


# --------------------------------------------------------------------------- benchmark replication


def _one_benchmark_seed(args: tuple[str, dict, int]) -> list[dict]:
    """Worker: simulate one seed, fit the whole model zoo, score it."""
    from threadpoolctl import threadpool_limits

    config_path, overrides, seed = args
    with threadpool_limits(limits=1):
        from tierlift.features import build_features
        from tierlift.simulate import simulate
        from tierlift.uplift.evaluate import benchmark_table
        from tierlift.uplift.learners import fit_zoo, model_zoo

        cfg = load_config(config_path, seed=seed, **overrides)
        data = simulate(cfg)
        e, truth = data.experiment, data.truth.set_index("client_id")
        primary, control = cfg.analysis.primary_contrast
        s = e[e["cell"].isin([primary, control])].reset_index(drop=True)
        d = (s["cell"] == primary).to_numpy()
        X = build_features(s)
        y_index = (s["exp_revenue"] / s["pre_revenue"]).to_numpy()
        ite_index = (
            truth.loc[s["client_id"], f"ite_revenue_{primary}"].to_numpy()
            / s["pre_revenue"].to_numpy()
        )
        specs = model_zoo(seed, cfg.analysis.n_folds, cf_trees=500)
        res = fit_zoo(
            X,
            y_index,
            d,
            s["pre_revenue"].to_numpy(),
            n_folds=4,
            seed=seed,
            specs=specs,
            n_repeats=1,
        )
        bench = benchmark_table(res, y_index, d, ite_index, n_bootstrap=30, seed=seed)
        bench["auuc_rank"] = bench["auuc"].rank(ascending=False)
        bench["spearman_rank"] = bench["spearman_vs_truth"].rank(ascending=False)
        bench["perfect_auuc"] = bench.attrs["perfect_auuc"]
        return [{"seed": seed, "model": m, **row} for m, row in bench.to_dict("index").items()]


def run_benchmark_montecarlo(
    cfg: Config, n_seeds: int = 5, workers: int = 2, out_dir: str | Path | None = None, **overrides
) -> pd.DataFrame:
    """Replicate the uplift benchmark over seeds: is the leaderboard stable?"""
    out_dir = Path(out_dir or REPO_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(cfg.source, overrides, seed) for seed in range(n_seeds)]
    log.info(
        "benchmark Monte Carlo: %d seeds on %d workers (several minutes per seed)", n_seeds, workers
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for rows in pool.map(_one_benchmark_seed, jobs) for r in rows]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "benchmark_montecarlo_seeds.csv", index=False)
    g = df.groupby("model")
    summary = pd.DataFrame(
        {
            "mean_auuc": g["auuc"].mean(),
            "sd_auuc": g["auuc"].std(ddof=1),
            "mean_auuc_rank": g["auuc_rank"].mean(),
            "share_first_by_auuc": g["auuc_rank"].apply(lambda r: (r == 1).mean()),
            "mean_rate_autoc": g["rate_autoc"].mean(),
            "mean_spearman_vs_truth": g["spearman_vs_truth"].mean(),
            "mean_spearman_rank": g["spearman_rank"].mean(),
            "mean_pehe": g["pehe"].mean(),
            "mean_blp_slope": g["blp_heterogeneity_coef"].mean(),
        }
    ).sort_values("mean_auuc_rank")
    summary.to_csv(out_dir / "benchmark_montecarlo_summary.csv")
    log.info("benchmark Monte Carlo done:\n%s", summary.round(4).to_string())
    return summary
