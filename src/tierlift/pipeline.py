"""The end-to-end pipeline: simulate -> validate -> analyse -> figures -> headline numbers.

``run_all`` writes every table the README quotes into ``results/`` (CSV/JSON,
small enough to commit) and every figure into ``reports/figures/``. The
notebooks import the same functions, so the two never disagree.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tierlift import plots
from tierlift.config import REPO_ROOT, Config, load_config
from tierlift.durability.roi import roi_projection
from tierlift.durability.survival import (
    cox_model,
    cox_table,
    fit_exponential_decay,
    kaplan_meier,
    weekly_effect_curve,
    weekly_hazard,
)
from tierlift.economics.decomposition import decompose_groups, decompose_true
from tierlift.economics.rollout import ContrastScores, rollout_table, targeting_band_grid
from tierlift.experiment.ate import ate_table, cuped
from tierlift.experiment.design import balance_table, power_table, srm_test
from tierlift.experiment.observational import (
    observational_estimates,
    observational_replication,
    simulate_opt_in,
)
from tierlift.experiment.robustness import holm, robustness_table
from tierlift.features import build_features
from tierlift.simulate import SimulatedData, simulate
from tierlift.uplift.evaluate import benchmark_table, qini_curve
from tierlift.uplift.learners import fit_zoo, model_zoo
from tierlift.uplift.policy import (
    compare_policies,
    dr_scores,
    fit_policy_tree,
    policy_tree_oof,
    policy_tree_text,
    rule_policy,
)
from tierlift.validation.checks import BALANCE_COVARIATES, run_all_checks

log = logging.getLogger("tierlift")
COX_COVARIATES = ["has_account_manager", "api_share", "engagement_z", "tenure_months"]


def _arm(df: pd.DataFrame, cell: str, control: str = "holdout") -> tuple[pd.DataFrame, np.ndarray]:
    s = df[df["cell"].isin([cell, control])].reset_index(drop=True)
    return s, (s["cell"] == cell).to_numpy()


def run_all(
    cfg: Config | None = None,
    out_dir: Path | None = None,
    fig_dir: Path | None = None,
    data_dir: Path | None = None,
    fast: bool = False,
) -> dict:
    """Run everything. ``fast`` shrinks permutations / bootstraps / forest size for CI."""
    t_start = time.perf_counter()
    cfg = cfg or load_config()
    out_dir = Path(out_dir or REPO_ROOT / "results")
    fig_dir = Path(fig_dir or REPO_ROOT / "reports" / "figures")
    data_dir = Path(data_dir or REPO_ROOT / "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    an = cfg.analysis
    n_perm = 500 if fast else an.n_permutations
    n_boot = 50 if fast else an.n_bootstrap
    primary, control = an.primary_contrast
    headline: dict = {
        "seed": cfg.seed,
        "n_clients": cfg.population.n_clients,
        "cells": cfg.experiment.cells,
    }

    # 1. Simulate and validate ----------------------------------------------------------------
    log.info("simulating %d clients", cfg.population.n_clients)
    data: SimulatedData = simulate(cfg)
    data.save(data_dir)
    checks = run_all_checks(data, cfg)
    checks.to_csv(out_dir / "data_quality_checks.csv")
    e, truth = data.experiment, data.truth.set_index("client_id")
    log.info("data-quality checks passed (%d)", len(checks))

    # 2. Design: SRM, balance, power -----------------------------------------------------------
    json.dump(srm_test(e["cell"], cfg.experiment.cells), open(out_dir / "srm.json", "w"), indent=2)
    balance_table(e, BALANCE_COVARIATES).to_csv(out_dir / "balance.csv", index=False)
    h = e[e["cell"] == control]
    rho_design = float(np.corrcoef(h["pre_revenue"], h["exp_revenue"])[0, 1])
    power = power_table(
        float(h["exp_revenue"].std(ddof=1)),
        float(h["exp_revenue"].mean()),
        rho_design,
        len(e),
        cfg.experiment.cells,
        an.alpha,
    )
    power.to_csv(out_dir / "power.csv", index=False)

    # 3. ATE per cell, CUPED, robustness ------------------------------------------------------
    ate_tables, truth_rel, p_values = {}, {}, {}
    for cell in cfg.treatment_cells:
        s, d = _arm(e, cell, control)
        tab = ate_table(
            s["exp_revenue"].to_numpy(),
            d,
            s["pre_revenue"].to_numpy(),
            build_features(s),
            seed=cfg.seed,
        )
        tab.to_csv(out_dir / f"ate_{cell}.csv")
        ate_tables[cell] = tab
        tt = truth.loc[s["client_id"]]
        truth_rel[cell] = float(tt[f"revenue_{cell}"].sum() / tt["revenue_holdout"].sum() - 1)
        rob = robustness_table(
            s["exp_revenue"].to_numpy(), d, s["stratum"].to_numpy(), n_perm, cfg.seed
        )
        rob.to_csv(out_dir / f"robustness_{cell}.csv")
        p_values[cell] = float(tab.loc["CUPED", "p_value"])
        log.info(
            "ATE %s: CUPED %.2f%% (truth %.2f%%), variance cut %.0f%%",
            cell,
            100 * tab.loc["CUPED", "rel_effect"],
            100 * truth_rel[cell],
            100 * tab.loc["CUPED", "variance_reduction"],
        )
    json.dump(
        {"raw": p_values, "holm": holm(p_values)},
        open(out_dir / "multiple_testing.json", "w"),
        indent=2,
    )
    headline["cuped_variance_reduction"] = float(
        ate_tables[primary].loc["CUPED", "variance_reduction"]
    )
    headline["cuped_rho"] = float(ate_tables[primary].loc["CUPED", "rho"])
    headline["ate"] = {
        c: {"cuped_rel": float(t.loc["CUPED", "rel_effect"]), "truth_rel": truth_rel[c]}
        for c, t in ate_tables.items()
    }
    plots.fig_ate_estimators(ate_tables, truth_rel, fig_dir / "01_ate_estimators.png")
    plots.fig_variance_reduction(ate_tables[primary], fig_dir / "02_variance_reduction.png")
    plots.fig_cuped_scatter(
        h["pre_revenue"].to_numpy(),
        h["exp_revenue"].to_numpy(),
        rho_design,
        fig_dir / "03_cuped_scatter.png",
    )

    # 4. Observational counterfactual ----------------------------------------------------------
    obs = simulate_opt_in(e, data.truth, seed=cfg.seed)
    obs_tab = observational_estimates(obs, build_features(obs).to_numpy(), seed=cfg.seed)
    obs_tab.to_csv(out_dir / "observational.csv")
    plots.fig_observational(obs_tab, fig_dir / "04_observational_bias.png")
    obs_rep = observational_replication(
        e,
        data.truth,
        lambda o: build_features(o).to_numpy(),
        n_draws=2 if fast else 10,
        seed=cfg.seed,
    )
    obs_rep.to_csv(out_dir / "observational_replication.csv")

    # 5. Uplift benchmark ------------------------------------------------------------------------
    s, d = _arm(e, primary, control)
    X = build_features(s)
    y_index = (s["exp_revenue"] / s["pre_revenue"]).to_numpy()
    ite_index = (
        truth.loc[s["client_id"], f"ite_revenue_{primary}"].to_numpy() / s["pre_revenue"].to_numpy()
    )
    specs = model_zoo(cfg.seed, an.n_folds, cf_trees=200 if fast else 1000)
    log.info("fitting %d uplift learners (cross-fitted)", len(specs))
    results = fit_zoo(
        X,
        y_index,
        d,
        s["pre_revenue"].to_numpy(),
        n_folds=4,
        seed=cfg.seed,
        specs=specs,
        n_repeats=1 if fast else 2,
    )
    bench = benchmark_table(results, y_index, d, ite_index, n_boot, cfg.seed)
    bench.to_csv(out_dir / "uplift_benchmark.csv")
    json.dump(
        {k: float(v) for k, v in bench.attrs.items()},
        open(out_dir / "uplift_ceiling.json", "w"),
        indent=2,
    )
    scores = pd.DataFrame(
        {name: r.tau_index for name, r in results.items()}, index=s["client_id"]
    ).assign(true_ite_index=ite_index)
    scores.to_parquet(data_dir / "uplift_scores.parquet")
    curves = {name: qini_curve(r.tau_index, y_index, d) for name, r in results.items()}
    # Every 10th point of each curve is plenty for a plot and keeps the committed CSV small.
    pd.concat(curves, names=["model"]).reset_index(level=0).iloc[::10].round(6).to_csv(
        out_dir / "qini_curves.csv", index=False
    )
    plots.fig_qini(curves, qini_curve(ite_index, y_index, d), fig_dir / "05_qini_curves.png")
    plots.fig_benchmark(bench, fig_dir / "06_uplift_benchmark.png")
    best = str(bench.index[0])
    headline["uplift"] = {
        "best_by_auuc": best,
        "auuc": bench["auuc"].to_dict(),
        "spearman_vs_truth": bench["spearman_vs_truth"].to_dict(),
        "perfect_auuc": bench.attrs["perfect_auuc"],
    }
    log.info("uplift benchmark: best by AUUC = %s", best)
    s_scored = s.assign(cf_score=results["CausalForestDML"].tau_index, true_ite_index=ite_index)
    plots.fig_cate_vs_distance(
        s_scored, "cf_score", "true_ite_index", fig_dir / "07_cate_vs_distance.png"
    )

    # 6. Policy -------------------------------------------------------------------------------
    tau_dollars = {name: r.tau_dollars for name, r in results.items()}
    y_dollars = s["exp_revenue"].to_numpy(float)
    gamma_index = dr_scores(y_index, d, X.to_numpy(), cfg.seed)
    tree = fit_policy_tree(X, gamma_index, max_depth=2, seed=cfg.seed)
    (out_dir / "policy_tree.txt").write_text(policy_tree_text(tree, list(X.columns)))
    tree_oof = policy_tree_oof(X, gamma_index, max_depth=2, seed=cfg.seed)
    pol = compare_policies(
        s,
        X.to_numpy(),
        y_dollars,
        d,
        tau_dollars,
        an.targeting_band,
        cfg.seed,
        extra_policies={
            "DR policy tree (depth 2, in-sample)": tree.predict(X.to_numpy(float)).astype(bool),
            "DR policy tree (depth 2, cross-fitted)": tree_oof,
        },
    )
    pol.to_csv(out_dir / "policy_comparison.csv")
    plots.fig_policy_tree_values(pol, fig_dir / "08_policy_values.png")

    # 7. Economics: decomposition and rollout -------------------------------------------------
    dec = decompose_groups(e, primary)
    dec.to_csv(out_dir / f"decomposition_{primary}.csv")
    dec_adj = decompose_groups(e, primary, adjust=True, seed=cfg.seed)
    dec_adj.to_csv(out_dir / f"decomposition_{primary}_adjusted.csv")
    decompose_true(truth, primary, e.set_index("client_id")["distance_bucket"]).to_csv(
        out_dir / "decomposition_true.csv"
    )
    plots.fig_decomposition(dec_adj, fig_dir / "09_revenue_decomposition.png")
    band = an.targeting_band
    policies = {
        "Blanket rollout": np.ones(len(e), bool),
        f"Targeted: {band[0]:.0%}-{band[1]:.0%} from tier": rule_policy(e, band),
    }
    scores = ContrastScores(e, primary, control, cfg.seed)
    roll = rollout_table(e, data.truth, policies, primary, control, cfg.seed, scores)
    roll.to_csv(out_dir / "rollout.csv")
    grid = targeting_band_grid(
        e,
        primary,
        [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)],
        seed=cfg.seed,
        scores=scores,
    )
    grid.to_csv(out_dir / "targeting_band_grid.csv")
    plots.fig_rollout(roll, grid, fig_dir / "10_rollout_comparison.png")
    headline["rollout"] = {
        k: {
            "est_net_revenue_pct": float(v["est_net_revenue_pct"]),
            "ci": [float(v["ci_low_pct"]), float(v["ci_high_pct"])],
            "true_net_revenue_pct": float(v["true_net_revenue_pct"]),
        }
        for k, v in roll.iterrows()
    }
    log.info("rollout: %s", {k: f"{v['est_net_revenue_pct']:+.1%}" for k, v in roll.iterrows()})

    # 8. Durability: weekly curve, survival, ROI ---------------------------------------------
    pre_weekly = data.pre_weekly.groupby("client_id")["volume"].mean()
    curve = weekly_effect_curve(data.weekly, data.assignment, primary, control, pre_weekly)
    curve.to_csv(out_dir / "weekly_effect.csv")
    decay = fit_exponential_decay(curve)
    json.dump(decay, open(out_dir / "decay_fit.json", "w"), indent=2)
    plots.fig_weekly_effect(
        curve, decay, cfg.experiment.experiment_weeks, fig_dir / "11_weekly_effect.png"
    )
    treated = e[e["cell"] != control]
    km = kaplan_meier(treated["lapse_week"].to_numpy(), treated["lapsed"].to_numpy())
    km.to_csv(out_dir / "kaplan_meier.csv", index=False)
    hz = weekly_hazard(treated["lapse_week"].to_numpy(), treated["lapsed"].to_numpy())
    hz.to_csv(out_dir / "weekly_hazard.csv")
    json.dump(dict(hz.attrs), open(out_dir / "hazard_summary.json", "w"), indent=2)
    plots.fig_survival(km, hz, cfg.truth.base_weekly_hazard, fig_dir / "12_survival.png")
    cox = cox_table(cox_model(treated, "lapse_week", "lapsed", COX_COVARIATES), COX_COVARIATES)
    cox.to_csv(out_dir / "cox.csv")
    plots.fig_cox(cox, fig_dir / "13_cox_hazard_ratios.png")
    # ROI: project the targeted rollout, scaled to the whole targeted population.
    target = rule_policy(e, band)
    n_target = int(target.sum())
    share_treated = float((e.loc[target, "cell"] == primary).mean())
    bucket = dec_adj.loc["10_to_20"] if "10_to_20" in dec_adj.index else dec_adj.iloc[0]
    roi = roi_projection(
        (bucket["volume_effect"] + bucket["interaction"]) * n_target,
        bucket["rate_effect"] * n_target,
        hz.attrs["pooled_hazard"],
        an.horizon_weeks,
        cfg.experiment.experiment_weeks,
    )
    roi.to_csv(out_dir / "roi_projection.csv")
    json.dump(dict(roi.attrs), open(out_dir / "roi_summary.json", "w"), indent=2)
    plots.fig_roi(roi, fig_dir / "14_roi_projection.png")
    headline["durability"] = {
        "weekly_hazard": hz.attrs["pooled_hazard"],
        "weekly_hazard_se": hz.attrs["pooled_hazard_se"],
        "curve_decay_per_week": decay["weekly_decay"],
        "curve_decay_se": decay["weekly_decay_se"],
        "benefit_overstatement_pct": float(roi["benefit_overstatement_pct"].iloc[0]),
        "net_overstatement_pct": float(roi["net_overstatement_pct"].iloc[0]),
    }
    headline["share_treated_in_band"] = share_treated
    headline["runtime_seconds"] = time.perf_counter() - t_start
    json.dump(headline, open(out_dir / "headline.json", "w"), indent=2)
    log.info("done in %.0fs", headline["runtime_seconds"])
    return headline


def cuped_rel(df: pd.DataFrame, cell: str, control: str = "holdout") -> float:
    s, d = _arm(df, cell, control)
    return cuped(s["exp_revenue"].to_numpy(), d, s["pre_revenue"].to_numpy()).rel_effect
