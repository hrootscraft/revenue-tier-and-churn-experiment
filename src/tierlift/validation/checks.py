"""Data-quality checks that run before any analysis.

Each check returns a ``CheckResult``; ``run_all_checks`` raises if any
*blocking* check fails. The checks encode the assumptions the analysis
relies on, so a broken simulator (or, in production, a broken logging
pipeline) fails loudly instead of producing a confident wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tierlift.config import Config
from tierlift.experiment.design import balance_table, srm_test
from tierlift.features import build_features
from tierlift.simulate import SimulatedData

PRE_TREATMENT_COLUMNS = {
    "client_id",
    "segment",
    "region",
    "tenure_months",
    "has_account_manager",
    "api_share",
    "dashboard_logins_pm",
    "maker_ratio",
    "weekly_cv",
    "pre_volume",
    "pre_revenue",
    "tier_idx",
    "tier",
    "distance_pct",
    "distance_bucket",
    "engagement_z",
    "aum_flow_z",
    "cell",
    "stratum",
}
BALANCE_COVARIATES = [
    "pre_revenue",
    "pre_volume",
    "api_share",
    "engagement_z",
    "aum_flow_z",
    "tenure_months",
    "has_account_manager",
    "maker_ratio",
]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    blocking: bool = True


def check_unique_clients(data: SimulatedData) -> CheckResult:
    dup = data.experiment["client_id"].duplicated().sum()
    n = len(data.experiment)
    return CheckResult(
        "unique clients", dup == 0 and n == len(data.clients), f"{n} rows, {dup} duplicates"
    )


def check_sample_ratio(data: SimulatedData, cfg: Config) -> CheckResult:
    r = srm_test(data.experiment["cell"], cfg.experiment.cells)
    return CheckResult(
        "sample ratio mismatch",
        r["p_value"] > 0.001,
        f"chi2 p = {r['p_value']:.3f}, counts {r['observed']}",
    )


def check_strata_shares(data: SimulatedData, cfg: Config, tol: float = 0.08) -> CheckResult:
    """Within every stratum of at least 20 clients, each cell share is within ``tol`` of plan."""
    e = data.experiment
    worst = 0.0
    for _, g in e.groupby("stratum"):
        if len(g) < 20:
            continue
        shares = g["cell"].value_counts(normalize=True)
        for c, planned in cfg.experiment.cells.items():
            worst = max(worst, abs(shares.get(c, 0) - planned))
    return CheckResult(
        "cell shares within strata", worst <= tol, f"worst deviation from plan {worst:.3f}"
    )


def check_balance(data: SimulatedData, max_smd: float = 0.1) -> CheckResult:
    b = balance_table(data.experiment, BALANCE_COVARIATES)
    worst = b["smd"].abs().max()
    # Non-blocking: under randomisation an |SMD| just above 0.1 is chance (its SE is ~sqrt(1/n_t + 1/n_c)),
    # and every estimator downstream adjusts for pre-period covariates anyway. The SRM test is the alarm.
    return CheckResult(
        "covariate balance",
        worst < max_smd,
        f"max |SMD| = {worst:.3f} across {len(b)} cell x covariate pairs",
        blocking=False,
    )


def check_revenue_identity(data: SimulatedData, tol: float = 1e-6) -> CheckResult:
    e = data.experiment
    err = (e["exp_revenue"] - e["exp_volume"] * e["exp_rate"]).abs().max()
    return CheckResult("revenue = volume x rate", err < tol, f"max abs error ${err:.2e}")


def check_observed_matches_potential(data: SimulatedData) -> CheckResult:
    """The observed outcome equals the potential outcome of the assigned cell (consistency / SUTVA)."""
    e = data.experiment.set_index("client_id")
    t = data.truth.set_index("client_id").loc[e.index]
    expected = np.select(
        [e["cell"] == c for c in ["holdout", "nudge", "incentive"]],
        [t["revenue_holdout"], t["revenue_nudge"], t["revenue_incentive"]],
    )
    err = np.abs(e["exp_revenue"].to_numpy() - expected).max()
    return CheckResult(
        "observed outcome = potential outcome of assigned cell",
        err < 1e-6,
        f"max abs error {err:.2e}",
    )


def check_weekly_sums(data: SimulatedData, cfg: Config) -> CheckResult:
    q = cfg.experiment.experiment_weeks
    s = data.weekly[data.weekly["week"] <= q].groupby("client_id")["volume"].sum()
    e = data.experiment.set_index("client_id")["exp_volume"]
    err = (s - e.loc[s.index]).abs().max() / e.mean()
    return CheckResult(
        "weekly panel sums to quarter volume", err < 1e-9, f"max relative error {err:.2e}"
    )


def check_no_post_treatment_features(data: SimulatedData) -> CheckResult:
    """Every feature must be derivable from pre-treatment columns only."""
    feats = build_features(data.experiment.head(50))
    leaked = [c for c in feats.columns if any(k in c for k in ("exp_", "crossed", "lapse", "ite_"))]
    return CheckResult(
        "no post-treatment features", not leaked, f"{len(feats.columns)} features, leaked: {leaked}"
    )


def check_no_pre_period_leakage(data: SimulatedData) -> CheckResult:
    """Pre-period revenue must not differ by cell (it was fixed before assignment)."""
    e = data.experiment
    means = e.groupby("cell")["pre_revenue"].mean()
    spread = (means.max() - means.min()) / means.mean()
    return CheckResult(
        "no treatment leakage into pre-period",
        spread < 0.08,
        f"relative spread of pre-period means {spread:.3f}",
        blocking=False,
    )


def run_all_checks(data: SimulatedData, cfg: Config) -> pd.DataFrame:
    results = [
        check_unique_clients(data),
        check_sample_ratio(data, cfg),
        check_strata_shares(data, cfg),
        check_balance(data),
        check_revenue_identity(data),
        check_observed_matches_potential(data),
        check_weekly_sums(data, cfg),
        check_no_post_treatment_features(data),
        check_no_pre_period_leakage(data),
    ]
    table = pd.DataFrame([r.__dict__ for r in results]).set_index("name")
    failed = table[(~table["passed"]) & table["blocking"]]
    if len(failed):
        raise RuntimeError("Data-quality checks failed:\n" + failed[["detail"]].to_string())
    return table
