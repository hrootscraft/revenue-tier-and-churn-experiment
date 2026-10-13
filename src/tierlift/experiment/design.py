"""Design-stage checks: sample-ratio mismatch, covariate balance, and power.

These checks come *before* any effect is estimated. An effect estimate from a
broken randomisation is worse than no estimate, because it looks authoritative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def srm_test(cells: pd.Series, planned: dict[str, float]) -> dict:
    """Sample-ratio mismatch: chi-square test of observed vs planned cell counts.

    A tiny p-value (< 0.001 is the usual alarm) means assignment or logging is
    broken, e.g. a bug drops treated clients who never loaded the dashboard.
    """
    counts = cells.value_counts().reindex(list(planned)).fillna(0).to_numpy()
    expected = np.array(list(planned.values())) * counts.sum()
    chi2, p = stats.chisquare(counts, expected)
    return {
        "observed": dict(zip(planned, counts.astype(int).tolist(), strict=True)),
        "expected": dict(zip(planned, expected.round(1).tolist(), strict=True)),
        "chi2": float(chi2),
        "p_value": float(p),
    }


def standardized_mean_difference(x1: np.ndarray, x0: np.ndarray) -> float:
    """SMD = (mean1 - mean0) / sqrt((var1 + var0) / 2). |SMD| < 0.1 is the usual bar."""
    s = np.sqrt((np.var(x1, ddof=1) + np.var(x0, ddof=1)) / 2)
    return float((np.mean(x1) - np.mean(x0)) / s) if s > 0 else 0.0


def balance_table(
    df: pd.DataFrame, covariates: list[str], cell_col: str = "cell", control: str = "holdout"
) -> pd.DataFrame:
    rows = []
    c = df[df[cell_col] == control]
    for cell in [x for x in df[cell_col].unique() if x != control]:
        t = df[df[cell_col] == cell]
        for cov in covariates:
            rows.append(
                {
                    "cell": cell,
                    "covariate": cov,
                    "mean_treated": t[cov].mean(),
                    "mean_holdout": c[cov].mean(),
                    "smd": standardized_mean_difference(
                        t[cov].to_numpy(float), c[cov].to_numpy(float)
                    ),
                }
            )
    return pd.DataFrame(rows)


def mde(
    sd: float, n_treated: int, n_control: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Minimum detectable effect (absolute) for a two-sided z-test.

    MDE = (z_{1-alpha/2} + z_{power}) * sd * sqrt(1/n_t + 1/n_c)
    """
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd * np.sqrt(1 / n_treated + 1 / n_control))


def power_at(
    effect: float, sd: float, n_treated: int, n_control: int, alpha: float = 0.05
) -> float:
    se = sd * np.sqrt(1 / n_treated + 1 / n_control)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(z - abs(effect) / se) + stats.norm.cdf(-z - abs(effect) / se))


def power_table(
    sd: float,
    mean: float,
    rho: float,
    n_total: int,
    cells: dict[str, float],
    alpha: float = 0.05,
    power: float = 0.8,
) -> pd.DataFrame:
    """MDE (as % of the holdout mean) for each treatment cell vs holdout, raw and CUPED-adjusted.

    CUPED shrinks the effective sd by sqrt(1 - rho^2), where rho is the
    correlation between the pre-period metric and the outcome. At design time
    rho comes from history (e.g. one quarter vs the next).
    """
    n_c = int(n_total * cells["holdout"])
    rows = []
    for cell, share in cells.items():
        if cell == "holdout":
            continue
        n_t = int(n_total * share)
        raw = mde(sd, n_t, n_c, alpha, power)
        adj = mde(sd * np.sqrt(1 - rho**2), n_t, n_c, alpha, power)
        rows.append(
            {
                "cell": cell,
                "n_treated": n_t,
                "n_holdout": n_c,
                "mde_raw_pct": raw / mean,
                "mde_cuped_pct": adj / mean,
                "rho": rho,
                "equivalent_sample_multiplier": 1 / (1 - rho**2),
            }
        )
    return pd.DataFrame(rows)
