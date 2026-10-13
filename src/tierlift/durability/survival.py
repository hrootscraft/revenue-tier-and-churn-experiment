"""Does the effect last? Survival analysis of campaign engagement.

Two complementary views of decay:

1. **Weekly treatment effect curve** (model-free). For each week t after
   launch, the treated-minus-holdout difference in volume, as a % of the
   holdout mean. If the effect decays, this curve falls. We fit
   log(effect_t) = a - lambda * t, so lambda is the weekly decay rate.

2. **Survival analysis of engagement** (per-client). A treated client stays
   "engaged" with the campaign until a lapse event: the last week they
   interacted with the Tier Accelerator feature (progress-bar views, email
   clicks). That is product telemetry, observable in production, and the
   simulator exposes it as ``lapse_week`` / ``lapsed``. Clients still engaged
   at the end of the follow-up are *right-censored*, which is exactly what
   Kaplan-Meier and Cox handle and a naive "average weeks engaged" does not.

   Why not detect the lapse from volume alone? For most clients the planted
   lift (3-8%) is smaller than their week-to-week noise (20-35%), so a
   per-client "volume went back to baseline" rule fires on noise. We tried
   it: the detected lapse week had zero correlation with the true one. The
   aggregate curve (view 1) is the volume-only check that *does* work.

   * Kaplan-Meier: S(t) = prod_{t_j <= t} (1 - d_j / n_j), the probability
     of still being engaged at week t.
   * Weekly hazard: h(t) = d_t / n_t, the chance of lapsing in week t given
     engagement at the start of it. A flat hazard around 2%/week means
     "memoryless" decay: S(t) = (1 - h)^t.
   * Cox proportional hazards: h_i(t) = h_0(t) exp(beta' x_i). Which clients
     lapse faster? (account manager, API-driven flow, engagement).

The two views must agree: if h = 2%/week, the average effect in week t is
about (1 - h)^t of the initial effect, so view 1's lambda ~ -log(1 - h) ~ h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight


def weekly_effect_curve(
    weekly: pd.DataFrame,
    assignment: pd.DataFrame,
    cell: str,
    control: str = "holdout",
    pre_volume: pd.Series | None = None,
) -> pd.DataFrame:
    """Treated-vs-holdout volume gap per week, as a share of the holdout mean.

    If ``pre_volume`` (indexed by client_id) is given, each client's weekly
    volume is first divided by their average pre-period weekly volume. That
    is the CUPED idea applied week by week, and it removes the whale noise.
    """
    df = weekly.merge(assignment[["client_id", "cell"]], on="client_id")
    df = df[df["cell"].isin([cell, control])].copy()
    if pre_volume is not None:
        df["volume"] = df["volume"] / df["client_id"].map(pre_volume).to_numpy()
    rows = []
    for week, sub in df.groupby("week"):
        t, c = sub.loc[sub["cell"] == cell, "volume"], sub.loc[sub["cell"] == control, "volume"]
        eff = t.mean() - c.mean()
        se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
        rows.append({"week": week, "effect_pct": eff / c.mean(), "se_pct": se / c.mean()})
    out = pd.DataFrame(rows).set_index("week")
    out["ci_low"], out["ci_high"] = (
        out["effect_pct"] - 1.96 * out["se_pct"],
        out["effect_pct"] + 1.96 * out["se_pct"],
    )
    return out


def fit_exponential_decay(curve: pd.DataFrame, max_week: int | None = None) -> dict:
    """Weighted log-linear fit log(effect) = a - lambda * (week - 1).

    Weeks with a non-positive point estimate carry no information about the
    log and are dropped; the weights are 1/se^2 of the log effect (delta method).
    """
    c = curve if max_week is None else curve.loc[:max_week]
    c = c[c["effect_pct"] > 0]
    y = np.log(c["effect_pct"].to_numpy())
    w = (c["effect_pct"] / c["se_pct"]).to_numpy() ** 2
    X = sm.add_constant(c.index.to_numpy(float) - 1)
    fit = sm.WLS(y, X, weights=w).fit()
    # The slope is the continuous rate -ln(1 - h); report the discrete weekly rate h so that
    # effect_t = effect_1 * (1 - h)^(t - 1), the same form as the survival analysis.
    lam = -fit.params[1]
    h = 1 - np.exp(-lam)
    return {
        "initial_effect_pct": float(np.exp(fit.params[0])),
        "weekly_decay": float(h),
        "weekly_decay_se": float(fit.bse[1] * np.exp(-lam)),  # delta method
        "half_life_weeks": float(np.log(2) / lam) if lam > 0 else np.inf,
        "r2": float(fit.rsquared),
    }


def kaplan_meier(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    sf = SurvfuncRight(np.asarray(time, float), np.asarray(event, int))
    return pd.DataFrame(
        {
            "week": sf.surv_times,
            "survival": sf.surv_prob,
            "se": sf.surv_prob_se,
            "n_at_risk": sf.n_risk,
        }
    )


def weekly_hazard(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """Discrete-time hazard: events / at-risk per week, with a pooled estimate."""
    time, event = np.asarray(time, float), np.asarray(event, int)
    weeks = np.arange(1, int(np.nanmax(time)) + 1)
    rows = []
    for w in weeks:
        at_risk = (time >= w).sum()
        events = ((time == w) & (event == 1)).sum()
        rows.append(
            {
                "week": w,
                "n_at_risk": at_risk,
                "events": events,
                "hazard": events / at_risk if at_risk else np.nan,
            }
        )
    out = pd.DataFrame(rows).set_index("week")
    out.attrs["pooled_hazard"] = float(out["events"].sum() / out["n_at_risk"].sum())
    out.attrs["pooled_hazard_se"] = float(
        np.sqrt(
            out.attrs["pooled_hazard"] * (1 - out.attrs["pooled_hazard"]) / out["n_at_risk"].sum()
        )
    )
    return out


def cox_model(df: pd.DataFrame, time_col: str, event_col: str, covariates: list[str]):
    """Cox proportional-hazards fit. Returns the statsmodels results object."""
    X = df[covariates].astype(float)
    X = (X - X.mean()) / X.std()
    model = PHReg(df[time_col].to_numpy(float), X, status=df[event_col].to_numpy(int), ties="efron")
    return model.fit()


def cox_table(result, covariates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "covariate": covariates,
            "log_hr": result.params,
            "hazard_ratio": np.exp(result.params),
            "se": result.bse,
            "p_value": result.pvalues,
        }
    ).set_index("covariate")
