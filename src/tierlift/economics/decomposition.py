"""Revenue decomposition: volume effect vs rate-compression effect.

Revenue is volume times rate, R = V * r. If a treatment changes both,

    dR = R_1 - R_0
       = (V_1 - V_0) * r_0          <- volume effect: more trading at the old price
       + V_0 * (r_1 - r_0)          <- rate effect: the old volume at a cheaper price
       + (V_1 - V_0) * (r_1 - r_0)  <- interaction: the *new* volume at the cheaper price

This is an exact identity (expand the right-hand side and the cross terms
cancel), so the three pieces always sum to the total, which the unit tests
check. It is the same trick as a Laspeyres price/volume split in economics.

Why it matters here: the fee-tier incentive raises volume (good) *and* lowers
the rate (bad). The sign of dR depends on which is bigger, and that differs by
how close the client was to the tier boundary:

* within 10%: they were about to cross anyway, so the incentive mostly gives
  away rate on volume the exchange would have earned at full price.
* 10-20%: the push to cross is real, and most of the extra volume is at the
  new rate but there is enough of it to pay for the discount.
* 20%+: far from the boundary, small lift, half-step discount on everything.

At the *group* level we work with aggregate volume and the revenue-weighted
average rate, so the identity holds for groups too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tierlift.experiment.ate import aipw
from tierlift.features import build_features


def decompose(v0: np.ndarray, r0: np.ndarray, v1: np.ndarray, r1: np.ndarray) -> pd.DataFrame:
    """Elementwise decomposition. Returns volume_effect, rate_effect, interaction, total."""
    v0, r0, v1, r1 = (np.atleast_1d(np.asarray(a, float)) for a in (v0, r0, v1, r1))
    vol = (v1 - v0) * r0
    rate = v0 * (r1 - r0)
    inter = (v1 - v0) * (r1 - r0)
    return pd.DataFrame(
        {
            "volume_effect": vol,
            "rate_effect": rate,
            "interaction": inter,
            "total": v1 * r1 - v0 * r0,
        }
    )


MIN_ARM_FOR_ADJUSTMENT = 25


def _arm_means(sub: pd.DataFrame, cell: str, control: str, col: str, adjust: bool, seed: int):
    """Per-client arm means of ``col``: raw, or AIPW covariate-adjusted (same estimator as the rollout table)."""
    d = (sub["cell"] == cell).to_numpy()
    y = sub[col].to_numpy(float)
    # Groups too small to cross-fit an outcome model (the top tier has a handful of clients)
    # fall back to raw means; the adjustment only matters where there are clients to adjust.
    if not adjust or min(d.sum(), (~d).sum()) < MIN_ARM_FOR_ADJUSTMENT:
        return float(y[~d].mean()), float(y[d].mean())
    est = aipw(y, d, build_features(sub), seed=seed)
    return float(est.control_mean), float(est.control_mean + est.effect)


def decompose_groups(
    df: pd.DataFrame,
    cell: str,
    by: str = "distance_bucket",
    control: str = "holdout",
    adjust: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Observed decomposition per group, comparing the ``cell`` arm with the holdout arm.

    Within a group the two arms are exchangeable (randomisation), so the arm
    means of volume and revenue estimate V_0, V_1, R_0, R_1, and the
    volume-weighted rates are r = R / V. Numbers are per client, so groups of
    different sizes are comparable.

    ``adjust=True`` replaces the raw arm means by AIPW covariate-adjusted
    means (write-up 4), the estimator the rollout table uses. The pieces then
    sum to the rollout estimate instead of to the noisier raw difference.
    """
    rows = []
    for g, sub in df.groupby(by):
        sub = sub[sub["cell"].isin([cell, control])].reset_index(drop=True)
        t = sub[sub["cell"] == cell]
        c = sub[sub["cell"] == control]
        v0, v1 = _arm_means(sub, cell, control, "exp_volume", adjust, seed)
        rev0, rev1 = _arm_means(sub, cell, control, "exp_revenue", adjust, seed)
        r0, r1 = rev0 / v0, rev1 / v1
        parts = decompose(v0, r0, v1, r1).iloc[0]
        base = rev0
        rows.append(
            {
                by: g,
                "n_treated": len(t),
                "n_holdout": len(c),
                "holdout_revenue_per_client": base,
                "volume_lift_pct": v1 / v0 - 1,
                "rate_change_pct": r1 / r0 - 1,
                **parts.to_dict(),
                "net_revenue_pct": parts["total"] / base,
            }
        )
    return pd.DataFrame(rows).set_index(by)


def decompose_true(truth: pd.DataFrame, cell: str, groups: pd.Series) -> pd.DataFrame:
    """Same decomposition on the planted potential outcomes (for validation)."""
    rows = []
    for g, idx in groups.groupby(groups).groups.items():
        t = truth.loc[idx]
        v0, v1 = t["volume_holdout"].mean(), t[f"volume_{cell}"].mean()
        r0 = t["revenue_holdout"].sum() / t["volume_holdout"].sum()
        r1 = t[f"revenue_{cell}"].sum() / t[f"volume_{cell}"].sum()
        parts = decompose(v0, r0, v1, r1).iloc[0]
        rows.append(
            {
                "group": g,
                **parts.to_dict(),
                "net_revenue_pct": parts["total"] / t["revenue_holdout"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("group")
