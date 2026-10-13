"""Pre-treatment feature matrix shared by AIPW, the uplift models and the policy tree.

Rule: only variables fixed *before* randomisation go in. Anything measured
after launch (crossed_tier, lapse_week, exp_volume, ...) could itself be
affected by treatment. Conditioning on it would bias every CATE estimate
(a "bad control").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = [
    "distance_pct_filled",
    "is_top_tier",
    "tier_idx",
    "log_pre_volume",
    "log_pre_revenue",
    "maker_ratio",
    "api_share",
    "engagement_z",
    "aum_flow_z",
    "tenure_months",
    "has_account_manager",
    "weekly_cv",
]
CATEGORICAL = ["segment", "region"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["distance_pct_filled"] = df["distance_pct"].fillna(1.0)  # top tier: "infinitely far"
    out["is_top_tier"] = df["distance_pct"].isna().astype(int)
    out["tier_idx"] = df["tier_idx"]
    out["log_pre_volume"] = np.log(df["pre_volume"])
    out["log_pre_revenue"] = np.log(df["pre_revenue"])
    for c in NUMERIC[5:]:
        out[c] = df[c].astype(float)
    for c in CATEGORICAL:
        dummies = pd.get_dummies(df[c], prefix=c, dtype=float)
        out = out.join(dummies.drop(columns=dummies.columns[0]))  # drop one level as baseline
    return out
