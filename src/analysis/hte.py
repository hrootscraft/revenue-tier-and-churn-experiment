"""
Heterogeneous Treatment Effects (HTE)
======================================
Not all clients respond equally. This module finds which segments
benefit most and which don't — the key to a targeted rollout.

Uses Empirical Bayes shrinkage to avoid false discoveries in small segments.
Without shrinkage, a segment with n=8 showing +50% lift looks amazing
but is almost certainly noise. Shrinkage pulls it toward the grand mean.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_hte(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str = "post_volume",
    treatment_col: str = "group",
) -> pd.DataFrame:
    """
    Compute treatment effect for each level of group_col.
    
    For each group: ATE = mean(treatment) - mean(control)
    Then apply Empirical Bayes shrinkage.
    
    Args:
        df: Experiment data
        group_col: Column to segment by (e.g., 'distance_bucket', 'segment')
        outcome_col: Outcome variable
        treatment_col: Column with 'treatment'/'control' labels
    
    Returns:
        DataFrame with raw and shrunk effects per group
    """
    results = []
    
    for group_val, group_df in df.groupby(group_col):
        t = group_df[group_df[treatment_col] == "treatment"][outcome_col].values
        c = group_df[group_df[treatment_col] == "control"][outcome_col].values
        
        if len(t) < 5 or len(c) < 5:
            continue  # Skip tiny groups
        
        raw_effect = np.mean(t) - np.mean(c)
        se = np.sqrt(np.var(t, ddof=1) / len(t) + np.var(c, ddof=1) / len(c))
        
        # Percentage lift relative to control mean
        c_mean = np.mean(c)
        pct_lift = (raw_effect / c_mean * 100) if c_mean != 0 else np.nan
        
        # t-test for this segment
        _, p_value = stats.ttest_ind(t, c, equal_var=False)
        
        results.append({
            "group": group_val,
            "n_total": len(group_df),
            "n_treatment": len(t),
            "n_control": len(c),
            "treatment_mean": np.mean(t),
            "control_mean": c_mean,
            "raw_effect": raw_effect,
            "pct_lift": pct_lift,
            "se": se,
            "p_value": p_value,
            "significant_5pct": p_value < 0.05,
        })
    
    result_df = pd.DataFrame(results)
    
    # Apply Empirical Bayes shrinkage
    if len(result_df) >= 2:
        result_df = _apply_shrinkage(result_df)
    else:
        result_df["shrunk_effect"] = result_df["raw_effect"]
        result_df["shrinkage_factor"] = 1.0
    
    return result_df


def _apply_shrinkage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Empirical Bayes shrinkage: pull noisy estimates toward the grand mean.
    
    shrunk_effect = B * raw_effect + (1 - B) * grand_mean
    where B = tau² / (tau² + se²)
    
    tau² = between-group variance of true effects
    se² = within-group sampling variance
    
    Small groups (high se²) get pulled more toward the grand mean.
    Large groups (low se²) keep most of their raw estimate.
    """
    # Grand mean (weighted by sample size)
    grand_mean = np.average(df["raw_effect"], weights=df["n_total"])
    
    # Estimate between-group variance (tau²) using DerSimonian-Laird
    weights = 1 / df["se"] ** 2
    q_stat = np.sum(weights * (df["raw_effect"] - grand_mean) ** 2)
    df_k = len(df) - 1
    c = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
    tau_sq = max(0, (q_stat - df_k) / c)
    
    # Shrinkage factor for each group
    df = df.copy()
    df["shrinkage_factor"] = tau_sq / (tau_sq + df["se"] ** 2)
    df["shrunk_effect"] = (
        df["shrinkage_factor"] * df["raw_effect"]
        + (1 - df["shrinkage_factor"]) * grand_mean
    )
    df["shrunk_pct_lift"] = df["shrunk_effect"] / df["control_mean"] * 100
    
    return df


def compute_hte_revenue(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    HTE on NET REVENUE (not just volume) — the metric that matters for
    the business decision.
    
    This combines the volume HTE with the revenue decomposition to show:
    "Which segments generate positive net revenue?"
    """
    volume_hte = compute_hte(df, group_col, outcome_col="post_volume")
    revenue_hte = compute_hte(df, group_col, outcome_col="revenue_change")
    
    # Merge with tier-crossing rates
    tier_info = (
        df[df["group"] == "treatment"]
        .groupby(group_col)
        .agg(pct_crossed_tier=("crossed_tier", "mean"))
        .reset_index()
        .rename(columns={group_col: "group"})
    )
    
    combined = volume_hte[["group", "n_total", "pct_lift", "p_value", "significant_5pct"]].rename(
        columns={"pct_lift": "volume_lift_pct", "p_value": "volume_p_value"}
    )
    combined = combined.merge(
        revenue_hte[["group", "raw_effect", "pct_lift"]].rename(
            columns={"raw_effect": "net_revenue_effect", "pct_lift": "net_revenue_pct"}
        ),
        on="group"
    )
    combined = combined.merge(tier_info, on="group", how="left")
    
    return combined
