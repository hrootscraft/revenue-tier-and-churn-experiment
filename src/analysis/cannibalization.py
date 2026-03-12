"""
Cannibalization Check
In institutional crypto, the market can be zero-sum.
If treatment clients trade more, are they stealing volume from control clients?

Check: Did control group volume DECLINE during the experiment
relative to their pre-period baseline?
"""

import numpy as np
import pandas as pd
from scipy import stats


def check_cannibalization(
    pre_activity: pd.DataFrame,
    post_activity: pd.DataFrame,
    assignment: pd.DataFrame,
) -> dict:
    """
    Check whether control group volume declined during the experiment.
    
    If control volume dropped significantly, the treatment might be
    cannibalizing rather than growing the pie.
    """
    control_ids = set(
        assignment[assignment["group"] == "control"]["client_id"]
    )
    
    pre_control = pre_activity[pre_activity["client_id"].isin(control_ids)]
    post_control = post_activity[post_activity["client_id"].isin(control_ids)]
    
    pre_quarterly = pre_control.groupby("client_id")["weekly_volume"].sum()
    post_quarterly = post_control.groupby("client_id")["weekly_volume"].sum()
    
    pre_mean = pre_quarterly.mean()
    post_mean = post_quarterly.mean()
    change_pct = (post_mean - pre_mean) / pre_mean * 100
    
    # Paired t-test: did individual control clients change?
    common_ids = pre_quarterly.index.intersection(post_quarterly.index)
    t_stat, p_value = stats.ttest_rel(
        post_quarterly.loc[common_ids].values,
        pre_quarterly.loc[common_ids].values,
    )
    
    # Also check by segment
    segment_map = assignment.merge(
        pre_activity[["client_id"]].drop_duplicates()
    )
    
    return {
        "control_pre_mean": pre_mean,
        "control_post_mean": post_mean,
        "control_change_pct": change_pct,
        "paired_t_stat": t_stat,
        "p_value": p_value,
        "evidence_of_cannibalization": change_pct < -5 and p_value < 0.05,
        "interpretation": _interpret(change_pct, p_value),
    }


def _interpret(change_pct, p_value):
    if change_pct < -5 and p_value < 0.05:
        return (
            "⚠ WARNING: Control volume declined significantly. "
            "Treatment may be cannibalizing, not growing the pie."
        )
    elif change_pct < -3:
        return (
            "Marginal decline in control volume. Not statistically "
            "significant but worth monitoring in a longer experiment."
        )
    else:
        return "No evidence of cannibalization. Control volume stable."
