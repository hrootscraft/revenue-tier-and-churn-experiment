"""
Durability Analysis
====================
Does the treatment effect persist or fade?
A naive annualization of a single quarter's result is a red flag.
This module tracks the week-by-week treatment effect.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_weekly_effects(
    post_activity: pd.DataFrame,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each week of the experiment, compute the treatment-control
    difference in average volume.
    
    Returns a DataFrame with one row per week showing:
    - Treatment and control means
    - Effect size (absolute and percentage)
    - 95% confidence interval
    """
    merged = post_activity.merge(assignment[["client_id", "group"]], on="client_id")
    
    weekly = []
    for week in sorted(merged["week"].unique()):
        week_data = merged[merged["week"] == week]
        t = week_data[week_data["group"] == "treatment"]["weekly_volume"].values
        c = week_data[week_data["group"] == "control"]["weekly_volume"].values
        
        effect = np.mean(t) - np.mean(c)
        se = np.sqrt(np.var(t, ddof=1) / len(t) + np.var(c, ddof=1) / len(c))
        c_mean = np.mean(c)
        
        weekly.append({
            "week": week,
            "treatment_mean": np.mean(t),
            "control_mean": c_mean,
            "effect": effect,
            "effect_pct": effect / c_mean * 100 if c_mean > 0 else 0,
            "se": se,
            "ci_lower": effect - 1.96 * se,
            "ci_upper": effect + 1.96 * se,
        })
    
    return pd.DataFrame(weekly)


def estimate_decay_rate(weekly_effects: pd.DataFrame) -> dict:
    """
    Fit a linear model: effect_pct = a - b × week
    to estimate weekly decay rate.
    
    Returns: initial_effect, weekly_decay, r_squared, and
    projected annual revenue under different scenarios.
    """
    x = weekly_effects["week"].values
    y = weekly_effects["effect_pct"].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return {
        "initial_effect_pct": intercept,
        "weekly_decay_pct": -slope,  # positive = decaying
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "week_1_effect": intercept + slope * 1,
        "week_13_effect": intercept + slope * 13,
        "interpretation": _interpret_decay(slope, intercept),
    }


def annualization_scenarios(
    weekly_effects: pd.DataFrame,
    total_treatment_revenue_q: float,
) -> pd.DataFrame:
    """
    Project annual revenue under different decay assumptions.
    
    This is the table that prevents naive 4x annualization:
    - No decay (naive)
    - Observed decay rate
    - Conservative (50% faster decay)
    - Pessimistic (effect goes to 0 after Q1)
    """
    q1_incremental = total_treatment_revenue_q  # From experiment quarter
    decay = estimate_decay_rate(weekly_effects)
    
    scenarios = [
        {
            "scenario": "No decay (naive)",
            "assumption": "Q1 effect persists unchanged",
            "annual_revenue": q1_incremental * 4,
        },
        {
            "scenario": "Observed decay",
            "assumption": f"{decay['weekly_decay_pct']:.1f}% per week",
            "annual_revenue": q1_incremental * _decay_multiplier(
                decay["weekly_decay_pct"] / 100, weeks=52
            ),
        },
        {
            "scenario": "Conservative",
            "assumption": "50% effect remaining after 6 months",
            "annual_revenue": q1_incremental * 3.0,  # ~75% average over year
        },
        {
            "scenario": "Pessimistic",
            "assumption": "Effect disappears after Q1",
            "annual_revenue": q1_incremental * 1.0,
        },
    ]
    
    return pd.DataFrame(scenarios)


def _decay_multiplier(weekly_decay_rate: float, weeks: int) -> float:
    """Compute cumulative multiplier accounting for weekly decay."""
    total = sum((1 - weekly_decay_rate) ** w for w in range(weeks))
    return total / 13  # Normalize to quarterly units


def _interpret_decay(slope: float, intercept: float) -> str:
    if slope > -0.1:
        return "Effect is stable — safe to project forward"
    elif slope > -0.5:
        return "Moderate decay — use conservative annualization"
    else:
        return "Rapid decay — likely novelty effect, do NOT annualize at face value"
