"""
Churn Analysis (Secondary Outcome)
Two analyses, one causal and one observational:

1. CAUSAL: Treatment vs Control churn rates
   → Randomization guarantees this is a valid causal comparison.
   → "Showing tier progress CAUSED a reduction in churn."

2. OBSERVATIONAL: Within treatment, crossed-tier vs didn't-cross
   → This is NOT causal. Tier crossing was not randomized.
   → Clients who crossed are different (growing, more engaged).
   → "Tier-crossers churn less, but we can't be sure crossing CAUSED it."
"""

import numpy as np
import pandas as pd
from scipy import stats


def analyze_churn(df: pd.DataFrame) -> dict:
    """
    Complete churn analysis with both causal and observational components.
    
    Args:
        df: Experiment summary table with churned_6mo, group, crossed_tier
    
    Returns:
        Dict with causal results, observational results, and IV estimate
    """
    treatment = df[df["group"] == "treatment"]
    control = df[df["group"] == "control"]
    
    # ──────────────────────────────────────────────────
    # PART 1: CAUSAL (Treatment vs Control)
    # This comparison IS causal because of randomization
    # ──────────────────────────────────────────────────
    t_churn_rate = treatment["churned_6mo"].mean()
    c_churn_rate = control["churned_6mo"].mean()
    ate_churn = c_churn_rate - t_churn_rate  # Reduction (positive = good)
    
    # Fisher's exact test (better than chi-squared for binary outcomes)
    contingency = pd.crosstab(df["group"], df["churned_6mo"])
    _, p_value_fisher = stats.fisher_exact(contingency.values)
    
    # Also chi-squared for comparison
    chi2, p_value_chi2, _, _ = stats.chi2_contingency(contingency.values)
    
    causal = {
        "treatment_churn_rate": t_churn_rate,
        "control_churn_rate": c_churn_rate,
        "absolute_reduction": ate_churn,
        "relative_reduction": ate_churn / c_churn_rate if c_churn_rate > 0 else 0,
        "p_value_fisher": p_value_fisher,
        "p_value_chi2": p_value_chi2,
        "significant": p_value_fisher < 0.05,
        "n_churns_prevented": ate_churn * len(treatment),
    }
    
    # ──────────────────────────────────────────────────
    # PART 2: OBSERVATIONAL (Crossed vs Not, within treatment)
    # This comparison is NOT causal — be explicit about this!
    # ──────────────────────────────────────────────────
    crossed = treatment[treatment["crossed_tier"] == True]
    not_crossed = treatment[treatment["crossed_tier"] == False]
    
    crossed_churn = crossed["churned_6mo"].mean() if len(crossed) > 0 else np.nan
    not_crossed_churn = not_crossed["churned_6mo"].mean() if len(not_crossed) > 0 else np.nan
    
    observational = {
        "crossed_churn_rate": crossed_churn,
        "not_crossed_churn_rate": not_crossed_churn,
        "n_crossed": len(crossed),
        "n_not_crossed": len(not_crossed),
        "raw_difference": not_crossed_churn - crossed_churn if not np.isnan(crossed_churn) else np.nan,
        "CAUTION": (
            "This comparison is OBSERVATIONAL, not causal. "
            "Clients who crossed the tier are self-selected — they may be "
            "growing firms that would have churned less regardless. "
            "Do NOT claim crossing the tier CAUSED lower churn based on this alone."
        ),
    }
    
    # ──────────────────────────────────────────────────
    # PART 3: INSTRUMENTAL VARIABLE ESTIMATE (advanced)
    # Treatment assignment instruments for tier crossing.
    # Wald estimator: LATE = ΔChurn / ΔCrossing
    # ──────────────────────────────────────────────────
    t_cross_rate = treatment["crossed_tier"].mean()
    c_cross_rate = control["crossed_tier"].mean()
    
    iv_estimate = None
    if t_cross_rate != c_cross_rate:
        # Wald (IV) estimator
        # = (effect of treatment on churn) / (effect of treatment on crossing)
        late = ate_churn / (t_cross_rate - c_cross_rate)
        iv_estimate = {
            "late": late,  # Local Average Treatment Effect of CROSSING
            "treatment_cross_rate": t_cross_rate,
            "control_cross_rate": c_cross_rate,
            "first_stage_strength": t_cross_rate - c_cross_rate,
            "interpretation": (
                f"For clients who crossed a tier BECAUSE of the treatment "
                f"(the 'compliers'), crossing reduced churn probability by "
                f"{late:.1%}. This is the causal effect of tier crossing, "
                f"instrumented by random treatment assignment."
            ),
        }
    
    return {
        "causal": causal,
        "observational": observational,
        "iv_estimate": iv_estimate,
    }


def compute_ltv_impact(
    churn_results: dict,
    avg_annual_revenue_per_client: float = 150_000,
    n_treatment_eligible: int = 425,
) -> dict:
    """
    Translate churn reduction into dollar impact.
    
    Args:
        churn_results: Output from analyze_churn()
        avg_annual_revenue_per_client: Average annual fee revenue per client
        n_treatment_eligible: Number of clients who would get the treatment
    """
    causal = churn_results["causal"]
    
    churns_prevented = causal["absolute_reduction"] * n_treatment_eligible
    annual_value_saved = churns_prevented * avg_annual_revenue_per_client
    
    return {
        "churns_prevented_per_year": churns_prevented,
        "annual_retention_value": annual_value_saved,
        "avg_client_ltv": avg_annual_revenue_per_client,
        "interpretation": (
            f"The treatment prevents ~{churns_prevented:.0f} churns per 6 months, "
            f"worth ~${annual_value_saved:,.0f}/year in retained revenue."
        ),
    }
