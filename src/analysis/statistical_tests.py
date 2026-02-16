"""
Core Statistical Analysis Functions
====================================
Reusable functions for experiment analysis.
Each function is self-contained, tested, and documented.

These get imported into the analysis notebooks.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


# ─────────────────────────────────────────────────────────────
# AVERAGE TREATMENT EFFECT (ATE)
# ─────────────────────────────────────────────────────────────

def compute_ate(
    treatment: np.ndarray,
    control: np.ndarray,
) -> dict:
    """
    Compute the Average Treatment Effect with confidence interval.
    
    ATE = mean(treatment) - mean(control)
    
    Returns dict with: ate, se, ci_lower, ci_upper, t_mean, c_mean
    """
    t_mean = np.mean(treatment)
    c_mean = np.mean(control)
    ate = t_mean - c_mean
    
    # Standard error of difference in means
    se = np.sqrt(np.var(treatment, ddof=1) / len(treatment) +
                 np.var(control, ddof=1) / len(control))
    
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    
    return {
        "ate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "treatment_mean": t_mean,
        "control_mean": c_mean,
        "pct_lift": ate / c_mean * 100 if c_mean != 0 else np.nan,
    }


# ─────────────────────────────────────────────────────────────
# STATISTICAL TESTS (Three independent tests for robustness)
# ─────────────────────────────────────────────────────────────

def run_welch_ttest(treatment: np.ndarray, control: np.ndarray) -> dict:
    """
    Welch's t-test: parametric, handles unequal variance.
    Most standard test for comparing two means.
    """
    stat, pvalue = stats.ttest_ind(treatment, control, equal_var=False)
    return {"test": "Welch's t-test", "statistic": stat, "p_value": pvalue}


def run_mann_whitney(treatment: np.ndarray, control: np.ndarray) -> dict:
    """
    Mann-Whitney U test: non-parametric, robust to skew.
    Doesn't assume normal distribution — important for trading volume
    which is heavily right-skewed.
    """
    stat, pvalue = stats.mannwhitneyu(treatment, control, alternative="two-sided")
    return {"test": "Mann-Whitney U", "statistic": stat, "p_value": pvalue}


def run_permutation_test(
    treatment: np.ndarray,
    control: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Permutation test: assumption-free.
    
    Shuffles treatment/control labels 10,000 times and computes
    how often the random shuffle produces a difference as large
    as the observed one.
    
    If the observed difference is larger than 99.5% of random shuffles,
    p < 0.005. This is the gold standard for small samples.
    """
    rng = np.random.default_rng(seed)
    combined = np.concatenate([treatment, control])
    n_treatment = len(treatment)
    observed_diff = np.mean(treatment) - np.mean(control)
    
    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[:n_treatment]) - np.mean(combined[n_treatment:])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    
    pvalue = count_extreme / n_permutations
    return {
        "test": "Permutation",
        "statistic": observed_diff,
        "p_value": pvalue,
        "n_permutations": n_permutations,
    }


def run_all_tests(treatment: np.ndarray, control: np.ndarray) -> pd.DataFrame:
    """Run all three tests and return as a clean DataFrame."""
    results = [
        run_welch_ttest(treatment, control),
        run_mann_whitney(treatment, control),
        run_permutation_test(treatment, control),
    ]
    df = pd.DataFrame(results)
    df["significant_5pct"] = df["p_value"] < 0.05
    df["significant_1pct"] = df["p_value"] < 0.01
    return df


# ─────────────────────────────────────────────────────────────
# CUPED (Controlled-experiment Using Pre-Experiment Data)
# ─────────────────────────────────────────────────────────────

def apply_cuped(
    post_values: np.ndarray,
    pre_values: np.ndarray,
) -> np.ndarray:
    """
    CUPED variance reduction.
    
    The idea: pre-period volume predicts post-period volume.
    Remove this predictable component to reduce noise.
    
    adjusted_Y = Y - θ(X - X̄)
    where θ = Cov(Y, X) / Var(X)
    
    This doesn't change the expected treatment effect (both groups
    get the same adjustment), but reduces variance by removing
    the predictable component.
    
    Args:
        post_values: Post-period outcome (Y)
        pre_values:  Pre-period covariate (X)
    
    Returns:
        CUPED-adjusted post values
    """
    theta = np.cov(post_values, pre_values)[0, 1] / np.var(pre_values)
    adjusted = post_values - theta * (pre_values - np.mean(pre_values))
    return adjusted


def cuped_comparison(
    treatment_post: np.ndarray,
    control_post: np.ndarray,
    treatment_pre: np.ndarray,
    control_pre: np.ndarray,
) -> dict:
    """
    Compute ATE with and without CUPED, showing the variance reduction.
    
    Returns dict with raw_ate, cuped_ate, variance_reduction_pct,
    and both confidence intervals.
    """
    # Raw ATE
    raw = compute_ate(treatment_post, control_post)
    
    # CUPED-adjusted
    all_post = np.concatenate([treatment_post, control_post])
    all_pre = np.concatenate([treatment_pre, control_pre])
    all_adjusted = apply_cuped(all_post, all_pre)
    
    n_t = len(treatment_post)
    adj_treatment = all_adjusted[:n_t]
    adj_control = all_adjusted[n_t:]
    
    cuped = compute_ate(adj_treatment, adj_control)
    
    raw_var = raw["se"] ** 2
    cuped_var = cuped["se"] ** 2
    variance_reduction = 1 - cuped_var / raw_var if raw_var > 0 else 0
    
    return {
        "raw_ate": raw["ate"],
        "raw_se": raw["se"],
        "raw_ci": (raw["ci_lower"], raw["ci_upper"]),
        "cuped_ate": cuped["ate"],
        "cuped_se": cuped["se"],
        "cuped_ci": (cuped["ci_lower"], cuped["ci_upper"]),
        "variance_reduction_pct": variance_reduction * 100,
        "theta": np.cov(all_post, all_pre)[0, 1] / np.var(all_pre),
    }


# ─────────────────────────────────────────────────────────────
# BALANCE TABLE (Randomization Check)
# ─────────────────────────────────────────────────────────────

def compute_balance_table(
    df: pd.DataFrame,
    covariates: list[str],
    group_col: str = "group",
    treatment_label: str = "treatment",
    control_label: str = "control",
) -> pd.DataFrame:
    """
    Check covariate balance between treatment and control.
    
    For each covariate, computes:
    - Treatment mean, Control mean
    - Standardized Mean Difference (SMD)
    - p-value from t-test
    - Pass/Fail flag (SMD < 0.1 is standard threshold)
    
    This is the FIRST thing you show in any experiment analysis.
    If balance is bad, nothing downstream can be trusted.
    """
    treatment = df[df[group_col] == treatment_label]
    control = df[df[group_col] == control_label]
    
    rows = []
    for cov in covariates:
        t_vals = treatment[cov].dropna()
        c_vals = control[cov].dropna()
        
        t_mean = t_vals.mean()
        c_mean = c_vals.mean()
        
        pooled_std = df[cov].std()
        smd = abs(t_mean - c_mean) / pooled_std if pooled_std > 0 else 0
        
        _, p_value = stats.ttest_ind(t_vals, c_vals, equal_var=False)
        
        rows.append({
            "covariate": cov,
            "treatment_mean": t_mean,
            "control_mean": c_mean,
            "difference": t_mean - c_mean,
            "smd": smd,
            "p_value": p_value,
            "balanced": "✓ Pass" if smd < 0.1 else "✗ FAIL",
        })
    
    return pd.DataFrame(rows)
