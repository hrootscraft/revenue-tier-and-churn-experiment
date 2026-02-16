"""
Pipeline Step 3: Treatment Assignment (Stratified Randomization)
================================================================
Randomly assigns clients to treatment or control groups.
Uses stratified randomization to ensure balance on key variables.

Why stratified (not simple random):
- With only 850 clients, simple random could produce imbalanced groups
  (e.g., all Enterprise clients in treatment by chance)
- Stratification guarantees balance on the variables that matter most:
  tier, distance to next tier, and segment
- This also improves statistical power (smaller residual variance)

Output: data/raw/assignment.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RANDOM_SEED, STRATIFICATION_VARS, MAX_ALLOWED_SMD


def stratified_randomize(
    clients: pd.DataFrame,
    strat_vars: list[str] = STRATIFICATION_VARS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Assign clients to treatment/control using stratified randomization.
    
    Within each stratum (defined by the combination of strat_vars),
    randomly assign ~50% to treatment and ~50% to control.
    
    Args:
        clients: Client dataframe with columns matching strat_vars
        strat_vars: Variables to stratify on
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame with columns: client_id, group, strata
    """
    rng = np.random.default_rng(seed)
    
    assignments = []
    
    # Create strata from combination of stratification variables
    clients = clients.copy()
    clients["strata"] = clients[strat_vars].astype(str).agg("_".join, axis=1)
    
    for strata_name, strata_df in clients.groupby("strata"):
        n = len(strata_df)
        n_treatment = n // 2
        
        # Shuffle and assign
        indices = strata_df.index.tolist()
        rng.shuffle(indices)
        
        for i, idx in enumerate(indices):
            assignments.append({
                "client_id": clients.loc[idx, "client_id"],
                "group": "treatment" if i < n_treatment else "control",
                "strata": strata_name,
            })
    
    return pd.DataFrame(assignments)


def validate_assignment(assignment: pd.DataFrame, clients: pd.DataFrame) -> None:
    """
    Check that assignment is valid and balanced.
    
    Critical checks:
    1. Every client is assigned exactly once
    2. ~50/50 split overall
    3. ~50/50 split within each stratum
    4. No systematic imbalance on pre-period covariates
    """
    # Check 1: Every client assigned exactly once
    assert set(assignment["client_id"]) == set(clients["client_id"]), \
        "Assignment doesn't cover all clients!"
    assert assignment["client_id"].is_unique, "Duplicate assignments!"
    
    # Check 2: Overall balance
    t_frac = (assignment["group"] == "treatment").mean()
    print(f"  Overall split: {t_frac:.1%} treatment / {1-t_frac:.1%} control")
    assert 0.45 < t_frac < 0.55, f"Overall split too imbalanced: {t_frac:.1%}"
    
    # Check 3: Within-stratum balance
    for strata, group_df in assignment.groupby("strata"):
        strata_t_frac = (group_df["group"] == "treatment").mean()
        if len(group_df) >= 4:  # Only check strata with enough clients
            assert 0.30 < strata_t_frac < 0.70, \
                f"Stratum '{strata}' imbalanced: {strata_t_frac:.0%} treatment (n={len(group_df)})"
    
    # Check 4: Covariate balance (join with client data)
    merged = clients.merge(assignment, on="client_id")
    numeric_cols = ["aum", "baseline_weekly_volume", "tenure_months", "n_products"]
    
    print("  Covariate balance:")
    for col in numeric_cols:
        t_mean = merged.loc[merged["group"] == "treatment", col].mean()
        c_mean = merged.loc[merged["group"] == "control", col].mean()
        pooled_std = merged[col].std()
        smd = abs(t_mean - c_mean) / pooled_std if pooled_std > 0 else 0
        status = "✓" if smd < MAX_ALLOWED_SMD else "✗ FAIL"
        print(f"    {col}: T={t_mean:,.0f}, C={c_mean:,.0f}, SMD={smd:.3f} {status}")
        
        if smd >= MAX_ALLOWED_SMD:
            print(f"    ⚠ WARNING: {col} SMD ({smd:.3f}) exceeds threshold ({MAX_ALLOWED_SMD})")
    
    print("  ✓ Assignment validated")


if __name__ == "__main__":
    print("Step 3: Assigning treatment groups...")
    
    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    clients = pd.read_csv(data_dir / "clients.csv")
    
    assignment = stratified_randomize(clients)
    validate_assignment(assignment, clients)
    
    output_path = data_dir / "assignment.csv"
    assignment.to_csv(output_path, index=False)
    print(f"  → Saved {len(assignment)} assignments to {output_path}")
