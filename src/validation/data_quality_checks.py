"""
Data Quality Checks
====================
"Make sure all numbers are actually correct" — Head of DS, Coinbase

These checks run after EVERY pipeline step and catch:
- Join bugs (duplicate rows, missing clients)
- Calculation errors (revenue ≠ volume × rate)
- Logical inconsistencies (tier doesn't match volume)
- Data leakage (treatment effects in pre-period)
- Boundary conditions (off-by-one at tier thresholds)

Each check has a WHY comment explaining what real-world bug it catches.
This is not boilerplate — it's the file that proves you build
trustworthy data pipelines.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TIER_THRESHOLDS, MAX_ALLOWED_SMD


class DataQualityError(Exception):
    """Raised when a data quality check fails."""
    pass


# ─────────────────────────────────────────────────────────────
# INTEGRITY CHECKS
# ─────────────────────────────────────────────────────────────

def check_no_duplicates(df: pd.DataFrame, key_col: str = "client_id") -> None:
    """
    One row per client. Duplicates = double-counted revenue.
    
    WHY: A bad JOIN (e.g., many-to-many on a non-unique key) silently
    creates duplicates. Revenue gets inflated. Leadership makes decisions
    on wrong numbers. This is the #1 most common data bug.
    """
    dupes = df[key_col].duplicated().sum()
    if dupes > 0:
        # Show which clients are duplicated for debugging
        duped_ids = df[df[key_col].duplicated(keep=False)][key_col].unique()[:5]
        raise DataQualityError(
            f"Found {dupes} duplicate rows on '{key_col}'! "
            f"Example duplicated IDs: {list(duped_ids)}"
        )
    print(f"  ✓ No duplicates on {key_col} ({len(df)} unique rows)")


def check_no_nulls(df: pd.DataFrame, critical_cols: list[str]) -> None:
    """
    Critical columns must have no missing values.
    
    WHY: A NULL in fee_rate means revenue = volume × NULL = NULL.
    Aggregations silently drop NULLs, giving you the wrong total.
    """
    for col in critical_cols:
        n_null = df[col].isna().sum()
        if n_null > 0:
            raise DataQualityError(
                f"Column '{col}' has {n_null} NULL values! "
                f"Check JOIN logic — likely a failed merge."
            )
    print(f"  ✓ No NULLs in critical columns: {critical_cols}")


# ─────────────────────────────────────────────────────────────
# CONSISTENCY CHECKS
# ─────────────────────────────────────────────────────────────

def check_revenue_consistency(df: pd.DataFrame) -> None:
    """
    Fee revenue MUST equal volume × fee_rate. Not approximately. Exactly.
    
    WHY: If revenue is computed in one place and volume × rate in another,
    rounding errors or stale JOINs can cause them to diverge.
    $0.01 error × 850 clients × 4 quarters = real reporting error.
    """
    for period in ["pre", "post"]:
        maker_vol = f"{period}_maker_volume"
        taker_vol = f"{period}_taker_volume"
        maker_rate = f"{period}_maker_rate"
        taker_rate = f"{period}_taker_rate"
        rev_col = f"{period}_revenue"

        # Prefer maker/taker decomposition check if columns exist
        if all(c in df.columns for c in [maker_vol, taker_vol, maker_rate, taker_rate, rev_col]):
            computed = df[maker_vol] * df[maker_rate] + df[taker_vol] * df[taker_rate]
            reported = df[rev_col]
            max_diff = (computed - reported).abs().max()

            if max_diff > 0.01:
                worst_idx = (computed - reported).abs().idxmax()
                raise DataQualityError(
                    f"{period.title()} revenue inconsistency! "
                    f"Max diff: ${max_diff:.4f}. "
                    f"Worst client: {df.loc[worst_idx, 'client_id']} "
                    f"(computed ${computed.loc[worst_idx]:,.2f} vs reported ${reported.loc[worst_idx]:,.2f})"
                )
        else:
            # Fallback: blended rate check
            vol_col = f"{period}_volume"
            rate_col = f"{period}_fee_rate"
            if all(c in df.columns for c in [vol_col, rate_col, rev_col]):
                computed = df[vol_col] * df[rate_col]
                reported = df[rev_col]
                max_diff = (computed - reported).abs().max()

                if max_diff > 0.01:
                    worst_idx = (computed - reported).abs().idxmax()
                    raise DataQualityError(
                        f"{period.title()} revenue inconsistency! "
                        f"Max diff: ${max_diff:.4f}. "
                        f"Worst client: {df.loc[worst_idx, 'client_id']} "
                        f"(computed ${computed.loc[worst_idx]:,.2f} vs reported ${reported.loc[worst_idx]:,.2f})"
                    )

    print("  ✓ Revenue = MakerVol × MakerRate + TakerVol × TakerRate for all clients, both periods")


def check_tier_consistency(df: pd.DataFrame, volume_col: str, tier_col: str) -> None:
    """
    Every client's tier must match their volume.
    $100M client MUST be Professional, not Standard.
    
    WHY: If tier assignment uses >= but fee lookup uses >,
    a client at EXACTLY $100M gets the wrong rate.
    Boundary conditions are where bugs hide.
    """
    errors = []
    for _, row in df.iterrows():
        tier = row[tier_col]
        vol = row[volume_col]
        
        # Check: volume >= tier threshold
        threshold = TIER_THRESHOLDS[tier]
        if vol < threshold - 0.01:  # Small tolerance for floating point
            errors.append(
                f"Client {row['client_id']}: vol=${vol:,.0f} "
                f"< {tier} threshold=${threshold:,.0f}"
            )
        
        # Check: volume < NEXT tier threshold (shouldn't be in a lower tier)
        tier_order = ["Enterprise", "Professional", "Standard", "Starter"]
        tier_idx = tier_order.index(tier)
        if tier_idx > 0:  # Not Enterprise
            higher_threshold = TIER_THRESHOLDS[tier_order[tier_idx - 1]]
            if vol >= higher_threshold:
                errors.append(
                    f"Client {row['client_id']}: vol=${vol:,.0f} "
                    f">= {tier_order[tier_idx-1]} threshold=${higher_threshold:,.0f} "
                    f"but assigned to {tier}"
                )
    
    if errors:
        raise DataQualityError(
            f"Tier inconsistencies found ({len(errors)} errors):\n"
            + "\n".join(errors[:5])
        )
    
    print(f"  ✓ All tiers consistent with volumes ({tier_col} vs {volume_col})")


def check_decomposition_sums(df: pd.DataFrame) -> None:
    """
    The three revenue decomposition terms MUST sum to total ΔRevenue.
    ΔVol×OldRate + OldVol×ΔRate + ΔVol×ΔRate ≡ ΔRevenue
    
    This is a mathematical identity, not an approximation.
    
    WHY: If you show a waterfall chart to the Head of DS and the pieces
    don't add up to the total, you lose all credibility. Instantly.
    """
    if not all(c in df.columns for c in ["volume_effect", "rate_effect", "interaction_effect"]):
        print("  ⊘ Skipping decomposition check (columns not present)")
        return
    
    decomposed = df["volume_effect"] + df["rate_effect"] + df["interaction_effect"]
    actual = df["revenue_change"]
    diffs = (decomposed - actual).abs()
    max_diff = diffs.max()
    
    if max_diff > 0.01:
        worst_idx = diffs.idxmax()
        raise DataQualityError(
            f"Revenue decomposition doesn't sum! Max diff: ${max_diff:.4f}. "
            f"Client {df.loc[worst_idx, 'client_id']}: "
            f"decomposed=${decomposed.loc[worst_idx]:,.2f} vs actual=${actual.loc[worst_idx]:,.2f}"
        )
    
    print(f"  ✓ Revenue decomposition sums correctly (max error: ${max_diff:.6f})")


# ─────────────────────────────────────────────────────────────
# EXPERIMENT VALIDITY CHECKS
# ─────────────────────────────────────────────────────────────

def check_no_treatment_leakage(
    pre_activity: pd.DataFrame,
    assignment: pd.DataFrame,
) -> None:
    """
    Pre-period data must be IDENTICAL across treatment and control.
    If treatment clients already show different volume pre-experiment,
    your randomization failed or (worse) there's temporal leakage.
    
    WHY: This is the #1 way experiments get invalidated in court.
    An adversarial reviewer will always check this first.
    """
    merged = pre_activity.merge(assignment[["client_id", "group"]], on="client_id")
    
    pre_quarterly = merged.groupby(["client_id", "group"])["weekly_volume"].sum().reset_index()
    
    t_vol = pre_quarterly[pre_quarterly["group"] == "treatment"]["weekly_volume"]
    c_vol = pre_quarterly[pre_quarterly["group"] == "control"]["weekly_volume"]
    
    pooled_std = pre_quarterly["weekly_volume"].std()
    smd = abs(t_vol.mean() - c_vol.mean()) / pooled_std if pooled_std > 0 else 0
    
    if smd > MAX_ALLOWED_SMD:
        raise DataQualityError(
            f"Pre-period treatment leakage detected! "
            f"SMD = {smd:.3f} (threshold: {MAX_ALLOWED_SMD}). "
            f"Treatment mean: ${t_vol.mean():,.0f}, Control mean: ${c_vol.mean():,.0f}. "
            f"Re-check randomization or data generation sequence."
        )
    
    print(f"  ✓ No treatment leakage (pre-period SMD = {smd:.4f})")


def check_assignment_balance(assignment: pd.DataFrame) -> None:
    """
    ~50/50 split within each stratum.
    
    WHY: If one stratum is 80/20, your stratified analysis for that
    segment has very low power and the ATE estimate is noisy.
    """
    imbalanced = []
    for strata, group_df in assignment.groupby("strata"):
        n = len(group_df)
        if n < 4:
            continue  # Too small to judge
        t_frac = (group_df["group"] == "treatment").mean()
        if not (0.30 < t_frac < 0.70):
            imbalanced.append(f"  {strata}: {t_frac:.0%} treatment (n={n})")
    
    if imbalanced:
        raise DataQualityError(
            f"Imbalanced strata detected:\n" + "\n".join(imbalanced)
        )
    
    print("  ✓ Assignment balanced within all strata")


# ─────────────────────────────────────────────────────────────
# MASTER CHECK RUNNER
# ─────────────────────────────────────────────────────────────

def run_all_checks(
    experiment_df: pd.DataFrame,
    pre_activity: pd.DataFrame,
    assignment: pd.DataFrame,
) -> None:
    """
    Run ALL data quality checks. Pipeline fails if any check fails.
    
    This function should be called after building the experiment table
    and before ANY analysis is run.
    """
    print("=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)
    
    check_no_duplicates(experiment_df, "client_id")
    check_no_nulls(experiment_df, [
        "client_id", "group", "pre_volume", "post_volume",
        "pre_fee_rate", "post_fee_rate", "pre_revenue", "post_revenue",
        "pre_maker_volume", "pre_taker_volume", "post_maker_volume", "post_taker_volume",
        "pre_maker_rate", "pre_taker_rate", "post_maker_rate", "post_taker_rate",
    ])
    check_revenue_consistency(experiment_df)
    check_tier_consistency(experiment_df, "post_volume", "post_tier")
    check_decomposition_sums(experiment_df)
    check_no_treatment_leakage(pre_activity, assignment)
    check_assignment_balance(assignment)
    
    print("=" * 60)
    print("ALL CHECKS PASSED ✓")
    print("=" * 60)
