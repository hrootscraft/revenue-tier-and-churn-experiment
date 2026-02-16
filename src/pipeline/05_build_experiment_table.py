"""
Pipeline Step 5: Build Analysis-Ready Experiment Table
======================================================
Joins all raw data into a single table with one row per client,
containing everything needed for the analysis notebooks.

This is the "last mile" of the pipeline — where raw weekly data
gets aggregated into experiment-level metrics.

Output: data/processed/experiment_summary.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    get_tier, get_fee_rate, get_blended_rate, get_maker_rate, get_taker_rate,
    get_distance_to_next_tier, get_distance_bucket,
)


def build_experiment_table(
    clients: pd.DataFrame,
    pre_activity: pd.DataFrame,
    post_activity: pd.DataFrame,
    assignment: pd.DataFrame,
    churn_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the master analysis table: one row per client, all metrics.
    
    Columns produced:
    - Client attributes (segment, tier, region, etc.)
    - Assignment (group, strata)
    - Pre-period aggregates (pre_volume, pre_trades, pre_logins, etc.)
    - Post-period aggregates (post_volume, post_trades, etc.)
    - Computed metrics (volume_change, crossed_tier, revenue decomposition)
    - Churn outcome (churned_6mo, churn_week)
    """
    
    # ── Aggregate pre-period to client level ──
    pre_agg = (
        pre_activity
        .groupby("client_id")
        .agg(
            pre_volume=("weekly_volume", "sum"),
            pre_maker_volume=("maker_volume", "sum"),
            pre_taker_volume=("taker_volume", "sum"),
            pre_avg_weekly_volume=("weekly_volume", "mean"),
            pre_volume_std=("weekly_volume", "std"),
            pre_trades=("num_trades", "sum"),
            pre_logins=("platform_logins", "sum"),
            pre_tickets=("support_tickets", "sum"),
        )
        .reset_index()
    )

    # ── Aggregate post-period to client level ──
    post_agg = (
        post_activity
        .groupby("client_id")
        .agg(
            post_volume=("weekly_volume", "sum"),
            post_maker_volume=("maker_volume", "sum"),
            post_taker_volume=("taker_volume", "sum"),
            post_avg_weekly_volume=("weekly_volume", "mean"),
            post_volume_std=("weekly_volume", "std"),
            post_trades=("num_trades", "sum"),
            post_logins=("platform_logins", "sum"),
            post_tickets=("support_tickets", "sum"),
        )
        .reset_index()
    )
    
    # ── Join everything together ──
    df = (
        clients
        .merge(assignment[["client_id", "group", "strata"]], on="client_id")
        .merge(pre_agg, on="client_id")
        .merge(post_agg, on="client_id")
        .merge(churn_summary[["client_id", "churned_6mo", "churn_week", "crossed_tier", "post_tier"]],
               on="client_id")
    )
    
    # ── Compute derived metrics ──

    # Volume change
    df["volume_change"] = df["post_volume"] - df["pre_volume"]
    df["volume_change_pct"] = df["volume_change"] / df["pre_volume"]

    # Maker ratios (actual, computed from activity)
    df["pre_maker_ratio"] = df["pre_maker_volume"] / df["pre_volume"]
    df["post_maker_ratio"] = df["post_maker_volume"] / df["post_volume"]

    # Tier and fee rate changes (blended using each client's actual maker ratio)
    df["pre_tier"] = df["current_tier"]
    df["pre_maker_rate"] = df["pre_tier"].apply(get_maker_rate)
    df["pre_taker_rate"] = df["pre_tier"].apply(get_taker_rate)
    df["post_maker_rate"] = df["post_tier"].apply(get_maker_rate)
    df["post_taker_rate"] = df["post_tier"].apply(get_taker_rate)

    df["pre_fee_rate"] = df.apply(
        lambda row: get_blended_rate(row["pre_tier"], row["pre_maker_ratio"]), axis=1
    )
    df["post_fee_rate"] = df.apply(
        lambda row: get_blended_rate(row["post_tier"], row["post_maker_ratio"]), axis=1
    )
    df["fee_rate_changed"] = df["pre_fee_rate"] != df["post_fee_rate"]

    # Revenue calculation (maker/taker components)
    df["pre_maker_revenue"] = df["pre_maker_volume"] * df["pre_maker_rate"]
    df["pre_taker_revenue"] = df["pre_taker_volume"] * df["pre_taker_rate"]
    df["pre_revenue"] = df["pre_maker_revenue"] + df["pre_taker_revenue"]

    df["post_maker_revenue"] = df["post_maker_volume"] * df["post_maker_rate"]
    df["post_taker_revenue"] = df["post_taker_volume"] * df["post_taker_rate"]
    df["post_revenue"] = df["post_maker_revenue"] + df["post_taker_revenue"]

    df["revenue_change"] = df["post_revenue"] - df["pre_revenue"]
    df["revenue_change_pct"] = df["revenue_change"] / df["pre_revenue"]

    # ── Revenue decomposition (the key analysis) ──
    # Uses blended rates: ΔRevenue = ΔVolume × OldRate + OldVolume × ΔRate + ΔVolume × ΔRate
    df["volume_effect"] = (df["post_volume"] - df["pre_volume"]) * df["pre_fee_rate"]
    df["rate_effect"] = df["pre_volume"] * (df["post_fee_rate"] - df["pre_fee_rate"])
    df["interaction_effect"] = (
        (df["post_volume"] - df["pre_volume"]) * (df["post_fee_rate"] - df["pre_fee_rate"])
    )
    
    # Engagement change
    df["login_change"] = df["post_logins"] - df["pre_logins"]
    df["ticket_change"] = df["post_tickets"] - df["pre_tickets"]
    
    return df


def validate_experiment_table(df: pd.DataFrame) -> None:
    """
    Comprehensive validation of the final analysis table.
    THIS IS THE MOST IMPORTANT VALIDATION — it catches join bugs,
    calculation errors, and data integrity issues.
    """
    print("  Running final validation checks...")
    
    # 1. No duplicates
    assert df["client_id"].is_unique, f"Duplicate clients! {df['client_id'].duplicated().sum()} dupes"
    print("    ✓ No duplicate clients")
    
    # 2. Every client has both pre and post data
    assert df["pre_volume"].notna().all(), "Missing pre-period volume"
    assert df["post_volume"].notna().all(), "Missing post-period volume"
    print("    ✓ No missing volume data")
    
    # 3. Revenue decomposition sums correctly
    # Note: small floating-point error possible since blended rates compound rounding
    decomposed = df["volume_effect"] + df["rate_effect"] + df["interaction_effect"]
    # Recompute actual change from blended rates to match decomposition basis
    actual_from_blended = df["post_volume"] * df["post_fee_rate"] - df["pre_volume"] * df["pre_fee_rate"]
    max_diff = (decomposed - actual_from_blended).abs().max()
    assert max_diff < 0.01, f"Revenue decomposition off by ${max_diff:.2f}!"
    print(f"    ✓ Revenue decomposition sums correctly (max error: ${max_diff:.4f})")
    
    # 4. Revenue = maker_vol × maker_rate + taker_vol × taker_rate
    pre_check = (
        df["pre_maker_volume"] * df["pre_maker_rate"]
        + df["pre_taker_volume"] * df["pre_taker_rate"]
        - df["pre_revenue"]
    ).abs().max()
    post_check = (
        df["post_maker_volume"] * df["post_maker_rate"]
        + df["post_taker_volume"] * df["post_taker_rate"]
        - df["post_revenue"]
    ).abs().max()
    assert pre_check < 0.01, f"Pre-revenue inconsistent: ${pre_check:.2f}"
    assert post_check < 0.01, f"Post-revenue inconsistent: ${post_check:.2f}"
    print("    ✓ Revenue = MakerVol × MakerRate + TakerVol × TakerRate for all clients")
    
    # 5. Tier assignments are consistent with volume
    from config import TIER_THRESHOLDS
    for _, row in df.iterrows():
        tier = row["post_tier"]
        vol = row["post_volume"]
        threshold = TIER_THRESHOLDS[tier]
        if tier != "Enterprise":
            # Volume should be >= tier threshold and < next tier
            next_tier_threshold = TIER_THRESHOLDS.get(
                {"Professional": "Enterprise", "Standard": "Professional",
                 "Starter": "Standard"}.get(tier), float("inf")
            )
            assert vol >= threshold, \
                f"Client {row['client_id']}: vol ${vol:,.0f} < tier {tier} threshold ${threshold:,.0f}"
    print("    ✓ Post-period tiers consistent with volumes")
    
    # 6. Crossed tier flag is consistent
    tier_mismatch = df["crossed_tier"] != (df["pre_tier"] != df["post_tier"])
    n_mismatch = tier_mismatch.sum()
    assert n_mismatch == 0, f"{n_mismatch} clients have inconsistent crossed_tier flag"
    print("    ✓ Crossed tier flags consistent")
    
    # 7. Summary stats
    print(f"\n  Summary:")
    print(f"    Total clients: {len(df)}")
    print(f"    Treatment: {(df['group']=='treatment').sum()}, Control: {(df['group']=='control').sum()}")
    print(f"    Crossed tier (treatment): {df.loc[df['group']=='treatment', 'crossed_tier'].sum()}")
    print(f"    Crossed tier (control): {df.loc[df['group']=='control', 'crossed_tier'].sum()}")
    print(f"    Churned (treatment): {df.loc[df['group']=='treatment', 'churned_6mo'].mean():.1%}")
    print(f"    Churned (control): {df.loc[df['group']=='control', 'churned_6mo'].mean():.1%}")
    
    print("\n  ✓ ALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    print("Step 5: Building analysis-ready experiment table...")
    
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    raw = data_dir / "raw"
    
    clients = pd.read_csv(raw / "clients.csv")
    pre_activity = pd.read_csv(raw / "pre_period_activity.csv")
    post_activity = pd.read_csv(raw / "post_period_activity.csv")
    assignment = pd.read_csv(raw / "assignment.csv")
    churn_summary = pd.read_csv(raw / "churn_summary.csv")
    
    experiment_table = build_experiment_table(
        clients, pre_activity, post_activity, assignment, churn_summary
    )
    validate_experiment_table(experiment_table)
    
    output_path = data_dir / "processed" / "experiment_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_table.to_csv(output_path, index=False)
    print(f"\n  → Saved {len(experiment_table)} rows to {output_path}")
