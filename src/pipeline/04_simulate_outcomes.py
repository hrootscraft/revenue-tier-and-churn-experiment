"""
Pipeline Step 4: Simulate Experiment Outcomes
=============================================
Generates 13 weeks of post-period data WITH treatment effects applied.
Also generates 26 weeks of follow-up data for churn analysis.

This is the most complex step because it layers:
1. Baseline volume (from pre-period patterns)
2. Treatment effects (varying by distance-to-tier and segment)
3. Weekly decay (effect fades ~2% per week)
4. Confounders (market vol spike, competitor promo, seasonal rally)
5. Individual noise (not everyone responds the same)
6. Churn outcomes (6 months post-experiment)

The analysis pipeline must UNTANGLE these layers to recover the true
treatment effect. If the data generator makes this trivial, the project
is worthless for interview prep.

Output: data/raw/post_period_activity.csv
        data/raw/post_experiment_followup.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    EXPERIMENT_WEEKS, FOLLOWUP_WEEKS, RANDOM_SEED,
    TREATMENT_EFFECTS, WEEKLY_DECAY_RATE, SEGMENT_RESPONSE_MODIFIERS,
    CONFOUNDERS, BASE_QUARTERLY_CHURN_RATES,
    CHURN_REDUCTION_CROSSED_TIER, CHURN_REDUCTION_NOT_CROSSED,
    TREATMENT_MAKER_RATIO_SHIFT,
    get_tier, get_fee_rate,
)


def simulate_post_period(
    clients: pd.DataFrame,
    pre_activity: pd.DataFrame,
    assignment: pd.DataFrame,
    seed: int = RANDOM_SEED + 100,  # Different seed from pre-period!
) -> pd.DataFrame:
    """
    Simulate weekly trading activity during the experiment period.
    
    Treatment clients get a volume lift that:
    - Depends on their distance to the next fee tier
    - Is modified by their segment (MMs care less)
    - Decays ~2% per week (novelty wears off)
    - Has individual-level noise (some clients respond negatively)
    
    All clients (treatment AND control) are affected by confounders:
    - Market volatility spike in weeks 6-8
    - Competitor promotion from week 4
    - Seasonal rally in weeks 10-13
    """
    rng = np.random.default_rng(seed)
    
    # Merge client info with assignment
    client_info = clients.merge(assignment[["client_id", "group"]], on="client_id")
    
    records = []
    
    for _, client in client_info.iterrows():
        base_vol = client["baseline_weekly_volume"]
        volatility = client["volume_volatility"]
        segment = client["segment"]
        group = client["group"]
        distance_bucket = client.get("distance_bucket", "20_plus")
        current_tier = client["current_tier"]
        
        # Determine this client's individual treatment response
        # (fixed for all weeks — a client-level random effect)
        if group == "treatment" and current_tier != "Enterprise":
            bucket_key = distance_bucket if distance_bucket in TREATMENT_EFFECTS else "20_plus"
            mean_lift, std_lift = TREATMENT_EFFECTS[bucket_key]
            segment_modifier = SEGMENT_RESPONSE_MODIFIERS.get(segment, 1.0)
            
            # This client's personal response to treatment
            # Drawn ONCE and applied to all weeks (persistent individual effect)
            client_lift = rng.normal(mean_lift * segment_modifier, std_lift)
        elif group == "treatment" and current_tier == "Enterprise":
            mean_lift, std_lift = TREATMENT_EFFECTS["enterprise"]
            client_lift = rng.normal(mean_lift, std_lift)
        else:
            client_lift = 0.0
        
        for week in range(1, EXPERIMENT_WEEKS + 1):
            # Start with baseline + random noise
            noise = rng.normal(0, volatility)
            weekly_volume = base_vol * (1 + noise)
            
            # ── Apply treatment effect (with weekly decay) ──
            if group == "treatment":
                decay_factor = 1.0 - WEEKLY_DECAY_RATE * (week - 1)
                decay_factor = max(decay_factor, 0.3)  # Floor at 30% of original effect
                weekly_lift = client_lift * decay_factor
                weekly_volume *= (1 + weekly_lift)
            
            # ── Apply confounders (affect BOTH groups) ──
            
            # Volatility spike (weeks 6-8)
            vol_conf = CONFOUNDERS["volatility_spike"]
            if vol_conf["weeks"][0] <= week <= vol_conf["weeks"][1]:
                mult = vol_conf["segment_multipliers"].get(segment, 1.0)
                weekly_volume *= mult
            
            # Competitor promotion (from week 4, Starter tier only)
            comp_conf = CONFOUNDERS["competitor_promo"]
            if week >= comp_conf["start_week"] and current_tier in comp_conf["affected_tiers"]:
                weekly_volume *= comp_conf["volume_multiplier"]
            
            # Seasonal rally (weeks 10-13)
            rally_conf = CONFOUNDERS["seasonal_rally"]
            if rally_conf["weeks"][0] <= week <= rally_conf["weeks"][1]:
                weekly_volume *= rally_conf["volume_multiplier"]
            
            weekly_volume = max(0, weekly_volume)
            
            # Trades and logins (same logic as pre-period)
            avg_trade_size = _segment_trade_size(segment) * rng.lognormal(0, 0.3)
            num_trades = max(1, int(weekly_volume / avg_trade_size + rng.normal(0, 3)))
            
            platform_logins = rng.poisson(2 if client["api_integration"] else 8)
            # Treatment clients might login slightly more (checking progress bar)
            if group == "treatment":
                platform_logins += rng.poisson(1.5)
            
            support_tickets = rng.poisson(0.12)
            
            # Split into maker/taker volume
            base_maker_ratio = client.get("maker_ratio", 0.40)
            # Treatment clients shift toward taker (more aggressive to hit tier)
            if group == "treatment":
                effective_maker_ratio = base_maker_ratio + TREATMENT_MAKER_RATIO_SHIFT
            else:
                effective_maker_ratio = base_maker_ratio
            ratio_noise = rng.normal(0, 0.03)
            effective_maker_ratio = np.clip(effective_maker_ratio + ratio_noise, 0.05, 0.95)
            maker_volume = weekly_volume * effective_maker_ratio
            taker_volume = weekly_volume - maker_volume

            records.append({
                "client_id": client["client_id"],
                "week": week,
                "period": "experiment",
                "weekly_volume": round(weekly_volume, 2),
                "maker_volume": round(maker_volume, 2),
                "taker_volume": round(taker_volume, 2),
                "num_trades": num_trades,
                "platform_logins": platform_logins,
                "support_tickets": support_tickets,
            })
    
    return pd.DataFrame(records)


def simulate_followup(
    clients: pd.DataFrame,
    assignment: pd.DataFrame,
    post_activity: pd.DataFrame,
    seed: int = RANDOM_SEED + 200,
) -> pd.DataFrame:
    """
    Simulate 26 weeks of post-experiment follow-up.
    
    Used for:
    - Churn analysis (secondary outcome)
    - Durability: does the volume lift persist after the progress bar is gone?
    - Tier retention: do clients maintain their new tier?
    
    Churn is modeled as:
    - Base rate by segment
    - Reduced by treatment (engagement effect)
    - Further reduced if client crossed a tier (golden handcuff)
    - With noise so it's not deterministic
    """
    rng = np.random.default_rng(seed)
    
    # First, compute who crossed a tier during the experiment
    post_quarterly = (
        post_activity
        .groupby("client_id")["weekly_volume"]
        .sum()
        .reset_index()
        .rename(columns={"weekly_volume": "experiment_quarterly_volume"})
    )
    
    client_info = (
        clients
        .merge(assignment[["client_id", "group"]], on="client_id")
        .merge(post_quarterly, on="client_id")
    )
    client_info["post_tier"] = client_info["experiment_quarterly_volume"].apply(get_tier)
    client_info["crossed_tier"] = client_info["post_tier"] != client_info["current_tier"]
    
    records = []
    churn_records = []
    
    for _, client in client_info.iterrows():
        segment = client["segment"]
        group = client["group"]
        crossed = client["crossed_tier"]
        
        # ── Compute churn probability ──
        base_churn_6mo = 1 - (1 - BASE_QUARTERLY_CHURN_RATES[segment]) ** 2
        
        if group == "treatment":
            if crossed:
                churn_reduction = CHURN_REDUCTION_CROSSED_TIER
            else:
                churn_reduction = CHURN_REDUCTION_NOT_CROSSED
        else:
            churn_reduction = 0.0
        
        # Add individual noise to churn probability
        noise = rng.normal(0, 0.05)
        final_churn_prob = base_churn_6mo * (1 - churn_reduction) + noise
        final_churn_prob = np.clip(final_churn_prob, 0.01, 0.50)
        
        did_churn = rng.random() < final_churn_prob
        
        # If they churn, pick a week (weighted toward later weeks)
        if did_churn:
            # Churn is more likely later (exponential weights)
            week_weights = np.exp(np.linspace(0, 2, FOLLOWUP_WEEKS))
            week_weights /= week_weights.sum()
            churn_week = rng.choice(range(1, FOLLOWUP_WEEKS + 1), p=week_weights)
        else:
            churn_week = None
        
        churn_records.append({
            "client_id": client["client_id"],
            "group": group,
            "crossed_tier": crossed,
            "post_tier": client["post_tier"],
            "churned_6mo": did_churn,
            "churn_week": churn_week,
        })
        
        # Generate weekly follow-up activity
        base_vol = client["baseline_weekly_volume"]
        volatility = client["volume_volatility"]
        
        for week in range(1, FOLLOWUP_WEEKS + 1):
            if did_churn and week >= churn_week:
                # Client has churned — zero activity
                weekly_volume = 0.0
                num_trades = 0
                platform_logins = 0
            else:
                # Volume gradually reverts toward pre-experiment baseline
                # (progress bar is no longer showing, but tier benefit remains)
                reversion_rate = 0.05  # 5% reversion per week toward baseline
                if group == "treatment" and crossed:
                    # Tier-crossers maintain most of their volume (golden handcuff)
                    reversion_rate = 0.02
                
                experiment_vol = client["experiment_quarterly_volume"] / EXPERIMENT_WEEKS
                current_target = experiment_vol - reversion_rate * week * (experiment_vol - base_vol)
                current_target = max(current_target, base_vol * 0.8)
                
                noise = rng.normal(0, volatility)
                weekly_volume = current_target * (1 + noise)
                weekly_volume = max(0, weekly_volume)
                
                avg_trade_size = _segment_trade_size(segment) * rng.lognormal(0, 0.3)
                num_trades = max(0, int(weekly_volume / avg_trade_size + rng.normal(0, 2)))
                platform_logins = rng.poisson(2 if client["api_integration"] else 6)
            
            # Split into maker/taker volume
            base_maker_ratio = client.get("maker_ratio", 0.40)
            if did_churn and week >= churn_week:
                maker_volume = 0.0
                taker_volume = 0.0
            else:
                ratio_noise = rng.normal(0, 0.03)
                eff_ratio = np.clip(base_maker_ratio + ratio_noise, 0.05, 0.95)
                maker_volume = weekly_volume * eff_ratio
                taker_volume = weekly_volume - maker_volume

            records.append({
                "client_id": client["client_id"],
                "week": week,
                "period": "followup",
                "weekly_volume": round(weekly_volume, 2),
                "maker_volume": round(maker_volume, 2),
                "taker_volume": round(taker_volume, 2),
                "num_trades": num_trades,
                "platform_logins": platform_logins,
                "support_tickets": 0,
            })
    
    followup_activity = pd.DataFrame(records)
    churn_summary = pd.DataFrame(churn_records)
    
    return followup_activity, churn_summary


def _segment_trade_size(segment: str) -> float:
    """Average trade size by segment."""
    sizes = {
        "Market Maker": 50_000, "Quant Fund": 100_000,
        "Hedge Fund": 250_000, "Asset Manager": 500_000,
        "Corporate Treasury": 1_000_000, "Family Office": 300_000,
    }
    return sizes.get(segment, 200_000)


def validate_post_period(df: pd.DataFrame, clients: pd.DataFrame) -> None:
    """Sanity checks on post-period data."""
    expected_rows = len(clients) * EXPERIMENT_WEEKS
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    assert df["weekly_volume"].min() >= 0, "Negative volume detected"
    print("  ✓ Post-period activity validated")


if __name__ == "__main__":
    print("Step 4: Simulating experiment outcomes...")
    
    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    clients = pd.read_csv(data_dir / "clients.csv")
    assignment = pd.read_csv(data_dir / "assignment.csv")
    pre_activity = pd.read_csv(data_dir / "pre_period_activity.csv")
    
    post_activity = simulate_post_period(clients, pre_activity, assignment)
    validate_post_period(post_activity, clients)
    post_activity.to_csv(data_dir / "post_period_activity.csv", index=False)
    print(f"  → Saved {len(post_activity)} experiment records")
    
    print("  Simulating 6-month follow-up...")
    followup_activity, churn_summary = simulate_followup(
        clients, assignment, post_activity
    )
    followup_activity.to_csv(data_dir / "post_experiment_followup.csv", index=False)
    churn_summary.to_csv(data_dir / "churn_summary.csv", index=False)
    print(f"  → Saved {len(followup_activity)} follow-up records")
    print(f"  → Saved {len(churn_summary)} churn outcomes")
    print(f"  → Churn rate: {churn_summary['churned_6mo'].mean():.1%}")
