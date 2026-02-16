"""
Pipeline Step 1: Generate Client Universe

Creates ~850 institutional clients with realistic attributes.

Key design decisions:
- AUM is log-normal (most clients are small, few are huge)
- ~30-40% of clients naturally cluster near tier boundaries (this is realistic — clients accumulate near thresholds)
- Segment distribution matches typical institutional crypto exchange

Output: data/raw/clients.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    N_CLIENTS, SEGMENTS, REGIONS, FEE_TIERS, TIER_ORDER, TIER_THRESHOLDS,
    RANDOM_SEED, PRE_PERIOD_WEEKS, SEGMENT_MAKER_RATIOS,
    get_tier, get_blended_rate, get_maker_rate, get_taker_rate,
    get_distance_to_next_tier, get_distance_bucket,
)


def generate_clients(n: int = N_CLIENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate the full client universe.
    
    Each client gets:
    - Identifiers: client_id, segment, sub_segment, region
    - Financial: aum, baseline_weekly_volume, current_tier, fee_rate
    - Behavioral: tenure_months, api_integration, primary_products
    - Tier distance: distance_to_next_tier_pct, distance_bucket
    
    Returns:
        pd.DataFrame with one row per client
    """
    rng = np.random.default_rng(seed)
    
    clients = []
    client_counter = 1
    
    for segment, (proportion, avg_aum, std_aum, vol_mult, vol_volatility) in SEGMENTS.items():
        n_segment = int(n * proportion)
        
        for _ in range(n_segment):
            client_id = f"CB-{client_counter:05d}"
            client_counter += 1
            
            # AUM: log-normal distribution
            # np.random.lognormal takes mean and sigma of the UNDERLYING normal
            log_mean = np.log(avg_aum) - 0.5 * (std_aum / avg_aum) ** 2
            log_std = np.sqrt(np.log(1 + (std_aum / avg_aum) ** 2))
            aum = rng.lognormal(log_mean, log_std)
            aum = np.clip(aum, 10e6, 5e9)  # Floor $10M, cap $5B
            
            # Baseline weekly volume derived from AUM and segment behavior
            base_weekly = aum * vol_mult
            # Add client-specific persistent volume tendency
            client_volume_factor = rng.lognormal(0, 0.3)
            base_weekly *= client_volume_factor
            
            # ─── IMPORTANT: Nudge ~35% of clients toward tier boundaries ───
            # This makes the experiment interesting. Without this, most clients
            # would be far from boundaries and the tier-crossing dynamics
            # wouldn't matter.
            quarterly_volume_est = base_weekly * PRE_PERIOD_WEEKS
            current_tier = get_tier(quarterly_volume_est)
            
            if rng.random() < 0.35 and current_tier != "Enterprise":
                # Push this client to be within 25% of the NEXT tier threshold
                next_tier_info = get_distance_to_next_tier(quarterly_volume_est, current_tier)
                if next_tier_info["next_threshold"] is not None:
                    target_pct = rng.uniform(0.75, 0.98)  # 75-98% of way to next tier
                    target_quarterly = next_tier_info["next_threshold"] * target_pct
                    base_weekly = target_quarterly / PRE_PERIOD_WEEKS
            
            # Region
            region = rng.choice(
                list(REGIONS.keys()),
                p=list(REGIONS.values())
            )
            
            # Tenure
            tenure_months = int(rng.uniform(3, 60))
            
            # API integration (Market Makers and Quant Funds almost always use API)
            api_prob = {"Market Maker": 0.95, "Quant Fund": 0.90, "Hedge Fund": 0.60,
                        "Asset Manager": 0.40, "Corporate Treasury": 0.25, "Family Office": 0.20}
            api_integration = rng.random() < api_prob[segment]
            
            # Products used
            all_products = ["Spot", "Derivatives", "Prime Financing", "Custody", "Staking"]
            n_products = rng.choice([1, 2, 3, 4, 5], p=[0.15, 0.30, 0.30, 0.15, 0.10])
            primary_products = list(rng.choice(all_products, size=n_products, replace=False))
            
            # Maker ratio: Beta distribution centered on segment typical ratio
            seg_maker = SEGMENT_MAKER_RATIOS[segment]
            # Parameterize Beta so mean = seg_maker, concentration ~30
            alpha = seg_maker * 30
            beta_param = (1 - seg_maker) * 30
            maker_ratio = rng.beta(alpha, beta_param)
            maker_ratio = np.clip(maker_ratio, 0.05, 0.95)

            clients.append({
                "client_id": client_id,
                "segment": segment,
                "region": region,
                "aum": round(aum, 2),
                "baseline_weekly_volume": round(base_weekly, 2),
                "volume_volatility": vol_volatility,
                "tenure_months": tenure_months,
                "api_integration": api_integration,
                "primary_products": "|".join(primary_products),
                "n_products": n_products,
                "maker_ratio": round(maker_ratio, 4),
            })
    
    df = pd.DataFrame(clients)
    
    # Compute tier-related fields from baseline volume
    df["est_quarterly_volume"] = df["baseline_weekly_volume"] * PRE_PERIOD_WEEKS
    df["current_tier"] = df["est_quarterly_volume"].apply(get_tier)
    df["maker_rate"] = df["current_tier"].apply(get_maker_rate)
    df["taker_rate"] = df["current_tier"].apply(get_taker_rate)
    df["blended_fee_rate"] = df.apply(
        lambda row: get_blended_rate(row["current_tier"], row["maker_ratio"]), axis=1
    )
    df["fee_rate"] = df["blended_fee_rate"]  # backward-compatible alias
    
    # Distance to next tier
    tier_info = df.apply(
        lambda row: get_distance_to_next_tier(row["est_quarterly_volume"], row["current_tier"]),
        axis=1
    )
    tier_info_df = pd.DataFrame(tier_info.tolist())
    df = pd.concat([df, tier_info_df], axis=1)
    df["distance_bucket"] = df["distance_pct"].apply(get_distance_bucket)
    
    return df


def validate_clients(df: pd.DataFrame) -> None:
    """Quick sanity checks on generated client data."""
    assert len(df) >= N_CLIENTS * 0.95, f"Expected ~{N_CLIENTS} clients, got {len(df)}"
    assert df["client_id"].is_unique, "Duplicate client IDs!"
    assert df["aum"].min() >= 10e6, "AUM below $10M floor"
    assert df["baseline_weekly_volume"].min() > 0, "Negative or zero volume"
    
    # Check tier distribution is reasonable
    tier_counts = df["current_tier"].value_counts(normalize=True)
    print(f"  Tier distribution: {tier_counts.to_dict()}")
    
    # Check that we have clients near boundaries
    near_boundary = df[df["distance_bucket"].isin(["within_10", "10_to_20"])].shape[0]
    near_pct = near_boundary / len(df)
    print(f"  Clients within 20% of next tier: {near_boundary} ({near_pct:.0%})")
    assert near_pct > 0.20, f"Only {near_pct:.0%} near boundaries — need more for interesting HTE"
    
    print("  ✓ Client universe validated")


if __name__ == "__main__":
    print("Step 1: Generating client universe...")
    clients = generate_clients()
    validate_clients(clients)
    
    output_path = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "clients.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clients.to_csv(output_path, index=False)
    print(f"  → Saved {len(clients)} clients to {output_path}")
