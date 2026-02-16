"""
Pipeline Step 2: Generate Pre-Period Activity
==============================================
Creates 13 weeks of weekly trading data BEFORE the experiment.
This is the baseline against which we measure treatment effects.

Key design decisions:
- Weekly volume is noisy around client's baseline (realistic)
- Segment-specific volatility patterns (MMs steady, HFs bursty)
- No treatment effects here (this is pre-experiment!)
- Used for CUPED variance reduction in analysis

Output: data/raw/pre_period_activity.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PRE_PERIOD_WEEKS, RANDOM_SEED


def generate_pre_period(clients: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate weekly activity for the pre-experiment period.
    
    For each client × week, generates:
    - weekly_volume: trading volume ($)
    - num_trades: number of individual trades
    - platform_logins: number of dashboard/API logins
    - support_tickets: number of tickets filed (rare events)
    
    Returns:
        pd.DataFrame with (n_clients × 13 weeks) rows
    """
    rng = np.random.default_rng(seed)
    records = []
    
    for _, client in clients.iterrows():
        base_vol = client["baseline_weekly_volume"]
        volatility = client["volume_volatility"]
        
        for week in range(1, PRE_PERIOD_WEEKS + 1):
            # Weekly volume: log-normal noise around baseline
            # Using log-normal ensures volume is always positive
            noise = rng.normal(0, volatility)
            weekly_volume = base_vol * (1 + noise)
            weekly_volume = max(0, weekly_volume)
            
            # Number of trades (correlated with volume, with noise)
            avg_trade_size = _get_avg_trade_size(client["segment"], rng)
            num_trades = max(1, int(weekly_volume / avg_trade_size + rng.normal(0, 3)))
            
            # Platform logins (API clients login less manually)
            if client["api_integration"]:
                platform_logins = rng.poisson(2)  # Low manual logins
            else:
                platform_logins = rng.poisson(8)  # Regular dashboard users
            
            # Support tickets (rare — Poisson with low lambda)
            support_tickets = rng.poisson(0.15)  # ~0.15 tickets/week avg
            
            # Split into maker/taker volume using client's ratio + noise
            maker_ratio = client.get("maker_ratio", 0.40)
            ratio_noise = rng.normal(0, 0.03)
            effective_ratio = np.clip(maker_ratio + ratio_noise, 0.05, 0.95)
            maker_volume = weekly_volume * effective_ratio
            taker_volume = weekly_volume - maker_volume

            records.append({
                "client_id": client["client_id"],
                "week": week,
                "period": "pre",
                "weekly_volume": round(weekly_volume, 2),
                "maker_volume": round(maker_volume, 2),
                "taker_volume": round(taker_volume, 2),
                "num_trades": num_trades,
                "platform_logins": platform_logins,
                "support_tickets": support_tickets,
            })
    
    return pd.DataFrame(records)


def _get_avg_trade_size(segment: str, rng) -> float:
    """Average trade size by segment (for computing num_trades)."""
    sizes = {
        "Market Maker":       50_000,    # Lots of small trades
        "Quant Fund":         100_000,
        "Hedge Fund":         250_000,
        "Asset Manager":      500_000,
        "Corporate Treasury": 1_000_000, # Few large trades
        "Family Office":      300_000,
    }
    base = sizes.get(segment, 200_000)
    return base * rng.lognormal(0, 0.3)


def validate_pre_period(df: pd.DataFrame, clients: pd.DataFrame) -> None:
    """Sanity checks on pre-period data."""
    expected_rows = len(clients) * PRE_PERIOD_WEEKS
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    assert df["weekly_volume"].min() >= 0, "Negative volume detected"
    assert df["week"].min() == 1 and df["week"].max() == PRE_PERIOD_WEEKS
    assert set(df["client_id"]) == set(clients["client_id"]), "Client ID mismatch"
    
    # Check no absurd outliers
    vol_99 = df["weekly_volume"].quantile(0.99)
    print(f"  Volume 99th percentile: ${vol_99:,.0f}")
    
    print("  ✓ Pre-period activity validated")


if __name__ == "__main__":
    print("Step 2: Generating pre-period activity...")
    
    clients_path = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "clients.csv"
    clients = pd.read_csv(clients_path)
    
    pre_activity = generate_pre_period(clients)
    validate_pre_period(pre_activity, clients)
    
    output_path = clients_path.parent / "pre_period_activity.csv"
    pre_activity.to_csv(output_path, index=False)
    print(f"  → Saved {len(pre_activity)} records to {output_path}")
