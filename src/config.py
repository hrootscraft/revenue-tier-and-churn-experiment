"""
Coinbase Institutional Fee Tier Experiment — Configuration
==========================================================
All constants, thresholds, and experiment parameters in one place.
If a number appears in your analysis, it should be defined here.
"""

# ─────────────────────────────────────────────────────────────
# FEE TIER STRUCTURE
# Source: Modeled on Coinbase Institutional's tiered pricing
# bps = basis points (1 bp = 0.01% = 0.0001)
# ─────────────────────────────────────────────────────────────

FEE_TIERS = {
    "Enterprise": {
        "min_volume": 500_000_000,
        "maker_bps": 2,
        "taker_bps": 7,
        "maker_rate": 0.0002,
        "taker_rate": 0.0007,
    },
    "Professional": {
        "min_volume": 100_000_000,
        "maker_bps": 4,
        "taker_bps": 10,
        "maker_rate": 0.0004,
        "taker_rate": 0.0010,
    },
    "Standard": {
        "min_volume": 25_000_000,
        "maker_bps": 6,
        "taker_bps": 16,
        "maker_rate": 0.0006,
        "taker_rate": 0.0016,
    },
    "Starter": {
        "min_volume": 0,
        "maker_bps": 10,
        "taker_bps": 26,
        "maker_rate": 0.0010,
        "taker_rate": 0.0026,
    },
}

# Typical maker ratios by segment (fraction of volume that is maker/limit orders)
# Market Makers post passive liquidity; Corporate Treasuries mostly take liquidity
SEGMENT_MAKER_RATIOS = {
    "Market Maker": 0.80,
    "Quant Fund": 0.65,
    "Hedge Fund": 0.45,
    "Asset Manager": 0.35,
    "Family Office": 0.30,
    "Corporate Treasury": 0.25,
}

# Treatment clients trade more aggressively (taker-heavy) to hit tier thresholds
TREATMENT_MAKER_RATIO_SHIFT = -0.07  # ~7% shift toward taker

# Ordered from highest to lowest for tier assignment logic
TIER_ORDER = ["Enterprise", "Professional", "Standard", "Starter"]

TIER_THRESHOLDS = {
    "Enterprise": 500_000_000,
    "Professional": 100_000_000,
    "Standard": 25_000_000,
    "Starter": 0,
}


def get_tier(quarterly_volume: float) -> str:
    """Assign tier based on quarterly trading volume."""
    for tier in TIER_ORDER:
        if quarterly_volume >= TIER_THRESHOLDS[tier]:
            return tier
    return "Starter"  # fallback


def get_maker_rate(tier: str) -> float:
    """Get maker fee rate (as decimal) for a given tier."""
    return FEE_TIERS[tier]["maker_rate"]


def get_taker_rate(tier: str) -> float:
    """Get taker fee rate (as decimal) for a given tier."""
    return FEE_TIERS[tier]["taker_rate"]


def get_blended_rate(tier: str, maker_ratio: float) -> float:
    """Get blended fee rate: maker_ratio × maker_rate + (1 - maker_ratio) × taker_rate."""
    t = FEE_TIERS[tier]
    return maker_ratio * t["maker_rate"] + (1 - maker_ratio) * t["taker_rate"]


def get_fee_rate(tier: str) -> float:
    """Get taker fee rate (as decimal) for a given tier. Legacy helper."""
    return FEE_TIERS[tier]["taker_rate"]


def get_next_tier(current_tier: str) -> str | None:
    """Get the next tier up. Returns None if already Enterprise."""
    idx = TIER_ORDER.index(current_tier)
    if idx == 0:
        return None  # Already at top
    return TIER_ORDER[idx - 1]


def get_distance_to_next_tier(quarterly_volume: float, current_tier: str) -> dict:
    """
    Compute how far a client is from the next tier boundary.

    Returns:
        dict with 'distance_dollars', 'distance_pct', 'next_tier', 'next_threshold'
        Returns None values if already Enterprise.
    """
    next_tier = get_next_tier(current_tier)
    if next_tier is None:
        return {
            "distance_dollars": None,
            "distance_pct": None,
            "next_tier": None,
            "next_threshold": None,
        }

    next_threshold = TIER_THRESHOLDS[next_tier]
    distance_dollars = next_threshold - quarterly_volume
    distance_pct = distance_dollars / next_threshold

    return {
        "distance_dollars": distance_dollars,
        "distance_pct": distance_pct,
        "next_tier": next_tier,
        "next_threshold": next_threshold,
    }


# ─────────────────────────────────────────────────────────────
# CLIENT UNIVERSE PARAMETERS
# ─────────────────────────────────────────────────────────────

N_CLIENTS = 850

SEGMENTS = {
    # segment_name: (proportion, avg_aum, aum_std, volume_multiplier, volume_volatility)
    # volume_multiplier: what fraction of AUM they trade per week
    # volume_volatility: week-to-week standard deviation of volume (as fraction of mean)
    "Hedge Fund": (0.25, 800e6, 400e6, 0.015, 0.35),
    "Market Maker": (0.10, 500e6, 200e6, 0.040, 0.15),  # High volume, low variance
    "Corporate Treasury": (0.25, 300e6, 150e6, 0.005, 0.20),  # Low volume, predictable
    "Family Office": (0.15, 600e6, 300e6, 0.008, 0.25),
    "Asset Manager": (0.15, 1000e6, 500e6, 0.010, 0.20),
    "Quant Fund": (0.10, 400e6, 200e6, 0.025, 0.30),
}

REGIONS = {
    "Americas": 0.50,
    "EMEA": 0.30,
    "APAC": 0.20,
}


# ─────────────────────────────────────────────────────────────
# EXPERIMENT PARAMETERS
# ─────────────────────────────────────────────────────────────

PRE_PERIOD_WEEKS = 13  # Q1 2024 (Jan–Mar)
EXPERIMENT_WEEKS = 13  # Q3 2024 (Jul–Sep)
FOLLOWUP_WEEKS = 26  # 6 months post-experiment (Oct 2024–Mar 2025)

RANDOM_SEED = 42

# Stratification variables for randomization
STRATIFICATION_VARS = ["current_tier", "distance_bucket", "segment"]


# ─────────────────────────────────────────────────────────────
# TREATMENT EFFECT PARAMETERS
# These are the "true" effects baked into the simulation.
# The analysis must DISCOVER these — they should NOT be trivially
# recoverable. Noise, confounders, and decay make this non-trivial.
# ─────────────────────────────────────────────────────────────

TREATMENT_EFFECTS = {
    # Volume lift by distance-to-tier bucket
    # (mean_lift, std_of_individual_response)
    "within_10": (0.25, 0.12),  # 25% avg lift, but high variance
    "10_to_20": (0.15, 0.08),  # 15% avg lift
    "20_plus": (0.05, 0.06),  # 5% avg lift
    "enterprise": (0.00, 0.03),  # Already top tier — no incentive to move up
}

# Weekly decay rate: effect diminishes ~2% per week
WEEKLY_DECAY_RATE = 0.02

# Some segments respond differently (multiplicative modifier)
SEGMENT_RESPONSE_MODIFIERS = {
    "Hedge Fund": 1.0,
    "Market Maker": 0.3,  # They don't care about a progress bar
    "Corporate Treasury": 1.2,  # Relationship-sensitive, respond well to nudges
    "Family Office": 1.1,
    "Asset Manager": 0.9,
    "Quant Fund": 0.5,  # Automated strategies, less influenced by UI
}


# ─────────────────────────────────────────────────────────────
# CONFOUNDERS (things that affect BOTH groups)
# These exist to make the analysis harder and more realistic
# ─────────────────────────────────────────────────────────────

CONFOUNDERS = {
    # Market volatility spike in weeks 6-8
    # Affects all clients but differently by segment
    "volatility_spike": {
        "weeks": (6, 8),
        "segment_multipliers": {
            "Market Maker": 1.30,  # MMs love volatility
            "Hedge Fund": 1.15,
            "Quant Fund": 1.20,
            "Corporate Treasury": 0.85,  # Treasuries pull back
            "Family Office": 0.90,
            "Asset Manager": 0.95,
        },
    },
    # Competitor promotion starting week 4
    # Slightly reduces Starter-tier volume (they're most price-sensitive)
    "competitor_promo": {
        "start_week": 4,
        "affected_tiers": ["Starter"],
        "volume_multiplier": 0.95,
    },
    # Seasonal crypto rally weeks 10-13
    # Lifts everyone (could confuse naive analysts into thinking treatment
    # effect is growing, when really it's market-wide)
    "seasonal_rally": {
        "weeks": (10, 13),
        "volume_multiplier": 1.10,
    },
}


# ─────────────────────────────────────────────────────────────
# CHURN PARAMETERS (for 6-month follow-up)
# ─────────────────────────────────────────────────────────────

BASE_QUARTERLY_CHURN_RATES = {
    "Hedge Fund": 0.08,
    "Market Maker": 0.05,
    "Corporate Treasury": 0.12,
    "Family Office": 0.11,
    "Asset Manager": 0.09,
    "Quant Fund": 0.07,
}

# Treatment effects on churn
CHURN_REDUCTION_CROSSED_TIER = 0.60  # 60% lower churn if crossed tier (golden handcuff)
CHURN_REDUCTION_NOT_CROSSED = 0.20  # 20% lower churn from engagement effect alone

# Churn definition
CHURN_INACTIVE_DAYS = 28  # 28+ consecutive days with no activity = churned


# ─────────────────────────────────────────────────────────────
# DATA QUALITY THRESHOLDS
# ─────────────────────────────────────────────────────────────

MAX_ALLOWED_SMD = 0.10  # Standardized mean difference for balance checks
REVENUE_TOLERANCE = 0.01  # Max acceptable rounding error in revenue ($)
MIN_TREATMENT_FRACTION = 0.35  # Min fraction of treatment in any stratum
MAX_TREATMENT_FRACTION = 0.65  # Max fraction of treatment in any stratum


# ─────────────────────────────────────────────────────────────
# DISTANCE BUCKET DEFINITIONS
# ─────────────────────────────────────────────────────────────


def get_distance_bucket(distance_pct: float | None) -> str:
    """Categorize clients by how far they are from the next tier."""
    if distance_pct is None:
        return "at_top_tier"  # Enterprise clients
    if distance_pct <= 0.10:
        return "within_10"
    elif distance_pct <= 0.20:
        return "10_to_20"
    else:
        return "20_plus"
