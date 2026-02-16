"""
Revenue Decomposition Analysis:

The core insight of this experiment: Volume ↑ but Fee Rate ↓.
Does the volume gain outweigh the margin compression?

ΔRevenue = ΔVolume × OldRate    +  OldVolume × ΔRate   +   ΔVolume × ΔRate
           ───────────────────    ──────────────────      ──────────────────
              Volume Effect          Rate Effect             Interaction
               (positive)        (negative if crossed)     (usually small)

where
- ΔVolume = PostVolume - PreVolume
- ΔRate = PostFeeRate - PreFeeRate

and the three terms must sum to the actual ΔRevenue (it's a mathematical identity, not an approximation):
1. Volume Effect: If the rate HADN'T changed, the extra volume would've earned this
2. Rate Effect: The rate cut costs this much on existing volume
3. Interaction: The extra volume is ALSO at the cheaper rate

"""

import numpy as np
import pandas as pd


def decompose_single_client(
    pre_volume: float,
    post_volume: float,
    pre_rate: float,
    post_rate: float,
) -> dict:
    """
    Decompose revenue change for a single client into three components.

    Example:
        Client at $90M, 12 bps → crosses to $106M, 8 bps
        Volume effect:      +$16M × 0.0012  = +$19,200   (more trading)
        Rate effect:         $90M × -0.0004 = -$36,000   (cheaper rate on ALL volume)
        Interaction:        +$16M × -0.0004 = -$6,400    (new volume also cheaper)
        Net:                                  -$23,200   (Coinbase loses!)
    """
    delta_volume = post_volume - pre_volume
    delta_rate = post_rate - pre_rate

    volume_effect = delta_volume * pre_rate
    rate_effect = pre_volume * delta_rate
    interaction_effect = delta_volume * delta_rate
    total = volume_effect + rate_effect + interaction_effect

    # Sanity check: must equal actual revenue change
    actual = (post_volume * post_rate) - (pre_volume * pre_rate)
    assert (
        abs(total - actual) < 0.01
    ), f"Decomposition error: Total({total:.2f}) vs Actual({actual:.2f})\nWhere Total = volume_effect + rate_effect + interaction_effect \nAnd Actual = (post_volume * post_rate) - (pre_volume * pre_rate)"

    return {
        "volume_effect": volume_effect,
        "rate_effect": rate_effect,
        "interaction_effect": interaction_effect,
        "total_delta_revenue": total,
        "pre_revenue": pre_volume * pre_rate,
        "post_revenue": post_volume * post_rate,
    }


def decompose_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply revenue decomposition to all clients in the experiment table.

    Adds columns: volume_effect, rate_effect, interaction_effect
    (These may already exist from the pipeline; this recomputes and validates.)
    """
    results = df.apply(
        lambda row: decompose_single_client(
            row["pre_volume"],
            row["post_volume"],
            row["pre_fee_rate"],
            row["post_fee_rate"],
        ),
        axis=1,
    )
    result_df = pd.DataFrame(results.tolist(), index=df.index)

    # Validate: should match pipeline-computed values if they exist
    if "volume_effect" in df.columns:
        max_diff = (result_df["volume_effect"] - df["volume_effect"]).abs().max()
        assert max_diff < 0.01, f"Volume effect mismatch: {max_diff:.2f}"

    return result_df


def decompose_maker_taker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Break revenue into maker vs taker components, showing that
    volume increase was disproportionately taker-side.

    Returns a DataFrame with per-client maker/taker revenue split and changes.
    """
    result = pd.DataFrame(index=df.index)
    result["pre_maker_revenue"] = df["pre_maker_volume"] * df["pre_maker_rate"]
    result["pre_taker_revenue"] = df["pre_taker_volume"] * df["pre_taker_rate"]
    result["post_maker_revenue"] = df["post_maker_volume"] * df["post_maker_rate"]
    result["post_taker_revenue"] = df["post_taker_volume"] * df["post_taker_rate"]
    result["maker_revenue_change"] = (
        result["post_maker_revenue"] - result["pre_maker_revenue"]
    )
    result["taker_revenue_change"] = (
        result["post_taker_revenue"] - result["pre_taker_revenue"]
    )
    result["maker_volume_change"] = df["post_maker_volume"] - df["pre_maker_volume"]
    result["taker_volume_change"] = df["post_taker_volume"] - df["pre_taker_volume"]
    result["taker_share_of_volume_increase"] = np.where(
        (result["maker_volume_change"] + result["taker_volume_change"]) != 0,
        result["taker_volume_change"]
        / (result["maker_volume_change"] + result["taker_volume_change"]),
        np.nan,
    )
    return result


def summarize_by_group(
    df: pd.DataFrame,
    group_col: str,
    treatment_only: bool = True,
) -> pd.DataFrame:
    """
    Summarize revenue decomposition by a grouping variable.

    Args:
        df: Experiment table with decomposition columns
        group_col: Column to group by (e.g., 'distance_bucket', 'segment')
        treatment_only: If True, only summarize treatment group

    Returns:
        DataFrame with per-group: mean volume_effect, rate_effect,
        interaction_effect, net revenue change, and percentage of
        clients who crossed a tier
    """
    if treatment_only:
        df = df[df["group"] == "treatment"]

    summary = (
        df.groupby(group_col)
        .agg(
            n_clients=("client_id", "count"),
            mean_volume_effect=("volume_effect", "mean"),
            mean_rate_effect=("rate_effect", "mean"),
            mean_interaction=("interaction_effect", "mean"),
            mean_net_revenue_change=("revenue_change", "mean"),
            total_net_revenue_change=("revenue_change", "sum"),
            pct_crossed_tier=("crossed_tier", "mean"),
            mean_volume_change_pct=("volume_change_pct", "mean"),
        )
        .reset_index()
    )

    summary["mean_net_revenue_change_pct"] = (
        summary["mean_net_revenue_change"]
        / df.groupby(group_col)["pre_revenue"].mean().values
        * 100
    )

    return summary
