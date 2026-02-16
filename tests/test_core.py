"""
Unit Tests
Test the tricky math:
boundary conditions,
decomposition identity,
CUPED variance reduction, and
tier assignment.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (
    get_tier,
    get_maker_rate,
    get_taker_rate,
    get_blended_rate,
    get_distance_to_next_tier,
    get_distance_bucket,
    TIER_ORDER,
)
from analysis.revenue_decomposition import decompose_single_client
from analysis.statistical_tests import apply_cuped, compute_ate
from analysis.power_analysis import (
    minimum_detectable_effect,
    power_at_effect_size,
    required_sample_size,
    power_curve,
    experiment_sizing_table,
)

# FEE TIER LOGIC
class TestTierAssignment:
    """Tier boundaries are where bugs hide. Test every edge case."""

    def test_exact_boundary_enterprise(self):
        """Client at EXACTLY $500M should be Enterprise."""
        assert get_tier(500_000_000) == "Enterprise"

    def test_just_below_enterprise(self):
        """$499,999,999 should be Professional, not Enterprise."""
        assert get_tier(499_999_999) == "Professional"

    def test_exact_boundary_professional(self):
        assert get_tier(100_000_000) == "Professional"

    def test_just_below_professional(self):
        assert get_tier(99_999_999) == "Standard"

    def test_exact_boundary_standard(self):
        assert get_tier(25_000_000) == "Standard"

    def test_just_below_standard(self):
        assert get_tier(24_999_999) == "Starter"

    def test_zero_volume(self):
        assert get_tier(0) == "Starter"

    def test_very_large_volume(self):
        assert get_tier(10_000_000_000) == "Enterprise"

    def test_fee_rates_monotonically_decrease(self):
        """Higher tiers should have lower fee rates (both maker and taker)."""
        for rate_fn in [get_maker_rate, get_taker_rate]:
            assert rate_fn("Starter") > rate_fn("Standard")
            assert rate_fn("Standard") > rate_fn("Professional")
            assert rate_fn("Professional") > rate_fn("Enterprise")

    def test_maker_rate_less_than_taker_rate(self):
        """Maker rate should be lower than taker rate for all tiers."""
        for tier in TIER_ORDER:
            assert get_maker_rate(tier) < get_taker_rate(tier), (
                f"{tier}: maker {get_maker_rate(tier)} >= taker {get_taker_rate(tier)}"
            )

    def test_blended_rate_between_maker_and_taker(self):
        """Blended rate should be between maker and taker rates."""
        for tier in TIER_ORDER:
            for ratio in [0.0, 0.25, 0.50, 0.75, 1.0]:
                blended = get_blended_rate(tier, ratio)
                assert get_maker_rate(tier) <= blended <= get_taker_rate(tier), (
                    f"{tier} @ {ratio}: blended {blended} out of range"
                )


class TestDistanceToTier:
    """Distance calculations must be correct for HTE segmentation."""

    def test_client_near_professional(self):
        result = get_distance_to_next_tier(90_000_000, "Standard")
        assert result["next_tier"] == "Professional"
        assert result["next_threshold"] == 100_000_000
        assert abs(result["distance_pct"] - 0.10) < 0.001

    def test_enterprise_has_no_next_tier(self):
        result = get_distance_to_next_tier(600_000_000, "Enterprise")
        assert result["next_tier"] is None
        assert result["distance_pct"] is None

    def test_distance_bucket_within_10(self):
        assert get_distance_bucket(0.08) == "within_10"

    def test_distance_bucket_10_to_20(self):
        assert get_distance_bucket(0.15) == "10_to_20"

    def test_distance_bucket_20_plus(self):
        assert get_distance_bucket(0.35) == "20_plus"

    def test_distance_bucket_enterprise(self):
        assert get_distance_bucket(None) == "at_top_tier"


# REVENUE DECOMPOSITION


class TestRevenueDecomposition:
    """The decomposition is a mathematical identity — it MUST be exact."""

    def test_basic_decomposition_sums(self):
        """Three terms must sum to actual ΔRevenue."""
        result = decompose_single_client(
            pre_volume=90e6,
            post_volume=106e6,
            pre_rate=0.0012,
            post_rate=0.0008,
        )
        actual_delta = (106e6 * 0.0008) - (90e6 * 0.0012)
        assert abs(result["total_delta_revenue"] - actual_delta) < 0.01

    def test_no_tier_cross_no_rate_effect(self):
        """If fee rate doesn't change, rate_effect should be 0."""
        result = decompose_single_client(
            pre_volume=50e6,
            post_volume=60e6,
            pre_rate=0.0012,
            post_rate=0.0012,  # Same rate
        )
        assert result["rate_effect"] == 0
        assert result["interaction_effect"] == 0
        assert result["volume_effect"] == result["total_delta_revenue"]

    def test_no_volume_change(self):
        """If volume doesn't change, volume_effect should be 0."""
        result = decompose_single_client(
            pre_volume=100e6,
            post_volume=100e6,  # Same volume
            pre_rate=0.0012,
            post_rate=0.0008,
        )
        assert result["volume_effect"] == 0
        assert result["interaction_effect"] == 0

    def test_known_example_from_docs(self):
        """The $90M→$106M example from the case study."""
        result = decompose_single_client(
            pre_volume=90e6,
            post_volume=106e6,
            pre_rate=0.0012,
            post_rate=0.0008,
        )
        # Volume effect: +$16M × 0.0012 = +$19,200
        assert abs(result["volume_effect"] - 19_200) < 1
        # Rate effect: $90M × -0.0004 = -$36,000
        assert abs(result["rate_effect"] - (-36_000)) < 1
        # Interaction: +$16M × -0.0004 = -$6,400
        assert abs(result["interaction_effect"] - (-6_400)) < 1

    def test_starter_to_standard_cross(self):
        """Smaller client crossing from Starter to Standard."""
        result = decompose_single_client(
            pre_volume=24e6,
            post_volume=26e6,
            pre_rate=0.0020,
            post_rate=0.0012,
        )
        # This client: pre_rev = $48,000, post_rev = $31,200
        # Coinbase LOSES $16,800 on this client
        assert result["total_delta_revenue"] < 0

    def test_decomposition_with_zero_volume(self):
        """Edge case: client with zero pre-period volume."""
        result = decompose_single_client(
            pre_volume=0,
            post_volume=10e6,
            pre_rate=0.0020,
            post_rate=0.0020,
        )
        assert result["volume_effect"] == 10e6 * 0.0020
        assert result["rate_effect"] == 0


# CUPED


class TestCUPED:

    def test_cuped_reduces_variance(self):
        """CUPED-adjusted values should have lower variance than raw."""
        rng = np.random.default_rng(42)
        n = 500
        pre = rng.normal(100, 30, n)
        post = pre + rng.normal(5, 10, n)  # Correlated with pre

        adjusted = apply_cuped(post, pre)

        assert np.var(adjusted) < np.var(
            post
        ), "CUPED should reduce variance when pre and post are correlated"

    def test_cuped_preserves_mean_difference(self):
        """CUPED should not change the ATE (unbiased)."""
        rng = np.random.default_rng(42)
        n = 500

        pre_t = rng.normal(100, 30, n)
        pre_c = rng.normal(100, 30, n)
        post_t = pre_t + rng.normal(10, 10, n)  # Treatment: +10
        post_c = pre_c + rng.normal(0, 10, n)  # Control: +0

        raw_ate = np.mean(post_t) - np.mean(post_c)

        all_post = np.concatenate([post_t, post_c])
        all_pre = np.concatenate([pre_t, pre_c])
        adjusted = apply_cuped(all_post, all_pre)

        cuped_ate = np.mean(adjusted[:n]) - np.mean(adjusted[n:])

        # ATE should be approximately the same (within sampling noise)
        assert (
            abs(raw_ate - cuped_ate) < 2
        ), f"CUPED changed ATE too much: raw={raw_ate:.2f}, cuped={cuped_ate:.2f}"

    def test_cuped_with_uncorrelated_data(self):
        """If pre and post are uncorrelated, CUPED shouldn't help."""
        rng = np.random.default_rng(42)
        n = 500
        pre = rng.normal(100, 30, n)
        post = rng.normal(105, 30, n)  # Independent of pre

        adjusted = apply_cuped(post, pre)

        # Variance reduction should be minimal
        var_reduction = 1 - np.var(adjusted) / np.var(post)
        assert (
            abs(var_reduction) < 0.1
        ), f"CUPED shouldn't reduce variance much with uncorrelated data (got {var_reduction:.2%})"


# ATE COMPUTATION


class TestATE:

    def test_known_effect(self):
        """ATE should recover a known treatment effect."""
        rng = np.random.default_rng(42)
        treatment = rng.normal(110, 10, 1000)  # Mean 110
        control = rng.normal(100, 10, 1000)  # Mean 100

        result = compute_ate(treatment, control)
        assert abs(result["ate"] - 10) < 2  # Should be close to 10
        assert result["ci_lower"] > 0  # Should be significantly positive

    def test_no_effect(self):
        """ATE should be near zero when there's no treatment effect."""
        rng = np.random.default_rng(42)
        treatment = rng.normal(100, 10, 1000)
        control = rng.normal(100, 10, 1000)

        result = compute_ate(treatment, control)
        assert abs(result["ate"]) < 2  # Should be near zero
        # CI should include zero
        assert result["ci_lower"] < 0 < result["ci_upper"]


# POWER ANALYSIS


class TestPowerAnalysis:
    """Power calculations must be correct — they determine whether
    the experiment launches at all."""

    def test_mde_decreases_with_sample_size(self):
        """More clients → smaller MDE (we can detect smaller effects)."""
        mde_small = minimum_detectable_effect(100, 100, baseline_std=1e6)
        mde_large = minimum_detectable_effect(500, 500, baseline_std=1e6)
        assert mde_large["mde_absolute"] < mde_small["mde_absolute"]

    def test_mde_increases_with_variance(self):
        """Noisier data → larger MDE (harder to detect small effects)."""
        mde_low_var = minimum_detectable_effect(400, 400, baseline_std=1e6)
        mde_high_var = minimum_detectable_effect(400, 400, baseline_std=5e6)
        assert mde_high_var["mde_absolute"] > mde_low_var["mde_absolute"]

    def test_power_increases_with_effect_size(self):
        """Larger effects are easier to detect."""
        power_small = power_at_effect_size(1e5, 400, 400, baseline_std=1e6)
        power_large = power_at_effect_size(5e5, 400, 400, baseline_std=1e6)
        assert power_large > power_small

    def test_power_at_zero_effect_equals_alpha(self):
        """Power at effect=0 should equal the false positive rate (α)."""
        power = power_at_effect_size(0.0, 400, 400, baseline_std=1e6, alpha=0.05)
        assert abs(power - 0.025) < 0.01  # One-sided component of two-sided test

    def test_power_at_mde_equals_target(self):
        """Power evaluated AT the MDE should equal target power."""
        mde = minimum_detectable_effect(400, 400, baseline_std=1e6, power=0.80)
        power = power_at_effect_size(mde["mde_absolute"], 400, 400, baseline_std=1e6)
        assert abs(power - 0.80) < 0.01

    def test_required_sample_size_round_trip(self):
        """If we compute n for a given effect, power at that n should be ~80%."""
        result = required_sample_size(effect_size=1e5, baseline_std=1e6, power=0.80)
        power = power_at_effect_size(
            1e5, result["n_treatment"], result["n_control"], baseline_std=1e6
        )
        assert power >= 0.79  # At least 79% due to ceiling rounding

    def test_power_curve_shape(self):
        """Power curve should be monotonically increasing."""
        curve = power_curve(baseline_std=1e6, n_treatment=400, n_control=400)
        powers = curve["power"].values
        assert all(powers[i] <= powers[i + 1] + 0.001 for i in range(len(powers) - 1))

    def test_sizing_table_structure(self):
        """Sizing table should have correct dimensions and increasing power."""
        table = experiment_sizing_table(baseline_std=1e6, baseline_mean=5e6)
        assert len(table) == 6  # Default 6 sample sizes
        # Power should increase with sample size for any fixed effect
        assert table["power_at_15pct_lift"].is_monotonic_increasing
