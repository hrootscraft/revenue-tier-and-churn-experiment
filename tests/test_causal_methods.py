"""
Tests for Advanced Analysis Methods

Tests HTE (Empirical Bayes), Causal Forest, durability analysis, and
cannibalization checks — the methods used in notebooks 02 and 03.

Each test uses synthetic data with known properties so we can verify
the methods recover the right answers.

Run with: pytest tests/test_causal_methods.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# SHARED FIXTURES
@pytest.fixture
def mock_experiment_summary():
    """
    Experiment summary with known properties for HTE and Causal Forest tests.
    """
    rng = np.random.default_rng(42)
    n = 400
    groups = ["treatment"] * 200 + ["control"] * 200

    # Generate covariates
    aum = rng.lognormal(19, 1, n)
    tenure = rng.uniform(3, 60, n)
    n_products = rng.integers(1, 6, n)
    baseline_vol = aum * rng.uniform(0.005, 0.02, n)

    # Treatment increases crossing probability
    crossed = []
    for i, g in enumerate(groups):
        if g == "treatment":
            p = 0.4 + 0.1 * (aum[i] > np.median(aum))
        else:
            p = 0.15
        crossed.append(rng.random() < p)

    # Outcomes affected by treatment and crossing
    pre_volume = baseline_vol * 13
    volume_change = []
    revenue_change = []
    churned = []

    for i in range(n):
        base_change = rng.normal(0, 1e6)
        treat_effect = 2e6 if groups[i] == "treatment" else 0
        cross_effect = -1e6 if crossed[i] else 0  # Margin compression

        volume_change.append(base_change + treat_effect)
        revenue_change.append(base_change + treat_effect + cross_effect)

        # Churn: treatment reduces, crossing reduces more
        churn_prob = 0.20
        if groups[i] == "treatment":
            churn_prob *= 0.7
        if crossed[i]:
            churn_prob *= 0.5
        churned.append(rng.random() < churn_prob)

    segments = rng.choice(
        ["Hedge Fund", "Market Maker", "Corporate Treasury",
         "Family Office", "Asset Manager", "Quant Fund"],
        size=n,
    )
    tiers = rng.choice(["1M_15M", "15M_75M", "75M_250M"], size=n)
    regions = rng.choice(["Americas", "EMEA", "APAC"], size=n)

    # Distance to next tier
    next_thresholds = []
    for t in tiers:
        if t == "1M_15M":
            next_thresholds.append(15_000_000)
        elif t == "15M_75M":
            next_thresholds.append(75_000_000)
        else:
            next_thresholds.append(250_000_000)

    next_thresholds = np.array(next_thresholds, dtype=float)
    distance_pct = rng.uniform(0.01, 0.30, n)

    return pd.DataFrame({
        "client_id": [f"CB-{i:05d}" for i in range(n)],
        "group": groups,
        "segment": segments,
        "region": regions,
        "aum": aum,
        "tenure_months": tenure,
        "n_products": n_products,
        "baseline_weekly_volume": baseline_vol,
        "api_integration": rng.choice([True, False], size=n),
        "primary_products": ["Spot|Derivatives"] * n,
        "current_tier": tiers,
        "blended_fee_rate": rng.uniform(0.0005, 0.002, n),
        "maker_ratio": rng.uniform(0.2, 0.8, n),
        "next_tier": ["15M_75M" if t == "1M_15M" else "75M_250M" if t == "15M_75M" else "250M_400M" for t in tiers],
        "next_threshold": next_thresholds,
        "distance_pct": distance_pct,
        "distance_dollars": distance_pct * next_thresholds,
        "distance_bucket": ["within_10" if d < 0.10 else "10_to_20" if d < 0.20 else "20_plus" for d in distance_pct],
        "pre_volume": pre_volume,
        "pre_avg_weekly_volume": baseline_vol,
        "pre_volume_std": rng.uniform(1e5, 5e5, n),
        "pre_logins": rng.integers(20, 150, n),
        "pre_tickets": rng.integers(0, 5, n),
        "pre_trades": rng.integers(100, 1000, n),
        "pre_revenue": pre_volume * rng.uniform(0.0005, 0.002, n),
        "pre_maker_rate": rng.uniform(0.0002, 0.001, n),
        "pre_taker_rate": rng.uniform(0.0007, 0.003, n),
        "post_volume": pre_volume + np.array(volume_change),
        "post_maker_rate": rng.uniform(0.0002, 0.001, n),
        "post_taker_rate": rng.uniform(0.0007, 0.003, n),
        "post_fee_rate": rng.uniform(0.0005, 0.002, n),
        "pre_fee_rate": rng.uniform(0.0005, 0.002, n),
        "crossed_tier": crossed,
        "post_tier": tiers,
        "volume_change": volume_change,
        "volume_change_pct": np.array(volume_change) / pre_volume * 100,
        "revenue_change": revenue_change,
        "revenue_change_pct": np.array(revenue_change) / (pre_volume * 0.001) * 100,
        "churned_6mo": churned,
        "churn_week": [int(rng.integers(1, 27)) if c else np.nan for c in churned],
    })

# HTE TESTS
class TestHTE:

    def test_compute_hte_returns_all_groups(self, mock_experiment_summary):
        """HTE should return one row per group level."""
        from analysis.hte import compute_hte

        result = compute_hte(mock_experiment_summary, "distance_bucket", outcome_col="post_volume")
        groups_in_data = mock_experiment_summary["distance_bucket"].nunique()
        assert len(result) == groups_in_data

    def test_compute_hte_has_confidence_intervals(self, mock_experiment_summary):
        """HTE results should include 95% confidence intervals."""
        from analysis.hte import compute_hte

        result = compute_hte(mock_experiment_summary, "segment", outcome_col="post_volume")
        assert "ci_lower" in result.columns
        assert "ci_upper" in result.columns
        # CI should bracket the point estimate
        assert (result["ci_lower"] <= result["raw_effect"]).all()
        assert (result["ci_upper"] >= result["raw_effect"]).all()

    def test_hte_revenue_combines_volume_and_revenue(self, mock_experiment_summary):
        """compute_hte_revenue should return both volume and revenue columns."""
        from analysis.hte import compute_hte_revenue

        result = compute_hte_revenue(mock_experiment_summary, "distance_bucket")
        assert "volume_lift_pct" in result.columns
        assert "net_revenue_effect" in result.columns
        assert "pct_crossed_tier" in result.columns

    def test_hte_skips_tiny_groups(self):
        """Groups with fewer than 5 per arm should be excluded."""
        from analysis.hte import compute_hte

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "group": ["treatment"] * 3 + ["control"] * 3 + ["treatment"] * 50 + ["control"] * 50,
            "segment": ["Tiny"] * 6 + ["Big"] * 100,
            "post_volume": rng.normal(100, 10, 106),
        })
        result = compute_hte(df, "segment", outcome_col="post_volume")
        # Should skip 'Tiny' (n=3 per arm < 5)
        assert "Tiny" not in result["group"].values
        assert "Big" in result["group"].values

# DURABILITY TESTS
class TestDurability:

    @pytest.fixture
    def mock_weekly_activity(self):
        """Weekly activity data with a known decaying treatment effect."""
        rng = np.random.default_rng(42)
        n_clients = 200
        n_weeks = 13

        rows = []
        for i in range(n_clients):
            group = "treatment" if i < 100 else "control"
            base_vol = rng.normal(10000, 2000)

            for w in range(1, n_weeks + 1):
                noise = rng.normal(0, 500)
                vol = base_vol + noise
                if group == "treatment":
                    # Known decay: starts at +2000, decays 100/week
                    vol += 2000 - 100 * (w - 1)
                rows.append({
                    "client_id": f"CB-{i:05d}",
                    "week": w,
                    "weekly_volume": max(0, vol),
                })

        return pd.DataFrame(rows)

    @pytest.fixture
    def mock_assignment(self):
        return pd.DataFrame({
            "client_id": [f"CB-{i:05d}" for i in range(200)],
            "group": ["treatment"] * 100 + ["control"] * 100,
        })

    def test_weekly_effects_shape(self, mock_weekly_activity, mock_assignment):
        from analysis.durability import compute_weekly_effects

        result = compute_weekly_effects(mock_weekly_activity, mock_assignment)
        assert len(result) == 13, "Should have one row per week"
        assert "effect" in result.columns
        assert "ci_lower" in result.columns
        assert "ci_upper" in result.columns

    def test_weekly_effects_decreasing(self, mock_weekly_activity, mock_assignment):
        """With known decay, weekly effects should generally decrease."""
        from analysis.durability import compute_weekly_effects

        result = compute_weekly_effects(mock_weekly_activity, mock_assignment)
        # First week effect should be larger than last week effect
        assert result.iloc[0]["effect"] > result.iloc[-1]["effect"], (
            "Effect should decrease over time (known decay in mock data)"
        )

    def test_decay_rate_positive(self, mock_weekly_activity, mock_assignment):
        """Decay rate should be positive (effect is shrinking)."""
        from analysis.durability import compute_weekly_effects, estimate_decay_rate

        weekly = compute_weekly_effects(mock_weekly_activity, mock_assignment)
        decay = estimate_decay_rate(weekly)
        assert decay["weekly_decay_pct"] > 0, "Decay should be positive with decaying mock data"

    def test_annualization_scenarios_shape(self, mock_weekly_activity, mock_assignment):
        from analysis.durability import compute_weekly_effects, annualization_scenarios

        weekly = compute_weekly_effects(mock_weekly_activity, mock_assignment)
        scenarios = annualization_scenarios(weekly, total_treatment_revenue_q=100_000)
        assert len(scenarios) == 4, "Should have 4 scenarios"
        assert "scenario" in scenarios.columns
        assert "annual_revenue" in scenarios.columns

    def test_annualization_no_decay_is_4x(self, mock_weekly_activity, mock_assignment):
        """'No decay' scenario should be exactly 4× quarterly revenue."""
        from analysis.durability import compute_weekly_effects, annualization_scenarios

        weekly = compute_weekly_effects(mock_weekly_activity, mock_assignment)
        scenarios = annualization_scenarios(weekly, total_treatment_revenue_q=100_000)
        no_decay = scenarios[scenarios["scenario"] == "No decay (naive)"].iloc[0]
        assert abs(no_decay["annual_revenue"] - 400_000) < 1, "No decay = 4× quarterly"

    def test_alpha_parameter_changes_ci(self, mock_weekly_activity, mock_assignment):
        """Different alpha should produce different CI widths."""
        from analysis.durability import compute_weekly_effects

        result_95 = compute_weekly_effects(mock_weekly_activity, mock_assignment, alpha=0.05)
        result_99 = compute_weekly_effects(mock_weekly_activity, mock_assignment, alpha=0.01)
        # 99% CI should be wider than 95% CI
        width_95 = (result_95["ci_upper"] - result_95["ci_lower"]).mean()
        width_99 = (result_99["ci_upper"] - result_99["ci_lower"]).mean()
        assert width_99 > width_95, "99% CI should be wider than 95% CI"


# CANNIBALIZATION TESTS
class TestCannibalization:

    def test_no_cannibalization_detected(self):
        """With random pre/post data, no cannibalization should be detected."""
        from analysis.cannibalization import check_cannibalization

        rng = np.random.default_rng(42)
        n = 200

        assignment = pd.DataFrame({
            "client_id": [f"CB-{i:05d}" for i in range(n)],
            "group": ["treatment"] * 100 + ["control"] * 100,
        })

        # Generate pre and post with same distribution (no cannibalization)
        pre_rows = []
        post_rows = []
        for i in range(n):
            base = rng.normal(10000, 2000)
            for w in range(1, 14):
                pre_rows.append({
                    "client_id": f"CB-{i:05d}",
                    "week": w,
                    "weekly_volume": max(0, base + rng.normal(0, 500)),
                })
                post_rows.append({
                    "client_id": f"CB-{i:05d}",
                    "week": w,
                    "weekly_volume": max(0, base + rng.normal(0, 500)),
                })

        pre = pd.DataFrame(pre_rows)
        post = pd.DataFrame(post_rows)

        result = check_cannibalization(pre, post, assignment)
        assert "evidence_of_cannibalization" in result
        assert result["evidence_of_cannibalization"] == False

    def test_cannibalization_detected_with_control_decline(self):
        """If control volume drops 20%, cannibalization should be flagged."""
        from analysis.cannibalization import check_cannibalization

        rng = np.random.default_rng(42)
        n = 200

        assignment = pd.DataFrame({
            "client_id": [f"CB-{i:05d}" for i in range(n)],
            "group": ["treatment"] * 100 + ["control"] * 100,
        })

        pre_rows = []
        post_rows = []
        for i in range(n):
            group = "treatment" if i < 100 else "control"
            base = rng.normal(10000, 500)
            for w in range(1, 14):
                pre_rows.append({
                    "client_id": f"CB-{i:05d}",
                    "week": w,
                    "weekly_volume": base + rng.normal(0, 200),
                })
                if group == "control":
                    # Control drops 20%
                    post_vol = base * 0.80 + rng.normal(0, 200)
                else:
                    post_vol = base + rng.normal(0, 200)
                post_rows.append({
                    "client_id": f"CB-{i:05d}",
                    "week": w,
                    "weekly_volume": max(0, post_vol),
                })

        pre = pd.DataFrame(pre_rows)
        post = pd.DataFrame(post_rows)

        result = check_cannibalization(pre, post, assignment)
        assert result["evidence_of_cannibalization"] == True
        assert result["control_change_pct"] < -5

    def test_returns_expected_keys(self):
        """check_cannibalization should return all expected keys."""
        from analysis.cannibalization import check_cannibalization

        rng = np.random.default_rng(42)
        n = 50
        assignment = pd.DataFrame({
            "client_id": [f"CB-{i:05d}" for i in range(n)],
            "group": ["treatment"] * 25 + ["control"] * 25,
        })
        rows = []
        for i in range(n):
            for w in range(1, 14):
                rows.append({
                    "client_id": f"CB-{i:05d}",
                    "week": w,
                    "weekly_volume": rng.normal(1000, 100),
                })
        activity = pd.DataFrame(rows)

        result = check_cannibalization(activity, activity, assignment)
        expected_keys = {
            "control_pre_mean", "control_post_mean", "control_change_pct",
            "paired_t_stat", "p_value", "evidence_of_cannibalization", "interpretation",
        }
        assert expected_keys.issubset(set(result.keys()))
