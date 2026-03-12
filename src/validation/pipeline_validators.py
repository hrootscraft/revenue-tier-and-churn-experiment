"""
Per-step output validators for the data pipeline.

Functions:
    validate_clients     -01. client universe
    validate_pre_period  -02. pre-period activity
    validate_assignment  -03. treatment assignment
    validate_post_period -04. post-period activity
    validate_followup    -04. followup activity + churn summary
"""

import numpy as np
import pandas as pd
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    N_CLIENTS, PRE_PERIOD_WEEKS, EXPERIMENT_WEEKS, FOLLOWUP_WEEKS,
    SEGMENTS, REGIONS, TIER_ORDER,
    BALANCE_CHECK_VARS, MAX_ALLOWED_SMD,
    MIN_TREATMENT_FRACTION, MAX_TREATMENT_FRACTION,
    get_tier, get_blended_rate, get_distance_bucket,
)


def validate_clients(df: pd.DataFrame) -> None:
    """
    Validation of the generated client universe.

    Organized into five layers, from cheapest to most expensive:
    A. Column completeness      - are all expected columns present and non-null?
    B. Value ranges             - do individual values respect generation bounds?
    C. Categorical validity     - are categorical values drawn from the right sets?
    D. Distribution sanity      - do proportions roughly match config targets?
    E. Cross-column consistency - do derived columns agree with their inputs?
    """

    assert len(df) >= N_CLIENTS * 0.95, f"Expected ~{N_CLIENTS} clients, got {len(df)}"
    assert df["client_id"].is_unique, "Duplicate client IDs!"
    assert df["aum"].min() >= 10e6, "AUM below $10M floor"
    assert df["baseline_weekly_volume"].min() > 0, "Negative or zero volume"

    # A. Column completeness
    # WHY: A dropped column (e.g., from a bad pd.concat axis) silently becomes
    # NaN downstream. Catching it here prevents cryptic KeyErrors in later
    # pipeline steps and analysis notebooks.
    expected_cols = {
        "client_id",
        "segment",
        "region",
        "aum",
        "baseline_weekly_volume",
        "volume_volatility",
        "tenure_months",
        "api_integration",
        "primary_products",
        "n_products",
        "maker_ratio",
        "est_quarterly_volume",
        "current_tier",
        "maker_rate",
        "taker_rate",
        "blended_fee_rate",
        "fee_rate",
        "distance_dollars",
        "distance_pct",
        "next_tier",
        "next_threshold",
        "distance_bucket",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    # WHY: A NULL in fee_rate means revenue = volume × NULL = NULL.
    # Aggregations silently drop NULLs, giving you the wrong totals.
    critical_cols = [
        "client_id",
        "segment",
        "region",
        "aum",
        "baseline_weekly_volume",
        "maker_ratio",
        "current_tier",
        "blended_fee_rate",
        "distance_bucket",
    ]
    for col in critical_cols:
        n_null = df[col].isna().sum()
        assert (
            n_null == 0
        ), f"Column '{col}' has {n_null} NULLs — check generation logic"

    # B. Value range checks
    # WHY: Each of these corresponds to a np.clip or bounded draw in
    # generate_clients. If the clip breaks or the distribution parameters
    # are wrong, impossible values leak through and corrupt tier assignment.

    # AUM capped at $5B (np.clip ceiling)
    assert (
        df["aum"].max() <= 5e9
    ), f"AUM above $5B cap: ${df['aum'].max():,.0f} — np.clip may be broken"

    # Maker ratio clipped to [0.05, 0.95] to avoid unrealistic 100% maker or taker clients
    assert (
        df["maker_ratio"].min() >= 0.05
    ), f"maker_ratio below 0.05 floor: {df['maker_ratio'].min()}"
    assert (
        df["maker_ratio"].max() <= 0.95
    ), f"maker_ratio above 0.95 cap: {df['maker_ratio'].max()}"

    # Tenure drawn from uniform(3, 60) then int() → [3, 59]
    assert (
        df["tenure_months"].min() >= 3
    ), f"tenure_months below 3: {df['tenure_months'].min()}"
    assert (
        df["tenure_months"].max() <= 59
    ), f"tenure_months above 59: {df['tenure_months'].max()}"

    # Product count drawn from choice([1,2,3,4,5])
    assert df["n_products"].min() >= 1, f"n_products below 1: {df['n_products'].min()}"
    assert df["n_products"].max() <= 5, f"n_products above 5: {df['n_products'].max()}"

    # Blended fee rate must be positive — every tier has maker_bps >= 0 and
    # taker_bps >= 5, and maker_ratio < 1, so the blended rate is always > 0.
    assert (
        df["blended_fee_rate"] > 0
    ).all(), "Found zero or negative blended_fee_rate — fee lookup is broken"

    # Volume volatility comes directly from SEGMENTS config, always positive
    assert (
        df["volume_volatility"] > 0
    ).all(), "Found zero or negative volume_volatility — SEGMENTS config may be wrong"

    # C. Categorical validity
    # WHY: A typo in segment names (e.g., "HedgeFund" vs "Hedge Fund") or a
    # stale config silently creates a new category that won't match any
    # SEGMENT_MAKER_RATIOS key or SEGMENT_RESPONSE_MODIFIERS key downstream.
    invalid_segments = set(df["segment"].unique()) - set(SEGMENTS.keys())
    assert not invalid_segments, f"Unknown segments: {invalid_segments}"

    invalid_regions = set(df["region"].unique()) - set(REGIONS.keys())
    assert not invalid_regions, f"Unknown regions: {invalid_regions}"

    invalid_tiers = set(df["current_tier"].unique()) - set(TIER_ORDER)
    assert not invalid_tiers, f"Unknown tiers: {invalid_tiers}"

    valid_buckets = {"within_10", "10_to_20", "20_plus", "at_top_tier"}
    invalid_buckets = set(df["distance_bucket"].unique()) - valid_buckets
    assert not invalid_buckets, f"Unknown distance_buckets: {invalid_buckets}"

    # D. Distribution sanity
    # WHY: A loop bug (e.g., wrong iterator variable) could silently generate
    # 50% Hedge Funds and 0% Market Makers. The pipeline would run fine but
    # all segment-level analysis would be meaningless.
    segment_actual = df["segment"].value_counts(normalize=True)
    for seg, (target_prop, *_) in SEGMENTS.items():
        actual_prop = segment_actual.get(seg, 0)
        assert abs(actual_prop - target_prop) < 0.05, (
            f"Segment '{seg}' proportion {actual_prop:.1%} is >5pp from "
            f"target {target_prop:.1%}"
        )

    region_actual = df["region"].value_counts(normalize=True)
    for reg, target_prop in REGIONS.items():
        actual_prop = region_actual.get(reg, 0)
        assert abs(actual_prop - target_prop) < 0.05, (
            f"Region '{reg}' proportion {actual_prop:.1%} is >5pp from "
            f"target {target_prop:.1%}"
        )

    # Every tier should have at least some clients — an empty tier means the
    # volume generation is miscalibrated and that tier's HTE is unestimable.
    for tier in TIER_ORDER:
        assert (
            df["current_tier"] == tier
        ).any(), f"Tier '{tier}' has zero clients — volume generation is miscalibrated"

    # api_integration should be boolean with both values present
    assert (
        df["api_integration"].dtype == bool
    ), f"api_integration is {df['api_integration'].dtype}, expected bool"
    assert (
        df["api_integration"].nunique() == 2
    ), "api_integration has only one value — probability thresholds may be wrong"

    # E. Cross-column consistency
    # WHY: These are the most critical checks. Derived columns (tier, fee rate,
    # distance bucket) are computed from base columns. If a stale merge, a
    # column overwrite, or a reordering breaks the derivation, these catch it.
    # Revenue in every downstream step depends on these being correct.

    # At this point (step 01 output), est_quarterly_volume = baseline_weekly_volume × 13.
    # This is a temporary estimate — step 02 will overwrite it with the actual sum
    # of observed pre-period weekly volumes once activity is generated.
    # This assertion validates step 01's internal consistency only.
    expected_qvol = df["baseline_weekly_volume"] * PRE_PERIOD_WEEKS
    max_qvol_diff = (df["est_quarterly_volume"] - expected_qvol).abs().max()
    assert max_qvol_diff < 0.01, (
        f"est_quarterly_volume != baseline_weekly_volume * {PRE_PERIOD_WEEKS} "
        f"(max diff: ${max_qvol_diff:.4f})"
    )

    # Tier must match volume — the single most dangerous inconsistency.
    # A wrong tier means wrong fee rates, wrong revenue, wrong treatment effects.
    recomputed_tier = df["est_quarterly_volume"].apply(get_tier)
    tier_mismatches = (df["current_tier"] != recomputed_tier).sum()
    assert tier_mismatches == 0, (
        f"{tier_mismatches} clients have current_tier inconsistent with "
        f"est_quarterly_volume — tier assignment logic may have changed"
    )

    # Blended fee rate must be recomputable from tier + maker_ratio
    recomputed_rate = df.apply(
        lambda row: get_blended_rate(row["current_tier"], row["maker_ratio"]), axis=1
    )
    max_rate_diff = (df["blended_fee_rate"] - recomputed_rate).abs().max()
    assert max_rate_diff < 1e-9, (
        f"blended_fee_rate doesn't match get_blended_rate(tier, maker_ratio) "
        f"(max diff: {max_rate_diff:.2e})"
    )

    # Distance bucket must be consistent with distance_pct
    recomputed_bucket = df["distance_pct"].apply(get_distance_bucket)
    bucket_mismatches = (df["distance_bucket"] != recomputed_bucket).sum()
    assert bucket_mismatches == 0, (
        f"{bucket_mismatches} clients have distance_bucket inconsistent with "
        f"distance_pct — get_distance_bucket thresholds may have changed"
    )

    # Summary stats for sanity check and to confirm that the distribution looks reasonable.
    tier_counts = df["current_tier"].value_counts(normalize=True)
    print(f"Tier distribution: {tier_counts.to_dict()}")

    near_boundary = df[df["distance_bucket"].isin(["within_10", "10_to_20"])].shape[0]
    near_pct = near_boundary / len(df)
    print(f"Clients within 20% of next tier: {near_boundary} ({near_pct:.0%})")
    assert (
        near_pct > 0.20
    ), f"Only {near_pct:.0%} near boundaries — need more for interesting HTE"

    print("Client universe validated")


def validate_pre_period(df: pd.DataFrame, clients: pd.DataFrame) -> None:
    """
    Validate pre-period activity data against the client universe.

    Checks are organized as:
    A. Column completeness      - expected schema, no nulls
    B. Structural integrity     - exact row count, no duplicates, balanced panel
    C. Value ranges             - non-negative counts, volume bounds
    D. Cross-column consistency - maker + taker = total, mean volume ~ baseline
    """

    # A. Column completeness
    # WHY: A missing column (e.g., from a renamed field) causes a KeyError
    # downstream in 05_build_experiment_table when aggregating maker/taker splits.
    expected_cols = {
        "client_id",
        "week",
        "period",
        "weekly_volume",
        "maker_volume",
        "taker_volume",
        "num_trades",
        "platform_logins",
        "support_tickets",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    # WHY: A NULL in weekly_volume silently drops that row from sum/mean
    # aggregations, understating a client's pre-period baseline.
    for col in expected_cols:
        n_null = df[col].isna().sum()
        assert n_null == 0, f"Column '{col}' has {n_null} NULLs"

    # B. Structural integrity
    expected_rows = len(clients) * PRE_PERIOD_WEEKS
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    assert set(df["client_id"]) == set(clients["client_id"]), "Client ID mismatch"

    # WHY: A bad append could give one client 14 weeks and another 12.
    # Total row count would still be correct, but per-client aggregates
    # (used for CUPED and stratification) would be wrong.
    weeks_per_client = df.groupby("client_id")["week"].nunique()
    bad_clients = weeks_per_client[weeks_per_client != PRE_PERIOD_WEEKS]
    assert len(bad_clients) == 0, (
        f"{len(bad_clients)} clients don't have exactly {PRE_PERIOD_WEEKS} weeks. "
        f"Examples: {bad_clients.head().to_dict()}"
    )

    # WHY: Duplicate (client_id, week) pairs double-count volume in aggregation.
    dupes = df.duplicated(subset=["client_id", "week"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate (client_id, week) pairs"

    assert df["week"].min() == 1 and df["week"].max() == PRE_PERIOD_WEEKS, (
        f"Week range [{df['week'].min()}, {df['week'].max()}] "
        f"doesn't match expected [1, {PRE_PERIOD_WEEKS}]"
    )

    # WHY: This is pre-experiment data — if any other period label leaks in,
    # downstream joins that filter on period will silently include/exclude
    # wrong rows.
    assert (
        df["period"] == "pre"
    ).all(), f"Found non-'pre' period values: {df['period'].unique()}"

    # C. Value ranges
    # WHY: Negative volume is impossible (max(0, ...) guard in generation).
    # Negative counts would indicate a bug in the Poisson or trade-size logic.
    assert df["weekly_volume"].min() >= 0, "Negative weekly_volume detected"
    assert df["maker_volume"].min() >= 0, "Negative maker_volume detected"
    assert df["taker_volume"].min() >= 0, "Negative taker_volume detected"
    assert (
        df["num_trades"].min() >= 1
    ), f"num_trades below 1: {df['num_trades'].min()} — floor logic may be broken"
    assert df["platform_logins"].min() >= 0, "Negative platform_logins detected"
    assert df["support_tickets"].min() >= 0, "Negative support_tickets detected"

    # D. Cross-column consistency
    # WHY: maker_volume + taker_volume = weekly_volume is an arithmetic identity.
    # A violation means the split logic is broken,
    # which would corrupt the maker/taker revenue decomposition downstream.
    # Tolerance is 0.02 rather than 0.01 because maker_volume, taker_volume,
    # and weekly_volume are each independently rounded to 2 decimal places,
    # so round(maker) + round(taker) can differ from round(total) by up to $0.01.
    vol_diff = (df["maker_volume"] + df["taker_volume"] - df["weekly_volume"]).abs()
    max_vol_diff = vol_diff.max()
    assert (
        max_vol_diff < 0.02
    ), f"maker_volume + taker_volume != weekly_volume (max diff: ${max_vol_diff:.4f})"

    # WHY: The noise model is E[weekly_volume] = baseline_weekly_volume.
    # If the per-client mean deviates wildly (>50%) from baseline, either
    # the noise distribution is miscalibrated or the wrong baseline was used.
    # We check the correlation rather than exact equality because 13 weeks
    # of noisy draws won't match perfectly.
    client_mean_vol = df.groupby("client_id")["weekly_volume"].mean()
    baseline_vol = clients.set_index("client_id")["baseline_weekly_volume"]
    # Align indexes
    common = client_mean_vol.index.intersection(baseline_vol.index)
    corr = client_mean_vol.loc[common].corr(baseline_vol.loc[common])
    assert corr > 0.90, (
        f"Correlation between mean weekly volume and baseline is {corr:.3f} — "
        f"expected >0.90. Noise model may be miscalibrated."
    )

    # Summary
    vol_99 = df["weekly_volume"].quantile(0.99)
    print(f"Volume 99th percentile: ${vol_99:,.0f}")
    print("Pre-period activity validated")


def validate_assignment(assignment: pd.DataFrame, clients: pd.DataFrame) -> None:
    """
    Validate treatment assignment for correctness and balance.

    Checks are organized as:
    A. Structural integrity - every client assigned, no duplicates, valid groups
    B. Sample Ratio Mismatch - formal chi-square test on the overall split
    C. Stratified balance - within-stratum splits (guaranteed by design)
    D. Non-stratified covariate balance - SMD for continuous, proportion diff for categorical
    E. Joint balance - omnibus check across all covariates simultaneously
    """

    # A. STRUCTURAL INTEGRITY
    # A1. Every client assigned exactly once
    # WHY: If a client is missing, their data still shows up in downstream
    # analysis tables but with no group label → NaN joins → silently dropped
    # from the ATE calculation, biasing the estimate if missingness correlates
    # with outcomes (e.g., 400M+ clients all missing → ATE computed only on
    # lower tiers).
    assert set(assignment["client_id"]) == set(
        clients["client_id"]
    ), "Assignment doesn't cover all clients!"

    # A2. No duplicate assignments
    # WHY: A client in BOTH treatment and control contaminates both groups.
    # Their outcome counts toward both the treatment mean and control mean,
    # biasing the difference toward zero (attenuating the treatment effect).
    # This can happen from a bad merge or append in the randomization logic.
    assert assignment["client_id"].is_unique, "Duplicate assignments!"

    # A3. Only valid group labels exist
    # WHY: A typo ("Treatment" vs "treatment") or encoding issue creates a
    # third implicit group. Downstream filters like df[df.group == "treatment"]
    # silently exclude those clients, and you'd never notice unless you check.
    valid_groups = {"treatment", "control"}
    actual_groups = set(assignment["group"].unique())
    assert actual_groups == valid_groups, (
        f"Unexpected group labels: {actual_groups - valid_groups}. "
        f"Expected exactly {valid_groups}."
    )

    # A4. Every stratum contains both treatment AND control
    # WHY: If a stratum is all-treatment (or all-control), within-stratum
    # comparisons are undefined, you can't compute a treatment effect for that
    # segment. Stratified estimators (like the stratum-weighted ATE) would
    # either error or silently drop that stratum, losing those clients' data.
    # This can happen with very small strata (n=1 or n=2).
    for strata, group_df in assignment.groupby("strata"):
        groups_in_strata = set(group_df["group"].unique())
        assert groups_in_strata == valid_groups, (
            f"Stratum '{strata}' (n={len(group_df)}) is missing group(s): "
            f"{valid_groups - groups_in_strata}. "
            f"Cannot compute within-stratum treatment effect."
        )

    print("A. Structural integrity: ✓")

    # B. SAMPLE RATIO MISMATCH (SRM) TEST
    # WHY: The most important diagnostic in online experimentation. SRM means
    # the observed treatment/control ratio differs from the EXPECTED ratio
    # (50/50) more than chance alone would explain. This is a formal chi-square
    # test, not just eyeballing whether the split "looks close to 50%."
    #
    # SRM is the #1 signal that something is MECHANICALLY broken:
    # - Randomization bug (hash function collision, off-by-one)
    # - Logging bug (treatment events logged differently than control)
    # - Bot/crawler contamination concentrated in one group
    # - Survivorship bias (treatment causes more app crashes → fewer logged users)
    #
    # α = 0.01 rather than 0.05 to reduce false positives —
    # a spurious SRM flag would halt a valid experiment unnecessarily.
    # (At production scale with n=100K+, α=0.05 can fire on trivial noise;
    # at n=5000 the practical difference is small but the principle holds.)
    n_treatment = (assignment["group"] == "treatment").sum()
    n_control = (assignment["group"] == "control").sum()
    n_total = len(assignment)
    expected = n_total / 2

    chi2_stat = (n_treatment - expected) ** 2 / expected + (
        n_control - expected
    ) ** 2 / expected
    srm_pvalue = 1 - stats.chi2.cdf(
        chi2_stat, df=1
    )  # chi-square with 1 degree of freedom

    t_frac = n_treatment / n_total
    print(f"\nB. Sample Ratio Mismatch test:")
    print(f"Split: {n_treatment} T / {n_control} C " f"({t_frac:.2%} / {1-t_frac:.2%})")
    print(f"χ² = {chi2_stat:.4f}, p = {srm_pvalue:.4f}")

    if srm_pvalue < 0.01:
        print(
            f"SRM DETECTED (p={srm_pvalue:.4f} < 0.01). ✗"
            f"DO NOT TRUST RESULTS. Debug randomization."
        )
    else:
        print(f"No SRM (p={srm_pvalue:.4f} ≥ 0.01). ✓")

    assert srm_pvalue >= 0.01, (
        f"Sample Ratio Mismatch detected: {n_treatment}T / {n_control}C, "
        f"p={srm_pvalue:.6f}. Randomization is broken."
    )

    # C. WITHIN-STRATUM BALANCE (guaranteed by stratification)
    # WHY: Stratification GUARANTEES balance on tier × distance_bucket by
    # construction, we shuffle within each stratum and assign the first half
    # to treatment. This check verifies the guarantee held (catches code bugs).
    #
    # We use the MIN/MAX_TREATMENT_FRACTION thresholds from config (0.35/0.65)
    # which are tighter than a blanket 30/70 check. With our smallest stratum
    # at ~100 clients, we should see 45-55% treatment, never 35% or 65%.
    print(f"\nC. Within-stratum balance (guaranteed by design):")
    for strata, group_df in assignment.groupby("strata"):
        n_strata = len(group_df)
        strata_t_frac = (group_df["group"] == "treatment").mean()
        print(f"{strata}: n={n_strata}, {strata_t_frac:.1%}T / {1-strata_t_frac:.1%}C")

        assert MIN_TREATMENT_FRACTION <= strata_t_frac <= MAX_TREATMENT_FRACTION, (
            f"Stratum '{strata}' treatment fraction {strata_t_frac:.1%} "
            f"outside [{MIN_TREATMENT_FRACTION:.0%}, {MAX_TREATMENT_FRACTION:.0%}] "
            f"(n={n_strata}). Stratification logic may be broken."
        )

    # D. NON-STRATIFIED COVARIATE BALANCE
    # WHY: Stratification guarantees balance on tier and distance_bucket, but
    # other covariates (segment, AUM, tenure, maker_ratio) could still be
    # imbalanced by chance. With n=2500 per group, large imbalances are
    # unlikely but possible and if they occur on a variable correlated with
    # the outcome (eg, AUM strongly predicts volume), they bias the ATE.
    #
    # We use Standardized Mean Difference (SMD) rather than p-values because:
    # i. p-values are sample-size dependent: with n=2500, even a trivially
    # small difference can be "significant". SMD measures MAGNITUDE of
    # imbalance in standard deviation units, independent of sample size.
    # ii. Rule of thumb: SMD < 0.05 is excellent, < 0.10 is acceptable,
    # > 0.10 warrants investigation.
    #
    # For categorical variables, we use the max absolute proportion difference
    # across all categories. This catches scenarios like "65% of treatment is
    # Hedge Funds but only 45% of control" which a single SMD can't express.
    merged = clients.merge(assignment, on="client_id")
    continuous_vars = [
        v
        for v in BALANCE_CHECK_VARS
        if v in merged.columns
        and merged[v].dtype in ["float64", "int64", "float32", "int32"]
    ]
    categorical_vars = [
        v
        for v in BALANCE_CHECK_VARS
        if v in merged.columns and merged[v].dtype == "object"
    ]

    print(f"\nD. Non-stratified covariate balance (checked, not guaranteed):")
    smd_values = []
    print("i. Continuous variables (SMD):")
    # is the difference small relative to natural variation?
    for col in continuous_vars:
        t_vals = merged.loc[merged["group"] == "treatment", col]
        c_vals = merged.loc[merged["group"] == "control", col]
        pooled_std = merged[col].std()
        smd = abs(t_vals.mean() - c_vals.mean()) / pooled_std if pooled_std > 0 else 0
        smd_values.append(smd)
        status = "✓" if smd < MAX_ALLOWED_SMD else "⚠ FLAG"
        print(
            f"       {col}: T={t_vals.mean():,.2f}, C={c_vals.mean():,.2f}, "
            f"SMD={smd:.4f} {status}"
        )

    print("ii. Categorical variables (max abs proportion diff):")
    # is the compositional difference small in absolute terms?
    # eg. "treatment is 65% Hedge Funds but control is 45% Hedge Funds → 20% diff = FLAG"
    for col in categorical_vars:
        t_dist = merged.loc[merged["group"] == "treatment", col].value_counts(
            normalize=True
        )
        c_dist = merged.loc[merged["group"] == "control", col].value_counts(
            normalize=True
        )
        all_vals = set(t_dist.index) | set(
            c_dist.index
        )  # all categories across both groups
        max_diff = max(abs(t_dist.get(v, 0) - c_dist.get(v, 0)) for v in all_vals)
        status = "✓" if max_diff < 0.05 else "FLAG"
        print(f"{col}: max proportion diff = {max_diff:.4f} {status}")

    # E. JOINT BALANCE TEST (omnibus)
    # WHY: Individual SMDs can all pass while there's a SYSTEMATIC pattern.
    # Example: Treatment has slightly higher AUM (+0.04 SMD), slightly higher
    # tenure (+0.03 SMD), slightly higher maker_ratio (+0.03 SMD). Each one
    # is fine individually, but ALL tilting the same direction suggests a
    # non-random pattern,  maybe the randomizer has a subtle correlation with
    # client ordering.
    #
    # The joint test asks: "Looking at ALL covariates together, is the overall
    # pattern of imbalance consistent with random chance?"
    #
    # We use a simple approach: count how many covariates exceed SMD > 0.05.
    # Under random assignment, each has ~5% chance of exceeding 0.05 by chance.
    # With k covariates, we expect ~k × 0.05 flags. Significantly more flags
    # than expected suggests a systematic problem.
    n_covariates = len(smd_values)
    n_flagged = sum(1 for s in smd_values if s > 0.05)
    expected_flags = n_covariates * 0.05

    print(f"\nE. Joint balance (omnibus):")
    print(
        f"{n_flagged}/{n_covariates} continuous covariates have SMD > 0.05 "
        f"(expected ~{expected_flags:.1f} by chance)"
    )
    if n_flagged > max(2, n_covariates * 0.3):
        print("WARNING: systematic imbalance detected")
        assert False, (
            f"{n_flagged}/{n_covariates} covariates flagged — "
            f"expected ~{expected_flags:.1f}. Investigate randomizer."
        )
    print("No evidence of systematic imbalance ✓")
    print("\nAll assignment checks passed ✓")  # only prints if E passed


def validate_post_period(
    df: pd.DataFrame, clients: pd.DataFrame, assignment: pd.DataFrame
) -> None:
    """
    Validate post-period (experiment) activity data.

    Mirrors validate_pre_period structure, plus one post-period-specific check:
    A. Column completeness      - expected schema, no nulls
    B. Structural integrity     - exact row count, balanced panel, period label
    C. Value ranges             - non-negative volumes and counts
    D. Cross-column consistency - maker + taker = total, volume correlation with baseline
    E. Treatment effect direction - treatment mean volume > control mean volume
    """

    # A. Column completeness
    expected_cols = {
        "client_id", "week", "period", "weekly_volume",
        "maker_volume", "taker_volume", "num_trades",
        "platform_logins", "support_tickets",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    for col in expected_cols:
        n_null = df[col].isna().sum()
        assert n_null == 0, f"Column '{col}' has {n_null} NULLs"

    # B. Structural integrity
    expected_rows = len(clients) * EXPERIMENT_WEEKS
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    assert set(df["client_id"]) == set(clients["client_id"]), "Client ID mismatch"

    weeks_per_client = df.groupby("client_id")["week"].nunique()
    bad_clients = weeks_per_client[weeks_per_client != EXPERIMENT_WEEKS]
    assert len(bad_clients) == 0, (
        f"{len(bad_clients)} clients don't have exactly {EXPERIMENT_WEEKS} weeks. "
        f"Examples: {bad_clients.head().to_dict()}"
    )

    dupes = df.duplicated(subset=["client_id", "week"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate (client_id, week) pairs"

    assert df["week"].min() == 1 and df["week"].max() == EXPERIMENT_WEEKS, (
        f"Week range [{df['week'].min()}, {df['week'].max()}] "
        f"doesn't match expected [1, {EXPERIMENT_WEEKS}]"
    )

    # WHY: A period label bug could cause downstream joins that filter on
    # period="experiment" to silently exclude or double-count rows.
    assert (
        df["period"] == "experiment"
    ).all(), f"Found non-'experiment' period values: {df['period'].unique()}"

    # C. Value ranges
    assert df["weekly_volume"].min() >= 0, "Negative weekly_volume detected"
    assert df["maker_volume"].min() >= 0, "Negative maker_volume detected"
    assert df["taker_volume"].min() >= 0, "Negative taker_volume detected"
    assert (
        df["num_trades"].min() >= 1
    ), f"num_trades below 1: {df['num_trades'].min()} — floor logic may be broken"
    assert df["platform_logins"].min() >= 0, "Negative platform_logins detected"
    assert df["support_tickets"].min() >= 0, "Negative support_tickets detected"

    # D. Cross-column consistency
    vol_diff = (df["maker_volume"] + df["taker_volume"] - df["weekly_volume"]).abs()
    max_vol_diff = vol_diff.max()
    assert (
        max_vol_diff < 0.02
    ), f"maker_volume + taker_volume != weekly_volume (max diff: ${max_vol_diff:.4f})"

    # WHY: Treatment + confounders cause real divergence from baseline, so the
    # correlation threshold is 0.80 (not 0.90 like pre-period). Still catches
    # a completely broken baseline lookup (e.g., wrong client_id join).
    client_mean_vol = df.groupby("client_id")["weekly_volume"].mean()
    baseline_vol = clients.set_index("client_id")["baseline_weekly_volume"]
    common = client_mean_vol.index.intersection(baseline_vol.index)
    corr = client_mean_vol.loc[common].corr(baseline_vol.loc[common])
    assert corr > 0.80, (
        f"Correlation between mean post-period volume and baseline is {corr:.3f} — "
        f"expected >0.80. Baseline lookup or noise model may be broken."
    )

    # E. Treatment effect direction
    # WHY: The single most important simulation invariant. If treatment group
    # mean volume <= control, the effect was sign-flipped, applied to the wrong
    # group, or the treatment effect parameters are all zero/negative. This
    # check would have caught every treatment logic bug in this file.
    post_with_group = df.merge(assignment[["client_id", "group"]], on="client_id")
    t_mean = post_with_group.loc[
        post_with_group["group"] == "treatment", "weekly_volume"
    ].mean()
    c_mean = post_with_group.loc[
        post_with_group["group"] == "control", "weekly_volume"
    ].mean()
    assert t_mean > c_mean, (
        f"Treatment mean volume (${t_mean:,.0f}) is not greater than control "
        f"(${c_mean:,.0f}) — treatment effect may be sign-flipped or missing."
    )

    print(f"  ✓ Post-period activity validated "
          f"(T mean: ${t_mean:,.0f} vs C mean: ${c_mean:,.0f}, "
          f"lift: {(t_mean/c_mean - 1):.1%})")


def validate_followup(
    followup_activity: pd.DataFrame,
    churn_summary: pd.DataFrame,
    clients: pd.DataFrame,
) -> None:
    """
    Validate the 6-month follow-up outputs from simulate_followup.

    Checks are organized as:
    A. Activity structure   - row count, period label, maker+taker=total
    B. Churn summary        - one row per client, valid labels, churn_week bounds
    C. Churn-activity consistency - churned clients have zero volume after churn_week
    D. Churn direction      - treatment churn rate < control churn rate
    """
    n_clients = len(clients)

    # A. Activity structure
    expected_rows = n_clients * FOLLOWUP_WEEKS
    assert len(followup_activity) == expected_rows, (
        f"Expected {expected_rows} followup rows, got {len(followup_activity)}"
    )

    weeks_per_client = followup_activity.groupby("client_id")["week"].nunique()
    bad = weeks_per_client[weeks_per_client != FOLLOWUP_WEEKS]
    assert len(bad) == 0, (
        f"{len(bad)} clients don't have exactly {FOLLOWUP_WEEKS} followup weeks."
    )

    assert (
        followup_activity["period"] == "followup"
    ).all(), f"Found non-'followup' period values: {followup_activity['period'].unique()}"

    vol_diff = (
        followup_activity["maker_volume"]
        + followup_activity["taker_volume"]
        - followup_activity["weekly_volume"]
    ).abs()
    assert vol_diff.max() < 0.02, (
        f"maker + taker != weekly_volume in followup (max diff: ${vol_diff.max():.4f})"
    )

    # B. Churn summary structure
    assert len(churn_summary) == n_clients, (
        f"Expected {n_clients} churn rows, got {len(churn_summary)}"
    )
    assert churn_summary["client_id"].is_unique, "Duplicate client_ids in churn_summary"

    valid_groups = {"treatment", "control"}
    actual_groups = set(churn_summary["group"].unique())
    assert actual_groups == valid_groups, (
        f"Unexpected group labels in churn_summary: {actual_groups - valid_groups}"
    )

    # WHY: churn_week must be set iff churned_6mo is True. A mismatch means the
    # churn flag and the week assignment fell out of sync — churned clients would
    # show non-zero volume, or non-churned clients would have a stale week value.
    churned = churn_summary[churn_summary["churned_6mo"] == True]
    not_churned = churn_summary[churn_summary["churned_6mo"] == False]
    assert churned["churn_week"].notna().all(), "Churned clients missing churn_week"
    assert not_churned["churn_week"].isna().all(), "Non-churned clients have a churn_week"

    if len(churned) > 0:
        assert churned["churn_week"].between(1, FOLLOWUP_WEEKS).all(), (
            f"churn_week out of [1, {FOLLOWUP_WEEKS}] range"
        )

    # C. Churn-activity consistency
    # WHY: The most dangerous followup bug: churn flag says churned at week W,
    # but weekly_volume is non-zero for weeks >= W. This would corrupt durability
    # analysis (falsely showing persistent volume for churned clients) and
    # overstate follow-up revenue.
    if len(churned) > 0:
        churn_map = churned.set_index("client_id")["churn_week"]
        churned_activity = followup_activity[
            followup_activity["client_id"].isin(churn_map.index)
        ].copy()
        churned_activity["churn_week"] = churned_activity["client_id"].map(churn_map)
        post_churn = churned_activity[
            churned_activity["week"] >= churned_activity["churn_week"]
        ]
        nonzero_post_churn = (post_churn["weekly_volume"] > 0).sum()
        assert nonzero_post_churn == 0, (
            f"{nonzero_post_churn} rows have non-zero volume after churn_week — "
            f"churn enforcement in simulate_followup is broken."
        )

    # D. Churn direction
    # WHY: Treatment reduces churn (via engagement + golden handcuff effects).
    # If treatment churn rate >= control, the churn_reduction parameters were
    # sign-flipped or the group labels are swapped.
    t_churn = churn_summary.loc[churn_summary["group"] == "treatment", "churned_6mo"].mean()
    c_churn = churn_summary.loc[churn_summary["group"] == "control", "churned_6mo"].mean()
    assert t_churn < c_churn, (
        f"Treatment churn ({t_churn:.1%}) is not less than control churn ({c_churn:.1%}) — "
        f"churn reduction parameters may be sign-flipped."
    )

    print(f"  ✓ Follow-up validated "
          f"(T churn: {t_churn:.1%} vs C churn: {c_churn:.1%})")
