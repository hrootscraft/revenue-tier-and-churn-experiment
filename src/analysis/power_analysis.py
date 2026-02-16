"""
Power Analysis for the Fee Tier Experiment:
"How many clients do we need, and how long should we run?"

Three questions it answers:

1. PROSPECTIVE: Before the experiment - given our client base, what's
   the smallest effect we can reliably detect? (MDE; If the true effect
   is smaller than MDE, our experiment will miss it most of the time)
2. SENSITIVITY: Planning Phase - How does power change across plausible
   effect sizes, sample sizes, and variance reduction strategies? (Power curves)
3. POST-HOC: After the experiment - was our design adequate for what
   we actually observed? (Not "did we have enough power to see p<0.05"
   - that's circular - but "how precise is our estimate?")

Key insight for Coinbase: volume data is heavy-tailed (some may trade $500M
while some trade $10M) and autocorrelated (client who traded a lot last week
will likely trade a lot this week; data points aren't independent).

And so naive power calculations using normal theory overstate power.
(
MDE formula uses that giant σ and computes a standard error, then says "you need X power."
But because it assumes the sampling distribution of the mean is Normal (via CLT),
it overestimates how well-behaved your test statistic is. In reality:
a. The t-test's p-values are unreliable when distributions are this skewed
b. A single whale client landing in treatment vs control can flip your result
c. The "80% power" the formula promises might actually be 60% in practice
)

We handle this with simulation-based power analysis using the actual pre-period variance structure.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


# ANALYTICAL POWER FORMULAS
# Given our sample size and variance, what's the smallest effect we can reliably detect? Unit is the same as the outcome variable (e.g., dollars of quarterly volume)
def minimum_detectable_effect(
    n_treatment: int,
    n_control: int,
    baseline_std: float,  # From pre-period data; in our example it is variance of pre_volume
    alpha: float = 0.05,  # 95% confidence that a detected effect is real, not noise ie 5% False Positive Rate (Type I error = α)
    power: float = 0.80,  # 80% chance of detecting a true effect of this size ie 20% false negative rate (True positive rate = 1 - Type II error = 1 - β)
) -> dict:
    """
    Compute the MDE (is the effect size such that if the true effect equals the MDE,
    the test will reject 𝐻0 with probability `power`) for a two-sample t-test.

    MDE = (z_{1-α/2} + z_{power}) × σ × √(1/n_t + 1/n_c)

    This is the smallest true effect size we can detect with the given
    power. Effects smaller than MDE will be missed >20% of the time.

    Args:
        n_treatment: Number of treatment units
        n_control: Number of control units
        baseline_std: Standard deviation of the outcome (from pre-period)
        alpha: Significance level (default 0.05)
        power: Target power (default 0.80)

    Returns:
        dict with mde_absolute, se
    """
    # ppf = percent point function = inverse of the normal CDF. It answers: "What value has X% of the distribution below it?"
    z_alpha = stats.norm.ppf(
        1 - alpha / 2
    )  # z-score for two-tailed test at given alpha (e.g., 1.96 for 0.05) | Critical value for significance
    z_power = stats.norm.ppf(
        power
    )  # z-score corresponding to desired power (e.g., 0.84 for 80% power) | Critical value for power
    se = baseline_std * np.sqrt(
        1 / n_treatment + 1 / n_control
    )  # How precise is our estimate of the treatment effect?
    mde = (z_alpha + z_power) * se  # = 2.8 × SE for 80% power and 5% alpha

    return {
        "mde_absolute": mde,
        "se": se,
        "n_treatment": n_treatment,
        "n_control": n_control,
        "alpha": alpha,  # "I'm okay with 5% chance of saying there's an effect when there isn't"
        "power": power,  # "I want 80% chance of finding the effect if it exists"
    }


# If we have a target effect size in mind, how many clients do we need to reliably detect it?
def required_sample_size(
    effect_size: float,  # Absolute effect size we want to detect (for 10% lift on $100M baseline mean, this would be baseline_mean * 0.10 = $10M)
    baseline_std: float,  # Standard deviation of the outcome (from pre-period)
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,  # Ratio of treatment to control sample sizes (default 1:1)
) -> dict:
    """
    Compute required sample size per arm for a target effect size.

    n_control = ((z_α/2 + z_power) / (effect / σ))² × (1 + 1/ratio)
    n_treatment = n_control × ratio

    Args:
        effect_size: Minimum effect to detect (absolute)
        baseline_std: Standard deviation of the outcome
        alpha: Significance level
        power: Target power
        ratio: n_treatment / n_control ratio (default 1:1)

    Returns:
        dict with n_treatment, n_control, n_total
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # z-score for significance threshold
    z_power = stats.norm.ppf(power)  # z-score for desired power level
    standardized_effect = (
        effect_size / baseline_std
    )  # Effect size in units of standard deviations (Cohen's d)

    n_control = ((z_alpha + z_power) / standardized_effect) ** 2 * (1 + 1 / ratio)
    n_treatment = n_control * ratio

    return {
        "n_treatment": int(np.ceil(n_treatment)),
        "n_control": int(np.ceil(n_control)),
        "n_total": int(np.ceil(n_treatment + n_control)),
        "standardized_effect": standardized_effect,  # How big is the effect relative to the noise? eg. for a 10% lift on a $100M baseline mean with a $50M std, this would be 0.10 * 100M / 50M = 0.2 SDs
    }


# Given a specific effect size, what's the probability we detect it (ie reject H0)?
def power_at_effect_size(
    effect_size: float,  # Absolute effect size we want to evaluate (eg. $10M lift)
    n_treatment: int,
    n_control: int,
    baseline_std: float,
    alpha: float = 0.05,
) -> float:
    """
    Compute statistical power for a given effect size.

    Power = P(reject H0 | H1 true) = Φ(|δ|/SE - z_{α/2})

    Args:
        effect_size: True difference in means (absolute)
        n_treatment: Number of treatment units
        n_control: Number of control units
        baseline_std: Standard deviation of the outcome
        alpha: Significance level

    Returns:
        Power (probability of detecting the effect)
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    se = baseline_std * np.sqrt(1 / n_treatment + 1 / n_control)
    noncentrality = abs(effect_size) / se
    power = 1 - stats.norm.cdf(z_alpha - noncentrality)
    return power


# POWER CURVES
# How does power change as we vary the effect size? The curve we show leadership when planning the experiment.
def power_curve(
    baseline_std: float,
    n_treatment: int,
    n_control: int,
    effect_sizes: Optional[np.ndarray] = None,
    baseline_mean: Optional[float] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Generate a power curve: power as a function of effect size.
    X-axis = "how big is the real effect",
    Y-axis = "probability we detect it."

    Args:
        baseline_std: SD of outcome variable
        n_treatment: Treatment arm size
        n_control: Control arm size
        effect_sizes: Array of absolute effect sizes to evaluate. If None, generates a range from 0 to 3×MDE.
        baseline_mean: If provided, adds a "pct_lift" column
        alpha: Significance level

    Returns:
        DataFrame with columns: effect_size, power, and optionally pct_lift
    """
    if effect_sizes is None:
        mde_info = minimum_detectable_effect(
            n_treatment, n_control, baseline_std, alpha
        )
        max_effect = mde_info["mde_absolute"] * 3
        effect_sizes = np.linspace(0, max_effect, 50)

    rows = []
    for es in effect_sizes:
        pwr = power_at_effect_size(es, n_treatment, n_control, baseline_std, alpha)
        row = {"effect_size": es, "power": pwr}
        if baseline_mean is not None and baseline_mean != 0:
            row["pct_lift"] = es / baseline_mean * 100
        rows.append(row)

    return pd.DataFrame(rows)


def power_curve_with_cuped(
    baseline_std: float,
    n_treatment: int,
    n_control: int,
    pre_post_correlation: float,  # Correlation between pre and post outcomes (ρ), used to calculate CUPED-adjusted variance
    baseline_mean: Optional[float] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compare power curves with and without CUPED variance reduction.

    CUPED reduces variance by a factor of (1 - ρ²), where ρ is the
    correlation between pre and post outcomes.
    cuped_std = baseline_std * sqrt(1 - ρ²) gives the new standard deviation after CUPED adjustment.
    The higher the correlation, the more variance reduction we get, and the smaller the MDE.

    Why CUPED?
    The core problem: Trading volume is noisy. Client A trades $50M one quarter, $80M the next.
    Client B trades $10M, then $5M. This natural fluctuation makes it hard to tell if a 15% lift
    is real or just random variance.
    CUPED's insight: We have pre-experiment data. If a client traded $50M last quarter,
    they'll probably trade roughly $50M this quarter regardless of treatment. That predictable portion
    isn't caused by our experiment — it's just who the client is. So subtract it out.
    Before CUPED: Outcome = Treatment Effect + Client's Natural Level + Random Noise
                               ↑ signal           ↑ huge variance
    After CUPED: Adjusted Outcome = Treatment Effect + (smaller) Residual Noise
                                     ↑ same signal       ↑ variance reduced

    Args:
        baseline_std: SD of raw outcome
        n_treatment: Treatment arm size
        n_control: Control arm size
        pre_post_correlation: Correlation between pre and post (ρ)
        baseline_mean: If provided, adds pct_lift column
        alpha: Significance level

    Returns:
        DataFrame with columns: effect_size, power_raw, power_cuped,
        and the MDE for each
    """
    cuped_std = baseline_std * np.sqrt(1 - pre_post_correlation**2)

    # Generate effect sizes spanning both MDEs
    mde_raw = minimum_detectable_effect(n_treatment, n_control, baseline_std, alpha)
    mde_cuped = minimum_detectable_effect(n_treatment, n_control, cuped_std, alpha)

    # We want to see the curve well beyond the MDE to show how power approaches 1 for larger effects,
    # and to show the difference between raw and CUPED curves.
    # If we only went up to 3×MDE, the curve would be very flat near 1 and it would be hard to see
    # the difference in power for larger effects.
    # By going up to 5×MDE, we can show a more complete picture of how power increases with
    # effect size and how CUPED provides a boost across the range.
    max_effect = mde_raw["mde_absolute"] * 5

    # Generate 50 effect sizes from 0 to 5×MDE to show the full curve, including the
    # difference between raw and CUPED power at larger effects.
    effect_sizes = np.linspace(0, max_effect, 50)

    rows = []
    for es in effect_sizes:
        pwr_raw = power_at_effect_size(es, n_treatment, n_control, baseline_std, alpha)
        pwr_cuped = power_at_effect_size(es, n_treatment, n_control, cuped_std, alpha)
        row = {"effect_size": es, "power_raw": pwr_raw, "power_cuped": pwr_cuped}
        if baseline_mean is not None and baseline_mean != 0:
            row["pct_lift"] = es / baseline_mean * 100
        rows.append(row)

    df = pd.DataFrame(rows)
    # Add MDE info as attributes (metadata) to the DataFrame for easy reference in the report
    df.attrs["mde_raw"] = mde_raw["mde_absolute"]
    df.attrs["mde_cuped"] = mde_cuped["mde_absolute"]
    df.attrs["variance_reduction_pct"] = (1 - (cuped_std / baseline_std) ** 2) * 100
    df.attrs["cuped_std"] = cuped_std
    df.attrs["raw_std"] = baseline_std

    return df


# SIMULATION-BASED POWER: Realistic power for heavy-tailed data
# https://youtu.be/VX_M3tIyiYk : If we use sample size recommended by power analysis,
# we will know that regardless of the p-value, we used enough data to make a good decision.
# Power is the probability of correctly rejecting the null when the alternative is true.
# If power is 80%, then if the true effect is equal to the MDE, we will correctly reject the null 80% of the time.
def simulate_power(
    baseline_volumes: np.ndarray,
    effect_pct: float,
    n_treatment: int,
    n_control: int,
    n_simulations: int = 2_000,  # More simulations = more stable power estimate, but longer runtime. 2K is a good balance for this use case.
    alpha: float = 0.05,
    seed: int = 42,  # Reproducibility is key for simulation-based analysis to ensure results can be consistently replicated and verified by others
    use_cuped: bool = False,  # Whether to apply CUPED variance reduction in the simulated analysis. This allows us to estimate how much power improvement we get from CUPED in a realistic setting, using the actual pre-period volumes to calculate the CUPED adjustment.
    pre_volumes: Optional[
        np.ndarray
    ] = None,  # Pre-period volumes for each client, used to calculate the CUPED adjustment. Required if use_cuped is True
) -> dict:
    """
    Simulation-based power analysis using the ACTUAL volume distribution.
    This is the gold standard for power analysis when data is non-normal and has complex variance structure.

    Why this matters: trading volume is log-normal with heavy tails.
    A few Market Makers trading $500M+ inflate variance massively.
    Analytical formulas assume normality and understate the required
    sample size.

    This function simulates the entire experiment process, including sampling clients, adding noise,
    applying treatment effects, and running statistical tests. By repeating this process many times,
    we can empirically estimate the power of our experiment design under realistic conditions.

    Process:
    1. Sample n_t + n_c clients (with replacement) from baseline data
    2. Apply a known % lift to treatment group's volumes to simulate the true effect (e.g., 15% lift means multiplying their volumes by 1.15)
    3. Run the statistical test to see if we detect the effect (p < alpha)
    4. Repeat 2,000 times
    5. Power = fraction of simulations where p < alpha

    Args:
        baseline_volumes:   Array of real pre-period volumes to sample from (we use this to capture the true variance structure)
        effect_pct:         True treatment effect as a percentage (e.g., 0.15 = 15%); "Effect” is essentially the gap between
                            treatment and control means. By specifying it as a percentage, we can easily translate it into an
                            absolute effect size for each simulation by multiplying it with the baseline mean volume.
                            For example, if the baseline mean is $100M and effect_pct is 0.15, then in each simulation we will
                            apply a $15M lift to the treatment group’s volumes to reflect the assumed true effect we want to be powered to detect.
        n_treatment:        Treatment arm size. In each simulation, we will sample this many clients to be in the treatment group and apply the effect_pct lift to their volumes.
        n_control:          Control arm size. In each simulation, we will sample this many clients to be in the control group and leave their volumes unchanged (except for noise).
        n_simulations:      Number of Monte Carlo iterations (Monte Carlo simulation is a computational technique that uses repeated
                            random sampling to estimate the properties of a statistical function or model. In this context, it allows us to empirically
                            estimate the power of our experiment design by simulating the entire experiment process many times under realistic conditions).
        alpha:              Significance level for the statistical test (default 0.05 for 95% confidence)
        seed:               Random seed.
        use_cuped:          If True, apply CUPED adjustment using pre_volumes
        pre_volumes:        Pre-period volumes for CUPED (required if use_cuped=True); these are used to calculate
                            the CUPED adjustment for each simulated dataset, allowing us to estimate the variance
                            reduction and power improvement from CUPED in a realistic setting.

    Returns:
        dict with power, se_of_power, median_ate, median_se, rejection_rate, n_simulations
    """
    rng = np.random.default_rng(
        seed
    )  # Seed here too to ensure the same random sampling across runs for reproducibility
    n_total = n_treatment + n_control  # Total sample size for each simulation

    # Count how many times we reject H0 across simulations where H0 means "no effect" and in our case H1
    # means there is an effect of size effect_pct in the treatment group compared to control ie the mean
    # of the treatment group is (1 + effect_pct) times the mean of the control group. Basically, the control &
    # treatment data are generated from different distributions,
    # with the treatment distribution having a higher mean by the specified effect size.
    rejections = 0

    # Store the estimated average treatment effect and  standard error from each simulation to analyze the distribution
    # of estimates and their precision across simulations. This allows us to understand not just the power,
    # but also the typical magnitude of the estimated effect and its variability under the simulated conditions.
    ates = []
    ses = []

    for _ in range(n_simulations):
        # Sample clients with replacement
        idx = rng.choice(len(baseline_volumes), size=n_total, replace=True)
        volumes = baseline_volumes[idx].copy()

        # Add log-normal multiplicative noise to mimic heavy-tailed volume variability.
        noise = rng.lognormal(0, 0.25, size=n_total)
        post_volumes = volumes * noise

        # Simulate treatment impact: multiply treatment post-volumes by (1 + effect_pct)
        post_volumes[:n_treatment] *= 1 + effect_pct

        treatment = post_volumes[:n_treatment]
        control = post_volumes[n_treatment:]

        # Optionally apply CUPED with pre-period volumes to reduce variance in each simulation.
        if use_cuped and pre_volumes is not None:
            # Use the same sampled client indices so each post outcome is paired with its own pre value.
            pre_idx = pre_volumes[idx]

            # Combine treatment and control post volumes so CUPED is fit once on the full sample, using pre/post correlation to reduce variance.
            all_post = np.concatenate([treatment, control])

            # CUPED coefficient: theta = Cov(post, pre) / Var(pre).
            theta = np.cov(all_post, pre_idx)[0, 1] / np.var(pre_idx)
            adjusted = all_post - theta * (pre_idx - np.mean(pre_idx))

            # Re-split adjusted outcomes into treatment/control for the hypothesis test.
            treatment = adjusted[:n_treatment]
            control = adjusted[n_treatment:]

        # Welch's t-test because we can't assume equal variances given the heavy-tailed nature of the data
        _, p_value = stats.ttest_ind(treatment, control, equal_var=False)

        if p_value < alpha:
            rejections += 1

        ate = np.mean(treatment) - np.mean(control)
        # se of the difference in means = sqrt(var(treatment)/n_treatment + var(control)/n_control)
        se = np.sqrt(
            np.var(treatment, ddof=1) / n_treatment
            + np.var(control, ddof=1) / n_control
        )
        ates.append(ate)
        ses.append(se)

    power = rejections / n_simulations

    # Uncertainty in power due to limited simulations (SE of a proportion: sqrt(p*(1-p)/n_simulations)).
    se_of_power = np.sqrt(power * (1 - power) / n_simulations)

    return {
        "power": power,
        "se_of_power": se_of_power,
        "effect_pct": effect_pct,
        "n_treatment": n_treatment,
        "n_control": n_control,
        "n_simulations": n_simulations,
        "median_ate": np.median(ates),
        "median_se": np.median(ses),
        "use_cuped": use_cuped,
    }


# SUBGROUP POWER: Can we detect HTE in each bucket?
def subgroup_power_table(
    df: pd.DataFrame,
    subgroup_col: str,
    outcome_col: str = "pre_volume",
    group_col: str = "group",
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> pd.DataFrame:
    """
    For each subgroup, compute MDE and whether we're powered to detect
    the observed effect. This is critical for HTE analysis.

    The mistake junior DSs make: they slice the data into 6 segments,
    run tests on each, and report "significant" or "not significant."
    But they never check whether n=50 in a subgroup can even DETECT
    a 15% lift. If the answer is "no" then a null result means nothing.

    This table shows exactly which subgroup results are trustworthy
    and which are underpowered noise.

    Args:
        df: Experiment summary table
        subgroup_col: Column to segment by (e.g., 'distance_bucket', 'segment')
        outcome_col: Column whose pre-period std to use for power calc
        group_col: Treatment assignment column
        alpha: Significance level
        target_power: Power threshold

    Returns:
        DataFrame with: subgroup, n_treatment, n_control, baseline_std,
        mde_absolute, mde_pct, powered_for_10pct, powered_for_15pct
    """
    rows = []
    for subgroup, sub_df in df.groupby(subgroup_col):
        t_df = sub_df[sub_df[group_col] == "treatment"]
        c_df = sub_df[sub_df[group_col] == "control"]
        n_t = len(t_df)
        n_c = len(c_df)

        if n_t < 5 or n_c < 5:
            continue

        baseline_std = sub_df[outcome_col].std()
        baseline_mean = sub_df[outcome_col].mean()

        mde_info = minimum_detectable_effect(
            n_t, n_c, baseline_std, alpha, target_power
        )
        mde_pct = (
            mde_info["mde_absolute"] / baseline_mean * 100
            if baseline_mean > 0
            else np.nan
        )

        # Can we detect specific lifts?
        power_10 = power_at_effect_size(
            baseline_mean * 0.10, n_t, n_c, baseline_std, alpha
        )
        power_15 = power_at_effect_size(
            baseline_mean * 0.15, n_t, n_c, baseline_std, alpha
        )
        power_25 = power_at_effect_size(
            baseline_mean * 0.25, n_t, n_c, baseline_std, alpha
        )

        rows.append(
            {
                "subgroup": subgroup,
                "n_treatment": n_t,
                "n_control": n_c,
                "n_total": n_t + n_c,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "mde_absolute": mde_info["mde_absolute"],
                "mde_pct": mde_pct,
                "power_at_10pct_lift": round(power_10, 3),
                "power_at_15pct_lift": round(power_15, 3),
                "power_at_25pct_lift": round(power_25, 3),
                "adequately_powered": "Yes" if mde_pct < 20 else "No",
            }
        )

    return pd.DataFrame(rows)


# EXPERIMENT SIZING: The pre-experiment planning table for leadership
def experiment_sizing_table(
    baseline_std: float,
    baseline_mean: float,
    client_pool_sizes: Optional[list] = None,
    effect_sizes_pct: Optional[list] = None,
    alpha: float = 0.05,  #
    power: float = 0.80,
) -> pd.DataFrame:
    """
    Generate the experiment sizing grid so leadership can pick their desired power and effect size.

    Rows = different sample sizes (how many clients to include)
    Cols = different effect sizes (what lift we think is plausible)
    Cells = power

    Leadership picks: "We want 80% power to detect a 10% lift, so we need N=850 clients running for 13 weeks."

    Args:
        baseline_std: SD of quarterly volume
        baseline_mean: Mean quarterly volume
        client_pool_sizes: List of total sample sizes to evaluate
        effect_sizes_pct: List of effect sizes as percentages
        alpha: Significance level
        power: Target power (for MDE column)

    Returns:
        DataFrame grid with power at each (n, effect) combination
    """
    if client_pool_sizes is None:
        client_pool_sizes = [200, 400, 600, 850, 1200, 2000]
    if effect_sizes_pct is None:
        effect_sizes_pct = [0.05, 0.10, 0.15, 0.20, 0.25]

    rows = []
    for n_total in client_pool_sizes:
        n_t = n_total // 2
        n_c = n_total - n_t
        mde_info = minimum_detectable_effect(n_t, n_c, baseline_std, alpha, power)
        mde_pct = mde_info["mde_absolute"] / baseline_mean * 100

        row = {
            "n_total": n_total,
            "n_per_arm": n_t,
            "mde_pct": round(mde_pct, 1),
        }
        for pct in effect_sizes_pct:
            effect_abs = baseline_mean * pct
            pwr = power_at_effect_size(effect_abs, n_t, n_c, baseline_std, alpha)
            row[f"power_at_{int(pct*100)}pct_lift"] = round(pwr, 3)

        rows.append(row)

    return pd.DataFrame(rows)


# DURATION ANALYSIS: How many weeks do we need?
def duration_power_tradeoff(
    weekly_volumes: pd.DataFrame,  # df with client_id and weekly_volume columns
    n_treatment: int,
    n_control: int,
    effect_pct: float = 0.15,  # What lift we want to be powered to detect
    max_weeks: int = 26,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Show how power increases with experiment duration.

    Longer experiments reduce variance because weekly noise averages out.
    But longer experiments also cost more (opportunity cost, holdout cost).

    The output answers: "Is 13 weeks enough, or do we need 26?"

    Args:
        weekly_volumes: DataFrame with client_id and weekly_volume columns
        n_treatment: Treatment arm size
        n_control: Control arm size
        effect_pct: Assumed true effect (as fraction)
        max_weeks: Maximum weeks to evaluate
        alpha: Significance level

    Returns:
        DataFrame with weeks, quarterly_std, mde_absolute, mde_pct, power
    """
    # Compute per-client weekly volume stats
    client_stats = weekly_volumes.groupby("client_id")["weekly_volume"].agg(
        ["mean", "std"]
    )
    grand_mean = client_stats["mean"].mean()

    rows = []
    for weeks in range(1, max_weeks + 1):
        # Quarterly volume std decreases with √weeks due to averaging
        # But between-client variance stays constant
        between_client_std = client_stats["mean"].std()
        within_client_std = client_stats["std"].mean()
        # Total quarterly std: between-client + within-client/√weeks
        quarterly_std = np.sqrt(
            between_client_std**2 + (within_client_std / np.sqrt(weeks)) ** 2
        )
        quarterly_mean = grand_mean * weeks

        mde_info = minimum_detectable_effect(
            n_treatment, n_control, quarterly_std * weeks, alpha
        )
        mde_pct = (
            mde_info["mde_absolute"] / quarterly_mean * 100
            if quarterly_mean > 0
            else np.nan
        )

        effect_abs = quarterly_mean * effect_pct
        pwr = power_at_effect_size(
            effect_abs, n_treatment, n_control, quarterly_std * weeks, alpha
        )

        rows.append(
            {
                "weeks": weeks,
                "quarterly_mean": quarterly_mean,
                "quarterly_std": quarterly_std * weeks,
                "mde_absolute": mde_info["mde_absolute"],
                "mde_pct": round(mde_pct, 1),
                "power_at_target_effect": round(pwr, 3),
            }
        )

    return pd.DataFrame(rows)


# REPORT: One-call summary for the experiment design doc
def power_report(
    df: pd.DataFrame,
    pre_activity: pd.DataFrame,
    outcome_col: str = "pre_volume",
    group_col: str = "group",
    alpha: float = 0.05,
) -> dict:
    """
    Generate a complete power analysis report for the experiment.

    This is the function you call in the notebook. It returns everything
    the experiment design doc needs to answer the key questions for leadership:
    1. Overall MDE (can we detect the main effect?)
    2. Subgroup MDE (can we detect HTE in each distance bucket?)
    3. CUPED impact (how much does variance reduction help?)
    4. Duration analysis (was 13 weeks enough?)
    5. What we CANNOT detect (intellectual honesty)

    Args:
        df: Experiment summary table
        pre_activity: Pre-period weekly activity data
        outcome_col: Column to use for baseline variance
        group_col: Treatment assignment column
        alpha: Significance level

    Returns:
        dict with all analysis components
    """
    t_df = df[df[group_col] == "treatment"]
    c_df = df[df[group_col] == "control"]
    n_t = len(t_df)
    n_c = len(c_df)
    baseline_std = df[outcome_col].std()
    baseline_mean = df[outcome_col].mean()

    # 1. Overall MDE
    mde = minimum_detectable_effect(n_t, n_c, baseline_std, alpha)
    mde["mde_pct"] = mde["mde_absolute"] / baseline_mean * 100

    # 2. CUPED-adjusted MDE
    pre_post_corr = df[["pre_volume", "post_volume"]].corr().iloc[0, 1]
    cuped_std = baseline_std * np.sqrt(1 - pre_post_corr**2)
    mde_cuped = minimum_detectable_effect(n_t, n_c, cuped_std, alpha)
    mde_cuped["mde_pct"] = mde_cuped["mde_absolute"] / baseline_mean * 100

    # 3. Subgroup power (by distance bucket)
    subgroup_distance = subgroup_power_table(
        df, "distance_bucket", outcome_col, group_col, alpha
    )

    # 4. Subgroup power (by segment)
    subgroup_segment = subgroup_power_table(
        df, "segment", outcome_col, group_col, alpha
    )

    # 5. Experiment sizing grid
    sizing = experiment_sizing_table(baseline_std, baseline_mean)

    # 6. Power curve with CUPED comparison
    curves = power_curve_with_cuped(
        baseline_std, n_t, n_c, pre_post_corr, baseline_mean, alpha
    )

    # 7. Duration analysis
    duration = duration_power_tradeoff(pre_activity, n_t, n_c)

    # 8. What we cannot detect (honesty section)
    underpowered = []
    for _, row in subgroup_distance.iterrows():
        if row["mde_pct"] > 20:
            underpowered.append(
                f"  {row['subgroup']}: MDE = {row['mde_pct']:.0f}% "
                f"(n={row['n_total']}). Cannot reliably detect lifts below "
                f"{row['mde_pct']:.0f}%."
            )

    return {
        "overall_mde": mde,
        "cuped_mde": mde_cuped,
        "pre_post_correlation": pre_post_corr,
        "variance_reduction_from_cuped_pct": (1 - (cuped_std / baseline_std) ** 2)
        * 100,
        "subgroup_power_by_distance": subgroup_distance,
        "subgroup_power_by_segment": subgroup_segment,
        "sizing_table": sizing,
        "power_curves": curves,
        "duration_analysis": duration,
        "underpowered_subgroups": underpowered,
        "summary": {
            "n_treatment": n_t,
            "n_control": n_c,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "mde_raw_pct": mde["mde_pct"],
            "mde_cuped_pct": mde_cuped["mde_pct"],
        },
    }
