"""Step 3: potential outcomes, planted effects, and what the analyst observes.

The key idea: we simulate **every client under every cell** using the *same*
random noise, then reveal only the outcome of the cell the client was
randomised into. This is the potential-outcomes framework made literal:

    Y_i(holdout), Y_i(nudge), Y_i(incentive)  ->  observed Y_i = Y_i(cell_i)

Because we keep all three, we know each client's true individual treatment
effect (ITE). That lets us grade uplift models against the truth, which is
impossible with real data.

Layers of the weekly volume model (for t = 1..39 after launch)
-------------------------------------------------------------
    V_it(holdout) = latent_i * drift_i * market_t(segment, tier) * noise_it
    V_it(cell)    = V_it(holdout) * (1 + tau_i(cell) * engaged_it)

* ``drift_i``: quarter-over-quarter change in the client's business. Part of
  it is predictable from last quarter's net AUM flows, and part is noise. This
  is what keeps the pre-period from predicting the outcome perfectly, and it
  sets the CUPED variance cut.
* ``market_t``: confounders that hit every cell alike (a volatility spike, a
  competitor promo, a seasonal rally). Randomisation cancels them in the
  treatment-holdout difference, but they would fool a before/after comparison.
* ``tau_i``: the planted lift. It is goal-gradient shaped in distance to the
  next tier and modified by segment, engagement and API share.
* ``engaged_it``: the client's engagement with the campaign lapses at a weekly
  hazard of about 2%. Once it lapses the lift is gone. This is the decay that
  the survival analysis recovers. The analyst observes the lapse through
  product telemetry (``lapse_week``: the last week with a progress-bar view
  or email click; ``lapsed`` = 0 means still engaged when the data ends).

Revenue in the campaign quarter (weeks 1-13)
--------------------------------------------
* holdout: volume * blended rate of the current tier (status quo).
* nudge:   same fee rule; only behaviour changes.
* incentive (Tier Accelerator): the rate is cut halfway to the next tier. If
  quarter volume reaches the next threshold, the full next-tier rate applies
  retroactively to the whole quarter. This is the **rate-compression** cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tierlift.config import Config
from tierlift.fees import FeeSchedule


def planted_volume_lift(clients: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Individual volume lift for the incentive cell (the nudge gets a scaled copy)."""
    t = cfg.truth
    lift_map = t.incentive_lift_by_distance
    d = clients["distance_pct"].to_numpy()
    base = np.select(
        [np.isnan(d), d <= 0.10, d <= 0.20, d <= 0.40],
        [
            lift_map["top_tier"],
            lift_map["within_10"],
            lift_map["10_to_20"],
            lift_map["20_plus_near"],
        ],
        default=lift_map["20_plus_far"],
    )
    resp = (
        clients["segment"]
        .map({s.name: s.responsiveness for s in cfg.population.segments})
        .to_numpy()
    )
    eng = np.exp(t.engagement_lift_coef * clients["engagement_z"].to_numpy())
    api = np.clip(1 + t.api_lift_coef * (clients["api_share"].to_numpy() - 0.4), 0.2, None)
    sigma = t.lift_log_sd
    idio = np.exp(sigma * rng.standard_normal(len(clients)) - sigma**2 / 2)
    return base * resp * eng / eng.mean() * api * idio


def lapse_times(
    clients: pd.DataFrame, cfg: Config, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous time (in weeks) until a treated client disengages, plus their hazard.

    Proportional hazards: lambda_i = lambda_0 * exp(beta' x_i), centred so the
    population-average weekly hazard matches ``base_weekly_hazard``.
    """
    t = cfg.truth
    lin = np.zeros(len(clients))
    for col, beta in t.hazard_coefs.items():
        lin += beta * clients[col].to_numpy(dtype=float)
    lin -= lin.mean()
    lam0 = -np.log1p(-t.base_weekly_hazard)  # weekly hazard 2%  ->  rate 0.0202
    lam = lam0 * np.exp(lin) / np.mean(np.exp(lin))
    return rng.exponential(1 / lam), lam


def market_multipliers(clients: pd.DataFrame, cfg: Config, n_weeks: int) -> np.ndarray:
    """(n_clients, n_weeks) common shocks. Identical across cells by construction."""
    conf = cfg.truth.confounders
    m = np.ones((len(clients), n_weeks))
    weeks = np.arange(1, n_weeks + 1)

    vs = conf["volatility_spike"]
    in_spike = (weeks >= vs["weeks"][0]) & (weeks <= vs["weeks"][1])
    seg_mult = clients["segment"].map(vs["segment_mult"]).fillna(1.0).to_numpy()
    m[:, in_spike] *= seg_mult[:, None]

    cp = conf["competitor_promo"]
    hit = (clients["tier"] == cp["tier"]).to_numpy()
    after = weeks >= cp["start_week"]
    m[np.ix_(hit, after)] *= cp["mult"]

    sr = conf["seasonal_rally"]
    in_rally = (weeks >= sr["weeks"][0]) & (weeks <= sr["weeks"][1])
    m[:, in_rally] *= sr["mult"]

    # Confounders belong to the campaign quarter only; the follow-up is "normal" markets.
    m[:, weeks > cfg.experiment.experiment_weeks] = 1.0
    return m


def simulate_outcomes(
    clients: pd.DataFrame,
    assignment: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Return ``experiment`` (one row per client, observed), ``weekly`` (observed
    panel), and ``truth`` (all potential outcomes, kept out of the analysis)."""
    exp = cfg.experiment
    fees = FeeSchedule.from_config(cfg)
    n = len(clients)
    q_weeks = exp.experiment_weeks
    n_weeks = q_weeks + exp.followup_weeks
    df = clients.merge(assignment, on="client_id", validate="1:1")

    # Holdout potential outcome, shared noise across cells.
    # Part of the drift is predictable from last quarter's net AUM flows (an
    # observed covariate), and part is pure surprise. CUPED only uses pre-period
    # revenue, so it cannot see the predictable part. ML regression adjustment can.
    t = cfg.truth
    log_drift = t.drift_flow_coef * df[
        "aum_flow_z"
    ].to_numpy() + t.drift_log_sd * rng.standard_normal(n)
    drift = np.exp(log_drift - (t.drift_flow_coef**2 + t.drift_log_sd**2) / 2)
    sigma_w = np.sqrt(np.log1p(df["weekly_cv"].to_numpy() ** 2))[:, None]
    noise = np.exp(sigma_w * rng.standard_normal((n, n_weeks)) - sigma_w**2 / 2)
    v_hold = (
        df["latent_weekly_volume"].to_numpy()[:, None]
        * drift[:, None]
        * market_multipliers(df, cfg, n_weeks)
        * noise
    )

    tau_incentive = planted_volume_lift(df, cfg, rng)
    tau_nudge = cfg.truth.nudge_scale * tau_incentive
    lapse, hazard = lapse_times(df, cfg, rng)
    engaged = np.arange(n_weeks)[None, :] < lapse[:, None]  # engaged during week t if t-1 < lapse

    vols = {
        "holdout": v_hold,
        "nudge": v_hold * (1 + tau_nudge[:, None] * engaged),
        "incentive": v_hold * (1 + tau_incentive[:, None] * engaged),
    }

    # Campaign-quarter revenue under each cell.
    tier0 = df["tier_idx"].to_numpy()
    maker = df["maker_ratio"].to_numpy()
    maker_treated = np.clip(maker + exp.treated_maker_ratio_shift, 0, 1)
    top = fees.top
    nxt = np.minimum(tier0 + 1, top)

    q = {c: v[:, :q_weeks].sum(axis=1) for c, v in vols.items()}
    rate = {
        "holdout": fees.blended_rate(tier0, maker),
        "nudge": fees.blended_rate(tier0, maker_treated),
    }
    reached = np.maximum(fees.tier_index(q["incentive"]), tier0)
    crossed = (reached > tier0) & (tier0 < top)
    half = exp.incentive_half_step
    r_cur = fees.blended_rate(tier0, maker_treated)
    r_next = fees.blended_rate(nxt, maker_treated)
    rate["incentive"] = np.where(
        crossed, fees.blended_rate(reached, maker_treated), r_cur - half * (r_cur - r_next)
    )
    rev = {c: q[c] * rate[c] for c in q}

    # Natural crossing (would the client have earned the next tier anyway?)
    crossed_nat = {c: (fees.tier_index(q[c]) > tier0) & (tier0 < top) for c in q}

    truth = pd.DataFrame(
        {
            "client_id": df["client_id"],
            "drift": drift,
            "tau_volume_incentive": tau_incentive,
            "tau_volume_nudge": tau_nudge,
            "lapse_time": lapse,
            "weekly_hazard": 1 - np.exp(-hazard),
            **{f"volume_{c}": q[c] for c in q},
            **{f"rate_{c}": rate[c] for c in rate},
            **{f"revenue_{c}": rev[c] for c in rev},
            **{f"crossed_{c}": crossed_nat[c] for c in crossed_nat},
        }
    )
    for c in cfg.treatment_cells:
        truth[f"ite_revenue_{c}"] = truth[f"revenue_{c}"] - truth["revenue_holdout"]
        truth[f"ite_volume_{c}"] = truth[f"volume_{c}"] - truth["volume_holdout"]

    # What the analyst sees.
    cell = df["cell"].to_numpy()
    pick = lambda d: np.select([cell == c for c in d], [d[c] for c in d])  # noqa: E731
    obs_vol_weekly = np.select([(cell == c)[:, None] for c in vols], [vols[c] for c in vols])
    horizon = n_weeks
    treated = cell != "holdout"
    observed_lapse = np.where(treated, np.minimum(np.floor(lapse) + 1, horizon), np.nan)
    lapse_event = np.where(treated, (np.floor(lapse) + 1) <= horizon, False)

    experiment = df.drop(columns=["latent_weekly_volume"]).assign(
        exp_volume=pick(q),
        exp_revenue=pick(rev),
        exp_rate=pick(rate),
        crossed_tier=np.select([cell == "incentive"], [crossed], default=pick(crossed_nat)).astype(
            int
        ),
        lapse_week=observed_lapse,
        lapsed=lapse_event.astype(int),
    )
    weekly = pd.DataFrame(
        {
            "client_id": np.repeat(df["client_id"].to_numpy(), n_weeks),
            "week": np.tile(np.arange(1, n_weeks + 1), n),
            "volume": obs_vol_weekly.ravel(),
        }
    )
    return {"experiment": experiment, "weekly": weekly, "truth": truth}
