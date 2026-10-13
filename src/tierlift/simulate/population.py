"""Step 1: the client universe and its pre-period (baseline) activity.

What we generate, and why it matters downstream
-----------------------------------------------
* ``latent_weekly_volume``: each client's true trading capacity. It is never
  shown to the analysis, only its noisy footprint in the pre-period.
* 13 weeks of pre-period volume. Their sum (``pre_volume``) decides today's
  tier and the distance to the next tier, and it is the CUPED covariate.
* Covariates an analyst would really have: segment, region, tenure, account
  manager, API share, dashboard engagement, maker ratio, volatility.
  Some of them drive treatment-effect heterogeneity (see ``outcomes.py``), and
  the uplift models have to discover which ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tierlift.config import Config
from tierlift.fees import FeeSchedule, distance_bucket


def generate_population(cfg: Config, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(clients, pre_weekly)``.

    ``clients`` has one row per client; ``pre_weekly`` is long format
    (client_id, week, volume) with weeks -12..0.
    """
    pop = cfg.population
    n = pop.n_clients
    fees = FeeSchedule.from_config(cfg)
    seg_tbl = pd.DataFrame([s.__dict__ for s in pop.segments]).set_index("name")

    segment = rng.choice(seg_tbl.index.to_numpy(), size=n, p=seg_tbl["share"].to_numpy())
    s = seg_tbl.loc[segment]
    region = rng.choice(list(pop.regions), size=n, p=list(pop.regions.values()))

    # Latent quarterly capacity: log-normal around a segment-specific median.
    log_q = np.log(
        pop.median_quarterly_volume * s["volume_mult"].to_numpy()
    ) + pop.log_sd_quarterly_volume * rng.standard_normal(n)
    latent_weekly = np.exp(log_q) / cfg.experiment.pre_weeks

    # Client covariates. Bounded quantities use Beta draws centred on the segment prior.
    maker_ratio = _beta_around(rng, s["maker_ratio"].to_numpy(), concentration=40)
    api_share = _beta_around(rng, s["api_share"].to_numpy(), concentration=12)
    weekly_cv = s["weekly_cv"].to_numpy() * np.exp(0.2 * rng.standard_normal(n))
    tenure_months = np.clip(rng.gamma(shape=2.0, scale=18.0, size=n), 1, 180).round()
    has_am = rng.random(n) < np.clip(0.15 + 0.25 * (log_q - log_q.mean()), 0.05, 0.9)
    # Dashboard logins per month: humans log in, algorithms don't.
    engagement = rng.poisson(np.exp(1.8 - 1.5 * api_share + 0.2 * rng.standard_normal(n)))

    # Pre-period weekly volume: latent capacity times mean-one log-normal noise.
    weeks = cfg.experiment.pre_weeks
    noise = _mean_one_lognormal(rng, weekly_cv[:, None], (n, weeks))
    pre = latent_weekly[:, None] * noise
    pre_volume = pre.sum(axis=1)

    tier_idx = fees.tier_index(pre_volume)
    dist = fees.distance_to_next(pre_volume)
    pre_rate = fees.blended_rate(tier_idx, maker_ratio)

    clients = pd.DataFrame(
        {
            "client_id": np.arange(1, n + 1),
            "segment": segment,
            "region": region,
            "tenure_months": tenure_months.astype(int),
            "has_account_manager": has_am.astype(int),
            "api_share": api_share.round(4),
            "dashboard_logins_pm": engagement,
            "maker_ratio": maker_ratio.round(4),
            "weekly_cv": weekly_cv.round(4),
            "pre_volume": pre_volume,
            "pre_revenue": pre_volume * pre_rate,
            "tier_idx": tier_idx,
            "tier": fees.names[tier_idx],
            "distance_pct": dist,
            "distance_bucket": distance_bucket(dist),
            "latent_weekly_volume": latent_weekly,  # truth: dropped before analysis
        }
    )
    eng = np.log1p(clients["dashboard_logins_pm"])
    clients["engagement_z"] = ((eng - eng.mean()) / eng.std()).round(4)
    # Net AUM flows last quarter (standardised): a CRM signal of where the
    # client's business is heading. It predicts part of next quarter's drift.
    clients["aum_flow_z"] = rng.standard_normal(n).round(4)

    pre_weekly = pd.DataFrame(
        {
            "client_id": np.repeat(clients["client_id"].to_numpy(), weeks),
            "week": np.tile(np.arange(-weeks + 1, 1), n),
            "volume": pre.ravel(),
        }
    )
    return clients, pre_weekly


def _beta_around(rng: np.random.Generator, mean: np.ndarray, concentration: float) -> np.ndarray:
    mean = np.clip(mean, 0.02, 0.98)
    return rng.beta(mean * concentration, (1 - mean) * concentration)


def _mean_one_lognormal(
    rng: np.random.Generator, cv: np.ndarray, size: tuple[int, ...]
) -> np.ndarray:
    """Log-normal noise with mean exactly 1 and the given coefficient of variation."""
    sigma = np.sqrt(np.log1p(np.asarray(cv) ** 2))
    return np.exp(sigma * rng.standard_normal(size) - sigma**2 / 2)
