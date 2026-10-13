"""Annualising a quarter: the ROI overstatement a decay rate corrects.

The naive projection takes the campaign quarter's incremental revenue and
multiplies by four (or, equivalently, projects the *initial* weekly effect
flat across 52 weeks). If the effect decays at weekly hazard h, the real
cumulative effect over H weeks is

    sum_{t=0}^{H-1} (1 - h)^t  =  (1 - (1 - h)^H) / h

instead of H. With h = 2% and H = 52:  (1 - 0.98^52) / 0.02 = 32.6 weeks'
worth, not 52. But the naive analyst does not see the week-1 effect either:
the quarter they annualise already contains 13 weeks of decay, worth
(1 - 0.98^13) / 0.02 = 11.6 weeks of the initial effect. So the naive annual
figure is 4 x 11.6 = 46.2 weeks' worth against a true 32.5: an overstatement
of 46.2 / 32.5 - 1 = 42%, or about 40% at the hazard we actually estimate.

Two subtleties the code handles:

* The quarter itself already contains some decay (weeks 1-13), so the
  "initial" effect must be backed out from the quarter's average, not taken
  as the quarter's average.
* Rate compression is a *cost* that does not decay the same way: a client who
  has crossed a tier keeps the cheaper rate. We keep the volume and rate pieces
  separate and apply decay only to the volume effect in the projection.
"""

from __future__ import annotations

import pandas as pd


def decay_multiplier(weekly_hazard: float, horizon_weeks: int) -> float:
    """Average effect over the horizon relative to the initial effect: (1-(1-h)^H)/(hH)."""
    h = weekly_hazard
    if h <= 0:
        return 1.0
    return float((1 - (1 - h) ** horizon_weeks) / (h * horizon_weeks))


def initial_from_quarter_average(
    quarter_avg: float, weekly_hazard: float, quarter_weeks: int = 13
) -> float:
    """Back out the week-1 effect from a quarter average that already decayed."""
    return quarter_avg / decay_multiplier(weekly_hazard, quarter_weeks)


def roi_projection(
    quarter_volume_gain: float,
    quarter_rate_cost: float,
    weekly_hazard: float,
    horizon_weeks: int = 52,
    quarter_weeks: int = 13,
    fixed_cost: float = 0.0,
) -> pd.DataFrame:
    """Naive vs decay-adjusted projection of a targeted rollout's annual net revenue.

    ``quarter_volume_gain`` and ``quarter_rate_cost`` come from the revenue
    decomposition (volume effect + interaction, and rate effect) for the
    targeted population in the campaign quarter.
    """
    weekly_gain_avg = quarter_volume_gain / quarter_weeks
    weekly_gain_initial = initial_from_quarter_average(
        weekly_gain_avg, weekly_hazard, quarter_weeks
    )
    naive_gain = weekly_gain_avg * horizon_weeks
    adjusted_gain = (
        weekly_gain_initial * decay_multiplier(weekly_hazard, horizon_weeks) * horizon_weeks
    )
    rate_cost_h = quarter_rate_cost / quarter_weeks * horizon_weeks  # the discount does not fade
    rows = {
        "Naive (flat annualisation)": {
            "volume_gain": naive_gain,
            "rate_cost": rate_cost_h,
            "net": naive_gain + rate_cost_h - fixed_cost,
        },
        "Decay-adjusted": {
            "volume_gain": adjusted_gain,
            "rate_cost": rate_cost_h,
            "net": adjusted_gain + rate_cost_h - fixed_cost,
        },
    }
    out = pd.DataFrame(rows).T
    adj = out.loc["Decay-adjusted"]
    # "Overstatement" = how much larger the naive figure is than the decay-adjusted one.
    out["benefit_overstatement_pct"] = out["volume_gain"] / adj["volume_gain"] - 1
    out["net_overstatement_pct"] = out["net"] / adj["net"] - 1
    out.attrs["weekly_hazard"] = weekly_hazard
    out.attrs["decay_multiplier"] = decay_multiplier(weekly_hazard, horizon_weeks)
    return out
