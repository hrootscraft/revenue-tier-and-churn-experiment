import numpy as np
import pandas as pd
import pytest

from tierlift.durability.roi import decay_multiplier, initial_from_quarter_average, roi_projection
from tierlift.durability.survival import fit_exponential_decay, kaplan_meier, weekly_hazard
from tierlift.economics.decomposition import decompose, decompose_groups


def test_decomposition_identity(rng):
    v0, r0 = rng.uniform(1e6, 1e8, 200), rng.uniform(5e-4, 15e-4, 200)
    v1, r1 = v0 * rng.uniform(0.8, 1.5, 200), r0 * rng.uniform(0.8, 1.0, 200)
    parts = decompose(v0, r0, v1, r1)
    np.testing.assert_allclose(
        parts["volume_effect"] + parts["rate_effect"] + parts["interaction"],
        parts["total"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(parts["total"], v1 * r1 - v0 * r0)


def test_decompose_scalar_and_groups():
    one = decompose(100.0, 0.01, 120.0, 0.009).iloc[0]
    assert one["volume_effect"] == pytest.approx(0.2) and one["rate_effect"] == pytest.approx(-0.1)
    df = pd.DataFrame(
        {
            "cell": ["holdout", "incentive"] * 20,
            "distance_bucket": ["a"] * 40,
            "exp_volume": np.r_[np.full(20, 100.0), np.full(20, 120.0)][
                np.argsort(np.tile([0, 1], 20), kind="stable")
            ],
        }
    )
    df["exp_volume"] = np.where(df["cell"] == "holdout", 100.0, 120.0)
    df["exp_revenue"] = df["exp_volume"] * np.where(df["cell"] == "holdout", 0.01, 0.009)
    g = decompose_groups(df, "incentive")
    assert g.loc["a", "total"] == pytest.approx(120 * 0.009 - 100 * 0.01)
    assert g.loc["a", "net_revenue_pct"] == pytest.approx((120 * 0.009 - 1) / 1)


def test_decay_multiplier_formula():
    assert decay_multiplier(0.0, 52) == 1.0
    assert decay_multiplier(0.02, 52) == pytest.approx((1 - 0.98**52) / (0.02 * 52))
    assert initial_from_quarter_average(10.0, 0.02, 13) * decay_multiplier(
        0.02, 13
    ) == pytest.approx(10.0)


def test_roi_overstatement_about_forty_percent():
    roi = roi_projection(quarter_volume_gain=1000.0, quarter_rate_cost=0.0, weekly_hazard=0.02)
    # dm(13) / dm(52) at h = 2%: 0.8885 / 0.6253 = 1.42
    assert roi.loc["Naive (flat annualisation)", "benefit_overstatement_pct"] == pytest.approx(
        0.42, abs=0.01
    )
    assert roi.loc["Decay-adjusted", "benefit_overstatement_pct"] == 0


def test_kaplan_meier_matches_closed_form(rng):
    n = 20000
    h = 0.03
    t = rng.geometric(h, n).astype(float)
    event = (t <= 30).astype(int)
    t = np.minimum(t, 30)
    km = kaplan_meier(t, event)
    s20 = km.loc[km["week"] == 20, "survival"].iloc[0]
    assert s20 == pytest.approx((1 - h) ** 20, abs=0.01)
    hz = weekly_hazard(t, event)
    assert hz.attrs["pooled_hazard"] == pytest.approx(h, abs=0.003)


def test_exponential_decay_fit_recovers_rate():
    w = np.arange(1, 40)
    eff = 0.06 * 0.98 ** (w - 1)
    curve = pd.DataFrame({"effect_pct": eff, "se_pct": np.full(len(w), 0.005)}, index=w)
    fit = fit_exponential_decay(curve)
    assert fit["weekly_decay"] == pytest.approx(0.02, abs=1e-6)
    assert fit["initial_effect_pct"] == pytest.approx(0.06, abs=1e-6)
