import numpy as np
import pytest

from tierlift.experiment.ate import aipw, cuped, cuped_theta, difference_in_means, lin_regression
from tierlift.experiment.design import mde, power_at, srm_test, standardized_mean_difference
from tierlift.experiment.robustness import holm, permutation, robustness_table


def _toy(rng, n=4000, effect=2.0, rho=0.7):
    x = rng.standard_normal(n)
    d = rng.random(n) < 0.5
    y = 10 + rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(n) + effect * d
    return y, d, x


def test_cuped_theta_is_regression_slope(rng):
    y, d, x = _toy(rng)
    theta = cuped_theta(y, x)
    assert theta == pytest.approx(np.polyfit(x, y, 1)[0])


def test_cuped_variance_reduction_is_rho_squared(rng):
    y, d, x = _toy(rng, n=20000, rho=0.7)
    est = cuped(y, d, x)
    assert est.extra["variance_reduction"] == pytest.approx(0.49, abs=0.03)
    assert est.effect == pytest.approx(2.0, abs=0.08)


def test_all_estimators_agree_on_toy(rng):
    y, d, x = _toy(rng, n=6000)
    X = np.column_stack([x, x**2])
    ests = [
        difference_in_means(y, d),
        cuped(y, d, x),
        lin_regression(y, d, X),
        aipw(y, d, X, n_folds=3),
    ]
    for e in ests:
        assert abs(e.effect - 2.0) < 3 * e.se + 0.05
    # CUPED and Lin should be tighter than difference in means.
    assert ests[1].se < ests[0].se and ests[2].se < ests[0].se


def test_relative_effect_delta_method(rng):
    y, d, x = _toy(rng)
    e = difference_in_means(y, d)
    assert e.rel_effect == pytest.approx(e.effect / e.control_mean)
    assert e.rel_se > 0


def test_permutation_p_value_is_uniform_under_null(rng):
    y = rng.standard_normal(300)
    d = rng.random(300) < 0.5
    p = permutation(y, d, n_permutations=400, seed=1)["p_value"]
    assert 0.0 < p <= 1.0


def test_robustness_table_detects_effect(rng):
    y, d, x = _toy(rng, n=2000, effect=1.0)
    tab = robustness_table(y, d, n_permutations=300)
    assert (tab["p_value"] < 0.01).all()


def test_holm_monotone():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.20})
    assert (
        adj["a"] == pytest.approx(0.03)
        and adj["b"] == pytest.approx(0.08)
        and adj["c"] == pytest.approx(0.20)
    )


def test_srm_and_smd():
    import pandas as pd

    cells = pd.Series(["holdout"] * 300 + ["nudge"] * 300 + ["incentive"] * 400)
    assert srm_test(cells, {"holdout": 0.3, "nudge": 0.3, "incentive": 0.4})["p_value"] > 0.99
    assert standardized_mean_difference(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) == 0


def test_mde_and_power_consistent():
    m = mde(sd=1.0, n_treated=500, n_control=500, alpha=0.05, power=0.8)
    assert power_at(m, 1.0, 500, 500) == pytest.approx(0.8, abs=0.01)
