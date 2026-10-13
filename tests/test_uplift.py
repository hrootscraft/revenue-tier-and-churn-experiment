import numpy as np
import pytest

from tierlift.uplift.evaluate import (
    auuc,
    blp_calibration,
    oracle_metrics,
    qini_curve,
    rate,
    toc_curve,
    uplift_curve,
)
from tierlift.uplift.policy import cate_policy, policy_value, rule_policy


def _hetero(rng, n=6000):
    x = rng.uniform(-1, 1, n)
    tau = 2.0 * (x > 0)  # effect only for half the population
    d = rng.random(n) < 0.5
    y = 5 + x + tau * d + rng.standard_normal(n) * 0.5
    return x, tau, d, y


def test_qini_ends_at_ate_and_random_line_is_straight(rng):
    x, tau, d, y = _hetero(rng)
    q = qini_curve(tau, y, d)
    ate = y[d].mean() - y[~d].mean()
    assert q["qini"].iloc[-1] == pytest.approx(ate * d.mean(), rel=0.05)
    assert np.allclose(np.diff(q["random"]), q["random"].iloc[-1] / len(q))


def test_true_ranking_beats_random_and_reversed(rng):
    x, tau, d, y = _hetero(rng)
    good, bad, noise = auuc(tau, y, d), auuc(-tau, y, d), auuc(rng.standard_normal(len(y)), y, d)
    assert good > 0.1 and bad < -0.1 and abs(noise) < 0.05


def test_rate_significant_for_informative_score(rng):
    x, tau, d, y = _hetero(rng, n=3000)
    r = rate(tau + 0.1 * rng.standard_normal(len(tau)), y, d, n_bootstrap=40)
    assert r["autoc"] > 0 and r["p_value"] < 0.01


def test_toc_starts_above_zero_for_good_score(rng):
    x, tau, d, y = _hetero(rng)
    t = toc_curve(tau, y, d)
    assert t["toc"].iloc[0] > 0.5 and abs(t["toc"].iloc[-1]) < 1e-9


def test_blp_calibration_recovers_slope_one(rng):
    x, tau, d, y = _hetero(rng, n=20000)
    b = blp_calibration(tau, y, d)
    assert b["heterogeneity_coef"] == pytest.approx(1.0, abs=0.1)
    assert b["ate_coef"] == pytest.approx(1.0, abs=0.1)


def test_oracle_metrics():
    m = oracle_metrics(np.array([1, 2, 3.0]), np.array([1, 2, 3.0]))
    assert m["pehe"] == 0 and m["spearman_vs_truth"] == 1


def test_policy_value_with_dr_scores(rng):
    gamma = rng.standard_normal(1000) + np.r_[np.ones(500), -np.ones(500)]
    good = policy_value(np.r_[np.ones(500, bool), np.zeros(500, bool)], gamma)
    blanket = policy_value(np.ones(1000, bool), gamma)
    assert good["value_per_client"] > blanket["value_per_client"]
    assert cate_policy(np.array([-1.0, 0.5]), cost_per_client=0.2).tolist() == [False, True]


def test_rule_policy_band():
    import pandas as pd

    df = pd.DataFrame({"distance_pct": [0.05, 0.15, 0.2, 0.25, np.nan]})
    assert rule_policy(df, (0.1, 0.2)).tolist() == [False, True, True, False, False]


def test_uplift_curve_and_qini_agree_in_sign(rng):
    x, tau, d, y = _hetero(rng)
    assert np.sign(uplift_curve(tau, y, d)["gain_above_random"].mean()) == np.sign(
        qini_curve(tau, y, d)["qini_above_random"].mean()
    )


def test_policy_tree_finds_the_responsive_segment(rng):
    """A depth-1 policy tree on informative DR scores must treat exactly the x > 0 half."""
    import pandas as pd

    from tierlift.uplift.policy import fit_policy_tree, policy_tree_oof, policy_tree_text

    x, tau, d, y = _hetero(rng, n=4000)
    X = pd.DataFrame({"x": x, "noise": rng.standard_normal(len(x))})
    gamma = np.where(x > 0, 2.0, -2.0) + 0.5 * rng.standard_normal(len(x))  # noisy scores
    tree = fit_policy_tree(X, gamma, max_depth=1, seed=0)
    treat = tree.predict(X.to_numpy(float)).astype(bool)
    assert (treat == (x > 0)).mean() > 0.95
    assert "x <=" in policy_tree_text(tree, list(X.columns))
    oof = policy_tree_oof(X, gamma, max_depth=1, seed=0)
    assert (oof == (x > 0)).mean() > 0.95
