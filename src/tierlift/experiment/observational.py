"""What if we had NOT randomised? An observational counterfactual.

Course section 3 (videos 5-8) teaches why randomisation matters and how
propensity scores and AIPW rescue estimates when it is missing. We show it on
our own data. We reuse each client's two potential outcomes, Y(holdout) and
Y(incentive), but let clients *opt in* to the incentive the way they would if
it were simply launched with a sign-up button:

    P(opt in | x) = sigmoid(a + b1 * engagement_z + b2 * aum_flow_z + b3 * near_tier)

Engaged, growing clients near a tier boundary opt in more, and they would have
grown anyway (aum_flow_z drives drift). So the naive "opted-in vs not"
comparison mixes the treatment effect with selection. Because we know the true
ATE, we can watch each estimator succeed or fail:

* naive difference in means: biased by selection
* regression adjustment: fixes it only if the outcome model is right
* IPW with an estimated propensity: fixes it only if the propensity model is right
* AIPW: consistent if *either* model is right (double robustness)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from tierlift.experiment.ate import Estimate, _from_influence, difference_in_means, lin_regression


def simulate_opt_in(
    experiment: pd.DataFrame, truth: pd.DataFrame, seed: int = 7, strength: float = 1.0
) -> pd.DataFrame:
    """Observational dataset: self-selected incentive uptake using true potential outcomes."""
    rng = np.random.default_rng(seed)
    df = experiment.merge(
        truth[["client_id", "revenue_holdout", "revenue_incentive"]], on="client_id"
    )
    near = df["distance_pct"].fillna(1).le(0.2).astype(float)
    logit = -0.9 + strength * (0.8 * df["engagement_z"] + 0.9 * df["aum_flow_z"] + 0.7 * near)
    p = 1 / (1 + np.exp(-logit))
    d = rng.random(len(df)) < p
    df = df.assign(
        opted_in=d.astype(int),
        true_propensity=p,
        y=np.where(d, df["revenue_incentive"], df["revenue_holdout"]),
    )
    return df


def true_ate(obs: pd.DataFrame) -> float:
    return float((obs["revenue_incentive"] - obs["revenue_holdout"]).mean())


def _crossfit_nuisance(X, y, d, seed=0, n_folds=5, linear_propensity=False, scale=None):
    """Cross-fitted propensity e(x) and outcome models mu0(x), mu1(x).

    ``scale`` (e.g. each client's pre-period revenue) makes the outcome models
    predict y/scale and multiplies back. Revenue is heavy-tailed; a tree model
    fitted on raw dollars extrapolates badly for the few whales, and the AIPW
    correction term then divides those large residuals by small propensities.
    Modelling the *index* removes most of that scale before the model sees it.
    """
    n = len(y)
    scale = np.ones(n) if scale is None else np.asarray(scale, float)
    e, mu0, mu1 = np.zeros(n), np.zeros(n), np.zeros(n)
    reg = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=seed
    )
    clf = (
        LogisticRegression(max_iter=2000)
        if linear_propensity
        else HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=7, random_state=seed
        )
    )
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        e[te] = clone(clf).fit(X[tr], d[tr]).predict_proba(X[te])[:, 1]
        ys = y / scale
        mu1[te] = clone(reg).fit(X[tr][d[tr]], ys[tr][d[tr]]).predict(X[te]) * scale[te]
        mu0[te] = clone(reg).fit(X[tr][~d[tr]], ys[tr][~d[tr]]).predict(X[te]) * scale[te]
    return np.clip(e, 0.02, 0.98), mu0, mu1


def observational_estimates(obs: pd.DataFrame, X: np.ndarray, seed: int = 0) -> pd.DataFrame:
    y = obs["y"].to_numpy(float)
    d = obs["opted_in"].to_numpy(bool)
    X = np.asarray(X, float)
    e, mu0, mu1 = _crossfit_nuisance(
        X, y, d, seed, linear_propensity=True, scale=obs["pre_revenue"].to_numpy(float)
    )
    n1, n0 = int(d.sum()), int((~d).sum())

    ests: list[Estimate] = [difference_in_means(y, d)]
    ests[-1].estimator = "Naive difference in means"
    lin = lin_regression(y, d, X)
    lin.estimator = "Regression adjustment (linear)"
    ests.append(lin)
    # Hajek-normalised IPW
    w1, w0 = d / e, (~d) / (1 - e)
    psi1 = w1 * y / w1.mean()
    psi0 = w0 * y / w0.mean()
    ests.append(_from_influence("IPW (logistic propensity)", psi1, psi0, n1, n0))
    psi1 = mu1 + d * (y - mu1) / e
    psi0 = mu0 + (~d) * (y - mu0) / (1 - e)
    ests.append(_from_influence("AIPW (cross-fitted)", psi1, psi0, n1, n0))

    out = pd.DataFrame([x.to_dict() for x in ests]).set_index("estimator")
    out["true_ate"] = true_ate(obs)
    out["bias"] = out["effect"] - out["true_ate"]
    out["covers_truth"] = (out["ci_low"] <= out["true_ate"]) & (out["true_ate"] <= out["ci_high"])
    return out


def observational_replication(
    experiment: pd.DataFrame,
    truth: pd.DataFrame,
    feature_fn,
    n_draws: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """Repeat the opt-in draw ``n_draws`` times: mean bias, RMSE and CI coverage per estimator.

    One draw can mislead in either direction (heavy-tailed revenue makes the
    IPW-type correction terms noisy), so the README reports this table next to
    the single-draw figure.
    """
    rows = []
    for k in range(n_draws):
        obs = simulate_opt_in(experiment, truth, seed=seed * 1000 + k)
        tab = observational_estimates(obs, feature_fn(obs), seed=seed + k)
        rows.append(tab[["bias", "covers_truth", "se"]].assign(draw=k))
    df = pd.concat(rows)
    out = df.groupby(level=0, sort=False).agg(
        mean_bias=("bias", "mean"),
        rmse=("bias", lambda b: float(np.sqrt(np.mean(b**2)))),
        mean_se=("se", "mean"),
        ci95_coverage=("covers_truth", "mean"),
    )
    out["n_draws"] = n_draws
    return out
