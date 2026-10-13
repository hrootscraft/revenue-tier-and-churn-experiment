"""Average treatment effect (ATE) estimators for a randomised experiment.

Four estimators, all unbiased under randomisation. They differ in variance:

1. ``difference_in_means``: the baseline.
2. ``cuped``: subtract theta * (pre-period metric - its mean) from the outcome.
   It removes the part of the outcome that was predictable *before* launch.
   The variance shrinks by a factor (1 - rho^2).
3. ``lin_regression``: OLS of y on treatment, centred covariates, and their
   interactions (Lin, 2013), with HC2 robust standard errors. CUPED is the
   special case with one covariate and no interaction.
4. ``aipw``: augmented inverse-propensity weighting with cross-fitted ML
   outcome models (course section 3: double robustness). In an RCT the
   propensity is known, so the estimator is unbiased whatever the outcome
   model. A good outcome model just makes it tighter.

Every estimator returns an :class:`Estimate` with the absolute effect, the
relative effect (effect / holdout mean), and delta-method standard errors.
Relative effects are what the business quotes ("+11% revenue").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

Z95 = stats.norm.ppf(0.975)


@dataclass
class Estimate:
    estimator: str
    effect: float
    se: float
    control_mean: float
    rel_effect: float
    rel_se: float
    n_treated: int
    n_control: int
    extra: dict = field(default_factory=dict)

    @property
    def ci(self) -> tuple[float, float]:
        return self.effect - Z95 * self.se, self.effect + Z95 * self.se

    @property
    def rel_ci(self) -> tuple[float, float]:
        return self.rel_effect - Z95 * self.rel_se, self.rel_effect + Z95 * self.rel_se

    @property
    def p_value(self) -> float:
        return float(2 * stats.norm.sf(abs(self.effect / self.se)))

    def to_dict(self) -> dict:
        d = asdict(self)
        extra = d.pop("extra")
        d.update(
            ci_low=self.ci[0],
            ci_high=self.ci[1],
            rel_ci_low=self.rel_ci[0],
            rel_ci_high=self.rel_ci[1],
            p_value=self.p_value,
            **extra,
        )
        return d


def _from_influence(
    name: str, psi1: np.ndarray, psi0: np.ndarray, n1: int, n0: int, **extra
) -> Estimate:
    """Build an Estimate from per-unit influence-function scores of the two arm means.

    m1 = mean(psi1), m0 = mean(psi0); ATE = m1 - m0; relative = m1/m0 - 1.
    Delta method: IF(m1/m0) = psi1/m0 - m1*psi0/m0^2.
    """
    n = len(psi1)
    m1, m0 = psi1.mean(), psi0.mean()
    se = np.std(psi1 - psi0, ddof=1) / np.sqrt(n)
    if_rel = (psi1 - m1) / m0 - m1 * (psi0 - m0) / m0**2
    rel_se = np.std(if_rel, ddof=1) / np.sqrt(n)
    return Estimate(name, m1 - m0, se, m0, m1 / m0 - 1, rel_se, n1, n0, dict(extra))


def difference_in_means(y: np.ndarray, d: np.ndarray) -> Estimate:
    y, d = np.asarray(y, float), np.asarray(d, bool)
    y1, y0 = y[d], y[~d]
    m1, m0 = y1.mean(), y0.mean()
    v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
    rel_se = np.sqrt(v1 / m0**2 + m1**2 * v0 / m0**4)
    return Estimate(
        "Difference in means", m1 - m0, np.sqrt(v1 + v0), m0, m1 / m0 - 1, rel_se, len(y1), len(y0)
    )


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    """theta = Cov(y, x) / Var(x), pooled across arms (it must not use treatment)."""
    c = np.cov(y, x, ddof=1)
    return float(c[0, 1] / c[1, 1])


def cuped(y: np.ndarray, d: np.ndarray, x: np.ndarray) -> Estimate:
    """CUPED (Deng, Xu, Kohavi & Walker, 2013).

    y_adj = y - theta * (x - mean(x)). Same expected difference, smaller variance.
    The relative effect is still divided by the *raw* holdout mean (the
    adjustment term has mean zero), so it stays interpretable.
    """
    y, d, x = np.asarray(y, float), np.asarray(d, bool), np.asarray(x, float)
    theta = cuped_theta(y, x)
    y_adj = y - theta * (x - x.mean())
    raw = difference_in_means(y, d)
    adj = difference_in_means(y_adj, d)
    var_reduction = 1 - adj.se**2 / raw.se**2
    rho = np.corrcoef(y, x)[0, 1]
    return Estimate(
        "CUPED",
        adj.effect,
        adj.se,
        raw.control_mean,
        adj.effect / raw.control_mean,
        adj.se / raw.control_mean,
        adj.n_treated,
        adj.n_control,
        {"theta": theta, "rho": rho, "variance_reduction": var_reduction},
    )


def lin_regression(y: np.ndarray, d: np.ndarray, X: pd.DataFrame | np.ndarray) -> Estimate:
    """Lin (2013) fully interacted regression adjustment with HC2 standard errors."""
    y, d = np.asarray(y, float), np.asarray(d, float)
    Xc = np.asarray(X, float)
    Xc = Xc - Xc.mean(axis=0)
    design = np.column_stack([np.ones_like(d), d, Xc, Xc * d[:, None]])
    # A dummy that is constant inside one arm gives an all-zero interaction column; drop it.
    keep = np.r_[True, True, design[:, 2:].std(axis=0) > 1e-12]
    fit = sm.OLS(y, design[:, keep]).fit(cov_type="HC2")
    effect, se = fit.params[1], fit.bse[1]
    # With centred covariates the intercept is the covariate-adjusted holdout mean.
    m0 = fit.params[0]
    cov = fit.cov_params()
    rel = effect / m0
    grad = np.array([-effect / m0**2, 1 / m0])
    rel_se = float(np.sqrt(grad @ cov[:2, :2] @ grad))
    return Estimate(
        "Lin regression (HC2)", effect, se, m0, rel, rel_se, int(d.sum()), int((1 - d).sum())
    )


def aipw_scores(
    y: np.ndarray,
    d: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    propensity: float | np.ndarray | None = None,
    model: RegressorMixin | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cross-fitted AIPW influence scores ``(psi1, psi0)`` per unit, plus the outcome model's R².

    psi1_i = mu1(x_i) + d_i (y_i - mu1(x_i)) / e(x_i)
    psi0_i = mu0(x_i) + (1-d_i)(y_i - mu0(x_i)) / (1 - e(x_i))

    mean(psi1 - psi0) over *any* subgroup defined by pre-treatment variables is
    an unbiased estimate of that subgroup's ATE, which is how the rollout table
    scores a targeting rule without refitting models inside a small subgroup.

    Cross-fitting: mu0 and mu1 for client i are fitted on folds that exclude i.
    This stops an over-fitted model from "memorising" y_i and biasing the
    correction term (course section 2: prediction vs estimation).
    """
    y, d = np.asarray(y, float), np.asarray(d, bool)
    X = np.asarray(X, float)
    n = len(y)
    e = np.broadcast_to(d.mean() if propensity is None else propensity, (n,)).astype(float)
    model = model or HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=seed
    )
    mu0, mu1 = np.zeros(n), np.zeros(n)
    for train, test in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        for arm, mu in ((True, mu1), (False, mu0)):
            idx = train[d[train] == arm]
            mu[test] = clone(model).fit(X[idx], y[idx]).predict(X[test])
    psi1 = mu1 + d * (y - mu1) / e
    psi0 = mu0 + (~d) * (y - mu0) / (1 - e)
    r2 = 1 - np.mean((y - np.where(d, mu1, mu0)) ** 2) / np.var(y)
    return psi1, psi0, float(r2)


def aipw(
    y: np.ndarray,
    d: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    propensity: float | np.ndarray | None = None,
    model: RegressorMixin | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> Estimate:
    """Cross-fitted AIPW (augmented IPW, the doubly robust estimator) of the ATE.

    In an RCT the propensity is known, so the estimator is unbiased whatever
    the outcome model; a good outcome model just makes it tighter. See
    :func:`aipw_scores` for the influence-function formulas.
    """
    d = np.asarray(d, bool)
    psi1, psi0, r2 = aipw_scores(y, d, X, propensity, model, n_folds, seed)
    return _from_influence(
        "AIPW (cross-fitted GBM)", psi1, psi0, int(d.sum()), int((~d).sum()), outcome_r2=r2
    )


def aipw_subgroup(
    psi1: np.ndarray,
    psi0: np.ndarray,
    d: np.ndarray,
    mask: np.ndarray,
    name: str = "AIPW (subgroup, full-sample nuisance)",
) -> Estimate:
    """Subgroup ATE from AIPW scores that were fitted on the *full* sample.

    The nuisance models have seen every client, so they are far better
    estimated than models refit inside a 300-client subgroup; averaging the
    scores over the subgroup is still unbiased because membership is decided
    by pre-treatment variables only.
    """
    m, d = np.asarray(mask, bool), np.asarray(d, bool)
    return _from_influence(name, psi1[m], psi0[m], int(d[m].sum()), int((~d[m]).sum()))


def ate_table(
    y: np.ndarray, d: np.ndarray, x_pre: np.ndarray, X: pd.DataFrame, seed: int = 0
) -> pd.DataFrame:
    """All four estimators side by side, plus each one's variance relative to difference in means."""
    ests = [
        difference_in_means(y, d),
        cuped(y, d, x_pre),
        lin_regression(y, d, X),
        aipw(y, d, X, seed=seed),
    ]
    out = pd.DataFrame([e.to_dict() for e in ests]).set_index("estimator")
    out["variance_vs_dim"] = (out["se"] / out["se"].iloc[0]) ** 2
    return out
