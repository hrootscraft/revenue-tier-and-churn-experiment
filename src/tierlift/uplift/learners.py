"""The uplift model zoo: S-, T-, X-, R-, DR-learners and CausalForestDML.

All six estimate the same object, the conditional average treatment effect

    tau(x) = E[Y(1) - Y(0) | X = x],

and differ only in *how* they get there. The one-line summary of each:

* **S-learner**: one model mu(x, d). tau = mu(x, 1) - mu(x, 0).
  Simple, but regularisation can shrink the treatment term toward zero.
* **T-learner**: two models, mu1(x) and mu0(x). tau = mu1 - mu0.
  No shared strength: the difference of two noisy fits is noisier still.
* **X-learner**: T-learner, then regress the imputed effects
  D1 = Y - mu0(x) on treated and D0 = mu1(x) - Y on controls, and blend with
  the propensity. Shines when one arm is much smaller than the other.
* **R-learner** (orthogonal / "DML"): residualise Y and D on X, then fit
  tau by minimising  sum[(Y - m(x)) - tau(x)(D - e(x))]^2  (the R-loss).
  Course section 5. Robust to errors in the nuisance models.
* **DR-learner**: build AIPW pseudo-outcomes per client, then regress them
  on X. Also orthogonal; course sections 3 + 5.
* **CausalForestDML**: a forest of honest causal trees fitted on the
  R-loss residuals. Course section 4. Gives pointwise confidence intervals.

Outcome scale
-------------
Models are fit on the **revenue index** = campaign-quarter revenue /
pre-period revenue. Dollar revenue spans four orders of magnitude, and in
dollar space every learner spends its capacity on predicting client *size*
rather than *response*. Working in index space is the uplift analogue of
CUPED. Dollar effects are recovered by multiplying by pre-period revenue.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, NonParamDML
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold


def default_regressor(seed: int = 0) -> HistGradientBoostingRegressor:
    """The nuisance / outcome model shared by every learner, so the benchmark
    compares *strategies*, not base learners."""
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, random_state=seed
    )


def known_propensity() -> DummyClassifier:
    """In an RCT the propensity is a known constant. A DummyClassifier with
    strategy='prior' returns exactly the assignment share, so EconML's
    propensity-aware learners use the truth instead of an estimate."""
    return DummyClassifier(strategy="prior")


@dataclass
class LearnerSpec:
    name: str
    family: str
    build: Any  # callable(seed) -> fitted-able EconML estimator
    notes: str = ""


def model_zoo(seed: int = 0, n_folds: int = 5, cf_trees: int = 1000) -> list[LearnerSpec]:
    reg = default_regressor
    return [
        LearnerSpec(
            "S-learner",
            "meta",
            lambda s: SLearner(overall_model=reg(s)),
            "single model with treatment as a feature",
        ),
        LearnerSpec(
            "T-learner", "meta", lambda s: TLearner(models=reg(s)), "separate model per arm"
        ),
        LearnerSpec(
            "X-learner",
            "meta",
            lambda s: XLearner(models=reg(s), propensity_model=known_propensity()),
            "T-learner + imputed-effect regression, propensity-blended",
        ),
        LearnerSpec(
            "R-learner",
            "orthogonal",
            lambda s: NonParamDML(
                model_y=reg(s),
                model_t=known_propensity(),
                model_final=reg(s),
                discrete_treatment=True,
                cv=n_folds,
                random_state=s,
            ),
            "residual-on-residual (R-loss) with GBM final stage",
        ),
        LearnerSpec(
            "DR-learner",
            "orthogonal",
            lambda s: DRLearner(
                model_propensity=known_propensity(),
                model_regression=reg(s),
                model_final=reg(s),
                cv=n_folds,
                random_state=s,
            ),
            "AIPW pseudo-outcome regression",
        ),
        LearnerSpec(
            "CausalForestDML",
            "forest",
            lambda s: CausalForestDML(
                model_y=reg(s),
                model_t=known_propensity(),
                discrete_treatment=True,
                n_estimators=cf_trees,
                min_samples_leaf=10,
                cv=n_folds,
                random_state=s,
            ),
            "honest causal forest on R-loss residuals",
        ),
    ]


@dataclass
class UpliftResult:
    name: str
    family: str
    tau_index: np.ndarray  # CATE in revenue-index units, cross-fitted (out-of-fold)
    tau_dollars: np.ndarray  # tau_index * pre_revenue
    fit_seconds: float
    models: list = field(default_factory=list)  # fitted per-fold estimators


def fit_crossfit(
    spec: LearnerSpec,
    X: np.ndarray,
    y_index: np.ndarray,
    d: np.ndarray,
    pre_revenue: np.ndarray,
    n_folds: int = 4,
    seed: int = 0,
    n_repeats: int = 2,
) -> UpliftResult:
    """Fit a learner K times, each time predicting tau for the held-out fold.

    Every client's score comes from a model that never saw their outcome.
    That is what makes Qini / AUUC on these scores an honest evaluation, and
    it is the same cross-fitting idea the course uses for nuisance models.

    ``n_repeats`` repeats the whole K-fold split with a different shuffle and
    averages the scores. With 5,000 clients a single split leaves enough
    Monte-Carlo noise in the scores to reorder the leaderboard; averaging two
    or three splits is the standard remedy (Chernozhukov et al. 2018, sec. 3.4).
    """
    import time

    t0 = time.perf_counter()
    tau = np.zeros(len(y_index))
    models = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(n_repeats):
            splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed + 1000 * r)
            for k, (tr, te) in enumerate(splitter.split(X, d)):
                est = spec.build(seed + 10 * r + k)
                est.fit(y_index[tr], d[tr], X=X[tr])
                tau[te] += np.asarray(est.effect(X[te])).ravel() / n_repeats
                models.append(est)
    return UpliftResult(
        spec.name, spec.family, tau, tau * pre_revenue, time.perf_counter() - t0, models
    )


def fit_zoo(
    X: pd.DataFrame | np.ndarray,
    y_index: np.ndarray,
    d: np.ndarray,
    pre_revenue: np.ndarray,
    n_folds: int = 4,
    seed: int = 0,
    specs: list[LearnerSpec] | None = None,
    n_repeats: int = 2,
) -> dict[str, UpliftResult]:
    X = np.asarray(X, float)
    out = {}
    for spec in specs or model_zoo(seed):
        out[spec.name] = fit_crossfit(spec, X, y_index, d, pre_revenue, n_folds, seed, n_repeats)
    return out


def fit_full(spec: LearnerSpec, X: np.ndarray, y_index: np.ndarray, d: np.ndarray, seed: int = 0):
    """Fit one learner on all data (for inference / interpretation, not evaluation)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = spec.build(seed)
        est.fit(y_index, d, X=X)
    return est


__all__ = [
    "LearnerSpec",
    "UpliftResult",
    "model_zoo",
    "fit_crossfit",
    "fit_zoo",
    "fit_full",
    "default_regressor",
    "known_propensity",
    "copy",
]
