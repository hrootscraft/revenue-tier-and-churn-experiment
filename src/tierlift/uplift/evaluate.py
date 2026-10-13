"""Evaluating uplift models: Qini, AUUC, RATE/TOC, calibration, and oracle metrics.

The central difficulty (course section 6, "validating CATE estimates"): we
never observe a client's true effect, so we cannot compute an RMSE against
it. Instead we ask: *if we rank clients by the model's score, do the
highest-ranked clients really show the largest treatment-vs-holdout gap?*

Every curve below is computed on **cross-fitted** scores (each client scored
by a model that did not see their outcome), with the randomised holdout as
the counterfactual.

Qini curve (Radcliffe, 2007)
----------------------------
Sort clients by score, descending. For the top k clients:

    Qini(k) = sum_{treated in top k} Y  -  sum_{holdout in top k} Y * (n_t(k) / n_c(k))

i.e. the cumulative treated outcome minus the holdout outcome scaled to the
same number of clients. We report it per client (divided by N) so it reads as
"incremental revenue index per client" and compare it against the random line
that goes straight from 0 to the ATE.

Uplift curve / AUUC
-------------------
Same idea but on *means*: gain(k) = (mean_t(k) - mean_c(k)) * k / N.
AUUC is the area under it. We centre both curves on the random-targeting line,
so a model with no ranking skill scores ~0 and a negative area means the model
ranks backwards.

TOC / RATE (Yadlowsky, Fleming, Shah, Brunskill & Wager, 2021)
---------------------------------------------------------------
TOC(q) = ATE among the top q fraction  -  overall ATE. RATE (AUTOC) is its
integral, a single number for "how much better than random is this ranking".
Standard errors via the bootstrap.

Calibration / BLP (Chernozhukov, Demirer, Duflo & Fernandez-Val, 2018)
------------------------------------------------------------------------
Regress Y on (D - e) * 1 and (D - e) * (tau_hat - mean tau_hat) with the
known propensity e. A slope of 1 on the second term means the model's
*spread* of effects is right; 0 means it carries no heterogeneity signal.

Oracle metrics
--------------
Because the data is simulated we also know each client's true ITE, so we
report PEHE (RMSE vs the truth) and the rank correlation. These exist only to
*validate the validators*: the observational metrics should agree with them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def _sorted_arms(score: np.ndarray, y: np.ndarray, d: np.ndarray):
    o = np.argsort(-score, kind="stable")
    return y[o], d[o].astype(bool)


def qini_curve(score: np.ndarray, y: np.ndarray, d: np.ndarray) -> pd.DataFrame:
    """Per-client Qini curve with the random-targeting baseline."""
    y, d = _sorted_arms(score, y, d)
    n = len(y)
    n_t = np.cumsum(d)
    n_c = np.cumsum(~d)
    s_t = np.cumsum(y * d)
    s_c = np.cumsum(y * ~d)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (s_t - s_c * np.where(n_c > 0, n_t / np.maximum(n_c, 1), 0)) / n
    frac = np.arange(1, n + 1) / n
    random = frac * q[-1]
    return pd.DataFrame(
        {"fraction_targeted": frac, "qini": q, "random": random, "qini_above_random": q - random}
    )


def uplift_curve(score: np.ndarray, y: np.ndarray, d: np.ndarray) -> pd.DataFrame:
    y, d = _sorted_arms(score, y, d)
    n = len(y)
    n_t = np.cumsum(d)
    n_c = np.cumsum(~d)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_t = np.cumsum(y * d) / np.maximum(n_t, 1)
        m_c = np.cumsum(y * ~d) / np.maximum(n_c, 1)
    frac = np.arange(1, n + 1) / n
    gain = np.where((n_t > 0) & (n_c > 0), (m_t - m_c) * frac, 0.0)
    return pd.DataFrame(
        {
            "fraction_targeted": frac,
            "gain": gain,
            "random": frac * gain[-1],
            "gain_above_random": gain - frac * gain[-1],
        }
    )


def toc_curve(
    score: np.ndarray, y: np.ndarray, d: np.ndarray, grid: np.ndarray | None = None
) -> pd.DataFrame:
    """Targeting Operator Characteristic: ATE(top q) - ATE(all)."""
    y, d = _sorted_arms(score, y, d)
    n = len(y)
    grid = np.linspace(0.02, 1.0, 50) if grid is None else grid
    ate_all = y[d].mean() - y[~d].mean()
    rows = []
    for q in grid:
        k = max(int(np.ceil(q * n)), 2)
        yt, dt = y[:k], d[:k]
        if dt.sum() == 0 or (~dt).sum() == 0:
            rows.append((q, np.nan))
            continue
        rows.append((q, yt[dt].mean() - yt[~dt].mean() - ate_all))
    return pd.DataFrame(rows, columns=["fraction_targeted", "toc"])


def rate(
    score: np.ndarray, y: np.ndarray, d: np.ndarray, n_bootstrap: int = 200, seed: int = 0
) -> dict:
    """AUTOC (area under TOC) with bootstrap SE and a one-sided p-value for > 0."""
    rng = np.random.default_rng(seed)
    score, y, d = map(np.asarray, (score, y, d))
    point = np.nanmean(toc_curve(score, y, d)["toc"])
    boots = np.empty(n_bootstrap)
    n = len(y)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boots[b] = np.nanmean(toc_curve(score[idx], y[idx], d[idx])["toc"])
    se = boots.std(ddof=1)
    return {
        "autoc": float(point),
        "se": float(se),
        "p_value": float(stats.norm.sf(point / se)) if se > 0 else np.nan,
    }


def auuc(score: np.ndarray, y: np.ndarray, d: np.ndarray) -> float:
    return float(uplift_curve(score, y, d)["gain_above_random"].mean())


def qini_coefficient(score: np.ndarray, y: np.ndarray, d: np.ndarray) -> float:
    return float(qini_curve(score, y, d)["qini_above_random"].mean())


def blp_calibration(
    score: np.ndarray, y: np.ndarray, d: np.ndarray, propensity: float | None = None
) -> dict:
    """Best Linear Predictor test of heterogeneity (Chernozhukov et al. 2018).

    y = a + b1 * (d - e) + b2 * (d - e) * (s - mean s) + error
    b1 estimates the ATE; b2 = 1 means perfectly calibrated heterogeneity.
    """
    score, y, d = map(lambda a: np.asarray(a, float), (score, y, d))
    e = d.mean() if propensity is None else propensity
    w = d - e
    X = np.column_stack([np.ones_like(y), w, w * (score - score.mean())])
    fit = sm.OLS(y, X).fit(cov_type="HC2")
    return {
        "ate_coef": float(fit.params[1]),
        "heterogeneity_coef": float(fit.params[2]),
        "heterogeneity_se": float(fit.bse[2]),
        "heterogeneity_p": float(fit.pvalues[2]),
    }


def oracle_metrics(score: np.ndarray, true_ite: np.ndarray) -> dict:
    """Only possible because the data is simulated."""
    score, true_ite = np.asarray(score, float), np.asarray(true_ite, float)
    return {
        "pehe": float(np.sqrt(np.mean((score - true_ite) ** 2))),
        "spearman_vs_truth": float(stats.spearmanr(score, true_ite)[0]),
        "pearson_vs_truth": float(np.corrcoef(score, true_ite)[0, 1]),
    }


def perfect_ranking_auuc(true_ite: np.ndarray, y: np.ndarray, d: np.ndarray) -> float:
    """What a model that knew the truth would score on the observed metric: the ceiling."""
    return auuc(true_ite, y, d)


def benchmark_table(
    results: dict,
    y: np.ndarray,
    d: np.ndarray,
    true_ite: np.ndarray | None = None,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        s = res.tau_index
        row = {
            "model": name,
            "family": res.family,
            "qini": qini_coefficient(s, y, d),
            "auuc": auuc(s, y, d),
            **{f"rate_{k}": v for k, v in rate(s, y, d, n_bootstrap, seed).items()},
            **{f"blp_{k}": v for k, v in blp_calibration(s, y, d).items()},
            "fit_seconds": res.fit_seconds,
        }
        if true_ite is not None:
            row.update(oracle_metrics(s, true_ite))
        rows.append(row)
    out = pd.DataFrame(rows).set_index("model")
    if true_ite is not None:
        out.attrs["perfect_auuc"] = perfect_ranking_auuc(true_ite, y, d)
        out.attrs["perfect_qini"] = qini_coefficient(true_ite, y, d)
    return out.sort_values("auuc", ascending=False)
