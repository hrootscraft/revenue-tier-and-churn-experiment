"""Robustness tests: do the conclusions survive different assumptions?

Revenue per client is heavily right-skewed: a few whales carry a large share of
dollars. That puts the t-test's reliance on approximate normality of the
*mean* under strain. So we ask the same question three ways:

* **Welch's t-test**: the difference in means, with no equal-variance
  assumption. It relies on the CLT for the sample means.
* **Mann-Whitney U**: a rank test. Is a random treated client likely to
  out-earn a random holdout client? It is insensitive to whales, but it tests a
  *different* null (stochastic equality, not equal means).
* **Permutation test**: the exact randomisation distribution of the
  difference in means. Labels are reshuffled *within strata*, because that is
  how they were assigned. It needs no distributional assumption at all.

When all three agree we can be confident. When they disagree, the
disagreement itself is informative (e.g. a mean effect driven by a few whales).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def welch(y1: np.ndarray, y0: np.ndarray) -> dict:
    t, p = stats.ttest_ind(y1, y0, equal_var=False)
    return {"test": "Welch t", "statistic": float(t), "p_value": float(p)}


def mann_whitney(y1: np.ndarray, y0: np.ndarray) -> dict:
    u, p = stats.mannwhitneyu(y1, y0, alternative="two-sided")
    # Common-language effect size: P(random treated > random control).
    return {
        "test": "Mann-Whitney U",
        "statistic": float(u),
        "p_value": float(p),
        "prob_superiority": float(u / (len(y1) * len(y0))),
    }


def permutation(
    y: np.ndarray,
    d: np.ndarray,
    strata: np.ndarray | None = None,
    n_permutations: int = 5000,
    seed: int = 0,
) -> dict:
    """Two-sided stratified permutation test of the difference in means."""
    rng = np.random.default_rng(seed)
    y, d = np.asarray(y, float), np.asarray(d, bool)
    strata = np.zeros(len(y), int) if strata is None else pd.factorize(np.asarray(strata))[0]
    observed = y[d].mean() - y[~d].mean()
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    null = np.empty(n_permutations)
    perm = d.copy()
    for b in range(n_permutations):
        for g in groups:
            perm[g] = rng.permutation(d[g])
        null[b] = y[perm].mean() - y[~perm].mean()
    p = (1 + np.sum(np.abs(null) >= abs(observed))) / (n_permutations + 1)
    return {
        "test": "Permutation (stratified)",
        "statistic": float(observed),
        "p_value": float(p),
        "null_sd": float(null.std()),
    }


def winsorize(y: np.ndarray, q: float) -> np.ndarray:
    return np.minimum(y, np.quantile(y, q))


def robustness_table(
    y: np.ndarray,
    d: np.ndarray,
    strata: np.ndarray | None = None,
    n_permutations: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    y, d = np.asarray(y, float), np.asarray(d, bool)
    rows = [
        welch(y[d], y[~d]),
        mann_whitney(y[d], y[~d]),
        permutation(y, d, strata, n_permutations, seed),
    ]
    return pd.DataFrame(rows).set_index("test")


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values. Controls the family-wise error rate
    across the treatment-vs-holdout comparisons of a multi-cell test."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    return adjusted
