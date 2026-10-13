"""From CATE to a decision: who gets the incentive? (course video 19, policy learning)

A CATE model ranks clients. A *policy* says where to stop. Two policies are
compared against blanket rollout:

1. **Rule policy**: the business-readable rule "clients 10-20% from the next
   tier".
2. **Learned threshold policies**: target client i iff tau_hat_i > 0 in *net
   revenue* terms (the incentive's rate cost is already inside revenue),
   optionally with a fixed per-client campaign cost.
3. **Policy tree**: a depth-2 tree grown directly on the doubly robust scores,
   reported both in-sample (optimistic) and cross-fitted (honest).

The value of a policy pi on the holdout-vs-treated data is estimated with the
AIPW / doubly-robust scores Gamma_i (course video 19):

    V(pi) = mean_i [ pi(x_i) * Gamma_i ]       where E[Gamma_i | x] = tau(x)

so it needs no model of Y, only the randomisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from econml.policy import PolicyTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.tree import export_text

from tierlift.experiment.ate import aipw


def dr_scores(
    y: np.ndarray, d: np.ndarray, X: np.ndarray, seed: int = 0, n_folds: int = 5
) -> np.ndarray:
    """Per-client doubly-robust effect scores Gamma_i (cross-fitted)."""
    y, d, X = np.asarray(y, float), np.asarray(d, bool), np.asarray(X, float)
    n = len(y)
    e = d.mean()
    mu0, mu1 = np.zeros(n), np.zeros(n)
    model = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, random_state=seed
    )
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        mu1[te] = model.fit(X[tr][d[tr]], y[tr][d[tr]]).predict(X[te])
        mu0[te] = model.fit(X[tr][~d[tr]], y[tr][~d[tr]]).predict(X[te])
    return (mu1 - mu0) + d * (y - mu1) / e - (~d) * (y - mu0) / (1 - e)


def policy_value(target: np.ndarray, gamma: np.ndarray) -> dict:
    """Mean incremental value per client of targeting ``target`` (bool), with SE."""
    target = np.asarray(target, bool)
    contrib = target * gamma
    return {
        "value_per_client": float(contrib.mean()),
        "se": float(contrib.std(ddof=1) / np.sqrt(len(gamma))),
        "share_targeted": float(target.mean()),
    }


def rule_policy(df: pd.DataFrame, band: tuple[float, float]) -> np.ndarray:
    d = df["distance_pct"]
    return ((d > band[0]) & (d <= band[1])).to_numpy()


def cate_policy(tau_dollars: np.ndarray, cost_per_client: float = 0.0) -> np.ndarray:
    return np.asarray(tau_dollars) > cost_per_client


def fit_policy_tree(
    X: pd.DataFrame, gamma: np.ndarray, max_depth: int = 2, seed: int = 0
) -> PolicyTree:
    """EconML's policy tree grown on doubly robust scores: an interpretable who-to-target rule.

    Each leaf picks "treat" if the mean score of its clients is positive. We
    grow it on the *index-scale* scores (revenue / pre-period revenue, the same
    outcome the uplift learners use) because dollar scores are dominated by a
    few whales and the greedy splits then chase their noise; the policy's value
    is still reported in dollars by :func:`compare_policies`.

    ``honest=False``: EconML's honest option grows the splits on one half of
    the data and fills the leaves with the other half; with 3,500 clients that
    made the splits unstable. Honesty comes from :func:`policy_tree_oof`
    instead (score the tree on clients it never saw). ``min_samples_leaf=100``
    keeps leaves large enough to mean something.
    """
    scores = np.c_[np.zeros(len(gamma)), np.asarray(gamma, float)]  # don't-treat scores 0
    tree = PolicyTree(max_depth=max_depth, min_samples_leaf=100, honest=False, random_state=seed)
    tree.fit(np.asarray(X, float), scores)
    return tree


def policy_tree_oof(
    X: pd.DataFrame, gamma: np.ndarray, max_depth: int = 2, seed: int = 0, n_folds: int = 2
) -> np.ndarray:
    """Out-of-fold treat/don't decisions of the policy tree, so its value can be scored honestly.

    A tree evaluated on the clients it was grown on looks better than it is
    (it chose its splits to make those very scores look good). Each fold's
    clients are decided by a tree grown on the other folds.
    """
    Xa, gamma = np.asarray(X, float), np.asarray(gamma, float)
    target = np.zeros(len(Xa), bool)
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(Xa):
        tree = fit_policy_tree(Xa[tr], gamma[tr], max_depth, seed)
        target[te] = tree.predict(Xa[te]).astype(bool)
    return target


def policy_tree_text(tree: PolicyTree, feature_names: list[str]) -> str:
    return export_text(tree.tree_model_, feature_names=list(feature_names), show_weights=False)


def compare_policies(
    df: pd.DataFrame,
    X: np.ndarray,
    y_dollars: np.ndarray,
    d: np.ndarray,
    tau_dollars: dict[str, np.ndarray],
    band: tuple[float, float],
    seed: int = 0,
    extra_policies: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Score every candidate policy with the same doubly robust scores Gamma."""
    gamma = dr_scores(y_dollars, d, X, seed)
    rows = {
        "Blanket rollout": policy_value(np.ones(len(df), bool), gamma),
        f"Rule: {int(band[0] * 100)}-{int(band[1] * 100)}% from tier": policy_value(
            rule_policy(df, band), gamma
        ),
    }
    for name, tau in tau_dollars.items():
        rows[f"CATE > 0: {name}"] = policy_value(cate_policy(tau), gamma)
    for name, target in (extra_policies or {}).items():
        rows[name] = policy_value(target, gamma)
    out = pd.DataFrame(rows).T
    out["total_value"] = out["value_per_client"] * len(df)
    return out


__all__ = [
    "dr_scores",
    "policy_value",
    "rule_policy",
    "cate_policy",
    "fit_policy_tree",
    "policy_tree_oof",
    "policy_tree_text",
    "compare_policies",
    "aipw",
]
