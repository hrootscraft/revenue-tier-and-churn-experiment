"""Blanket rollout vs targeted rollout: the business case.

A rollout policy is a set S of clients who get the incentive. Its quarterly
net revenue impact is

    Impact(S) = sum_{i in S} tau_i

where tau_i is client i's incremental net revenue (treated minus holdout,
with rate compression already inside). We estimate it two ways and show
they agree:

1. **Experimental, doubly robust**: AIPW influence scores are fitted once on
   the whole incentive-vs-holdout sample (3,500 clients) and averaged over S.
   Because S is defined by pre-treatment variables, randomisation still holds
   inside it, so the average is unbiased; because the nuisance models saw
   every client, it is far more precise than refitting inside S. CUPED on the
   clients in S is reported alongside as the simpler estimator.
2. **Oracle**: the planted truth, available only because this is a simulation.

"Net revenue %" is Impact(S) divided by the holdout revenue of S's clients:
"targeting these clients changes *their* revenue by x%". For a blanket
rollout S is everyone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tierlift.experiment.ate import Estimate, aipw_scores, aipw_subgroup, cuped
from tierlift.features import build_features


class ContrastScores:
    """AIPW scores for one treatment-vs-holdout contrast, computed once and reused for every policy."""

    def __init__(self, df: pd.DataFrame, cell: str, control: str = "holdout", seed: int = 0):
        self.in_contrast = df["cell"].isin([cell, control]).to_numpy()
        sub = df[self.in_contrast].reset_index(drop=True)
        self.d = (sub["cell"] == cell).to_numpy()
        self.psi1, self.psi0, self.outcome_r2 = aipw_scores(
            sub["exp_revenue"].to_numpy(float), self.d, build_features(sub), seed=seed
        )
        self.sub = sub

    def estimate(self, target: np.ndarray) -> Estimate:
        """Relative revenue effect of the cell inside ``target`` (a boolean mask over the full df)."""
        mask = np.asarray(target, bool)[self.in_contrast]
        return aipw_subgroup(self.psi1, self.psi0, self.d, mask)

    def cuped(self, target: np.ndarray) -> Estimate:
        mask = np.asarray(target, bool)[self.in_contrast]
        s = self.sub[mask]
        return cuped(
            s["exp_revenue"].to_numpy(float), self.d[mask], s["pre_revenue"].to_numpy(float)
        )


def rollout_table(
    df: pd.DataFrame,
    truth: pd.DataFrame,
    policies: dict[str, np.ndarray],
    cell: str,
    control: str = "holdout",
    seed: int = 0,
    scores: ContrastScores | None = None,
) -> pd.DataFrame:
    """One row per policy: experimental estimate (with CI), oracle truth, cost side."""
    scores = scores or ContrastScores(df, cell, control, seed)
    t = truth.set_index("client_id").loc[df["client_id"]]
    rows = []
    for name, target in policies.items():
        target = np.asarray(target, bool)
        est, est_c = scores.estimate(target), scores.cuped(target)
        n_target = int(target.sum())
        tt = t[target]
        true_total = float(tt[f"ite_revenue_{cell}"].sum())
        true_base = float(tt["revenue_holdout"].sum())
        # Rate compression cost: revenue given away on the volume the exchange would have had anyway.
        rate_cost = float((tt["volume_holdout"] * (tt[f"rate_{cell}"] - tt["rate_holdout"])).sum())
        rows.append(
            {
                "policy": name,
                "clients_targeted": n_target,
                "share_targeted": n_target / len(df),
                "est_net_revenue_pct": est.rel_effect,
                "ci_low_pct": est.rel_ci[0],
                "ci_high_pct": est.rel_ci[1],
                "p_value": est.p_value,
                "est_quarterly_impact": est.effect * n_target,
                "cuped_net_revenue_pct": est_c.rel_effect,
                "cuped_ci_low_pct": est_c.rel_ci[0],
                "cuped_ci_high_pct": est_c.rel_ci[1],
                "true_net_revenue_pct": true_total / true_base,
                "true_quarterly_impact": true_total,
                "true_rate_compression_cost": rate_cost,
                "true_volume_gain": true_total - rate_cost,
            }
        )
    return pd.DataFrame(rows).set_index("policy")


def targeting_band_grid(
    df: pd.DataFrame,
    cell: str,
    bands: list[tuple[float, float]],
    seed: int = 0,
    scores: ContrastScores | None = None,
) -> pd.DataFrame:
    """Sweep candidate distance bands to show where the sweet spot is (and that it is a band, not a point)."""
    scores = scores or ContrastScores(df, cell, seed=seed)
    rows = []
    for lo, hi in bands:
        target = ((df["distance_pct"] > lo) & (df["distance_pct"] <= hi)).to_numpy()
        est = scores.estimate(target)
        rows.append(
            {
                "band": f"{lo:.0%}-{hi:.0%}",
                "n": int(target.sum()),
                "net_revenue_pct": est.rel_effect,
                "ci_low": est.rel_ci[0],
                "ci_high": est.rel_ci[1],
            }
        )
    return pd.DataFrame(rows).set_index("band")
