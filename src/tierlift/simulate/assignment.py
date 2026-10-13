"""Step 2: stratified multi-cell randomisation with a global holdout.

Why stratify? With 5,000 clients, simple randomisation would still be
balanced on average, but any one draw can put, say, 25% of the 10-20% bucket
in the holdout instead of 20%. That is exactly the subgroup the business
decision depends on. Stratifying on tier x distance bucket x segment pins the
cell shares inside every stratum, which

1. removes chance imbalance on the variables that matter most, and
2. lowers the variance of the effect estimates (stratification is a form of
   covariate adjustment that happens at design time instead of analysis time).

Inside each stratum we build a label vector with the target shares (remainders
are allocated at random, so small strata don't systematically favour one cell),
then shuffle it. This is the "permuted block" idea used in clinical trials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tierlift.config import Config


def assign_cells(clients: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    cells = list(cfg.experiment.cells)
    shares = np.array([cfg.experiment.cells[c] for c in cells])
    strata = clients[list(cfg.experiment.stratify_on)].astype(str).agg("|".join, axis=1)

    out = pd.Series(index=clients.index, dtype=object)
    for _, idx in strata.groupby(strata).groups.items():
        n = len(idx)
        exact = shares * n
        counts = np.floor(exact).astype(int)
        # Hand the leftover units to cells at random, weighted by their fractional part.
        leftover = n - counts.sum()
        if leftover:
            frac = exact - counts
            extra = rng.choice(len(cells), size=leftover, replace=False, p=frac / frac.sum())
            counts[extra] += 1
        labels = np.repeat(cells, counts)
        rng.shuffle(labels)
        out.loc[idx] = labels

    return pd.DataFrame(
        {
            "client_id": clients["client_id"].to_numpy(),
            "cell": out.to_numpy(),
            "stratum": strata.to_numpy(),
        }
    )
