"""Simulation of the fee-tier incentive experiment.

``simulate(cfg)`` runs the three steps (population, assignment, outcomes)
with one seeded random generator, so the same config always gives the same data.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from tierlift.config import Config
from tierlift.simulate.assignment import assign_cells
from tierlift.simulate.outcomes import simulate_outcomes
from tierlift.simulate.population import generate_population


@dataclass
class SimulatedData:
    clients: pd.DataFrame  # covariates + pre-period summary (observed)
    pre_weekly: pd.DataFrame  # weekly pre-period volume (observed)
    assignment: pd.DataFrame  # client -> cell (observed)
    experiment: pd.DataFrame  # one row per client, analysis-ready (observed)
    weekly: pd.DataFrame  # weekly post-launch volume panel (observed)
    truth: pd.DataFrame  # potential outcomes and planted effects (NOT observed)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for f in fields(self):
            getattr(self, f.name).to_parquet(directory / f"{f.name}.parquet", index=False)

    @classmethod
    def load(cls, directory: str | Path) -> SimulatedData:
        directory = Path(directory)
        return cls(
            **{f.name: pd.read_parquet(directory / f"{f.name}.parquet") for f in fields(cls)}
        )


def simulate(cfg: Config) -> SimulatedData:
    rng = np.random.default_rng(cfg.seed)
    clients, pre_weekly = generate_population(cfg, rng)
    assignment = assign_cells(clients, cfg, rng)
    out = simulate_outcomes(clients, assignment, cfg, rng)
    return SimulatedData(
        clients=clients.drop(columns=["latent_weekly_volume"]),
        pre_weekly=pre_weekly,
        assignment=assignment,
        experiment=out["experiment"],
        weekly=out["weekly"],
        truth=out["truth"],
    )


__all__ = ["SimulatedData", "simulate"]
