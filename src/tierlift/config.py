"""Typed configuration loaded from ``configs/default.yaml``.

All tunable numbers live in YAML so a reviewer can see the planted truth in one
place. This module turns that YAML into frozen dataclasses so typos fail loudly
(``cfg.experiment.cels`` raises) instead of silently returning ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Segment:
    name: str
    share: float
    volume_mult: float
    maker_ratio: float
    api_share: float
    responsiveness: float
    weekly_cv: float


@dataclass(frozen=True)
class Tier:
    name: str
    min_volume: float
    maker_bps: float
    taker_bps: float


@dataclass(frozen=True)
class Population:
    n_clients: int
    median_quarterly_volume: float
    log_sd_quarterly_volume: float
    segments: tuple[Segment, ...]
    regions: dict[str, float]


@dataclass(frozen=True)
class Experiment:
    pre_weeks: int
    experiment_weeks: int
    followup_weeks: int
    cells: dict[str, float]
    stratify_on: tuple[str, ...]
    incentive_half_step: float
    treated_maker_ratio_shift: float


@dataclass(frozen=True)
class Truth:
    drift_flow_coef: float
    drift_log_sd: float
    incentive_lift_by_distance: dict[str, float]
    nudge_scale: float
    lift_log_sd: float
    engagement_lift_coef: float
    api_lift_coef: float
    base_weekly_hazard: float
    hazard_coefs: dict[str, float]
    confounders: dict[str, Any]


@dataclass(frozen=True)
class Analysis:
    alpha: float
    n_permutations: int
    n_bootstrap: int
    winsor_quantile: float
    test_size: float
    n_folds: int
    primary_contrast: tuple[str, str]
    targeting_band: tuple[float, float]
    horizon_weeks: int


@dataclass(frozen=True)
class Config:
    seed: int
    population: Population
    tiers: tuple[Tier, ...]
    experiment: Experiment
    truth: Truth
    analysis: Analysis
    source: str = field(default="<dict>")

    @property
    def treatment_cells(self) -> list[str]:
        """Cells other than the holdout, in config order."""
        return [c for c in self.experiment.cells if c != "holdout"]


def _build(raw: dict[str, Any], source: str) -> Config:
    pop = raw["population"]
    segments = tuple(Segment(name=k, **v) for k, v in pop["segments"].items())
    population = Population(
        n_clients=int(pop["n_clients"]),
        median_quarterly_volume=float(pop["median_quarterly_volume"]),
        log_sd_quarterly_volume=float(pop["log_sd_quarterly_volume"]),
        segments=segments,
        regions=dict(pop["regions"]),
    )
    tiers = tuple(Tier(**t) for t in raw["fees"]["tiers"])
    exp = raw["experiment"]
    experiment = Experiment(
        pre_weeks=exp["pre_weeks"],
        experiment_weeks=exp["experiment_weeks"],
        followup_weeks=exp["followup_weeks"],
        cells=dict(exp["cells"]),
        stratify_on=tuple(exp["stratify_on"]),
        incentive_half_step=float(exp["incentive_half_step"]),
        treated_maker_ratio_shift=float(exp["treated_maker_ratio_shift"]),
    )
    truth = Truth(**raw["truth"])
    an = raw["analysis"]
    analysis = Analysis(
        **{k: v for k, v in an.items() if k not in {"primary_contrast", "targeting_band"}},
        primary_contrast=tuple(an["primary_contrast"]),
        targeting_band=tuple(an["targeting_band"]),
    )
    cfg = Config(
        seed=int(raw["seed"]),
        population=population,
        tiers=tiers,
        experiment=experiment,
        truth=truth,
        analysis=analysis,
        source=source,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    shares = sum(s.share for s in cfg.population.segments)
    if abs(shares - 1) > 1e-9:
        raise ValueError(f"segment shares must sum to 1, got {shares}")
    if abs(sum(cfg.experiment.cells.values()) - 1) > 1e-9:
        raise ValueError("cell shares must sum to 1")
    if "holdout" not in cfg.experiment.cells:
        raise ValueError("the design needs a 'holdout' cell")
    mins = [t.min_volume for t in cfg.tiers]
    if mins != sorted(mins) or mins[0] != 0:
        raise ValueError("tiers must be ordered by min_volume, starting at 0")


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a YAML config. ``overrides`` replace top-level sections' keys,
    e.g. ``load_config(population={"n_clients": 500})`` for fast tests."""
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text())
    for section, values in overrides.items():
        if isinstance(values, dict):
            raw[section] = {**raw[section], **values}
        else:
            raw[section] = values
    return _build(raw, str(path))
