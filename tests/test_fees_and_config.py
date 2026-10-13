import numpy as np
import pytest

from tierlift.config import load_config
from tierlift.fees import FeeSchedule, distance_bucket


def test_config_loads_and_validates():
    cfg = load_config()
    assert cfg.population.n_clients == 5000
    assert abs(sum(cfg.experiment.cells.values()) - 1) < 1e-9
    assert cfg.treatment_cells == ["nudge", "incentive"]


def test_config_override_rejects_bad_shares():
    with pytest.raises(ValueError):
        load_config(experiment={"cells": {"holdout": 0.5, "nudge": 0.6, "incentive": 0.1}})


def test_tier_index_boundaries():
    fees = FeeSchedule.from_config(load_config())
    v = np.array([0, 24_999_999, 25_000_000, 99_999_999, 100_000_000, 500_000_000, 1e10])
    assert fees.tier_index(v).tolist() == [0, 0, 1, 1, 2, 3, 3]


def test_blended_rate_is_between_maker_and_taker():
    fees = FeeSchedule.from_config(load_config())
    t = np.array([0, 1, 2, 3])
    m = np.array([0.0, 0.5, 1.0, 0.3])
    r = fees.blended_rate(t, m)
    assert np.all(r <= fees.taker[t] + 1e-12) and np.all(r >= fees.maker[t] - 1e-12)
    assert r[0] == pytest.approx(fees.taker[0])
    assert r[2] == pytest.approx(fees.maker[2])


def test_distance_and_buckets():
    fees = FeeSchedule.from_config(load_config())
    v = np.array([90e6, 22e6, 10e6, 600e6])
    d = fees.distance_to_next(v)
    assert d[0] == pytest.approx(0.10)
    assert d[1] == pytest.approx(0.12)
    assert np.isnan(d[3])
    assert distance_bucket(d).tolist() == ["within_10", "10_to_20", "20_plus", "top_tier"]
