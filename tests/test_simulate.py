import numpy as np

from tierlift.simulate import simulate
from tierlift.validation.checks import run_all_checks


def test_reproducible(small_cfg):
    a, b = simulate(small_cfg), simulate(small_cfg)
    assert a.experiment.equals(b.experiment)


def test_all_quality_checks_pass(small_data, small_cfg):
    table = run_all_checks(small_data, small_cfg)  # raises if a blocking check fails
    assert table.loc[table["blocking"], "passed"].all(), table[~table["passed"]]


def test_cells_balanced_within_strata(small_data, small_cfg):
    e = small_data.experiment
    big = e.groupby("stratum").filter(lambda g: len(g) >= 40)
    shares = big.groupby("stratum")["cell"].value_counts(normalize=True).unstack(fill_value=0)
    for cell, planned in small_cfg.experiment.cells.items():
        assert (shares[cell] - planned).abs().max() < 0.06


def test_potential_outcomes_consistent(small_data):
    t = small_data.truth
    # Holdout potential outcome is never affected by the incentive mechanics.
    assert (t["volume_incentive"] >= t["volume_holdout"] * 0.999).mean() > 0.95
    # ITE columns are differences of the stored potential outcomes.
    np.testing.assert_allclose(
        t["ite_revenue_incentive"], t["revenue_incentive"] - t["revenue_holdout"]
    )


def test_planted_truth_has_the_right_shape(small_data):
    e = small_data.experiment.merge(small_data.truth, on="client_id")
    lift = e.groupby("distance_bucket")["tau_volume_incentive"].mean()
    assert lift["10_to_20"] > lift["within_10"] > lift["20_plus"] > lift.get("top_tier", -1)


def test_lapse_hazard_near_target(small_data, small_cfg):
    t = small_data.experiment[small_data.experiment["cell"] != "holdout"]
    events = t["lapsed"].sum()
    exposure = t["lapse_week"].sum()
    assert abs(events / exposure - small_cfg.truth.base_weekly_hazard) < 0.006


def test_truth_not_in_observed_tables(small_data):
    forbidden = {
        "tau_volume_incentive",
        "revenue_holdout",
        "revenue_incentive",
        "lapse_time",
        "drift",
        "latent_weekly_volume",
    }
    assert not forbidden & set(small_data.experiment.columns)
    assert not forbidden & set(small_data.clients.columns)
