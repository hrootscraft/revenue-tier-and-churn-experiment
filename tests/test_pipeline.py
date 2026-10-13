"""End-to-end: the fast pipeline runs on a small population and recovers the planted truth."""

import json

import pytest

from tierlift.config import load_config
from tierlift.pipeline import run_all


@pytest.mark.slow
def test_fast_pipeline_recovers_truth(tmp_path):
    cfg = load_config(seed=3, population={"n_clients": 2000})
    head = run_all(
        cfg,
        out_dir=tmp_path / "results",
        fig_dir=tmp_path / "figs",
        data_dir=tmp_path / "data",
        fast=True,
    )
    assert (tmp_path / "results" / "headline.json").exists()
    assert (tmp_path / "figs" / "05_qini_curves.png").exists()
    # Direction of the headline findings must hold even on a small sample.
    assert 0.3 < head["cuped_variance_reduction"] < 0.7
    roll = head["rollout"]
    blanket = next(v for k, v in roll.items() if k.startswith("Blanket"))
    targeted = next(v for k, v in roll.items() if k.startswith("Targeted"))
    assert blanket["true_net_revenue_pct"] < 0 < targeted["true_net_revenue_pct"]
    assert abs(head["durability"]["weekly_hazard"] - cfg.truth.base_weekly_hazard) < 0.005
    saved = json.load(open(tmp_path / "results" / "headline.json"))
    assert saved["seed"] == 3
