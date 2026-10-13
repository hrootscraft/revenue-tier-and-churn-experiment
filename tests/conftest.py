import numpy as np
import pytest

from tierlift.config import load_config
from tierlift.simulate import simulate


@pytest.fixture(scope="session")
def small_cfg():
    """A 1,200-client config so the whole suite runs in well under a minute."""
    return load_config(seed=7, population={"n_clients": 1200})


@pytest.fixture(scope="session")
def small_data(small_cfg):
    return simulate(small_cfg)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
