from collections import Counter

import pytest
from pydantic import ValidationError

from gwbenchmark2g.config import DetectorNetworkConfig, Level0Config, Level1Config


def test_detector_network_config(rng):
    config = DetectorNetworkConfig(
        detector_combinations=[
            ("H1", "L1", "V1"),
            ("H1", "L1"),
            ("H1",),
        ],
        weights=[0.5, 0.3, 0.2],
    )
    # Test that sampling works and respects weights
    samples = [tuple(config.sample_network(rng)) for _ in range(1000)]
    counts = Counter(samples)
    assert counts[("H1", "L1", "V1")] > counts[("H1", "L1")] > counts[("H1",)]


@pytest.mark.parametrize("config_class", [Level0Config, Level1Config])
@pytest.mark.parametrize(
    "bounds",
    [
        (0, 2000),
        (-1, 2000),
        (2000, 2000),
        (3000, 1000),
        (1000, float("inf")),
        (float("nan"), 2000),
        [1000],
        [1000, 2000, 3000],
    ],
)
def test_invalid_distance_range(config_class, bounds):
    with pytest.raises(ValidationError):
        config_class(seed=42, n_simulations=1, distance_range=bounds)


@pytest.mark.parametrize("config_class", [Level0Config, Level1Config])
def test_default_distance_range(config_class):
    assert config_class(seed=42, n_simulations=1).distance_range == (1750, 2250)
