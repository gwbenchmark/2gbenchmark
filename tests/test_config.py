from collections import Counter

from gwbenchmark2g.config import DetectorNetworkConfig


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
