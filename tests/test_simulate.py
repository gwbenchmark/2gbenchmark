from gwbenchmark2g import simulate, config


def test_simulate_level0():
    cfg = config.Level0Config(
        n_simulations=5,
        sampling_frequency=2048,
        duration=4,
        seed=10,
    )
    for data, metadata in simulate.simulate_level_0(cfg):
        continue


def test_simulate_level0_doesnt_contain_truth_with_blinding():
    cfg = config.Level0Config(
        n_simulations=5,
        sampling_frequency=2048,
        duration=4,
        seed=10,
        blind=True,
    )
    for data, metadata in simulate.simulate_level_0(cfg):
        assert metadata.injection_parameters is None
