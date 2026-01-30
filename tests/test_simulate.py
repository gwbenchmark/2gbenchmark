import tempfile
from pathlib import Path

from gwbenchmark2g import simulate, config
from gwbenchmark2g.io import save_metadata, read_metadata, read_single_metadata


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


def test_level0_fixed_parameters():
    """Test that Level0Config fixed parameters are correctly applied."""
    cfg = config.Level0Config(
        n_simulations=5,
        sampling_frequency=2048,
        duration=4,
        seed=42,
    )

    # Expected fixed parameter values for Level 0
    expected_fixed = {
        "geocent_time": -0.01621880385450652,
        "phase": 0.0,
        "psi": 0.0,
        "theta_jn": 0.0,
        "dec": 2.058804189275143,
        "ra": -1.595801372295631,
    }

    # Verify config has the correct fixed parameters
    assert cfg.fixed_parameters == expected_fixed

    # Verify that simulations actually use these fixed values
    for data, metadata in simulate.simulate_level_0(cfg):
        injection_params = metadata.injection_parameters
        assert injection_params is not None

        # Check each fixed parameter has the expected value
        for param_name, expected_value in expected_fixed.items():
            assert param_name in injection_params
            assert injection_params[param_name] == expected_value, (
                f"{param_name} should be fixed to {expected_value}, got {injection_params[param_name]}"
            )


def test_save_many_simulations_metadata_to_parquet():
    """Test saving metadata from many simulations to parquet format."""
    cfg = config.Level0Config(
        n_simulations=10,
        sampling_frequency=2048,
        duration=4,
        seed=42,
    )

    # Collect metadata from all simulations
    all_metadata = []
    for data, metadata in simulate.simulate_level_0(cfg):
        all_metadata.append(metadata)

    # Verify we got all simulations
    assert len(all_metadata) == 10

    # Save to parquet using the new function
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "metadata.parquet"
        save_metadata(all_metadata, parquet_path)

        # Verify file was created
        assert parquet_path.exists()

        # Test reading all metadata as objects
        read_metadata_list = read_metadata(parquet_path)
        assert len(read_metadata_list) == 10

        # Verify all records match
        for original, read_back in zip(all_metadata, read_metadata_list):
            assert original == read_back

        # Test reading individual rows as objects
        for i in range(10):
            row_metadata = read_single_metadata(parquet_path, i)
            assert row_metadata == all_metadata[i]
