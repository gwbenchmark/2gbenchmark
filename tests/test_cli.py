"""Tests for the CLI tool."""

import subprocess
import sys

import pytest


def test_cli_help():
    """Test that the CLI help command works."""
    result = subprocess.run(
        [sys.executable, "-m", "gwbenchmark2g", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "2GBenchmark" in result.stdout
    assert "--config" in result.stdout
    assert "--level" in result.stdout
    assert "--output-dir" in result.stdout


def test_cli_entry_point():
    """Test that the gwbenchmark2g entry point exists and works."""
    result = subprocess.run(
        ["gwbenchmark2g", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "2GBenchmark" in result.stdout


@pytest.mark.parametrize(
    ("level", "output_name", "config_name", "config_text"),
    [
        (
            0,
            "output",
            "config.yaml",
            """
seed: 42
n_simulations: 2
duration: 4.0
sampling_frequency: 2048.0
""",
        ),
        (
            1,
            "output_level1",
            "level1_config.yaml",
            """
seed: 42
n_simulations: 2
duration: 4.0
sampling_frequency: 2048.0
detectors:
  detector_combinations:
    - [H1, L1, V1]
    - [H1, L1]
    - [H1]
  weights: [1.0, 1.0, 1.0]
""",
        ),
    ],
)
def test_cli_generates_dataset(tmp_path, level, output_name, config_name, config_text):
    """Test that the CLI can generate datasets for supported levels."""
    output_dir = tmp_path / output_name
    output_dir.mkdir()

    config_file = tmp_path / config_name
    config_file.write_text(config_text)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gwbenchmark2g",
            "--level",
            str(level),
            "--config",
            str(config_file),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
    assert (output_dir / "injection_metadata.parquet").exists()
    sim_files = list(output_dir.glob("simulation_*.npz"))
    assert len(sim_files) == 2, f"Expected 2 simulation files, got {len(sim_files)}"


def test_cli_missing_config():
    """Test that the CLI fails gracefully when config is missing."""
    result = subprocess.run(
        [sys.executable, "-m", "gwbenchmark2g", "--level", "0"],
        capture_output=True,
        text=True,
    )
    # Should fail with non-zero exit code
    assert result.returncode != 0
