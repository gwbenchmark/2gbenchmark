"""Tests for the CLI tool."""

import subprocess
import sys


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


def test_cli_generates_dataset(tmp_path):
    """Test that the CLI can generate a basic dataset."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create a minimal config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
seed: 42
n_simulations: 2
duration: 4.0
sampling_frequency: 2048.0
"""
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gwbenchmark2g",
            "--level",
            "0",
            "--config",
            str(config_file),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,  # Set timeout to avoid hanging
    )

    # Check that the command succeeded
    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"

    # Check that output files were created
    metadata_file = output_dir / "injection_metadata.parquet"
    assert metadata_file.exists(), "Metadata file was not created"

    # Check that simulation files were created (should be 2 based on config)
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
