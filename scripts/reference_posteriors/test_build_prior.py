"""Unit test: build_prior.py reproduces the generation prior for each level.

Runs only where bilby and gwbenchmark2g are installed; skipped otherwise.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("bilby")
pytest.importorskip("gwbenchmark2g")
import bilby
from gwbenchmark2g.config import Level0Config, Level1Config

sys.path.insert(0, str(Path(__file__).parent))
from build_prior import build_recovery_prior, default_level0_config, main

HERE = Path(__file__).parent
DISTANCE_RANGE = (500.0, 1000.0)


def level1_config(**overrides):
    return Level1Config(
        seed=0, n_simulations=0, distance_range=DISTANCE_RANGE, **overrides
    )


def write_prior(prior, directory, label):
    prior.to_file(outdir=str(directory), label=label)
    return (directory / f"{label}.prior").read_text()


def test_default_matches_committed_level0_prior(tmp_path):
    written = write_prior(build_recovery_prior(default_level0_config()), tmp_path, "l0")
    assert written == (HERE / "level0_recovery.prior").read_text()


def test_level0_yaml_without_optional_fields_matches_default(tmp_path):
    # Like configs/gwbenchmark2g/level0.yaml: no fixed_parameters or distance_range.
    config = Level0Config(
        duration=8.0,
        sampling_frequency=2048,
        waveform_approximant="IMRPhenomXAS",
        seed=42,
        n_simulations=10,
        detectors=["H1", "L1", "V1"],
    )
    written = write_prior(build_recovery_prior(config), tmp_path, "l0")
    assert written == (HERE / "level0_recovery.prior").read_text()


def test_level1_prior():
    prior = build_recovery_prior(level1_config())
    distance = prior["luminosity_distance"]
    assert isinstance(distance, bilby.gw.prior.UniformSourceFrame)
    assert (distance.minimum, distance.maximum) == DISTANCE_RANGE
    # bilby_pipe adds the per-event geocent_time prior from the trigger time.
    assert "geocent_time" not in prior
    # No fixed parameters: every entry is a distribution, not a fixed value.
    assert all(isinstance(value, bilby.core.prior.Prior) for value in prior.values())
    assert not any(
        isinstance(value, bilby.core.prior.DeltaFunction) for value in prior.values()
    )

    expected = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
    assert set(prior) == set(expected)
    for key in expected:
        if key != "luminosity_distance":
            assert repr(prior[key]) == repr(expected[key]), key


def test_level1_fixed_parameters_are_fixed(tmp_path):
    prior = build_recovery_prior(level1_config(fixed_parameters={"psi": 0.5}))
    assert prior["psi"] == 0.5
    write_prior(prior, tmp_path, "fixed")
    reloaded = bilby.gw.prior.BBHPriorDict(filename=str(tmp_path / "fixed.prior"))
    assert reloaded.fixed_keys == ["psi"]


def test_level1_matches_committed_level1_prior(tmp_path):
    # The committed file was built from configs/gwbenchmark2g/level1.yaml.
    written = write_prior(build_recovery_prior(level1_config()), tmp_path, "l1")
    assert written == (HERE / "level1_recovery.prior").read_text()


def test_cli_level1(tmp_path):
    config = tmp_path / "level1.yaml"
    config.write_text("seed: 1\nn_simulations: 2\ndistance_range: [500.0, 1000.0]\n")
    subprocess.run(
        [
            sys.executable,
            str(HERE / "build_prior.py"),
            "--dataset-config",
            str(config),
            "--level",
            "1",
            "--outdir",
            str(tmp_path),
        ],
        check=True,
    )
    assert (tmp_path / "level1_recovery.prior").read_text() == (
        HERE / "level1_recovery.prior"
    ).read_text()


def test_cli_dataset_config_needs_level(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_prior.py", "--dataset-config", "x.yaml", "--outdir", str(tmp_path)],
    )
    with pytest.raises(SystemExit):
        main()
    assert not list(tmp_path.iterdir())
