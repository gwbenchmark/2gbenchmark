"""Unit test: NpzDataGenerationInput loads strain and PSD from the npz files.

Runs only where bilby_pipe is installed (the workflow venv); skipped otherwise.
A small synthetic npz stands in for the real data, so the test needs neither the
published dataset nor gwbenchmark2g.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("bilby_pipe")
from bilby_pipe.data_generation import create_generation_parser
from bilby_pipe.utils import BilbyPipeError, parse_args

sys.path.insert(0, str(Path(__file__).parent))
from npz_generation import NpzDataGenerationInput

DETECTORS = ["H1", "L1", "V1"]
DURATION = 4.0
SAMPLING_FREQUENCY = 2048.0
TRIGGER_TIME = -0.01621880385450652
N_FREQ = int(DURATION * SAMPLING_FREQUENCY / 2) + 1


def write_synthetic_npz(directory, idx, detectors=DETECTORS):
    """Write simulation_<idx>.npz with random strain/PSD; return the source dict."""
    rng = np.random.default_rng(idx)
    frequency_array = np.linspace(0.0, SAMPLING_FREQUENCY / 2, N_FREQ)
    data = {}
    for det in detectors:
        data[det] = types.SimpleNamespace(
            strain=(rng.normal(size=N_FREQ) + 1j * rng.normal(size=N_FREQ)),
            psd=rng.uniform(1e-46, 1e-44, size=N_FREQ),
            frequency_array=frequency_array,
        )
    np.savez(directory / f"simulation_{idx}.npz", data=data)
    return data


def write_ini(directory, detectors=DETECTORS):
    ini = directory / "test.ini"
    ini.write_text(
        f"""
trigger-time = {TRIGGER_TIME}
detectors = [{", ".join(detectors)}]
duration = {DURATION}
sampling-frequency = {SAMPLING_FREQUENCY}
minimum-frequency = 20
maximum-frequency = 1024
gaussian-noise = True
injection = False
n-simulation = 1
prior-file = {Path(__file__).parent / "level0_recovery.prior"}
frequency-domain-source-model = lal_binary_black_hole
waveform-approximant = IMRPhenomXAS
reference-frequency = 50
"""
    )
    return ini


def load(tmp_path, ini):
    args, unknown = parse_args(
        [str(ini), "--idx", "0", "--label", "test", "--outdir", str(tmp_path)],
        create_generation_parser(),
    )
    return NpzDataGenerationInput(args, unknown, npz_directory=tmp_path)


def assert_strain_and_psd_match(inputs, source):
    for ifo in inputs.interferometers:
        # frequency_domain_strain zeros out-of-band bins, so compare in-band
        # (what the likelihood actually uses).
        mask = ifo.frequency_mask
        np.testing.assert_array_equal(
            ifo.frequency_domain_strain[mask], source[ifo.name].strain[mask]
        )
        np.testing.assert_array_equal(
            ifo.power_spectral_density.psd_array, source[ifo.name].psd
        )


def test_strain_and_psd_match_npz(tmp_path):
    source = write_synthetic_npz(tmp_path, 0)
    inputs = load(tmp_path, write_ini(tmp_path))
    assert [ifo.name for ifo in inputs.interferometers] == DETECTORS
    assert_strain_and_psd_match(inputs, source)


@pytest.mark.parametrize("network", [["H1", "L1"], ["H1"]])
def test_subset_network_loads_only_event_detectors(tmp_path, network):
    # Level 1: the ini lists all detectors, each event holds its own network.
    source = write_synthetic_npz(tmp_path, 0, detectors=network)
    inputs = load(tmp_path, write_ini(tmp_path))
    assert [ifo.name for ifo in inputs.interferometers] == network
    assert_strain_and_psd_match(inputs, source)


def test_npz_detector_missing_from_ini_raises(tmp_path):
    write_synthetic_npz(tmp_path, 0)
    with pytest.raises(BilbyPipeError, match="V1"):
        load(tmp_path, write_ini(tmp_path, detectors=["H1", "L1"]))


@pytest.mark.parametrize(
    "parameters", [None, [], {}, "missing", [("chirp_mass", 31.0)]]
)
def test_optional_injection_truth(tmp_path, monkeypatch, parameters):
    from gwbenchmark2g import io

    (tmp_path / "injection_metadata.parquet").touch()
    calls = []

    def read_metadata(path, idx):
        calls.append((path, idx))
        return {} if parameters == "missing" else {"injection_parameters": parameters}

    monkeypatch.setattr(io, "read_single_metadata_raw", read_metadata)
    inputs = NpzDataGenerationInput.__new__(NpzDataGenerationInput)
    inputs.npz_directory = tmp_path
    inputs.idx = 3
    inputs.calibration_model = None
    inputs.meta_data = {"injection_parameters": None}
    inputs.injection_parameters = None
    inputs._load_injection_parameters()

    expected = dict(parameters) if parameters and parameters != "missing" else None
    assert inputs.injection_parameters == expected
    assert inputs.meta_data["injection_parameters"] == expected
    assert calls == [(tmp_path / "injection_metadata.parquet", 3)]


def test_missing_metadata_file_leaves_truth_unset(tmp_path):
    inputs = NpzDataGenerationInput.__new__(NpzDataGenerationInput)
    inputs.npz_directory = tmp_path
    inputs.calibration_model = None
    inputs.meta_data = {"injection_parameters": None}
    inputs.injection_parameters = None
    inputs._load_injection_parameters()
    assert inputs.injection_parameters is None
    assert inputs.meta_data["injection_parameters"] is None


def write_metadata(directory, trigger_time, detectors):
    from gwbenchmark2g.io import save_metadata
    from gwbenchmark2g.simulate import InjectionMetaData

    save_metadata(
        [
            InjectionMetaData(
                injection_parameters=None,
                fixed_parameters=None,
                waveform_kwargs={},
                seed=None,
                detectors={det: {"minimum_frequency": 20.0} for det in detectors},
                duration=DURATION,
                sampling_frequency=SAMPLING_FREQUENCY,
                trigger_time=trigger_time,
                post_trigger_duration=2.0,
            )
        ],
        directory / "injection_metadata.parquet",
    )


@pytest.mark.parametrize("trigger_time", [TRIGGER_TIME, None])
def test_metadata_trigger_time_and_detectors_agree(tmp_path, trigger_time):
    write_synthetic_npz(tmp_path, 0, detectors=["H1", "L1"])
    write_metadata(tmp_path, trigger_time, ["H1", "L1"])
    inputs = load(tmp_path, write_ini(tmp_path))
    assert [ifo.name for ifo in inputs.interferometers] == ["H1", "L1"]


def test_trigger_time_differing_from_metadata_raises(tmp_path):
    write_synthetic_npz(tmp_path, 0)
    write_metadata(tmp_path, TRIGGER_TIME + 1.0, DETECTORS)
    with pytest.raises(BilbyPipeError, match="does not match the metadata"):
        load(tmp_path, write_ini(tmp_path))


def test_npz_detectors_differing_from_metadata_raise(tmp_path):
    write_synthetic_npz(tmp_path, 0, detectors=["H1", "L1"])
    write_metadata(tmp_path, TRIGGER_TIME, ["H1"])
    with pytest.raises(BilbyPipeError, match="metadata lists detectors"):
        load(tmp_path, write_ini(tmp_path))
