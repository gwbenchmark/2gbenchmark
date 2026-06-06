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
from bilby_pipe.utils import parse_args

sys.path.insert(0, str(Path(__file__).parent))
from npz_generation import NpzDataGenerationInput

DETECTORS = ["H1", "L1", "V1"]
DURATION = 4.0
SAMPLING_FREQUENCY = 2048.0
TRIGGER_TIME = -0.01621880385450652
N_FREQ = int(DURATION * SAMPLING_FREQUENCY / 2) + 1


def write_synthetic_npz(directory, idx):
    """Write simulation_<idx>.npz with random strain/PSD; return the source dict."""
    rng = np.random.default_rng(idx)
    frequency_array = np.linspace(0.0, SAMPLING_FREQUENCY / 2, N_FREQ)
    data = {}
    for det in DETECTORS:
        data[det] = types.SimpleNamespace(
            strain=(rng.normal(size=N_FREQ) + 1j * rng.normal(size=N_FREQ)),
            psd=rng.uniform(1e-46, 1e-44, size=N_FREQ),
            frequency_array=frequency_array,
        )
    np.savez(directory / f"simulation_{idx}.npz", data=data)
    return data


def write_ini(directory):
    ini = directory / "test.ini"
    ini.write_text(
        f"""
trigger-time = {TRIGGER_TIME}
detectors = [H1, L1, V1]
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


def test_strain_and_psd_match_npz(tmp_path):
    source = write_synthetic_npz(tmp_path, 0)
    ini = write_ini(tmp_path)
    args, unknown = parse_args(
        [str(ini), "--idx", "0", "--label", "test", "--outdir", str(tmp_path)],
        create_generation_parser(),
    )
    inputs = NpzDataGenerationInput(args, unknown, npz_directory=tmp_path)

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
