"""Unit test: the workflow gets each event's trigger time and detector network.

Builds real bilby_pipe workflows (Slurm job scripts, nothing is submitted) from
a synthetic three-event dataset, and checks the generated job commands. Runs
only where bilby_pipe and gwbenchmark2g are installed; skipped otherwise.
"""

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("bilby_pipe")
pytest.importorskip("gwbenchmark2g")
from bilby_pipe.utils import BilbyPipeError
from gwbenchmark2g.io import save_metadata
from gwbenchmark2g.simulate import InjectionMetaData

sys.path.insert(0, str(Path(__file__).parent))
from run_reference_posteriors import (
    EventMetadata,
    choose_network,
    choose_trigger_time,
    event_metadata,
    read_event_metadata,
)

HERE = Path(__file__).parent

NETWORKS = [["H1", "L1", "V1"], ["H1", "L1"], ["H1"]]
TRIGGER_TIMES = [24008.161227123456, 56687.945152987654, 75317.337191111111]
LEVEL0_TRIGGER_TIME = -0.01621880385450652


def write_metadata(directory, trigger_times=TRIGGER_TIMES, networks=NETWORKS):
    directory.mkdir(exist_ok=True)
    save_metadata(
        [
            InjectionMetaData(
                injection_parameters=None,
                fixed_parameters=None,
                waveform_kwargs={},
                seed=None,
                detectors={det: {"minimum_frequency": 20.0} for det in network},
                duration=8.0,
                sampling_frequency=2048.0,
                trigger_time=trigger_time,
                post_trigger_duration=2.0,
            )
            for trigger_time, network in zip(trigger_times, networks)
        ],
        directory / "injection_metadata.parquet",
    )
    return directory


def write_ini(directory, n_simulation=3, trigger_time=None):
    ini = directory / "test.ini"
    ini.write_text(
        f"""
label = test
outdir = {directory / "out"}
scheduler = slurm
gaussian-noise = True
injection = False
n-simulation = {n_simulation}
{"" if trigger_time is None else f"trigger-time = {trigger_time}"}
detectors = [H1, L1, V1]
duration = 8
sampling-frequency = 2048
minimum-frequency = 20
maximum-frequency = 1024
deltaT = 0.2
post-trigger-duration = 2
prior-file = {HERE / "level1_recovery.prior"}
frequency-domain-source-model = lal_binary_black_hole
waveform-approximant = IMRPhenomXHM
reference-frequency = 50
sampler = dynesty
"""
    )
    return ini


def build_workflow(tmp_path, ini, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(HERE / "run_reference_posteriors.py"),
            str(ini),
            "--npz-directory",
            str(tmp_path / "data"),
            *extra,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def jobs(tmp_path, kind):
    """{idx: (job name, command)} for the generation or analysis jobs."""
    found = {}
    for script in (tmp_path / "out" / "submit").glob(f"*_{kind}*.sh"):
        if script.stem.endswith("final_result"):
            continue
        idx = int(re.search(r"_data(\d+)_", script.stem).group(1))
        found[idx] = (script.stem, script.read_text())
    return found


def option(command, name):
    return re.findall(rf"--{name} (\S+)", command)


def test_per_event_trigger_times_and_networks(tmp_path):
    write_metadata(tmp_path / "data")
    result = build_workflow(tmp_path, write_ini(tmp_path))
    assert result.returncode == 0, result.stderr

    generation = jobs(tmp_path, "generation")
    assert sorted(generation) == [0, 1, 2]
    for idx, (name, command) in generation.items():
        (trigger_time,) = option(command, "trigger-time")
        assert float(trigger_time) == TRIGGER_TIMES[idx]
        assert trigger_time.replace(".", "-") in name

    analysis = jobs(tmp_path, "analysis")
    for idx, (name, command) in analysis.items():
        network = NETWORKS[idx]
        assert option(command, "detectors") == network
        assert name.endswith(f"_analysis_{''.join(network)}")
        assert name in option(command, "label")[0]


def test_start_index_selects_metadata_rows(tmp_path):
    write_metadata(tmp_path / "data")
    result = build_workflow(
        tmp_path, write_ini(tmp_path, n_simulation=2), "--start-index", "1"
    )
    assert result.returncode == 0, result.stderr

    generation = jobs(tmp_path, "generation")
    assert sorted(generation) == [1, 2]
    for idx, (_, command) in generation.items():
        assert option(command, "idx") == [str(idx)]
        assert float(option(command, "trigger-time")[0]) == TRIGGER_TIMES[idx]
    analysis = jobs(tmp_path, "analysis")
    assert {idx: option(c, "detectors") for idx, (_, c) in analysis.items()} == {
        1: NETWORKS[1],
        2: NETWORKS[2],
    }


def test_without_metadata_uses_ini_trigger_time_and_detectors(tmp_path):
    # Level 0 without a metadata file: unchanged bilby_pipe behaviour.
    (tmp_path / "data").mkdir()
    ini = write_ini(tmp_path, trigger_time=LEVEL0_TRIGGER_TIME)
    result = build_workflow(tmp_path, ini)
    assert result.returncode == 0, result.stderr

    for _, command in jobs(tmp_path, "generation").values():
        assert float(option(command, "trigger-time")[0]) == LEVEL0_TRIGGER_TIME
    for name, command in jobs(tmp_path, "analysis").values():
        assert option(command, "detectors") == ["H1", "L1", "V1"]
        assert name.endswith("_analysis_H1L1V1")


def test_ini_trigger_time_disagreeing_with_metadata_fails(tmp_path):
    write_metadata(tmp_path / "data")
    result = build_workflow(tmp_path, write_ini(tmp_path, trigger_time=0.0))
    assert result.returncode != 0
    assert "Remove trigger-time from the ini" in result.stderr


def test_read_event_metadata(tmp_path):
    write_metadata(tmp_path)
    assert read_event_metadata(tmp_path) == [
        EventMetadata(t, n) for t, n in zip(TRIGGER_TIMES, NETWORKS)
    ]
    assert read_event_metadata(tmp_path / "missing") is None


def test_read_event_metadata_without_trigger_times(tmp_path):
    # Data generated before trigger_time was added to the metadata.
    write_metadata(tmp_path, trigger_times=[None] * 3)
    assert [event.trigger_time for event in read_event_metadata(tmp_path)] == [None] * 3


def test_event_metadata_missing_row_raises():
    events = [EventMetadata(1.0, ["H1"])]
    assert event_metadata(None, 5) == EventMetadata(None, None)
    with pytest.raises(BilbyPipeError, match="no metadata for event 1"):
        event_metadata(events, 1)


def test_choose_trigger_time():
    assert choose_trigger_time(0, None, 12.5) == 12.5
    assert choose_trigger_time(0, 3.0, None) == 3.0
    assert choose_trigger_time(0, LEVEL0_TRIGGER_TIME, LEVEL0_TRIGGER_TIME) == (
        LEVEL0_TRIGGER_TIME
    )
    with pytest.raises(BilbyPipeError, match="No trigger time for event 4"):
        choose_trigger_time(4, None, None)
    with pytest.raises(BilbyPipeError, match="differs from the metadata"):
        choose_trigger_time(0, 3.0, 12.5)


def inputs(coherence_test=False):
    return SimpleNamespace(detectors=["H1", "L1", "V1"], coherence_test=coherence_test)


def test_choose_network():
    full = ["H1", "L1", "V1"]
    assert choose_network(0, inputs(), full, None) == full
    assert choose_network(0, inputs(), full, ["V1", "H1", "L1"]) == full
    assert choose_network(0, inputs(), full, ["L1", "H1"]) == ["H1", "L1"]
    assert choose_network(0, inputs(), full, ["H1"]) == ["H1"]
    with pytest.raises(BilbyPipeError, match="K1"):
        choose_network(0, inputs(), full, ["H1", "K1"])
    with pytest.raises(BilbyPipeError, match="coherence-test"):
        choose_network(0, inputs(coherence_test=True), full, ["H1"])
