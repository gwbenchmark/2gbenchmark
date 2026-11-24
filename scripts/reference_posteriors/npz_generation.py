#!/usr/bin/env python3
"""bilby_pipe data generation that reads strain from the benchmark npz files.

The published datasets store each event as ``simulation_<idx>.npz`` holding
the exact frequency-domain strain and PSD per detector. Rather than simulate
noise, this input loads that strain straight into the interferometers so the
rest of the bilby_pipe workflow (analysis, plots, summary) runs on the real
benchmark data.

An event can hold a subset of the ini detectors (level 1 events are H1L1V1,
H1L1 or H1); only the detectors present in its npz file are loaded.

bilby_pipe runs the generation step for each event as a separate command, so
``main`` here is the executable the workflow calls in place of
``bilby_pipe_generation``. It takes one extra option, ``--npz-directory``.
"""

import sys
from pathlib import Path

import numpy as np
from bilby.gw.detector import InterferometerList, PowerSpectralDensity
from bilby_pipe.data_generation import (
    DataGenerationInput,
    create_generation_parser,
    log_version_information,
    logger,
    parse_args,
)
from bilby_pipe.utils import BilbyPipeError

# How closely the job's trigger time must match the metadata trigger_time (s).
# Both are the same float64 written out exactly, so any difference is an error.
TRIGGER_TIME_TOLERANCE = 1e-9


def metadata_detector_names(detectors):
    """Detector names from a metadata ``detectors`` entry, or None if absent.

    Arrow maps may be returned as lists of (key, value) pairs rather than
    dictionaries.
    """
    if not detectors:
        return None
    if isinstance(detectors, dict):
        return list(detectors)
    return [name for name, _ in detectors]


class NpzDataGenerationInput(DataGenerationInput):
    """DataGenerationInput that takes its strain from the benchmark npz files.

    Run with ``gaussian-noise=True`` and ``injection=False`` so bilby_pipe
    dispatches to ``_set_interferometers_from_gaussian_noise``, which we
    override to load ``simulation_<idx>.npz`` instead of simulating noise.

    Parameters
    ----------
    npz_directory : str | Path
        Directory holding the ``simulation_<idx>.npz`` files.
    """

    def __init__(self, args, unknown_args, npz_directory, create_data=True):
        # Set before super().__init__: the parent constructor immediately runs
        # data generation, which calls our override and needs this path.
        self.npz_directory = Path(npz_directory)
        super().__init__(args, unknown_args, create_data=create_data)
        self._load_injection_parameters()

    def _load_injection_parameters(self):
        """Check the event against its metadata and attach available truth.

        Truth is optional, so blinded datasets still run. The trigger time and
        detectors are checked whenever the metadata has them (blind included).
        """
        metadata_file = self.npz_directory / "injection_metadata.parquet"
        if not metadata_file.is_file():
            logger.info(
                "No injection metadata file; leaving injection parameters unset"
            )
            return

        from gwbenchmark2g.io import read_single_metadata_raw

        metadata = read_single_metadata_raw(metadata_file, self.idx)
        self._check_against_metadata(metadata)
        parameters = metadata.get("injection_parameters")
        if not parameters:
            logger.info("No injection truth for event %s; leaving it unset", self.idx)
            return

        # The setter also updates meta_data, which is saved in the data dump
        # and passed to bilby's sampler by the analysis job. Arrow maps may
        # be returned as lists of pairs rather than dictionaries.
        self.injection_parameters = dict(parameters)

    def _check_against_metadata(self, metadata):
        """Raise if the job's trigger time or the npz detectors disagree with it."""
        trigger_time = metadata.get("trigger_time")
        if (
            trigger_time is not None
            and abs(self.trigger_time - trigger_time) > TRIGGER_TIME_TOLERANCE
        ):
            raise BilbyPipeError(
                f"Event {self.idx}: --trigger-time {self.trigger_time!r} does not "
                f"match the metadata trigger_time {trigger_time!r}."
            )
        names = metadata_detector_names(metadata.get("detectors"))
        if names is None:
            return
        loaded = [ifo.name for ifo in self.interferometers]
        if sorted(names) != sorted(loaded):
            raise BilbyPipeError(
                f"Event {self.idx}: the npz file has data for {loaded}, but the "
                f"metadata lists detectors {names}."
            )

    def _set_interferometers_from_gaussian_noise(self):
        npz_file = self.npz_directory / f"simulation_{self.idx}.npz"
        data = np.load(npz_file, allow_pickle=True)["data"].item()

        unexpected = sorted(set(data) - set(self.detectors))
        if unexpected:
            raise BilbyPipeError(
                f"{npz_file.name} has data for {unexpected}, which are not in the "
                f"ini detectors {self.detectors}; they would be silently left out "
                "of the analysis."
            )
        # The event's own network, in ini order (e.g. [H1, L1] for an H1L1 event).
        event_detectors = [det for det in self.detectors if det in data]
        if not event_detectors:
            raise BilbyPipeError(
                f"{npz_file.name} has no data for any of {self.detectors}."
            )
        logger.info(
            "Event %s: loading data for detectors %s", self.idx, event_detectors
        )

        ifos = InterferometerList(event_detectors)
        for ifo in ifos:
            event = data[ifo.name]
            ifo.set_strain_data_from_frequency_domain_strain(
                frequency_domain_strain=np.asarray(event.strain),
                sampling_frequency=self.sampling_frequency,
                duration=self.duration,
                start_time=self.start_time,
            )
            ifo.power_spectral_density = PowerSpectralDensity(
                frequency_array=np.asarray(event.frequency_array),
                psd_array=np.asarray(event.psd),
            )

        self.interferometers = ifos


def create_npz_generation_parser():
    """The bilby_pipe generation parser plus our --npz-directory option."""
    parser = create_generation_parser()
    parser.add_argument(
        "--npz-directory",
        type=str,
        required=True,
        help="Directory holding the simulation_<idx>.npz files.",
    )
    return parser


def main():
    args, unknown_args = parse_args(sys.argv[1:], create_npz_generation_parser())
    log_version_information()
    data = NpzDataGenerationInput(args, unknown_args, npz_directory=args.npz_directory)
    data.save_data_dump()
    logger.info("Completed npz data generation")


if __name__ == "__main__":
    main()
