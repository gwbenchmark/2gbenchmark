#!/usr/bin/env python3
"""bilby_pipe data generation that reads strain from the level 0 npz files.

The published level 0 dataset stores each event as ``simulation_<idx>.npz``
holding the exact frequency-domain strain and PSD per detector. Rather than
simulate noise, this input loads that strain straight into the interferometers
so the rest of the bilby_pipe workflow (analysis, plots, summary) runs on the
real benchmark data.

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


class NpzDataGenerationInput(DataGenerationInput):
    """DataGenerationInput that takes its strain from the level 0 npz files.

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

    def _set_interferometers_from_gaussian_noise(self):
        npz_file = self.npz_directory / f"simulation_{self.idx}.npz"
        data = np.load(npz_file, allow_pickle=True)["data"].item()

        ifos = InterferometerList(self.detectors)
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
