#!/usr/bin/env python3
"""Run the level 0 reference-posterior workflow.

A thin wrapper around ``bilby_pipe`` that swaps the data-generation step for
``npz_generation.py``, so each event's strain is read from its
``simulation_<idx>.npz`` file instead of being simulated. Analysis, plots,
summary and scheduling are all standard bilby_pipe.

Usage mirrors bilby_pipe, with one extra option::

    run_reference_posteriors.py level0.ini --npz-directory /path/to/level0/data

``--npz-directory`` is consumed here (bilby_pipe never sees it) and added to the
generation jobs only.
"""

import argparse
import sys
from pathlib import Path

from bilby_pipe.job_creation.nodes.generation_node import GenerationNode
from bilby_pipe.main import main as bilby_pipe_main

NPZ_GENERATION = str(Path(__file__).resolve().parent / "npz_generation.py")


def use_npz_generation(npz_directory):
    """Run generation jobs via npz_generation.py, passing them the npz directory."""
    GenerationNode.executable = property(lambda self: NPZ_GENERATION)

    base_setup_arguments = GenerationNode.setup_arguments

    def setup_arguments(self, *args, **kwargs):
        base_setup_arguments(self, *args, **kwargs)
        self.arguments.add("npz-directory", npz_directory)

    GenerationNode.setup_arguments = setup_arguments


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--npz-directory", required=True)
    args, remaining = parser.parse_known_args()

    use_npz_generation(args.npz_directory)

    # Hand the remaining arguments (ini + any bilby_pipe options) to bilby_pipe.
    sys.argv = [sys.argv[0]] + remaining
    bilby_pipe_main()


if __name__ == "__main__":
    main()
