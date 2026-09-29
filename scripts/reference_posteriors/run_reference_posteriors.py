#!/usr/bin/env python3
"""Run the level 0 reference-posterior workflow.

A thin wrapper around ``bilby_pipe`` that swaps the data-generation step for
``npz_generation.py``, so each event's strain is read from its
``simulation_<idx>.npz`` file instead of being simulated. Analysis, plots,
summary and scheduling are all standard bilby_pipe.

Usage mirrors bilby_pipe, with one extra required option::

    run_reference_posteriors.py level0.ini --npz-directory /path/to/level0/data

``--npz-directory`` is consumed here (bilby_pipe never sees it) and added to the
generation jobs only.

To run every job inside an Apptainer/Singularity image, add
``--container /path/to/image.sif`` (and ``--bind PATH`` for any extra
directories the jobs need to see). On HTCondor this uses bilby_pipe's own
``container`` support; on Slurm each job script is wrapped in
``apptainer exec``.
"""

import argparse
import os
import shlex
import sys
from pathlib import Path

from bilby_pipe.job_creation.dag import Dag
from bilby_pipe.job_creation.nodes.generation_node import GenerationNode
from bilby_pipe.job_creation.slurm import SubmitSLURM
from bilby_pipe.main import main as bilby_pipe_main
from bilby_pipe.utils import BilbyPipeError, logger

NPZ_GENERATION = str(Path(__file__).resolve().parent / "npz_generation.py")


def use_npz_generation(npz_directory):
    """Run generation jobs via npz_generation.py, passing them the npz directory."""
    base_setup_arguments = GenerationNode.setup_arguments

    def executable(self):
        # Swap the generation executable to python so we can run
        # npz_generation.py instead of bilby_pipe's default generation script.
        # This could be changed if this script was made an executable itself.
        return self._get_executable_path("python")

    def setup_arguments(self, *args, **kwargs):
        base_setup_arguments(self, *args, **kwargs)
        self.arguments.argument_list.insert(0, NPZ_GENERATION)
        self.arguments.add("npz-directory", npz_directory)

    GenerationNode.executable = property(executable)
    GenerationNode.setup_arguments = setup_arguments


def use_slurm_container(container, binds):
    """Wrap every Slurm job script's command in ``apptainer exec``.

    bilby_pipe only supports containers on HTCondor, so for Slurm we prefix the
    command at the point each job script is written. The run directory (parent
    of the submit directory) is always bound so the jobs can write results.
    """
    base_write_individual_processes = SubmitSLURM._write_individual_processes

    def _write_individual_processes(self, name, executable, args):
        if self.scheduler_env is not None:
            raise BilbyPipeError(
                "--container cannot be combined with scheduler-env: the "
                "environment comes from the container."
            )
        outdir = os.path.dirname(os.path.abspath(self.submit_dir))
        bind = ",".join(dict.fromkeys(binds + [outdir]))
        prefix = f"apptainer exec --bind {shlex.quote(bind)} {shlex.quote(container)}"
        return base_write_individual_processes(
            self, name, f"{prefix} {executable}", args
        )

    SubmitSLURM._write_individual_processes = _write_individual_processes


def warn_condor_container(folders):
    """Warn that HTCondor container jobs need ``folders`` mounted by the cluster.

    On HTCondor we can't bind folders ourselves: the cluster decides what is
    visible inside a container. If these folders aren't mounted, generation
    jobs only fail once they run, so say so while the workflow is being built.
    """
    base_build_pycondor_dag = Dag.build_pycondor_dag

    def build_pycondor_dag(self):
        logger.warning(
            "Running HTCondor jobs in a container: the cluster must make these "
            "folders visible inside the container, or the jobs will fail "
            "(--bind has no effect on HTCondor):\n  " + "\n  ".join(folders)
        )
        base_build_pycondor_dag(self)

    Dag.build_pycondor_dag = build_pycondor_dag


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--npz-directory", required=True)
    parser.add_argument("--container", default=None)
    parser.add_argument("--bind", action="append", default=[])
    args, remaining = parser.parse_known_args()

    use_npz_generation(args.npz_directory)

    if args.container is not None:
        container = args.container
        if os.path.exists(container):
            # A local image file; leave URIs such as docker:// untouched.
            container = os.path.abspath(container)
        # The jobs read the npz files and run npz_generation.py from this
        # directory, so both must be visible inside the container.
        binds = [
            os.path.abspath(args.npz_directory),
            os.path.dirname(NPZ_GENERATION),
        ] + [os.path.abspath(path) for path in args.bind]
        use_slurm_container(container, binds)
        warn_condor_container(binds)
        # bilby_pipe's own option: HTCondor jobs are submitted to run in the
        # container, and executables are referred to by name so they resolve
        # in the container.
        remaining += ["--container", container]

    # Hand the remaining arguments (ini + any bilby_pipe options) to bilby_pipe.
    sys.argv = [sys.argv[0]] + remaining
    bilby_pipe_main()


if __name__ == "__main__":
    main()
