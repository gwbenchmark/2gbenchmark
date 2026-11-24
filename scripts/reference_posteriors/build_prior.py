#!/usr/bin/env python3
"""Write the recovery prior used for the reference posteriors.

This reproduces the prior that ``gwbenchmark2g.simulate`` used to generate the
data, so recovery samples under the same prior the injections were drawn from:
aligned-spin BBH, a UniformSourceFrame distance over the dataset's
``distance_range``, and the dataset's ``fixed_parameters`` fixed to their
injected values.

Level 0 (the default) fixes six extrinsic parameters, including
``geocent_time``. For level 1, pass the dataset config from the
``gwbenchmark/configs`` repo (or the ``config.yaml`` saved with the dataset):

    python build_prior.py --dataset-config path/to/level1.yaml --level 1

Level 1 events each have their own trigger time, so ``geocent_time`` is left
out of the file: bilby_pipe then adds ``Uniform(trigger_time - deltaT/2,
trigger_time + deltaT/2)`` for each event, which with ``deltaT = 0.2`` is the
generation prior ``trigger_time + Uniform(-0.1, 0.1)``.
"""

import argparse
from pathlib import Path

import bilby
import yaml
from gwbenchmark2g.config import DatasetConfig, Level0Config, level_registry


def build_recovery_prior(config: DatasetConfig) -> bilby.gw.prior.BBHPriorDict:
    prior = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
    prior["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        name="luminosity_distance",
        minimum=config.distance_range[0],
        maximum=config.distance_range[1],
    )
    for key, value in (config.fixed_parameters or {}).items():
        prior[key] = value
    return prior


def load_dataset_config(path: Path, level: int) -> DatasetConfig:
    with open(path) as f:
        return level_registry[level](**yaml.safe_load(f))


def default_level0_config() -> Level0Config:
    # seed and n_simulations are required but don't affect the prior.
    return Level0Config(seed=0, n_simulations=0)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="Dataset yaml the data were generated with. Defaults to the level 0 defaults.",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=sorted(level_registry),
        default=None,
        help="Level of --dataset-config (required with it).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Prior file name without .prior. Defaults to level<N>_recovery.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to write the prior file into.",
    )
    args = parser.parse_args()

    if args.dataset_config is None:
        if args.level not in (None, 0):
            parser.error("--level other than 0 needs --dataset-config.")
        config = default_level0_config()
    else:
        if args.level is None:
            parser.error("--dataset-config needs --level.")
        config = load_dataset_config(args.dataset_config, args.level)

    label = args.label or f"level{config.level}_recovery"
    build_recovery_prior(config).to_file(outdir=str(args.outdir), label=label)


if __name__ == "__main__":
    main()
