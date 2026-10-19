#!/usr/bin/env python3
"""Write the level 0 recovery prior used for the reference posteriors.

This reproduces the prior that ``gwbenchmark2g.simulate.simulate_level_0`` used
to generate the data, so recovery samples under the same prior the injections
were drawn from: aligned-spin BBH, a UniformSourceFrame distance, and the six
extrinsic parameters fixed to their injected values.
"""

import argparse
from pathlib import Path

import bilby
from gwbenchmark2g.config import Level0Config

# Luminosity distance bounds, matching simulate_level_0 (the values the data
# were generated with). Keep in sync with that function if it ever changes.
DISTANCE_MINIMUM = 1750.0
DISTANCE_MAXIMUM = 2250.0


def build_recovery_prior() -> bilby.gw.prior.BBHPriorDict:
    prior = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
    prior["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        name="luminosity_distance",
        minimum=DISTANCE_MINIMUM,
        maximum=DISTANCE_MAXIMUM,
    )
    for key, value in Level0Config.model_fields["fixed_parameters"].default.items():
        prior[key] = value
    return prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to write level0_recovery.prior into.",
    )
    args = parser.parse_args()
    build_recovery_prior().to_file(outdir=str(args.outdir), label="level0_recovery")


if __name__ == "__main__":
    main()
