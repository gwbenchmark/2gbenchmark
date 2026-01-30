"""Command-line interface for 2GBenchmark."""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .io import INJECTION_METADATA_SCHEMA


def get_parser():
    """Get the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="2GBenchmark: Generating benchmark datasets for 2G GW detectors"
    )
    # Additional arguments can be added here
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the configuration file for the benchmark dataset.",
    )
    parser.add_argument(
        "--level",
        type=int,
        help="",
    )
    parser.add_argument(
        "--filename",
        type=Path,
        help="Output filename for the generated dataset.",
    )
    return parser


def main():
    from .config import level_registry
    from .simulate import simulate_registry

    parser = get_parser()
    args = parser.parse_args()

    LevelConfigClass = level_registry[args.level]

    with open(args.config) as f:
        config_data = yaml.safe_load(f)

    config = LevelConfigClass.model_validate(config_data)

    simulate_function = simulate_registry.get(args.level)

    if simulate_function is None:
        raise ValueError(f"No simulation function found for level {args.level}")

    all_metadata = list()

    # Save data to .npy file and metadata to a parquet file
    for data, metadata in simulate_function(config):
        all_metadata.append(metadata)
        # Save the data

    table = pa.Table.from_pylist([all_metadata], schema=INJECTION_METADATA_SCHEMA)
    pq.write_table(table, "injection_metadata.parquet")
