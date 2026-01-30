import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import asdict
from pathlib import Path

from .simulate import InjectionMetaData


INJECTION_METADATA_SCHEMA = pa.schema(
    [
        # dict[str, float] | None
        pa.field(
            "injection_parameters",
            pa.map_(pa.string(), pa.float64()),
            nullable=True,  # allow None for blinding
        ),
        # dict[str, int | float | str]
        # Arrow cannot store heterogeneous map values directly,
        # so we split them into separate typed maps.
        pa.field(
            "waveform_kwargs",
            pa.struct(
                [
                    pa.field(
                        "ints",
                        pa.map_(pa.string(), pa.int64()),
                        nullable=False,
                    ),
                    pa.field(
                        "floats",
                        pa.map_(pa.string(), pa.float64()),
                        nullable=False,
                    ),
                    pa.field(
                        "strings",
                        pa.map_(pa.string(), pa.string()),
                        nullable=False,
                    ),
                ]
            ),
            nullable=False,
        ),
        # int
        pa.field("seed", pa.int64(), nullable=True),
        # dict[str, dict[str, float]]
        pa.field(
            "detectors",
            pa.map_(
                pa.string(),
                pa.map_(pa.string(), pa.float64()),
            ),
            nullable=False,
        ),
        # float
        pa.field("duration", pa.float64(), nullable=False),
        # float
        pa.field("sampling_frequency", pa.float64(), nullable=False),
        # float | None
        pa.field("network_optimal_snr", pa.float64(), nullable=True),
        # float | None
        pa.field("network_matched_filter_snr", pa.float64(), nullable=True),
    ]
)


def save_metadata(metadata: list, filepath: str | Path) -> None:
    """Save a list of InjectionMetaData objects to a parquet file.

    Parameters
    ----------
    metadata : list
        List of InjectionMetaData objects to save
    filepath : str | Path
        Path where the parquet file will be saved
    """
    metadata_dicts = [asdict(m) for m in metadata]
    table = pa.Table.from_pylist(metadata_dicts, schema=INJECTION_METADATA_SCHEMA)
    pq.write_table(table, filepath)


def read_metadata_raw(filepath: str | Path) -> pa.Table:
    """Read all metadata from a parquet file.

    Parameters
    ----------
    filepath : str | Path
        Path to the parquet file

    Returns
    -------
    pa.Table
        PyArrow Table containing all metadata records
    """
    return pq.read_table(filepath)


def read_single_metadata_raw(filepath: str | Path, row_index: int) -> dict:
    """Read a single metadata record from a parquet file by row index.

    Parameters
    ----------
    filepath : str | Path
        Path to the parquet file
    row_index : int
        Index of the row to read

    Returns
    -------
    dict
        Dictionary containing the metadata for the specified row
    """
    table = pq.read_table(filepath)
    if row_index < 0 or row_index >= len(table):
        raise IndexError(f"Row index {row_index} out of range [0, {len(table)})")
    return table.slice(row_index, 1).to_pylist()[0]


def _parse_metadata_dict(data: dict) -> dict:
    """Parse metadata dictionary from parquet, handling type conversions.

    Converts PyArrow-serialized types back to Python types:
    - Maps (list of tuples) back to dicts
    - Handles nested map structures

    Parameters
    ----------
    data : dict
        Raw dictionary from parquet

    Returns
    -------
    dict
        Parsed dictionary suitable for creating InjectionMetaData
    """
    parsed = {}

    for key, value in data.items():
        if key == "injection_parameters":
            # Convert list of tuples back to dict (or None if empty/null)
            parsed[key] = dict(value) if value else None
        elif key == "waveform_kwargs":
            # Reconstruct from structured format (floats/ints/strings)
            if isinstance(value, dict) and (
                "floats" in value or "ints" in value or "strings" in value
            ):
                reconstructed = {}
                for category in ["ints", "floats", "strings"]:
                    if category in value and value[category]:
                        reconstructed.update(value[category])
                parsed[key] = reconstructed
            else:
                parsed[key] = value
        elif key == "detectors":
            # Convert nested map structure: list of (str, list of tuples) back to dict[str, dict]
            detectors = {}
            if isinstance(value, list):
                for det_name, det_data in value:
                    detectors[det_name] = (
                        dict(det_data) if isinstance(det_data, list) else det_data
                    )
            elif isinstance(value, dict):
                for det_name, det_data in value.items():
                    detectors[det_name] = (
                        dict(det_data) if isinstance(det_data, list) else det_data
                    )
            parsed[key] = detectors
        else:
            parsed[key] = value

    return parsed


def read_metadata(filepath: str | Path) -> list[InjectionMetaData]:
    """Read all metadata from a parquet file as InjectionMetaData objects.

    Parameters
    ----------
    filepath : str | Path
        Path to the parquet file

    Returns
    -------
    list[InjectionMetaData]
        List of InjectionMetaData objects
    """
    table = read_metadata_raw(filepath)
    data_dicts = table.to_pylist()
    return [InjectionMetaData(**_parse_metadata_dict(d)) for d in data_dicts]


def read_single_metadata(filepath: str | Path, row_index: int) -> InjectionMetaData:
    """Read a single metadata record from a parquet file as an InjectionMetaData object.

    Parameters
    ----------
    filepath : str | Path
        Path to the parquet file
    row_index : int
        Index of the row to read

    Returns
    -------
    InjectionMetaData
        The metadata object for the specified row
    """
    data = read_single_metadata_raw(filepath, row_index)
    return InjectionMetaData(**_parse_metadata_dict(data))
