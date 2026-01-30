import pyarrow as pa


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
        pa.field("seed", pa.int64(), nullable=False),
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
    ]
)
