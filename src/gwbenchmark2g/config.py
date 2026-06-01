"""Configuration dataclasses for the 2gbenchmark package.

This may move to gwbenchmark in the future.
"""

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

SUPPORTED_DETECTORS = ("H1", "L1", "V1")
DEFAULT_LEVEL1_NETWORKS = (
    ("H1", "L1", "V1"),
    ("H1", "L1"),
    ("H1",),
)


class DatasetConfig(BaseModel):
    duration: float = 4.0
    sampling_frequency: float = 2048.0
    waveform_approximant: str = "IMRPhenomD"
    seed: int
    blind: bool = False
    n_simulations: int
    fixed_parameters: dict[str, float] | None = None


class Level1NetworkConfig(BaseModel):
    detectors: list[str]
    weight: float = 1.0

    @field_validator("detectors")
    @classmethod
    def validate_detectors(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("detectors must not be empty.")
        if len(set(value)) != len(value):
            raise ValueError("detectors must not contain duplicates.")
        invalid = set(value) - set(SUPPORTED_DETECTORS)
        if invalid:
            raise ValueError(
                "detectors contains unsupported entries: "
                + ", ".join(sorted(invalid))
            )
        return [detector for detector in SUPPORTED_DETECTORS if detector in value]

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"weight must be positive, got {value}.")
        return value

    @property
    def label(self) -> str:
        return "-".join(self.detectors)


class Level0Config(DatasetConfig):
    detectors: list[str] = ["H1", "L1", "V1"]
    """List of detectors to simulate data for."""
    fixed_parameters: dict[str, float] = dict(
        geocent_time=-0.01621880385450652,
        phase=0.0,
        psi=0.0,
        theta_jn=0.0,
        dec=2.058804189275143,
        ra=-1.595801372295631,
    )

    @computed_field
    @property
    def level(self) -> int:
        return 0


class Level1Config(DatasetConfig):
    waveform_approximant: str = "IMRPhenomHM"
    fixed_parameters: dict[str, float] | None = None
    geocent_time_range: tuple[float, float] = (-0.1, 0.1)
    detectors: list[Level1NetworkConfig] = Field(
        default_factory=lambda: [
            Level1NetworkConfig(detectors=list(detectors))
            for detectors in DEFAULT_LEVEL1_NETWORKS
        ]
    )

    @field_validator("geocent_time_range")
    @classmethod
    def validate_geocent_time_range(
        cls, value: tuple[float, float]
    ) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError("geocent_time_range must contain exactly two values.")
        minimum, maximum = value
        if minimum >= maximum:
            raise ValueError(
                "geocent_time_range minimum must be strictly less than maximum."
            )
        return value

    @field_validator("detectors")
    @classmethod
    def validate_detectors(
        cls, value: list[Level1NetworkConfig]
    ) -> list[Level1NetworkConfig]:
        if not value:
            raise ValueError("detectors must not be empty.")
        return value

    @model_validator(mode="after")
    def validate_unique_networks(self):
        labels = [network.label for network in self.detectors]
        if len(set(labels)) != len(labels):
            raise ValueError("detectors must not contain duplicate detector sets.")
        return self

    @computed_field
    @property
    def level(self) -> int:
        return 1


level_registry = {
    0: Level0Config,
    1: Level1Config,
}
