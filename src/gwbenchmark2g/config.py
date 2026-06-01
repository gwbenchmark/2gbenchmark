"""Configuration dataclasses for the 2gbenchmark package.

This may move to gwbenchmark in the future.
"""

import numpy as np
from pydantic import BaseModel, computed_field, field_validator


class DetectorNetworkConfig(BaseModel):
    detector_combinations: list[tuple[str, ...]]
    weights: list[float] | None = None

    def sample_network(self, rng: np.random.Generator) -> list[str]:
        if self.weights is not None:
            if len(self.weights) != len(self.detector_combinations):
                raise ValueError(
                    "Length of weights must match length of detector_combinations."
                )
            probabilities = [weight / sum(self.weights) for weight in self.weights]
            selected_combination = rng.choice(
                len(self.detector_combinations), p=probabilities
            )
        else:
            selected_combination = rng.choice(len(self.detector_combinations))
        detectors = self.detector_combinations[selected_combination]
        return list(detectors)


class DatasetConfig(BaseModel):
    duration: float = 4.0
    sampling_frequency: float = 2048.0
    waveform_approximant: str = "IMRPhenomD"
    seed: int
    blind: bool = False
    n_simulations: int
    fixed_parameters: dict[str, float] | None = None
    geocent_time_range: tuple[float, float] | None = None
    detectors: list[str] | DetectorNetworkConfig


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
    waveform_approximant: str = "IMRPhenomXHM"
    fixed_parameters: dict[str, float] | None = None
    geocent_time_range: tuple[float, float] = (-0.1, 0.1)
    detectors: DetectorNetworkConfig = DetectorNetworkConfig(
        detector_combinations=[
            ("H1", "L1", "V1"),
            ("H1", "L1"),
            ("H1",),
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

    @computed_field
    @property
    def level(self) -> int:
        return 1


level_registry = {
    0: Level0Config,
    1: Level1Config,
}
