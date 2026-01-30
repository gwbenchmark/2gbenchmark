"""Configuration dataclasses for the 2gbenchmark package.

This may move to gwbenchmark in the future.
"""

from pydantic import BaseModel, computed_field


class DatasetConfig(BaseModel):
    duration: float = 4.0
    sampling_frequency: float = 2048.0
    waveform_approximant: str = "IMRPhenomD"
    detectors: list[str] = ["H1", "L1", "V1"]
    """List of detectors to simulate data for."""
    seed: int
    blind: bool = False
    n_simulations: int
    fixed_parameters: dict[str, float] | None = None


class Level0Config(DatasetConfig):
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
    fixed_parameters: dict[str, float] = dict(geocent_time=0.0)

    @computed_field
    @property
    def level(self) -> int:
        return 1


level_registry = {
    0: Level0Config,
    1: Level1Config,
}
