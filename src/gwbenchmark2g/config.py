"""Configuration dataclasses for the 2gbenchmark package.

This may move to gwbenchmark in the future.
"""

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    duration: float = 4.0
    sampling_frequency: float = 2048.0
    waveform_approximant: str = "IMRPhenomD"
    detectors: list[str] = ["H1", "L1", "V1"]
    """List of detectors to simulate data for."""
    seed: int
    blind: bool = False
    n_simulations: int


class Level0Config(DatasetConfig):
    pass


class Level1Config(DatasetConfig):
    pass


level_registry = {
    0: Level0Config,
    1: Level1Config,
}
