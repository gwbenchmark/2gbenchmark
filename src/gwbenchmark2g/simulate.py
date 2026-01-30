from dataclasses import dataclass
from typing import Any, Generator

import bilby

from .config import DatasetConfig

ArrayLike = Any  # Placeholder for actual type once gwbenchmark.types is available

__all__ = [
    "FrequencyDomainInterferometerData",
    "InjectionMetaData",
    "simulate_level_0",
    "simulate_registry",  # hide?
]


@dataclass
class FrequencyDomainInterferometerData:
    strain: ArrayLike
    psd: ArrayLike
    frequency_array: ArrayLike


@dataclass
class InjectionMetaData:
    injection_parameters: dict[str, float] | None  # allow none to enable blinding
    waveform_kwargs: dict[str, int | float | str]
    seed: int
    detectors: dict[str, dict]
    duration: float
    sampling_frequency: float


def simulate_level_0(
    config: DatasetConfig,
) -> Generator[
    tuple[dict[str, FrequencyDomainInterferometerData], dict[str, Any]], None, None
]:
    """Simulate a level 0 benchmark dataset.

    Parameters
    ----------
    config : DatasetConfig
        Configuration parameters for the simulation.

    Yields
    ------
    data: dict[str, FrequencyDomainInterferometerData]
        Dictionary of data in each detector
    metadata: dict[str, Any]
        Information about the injected parameters, random seed, and
        per-detector metadata, e.g., frequency bounds.

    """
    bilby.core.utils.random.seed(config.seed)
    dist = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
    for key, parameters in (config.fixed_parameters or {}).items():
        dist[key] = parameters
    ifos = bilby.gw.detector.InterferometerList(config.detectors)
    wfg = bilby.gw.waveform_generator.WaveformGenerator(
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        duration=config.duration,
        sampling_frequency=config.sampling_frequency,
    )
    for _ in range(config.n_simulations):
        parameters = dist.sample()
        wfg.start_time = parameters["geocent_time"] - config.duration + 2
        ifos.set_strain_data_from_power_spectral_densities(
            duration=config.duration,
            sampling_frequency=config.sampling_frequency,
            start_time=parameters["geocent_time"] - config.duration + 2,
        )
        ifos.inject_signal(waveform_generator=wfg, parameters=parameters)

        metadata = InjectionMetaData(
            injection_parameters=parameters
            if not config.blind
            else None,  # this feels ugly, maybe we need a function to strip metadata out instead
            waveform_kwargs=wfg.waveform_arguments,
            detectors=dict(),
            seed=config.seed if not config.blind else None,
            duration=config.duration,
            sampling_frequency=config.sampling_frequency,
        )
        data = dict()
        for ifo in ifos:
            data[ifo.name] = FrequencyDomainInterferometerData(
                strain=ifo.frequency_domain_strain,
                psd=ifo.power_spectral_density_array,
                frequency_array=ifo.frequency_array,
            )
            metadata.detectors[ifo.name] = dict(
                minimum_frequency=ifo.minimum_frequency,
                maximum_frequency=ifo.maximum_frequency,
            )

        yield data, metadata


simulate_registry = {
    0: simulate_level_0,
}
