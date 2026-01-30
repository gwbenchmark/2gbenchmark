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
    fixed_parameters: dict[str, float] | None
    waveform_kwargs: dict[str, int | float | str]
    seed: int
    detectors: dict[str, dict]
    duration: float
    sampling_frequency: float
    network_optimal_snr: float | None = None
    network_matched_filter_snr: float | None = None


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
    if config.level != 0:
        raise ValueError("Config level must be 0 for level 0 simulation.")
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

        # Calculate network SNRs from individual detector SNRs
        if not config.blind:
            network_optimal_snr = (
                sum(ifo.meta_data["optimal_SNR"] ** 2 for ifo in ifos) ** 0.5
            )
            network_matched_filter_snr = (
                sum(ifo.meta_data["matched_filter_SNR"].real ** 2 for ifo in ifos)
                ** 0.5
            )
        else:
            network_optimal_snr = None
            network_matched_filter_snr = None

        metadata = InjectionMetaData(
            injection_parameters=parameters
            if not config.blind
            else None,  # this feels ugly, maybe we need a function to strip metadata out instead
            waveform_kwargs=wfg.waveform_arguments,
            detectors=dict(),
            seed=config.seed if not config.blind else None,
            duration=config.duration,
            fixed_parameters=config.fixed_parameters,
            sampling_frequency=config.sampling_frequency,
            network_optimal_snr=network_optimal_snr,
            network_matched_filter_snr=network_matched_filter_snr,
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
                optimal_snr=ifo.meta_data["optimal_SNR"] if not config.blind else None,
                matched_filter_snr=ifo.meta_data["matched_filter_SNR"].real
                if not config.blind
                else None,
            )

        yield data, metadata


simulate_registry = {
    0: simulate_level_0,
}
