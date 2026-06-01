from dataclasses import dataclass
from typing import Any, Generator

import bilby
import numpy as np

from .config import DatasetConfig, Level1Config, Level1NetworkConfig

ArrayLike = Any  # Placeholder for actual type once gwbenchmark.types is available

__all__ = [
    "FrequencyDomainInterferometerData",
    "InjectionMetaData",
    "simulate_level_0",
    "simulate_level_1",
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
    waveform_approximant: str
    seed: int | None
    detectors: dict[str, dict]
    duration: float
    sampling_frequency: float
    level: int | None = None
    network_label: str | None = None
    network_optimal_snr: float | None = None
    network_matched_filter_snr: float | None = None


def _create_prior(
    config: DatasetConfig,
    *,
    geocent_time_range: tuple[float, float] | None = None,
) -> bilby.gw.prior.BBHPriorDict:
    dist = bilby.gw.prior.BBHPriorDict(aligned_spin=True)
    dist["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        name="luminosity_distance", minimum=1750.0, maximum=2250.0
    )
    if geocent_time_range is not None:
        minimum, maximum = geocent_time_range
        dist["geocent_time"] = bilby.core.prior.Uniform(
            minimum=minimum,
            maximum=maximum,
            name="geocent_time",
        )
    for key, parameters in (config.fixed_parameters or {}).items():
        dist[key] = parameters
    return dist


def _create_waveform_generator(config: DatasetConfig):
    return bilby.gw.waveform_generator.WaveformGenerator(
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        duration=config.duration,
        sampling_frequency=config.sampling_frequency,
        waveform_arguments={"waveform_approximant": config.waveform_approximant},
    )


def _sample_network_label(config: Level1Config, rng: np.random.Generator) -> str:
    labels = [network.label for network in config.detectors]
    weights = np.asarray([network.weight for network in config.detectors])
    probabilities = weights / weights.sum()
    return str(rng.choice(labels, p=probabilities))


def _get_level1_network(config: Level1Config, label: str) -> Level1NetworkConfig:
    for network in config.detectors:
        if network.label == label:
            return network
    raise KeyError(f"Unknown Level 1 network label {label!r}")


def _simulate_dataset(
    config: DatasetConfig,
    *,
    level: int,
    prior: bilby.gw.prior.BBHPriorDict,
    network_selector,
) -> Generator[tuple[dict[str, FrequencyDomainInterferometerData], InjectionMetaData], None, None]:
    bilby.core.utils.random.seed(config.seed)
    waveform_generator = _create_waveform_generator(config)

    for _ in range(config.n_simulations):
        parameters = prior.sample()
        detectors, network_label = network_selector()
        ifos = bilby.gw.detector.InterferometerList(detectors)
        start_time = parameters["geocent_time"] - config.duration + 2
        waveform_generator.start_time = start_time
        ifos.set_strain_data_from_power_spectral_densities(
            duration=config.duration,
            sampling_frequency=config.sampling_frequency,
            start_time=start_time,
        )
        ifos.inject_signal(waveform_generator=waveform_generator, parameters=parameters)

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
            injection_parameters=parameters if not config.blind else None,
            fixed_parameters=config.fixed_parameters,
            waveform_approximant=config.waveform_approximant,
            seed=config.seed if not config.blind else None,
            detectors=dict(),
            duration=config.duration,
            sampling_frequency=config.sampling_frequency,
            level=level,
            network_label=network_label,
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


def simulate_level_0(
    config: DatasetConfig,
) -> Generator[
    tuple[dict[str, FrequencyDomainInterferometerData], InjectionMetaData], None, None
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
    prior = _create_prior(config)
    yield from _simulate_dataset(
        config,
        level=0,
        prior=prior,
        network_selector=lambda: (config.detectors, "-".join(config.detectors)),
    )


def simulate_level_1(
    config: Level1Config,
) -> Generator[
    tuple[dict[str, FrequencyDomainInterferometerData], InjectionMetaData], None, None
]:
    """Simulate a level 1 benchmark dataset."""
    if config.level != 1:
        raise ValueError("Config level must be 1 for level 1 simulation.")
    prior = _create_prior(config, geocent_time_range=config.geocent_time_range)
    rng = np.random.default_rng(config.seed)
    yield from _simulate_dataset(
        config,
        level=1,
        prior=prior,
        network_selector=lambda: (
            _get_level1_network(config, label := _sample_network_label(config, rng)).detectors,
            label,
        ),
    )


simulate_registry = {
    0: simulate_level_0,
    1: simulate_level_1,
}
