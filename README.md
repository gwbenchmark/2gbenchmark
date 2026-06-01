# 2gbenchmark

A Python package for generating standardised benchmark datasets of simulated
gravitational-wave (GW) signals from merging binary black holes, as they would
be observed by the current (2nd-generation) detector network: LIGO Hanford (H1),
LIGO Livingston (L1), and Virgo (V1).

The package draws source parameters from astrophysical prior distributions,
generates frequency-domain detector data (signal + simulated Gaussian noise),
and writes the results to disk in a structured format suitable for benchmarking
GW data-analysis pipelines.

## Installation

Requires Python >= 3.11. Install from the repository root:

```bash
pip install -e .
```

## Quick start — generating a Level 0 dataset

Level 0 is the simplest benchmark tier. Several "extrinsic" parameters (sky
location, polarisation angle, orbital phase, inclination, and coalescence time)
are held fixed so that the inference problem reduces to estimating the masses,
spins, and distance of the source.

Create a YAML configuration file, e.g. `level0_config.yaml`:

```yaml
seed: 42
n_simulations: 100
```

Only `seed` (random seed for reproducibility) and `n_simulations` (number of
signals to generate) are required. All other settings use sensible defaults for
Level 0:

| Parameter               | Default        | Description                                      |
|-------------------------|----------------|--------------------------------------------------|
| `duration`              | 4.0 s          | Length of each data segment                       |
| `sampling_frequency`    | 2048.0 Hz      | Sampling rate (Nyquist frequency = 1024 Hz)       |
| `waveform_approximant`  | `IMRPhenomD`   | Waveform model used to generate signals           |
| `detectors`             | `[H1, L1, V1]` | Detector network                                 |
| `blind`                 | `false`        | If true, injection parameters are hidden from output |

Run the CLI:

```bash
gwbenchmark2g --config level0_config.yaml --level 0 --output-dir output_level0
```

## Level definitions

| Level | IFOs | Waveform | Physics |
|-------|------|----------|---------|
| 0 | H1-L1-V1 | IMRPhenomXAS | Aligned spins, fixed sky location and time |
| 1 | H1-L1-V1/H1-L1/H1 | IMRPhenomXHM | Aligned spins with higher-order multipoles, all parameters randomized. Different detector combinations. |

## Quick start — generating a Level 1 dataset

Level 1 extends the benchmark to randomized extrinsic parameters, higher-order
multipoles, and mixed detector networks. The coalescence time is sampled
uniformly in `[-0.1, 0.1]`, while the segment start follows the same convention
as Level 0 so that the merger remains `duration - 2` seconds after segment
start.

Create a YAML configuration file, e.g. `level1_config.yaml`:

```yaml
seed: 42
n_simulations: 100
duration: 8.0
detectors:
  detector_combinations:
    - [H1, L1, V1]
    - [H1, L1]
    - [H1]
  weights: [1.0, 1.0, 1.0]
```

Level 1 defaults:

| Parameter               | Default                                | Description |
|-------------------------|----------------------------------------|-------------|
| `waveform_approximant`  | `IMRPhenomXHM`                          | Waveform model with higher-order multipoles |
| `geocent_time_range`    | `[-0.1, 0.1]`                          | Uniform absolute merger-time sampling window |
| `detectors`             | `{detector_combinations: [[H1, L1, V1], [H1, L1], [H1]], weights: [1, 1, 1]}` | Allowed detector subsets and their relative sampling weights |
| `fixed_parameters`      | `null`                                 | No Level 0-style fixed extrinsic parameters |

Run the CLI:

```bash
gwbenchmark2g --config level1_config.yaml --level 1 --output-dir output_level1
```

Each Level 1 simulation may contain a different detector subset. The selected
network is recorded in the metadata as `network_label`, and the strain file only
contains the detectors active for that simulation.

## Output format

The output directory will contain:

```
output_level0/
  simulation_0.npz
  simulation_1.npz
  ...
  simulation_99.npz
  injection_metadata.parquet
```

### Strain data files (`.npz`)

There is one `.npz` file per simulated signal. Each file contains a single key,
`"data"`, which holds a Python dictionary keyed by detector name (`"H1"`,
`"L1"`, `"V1"`). Each detector entry contains three arrays:

| Array             | dtype       | Shape     | Description                                                              |
|-------------------|-------------|-----------|--------------------------------------------------------------------------|
| `strain`          | `complex128`| `(N,)`    | Frequency-domain detector data (noise + injected signal)                 |
| `psd`             | `float64`   | `(N,)`    | One-sided power spectral density of the detector noise                   |
| `frequency_array` | `float64`   | `(N,)`    | Frequency bin centres, from 0 Hz to the Nyquist frequency                |

The array length is `N = duration × sampling_frequency / 2 + 1`. With the
defaults (4 s, 2048 Hz) this gives **N = 4097** frequency bins spaced at
0.25 Hz, covering 0–1024 Hz.

PSD values are `inf` below the detector's minimum sensitive frequency (20 Hz),
indicating that those bins carry no useful information.

**Note:** Because the data is stored as a pickled dictionary of dataclass
objects, loading requires `numpy.load(..., allow_pickle=True)`.

Example usage:

```python
import numpy as np

sim = np.load("output_level0/simulation_0.npz", allow_pickle=True)
data = sim["data"].item()  # unwrap the 0-d object array

h1 = data["H1"]
print(h1.strain.shape)          # (4097,)
print(h1.frequency_array[:5])   # [0.   0.25 0.5  0.75 1.  ]
```

### Metadata file (`.parquet`)

A single Apache Parquet file, `injection_metadata.parquet`, contains one row per
simulation with the following fields:

| Field                    | Type                            | Description                                                             |
|--------------------------|---------------------------------|-------------------------------------------------------------------------|
| `injection_parameters`   | `map<string, float64>`          | True source parameters of the injected signal (null if `blind=true`)    |
| `waveform_kwargs`        | `struct{ints, floats, strings}` | Extra arguments passed to the waveform generator
| `fixed_parameters`       | `map<string, float64>`          | Parameters held fixed for the level (null for Level 1 defaults)         |
| `waveform_approximant`   | `string`                        | Waveform approximant requested by the dataset config                    |
| `seed`                   | `int64`                         | Random seed used for the simulation                                     |
| `detectors`              | `map<string, map<string, float64>>` | Per-detector metadata (e.g. `minimum_frequency`, `maximum_frequency`) |
| `duration`               | `float64`                       | Segment duration in seconds                                             |
| `sampling_frequency`     | `float64`                       | Sampling rate in Hz                                                     |
| `level`                  | `int64`                         | Benchmark level used for the simulation                                 |
| `network_label`          | `string`                        | Detector network selected for this simulation                           |

The `injection_parameters` dictionary contains the physical parameters of the
simulated source. For Level 0, the free (non-fixed) parameters are:

- `chirp_mass` — a combination of the two component masses that governs the
  signal's frequency evolution (solar masses)
- `mass_ratio` — ratio of the lighter to the heavier component mass (0 < q <= 1)
- `luminosity_distance` — distance to the source (Mpc)
- `chi_1`, `chi_2` — dimensionless spin of the heavier and lighter black hole
  along the orbital angular momentum axis

The remaining parameters (`geocent_time`, `phase`, `psi`, `theta_jn`, `dec`,
`ra`) are held at fixed values in Level 0. In Level 1, these parameters are
sampled from the prior instead, with `geocent_time` restricted to the configured
`geocent_time_range`.

Example usage:

```python
import pyarrow.parquet as pq

table = pq.read_table("output_level0/injection_metadata.parquet")
row = table.to_pylist()[0]
print(dict(row["injection_parameters"]))
# {'mass_ratio': 0.462, 'chirp_mass': 37.64, 'luminosity_distance': 3070.9, ...}
```
