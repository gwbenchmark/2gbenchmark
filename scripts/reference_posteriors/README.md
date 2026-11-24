# Reference posteriors for the level 0 and level 1 datasets

These scripts run bilby_pipe on the level 0 and level 1 benchmark events to
produce the reference posteriors. The one unusual thing is that the strain isn't
simulated: each event is read straight from its `simulation_<idx>.npz` file, so
the posteriors come from exactly the data we published.

Everything else (the sampler, plots, job submission) is just normal bilby_pipe.

## Existing signals and result metadata

The NPZ strain already contains the injected signal and noise. Recovery reads
that strain unchanged; it does not inject another signal or generate new noise.
Keep these settings in every `.ini`:

```ini
gaussian-noise = True
injection = False
```

Here `gaussian-noise = True` selects the bilby_pipe method that our NPZ loader
overrides. That method reads the saved strain and PSDs instead of generating
noise.

For non-blind data, the loader also reads the true parameter values from row
`idx` of `injection_metadata.parquet`, matching `simulation_<idx>.npz`. After
loading the strain, it records these values as `injection_parameters` in the
result metadata for p–p plots. This only records the truth; it does not change
the strain or enable signal injection.

Blinded datasets with null, empty or absent `injection_parameters`, and datasets
without a metadata file, still run with truth unset. Unreadable metadata or a
missing event row raises an error.

## Per-event trigger times and detectors (level 1)

Level 1 events each have their own trigger time and detector network (H1L1V1,
H1L1 or H1). When the workflow is built, `run_reference_posteriors.py` reads
both from `injection_metadata.parquet`:

- each generation job gets that event's `--trigger-time`, which sets the
  segment start (`trigger_time + post-trigger-duration - duration`) and the
  `geocent_time` prior;
- only the detectors in the event's npz file are loaded, and its analysis job
  gets those `--detectors` and a label ending in them (e.g. `..._analysis_H1`).

`geocent_time` is left out of `level1_recovery.prior`, so bilby_pipe adds
`Uniform(trigger_time - deltaT/2, trigger_time + deltaT/2)` for each event. With
`deltaT = 0.2` (set in the level 1 `.ini` files) that is exactly the generation
prior, `trigger_time + Uniform(-0.1, 0.1)`. Don't set `trigger-time` in a
level 1 `.ini`: the build stops if it disagrees with the metadata.

For level 0 nothing changes: every event uses the `.ini` `trigger-time` (which
matches the metadata, where present) and all three detectors.

Each generation job also checks its trigger time and loaded detectors against
the event's metadata row, and raises if they disagree.

## What you need

Run this in the environment that has **bilby 2.7.1**, **bilby_pipe** and
**gwbenchmark2g** installed together. bilby has to be 2.7.1 because that's the
version the data was generated with, and gwbenchmark2g is needed to open the
npz files and the metadata (also when building the workflow). The `2g-pe`
image from `gwbenchmark/containers` has all three.

## How to run it

From inside this folder (the `.ini` points at its prior with a relative path,
and its `outdir` has to be inside the folder you run from, so it matters where
you are):

| Dataset | `.ini` |
|---|---|
| level 0 | `level0.ini` |
| level 1 | `level1.ini` |
| level 1 blind | `level1-blind.ini` |

```bash
python run_reference_posteriors.py level1.ini --npz-directory /path/to/level1
```

That builds the workflow but doesn't run anything yet. It prints the submit
command, which is just:

```bash
sbatch outdir_level1/submit/slurm_level1_master.sh
```

and that launches one job per event on the cluster. Results land in
`outdir_level1/`.

If you want to try it on a couple of events first, drop `n-simulation` in the
`.ini` to e.g. 2 before running.

## Blinded data

Use `level1-blind.ini` with the `level1-blind` data. It differs from
`level1.ini` only in its label and outdir, so blind results can't land in the
unblinded outdir. The blind metadata has no truth, so none is attached to the
results. The blind posteriors still give the hidden parameters away
approximately, so keep `outdir_level1-blind/` under the same access control as
the unblinded truth.

## Running inside a container

To run every job inside an Apptainer/Singularity image, add `--container`:

```bash
python run_reference_posteriors.py level0.ini --npz-directory /path/to/level0 \
    --container /path/to/image.sif
```

- **HTCondor** (`scheduler = condor`): uses bilby_pipe's own container support,
  so the submit files ask HTCondor to run each job in the image (the exact
  lines depend on the bilby_pipe version). What is visible inside the container is
  set by the cluster, so the npz directory and this folder need to be on a
  filesystem the cluster mounts into containers. `--bind` does nothing here.
  The script prints a warning listing the folders that need to be visible.
  For this shared-filesystem setup, pass `--transfer-files False`; the custom
  loader does not stage its script or NPZ inputs for file-transfer jobs.
  The image and output directory must also be accessible on the workers.
  On pools configured with `SINGULARITY_BIND_EXPR = TARGET.SingularityBind`,
  use `--condor-bind /path/to/level0` to request the dataset mount (repeatable).
  This writes `+SingularityBind` in every job's submit file; the source paths
  must exist on the execute nodes. It does not transfer files. This option
  depends on the pool configuration, unlike Slurm's direct `--bind` option.
- **Slurm** (`scheduler = slurm`): each job script runs its command through
  `apptainer exec`. The npz directory, this folder and the run directory are
  bound automatically; add `--bind PATH` (repeatable) for anything else. The
  compute nodes need `apptainer` on the `PATH` (add `scheduler-module` to the
  `.ini` if it has to be loaded), and `scheduler-env` can't be used with it.

Pass the image with `--container`, not `container =` in the `.ini`, and don't
set `conda-env`.

Run `run_reference_posteriors.py` itself inside the same image, e.g.
`apptainer exec image.sif python run_reference_posteriors.py ...`, then submit
from the host, since `sbatch`/`condor_submit_dag` aren't in the container.
The wrapper records absolute executable paths from that image for Condor;
bare names such as `python` can otherwise be resolved under Condor's scratch
directory instead of the image's `PATH`.

## A few things to know

- **The npz directory goes on the command line, not in the `.ini`.** Kept out of
  config on purpose so the config is just about analysis settings.
- **The recovery prior is the same one the data was generated with.** If the
  generation prior changes, rebuild it. Level 0 uses the defaults:
  `python build_prior.py`. Level 1 is built from the dataset config in the
  `gwbenchmark/configs` repo (or the `config.yaml` saved with the data):
  `python build_prior.py --dataset-config /path/to/configs/gwbenchmark2g/level1.yaml --level 1`.
  The committed `level1_recovery.prior` was built from `level1.yaml` at
  `configs` commit `083ac38`; `level1-blind.yaml` gives the same file.
- **Keep the outdir inside this folder.** With bilby_pipe's default file
  transfer, a `--outdir` outside the folder you run from fails when the
  workflow is built.
- The waveform for recovery is **IMRPhenomXAS** for level 0 and
  **IMRPhenomXHM** for level 1, matching the data.

## What's in here

- `npz_generation.py` – the bit that reads the npz strain into bilby_pipe.
- `run_reference_posteriors.py` – the script you actually run.
- `level0.ini`, `level1.ini`, `level1-blind.ini` – all the analysis settings
  (detectors, sampler, waveform, ...).
- `build_prior.py` / `level0_recovery.prior` / `level1_recovery.prior` – the
  recovery priors.
- `test_*.py` – tests for the loader, the workflow and the prior builder.

## Continuing a partial run

`--start-index 100 --n-simulation 900` selects `simulation_100.npz` through
`simulation_999.npz`, including the corresponding metadata rows (trigger times
and detectors) and job labels.
Use a separate label and outdir so the completed first 100 stay available.
`--start-index` is a wrapper CLI option, rather than a bilby_pipe INI setting.
