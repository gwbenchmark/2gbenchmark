# Reference posteriors for the level 0 dataset

These scripts run bilby_pipe on the level 0 benchmark events to produce the
reference posteriors. The one unusual thing is that the strain isn't simulated:
each event is read straight from its `simulation_<idx>.npz` file, so the
posteriors come from exactly the data we published.

Everything else (the sampler, plots, job submission) is just normal bilby_pipe.

## What you need

Run this in the environment that has **bilby 2.7.1**, **bilby_pipe** and
**gwbenchmark2g** installed together. bilby has to be 2.7.1 because that's the
version the data was generated with, and gwbenchmark2g is needed to open the
npz files. 

## How to run it

From inside this folder (the `.ini` points at `level0_recovery.prior` with a
relative path, so it matters where you are):

```bash
python run_reference_posteriors.py level0.ini --npz-directory /path/to/level0_data/level0
```

That builds the workflow but doesn't run anything yet. It prints the submit
command, which is just:

```bash
sbatch outdir_level0/submit/slurm_level0_master.sh
```

and that launches the 1000 jobs on the cluster. Results land in `outdir_level0/`.

If you want to try it on a couple of events first, drop `n-simulation` in
`level0.ini` to e.g. 2 before running.

## A few things to know

- **The npz directory goes on the command line, not in the `.ini`.** Kept out of 
  config on purpose so the config is just about analysis settings.
- **The recovery prior is the same one the data was generated with.** It lives
  in `level0_recovery.prior`. If generation prior is changed, must rebuild it
  with `python build_prior.py`.
- **One trigger time is used for every event.** This is fine for level 0 because
  all the events share the same merger time, but it won't be true for level 1,
  so that will need handling differently when we get there.
- The waveform for recovery is **IMRPhenomXAS**, matching the (to be regenerated)
  level 0 events. Change `waveform-approximant` in the `.ini` if that changes.

## What's in here

- `npz_generation.py` – the bit that reads the npz strain into bilby_pipe.
- `run_reference_posteriors.py` – the script you actually run.
- `level0.ini` – all the analysis settings (detectors, sampler, waveform, ...).
- `build_prior.py` / `level0_recovery.prior` – the recovery prior.
- `test_npz_generation.py` – a quick check that the strain is read in correctly.
