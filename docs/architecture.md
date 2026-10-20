---
type: architecture
status: current
updated: 2026-10-21
sources: [code]
verified_by: pending (Joon)
---

# Architecture

What the code is now, read from the tree at `3b12b0d` (the ICLR 2027 submission state), with
the problems Phase 0–1 of `docs/roadmap.md` address. Decisions live in `docs/decisions/`; the
plan in `docs/roadmap.md`; branch-level detail in `docs/changes/`. This page is kept current
with the code: when a module moves, this changes in the same commit.

## Purpose

Train Neural ODEs on chaotic time series so that the learned vector field is accurate *locally*
(Jacobian) as well as *statistically* (long-term invariant measure). The Neighborhood method
replaces per-trajectory MSE with a loss on the deformation of phase-space neighborhoods, which
carries Jacobian information without observing the Jacobian. Everything else here —
systems, metrics, baselines — exists to generate the data and evaluate that claim.

## Layout

A `uv` workspace: a root project (`sandbox`, to be renamed) that depends on the packages
under `packages/`, plus Hydra scripts and notebooks.

```
packages/
├── dynamical_systems/        the ground-truth side: ODEs, tangent dynamics, metrics
│   └── src/dynamical_systems/
│       ├── continuous/       AbstractODE (rhs, solve_ode) · Lorenz63/96 · Rossler76/79 ·
│       │                     Chen99 / HyperChen05 · FitzhughNagumo · TangentODE
│       ├── analysis/         lyapunov (naive, Gram–Schmidt, nonlinear) · jacobian
│       │                     (one-step, empirical) · characteristic_time (power spectrum)
│       ├── metrics/          pointwise (MSE, RMSE, cosine, valid_prediction_time) ·
│       │                     statistical (spectral Hellinger) · measure_distances
│       │                     (MMD, Sinkhorn via ott-jax)
│       ├── transforms/       AbstractBijection, Standardize, TransformedODE
│       └── linalg.py, utils.py
├── dynamics_discovery/       the learned side: data, models, losses, training
│   └── src/dynamics_discovery/
│       ├── data/             TimeSeriesDataset (HDF5, downsample, noise, standardize) ·
│       │                     loaders (SegmentLoader + Full/RandomSample/MiniBatching) ·
│       │                     utils.generate_ode_dataset
│       ├── models/           AbstractDynamicsModel / AbstractLatentDynamicsModel ·
│       │                     NeuralODE · ODEModel · LatentRNN / LatentLSTM
│       ├── neighborhood/     ★ the method: AdjacencyMatrix, NeighborhoodSegmentLoader,
│       │                     NeuralNeighborhoodFlow, NeighborhoodMSELoss,
│       │                     EfficientNeighborhoodMSELoss; radius_selection (r_min, r_max)
│       ├── loss_functions.py ★ MSELoss, JacobianMatchingMSE, TangentEvolutionMatchingMSE,
│       │                     PushforwardMatchingMSE, DySLIMLoss (the baselines)
│       ├── pushforward.py    ★ local pushforward-matrix estimation (LS / TLS)
│       ├── training/         BaseTrainer · VanillaTrainer · MultitermTrainer (ConFIG)
│       ├── nn/               InvertibleLinear · RNNCell
│       ├── io.py             save/load model (orbax and eqx)
│       ├── misc.py           load_experiment, calculate_lyapunov, solve_batch
│       └── metrics/error.py, preprocessing.py, utils/{array,tree}.py
├── lyapunov/                 empty stub ("Hello from lyapunov!") — removed in Phase 0
└── matplotlib_utils/         axes_setters, plots (line+band, violin, comparison, scatter)

scripts/
├── data_generation/generate_ode_data.py   Hydra; configs/system/{lorenz63, lorenz96_{6,18,54}d, hyperchen05}
├── training/train_baselines.py            Hydra; groups model/ data/ method/ training/ launcher/
├── analysis_lyapunov.py, analysis_sinkhorn.py   configs_analysis/
notebooks/   datasets/ · analysis/ · evaluation/ (per-system *_eval, metrics, sweeps) · visualization/ · isochrons/
```

★ = method core, Joon's (see `CLAUDE.md`, *AI use*).

## Data flow of one training run

`generate_ode_data.py` integrates a `dynamical_systems` ODE into HDF5 →
`train_baselines.py` loads it as a `TimeSeriesDataset`, standardizes, adds noise →
`cfg.method.loader_partial` builds the loader (for the Neighborhood method, a
`NeighborhoodSegmentLoader` that selects `r_min`/`r_max` and samples neighbor segments) →
`cfg.method.loss_fn` (a `NeighborhoodMSELoss` or a baseline) → `VanillaTrainer.train`
(optax; wandb config from the resolved Hydra config) → checkpoints under
`results/<name>/<subsavedir>/`. Evaluation currently lives in notebooks (`notebooks/evaluation/`).

## Known issues in the structure (addressed by the roadmap)

- **[code]** Metrics are split across `dynamical_systems.metrics` and
  `dynamics_discovery.metrics.error`; Lyapunov exponents are computed both in
  `dynamical_systems.analysis.lyapunov` and `dynamics_discovery.misc.calculate_lyapunov`.
- **[code]** `pushforward.py` keeps `estimate_pushforward_matrices_old` beside the current one.
- **[code]** The root `pyproject.toml` sets `testpaths = ["tests"]`, which does not exist; the
  one test lives in `packages/dynamical_systems/tests/`. No property or regression tests.
- **[code]** `jax[cuda12]` is a hard dependency of two library packages, so a CPU install
  (CI, laptops) is not expressible.
- **[code]** Two notebooks at the repository root (`prototype.ipynb`,
  `neuralode_lyapunov.ipynb`); 22 MB of notebook outputs tracked.
- **[deduced]** Evaluation logic (metrics, LE spectra, Sinkhorn, VPT) is defined per
  notebook; the paper's tables cannot be regenerated by a script. Phase 3.
