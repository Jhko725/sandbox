# neighborwatch

Code for *Watch Your Neighbors* — training Neural ODEs on chaotic time series with a
neighborhood-based loss that learns the local Jacobian structure without observing it
(submitted to ICLR 2027). The repository holds the ground-truth systems and metrics
(`packages/dynamical_systems`), the models, losses and training loop
(`packages/dynamics_discovery`), the Hydra scripts that generate data and train
(`scripts/`), and the notebooks used for evaluation.

## Install

```sh
uv sync                # CPU (adds the workspace packages under packages/)
uv sync --extra cuda   # GPU (from Phase 0; until then jax[cuda12] is a hard dependency)
uv run pytest          # fast tiers; see docs/decisions/0001-test-tiers-and-freeze.md
```

## Run

```sh
uv run python scripts/data_generation/generate_ode_data.py system=lorenz63
uv run python scripts/training/train_baselines.py data=lorenz63 method=neighborhood
```

## Documentation

`docs/index.md` is the entry point: architecture, roadmap, decision records, the results
and findings ledgers, and the paper map. `CLAUDE.md` describes how the repository is
worked on together with Claude, including which parts of the code are written by whom.
