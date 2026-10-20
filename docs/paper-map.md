---
type: paper-map
status: current
updated: 2026-10-21
sources: [paper, claude]
verified_by: pending (Joon fills the run columns; Phase 0.8)
---

# Paper map — ICLR 2027 submission

Every figure and table of the submission (`ICLR2027 submission.pdf`) traced to what produced
it: the script or notebook, the config group values, the dataset file, the run directories or
wandb group, and the metric rows in `metrics.md`. The goal of Phase 0.8 is no `?` left; the goal
of Phase 1 is that the *Regenerate* column is a command.

Captions are abbreviated; systems are the three of the paper (Lorenz63, Lorenz96, hyperchaotic
Chen) unless the row says otherwise.

| Item | What it shows | Produced by (now) | Config / data | Runs | Metrics | Regenerate (target) |
|---|---|---|---|---|---|---|
| Fig. 1 | schematic of the Neighborhood method | drawn | — | — | — | — |
| Fig. 2 | short-term accuracy: data vs model predictions | `notebooks/evaluation/<system>_eval.ipynb` ? | `data=*`, all methods | ? | M1 | `scripts/figures/short_term.py` |
| Fig. 3 | long-term statistical accuracy: 2D histograms of time-evolved samples | `notebooks/evaluation/long_term.ipynb` ? | ? | ? | M2 | `scripts/figures/long_term.py` |
| Fig. 4 | local behavior: relative error of vector field and Jacobian over the attractor | `notebooks/evaluation/jacobian_error.ipynb` ? | ? | ? | M5, M6 | — |
| Fig. 5 | Lyapunov exponents of the trained models | `notebooks/evaluation/lyapunov_exponents.ipynb`, `scripts/analysis_lyapunov.py` | `configs_analysis/config_lyapunov.yaml` | ? | M4 | `scripts/evaluate.py --metric lyapunov` |
| Table 1 | mean Lyapunov spectra (three seeds), hyperchaotic Chen | `notebooks/evaluation/hyperchen05_eval.ipynb` ? | `data=hyperchen05` | ? | M4 | — |
| Fig. 6 | DySLIM hyperparameter sweep | `notebooks/evaluation/sweep.ipynb` ? | `method=dyslim` multirun | ? | — | — |
| Fig. 7 | Neighborhood hyperparameter sweep | `notebooks/evaluation/hyperparameter_selection.ipynb` ? | `method=neighborhood` multirun (`num_neighbors`, `weight`) | ? | — | — |
| Fig. 8 | baseline MMD² for hyperchaotic Chen | `notebooks/evaluation/metrics.ipynb` ? | ? | ? | M2 | — |
| Table 2 | ablation of $r_{\min} = c\tilde\epsilon$, Lorenz63 | ? | `method.loader_partial.r_min_kwargs.multiplier` | ? | M1, M2 | — |
| Table 3 | ablation of $r_{\max}$ (annulus with $p\%$ of the data) | ? | `…r_max_kwargs.target_neighborhood_rel_size` | ? | M1, M2 | — |
| Table 4 | VPT, Jacobian-oracle baseline vs methods | ? | `method=jacobian_matching` ? (not in `configs/method/`) | ? | M1 | — |
| Table 5 | MMD² at $\bar t = 100$, oracle vs methods | ? | as Table 4 | ? | M2 | — |
| Table 6 | average log relative Jacobian error, oracle vs methods | `notebooks/evaluation/jacobian_error.ipynb` ? | as Table 4 | ? | M5 | — |
| Table 7 | average log relative vector-field error, oracle vs methods | as Table 6 | as Table 4 | ? | M6 | — |
| Table 8 | relative error of $\partial\phi_{\Delta t}$ for the local quadratic estimator | `notebooks/analysis/tangent_evolution.ipynb` ? | — | — | M7 | — |
| Table 9 | FLOPs and peak memory of the Neighborhood evaluation relative to vanilla | NeurIPS-round benchmark script ? | synthetic point clouds | — | M9, M10 | — |
| Table 10 | training wall time (three seeds, V100) | wandb | all | ? | M11 | — |
| Table 11 | Sinkhorn divergence at $\bar t$ | `notebooks/evaluation/sinkhorn.ipynb`, `scripts/analysis_sinkhorn.py` | `configs_analysis/config_sinkhorn.yaml` | ? | M3 | `scripts/evaluate.py --metric sinkhorn` |

## Open questions for Phase 0.8

- The Jacobian-oracle baseline (Tables 4–7) has no `configs/method/*.yaml`; which loss
  (`JacobianMatchingMSE`?) and which runs produced it?
- `configs/training/multiterm.yaml` (ConFIG) — used in any reported run, or an unreported
  experiment?
- Rössler76 and Rössler79 are in `continuous/` and `data/rossler76.yaml` but not in the
  paper; keep as an unreported system or delete (Principle 5)?
