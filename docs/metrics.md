---
type: metrics
status: current
updated: 2026-10-21
sources: [code, claude]
verified_by: pending (Joon: definitions against the paper)
---

# Metrics matrix

Every quantity the paper reports, with where it is defined, where it is implemented, what
proves the implementation, and the status. The counterpart of a verification matrix for a
numerics code: here the oracle problem is the nonlinear time-series literature (Lyapunov
spectra, invariant measures) rather than convergence orders. A metric with no green row is
unverified; a metric the paper reports that is not here is a finding.

Implementation column is the state at `3b12b0d` **[code]**; the Phase 1.3 target is one
`evaluation/` module. Tier: `property` (fast, every commit) or `slow`.

| # | Metric | Paper | Implementation now | Test / oracle | Tier | Status |
|---|---|---|---|---|---|---|
| M1 | Valid prediction time (VPT) | Table 4, Fig. 2 | `dynamical_systems.metrics.pointwise.valid_prediction_time` | on an exact model, VPT = horizon; threshold monotone in the tolerance | property | planned |
| M2 | MMD² between ground-truth and model-evolved attractor samples at $\bar t$ | Table 5, Fig. 3, Fig. 8 | `dynamical_systems.metrics.measure_distances.maximum_mean_discrepancy` | MMD²(X, X) = 0; matches a NumPy reference on a small set; the baseline value of Fig. 8 reproduced | property | planned |
| M3 | Sinkhorn divergence | Table 11 | `…measure_distances.sinkhorn_divergence` (ott-jax) | zero on identical clouds; symmetric; against `ott` reference values | property | planned |
| M4 | Lyapunov spectrum of the trained model | Table 1, Fig. 5 | `dynamical_systems.analysis.lyapunov` (`lyapunov_gr`); duplicate in `dynamics_discovery.misc.calculate_lyapunov` | Lorenz63 reference spectrum (0.906, 0, −14.57) [cited: Sprott 2003]; Lorenz96 F = 8 literature values; sum of exponents = mean divergence | slow | planned |
| M5 | Average log relative Jacobian error over the attractor | Table 6, Fig. 4 | `dynamical_systems.analysis.jacobian` + notebook code | exact model → 0; against `jax.jacfwd` of the true rhs | property | planned |
| M6 | Average log relative vector-field error over the attractor | Table 7, Fig. 4 | notebook code | exact model → 0 | property | planned |
| M7 | Relative error of the time-$\Delta t$ map Jacobian $\partial\phi_{\Delta t}$ | Table 8 | `dynamical_systems.analysis.jacobian.one_step_jacobian`, `empirical_one_step_jacobian` | linear system: closed form $e^{A\Delta t}$ | property | planned |
| M8 | Spectral Hellinger distance | appendix (to confirm) | `dynamical_systems.metrics.statistical` | identical spectra → 0; the one existing test (`test_gaussian_filter1d`) | property | one test |
| M9 | FLOPs of the Neighborhood loss relative to a vanilla Neural ODE step | Table 9 | NeurIPS-round benchmark script (to locate) | synthetic Gaussian point clouds; ratio normalization | slow | planned |
| M10 | Peak device memory | Table 9 | — | `memory_stats()["peak_bytes_in_use"]`, not wandb (F3) | slow | planned |
| M11 | Training wall time | Table 10 | wandb | recorded per run, not tested | — | — |

## Notes

- Fractal dimension (`notebooks/analysis/fractal_dimension.ipynb`) and the characteristic
  time (`analysis.characteristic_time`) are analysis tools, not reported metrics; they get a
  row only if the revised paper reports them.
- Tolerances for M4 are set by the integration length and recorded here when measured.
