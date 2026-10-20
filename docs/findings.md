---
type: findings
status: current
updated: 2026-10-21
sources: [joon, claude]
---

# Findings ledger

Append-only. Anything discovered while working that bears on a *reported result* (paper
tables, datasets, claims, the rebuttal) or constrains a later phase. Each row names its
evidence and the action taken. Items that are only cleanup go in the roadmap, not here.

| # | Date | Finding | Evidence | Affects | Action / status |
|---|---|---|---|---|---|
| F1 | 2026-10-21 | Lorenz96 (18D) runs from the NeurIPS round showed a low-noise win for the Neighborhood method that was not used in either submission. | Joon (stated); runs to be located on disk | the dimensionality question (reviewer Hd1Z); experiment 2.1 | Becomes the first result of `experiments/lorenz96-dimension.md`; run directories recorded there. |
| F2 | 2026-10-21 | The double-JVP identity used for the nested forward-mode AD (submission Eq. 7) is due to Nzoyem et al. (ICLR 2025) and is not cited. | rebuttal drafts (NeurIPS round) | the paper's related-work and method sections | Citation added in the revised paper (Phase 3). |
| F3 | 2026-10-21 | JAX's GPU preallocation makes wandb system-memory metrics unreliable; peak memory must be read from `jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]`. | Joon (stated); NeurIPS-round benchmarking | Table 9 (FLOPs and peak memory) and any cost claim in the rebuttal | Experiment 2.3 measures memory this way; `metrics.md` row M10. |
| F4 | 2026-10-21 | Evaluation metrics are defined per notebook (`notebooks/evaluation/*`) and the paper's tables were assembled by hand; no script regenerates them. | `architecture.md` [deduced]; `paper-map.md` | reproducibility of Tables 1, 4–8, 11 | Phase 1.3 (`evaluation/`), Phase 0.8 (`paper-map.md`); any discrepancy found when regenerating gets its own row here. |
