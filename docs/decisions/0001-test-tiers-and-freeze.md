---
type: decision
id: ADR-0001
status: accepted
updated: 2026-10-21
verified_by: pending (Joon; proposed by Claude 2026-10-21 following CONTEXT_FLUX_NO ADR-0008)
---

# ADR-0001 — Freeze tag and four test tiers; regression goldens from the submission checkpoints

**Status**: accepted pending review (branch `docs-conventions`).

## Context

The repository has one unit test. The code that produced the ICLR 2027 submission is about to
be reorganized (packages consolidated, evaluation moved out of notebooks, dependencies
split) while new experiments for the rebuttal are launched from it. Without a safety net,
a refactor that changes a number silently invalidates a reported result. The submission's
trained checkpoints exist on disk (Joon, 2026-10-21), which makes a regression tier cheap.

## Decision

1. **Freeze.** Tag `iclr2027-submission` at `3b12b0d` (the tree the paper was produced from).
   The tag is the archive; nothing is kept in the tree for the paper's sake.
2. **Four tiers.**
   - `tests/unit` — examples, shapes, error paths; every commit.
   - `tests/property` — Hypothesis: what must hold for all inputs (a metric is zero on
     identical inputs and symmetric; `vmap` equals a Python loop; a bijection round-trips;
     the loader's neighbor indices are within `r_max` and outside `r_min`; a dataset
     round-trips through HDF5); every commit.
   - `tests/regression` — for each (system, method) pair with a submission checkpoint: the
     vector field on a fixed batch, the Jacobian on a few points, a short rollout, compared
     to goldens in `tests/regression/goldens/*.npz` at `rtol 1e-5, atol 1e-6`; **CPU only**
     (`JAX_PLATFORMS=cpu`) for determinism; `-m regression`, needs `--checkpoints-root`.
   - `-m slow` — anything that trains or integrates long (an end-to-end 20-step run; a
     Lyapunov spectrum against literature values).
3. **Goldens are never regenerated without a change-document entry** naming the commit that
   changed behavior and why; the entry is also a row in `docs/findings.md` if a reported
   number moves.
4. CI runs the two fast tiers on CPU; the regression tier runs locally (the checkpoints are
   not in the repository) and before every merge to `master`.

## Rejected alternatives

- *Unit tests only* — cannot express the invariants the Neighborhood loader and the metrics
  must satisfy.
- *Regression on GPU* — nondeterministic reductions; the tolerance would have to be loose
  enough to hide real changes.
- *Re-training as the regression test* — hours, and seeds do not reproduce across JAX
  versions.

## Consequences

- Phase 0.5–0.6 of the roadmap. `pytest` markers `regression`, `slow`; `addopts` deselects
  both by default.
- JAX practicalities for Hypothesis: shapes from a small fixed set (retracing),
  `deadline=None`, bounded finite floats, tolerance-based assertions, `derandomize=True` in
  CI, shrunk counterexamples pinned with `@example`.
- `docs/metrics.md` maps each reported metric to its tier and oracle.
