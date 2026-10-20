---
type: change
status: landed
updated: 2026-10-21
branch: docs-conventions
roadmap: Phase 0 (0.1)
---

# 2026-10-21 — `docs-conventions`

The coworking system for this repository, derived from `deep_isochron` (the first version)
and `CONTEXT_FLUX_NO` (its revision), with what a paper-and-rebuttal repository needs on top:
experiment documents, a results ledger, a paper map, a metrics matrix, and a written AI-use
boundary. No code changes.

## Summary

`CLAUDE.md` (session protocol, change and experiment documents, the AI-use split, repository
facts), the `docs/` tree (index, architecture, roadmap, findings, results, metrics, paper-map,
experiment template), ADR-0001 (freeze + test tiers) and ADR-0002 (experiment documents),
`scripts/check_docs.py` ported from `CONTEXT_FLUX_NO`, a README, and the Phase 0 change
document as a plan. Roadmap item 0.1.

## Files

- `CLAUDE.md` — new.
- `README.md` — new (was empty): what the repository is, how to install, where the docs are.
- `docs/index.md`, `docs/architecture.md`, `docs/roadmap.md`, `docs/findings.md`,
  `docs/results.md`, `docs/metrics.md`, `docs/paper-map.md` — new.
- `docs/experiments/_template.md` — new.
- `docs/decisions/0001-test-tiers-and-freeze.md`, `docs/decisions/0002-experiment-documents.md` — new.
- `docs/changes/2026-10-21-docs-conventions.md` (this file), `docs/changes/2026-10-21-phase0-freeze.md` (planned) — new.
- `scripts/check_docs.py` — new; ported from `CONTEXT_FLUX_NO` with `master` as the base
  branch, the experiment statuses, and the `id`/`affects` requirement on experiment documents.

## Design

- **Base: the `CONTEXT_FLUX_NO` version, not `deep_isochron`'s.** It is the later one and adds
  the session protocol, the branch-plan role of change documents, `findings.md`, the
  per-item *Who* column and mechanical checking. *Rejected*: starting from `deep_isochron`'s
  `CLAUDE.md` and adding — the same content, less structure.
- **Experiment documents (ADR-0002).** Neither reference repo records experiments; here they
  are the main work. One file per question so that documents close (the `analysis` change
  document in `deep_isochron` has carried three review rounds and three roadmap items since
  2026-10-06 — the growth rule in `CLAUDE.md` comes from that).
- **`metrics.md` instead of `verification.md`.** The claims here are metric definitions with
  literature oracles, not convergence orders; the matrix shape is the same.
- **AI-use boundary in `CLAUDE.md`** (Joon, 2026-10-21: the split is correct; the revised paper
  will carry a new AI-use section). Method core = `neighborhood/`, `loss_functions.py`,
  `pushforward.py`; Claude reviews and refactors after a prototype.
- **Dropped from `CONTEXT_FLUX_NO`**: the `[C]/[M]/[R]` tracks (a deadline row in the roadmap
  does the job for one person and one paper), `glossary.md` (nothing to define yet),
  `verification.md` (replaced), the frozen assessment document (`architecture.md` carries the
  known issues; the tree is small enough).
- **`[measured]`** added to the provenance labels, for numbers from recorded runs.
- Package decisions deferred to ADR-0003 (Phase 1.1): `lyapunov` goes in Phase 0 (Joon); the
  monorepo idea is parked (Joon: new shared code grows in `deep_isochron`).

## Bugs fixed

- none (no code touched)

## Findings

- F1–F4 in `docs/findings.md`: the unused Lorenz96-18D result, the Nzoyem et al. citation
  gap, wandb memory metrics under JAX preallocation, and metrics defined per notebook.

## Tests

- `uv run python scripts/check_docs.py` — 0 problems. `--files` cannot run until the
  branch has a remote (`origin/master` base).

## Open issues

- The repository rename (`neighborwatch`) and the `pyproject.toml` name are Phase 0.2 (Joon;
  needs `uv lock`); documents already use the new name.
- `check_md_math.py` from `deep_isochron` is not ported; the Markdown rules are in
  `CLAUDE.md` and the few formulas here follow them. Port it when a design document appears.
- Pushes from Claude's sessions are refused for this repository (no GitHub App); this branch
  is delivered as a patch.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `CLAUDE.md`: AI-use boundary, session protocol, change + experiment documents, growth rule, repo facts | | |
| `README.md`: purpose, install, pointers | | |
| `docs/index.md`: the catalog and status vocabulary | | |
| `docs/architecture.md`: package map at `3b12b0d`, data flow, known issues | | |
| `docs/roadmap.md`: principles, deadline, Phases 0–3, Parked, Done | | |
| `docs/findings.md`: F1–F4 | | |
| `docs/results.md`: empty ledger | | |
| `docs/metrics.md`: M1–M11 against the paper's figures and tables | | |
| `docs/paper-map.md`: every figure/table → producer, with `?` where Joon fills | | |
| `docs/experiments/_template.md`: the experiment document shape | | |
| `docs/decisions/0001-test-tiers-and-freeze.md` | | |
| `docs/decisions/0002-experiment-documents.md` | | |
| `docs/changes/2026-10-21-phase0-freeze.md`: Phase 0 as a plan with a Who column | | |
| `scripts/check_docs.py`: ported; `master` base; experiment statuses and fields | | |
