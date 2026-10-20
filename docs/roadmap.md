---
type: roadmap
status: current
updated: 2026-10-21
sources: [joon, claude]
---

# Roadmap

The single current plan for `neighborwatch`. Kept *correct now*: finished items move to the
**Done** ledger at the bottom (append-only, chronological, one row per item, linking the
branch's change document or experiment document); parked work is listed with its reason.
Updated in the same commit as the work that changes it. History: `git log -p docs/roadmap.md`.

Decisions are ADRs in `decisions/`; branch-level detail in `changes/`; experiments in
`experiments/` with the `results.md` ledger.

## Principles

1. The ICLR 2027 submission state is the archive (tag `iclr2027-submission` at `3b12b0d`); the
   repository is free to change behind it.
2. Refactor only behind the regression tier built from the submission checkpoints (ADR-0001).
3. The method core is Joon's; Claude builds the infrastructure around it (`CLAUDE.md`, *AI use*).
4. Experiments are documents before they are runs: a question, a hypothesis and an *Affects*
   line come first (ADR-0002). A run without a document is a prototype, not a result.
5. Delete by default; no `legacy/`, no `_old` suffixes. Nothing is kept "for the paper" that the
   paper map does not point to.
6. Every number in the paper or the rebuttal resolves to a command (`paper-map.md`).

## Deadlines

| What | When | Needs |
|---|---|---|
| ICLR 2027 rebuttal | reviews expected ~mid-November 2026 (to confirm) | Phase 2 experiments concluded and in `results.md` |

## Phase 0 — Freeze, hygiene, docs (current)

Branch `docs-conventions` carries 0.1 (this system); branch `phase0-freeze` the rest
(`changes/2026-10-21-phase0-freeze.md`, planned). *Who* follows `CLAUDE.md`.

| # | Item | Done when | Who | Status |
|---|---|---|---|---|
| 0.1 | `CLAUDE.md`, `docs/` tree, `scripts/check_docs.py`, ADR-0001/0002, templates | `check_docs.py` passes | Claude | landed (`docs-conventions`) |
| 0.2 | Tag `iclr2027-submission` at `3b12b0d`; rename repository and root project to `neighborwatch` (`pyproject.toml` name, `uv lock`) | tag on origin; `uv sync` clean | Joon | |
| 0.3 | Move root notebooks into `notebooks/`; `nbstripout` git filter; strip outputs; tag `pre-nbstripout` | `notebooks/` tracked size < 2 MB | Claude (filter), Joon (tags) | |
| 0.4 | Remove `packages/lyapunov`; fix package metadata (descriptions, `dynamics-discovery` stray script entry); `jax[cuda12]` → root optional extra `cuda`, libraries depend on `jax` only | `uv sync` (CPU) and `uv sync --extra cuda` both resolve | Claude | |
| 0.5 | Dev tooling: `ruff format` + a `style:` commit, `ty`, hypothesis, `prek` hooks; pytest `testpaths` fixed to `tests/`; markers `regression`, `slow` | hooks run; `uv run pytest` finds the tests | Claude | |
| 0.6 | Regression harness (ADR-0001): one case per (system, method) with a submission checkpoint — forward pass of the vector field on a fixed batch, a short rollout, the Jacobian on a few points — CPU goldens in `tests/regression/goldens/` | `JAX_PLATFORMS=cpu uv run pytest tests/regression` green twice | Claude builds; Joon names the checkpoints | |
| 0.7 | CPU CI: ruff, ty, `check_docs.py`, fast tiers | green on the first push | Claude | |
| 0.8 | `paper-map.md` filled for every figure and table of the submission | no `?` cells left | Joon (which run), Claude (which script/config) | |
| 0.9 | `scripts/check_configs.py`: every Hydra config instantiates | 0 failures | Claude | |

Exit: fast tiers and regression tier green on CPU; `train_baselines.py data=lorenz63
method=neighborhood training.num_steps=20` runs and checkpoints on CPU; `paper-map.md` complete.

## Phase 1 — Prune and consolidate

Branch `phase1-prune`. ADR-0003 reserved: one package or three.

| # | Item | Done when | Who |
|---|---|---|---|
| 1.1 | Decide the package layout (ADR-0003): fold `matplotlib_utils` and `dynamical_systems` into `dynamics_discovery` (renamed `neighborwatch`?) vs keep the workspace. Input: the monorepo-across-projects idea is parked; `deep_isochron` is where shared code now grows | ADR accepted | Joon decides |
| 1.2 | Dead and duplicate code: `estimate_pushforward_matrices_old`; `misc.calculate_lyapunov` vs `analysis.lyapunov`; the two `metrics` modules merged; `io.py` down to one save/load path | `ruff check` clean; regression unchanged | Claude; method-core files reviewed by Joon |
| 1.3 | Evaluation out of notebooks: `evaluation/` with `rollout`, `metrics` (VPT, MMD², Sinkhorn, LE spectrum, Jacobian and vector-field relative error, spectral Hellinger), `scripts/evaluate.py` over a run directory | no `def` of a metric in any notebook; `metrics.md` rows green | Claude |
| 1.4 | Run identity: a `run.json` (config, git SHA, dataset hash, checkpoint, best step) written by the trainer; `runs.index(root)` for notebooks | `notebooks/evaluation/*` select runs by id, not timestamp | Claude |

Exit: `evaluate.py` regenerates Tables 1, 4–8 and 11 of the submission from the checkpoints
on disk (within tolerance recorded in `paper-map.md`).

## Phase 2 — Experiments for the rebuttal (the point)

Each item is an experiment document (`experiments/<slug>.md`); the roadmap row only names it.
Runs may start before Phase 1 closes if they only need `train_baselines.py` as it is; their
evaluation then waits for 1.3 or is done in a notebook and marked `tentative`.

| # | Experiment | Reviewer liability it answers | Status |
|---|---|---|---|
| 2.1 | `lorenz96-dimension` — Lorenz96 at 6, 18, 54 dimensions; the unused 18D low-noise finding from the NeurIPS round goes in first | dimensionality (Hd1Z) | planned |
| 2.2 | `real-world-systems` — systems and datasets to be named by Joon | "only synthetic systems" | planned |
| 2.3 | `neighbor-search-scaling` — KD-tree vs chunked `cdist` vs FAISS as dimension grows; wall time and memory | computational cost | planned |
| 2.4 | `oracle-gap` — the Jacobian-oracle baseline (Tables 4–7) on the new systems | motivation (WQsw) | planned |

## Phase 3 — Figures and the revised paper

| # | Item | Done when |
|---|---|---|
| 3.1 | `scripts/figures/` regenerates every figure of the paper from `results/` | `paper-map.md` lists a script for every figure |
| 3.2 | Revised AI-use section of the paper drafted from `CLAUDE.md` and the git history | Joon's text |

## Parked (with reason)

- **Monorepo across projects** (the original purpose of `sandbox`) — unclear whether it
  continues; new shared code is being written in `deep_isochron`. Revisit in Phase 1.1.
- **Typed (dataclass) Hydra configs** — not while the config surface is still changing.
- **Multi-device training** — single GPU assumed.

---

## Done

| When | Branch | What | Record |
|---|---|---|---|
| 2026-10-21 | `docs-conventions` | 0.1: `CLAUDE.md` (AI-use boundary, session protocol, change and experiment documents, repo facts), `docs/` tree (index, architecture, roadmap, findings, results, metrics, paper-map, experiment template), ADR-0001/0002, `scripts/check_docs.py`, Phase 0 change document planned | `changes/2026-10-21-docs-conventions.md` |
