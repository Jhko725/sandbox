---
type: change
status: planned
updated: 2026-10-21
branch: phase0-freeze
roadmap: Phase 0 (0.2–0.9)
---

# 2026-10-21 — `phase0-freeze` (planned)

Phase 0 of `docs/roadmap.md` after 0.1: freeze the submission state, build the regression
safety net from the checkpoints on disk, and clean the repository's hygiene before any
refactoring. Joon's items (0.2 tags and rename, 0.3 tags, 0.6 naming the checkpoints, 0.8 the
run columns of `paper-map.md`); the rest Claude's. This document is the plan of record for
the branch and is filled in as items land.

## Summary

Tag `iclr2027-submission` at `3b12b0d` and rename to `neighborwatch`; move the two root
notebooks under `notebooks/` and strip outputs with a git filter; remove the empty `lyapunov`
package and make CUDA an optional extra; dev tooling (ruff format in a `style:` commit, ty,
hypothesis, prek), pytest paths and markers; the regression tier (ADR-0001) with CPU goldens;
CPU CI; the paper map filled; a config census. Nothing in `src/` changes behavior.

## Files (planned)

- `pyproject.toml` — name `neighborwatch`; optional extra `cuda = ["jax[cuda12]"]`; dev group
  (ruff, ty, hypothesis, pytest-xdist, prek, nbstripout); `[tool.pytest.ini_options]`
  `testpaths = ["tests", "packages/*/tests"]`, markers, `addopts = "-m 'not regression and not slow'"`;
  `[tool.ruff]` format settings. `uv.lock` relocked (Joon: `uv lock`).
- `packages/lyapunov/` — deleted. `packages/*/pyproject.toml` — descriptions; `jax` without
  the `cuda12` extra; the stray `dynamics-discovery` script entry removed.
- `.gitattributes` — nbstripout filter. `notebooks/prototyping/{prototype,neuralode_lyapunov}.ipynb` — moved from the root.
- `prek.toml` — ruff check `--fix`, ruff format, whitespace fixers, nbstripout.
- `.github/workflows/ci.yml` — ruff, ty, `check_docs.py`, `pytest` fast tiers on CPU.
- `tests/conftest.py` — `--checkpoints-root`, hypothesis profiles (`dev` 10, default 50, `ci` 200 derandomized).
- `tests/regression/{cases.py,test_regression.py,goldens/*.npz}` — one case per (system, method) checkpoint (Joon names them).
- `tests/unit/`, `tests/property/` — the first tests (dataset round-trip, loader radius invariants, metric zero/symmetry).
- `scripts/check_configs.py` — instantiates every config under `scripts/**/configs`; census to `docs/config-census-phase0.txt`.
- `docs/paper-map.md` — run columns filled. `docs/architecture.md`, `docs/roadmap.md`, `docs/index.md` — updated.

## Steps

| # | Step | Who | Done when |
|---|---|---|---|
| 1 | `git tag iclr2027-submission 3b12b0d && git push origin iclr2027-submission`; rename the GitHub repository | Joon | tag on origin; clone URL is `neighborwatch` |
| 2 | Notebooks moved; `nbstripout --install`; `git tag pre-nbstripout`; outputs stripped in one commit | Claude (filter, strip), Joon (tag) | `git ls-files notebooks \| xargs du -ch` < 2 MB |
| 3 | `lyapunov` removed; metadata; CUDA extra; relock | Claude, Joon relocks | `uv sync` and `uv sync --extra cuda` resolve |
| 4 | Tooling; `style:` commit isolating the reformat | Claude | hooks run |
| 5 | Regression cases: Joon lists checkpoint paths → Claude writes `cases.py`, generates goldens on CPU, tests green twice | both | `tests/regression` green ×2 |
| 6 | CI | Claude | green on first push |
| 7 | Paper map | both | no `?` cells |
| 8 | Config census | Claude | 0 failures |

## Design

- (to be written as decisions arise; the structural ones are ADR-0001 already)

## Bugs fixed

- (none yet)

## Findings

- (none yet)

## Tests

- (filled in with step 5)

## Open issues

- Whether to keep Rössler76/79 (`paper-map.md`, open questions) — Joon.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| (rows added as files land) | | |
