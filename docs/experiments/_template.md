---
type: experiment
status: template
updated: 2026-10-21
id: E-template
affects: (paper section / table / reviewer)
---

# `<slug>` — one line naming the question

Copy this file to `docs/experiments/<slug>.md`, set `status: planned`, give it the next `id`
(`E1`, `E2`, …) and list it in `docs/index.md`. Keep the sections; write "none" rather than
deleting one. Provenance labels: **[cited]**, **[code]**, **[measured]**, **[deduced]**.

## Question

The single question this experiment answers, phrased so that a reviewer would recognize it.

## Hypothesis

What we expect and *why* (the mechanism, with a reference or an argument), and what result
would falsify it. Written before the runs.

## Systems and data

| System | Dimension | Source / generator | Dataset file (hash) | Noise | Train / valid / test |
|---|---|---|---|---|---|

Real-world data: provenance (where obtained, license, preprocessing), and what "ground
truth" means for each metric when there is no known vector field.

## Configs

The exact Hydra invocations (script, config name, overrides), one per row. Anything not
expressible as an override is a code change and gets a branch and a change document.

## Runs

| Run | Config row | Seeds | Run directory | wandb group | Status |
|---|---|---|---|---|---|

Runs are named here when launched, not when they finish. Failed or discarded runs stay listed
with the reason.

## Result

Tables and figures as produced by `scripts/evaluate.py` / `scripts/figures/` (paths under
`results/` or `figures/`), with the metric rows of `docs/metrics.md` they use. Numbers copied
from a notebook are marked **[tentative]**.

## Conclusion

Two to five sentences: what the result says about the hypothesis, what it does not say, and
the one-line form that goes into `docs/results.md`. `verified_by: joon` in the frontmatter
once Joon has signed it; until then the conclusion is `tentative`.

## Affects

Paper sections, tables, reviewer points, later experiments. Anything here that changes a
reported number is also a row in `docs/findings.md`.

## Log

Dated notes as the experiment runs (what was changed and why). Newest at the bottom.
