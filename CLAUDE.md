# Working conventions for Claude in this repository

This file is read at the start of every session. It says how work is recorded so that
knowledge survives between sessions and between Joon and Claude, and what Claude may and may
not write here. The documents it refers to live under `docs/`; `docs/index.md` lists them all.

## What this repository is

The code behind the paper *Watch Your Neighbors* (the Neighborhood method: training Neural
ODEs on chaotic time series with a neighborhood-based loss), submitted to ICLR 2027. The
repository is being renamed `neighborwatch`. Its work falls into two kinds, recorded
differently:

- **Infrastructure** — packaging, tests, data pipelines, evaluation code, docs: branches with
  change documents (below).
- **Experiments** — runs that answer a question for the paper or the rebuttal: experiment
  documents under `docs/experiments/` and the `docs/results.md` ledger (ADR-0002).

## AI use (the boundary)

The paper's stated policy was that its code and drafts were written without AI. From the
rebuttal on, the policy is the split below; the revised paper will carry an AI-use section
describing it, so the split must be kept true in the history.

- **Claude writes**: infrastructure — packaging and tooling, tests, data generation and
  loading, evaluation and analysis scripts, figures from recorded results, documentation.
- **Joon writes**: the method core — `dynamics_discovery/neighborhood/`,
  `dynamics_discovery/loss_functions.py`, `dynamics_discovery/pushforward.py` — and every
  claim that appears in the paper. Claude's role on these is review, and refactoring *after*
  Joon has a working prototype, in a branch Joon reviews file by file.
- **Analysis of results**: Claude may compute, tabulate and plot; conclusions recorded in
  `docs/results.md` and `docs/findings.md` are Joon's (`verified_by: joon`) or marked
  `tentative` until they are.
- A commit that touches the method core and is authored by Claude says so in its change
  document's *Design* section, with the prototype it refactors.

## Start of a session

1. Read `docs/index.md`, then `docs/architecture.md` (what the code is now), then the current
   phase in `docs/roadmap.md` (what is being done and why).
2. Find the branch's change document `docs/changes/<YYYY-MM-DD>-<branch>.md`. Its **Review
   notes** table is the current state of the review: rows with *Thoughts* or *Modifications*
   filled in by Joon are instructions for this session. Read them before touching code.
3. Check `docs/findings.md` for anything that constrains the task. For experiment work, read
   the experiment document and `docs/results.md`.

## Change documents

Every branch has a companion document `docs/changes/<YYYY-MM-DD>-<branch>.md`, committed with
the code. It is the reviewer's map of the change, not a changelog, and it is also the *plan*
for the branch: written before the work (`status: planned`), updated as the work lands
(`in-progress`, `landed`), closed at merge (`merged`). Sections:

- **Summary** — what the branch does and why, in a few sentences; which roadmap items.
- **Files** — every file touched, one line each. Must match `git diff --stat master...HEAD`
  (names); `scripts/check_docs.py --files` checks this. Planned files are marked *(planned)*.
- **Design** — decisions not obvious from the code, each with the rejected alternative and
  why; link to an ADR in `docs/decisions/` when the decision is structural.
- **Bugs fixed** — pre-existing bugs found on the way, with the symptom.
- **Findings** — anything that bears on a *reported result* or on later phases; also appended
  to `docs/findings.md`.
- **Tests** — what the new tests pin down and how to run them (tier, markers, env vars).
- **Open issues** — things deliberately left alone, with the reason.
- **Review notes** — a table *Change* | *Thoughts* | *Modifications*. Claude pre-populates one
  row per file touched (*Change* = file + one-line summary) and leaves the other two columns
  blank. Joon fills them in while vetting. Claude never edits *Thoughts* or *Modifications*
  except when asked to, and treats them as the current state of the review. Later rounds are
  appended as dated sections (`## Review round N (date)`), with the decisions attributed.

Claude updates the change document in the same commit as any code change on the branch.

**Growth rule.** A change document closes when its branch has carried more than one roadmap
item *or* more than two review rounds: the branch is merged (or split) and the next item gets
a new branch and document. ADR numbers are reserved in the roadmap table when a branch is
cut, so parallel branches do not collide.

## Experiment documents

One document per experimental question, `docs/experiments/<slug>.md`, from
`docs/experiments/_template.md`: *Question · Hypothesis · Systems and data · Configs · Runs ·
Result · Conclusion · Affects*. Status `planned → running → concluded` (or `superseded`).
Concluded experiments get one row in `docs/results.md`; anything that changes a reported
number also goes to `docs/findings.md`. Runs are identified by run directory and wandb
group, never by a timestamp copied into a notebook. Figures in the paper are traced in
`docs/paper-map.md`.

## End of a session (or of any commit that changes structure)

- Change document updated (Files, Design, Tests, Review-notes rows for new files).
- `docs/roadmap.md`: finished items moved to the **Done** ledger (append at the end, never
  insert, never delete); phase tables kept *correct now*.
- `docs/architecture.md` updated if a module moved, was renamed, added or removed.
- `docs/findings.md` appended if anything concerns a reported result; `docs/results.md`
  if an experiment concluded; `docs/metrics.md` if a metric gained or lost a test.
- `uv run python scripts/check_docs.py` passes.

## Repository facts

- Python 3.13, a `uv` workspace (`[tool.uv.workspace]`): `uv sync` installs the root and the
  packages under `packages/`; tests with `uv run pytest`. Training scripts enable
  `jax_enable_x64`. GPU: `jax[cuda12]` (to become an optional extra in Phase 0).
- Stack: JAX, equinox, diffrax, optax, orbax-checkpoint, hydra (+ submitit launcher), wandb,
  ott-jax, xarray/h5netcdf, scipy.
- Experiments are Hydra scripts under `scripts/`: `data_generation/generate_ode_data.py`
  (configs in `system/`), `training/train_baselines.py` (groups `model/`, `data/`, `method/`,
  `training/`, `launcher/`), `analysis_*.py` with `configs_analysis/`. Trained runs land
  under `results/` (git-ignored); the submission's checkpoints are kept on disk and are the
  source of the regression goldens (ADR-0001).
- Test tiers (ADR-0001): `tests/unit` and `tests/property` on every commit; `tests/regression`
  needs the submission checkpoints and runs on CPU (`JAX_PLATFORMS=cpu`); `-m slow` for
  anything that trains. Regression goldens are never regenerated without a change-document
  entry saying which commit changed behavior and why.
- Notebook outputs are stripped by a git filter (`nbstripout --install` once per clone, from
  Phase 0.3). Paper figures are regenerated by scripts, not kept as outputs.
- Markdown must render on GitHub *and* in VS Code's preview. GitHub runs Markdown before
  MathJax, so: an inline formula `$…$` never crosses a line; display math only as a
  standalone block (`$$` alone on a line, formula, `$$` alone on a line), never `$$…$$`
  inside a paragraph; inside inline math no backslash + ASCII punctuation (`\,` `\;` `\{`
  `\|` `\\` → `\thinspace` `\medspace` `\lbrace` `\Vert`), no `*` (use `\ast`), spaces around
  `<` `>`, and no `_` right after `}` `)` `|` (write `\mathbf e_r`, `\vert_{E}`), because such an
  `_` opens Markdown emphasis.
- Every document under `docs/` starts with a frontmatter block (`type`, `status`, `updated`,
  plus `verified_by` / `sources` / `id` / `branch` where relevant). `status` ∈ {`current`,
  `planned`, `in-progress`, `landed`, `merged`, `running`, `concluded`, `superseded`,
  `agreed`, `accepted`, `amended`, `tentative`, `historical`, `template`}.
- Provenance labels in design and experiment documents: **[cited]** = from a named source,
  **[code]** = verified in the tree, **[measured]** = from a recorded run, **[deduced]** =
  Claude's inference. Keep them.
- American English throughout (code, docstrings, documents). Docstrings follow the
  equinox/diffrax convention (summary paragraph, `**Arguments:**`, `**Returns:**`); existing
  docstrings are migrated when their file is touched, not in a sweep.
- Jit and vmap: a function that is naturally jit- and vmap-compatible is left undecorated —
  the caller jits at the level they want; host-side drivers jit the subcomponents they call.
- `git log -p docs/roadmap.md` is the project diary; `git blame` dates every Done row.
- Pushes and tags from Claude's sessions are currently refused (no GitHub App on this
  repository); Claude hands Joon a patch, Joon pushes.
