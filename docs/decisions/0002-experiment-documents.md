---
type: decision
id: ADR-0002
status: accepted
updated: 2026-10-21
verified_by: pending (Joon; proposed by Claude 2026-10-21)
---

# ADR-0002 — Experiments are documents: question-level files, a results ledger, a paper map

**Status**: accepted pending review (branch `docs-conventions`).

## Context

`deep_isochron` and `CONTEXT_FLUX_NO` record *code* changes well (change documents, ADRs,
findings) but have no convention for *experiments*: which runs answered which question, with
which configs, and what was concluded. In this repository that is the main work ahead — the
rebuttal is a set of experiments — and the NeurIPS round already produced a result (Lorenz96
18D, low noise) that was never used because it had no home. Checkpoint selection in the
evaluation notebooks is by timestamp dictionaries, so a table cannot be traced to its runs.

## Decision

1. **One document per experimental question**, `docs/experiments/<slug>.md`, from the
   template: Question, Hypothesis (written before the runs, with what would falsify it),
   Systems and data, Configs (exact Hydra invocations), Runs (named at launch, with run
   directory and wandb group; discarded runs kept with the reason), Result, Conclusion,
   Affects, Log. Status `planned → running → concluded`, or `superseded` by a later document.
2. **`docs/results.md`** is the append-only ledger of concluded experiments: one row with the
   headline number and `verified_by`. The rebuttal and the revised paper are written from it.
3. **`docs/paper-map.md`** traces every figure and table of the submission to script, config,
   dataset, runs and metric rows; its *Regenerate* column becomes a command as Phase 1 lands.
4. **Provenance of numbers**: `[measured]` means produced by a script from a run named in the
   document; `[tentative]` means read off a notebook. Conclusions are Joon's: `tentative`
   until `verified_by: joon`.
5. A run with no document is a prototype. It may be launched, but its numbers do not enter
   `results.md` until a document exists.

## Rejected alternatives

- *wandb as the record* — holds runs, not questions or conclusions; not reviewable in git;
  the NeurIPS-round finding was in wandb and still got lost.
- *One experiments page* — becomes the `analysis` change document of `deep_isochron`: a
  single file carrying unrelated rounds. Per-question files close.
- *Notebooks as the record* — outputs are stripped (Phase 0.3) and metric code is being
  moved out of them (Phase 1.3).

## Consequences

- `docs/index.md` lists experiment documents; `check_docs.py` accepts the experiment statuses
  and requires `id` and `affects` on `type: experiment`.
- Phase 1.4 (run identity: `run.json`, `runs.index`) exists so that the Runs table can be
  filled from disk rather than by hand.
- `docs/findings.md` keeps its role: a finding that changes a *reported* number goes there as
  well, with the experiment as evidence.
