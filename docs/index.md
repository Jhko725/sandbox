---
type: index
status: current
updated: 2026-10-21
---

# `neighborwatch` documentation — index

Progressive disclosure: start here, read only what the task needs. Every document carries a
frontmatter block (`type`, `status`, `updated`, plus `verified_by` / `sources` / `id` / `branch`
where relevant) so its standing is machine-readable. `scripts/check_docs.py` enforces that every
document is listed here and has frontmatter.

`status` values — `current` (kept true at all times) · `planned` / `in-progress` / `landed` /
`merged` (change documents) · `planned` / `running` / `concluded` / `superseded` (experiment
documents) · `agreed` (design settled with Joon) · `accepted` / `amended` (decisions) ·
`tentative` (written by Claude, not yet checked by Joon) · `historical` (frozen) · `template`.

## Read first

| Document | Type | What it answers |
|---|---|---|
| [`architecture.md`](architecture.md) | architecture | what the code is now, package by package; where each decision is recorded |
| [`roadmap.md`](roadmap.md) | roadmap | the single current plan: phase tables with *Done when* and *Who*, Parked, the Done ledger |
| [`../CLAUDE.md`](../CLAUDE.md) | conventions | how Claude works here: the AI-use boundary, session protocol, change and experiment documents, repo facts |

## Ledgers

| Document | What it holds |
|---|---|
| [`findings.md`](findings.md) | append-only: anything that bears on a *reported result* or constrains a later phase |
| [`results.md`](results.md) | one row per concluded experiment: question → systems → headline number → document |
| [`metrics.md`](metrics.md) | every metric the paper reports: definition → implementation → test/oracle → status |
| [`paper-map.md`](paper-map.md) | every figure and table of the ICLR 2027 submission → script, config, dataset, run |

## Experiment documents (`experiments/`) — one per question

| Experiment | Status | Question |
|---|---|---|
| [`_template.md`](experiments/_template.md) | template | the shape of an experiment document |

## Decision records (`decisions/`)

| ADR | Status | Decision |
|---|---|---|
| [0001](decisions/0001-test-tiers-and-freeze.md) | accepted (review pending) | freeze tag + four test tiers; regression goldens from the submission checkpoints |
| [0002](decisions/0002-experiment-documents.md) | accepted (review pending) | experiments are documents: question-level files, a results ledger, a paper map |

## Change documents (`changes/`) — one per branch, the reviewer's map and the branch plan

| Branch | Status | Document |
|---|---|---|
| `docs-conventions` | landed | [`2026-10-21-docs-conventions.md`](changes/2026-10-21-docs-conventions.md) |
| `phase0-freeze` | planned | [`2026-10-21-phase0-freeze.md`](changes/2026-10-21-phase0-freeze.md) |
