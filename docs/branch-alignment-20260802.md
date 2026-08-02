# Branch Alignment Audit

Audit date: 2026-08-02.

## Repository Facts

At the audited snapshot:

```text
main:                              7333b46
research/daily-coarse-graining:    d917498
merge base:                        7333b46
research ahead / behind main:      281 / 0 commits
changed files relative to main:    256
changed jax_orchidee files:        6
```

The research branch contains every `main` commit, so there is no bidirectional
history divergence and no missing main-side fix. However, code identity does
not hold because the research branch modified six shared Teacher files in
addition to adding research code. The relevant question is therefore whether
ordinary Teacher behavior remains aligned, not whether the branches should be
merged.

## Shared-Core Differences

Only these six `jax_orchidee/` files differ:

| File | Difference class | Teacher-parity status |
| --- | --- | --- |
| `ad_primitives.py` | source-primal-preserving finite AD rules | research addition; forward primal must be checked |
| `coupled.py` | wires source `min_stomate` into retained daily carbon | intentional edge-semantic difference is possible |
| `driver/orchestration.py` | compiled capture/replay diagnostics and optional full-step output | hooks verified default-off |
| `stomate/carbon_kernels.py` | source thresholds plus inactive-lane AD stabilization | intended primal preservation plus threshold corrections; A/B pending |
| `stomate/season.py` | inactive-lane safe divisions for reverse mode | intended forward-primal preservation; A/B pending |
| `stomate/soilcarbon_kernels.py` | source-primal-preserving DOC square-root derivative | intended forward-primal preservation; A/B pending |

Research-only model, training, dataset, experiment, and Slurm files remain on
the research branch. This audit does not authorize or propose a branch merge.

## Current Branch Contract

- `main` is the frozen PFT14 Teacher baseline and user-facing branch.
- `research/daily-coarse-graining` is an experimental descendant containing
  that baseline, accepted Teacher data plumbing, AD-safe retained processes,
  and neural research.
- Research capture hooks must default to off. A research experiment may add
  diagnostics but must not silently change the normal Teacher CLI.
- Scientific evidence remains scoped to PFT14 and the declared paper
  configuration.

## Required Cross-Branch Teacher Parity Gate

Keep both branches separate. In isolated worktrees, run the same ordinary
Teacher configuration with every research capture/surrogate option disabled.
The parity matrix must:

1. compare cold start, later day, restart-split, 365-day, and multiple
   landpoints;
2. compare the full canonical state, discrete fields, modelout, and restart
   packets, not only AGB/GPP summaries;
3. require optional capture hooks to have zero effect when disabled;
4. distinguish intended source-primal-preserving AD changes from the explicit
   `min_stomate` source-threshold correction;
5. record every nonzero difference by field and first day of occurrence;
6. require exact discrete equality and field-aware floating tolerances;
7. keep the result as a research-branch admission gate, without updating
   `main`.

The 669-landpoint acceptance remains a separate release gate. It is not needed
to determine whether ordinary Teacher mode diverged between these two branch
heads.

## Automated Checks To Preserve

- `git merge-base --is-ancestor main research/daily-coarse-graining`;
- clean worktree and commit-bound manifests;
- default-off capture diagnostics;
- exact state/discrete behavior for ordinary Teacher runs;
- finite parameter gradients only where the parameter contract declares them;
- no research import from the user-facing Teacher CLI.

## Audit Verification

The 2026-08-02 worktree audit verified:

- `git merge-base --is-ancestor main HEAD` succeeds;
- `jax_orchidee/` contains no import of `research` or
  `research.daily_coarse_graining`;
- optional `retain_step_results`, compiled-driver capture, and pre-daily
  training-boundary capture arguments all default to `False`;
- 82 targeted tests covering AD primitives, source-backed carbon budgets,
  persisted `OK_LEAK` capture/replay behavior, season, and soil-carbon kernels
  pass;
- all relative Markdown links in the changed contributor documentation
  resolve, and `git diff --check` passes.

These checks establish research isolation and current-branch integrity. They
do not establish full cross-branch Teacher numerical parity; the matrix above
remains pending.
