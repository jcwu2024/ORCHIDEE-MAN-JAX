# Branch Alignment Audit

Audit date: 2026-08-02.

## Repository Facts

At the audited snapshot:

```text
main:                              7333b46
research/daily-coarse-graining:    a779663
merge base:                        7333b46
research ahead / behind main:      284 / 0 commits
changed files relative to main:    263
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
| `ad_primitives.py` | source-primal-preserving finite AD rules | accepted; forward primals preserved |
| `coupled.py` | wires source `min_stomate` into retained daily carbon | accepted source correction |
| `driver/orchestration.py` | compiled capture/replay diagnostics and optional full-step output | hooks verified default-off |
| `stomate/carbon_kernels.py` | source thresholds plus inactive-lane AD stabilization | accepted; one Oracle-bound threshold disposition |
| `stomate/season.py` | inactive-lane safe divisions for reverse mode | accepted; forward primals preserved |
| `stomate/soilcarbon_kernels.py` | source-primal-preserving DOC square-root derivative | accepted; forward primal preserved |

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

## Completed Cross-Branch Teacher Parity Gate

The isolated-worktree matrix is complete. It:

1. compared cold start, later day, restart split, 365 days, and multiple
   landpoints;
2. compared the full canonical state, discrete fields, modelout, and restart
   packets, not only AGB/GPP summaries;
3. required optional capture hooks to have zero effect when disabled;
4. distinguished intended source-primal-preserving AD changes from the explicit
   `min_stomate` source-threshold correction;
5. recorded every nonzero difference by field and first day of occurrence;
6. required exact discrete equality and `1e-12` floating tolerances;
7. kept the result as a research-branch admission gate without updating
   `main`.

All seven cases are accepted. Six have no differing field. The only raw
difference begins on day 274 at landpoint `069.0-119.0` and is the
source-backed `min_stomate` correction, constrained by an exact first-day
field set and passed Fortran Oracle hash. Full evidence and the canonical
Teacher decision are in
[`teacher-branch-parity-acceptance.md`](teacher-branch-parity-acceptance.md).

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

Together with the completed numerical matrix, these checks close Gate A. The
research shared core is the sole Teacher development authority; `main`
remains a frozen release baseline.
