# Branch Alignment Audit

Audit date: 2026-08-02.

## Repository Facts

At the audited snapshot:

```text
main:                              7333b46
research/daily-coarse-graining:    072e136
merge base:                        7333b46
research ahead / behind main:      280 / 0 commits
changed files relative to main:    256
changed jax_orchidee files:        6
```

The research branch contains every `main` commit, so there is no bidirectional
history divergence and no missing main-side fix. However, code identity does
not hold: `main` has not received later shared-core source-threshold and
autodiff-safety changes. The previous documentation statement that `main`
already represented the complete currently validated Teacher was therefore
too strong.

## Shared-Core Differences

Only these six `jax_orchidee/` files differ:

| File | Difference class | Promotion policy |
| --- | --- | --- |
| `ad_primitives.py` | source-primal-preserving finite AD rules | promote only with primal and gradient gates |
| `coupled.py` | wires source `min_stomate` into retained daily carbon | production parity candidate |
| `driver/orchestration.py` | compiled capture/replay diagnostics and optional full-step output | keep hooks default-off; promote only required generic boundaries |
| `stomate/carbon_kernels.py` | source thresholds plus inactive-lane AD stabilization | production parity candidate after field-level primal comparison |
| `stomate/season.py` | inactive-lane safe divisions for reverse mode | promote after primal comparison |
| `stomate/soilcarbon_kernels.py` | source-primal-preserving DOC square-root derivative | promote if differentiable Teacher is a production requirement |

Research-only model, training, dataset, experiment, and Slurm files must not be
merged wholesale into `main` merely to synchronize these six files.

## Current Branch Contract

- `main` is the published PFT14 Teacher baseline, not yet the curated final
  production release.
- `research/daily-coarse-graining` is the current integration branch containing
  that baseline, accepted Teacher data plumbing, AD-safe retained processes,
  and neural research.
- Research capture hooks must default to off. A research experiment may add
  diagnostics but must not silently change the normal Teacher CLI.
- Scientific evidence remains scoped to PFT14 and the declared paper
  configuration.

## Required Production Promotion

Create a dedicated promotion change from `main`; do not fast-forward `main`
through the entire research history. The promotion must:

1. select the source-backed semantic corrections and generally useful AD-safe
   retained-process changes;
2. exclude neural architectures, rejected experiments, checkpoints, and
   research-only Slurm launchers;
3. run unit and Fortran-Oracle regression for each selected core change;
4. compare ordinary non-differentiated Teacher outputs before and after the
   promotion on cold start, later day, restart, 365-day, and multiple
   landpoints;
5. explain any intentional source-threshold difference instead of requiring
   bit identity to the stale baseline;
6. rerun the production performance benchmark;
7. update `main`, release documentation, and remote branch only after those
   gates pass.

The 669-landpoint acceptance remains a separate release gate. It must not be
used to discover which commits belong in the production promotion.

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
do not replace the cross-branch production-promotion numerical matrix listed
above.
