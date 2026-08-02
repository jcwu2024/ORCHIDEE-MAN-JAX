# Project Start Here

This page is the shortest reliable entry point for a new contributor. It
separates the stable Teacher, the daily-surrogate research program, historical
evidence, and generated data.

## Two Model Products

### PFT14 Teacher

The Teacher is the source-backed JAX port of the paper-era ORCHIDEE-MAN PFT14
configuration. It retains the original 48 half-hour process transitions and
the daily STOMATE tail. Its release gate is the complete 669-landpoint
1961-2010 acceptance run.

### Daily Surrogate

The daily surrogate is a research model trained from Teacher data. It is not a
Fortran-equivalent model and is not yet user-facing. Its active target is a
true daily operator:

```text
day-start state + native 6-hour forcing + parameters/static conditions
  -> process-structured neural daily fluxes and transfer fractions
  -> one constrained water/carbon/energy state update
  -> retained exact daily processes
  -> next-day state
```

The final inference path must not reconstruct 48 interpolated forcing steps or
execute a 48-step state scan. The existing exact 48-step paths remain Teacher,
diagnostic, and Oracle assets only.

## Read Order

1. [`current-status.md`](current-status.md): stable scientific and release
   facts.
2. [`../PROJECT_MANIFEST.md`](../PROJECT_MANIFEST.md): repository and external
   asset ownership.
3. [`research/daily_coarse_graining/HANDOFF.md`](research/daily_coarse_graining/HANDOFF.md):
   the current research operation.
4. [`research/daily_coarse_graining/conservative_daily_process_operator_v1.md`](research/daily_coarse_graining/conservative_daily_process_operator_v1.md):
   the active daily architecture decision.
5. [`branch-alignment-20260802.md`](branch-alignment-20260802.md): current
   `main`/research relationship and cross-branch Teacher parity work.

Read dated experiment reports only when investigating their named result.
They are evidence snapshots, not roadmaps.

## Code and Data Boundaries

- `jax_orchidee/`: Teacher and source-backed retained process code.
- `research/daily_coarse_graining/`: datasets, models, training, and rollout
  research.
- `manifests/coarse_graining/`: immutable dataset and experiment identities.
- `scripts/hpc/`: Explore1000 launchers; generated logs and checkpoints remain
  under the external runtime root.
- `data/`, `reference/`, `outputs/`, `traces/`: external or generated assets,
  not source code.

## Current Next Milestone

Do not launch another neural training experiment yet. First freeze and verify:

1. a parameter-ownership contract;
2. a daily water/carbon/energy flux-label contract;
3. a Teacher-label inventory showing which labels already exist, which are
   exactly derivable, and which require supplemental diagnostic capture;
4. a non-neural replay gate proving that true daily Teacher fluxes reconstruct
   the accepted next-day boundary with valid budgets and stocks.

Only after that replay gate passes should the new neural architecture be
implemented or GPU training resume.
