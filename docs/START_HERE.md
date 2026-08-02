# Project Start Here

This page is the shortest reliable entry point for a new contributor. It
separates the stable Teacher, the daily-surrogate research program, historical
evidence, and generated data. A new contributor should not read the full
research `HANDOFF.md` before understanding this page.

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

1. [`NEXT_STEPS.md`](NEXT_STEPS.md): the four separate claims, dependency
   order, and immediate next technical gate.
2. [`CODE_MAP.md`](CODE_MAP.md): production, research, historical, test, and
   generated code ownership.
3. [`DOCUMENT_STATUS.md`](DOCUMENT_STATUS.md): which documents are active,
   conditional, superseded, or historical.
4. [`current-status.md`](current-status.md): detailed facts when a gate needs
   its evidence history. It is not necessary to read all of it during
   onboarding.

Read `PROJECT_MANIFEST.md`, named contracts, and dated reports only when the
three pages above direct you to them.

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

Gate A is complete and the research shared core is the canonical Teacher for
future development. Do not launch another neural training experiment yet.
The next work is Gate B: freeze the Teacher PFT registry and parameter
ownership. Gate C then freezes daily flux labels and proves non-neural
constrained replay before a new network is trained.

## New Task Bootstrap

A new contributor can resume from a clean clone with:

```bash
git switch research/daily-coarse-graining
git pull --ff-only
git status --short --branch
git log -1 --oneline
```

Then read `NEXT_STEPS.md`, `CODE_MAP.md`, and `DOCUMENT_STATUS.md`. Report the
active gate before editing or submitting a job. Gate A must not be reopened
unless the six shared Teacher files change or its evidence hashes drift. The
active technical task is Gate B; later gates are queued work, not parallel
instructions. A local `AGENTS.md`, when present, adds workspace and server
rules but is deliberately not required for a clean Git handoff.

A Git clone contains source, tests, contracts, and small manifests. It does
not contain forcing, Fortran reference packages, generated Teacher shards,
checkpoints, or server runtime outputs. Resolve those through
`PROJECT_MANIFEST.md`, `data-layout.md`, environment variables, and the
Explore1000 deployment document only when the active gate needs them.

Do not infer current work from the newest dated experiment report and do not
restart an old GPU job.
