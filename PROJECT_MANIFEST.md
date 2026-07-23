# Project Manifest

## Branch Roles

- `main`: stable PFT14 Teacher and user-facing production CLI.
- `research/daily-coarse-graining`: the complete Teacher plus isolated
  capture, dataset, and neural-surrogate research. Research hooks are disabled
  by default and must not alter Teacher numerical behavior.

Documentation authority is deliberately layered:

- `docs/current-status.md` owns stable current scientific and release facts;
- `docs/research/daily_coarse_graining/HANDOFF.md` owns the current research
  operation and immediate next action;
- contract and development-standard documents own durable interfaces and
  acceptance policy;
- dated audit and experiment documents are immutable evidence snapshots, not
  current roadmaps.

## Public Repository

- `jax_orchidee/`: production model and user CLI.
- `jax_orchidee/runners/`: production multiyear and acceptance runners.
- `configs/`: portable paper/PFT14 configuration templates.
- `manifests/`: external data and landpoint metadata.
- `scripts/dev/`: Oracle, diagnostics, and performance tooling.
- `scripts/hpc/`: scheduler examples for isolated landpoint jobs.
- `tests/`: unit, parity, and external-data integration tests.
- `docs/`: user documentation and source-driven equivalence evidence.
- `docs/research/`: research concepts and development standards kept separate
  from Teacher equivalence evidence.
- `fortran_run_scripts/paper_250919/`: small paper protocol metadata, subject
  to a final redistribution review; historical absolute paths are retained as
  provenance and are not portable launch commands.

## External Assets

The following directories are local mounts or runtime products and are
excluded from Git:

- `data/`: forcing and static scientific inputs.
- `reference/`: full Fortran NetCDF and restart packages.
- `outputs/`: model output, checkpoints, validation reports, and XLA cache.
- `traces/`: raw development trace packages.
- `fortran_source/ORCHIDEE/`: exclude from a public release until its
  redistribution license is confirmed.

Small README files and machine-readable manifests remain tracked so the
external layouts are reproducible.

## Runtime Contract

The published CLI resolves paths from `ORCHIDEE_DATA_ROOT`,
`ORCHIDEE_REFERENCE_ROOT`, and `ORCHIDEE_OUTPUT_ROOT`. The same checkout can
therefore run on a workstation or cluster without editing source files.

The production PFT14 path uses seven-day compiled complete-day blocks and
retains per-year restart checkpoints. Scientific acceptance runs should use
one landpoint per process.

Explore1000 is the current remote compute platform. Its path, environment,
transfer, node, and cost contracts are in `docs/deployment-explore1000.md`.
Historical qhcess/Cancon paths inside Fortran protocol and source-audit assets
remain provenance only.
