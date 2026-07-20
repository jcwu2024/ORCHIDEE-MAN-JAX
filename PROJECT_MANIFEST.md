# Project Manifest

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
