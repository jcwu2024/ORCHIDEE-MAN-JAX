# Running PFT14

Use `orchidee-jax run --help` for the multiyear runner. The production defaults
enable compiled SECHIBA and a seven-day complete-day block. A complete day
contains 48 half-hour SECHIBA transitions followed by OK_LEAK, STOMATE daily
carbon, modelout, and full day-end state writeback.

Long runs should always specify a checkpoint directory and enable resume.
Each scheduler task should own one landpoint and an isolated output directory.
Aggregate annual acceptance only after every task has produced its final JSON
and all requested year checkpoints.

The runner normally resolves the selected paper landpoint's archived
`used_run.def` from `ORCHIDEE_REFERENCE_ROOT`. A model-only deployment without
the Fortran reference tree must provide a materialized configuration through
`--run-def`; annual Fortran deltas are then omitted.
