# Structured-GPP Broad Training Launch Failure

Date: 2026-08-31

Failed Slurm job: `14881105`

Failed executable commit: `af5f426b51703afd3f730a2f9000449c7ba0405b`

Corrected executable commit: `c24f21b3f736f24b9e22d7448ecee61f79c77ec4`

## Failure

Slurm allocated the approved one-GPU job, but the worker exited before invoking
Python. The production path leaves the optional admission-only argument array
empty. Explore1000's older Bash treats `"${EXTRA_ARGS[@]}"` as an unbound
variable under `set -u`, despite the preceding `EXTRA_ARGS=()` initialization.

The log reached the clean commit, output-path, and idle-GPU checks, then failed
at the empty array expansion. `sacct` records state `FAILED`, exit code `1:0`,
and elapsed time `00:00:02`. No training update, model load, checkpoint, report,
or recovery occurred. At the published additive rates, two allocated seconds
correspond to about CNY 0.0015 before any center-side billing granularity.

## Correction And Qualification

Commit `c24f21b` removes the optional array. Both modes now always pass an
explicit `--maximum-updates`: two for free admission and 1,000 for production.
Ruff, Bash syntax, and the 11 focused broad-training tests pass.

A one-second free `gln01` production-branch smoke used an empty
`MAXIMUM_UPDATES` with a synthetic Slurm ID. It printed
`maximum_updates=1000`, passed the former failure line, and entered the
Singularity launch before the deliberate timeout. No GPU training was run by
that smoke.

## Decision

The scientific contract, data schedule, model parameters, and accepted free
admission are unchanged. A corrected paid submission requires a new explicit
approval of the same resources, wall time, directory, command, and worst-case
cost, with expected commit `c24f21b`.
