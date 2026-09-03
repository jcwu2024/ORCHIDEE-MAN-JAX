# Persistent Scale-Selected Training Launch Failure

Date: 2026-09-03

Failed Slurm job: `14911154`

Failed executable commit: `b970cf46c1c8d52f8f0ea4cb228c7a886f3bbecf`

Corrected executable commit: `64185d802416721ca970162ec09632e404c4fcbe`

## Failure

Slurm allocated the approved one-V100 job on `ibc13b03n01`, but the worker
exited before invoking Python. The production path leaves the optional
admission argument empty. Explore1000's Bash 4.2 treats expansion of the empty
`ADMISSION_ARG` array as an unbound variable under `set -u`.

The log passed the clean-commit, output-path, input-asset, and idle-GPU checks,
then stopped at:

```text
ADMISSION_ARG[@]: unbound variable
```

Slurm records state `FAILED`, exit code `1:0`, and elapsed time `00:00:02`.
The run directory was empty: no Python runner, training update, report,
checkpoint, or recovery was produced. At the published additive rates, two
allocated seconds correspond to approximately CNY 0.0015 before any billing
granularity.

## Correction And Qualification

Commit `64185d8` replaces the optional Bash array with the scalar pattern
already used by the accepted bounded-training launcher. The admission mode
passes `--admission-only`; the production mode expands the defined empty
scalar to no argument.

The corrected launcher passes local and Explore1000 Bash syntax checks, Ruff,
format checking, `git diff --check`, and all four focused persistent-training
tests. A free one-second production-mode smoke on `gln01` then passed the
former failure line, reported `admission=0`, and entered the Singularity
launch before the deliberate timeout. It wrote no report, checkpoint, or
recovery; both V100s returned to `0 MiB`, and the empty smoke directory was
removed.

## Decision

The scientific contract, data schedule, Attention parameters, accepted free
admission, and 100-update resource estimate are unchanged. The initial paid
authorization was consumed by job `14911154`. A corrected paid submission
requires a new explicit approval of the same resources, wall-time limit,
working directory, command, and worst-case cost, with expected commit
`64185d8`.
