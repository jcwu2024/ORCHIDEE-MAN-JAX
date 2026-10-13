# CPU48 v3 Submission Audit

The only accepted `sbatch` invocation was at server-reported
`2026-10-13T02:36:23Z` from `cln01`; Slurm returned
`Submitted batch job 15820467` and exit status `0`.
Later bracketed snapshots measured a stable server/workstation offset of about
8m20s. Times in this audit use the server clock; accounting uses Slurm elapsed.
The immediate pre-submit stdout records commit
`03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`, registry marker
`submitted:false,invocations:0,job_id:null`, an empty `cnall` queue, and
healthy idle 56-CPU nodes. The run root and matching v3 log were absent, and
the target tree was clean. The accepted invocation and its stdout/stderr/status
are retained in `submission-attempt-20261013-final.*`.

Two earlier captured attempts exited before reaching the submission block:

Both used `set -euo pipefail`, the exact commit check, absent-run-root/log
checks, the v3 `submitted=false`/count-zero registry assertion, an empty
queue snapshot, and then the same authorized submission subshell reproduced
below. Their differing final pre-submit gates were:

```bash
# 02:32:21Z capture: Git 1.8 printed an error to stderr; empty stdout meant
# this test did not fail. The following awk exit is what stopped the script.
test -z "$(cd "$CODE_DIR" && git status --porcelain=v1)"
"$SLURM/sinfo" -p cnall -N -h -o '%N|%T|%C|%O' | awk -F'|' '$2 == "idle" && $3 == "0/56/0/56" {print; n++; if (n==5) exit}'

# 02:32:45Z capture: supported Git spelling, same early-exit pipeline.
test -z "$(cd "$CODE_DIR" && git status --porcelain)"
"$SLURM/sinfo" -p cnall -N -h -o '%N|%T|%C|%O' | awk -F'|' '$2 == "idle" && $3 == "0/56/0/56" {print; n++; if (n==5) exit}'
```

In both bodies, the approved `sbatch` subshell followed these gates. The
`pipefail` exit prevented control from reaching it. The final accepted body
used the corrected non-early-exit `awk` condition and a supported clean-tree
check.

1. At `02:32:21Z`, the server Git 1.8 client rejected
   `git status --porcelain=v1`, but the command was wrapped in
   `test -z "$(...)`; its empty stdout meant the test passed despite the Git
   error. The script then stopped at the five-row `sinfo | awk` summary:
   `awk` exited early, `pipefail` propagated upstream SIGPIPE, and `set -e`
   prevented entry into the submission subshell. Captured stderr contains the
   Git option error; stdout has no `Submitted batch job` line. Capture:
   `submission-attempt-20261013.stdout.txt`,
   `submission-attempt-20261013.stderr.txt`, and
   `submission-attempt-20261013.exitcode.txt`.
2. At `02:32:45Z`, the Git check used supported `git status --porcelain`, but
   the five-row `awk` summary still exited early while consuming `sinfo`. With
   `pipefail`, the resulting upstream SIGPIPE stopped
   the script before its submission block. Captured stdout has no submission
   response, and the exit file is `141`. Capture:
   `submission-attempt-20261013-accepted.stdout.txt`,
   `submission-attempt-20261013-accepted.stderr.txt`, and
   `submission-attempt-20261013-accepted.exitcode.txt`.

The successful command-generation record used the following gates (the two
failed captures differ in Git spelling and the early-exit `awk` condition
described above):

```bash
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_DIR=$PROJECT/runtime/worktrees/global-throughput-cpu48-b64-v3
RUN_ROOT=$PROJECT/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012
SLURM=/rmprog/slurm/v22.05.7/bin
test "$(cd "$CODE_DIR" && git rev-parse HEAD)" = 03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8
test -z "$(cd "$CODE_DIR" && git status --porcelain)"
test ! -e "$RUN_ROOT"
test -z "$(find "$PROJECT/runtime/logs" -maxdepth 1 -type f -name 'global_cpu48_b64_v3_*.txt' -print -quit)"
python3 -c 'import json,sys; x=json.load(open(sys.argv[1]))["latest_global_cpu48_b64_v3_deployment"]; assert x["execution_sha"]=="03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8" and x["submitted"] is False and x["sbatch_invocation_count"]==0 and x.get("job_id") is None; print("v3_marker=submitted:false,invocations:0,job_id:null")' "$PROJECT/runtime/deployment/registry.json"
"$SLURM/squeue" -p cnall -u "$USER" -o '%.18i|%.10P|%.30j|%.8u|%.2t|%.10M|%.10l|%.6D|%.6C|%R'
"$SLURM/sinfo" -p cnall -N -h -o '%N|%T|%C|%O' | awk -F'|' '$2 == "idle" && $3 == "0/56/0/56" {if (n<5) print; n++}'
```

It then executed this exact authorization subshell, which clears inherited
`SBATCH_*` variables and command-path overrides, exports the exact `CODE_DIR`
and `EXPECTED_CODE_COMMIT`, and `exec`s the one approved `sbatch` script.
The body was sent from Windows to `ssh cln01` after removing carriage returns:

```bash
(
  set -euo pipefail
  cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
  for name in $(compgen -A variable SBATCH_); do unset "$name"; done
  for name in CODE_DIR RUN_ROOT EVIDENCE_DIR PYTHON ENTRY COORDINATOR SLURM_BIN PREPARED_INPUT GPU_REFERENCE_DIR CONFIG; do unset "$name"; done
  export CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3
  export EXPECTED_CODE_COMMIT=03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8
  exec /rmprog/slurm/v22.05.7/bin/sbatch /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v3/scripts/hpc/slurm_global_server_cpu_48core.sh
)
```

The successful stdout contains the Slurm job ID. The immediate `scontrol`,
`squeue`, and parent/step `sacct` records are retained locally. The registry
was atomically advanced from SHA-256
`fb1d1e5fb05a4a55e7f4eccbecf36183a5e68f0f88fb0a187d2ede79b6ab8997` to
`f453142af6835d71a77bc56e05ba2a1c9a4ab06585c7558086d1e409c82373ae` with
`submitted=true`, invocation count `1`, and job ID `15820467`.

An intermediate Windows-to-SSH script transmission also stopped on CRLF shell
syntax before `sbatch`; its capture path was reused by the final accepted
attempt, so that transient response is not retained as a separate raw file.
It was followed by a fresh scheduler and registry check before the sole
successful submission. No earlier attempt was accepted by Slurm.
