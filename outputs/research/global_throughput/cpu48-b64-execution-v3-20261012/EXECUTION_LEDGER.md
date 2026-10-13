# CPU48 v3 Execution Ledger

Times are server-reported UTC unless explicitly labeled workstation. Bracketed
snapshots measured the server clock about 8m20s behind the workstation; derived
terminal times use that same server clock. Accounting uses Slurm elapsed time.

| UTC | Event | Evidence |
|---|---|---|
| 2026-10-13 02:27:54 | Exact pin deployed; registry deployment entry added with `submitted=false`, invocation count `0`. | `deployment.stdout.txt`, `server-registry-after-deployment.json` |
| 2026-10-13 02:28:17 | One-CPU `test01` identity/runtime/input/reference preflight passed; no model execution, lowering, or compilation. | `test01-v3-runtime-preflight.stdout.json` |
| 2026-10-13 02:32:21 | First pre-submit gate stopped before submission: Git 1.8 `--porcelain=v1` in a test substitution, then five-row `awk` exited early under `pipefail`. | `submission-attempt-20261013.stdout.txt`, `submission-attempt-20261013.stderr.txt`, `submission-attempt-20261013.exitcode.txt` |
| 2026-10-13 02:32:45 | Second pre-submit gate stopped before submission: supported Git `--porcelain`, but the five-row `awk` summary again exited early under `pipefail`. | `submission-attempt-20261013-accepted.stdout.txt`, `submission-attempt-20261013-accepted.stderr.txt`, `submission-attempt-20261013-accepted.exitcode.txt` |
| 2026-10-13 02:36:23 | Final fresh gates passed: exact clean pin, absent run root/log, registry false/count `0`, empty queue, healthy idle 56-CPU nodes. The authorized submission command ran once. | `submission-attempt-20261013-final.stdout.txt`, `submission-attempt-20261013-final.stderr.txt`, `submission-attempt-20261013-final.exitcode.txt`, `SUBMISSION_AUDIT.md` |
| 2026-10-13 02:36:24 | Slurm accepted job `15820467`; the single-submission grant was consumed. | `submission-attempt-20261013-final.stdout.txt`, `scontrol-job-15820467.stdout.txt` |
| 2026-10-13 02:36:41 | Job started on `ibc11b09n32`, one node/task, 48 CPUs, zero GPUs. | `scontrol-job-15820467.stdout.txt`, `15820467/orchestration.json` |
| 2026-10-13 02:38:16 | Registry atomically recorded `submitted=true`, invocation count `1`, and job ID; registry SHA-256 `f453142af6835d71a77bc56e05ba2a1c9a4ab06585c7558086d1e409c82373ae`. | `server-registry-after-submission.json` |
| 2026-10-13 02:46:55 | Progress: RUNNING at elapsed `00:10:15`; scalar lowering/compilation remained in progress. | `sacct-job-15820467.snapshot.raw`, `15820467/scalar/events.jsonl` |
| 2026-10-13 02:49:37 (derived from Slurm start + elapsed) | Read-only queue snapshot: RUNNING at elapsed `00:12:56`, on the same one-node, 48-CPU allocation. | `squeue-job-15820467.stdout.txt` |
| 2026-10-13 02:58:07 | Progress: RUNNING at elapsed `00:21:26`; scalar parity/gates remained in progress and batch was not dispatched. | `progress-snapshot-20261013T030627Z.txt`, `15820467/scalar-events.jsonl`, `15820467/orchestration.json` |
| 2026-10-13 03:01:29 remote / 03:09:49 workstation | RUNNING at Slurm elapsed `00:24:49`; same allocation. Bracketed snapshots measured a stable remote/workstation clock offset of about 8m20s. | `clock-and-job-snapshot-20261013T030949Z.txt` |
| 2026-10-13 03:13:18 (derived from Slurm start + elapsed) | Scalar step returned `0` after `2190.0347s`; its exploratory report completed, with 61/64 GPU-reference lane gates passing, three finite tolerance failures, and reset-repeat passing. Scalar strict numerical status is false. | `15820467/scalar/report.json`, `15820467/scalar.log`, `15820467/sacct-all-steps-15820467.terminal.txt` |
| 2026-10-13 03:13:18 (terminal time inferred) | Batch child failed about `0.42s` after dispatch at the OMP-budget guard, before JAX initialization, lowering, compilation, or model evaluation. The child's actual OMP value was not logged. | `15820467/batch.log`, `15820467/orchestration.json` |
| 2026-10-13 03:13:18 (parent terminal inferred) | Parent job ended `FAILED 1:0`, elapsed `00:36:37` / `2197s`, 48 CPUs, node `ibc11b09n32`; `TotalCPU=35:47.671`. One parent allocation is charged once, including the scalar step. | `15820467/sacct-all-steps-15820467.terminal.txt`, `15820467/parent-accounting.json` |
| Post-terminal | Read-only `scontrol show job` returned `Invalid job id specified`; prior live `scontrol` receipts remain preserved. The terminal full-step `sacct` record is authoritative for accounting. | `15820467/scontrol-terminal-15820467.txt`, `scontrol-job-15820467.stdout.txt` |
| Post-terminal | Bounded collector ran on `test01` with canonical project Python and returned `incomplete_run_failure_preserved`, `scientific_acceptance=false`, failure-accounting-only scope. Its early failure path did not independently admit scalar arrays, gates, or throughput. No model was rerun. | `15820467/collector-result.json`, `15820467/parent-accounting.json` |
| Post-terminal | Compact receipt manifest indexes preserved logs, scalar reports, collector result, parent accounting, and Slurm receipts. Serialized checkpoints remain separate raw artifacts under `15820467/scalar/`. | `TERMINAL_EVIDENCE_MANIFEST.md` |

The single submission count is final at one. No retry, requeue, second
submission, or renewed allocation is authorized by this execution record.
Scalar timings are exploratory one-CPU measurements only. No B64 compile or
timing, CPU48 throughput, same-workload CPU/GPU ratio, or scientific/physical
acceptance was obtained. Whole-parent cost estimates are CNY `1.4647` primary
and `2.0505` conservative at unverified rates, below the approved worst-case
ceilings of CNY `3.60` and `5.04`.
