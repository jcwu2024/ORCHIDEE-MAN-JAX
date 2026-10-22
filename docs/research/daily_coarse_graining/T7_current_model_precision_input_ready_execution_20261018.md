# T7 input-ready execution — 2026-10-18

Updated2026-10-22 from recovered raw terminal records: job **15941021 FAILED124:0**, running **2026-10-18 09:46:46–09:54:25 Asia/Shanghai**,459s,ibc13b03n03,4CPU/1V100. Earlier pending snapshots below are historical, not live status.

`result.v2.json`: `incomplete_timeout / deadline_exceeded / lower`. Sole lower started at50.685064s and deadline exceeded at350.690387s, i.e. the old300s phase cap. Trace attempts1, completed traces0, compile attempts/completions0, call attempts/completions0, results=[]: all52 numerical checks unstarted. This is an engineering timeout, without new derivative/precision evidence, not numerical rejection. State20 rawNaN/strict_failure=true remains separate and unwaived.

Retained inputs were actually consumed, missing static fields=[], original prepared case unchanged. Discovery passed in2.013170s and synthetic preflight in90.433149s; these do not certify real-model execution. Probe exited124 after355.530214s. Supervisor verified release and owned process-group cleanup, active groups=[], GPU0MiB/0%/no processes. Sampled peak345MiB is not full-model memory acceptance.

Occupied458.939854s / estimated¥0.316158566; sacct459s / estimated¥0.3162, actual bill unknown. Ledger closed; this single approval is consumed. No retry or additional GPU submission occurred. Old source `f2242fc5a7657a3f3061d85a5e1b998663c8f944` and server checkout remain immutable.

Raw result/supervisor/ledger/preflight/discovery/logs/approval/submission plus sacct are recovered under `outputs/T7-current-model-precision-input-ready-execution-20261018/terminal/`. `remote-sha256.txt` verifies all18 remote files byte-for-byte; `SHA256.json` records the initial recovery set and `summary.json` is derived. No scientific arrays exist to independently analyze because no calls completed.

The [2026-10-22 timing repair](T7_precision_shared_graph_budget_repair_20261022.md) prepares a new shared3300s lower/compile deadline without changing historical results or authorizing execution. Actual required lower duration remains unknown.

## Authorization and fixed identity

Human original instruction in supervisor chat `01a128cb-74c2-7fd0-822a-e411d7bcc64e`: **“继续。”** on2026-10-18. Supervisor reviewed proposal182da2a, its complete JSON and target-shell evidence, and arranged this same single execution under the user's previous delegated acceptance authority. This record does not invent a new human “批准”. The two historical grants remain consumed; neither old approval was reused.

Proposal SHA256 `492be6285b4491104019a5556d12c3aba9831c3a7ad30b9e95d0c40fdebbb2f4`. Immutable execution source `f2242fc5a7657a3f3061d85a5e1b998663c8f944`; existing checkout `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-retained-input-cpu-20261016`, clean with correct `jax_orchidee/__init__.py` resolution. Probe SHA256 `4db6e3209fa693136673792c7a7a99802555c33e43dfe4c0f5e2240507e1cad8`. No source changes or redeployment; borrowed Git stores preserved.

Fresh exact-argv approval was created in the new output only, binding probe hash, cwd, explicit retained-data-root, resource limits, full submission argv and original authorization wording. Actual test01 `--check-approval-only` returned0 / approval_checked=true / GPU_executed=false. Existing144/19 tests, real input consumption, seven-day binding and shell evidence reused without rerunning.

## Submission and bounds

Exact approved `submission_argv` executed once; raw response `Submitted batch job 15941021`. gnall, one node/task,4 actual Slurm CPUs/one V100, nonexclusive,5400s,24GiB GPU ceiling, no-requeue, no retry, one real-model compile maximum,52 planned calls(16native+36FD). Estimate ceiling `(4×0.07+2.20)×1.5 = CNY3.72`; actual bill unknown. Queue waiting is outside allocation runtime. Scheduler confirms ReqNodeList/ExcNodeList null, OverSubscribe OK, Requeue0, requested4CPU/1GPU and90min. No inherited SBATCH overrides.

All healthy nodes currently report all8 GPUs allocated, including a node labelled idle; idle label alone is not free-GPU evidence. Normal Priority queueing is accepted; no extra placement restrictions added. Initial scheduler query lacked Slurm PATH; retried readonly with existing `/rmprog/slurm/v22.05.7/bin`. A local decoding error on an overly broad readonly scheduler response was corrected to UTF-8 and partition-bounded output before submission. Neither issue initiated any sbatch or GPU action.

Scientific scope unchanged:66,123,713 parameters, B1/artifact299/window0/T7 public `(flux+stock)/9`, original case/weights, four directions434/2382/2621/control200 and six-step FD, full3854state, rtol1e-5/atol1e-6. State20 remains an independent strict failure without filling/waiver. FP64 diagnostic only; no training, Teacher update, precision adoption or numerical acceptance claimed.

## Evidence and handoff

Local evidence: `outputs/T7-current-model-precision-input-ready-execution-20261018/` (identity, fresh approval/CPU validation, raw submission, scheduler snapshots). Remote output intentionally retains proposal identity date16:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-current-model-precision-input-ready-20261016`.
Retained-data-root remains `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-current-model-forcing-asset-recovery-20261016/inputs`; source-cache remains in the existing checkout/outputs.

Manual recheck (no resubmission):
```bash
ssh gln01 '/rmprog/slurm/v22.05.7/bin/scontrol show job --details 15941021'
ssh gln01 '/rmprog/slurm/v22.05.7/bin/sacct -j 15941021 --format=JobID,State,ExitCode,Elapsed,AllocCPUS,AllocTRES,NodeList'
```
Terminal recovery and release verification are now complete as recorded above. No numerical gate can be assessed from the empty results. Preflight alone is not model acceptance.

Historical pending snapshot on2026-10-18 at00:41:07 reported a provisional scheduler StartTime2026-10-20 13:17:59; this is an estimate, not a reservation or actual allocation start. No watcher or automatic follow-up was created.
