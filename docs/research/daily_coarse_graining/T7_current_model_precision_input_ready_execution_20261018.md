# T7 input-ready execution — 2026-10-18

Job **15941021** submitted once from gln01 at **2026-10-18 00:39:59 Asia/Shanghai**. Final snapshot **00:41:07**: **PENDING / Priority**, runtime0, no assigned node. Slurm continues scheduling; no cancellation, retry, watcher or heartbeat. No model numerical results, traces, compiles or calls are available in this pending snapshot; no GPU allocation has started.

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
On eventual terminal state, preserve raw result/supervisor/ledger/logs, classify actual counts and numerical outcomes from scientific JSON (exit3 can reflect state20), and verify release. Preflight alone is not model acceptance. This handoff neither cancels the pending job nor declares the research stopped.

Final stable pending snapshot at00:41:07 reports a provisional scheduler StartTime2026-10-20 13:17:59; this is an estimate, not a reservation or actual allocation start. No further polling is scheduled.
