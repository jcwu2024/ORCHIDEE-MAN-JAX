# T7 shared-graph execution — 2026-10-24

Job **16060441** was submitted exactly once from gln01 at **2026-10-24 00:45:07 Asia/Shanghai**. Snapshot **00:45:19**: **PENDING / Priority**, no assigned node, runtime0. No model numerical results or completed measurement counts exist yet. Slurm continues independently; no cancellation, repeat submission, watcher, heartbeat or automatic follow-up.

## Authorization and immutable proposal

Human prior delegation in supervisor thread `01a128cb-74c2-7fd0-822a-e411d7bcc64e`: **“那边停了，你看没问题就可以批准，授权你了。”**. New human instruction on2026-10-24: **“Sol线程已经结束，可以继续让他干活。”**. Supervisor actually reviewed616faa7 complete proposal/REVIEW/deployment, Linux42-pass raw receipt, final-check and argument-chain evidence, accepted them, and approved this exact single execution under that continuing delegation. This does not claim the human newly said “批准”, and does not reuse any consumed historical grant.

Original proposal remains **approved=false**, unchanged in `outputs/T7-current-model-precision-shared-graph-proposal-20261023/proposal.json`, commit616faa7, SHA256 `069e8dcc114580314dffc173fdaab1158dc0ad2931520ec3a41e26061bee7a24`. New approved execution record is separate in the independent remote output and copied locally as `approval.v2.json`, binding exact argv/resources/cwd/source/probe/output and the authorization text above.

Execution source **96978caf04c9a56273ea7e9b9031f57aebf999ac**, probe SHA256 **3db6eaebbe7e0000e93f661d3264900e4d1d8ac92efe851a667bdc7392d1c863**. Cwd `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-shared-graph-20261023` remains clean and resolves `jax_orchidee/__init__.py` there. No source changes/redeployment; old f2242fc5 checkout, results, consumed approvals, data and borrowed Git stores retained.

## Pre-submit and exact request

Only freshness checks repeated: SHA/probe/import/clean identity, no prior independent output/submission/result/ledger/approval, necessary asset readability/size/mtime unchanged from the fully hashed2026-10-23 receipt, exact source-cache bytes. Existing42/59 tests and fixed case/input/science evidence reused. Fresh exact-argv approval was checked on test01 with actual outer runner `--check-approval-only`: return0, approval_checked=true, GPU_executed=false. No data consumption, model tracing or GPU bootstrap in that check.

Current gnall partition/node/CPU-load/allocated-GPU snapshot and submission environment saved. No inherited SBATCH overrides. Slurm automatic selection retained; normal Priority/Resources queueing accepted. New output and intent were created before the sole `submission_argv`; raw response **Submitted batch job16060441**, return0, no stderr. scontrol confirms gnall,1node/1task/4CPUs/1GPU,01:30:00,Requeue0,OverSubscribeOK,ReqNodeList/ExcNodeList null. No extra node restriction, CPU, exclusivity or budget.

Budget **load300 + startup/discovery/preflight240 + shared graph3300 + calls1380 + save/release180 =5400s**. Lower/compile share one absolute cutoff; import/startup consumes remaining outer time. GPU memory ceiling24GiB, at most one real-model compile and52 planned calls(16native+36FD). Rates CPU0.07 + GPU2.20 CNY/hour: worst-case `(4×0.07+2.20)×1.5 = ¥3.72`; actual bill unknown. Pending wait is outside allocation runtime. One-shot/no-requeue/no automatic retry or extension.

Scientific scope unchanged:66,123,713 parameters,B1/artifact299/window0/T7,original case/weights,public(flux+stock)/9,full3854state,four directions434/2382/2621/control200,fixed six-step FD,rtol1e-5/atol1e-6. FP64 diagnostic only. State20 rawNaN independent strict failure remains; no fill/waiver. No training, Teacher generation, architecture/capacity/precision-policy/physical/sampling/gradient-boundary changes. Preflight alone will not establish real-model acceptance.

## Evidence and handoff

Local raw records: `outputs/T7-current-model-precision-shared-graph-execution-20261024/`: freshness, separate fresh approval and CPU check, scheduler before submission, intent/raw submission response and post-submit snapshot. Original proposal/data hashes/CPU chain evidence remain in the20261023 proposal directory.

Remote output retains planned date identity:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-current-model-precision-shared-graph-20261023`.
Retained-data-root `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-current-model-forcing-asset-recovery-20261016/inputs`; source-cache is the accepted copy under the new checkout's outputs. GPU discovery/preflight/model will all run within this same single allocation, not separately.

Manual recheck (no submission):
```bash
ssh gln01 '/rmprog/slurm/v22.05.7/bin/scontrol show job --details 16060441'
ssh gln01 '/rmprog/slurm/v22.05.7/bin/sacct -j 16060441 --format=JobID,State,ExitCode,Start,End,Elapsed,AllocCPUS,AllocTRES,NodeList -P'
```
At real terminal state, retain raw JSON/logs/arrays/HLO/gradients, actual stage/attempt/completed counts/timing, cost and GPU/process release. Distinguish runtime/resource exit from numerical outcomes: exit3 may accompany completed direction results plus independent strict state20 failure. Preserve and report those results. If this shared budget still yields no result, report the actual blocked phase and stop; no additional GPU job or longer probe is authorized.
