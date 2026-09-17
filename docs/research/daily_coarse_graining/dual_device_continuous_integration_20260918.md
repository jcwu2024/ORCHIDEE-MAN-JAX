# Continuous dual-device training admission

Status: implementation, local and deployed tiny tests complete; real-model
paid integration is ready for resource approval and has not been submitted.

## Preparation evidence

- Implemented code9cc50dd; immutable remote checkout below has that exact HEAD.
  Sol/high implemented the executor/runner and reviewed orchestration; Luna/high
  implemented the tiny continuity test; lead integrated and checked results.
-41 focused local tests pass; changed Python passes Ruff and Git whitespace checks.
- Deployed JAX0.4.38 on test01, two CPU devices: continuity probe PASSED,
  measured probe1.9598s excluding interpreter/container startup. All three
  gradient/parameter/AdamW comparisons and checkpoint/peer-sync errors are zero.
  Clipping is active; changing daily andT7/T30 data, daily-to-mixed transition
  and physical-failure rejection are exercised. Evidence:
  `outputs/dual-training-continuity-9cc50dd-cpu.json`.
- Actual free V100 GPU1 on gln01: unchanged small runtime gate PASSED63.8724s
  under JAX0.4.38, no Teacher reads or real training updates. Evidence:
  `outputs/dual-training-runtime-9cc50dd.json`.
- Target-shell `bash -n` and launcher `--dry-run` with allocated mask3,5 pass;
  four generated commands also parse through the actual runner locally.
- Free node has only one idle GPU, so simultaneous two-GPU CUDA continuity
  remains the paid job's first tiny precheck before real-model work.
- Deployment uses a complete self-contained Git bundle. A new shared clone
  exposed a pre-existing chain of alternates exceeding Git's nesting limit;
  its incomplete directory was preserved with the `-incomplete` suffix, then
  a clean independent checkout was created. No old checkout was changed and
  no compute allocation was consumed by this transfer correction.

## Purpose and fixed method

Job15230086 admitted the real-model fixed-weight update and measured1.7426x
speedup. It discarded every proposal. This gate connects that execution to
the common training runner and checks actual changing-data updates and recovery.
It is not another architecture, optimizer, loss or scientific learning pilot.

Use the existing109,464,375-parameter full-network model, seed20260911, fresh
AdamW, v6 prepared pools and first three entries of the declared200-update
sampling plan. Each mixed update uses64 daily rows in8 microbatches plus oneT7
and oneT30 TRAIN window; unchanged weights1/.5/.5, native checkpointed AD and
FP32-Attention/FP64-physics execution policy. No Teacher regeneration or v5 join.
Daily draws192 and3windows/horizon are repeated identically across paths.

GPU0 owns parameters/AdamW, daily gradient, gradient merge and one optimizer
update. GPU1 computesT7 andT30 from the same old weights, then receives updated
weights after every retained update, including a daily-only phase. Raw expensive
backward AOT cache identities remain unchanged; device-bound helpers retain
the admitted device cache scopes. Checkpoint files hold one authoritative state.

## Bounded acceptance

1. Tiny two-device fixture: changing input data, real AdamW/clipping, continuous
   serial/dual gradient and parameter parity, peer synchronization, shared
   checkpoint restoration, daily-to-mixed transition and physical-failure rejection.
   Local CPU evidence does not certify deployed CUDA.
2. Three real runner paths, all from the same fresh initialization:
   serial3; uninterrupted dual3; dual2 then exit and resume in a new process to3.
   Four process starts and nine retained optimizer updates in total. The declared
   integration budget stays3 through recovery; `--stop-after-update2` is merely
   the temporary execution boundary, not a changed method identity.
3. Compare corresponding losses/norms/clip with existing1e-6absolute/1e-5relative
   scalar tolerances. Compare all final parameters and AdamW coordinates using
   the existing full-tree1e-6absolute/1e-4relative rules and L2 condition.
   Compare physical counters exactly and require zero. Require identical data
   descriptors between arms and changing successive daily/T7/T30 inputs.
4. Shared save/load verifies model, data, sampling, initialization, optimizer
   step and paired cursors. Dual checkpoint method explicitly records placement;
   serial/dual comparisons permit only that placement difference. Each runner
   replays persisted weights through its real daily forward path. The resumed
   arm must use update2 parameters to construct its peer copy before update3.
5. Report rotating-data update times, branch/sync timings and memory. Three
   updates include first-call effects and do not establish sustained throughput.
   Existing TRAIN/development monitors remain active. Scientific acceptance
   remains false regardless of this technical gate's outcome.

## Entry points and proposed resource envelope

- Runner: `scripts/hpc/run_mixed_learning.py --dual-device-training`.
- Reusable executor: `scripts/hpc/mixed_learning_dual_device.py`.
- Real orchestration/comparison: `scripts/hpc/run_dual_training_integration.py`.
- Paid launcher: `scripts/hpc/job_mixed_learning_dual_integration.sh`.
- Checkout: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-dual-integration-v1`.
- Reports/checkpoints: `runtime/outputs/diagnostics/mixed-learning-dual-integration-JOBID*`.

Proposed gnall1node/2V10032/8allocatedCPUs/64GiB,2h wall limit,115min inner
timeout, no requeue or automatic retries. Ceiling assuming additive charges:
CPU8*2*.07=1.12yuan plusGPU2*2*2.20=8.80yuan, total9.92yuan.
This budgets four startup/compile/monitor/checkpoint cycles; the nine updates
themselves are short. Existing raw cache assets are reused. Approximately six
checkpoint files (~8GB total) plus reports/logs, not per-sample files.

The paid entry first runs the tiny continuity fixture on its allocated GPUs,
then starts real runner legs only if that passes. Work already authorized on
free nodes continues without another permission request. A new paid submission
requires the explicit resource/time/directory/command/cost approval required by
AGENTS.md and the Explore1000 skill. Previous job15230086 approval is exhausted.

After this gate passes, return to a declared larger-coverage learning segment;
do not extend profiling or historical AD audits as an independent workstream.
