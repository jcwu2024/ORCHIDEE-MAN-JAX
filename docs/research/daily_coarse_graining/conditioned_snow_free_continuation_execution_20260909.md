# Corrected Attention continuation execution

The user approved implementation, free-node checks and gnall submission under
the [ten-update plan](conditioned_snow_free_continuation_plan_20260909.md).
Execution commit: `7f13f106e0fd06371cf092cd50b3bd7012bdfd0d`.

## Implementation and local evidence

- The exact 400-artifact training schedule is materialized, with no artifact
  overlap with the eight-artifact development monitor. Existing v6 data only.
- The runner starts from the saved corrected logical-update-4 candidate;
  it does not reconstruct the previously admitted step or overwrite the old
  logical-update-3 recovery.
- One corrected bootstrap and one streamed reverse are reused across the run.
  Each new update recomputes all 43 direction sources and admits its own
  64 seven-day and 32 thirty-day observation-score comparisons.
- `pending.pkl` precedes monitoring; `recovery.pkl` follows passing local and
  cumulative transfer gates. An interruption after pending save resumes its
  monitor without recomputing the proposal. Scientific rejection stops the run
  and preserves both the evidence and the last accepted state.
- Focused local lifecycle, runner, shell-launch and existing streamed-gradient
  regression tests pass (26 distinct tests across the four focused files).
  This includes changing parameter/data values with one reusable daily reverse,
  complete ten-update mocked execution, recovery after both save boundaries,
  rejection paths, metric-count mismatch, and parsing the actual free/Slurm CLI.
  Ruff and diff checks pass; remote Bash syntax checks pass.

## Free-node admission

The zero-update saved-candidate preflight runs on free `gln01` GPU 1 using
the same launcher and GPU environment as training. Its bounded timeout is
45 minutes. Report:
`runtime/outputs/diagnostics/conditioned-continuation-preflight-7f13f10/report.json`.
Log: `runtime/logs/conditioned-continuation-preflight-7f13f10.log`.

Admission completed and passed in 518.6363 seconds. All 408 required existing
artifacts are readable; candidate serialization/reload and full canonical
monitor reproduction against `monitor_after` from
`conditioned-snow-free-monitor-6ed00b0.json` pass. All physical gates pass.
All eight monitor artifacts completed, with hot artifact times about 9.4 s.
No parameters were updated. The complete report is also saved locally as
`outputs/conditioned-continuation-preflight-7f13f10.json`.

## Approved paid execution

One gnall node/task, one V100 32 GB, 8 Slurm CPUs, 32 GB host memory,
six-hour Slurm limit, 350-minute inner timeout, no requeue. Additive maximum:
GPU CNY 13.20 + CPU CNY 3.36 = CNY 16.56.
Working directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.
Entrypoint: `scripts/hpc/slurm_conditioned_snow_free_continuation.sh`.
After the passing free-node admission, job **14979295** was submitted through
`gln01` at 2026-09-09 12:59:14 cluster time using:

```bash
env EXPECTED_GIT_HEAD=7f13f106e0fd06371cf092cd50b3bd7012bdfd0d /rmprog/slurm/v22.05.7/bin/sbatch scripts/hpc/slurm_conditioned_snow_free_continuation.sh
```

The post-submission scheduler check reports `PENDING (Priority)`, requested
TRES `cpu=8,mem=32G,node=1,gres/gpu=1`, and time limit `06:00:00`.
No subsequent paid job is authorized by this envelope.
Log: `runtime/logs/corrected10_14979295.txt`.
Run directory: `runtime/outputs/training/conditioned-snow-free-continuation-14979295/`.
That directory will contain `report.json`, `pending.pkl`, and `recovery.pkl`.

Keep the remote checkout at execution commit `7f13f10` while queued: its
launcher checks that HEAD at startup. Local documentation commits made after
submission must not be fast-forwarded onto the queued job's checkout.

This is a sustained-learning experiment, not an independent validation or
model-promotion gate. Assess the fixed monitor curve and absolute GPP/NPP,
respiration and biomass accuracy after completion. Ten updates are not proof
of full training sufficiency or of architecture failure.

Next operation: inspect this job and its report, count accepted updates and
actual attempted data use, report any stopping constraint, and compare fixed
monitor losses/RMSE/bias against the corrected update-4 baseline. Check hot
update time and reverse reuse by the second update. Do not submit a replacement
or longer paid run automatically, and do not treat a rejected optimization
step as an Attention architecture rejection.
