# Corrected Attention continuation v2: free admission

Status: implementation, preparation and zero-update free-GPU admission pass.
The next paid run is ready for approval; no new paid submission has occurred.
Execution: `b6929153fc028330cbfc347f12565043a85e7aa6`. Subsequent local changes
are documentation only; keep the tested remote execution checkout for submission.

The [frozen plan](conditioned_continuation_v2_plan_20260909.md) continues the
accepted update-14 checkpoint for 20 updates, with the same Attention, corrected
physical boundary, train-only direction and scale selection, and monitor gates.
It adds paired independent validation and bounded progress-report IO.

## Admission evidence

- Local regression: 38 tests pass, including lifecycle/resume, launcher/CLI,
  independent selection and paired counts, reporting, and streamed derivatives.
- Free test01: 24 validation artifacts prepared in 783.53 seconds, 35,528,315
  bytes; 24 distinct points and point-years, excluding 215 known-panel point-years.
- Free test02 reporting benchmark: late full-event write 0.4603 seconds versus
  compact event 0.00244 seconds, without the additional benefit of throttling.
  This is measured CPU/IO reduction, not a measured training-speed improvement.
- Free gln01 GPU 1: completed in 1162.95 seconds (19m23s), zero trained updates.
  Accepted update-14 parameters reload and reproduce all fixed-monitor comparisons.
  The 678 selected training files plus eight monitor files are readable.
- All 24 independent windows (444 recurrent days) actually load and evaluate.
  Nonfinite predictions/states, unexpected masks, discrete mismatches, budget
  failures and negative declared stocks are all zero. There are 211 declared
  dynamic-state mask differences; the existing acceptance protocol reports these
  separately and excludes them from hard failures. No gate was relaxed for this run.

Independent evaluation took 627.21 seconds including first-use compilation;
median window evaluation was 1.418 seconds, maximum 306.73 seconds. Two graph
specializations dominated startup. These are full validation timings, including
Teacher-state GPP and diagnostics, not a production daily-inference benchmark.

Local evidence:

- `outputs/conditioned-v2-preflight-b692915.json`
- `outputs/conditioned-independent-validation-panel-20260909.json`
- `outputs/conditioned-independent-validation-prepared-20260909.json`
- `outputs/conditioned-reporting-db6ff1f.json`

Remote report:
`runtime/outputs/diagnostics/conditioned-v2-preflight-b692915/report.json`.
Remote log: `runtime/logs/conditioned-v2-preflight-b692915.log`.

## Independent starting accuracy

All values below are normalized errors, not physical units or percentages.
Each row contains four distinct windows. The overall science macro normalized
RMSE is 0.369501; it must not be compared numerically to the development
monitor's differently defined MSE/bias loss.

| Validation split | Horizon | Science macro | GPP RMSE | NPP RMSE |
| --- | ---: | ---: | ---: | ---: |
| Temporal | 7 | 0.386080 | 0.980775 | 1.272254 |
| Temporal | 30 | 0.404715 | 1.043262 | 1.226917 |
| Spatial | 7 | 0.282755 | 0.724035 | 0.860508 |
| Spatial | 30 | 0.443576 | 1.156636 | 1.277494 |
| Joint | 7 | 0.461547 | 1.193922 | 1.413022 |
| Joint | 30 | 0.238330 | 0.597125 | 0.715202 |

All six cells fail absolute GPP, NPP, growth/autotrophic respiration and NEP/NEE
gates. Maintenance additionally fails spatial/30. Aboveground/belowground and
total live biomass, ecosystem/nonliving carbon, and heterotrophic respiration
pass on this small panel. This does not establish 365-day stock accuracy.
Teacher-state direct GPP over 444 days has RMSE 0.788568 and signed bias 0.450362.

These are the starting model's measurements. They neither prove nor disprove
transfer from the proposed 20 updates. Do not tune the frozen schedule from this
panel or reopen the architecture based on its starting errors. After training,
evaluate the same windows once and apply the preregistered paired transfer gate.

## Proposed paid execution

Resources: gnall, one V100 32GB, eight Slurm CPUs, 32GB host memory, eight-hour
limit; 470-minute inner timeout, no requeue or automatic resubmission. Estimated
duration remains about 5--6.5 hours excluding queue, subject to compilation and
contention. Additive ceiling: `1 * 8 * 2.20 + 8 * 8 * 0.07 = CNY 22.08`.
The live gnall partition is UP with 10 nodes / 80 GPUs; availability is not a
promise of immediate scheduling.

Working directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`, via gln01.

```bash
env EXPECTED_GIT_HEAD=b6929153fc028330cbfc347f12565043a85e7aa6 /rmprog/slurm/v22.05.7/bin/sbatch scripts/hpc/slurm_conditioned_snow_free_continuation_v2.sh
```

Output: `runtime/outputs/training/conditioned-snow-free-continuation-v2-JOBID/`.
Start: accepted update-14 `recovery.pkl` under the completed job 14979295.
Review sustained development loss reduction, independent paired transfer,
absolute errors and actual hot-update/reporting time before any extension.
