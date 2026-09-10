# Corrected Attention continuation v3: carbon accumulation and sustained fit

Status: local and free-node admission complete; the user approved the 12h /
CNY 33.12 envelope and job **14991510** was submitted on 2026-09-10 at 15:15:57
Asia/Shanghai. It subsequently retained updates 35-41 and stopped at update 42
for a score discrepancy; the 40-update plan was not completed. See
`conditioned_continuation_v3_result_20260910.md` for the result and next gate,
and `conditioned_continuation_v3_admission_20260910.md` for admission provenance.
Rationale is in `conditioned_gpp_feedback_diagnosis_20260910.md`.

## Frozen scientific method

Start from job 14984032's accepted update-34 recovery. Run 40 updates, 35-74,
with the same 109,464,375-parameter Attention model, 15,960,211 trainable
parameters, inherited trained frozen modules, corrected physical boundary,
43-source normalized-gradient update, calibrated response and scale ladder.
Retain direct GPP, state anchoring and full 7/30-day streamed objectives and
their existing weights. No new architecture, Teacher data, v5, extra output
head, direct-only warm-up, or new gradient method is introduced.

Sampling continues blocks 21-60 of seed 20260910's existing shuffled train-only
pool. The v2 stage consumed blocks 1-20. Materialized manifest is
`manifests/coarse_graining/conditioned_snow_free_continuation_v3.json`, generated
by `python -m scripts.dev.prepare_conditioned_continuation_v3`.
The new stage uses 1,600 artifact exposures / 678 distinct existing IDs, five
groups of eight distinct packages per update; monitor IDs remain excluded.
This deliberately repeats the available prepared pool to measure further
learning. It is not expanded global data coverage or a full v6 dataset epoch.

Per update remains 8,320 direct day positions, 256 state anchors, 64 seven-day
and 32 thirty-day windows. Full-stage sample uses: 332,800 direct positions,
10,240 state anchors and 56,320 recurrent day positions, including overlaps.

## Recovery, reporting and decisions

Reuse the admitted pending/accepted recovery lifecycle and per-update local,
stage-cumulative and original-update-4 monitor rules. Preserve the original
monitor across the additional recovery boundary. Milestones are 5/10/20/40
new updates. Each accepted update is recoverable; no automatic resubmission.

Evaluate the existing 24 held-out temporal/spatial/joint windows before and
after. Their targets remain excluded from gradients and step-scale selection,
but they are now reused development validation, **not a fresh independent or
sealed test**. Keep the existing relative transfer and absolute acceptance
rules. Do not select the best checkpoint retrospectively from this panel.

Store the actual Teacher and predicted trajectories during these already
scheduled evaluations, through the previously exercised optional trajectory
return. This adds small host-side reporting and avoids a separate post-job
inference export. The default evaluator's old return/report behavior remains
available for earlier runs. Analyze an embedded-curve report with:

```text
conda run -n ORCJAX python scripts/dev/analyze_conditioned_gpp_feedback.py REPORT.json - OUTPUT_DIRECTORY
```

Report direct versus free-run daily GPP error on identical days; 7/30-day
GPP/NPP and respiration cumulative RMSE and signed bias; full-window AGB/BGB
net-change error inferred from the common initial state. Do not normalize
growth by the large initial stock when explaining the result. Existing stock
losses and formal acceptance thresholds remain unchanged.

The stage supports further unchanged continuation only if the existing relative
transfer passes AND 30-day GPP/NPP accumulated RMSE and AGB/BGB net-change RMSE
all decline, with maintenance/autotrophic respiration accumulated RMSE no
more than 2% worse. These are **end-of-stage review criteria**, not extra
per-step rejection, checkpoint-selection, or paid-process crash conditions.
Report failures and retain the last accepted checkpoint for diagnosis.
Absolute 7/30-day gates still control promotion; relative progress alone does
not authorize 365-day simulation, D2 or product acceptance. Even success here
does not establish final-method reproducibility: the consolidated architecture,
initialization/pretraining, data and staged optimization recipe must later be
executed end to end.

## Preparation and proposed compute envelope

Local checks cover both historical and new continuation lifecycles, original
monitor inheritance, uninterrupted sampling, completed resume, CLI and syntax.
Free CPU checks load the real update-34 parent. A zero-update free GPU admission
reloads its weights, reproduces the fixed monitor, and exercises all 24 windows
with embedded curves. Use an idle GPU selected after nvidia-smi; shared-node
preallocation is bounded. No duplicate gradient audit or test training is needed
because the forward model, loss and streamed derivative are unchanged.

Proposed paid resources: gnall, one node/task, one V100 32GB, 8 Slurm CPUs,
32GB host memory, **12h maximum**, 710m inner timeout, no requeue. The previous
20-update run took 5h08m; 40 updates are estimated at roughly 10-11h including
startup/reporting, without a guarantee against resource contention.
Worst-case additive charge: 1*12*2.20 + 8*12*0.07 = **CNY 33.12**.
Working directory `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.
Command from gln01, with the tested execution commit in EXPECTED_GIT_HEAD:
`sbatch scripts/hpc/slurm_conditioned_snow_free_continuation_v3.sh`.
The user approved this concrete envelope; submission provenance is in the
admission record. No automatic duplicate or paid resubmission is authorized.
