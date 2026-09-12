# Fresh-start learning v3: convergence-based admission

Frozen before execution in `fresh_start_one_day_probe_v3.json`. The preceding
seven-epsilon diagnostic completed at35e608a in2266.39s without updates. All
state masks and discrete values remained stable and every hard count passed.
Reverse/JVP agree at all tested scales. FD agrees at3e-4,1e-4,3e-5,1e-5;
the predeclared three-adjacent-small-scale plateau passes. The prior v2 rejection
at1e-3 remains evidence; no tolerance has been relaxed.

## Evidence and change

The larger finite difference is not a mask switch. Positive one-sided slopes
are near0.64, while the negative side differs at1e-3 and7e-4. The weighted
flux slope stays near0.01872; state and process terms account for the gap.
Offline state attribution identifies lignin_struc_below as the largest change
in state slope: -0.040607 at1e-3 versus0.024548 at1e-4. In deeper coordinates
the negative1e-3 perturbation pushes fractions from about1e-7 to0.99999+, while
negative3e-4 gives roughly0.00032-0.00252. This is a large nonlinear response,
not a local derivative measurement. Precise process-term attribution is not
claimed from the state-only export.

V3 retains the entire v2 model, DOC input treatment, gsmean applicability,
seed20260911, three samples, losses/scales, AdamW settings and 12-update budget.
Only the admission policy and diagnostic monitoring change. All seven fixed
scales are reported. Acceptance requires three consecutive epsilons <=3e-4
with unchanged defined/discrete state, passing physical counts and original
atol1e-6/rtol1% reverse/JVP/FD comparison. Diagnostic and ordinary scalar
agreement is checked. Failed larger-scale comparisons are recorded but do not
veto an established local convergence interval. This is a bounded local-
direction check, not all-regime gradient or Gate D2 acceptance.

After admission, the same process proceeds immediately to the already bounded
12 all-parameter updates, then saves/reloads the prototype checkpoint. No
checkpoint is inherited. Monitor update0/1/4/8/12 on all three samples, including
physical GPP, maintenance/growth/autotrophic respiration, NPP, AGB/BGB changes,
loss components, clipping and the lignin_struc_below fraction values. Count
near-zero/near-one predictions at tolerance1e-6, distinguish Teacher-supported
endpoints from opposite-end saturation, and report RMSE. Saturation counts are
diagnostic, not a new post-hoc accuracy veto; original hard gates and the
total/flux-loss learning screen remain.

The diagnostic return uses the existing optional state export; ordinary model
and loss code are unchanged. Historical v1/v2 keep their all-declared-epsilons
rule by default. Local control/monitor tests and a real-data CPU shape trace
precede one idle free GPU, maximum60minutes. No paid job or larger training is
authorized here. The canceled paid duplicate15005758 remains canceled.

Once actual learning/physical/fraction behavior is known, prepare a larger
data/batch/update plan with admission and training in one persistent job.
Do not claim that a drop in an initial invalid-output loss or a three-sample
learning probe establishes final training or generalization.

## Local integration checks

Seven focused tests pass in4.07s: unchanged scientific/optimizer settings,
legacy all-epsilon behavior, genuine distinct-scale mask-stable admission,
fraction monitoring with masked/legitimate/opposite endpoints, and the existing
physical extraction path. Ruff passes.

A GPT-5.6 Sol/max read-only integration review found no admission/ordering
blocker: all seven scales are evaluated, no optimizer is created before
admission, old protocols retain their defaults, and fraction monitoring is
diagnostic only. Its one robustness suggestion was applied: reject an invalid
ordinary monitor before running optional diagnostic comparisons, retaining
the ordinary failing values and attempted update number. This changes failure
reporting, not valid-run numerical behavior.

Free test02 traced both ordinary and diagnostic objectives on all three actual
samples at643533a and passed in208.4s. Report:
`runtime/outputs/diagnostics/fresh-start-learning-v3-shapes.json`.
The later failure-report ordering adjustment affects only rejected monitors,
not the traced model/gradient computation.

## Free execution

Started2026-09-12 around02:00 China time, free gln01 GPU1, source **0a3495e**,
Python PID **11826**, maximum60minutes. Both free GPUs were empty; only GPU1
is used. Working directory `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.

Command:
`GPU_DEVICE=1 RUN_TIMEOUT=60m OUTPUT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/fresh-start-learning-v3.json bash scripts/hpc/run_fresh_start_one_day.sh --protocol manifests/coarse_graining/fresh_start_one_day_probe_v3.json`.

Report: `runtime/outputs/diagnostics/fresh-start-learning-v3.json`.
Log: `runtime/logs/fresh-start-learning-v3.log`.
Optional final prototype: `fresh-start-learning-v3.checkpoint.pkl` beside report.
Read this run next. Do not modify its server checkout or launch a duplicate.
Paid job15005758 stays canceled; no new paid submission.

## Completed result

Completed at0a3495e in757.37s (12m37s), all12 full-network updates, convergence
admission and checkpoint replay pass. All monitored hard counts remain zero.
Mean total loss0.05371482 ->0.05056542 (5.863%); flux loss0.00246382 ->0.00194893
(20.898%). Only update12 clipped (scale0.7177); preceding updates had scale1.
Peak active JAX memory4.771GiB, allocator pool20.630GiB. Hot updates about0.85s.

Three-sample daily RMSE in gC m-2 day-1: GPP0.17902 ->0.15824, maintenance
0.08553 ->0.07572, autotrophic respiration0.07409 ->0.06682, NPP0.23592 ->0.20588;
growth respiration0.01792 ->0.01870. This is small-sample learning evidence,
not generalization or a final model.

Fraction monitoring shows a transient overshoot at update1: each sample has42
zero-Teacher deeper coordinates above0.9, maxima0.9971/0.9998/0.9994. They return
near zero by update4. The predeclared1e-6 opposite-end saturation count stayed
zero, so that count alone would miss the visible overshoot; raw values/RMSE
must remain part of review. Final fraction RMSE is near its initial value.

The user approved parallel broader single-day data work and consolidation of
multi-day AD before full-network7/30-day training. See
[`parallel_training_infrastructure_plan_20260912.md`](parallel_training_infrastructure_plan_20260912.md).
