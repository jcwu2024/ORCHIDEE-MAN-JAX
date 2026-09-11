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
