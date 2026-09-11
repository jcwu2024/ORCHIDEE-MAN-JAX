# Source-backed gsmean applicability, fresh-start probe v2

Frozen before corrected real-v6 execution. The user authorized three parallel
GPT-5.6 Sol agents with max reasoning: source audit, gsmean implementation,
and independent normalization/loss review. Root owns integration and execution.

## Mechanism and scope

The completed loss audit at 9a6edb6 localizes 99.75-99.77% of total initial
loss to bare-soil `diffuco_previous_step_state.gsmean`, predicted around 2769
against Teacher zero. Its output head supplies 99.99891% of the full initial
squared gradient norm. `diffuco.f90::diffuco_trans_co2`, lines 2289-2299,
initializes gsmean to zero and then loops only over vegetation PFTs 2..nvm;
line 2885 assigns vegetation conductance. Bare soil is a supported active
surface but does not have stomata. The full source audit records scope and
limitations separately before wider field corrections.

V2 applies exact zero only to the named gsmean output on PFTs with the
canonical bare-soil trait. It must be independent of PFT array position.
Preserve vegetation values, inactive-slot handling and PFT permutation.
Carry the corrected output into next-day state; keep its defined status valid
and keep zero Teacher values in the loss. Do not hide its error by excluding
the state coordinate from supervision. Other fields, topology and the
109,464,375-parameter tree remain unchanged. Default off retains historical
execution; only the v2 protocol enables this applicability rule.

DOC input conditioning remains enabled as in v1. Seed, trainable scope,
training samples, scales, all three loss weights, clipping, learning rate and
12-update budget remain exactly v1. The purpose is to remove a source-proven
invalid output before deciding whether loss reweighting is needed. Do not
zero fuel or dead-leaf fields by analogy; source/restart semantics decide them.

## Bounded acceptance

Local tests require exact bare-soil zero and zero parameter response there,
preserved vegetation response, inactive-slot behavior, permutation equivariance,
and unchanged default-off output. Run CPU abstract tracing of the complete
combined objective on the same three real samples before GPU execution.

Then one idle free gln01 GPU, at most 60 minutes. Initialize all weights anew
with seed 20260911. Check both named prediction and next-state bare-soil gsmean
are exactly zero at every monitored sample and finite-difference point. Require
the existing prediction/state/mask/discrete/budget hard gates. Retain the original
one-day reverse/JVP/rounded-FD tolerances at both epsilons. Record initial
per-group losses, all parameter-family gradient norms, gsmean head norm and
raw clipping norms; report physical GPP, respiration, NPP and AGB/BGB changes.

The initial science values should agree with the completed v1 initial values
within original atol=1e-6/rtol=1e-5 if retained processes do not consume the
corrected endpoint on that same day. Measure that correspondence; do not
assume it. State loss is expected to change. The old initial report is an
evidence comparator, never an inherited weight source or accuracy floor.

After numerical/physical/derivative admission, perform the same 12 updates
and checkpoint replay. Use the declared total/flux-loss learning screen and
physical-unit outcomes to decide the next larger experiment. A reduction in
initial loss from enforcing the invariant is not a learned improvement.
No new paid task, Teacher regeneration, v5 use, sealed test, 365-day or D2
acceptance is authorized by this bounded probe.

## Integrated implementation

The implementation agent completed the named-trait applicability helper and
default-off `apply_gsmean_applicability` head/factory option. Applicability runs
in `prediction_from_learned_outputs` before physical reconstruction, so the
bound model and named output share the corrected value. Head builders retain
the option across the existing module composition. No parameter tree change.

Six focused tests pass: named trait selection away from slot0, exact zero
value/Jacobian for bare soil, vegetation response, permutation/inactive-slot
handling, compact/full-PFT mapping, actual reconstruction-boundary dispatch,
and default-off behavior. Ruff/compile checks pass. Root independently reviewed
the placement and wired the v2 protocol into the clean bootstrap, plus exact
named-output/next-state invariant checks in the combined loss path.

An exact JSON comparison confirms seed, all numerical/data settings and the
update budget match v1. Root's offline loss accounting gives expected state
losses 0.48457/0.48618/0.47089 if only bare gsmean is zeroed, compared with
229.648/229.308/229.427. Actual model execution remains necessary. Broader
source and normalization reviews run independently and do not enable other
field changes in this v2 candidate.

The implementation agent's separate read-only integration review finds no
blocker: the flag reaches the reconstruction boundary and next state; zero
values remain defined/supervised; v1 defaults remain off. The independent
normalization review confirms pooled compact-PFT output statistics plus
missing applicability and a bare-column state scale of1 create the mismatch.
It finds no reason to alter v2 numerical settings before this correction.

The three-sample complete CPU shape preflight at 9394619 passed in 192.6s.
Both free gln01 GPUs were occupied after that check. Rather than wait without
numerical evidence, run the same initial samples on free test02 with
`--cpu-forward-only`, one process and a 15-minute cap. This uses the identical
objective and physical checks but performs no gradients or updates. It cannot
replace subsequent GPU derivative/resource/learning acceptance.

## CPU numerical admission and review outcome

Free test02 numerical forward completed at **f24225e** in 199.16s with zero
updates. Report: `runtime/outputs/diagnostics/fresh-start-gsmean-v2-cpu-forward.json`
(copied to local outputs). Shape report is
`runtime/outputs/diagnostics/fresh-start-gsmean-v2-shapes.json`, also copied.
The first CPU-forward invocation reached argument parsing before the server
fast-forward finished its automatic Git maintenance and was rejected; it did
no model computation. The actual numerical run began only after the completed
fast-forward was observed.

All three initial predictions, states, masks, discrete values, stock signs,
budgets and new applicability hard counts pass. Both named output and next-day
bare-soil gsmean are exactly zero. Vegetation gsmean remains around
2768.26497/2765.49098/2767.89066, consistent with the previous initial forward.
State losses are 0.484566612/0.486183763/0.470892702, matching offline accounting.
Total losses are 0.051294385/0.057645667/0.052204415, mean 0.053714822.

Root compared every stored initial physical flux/stock to the v1 GPU initial
report with atol=1e-6, rtol=1e-5; all pass (maximum absolute difference
3.244e-6 across physical stocks/fluxes). This is forward preservation of those
initial science outputs while removing an invalid state value. It is not
trained improvement or full-gradient acceptance of the corrected objective.

All three delegated tasks are complete. See
[`bare_soil_output_source_audit_20260911.md`](bare_soil_output_source_audit_20260911.md)
and
[`fresh_start_normalization_loss_review_20260911.md`](fresh_start_normalization_loss_review_20260911.md).
The source audit confirms unconditional bare-soil gsmean zero and rejects
unconditional zero for restart-carried dead_leaves/fuels. Its arbitrary
nonzero bare-soil fuel restart edge case remains documented and out of this
current paper-path correction; no Teacher source was changed.

Both free gln01 GPUs remained occupied at the last check (about 2505 MiB and
80% utilization each). No GPU v2 rerun has been started. Next, when an idle free
GPU is available, use the existing wrapper with
`--protocol manifests/coarse_graining/fresh_start_one_day_probe_v2.json`, a new
`OUTPUT` such as `runtime/outputs/diagnostics/fresh-start-gsmean-v2.json`, and
the unchanged 60-minute limit. Complete real GPU AD/JVP/FD and the bounded
12-update learning screen before judging clipping/loss balance further.
