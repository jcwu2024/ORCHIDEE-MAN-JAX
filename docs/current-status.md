# Current Project Status

Last updated: 2026-08-21.

This is the detailed status and evidence authority, not the shortest
onboarding page. Current work order is in [`NEXT_STEPS.md`](NEXT_STEPS.md),
code ownership is in [`CODE_MAP.md`](CODE_MAP.md), and document status is in
[`DOCUMENT_STATUS.md`](DOCUMENT_STATUS.md).

This page is the current status authority. Dated files under
`docs/source_audits/` and `docs/research/` are evidence snapshots and may
describe an earlier gate without being rewritten.

## Active Daily-Coarse-Graining Gate

The rejected mechanistic-output v2 isolated diagnostic remains the comparison
point: litter/POC/DOC fell `91.8014%` and passed only `16/28` dominant fields.
The user-approved `hierarchical_coordinate_query_attention_v3` repaired the
coordinate mapping and completed its bounded 100-epoch run under job
`14619803`. Loss fell `98.5804%`, all hard and gradient-coverage gates passed,
and `25/28` dominant fields passed. The result is a substantial improvement
but misses the frozen `99%` aggregate gate. DOC incoming amount
(`-0.3085%`), free-to-adsorbed transfer (`83.2158%`), and advective transfer
(`93.5753%`) remain unresolved, so no checkpoint was retained.

A zero-training audit on the same 364 frozen v6 days shows that all three
failed targets are active on every day, although only `9.88%` to `29.42%` of
their coordinate cells are meaningfully nonzero. Their daily normalized
intensities are strongly related: absolute daily-mean correlations are
`0.9597` for incoming versus adsorption and `0.8892` for incoming versus
advection. Per-coordinate constant means still leave `54.49%` to `91.00%` of
zero-predictor squared error. The likely remaining bottleneck is therefore
the model's pooled handoff between DOC stages: it does not provide
coordinate-aligned predicted transfers to the inventory endpoint decoder.

The user approved `coordinate_coupled_doc_transition_attention_v4`, and its
local foundation is complete. All 896 DOC inventory coordinates receive five
upstream reaction/transport tokens aligned by physical PFT identity, layer,
pool, and element. One shared signed transition decoder replaces the two
independent incoming/outgoing heads and materializes mutually exclusive
conservative endpoints. The model has `59,971,257` parameters and preserves
the unchanged daily boundary, conservative updater, retained tail, and all
non-DOC paths. The focused v4, runner, unchanged v3, and unchanged mechanistic
suites pass `48` tests in isolated processes; static checks also pass.

Paid job `14627588` completed the v4 100-epoch, 4,600-update diagnostic
normally in `05:30:23`. Aggregate litter/POC/DOC loss fell `99.1353%`, which
passes the aggregate gate for the first time, and all execution, gradient,
state, budget, inventory, count, and finite-value gates passed. The candidate
still fails the frozen field gate at `24/28`: DOC incoming improves strongly
to `90.5487%`, but DOC outgoing becomes `10.62` times its initial loss;
free-to-adsorbed and advective reductions are `85.1634%` and `92.6142%`.
No checkpoint was written. The coordinate-aligned handoff is useful evidence,
but the joint signed endpoint head is rejected. See
[`coordinate_coupled_doc_transition_v4_design_20260820.md`](research/daily_coarse_graining/coordinate_coupled_doc_transition_v4_design_20260820.md)
and
[`coordinate_coupled_doc_transition_v4_memorization_result_20260820.md`](research/daily_coarse_graining/coordinate_coupled_doc_transition_v4_memorization_result_20260820.md).

The following free-CPU, zero-training endpoint geometry audit passes integrity
on all `326,144` coordinates. The joint inverse reconstructs every finite
target to `8.88e-16`, and incoming/outgoing median normalized Jacobians are
balanced at `0.4730` and `0.5340`. The decisive asymmetry is the supervised
loss: incoming supplies `97.6209%` of the two fields' combined zero-predictor
squared error and outgoing only `2.3791%`. This gives the shared scalar a
strong incentive to repair incoming while sacrificing outgoing. The joint
head also has 110 active Jacobians at or below `1e-6`. Reverting to v3's
separate transforms is not sufficient because they have 563 such active
coordinates. Status plus branch-native loss-scaled magnitude has finite
targets throughout and closely matched active medians (`0.01843` versus
`0.01838`), so it is the accepted basis for v5.

The frozen
`coordinate_coupled_doc_status_branch_magnitude_attention_v5` implementation
retains v4's coordinate-aligned Attention and changes only the final endpoint
decoder and DOC loss. It emits incoming/unchanged/outgoing logits and separate
incoming/outgoing magnitude coordinates, uses differentiable soft status
gates, and preserves the conservative updater and retained exact tail. The
model has `59,973,309` parameters. Focused local forward/loss/gradient,
canonical binding, one-day state/budget, inactive-PFT, and strict permutation
tests pass; the runner and launcher now select its loss and report schema
explicitly.

Paid job `14632388` completed the V5 10-epoch screen normally in `01:14:07`.
All 460 updates were finite, selected gradient/update coverage passed, counts
were unchanged, and every hard count was zero. Epoch 5 passed. Epoch 10 failed
the frozen absolute gate: litter/POC/DOC fell `83.1178%` instead of `90%`, DOC
incoming fell `29.6589%` instead of `50%`, and DOC outgoing fell `26.1806%`
instead of `50%`. Adsorption and advection improved `16.6932%` and `58.9182%`.
No checkpoint was written.

Status and the two branch-latent losses fall `75.4655%`, `78.4438%`, and
`54.2036%`, while the materialized physical endpoint term falls only
`29.2932%`. V5 therefore fixes v4's incoming/outgoing sacrifice but does not
convert its learned latent variables into sufficiently accurate physical
endpoints. The coordinate-aligned source bank remains accepted evidence; V5's
materialization head remains rejected. See
[`doc_status_branch_magnitude_attention_v5_early_screen_result_20260821.md`](research/daily_coarse_graining/doc_status_branch_magnitude_attention_v5_early_screen_result_20260821.md).

The following target/output audit replaces the old field-by-field gate as the
current operation authority. Of the 161 learned fields, 2/26 scalar
coordinates are direct primary science outputs, 93/3,682 are state/budget
essential, and 66/35,298 are auxiliary supervision. Twenty complete-boundary
fields remain exact/diagnostic. The frozen target-oriented objective combines
an equal-group one-day state anchor, independent-basis 7/30-day science
losses, and a 0.1-weight necessary-process term. It avoids double-counting
NPP, NEP/NEE, total live biomass, and ecosystem-carbon identities while still
requiring their absolute 7/30/365-day gates. Focused contract, hierarchy,
identity, objective, and weighting tests pass `6/6` without Teacher data or
training.

The user-approved successor is now implemented as
`target_oriented_hierarchical_coordinate_attention_v1`. It retains the
source-routed 384-wide encoder, replaces the old uniform label heads with 12
domain-refinement blocks, five cross-domain science tokens, separate
high-capacity primary/state coordinate decoders, and a 192-wide auxiliary
bottleneck with label-shared heads. The fresh path does not use v4's joint
signed endpoint or v5's status-times-magnitude materializer. It has exactly
`83,755,173` float32 parameters.

The focused local gate passes `10/10` with the frozen target contract: all 161
learned fields and 39,006 coordinates are finite in eager/JIT execution;
primary, state, auxiliary, domain, and science-token paths receive finite
nonzero gradients; inactive-PFT perturbations are exactly isolated; the full
learned boundary is PFT-permutation equivariant within float32 tolerance; and
canonical binding preserves the one-day water/carbon budget tolerances. The
active operation is preparation of one bounded, no-checkpoint real-v6 GPU
lifecycle smoke. No server submission, multi-point training, 7/30-day
rollout, sealed-test read, Gate D2, or product work is authorized by the local
result alone. See
[`target_oriented_daily_operator_contract_v1.md`](research/daily_coarse_graining/target_oriented_daily_operator_contract_v1.md)
and
[`target_oriented_attention_v1.md`](research/daily_coarse_graining/target_oriented_attention_v1.md).

The update-1000 `source_routed_process_attention_v2` checkpoint has a strong
common first-order direction across 10 train/train groups and the four
one-day-anchor/NPP/AGB-tendency/BGB-tendency objectives. The group-aware
minimum-norm simplex audit has minimum cosine `0.050356` over all 40
constraints. Every tested global norm through `0.0003` improves every
component in all `10/10` groups, with zero hard or nonfinite failures.

The bounded step extension passes at norm `0.002`. All 40 group-component
losses decrease, with medians `-1.2176e-4` anchor, `-1.0193e-3` NPP,
`-1.3361e-3` AGB tendency, and `-3.4462e-4` BGB tendency. The least negative
individual change is `-1.8028e-5`; hard and nonfinite counts are zero. See
[`successor_group_aware_step_extension_20260815.md`](research/daily_coarse_graining/successor_group_aware_step_extension_20260815.md).

The efficient parent-frozen scalarization exactly reproduces the admitted
direction and applies one `0.002` update. A second fixed `0.002` proposal
regresses several constraints, worst group-5 NPP by `0.0861%`, and is stopped
before application. After the single accepted step, all six independent
7/30-day cells and all four NPP/AGB/BGB/total-live aggregates improve, but the
combined improvement is only `0.00922%`. See
[`successor_group_aware_scalarization_smoke_20260815.md`](research/daily_coarse_graining/successor_group_aware_scalarization_smoke_20260815.md).

Backtracking closes parent-frozen weights. At the second proposal, norms
`0.002, 0.001, 0.0005` regress the worst constraint by `8.6109e-4,
4.3088e-4, 2.1552e-4`. This near-linear scaling shows that the direction, not
just the finite step, has ceased to be common. See
[`successor_group_aware_backtracking_smoke_20260815.md`](research/daily_coarse_graining/successor_group_aware_backtracking_smoke_20260815.md).

The second-state exact recalibration audit completed at commit `1902936`.
The first update is reproduced to `4.96e-15` relative direction-norm error.
At that state, all 40 objectives retain a common direction with minimum cosine
`0.0486966`; norm `0.002` improves every loss and passes all component-effect
gates. The recalibrated support has 13 constraints, 12 overlapping the
parent's 14. Hard and nonfinite counts are zero, no validation values are
read, and no parameters are retained. Report SHA256 is
`c4ddb39b...10e97`. See
[`successor_second_state_recalibration_audit_20260815.md`](research/daily_coarse_graining/successor_second_state_recalibration_audit_20260815.md).

The bounded active-set implementation audit completed at commit `217876d`.
Recalibrating the 14 parent-support gradients closes all 40 directional checks
in one round with minimum relative derivative `0.0107166`; no constraint is
added. Norm `0.002` passes all ten groups and closely reproduces the exact
reference medians. Component reverse passes fall from 40 to 14, while hard
and nonfinite counts remain zero. Report SHA256 is `e2a9e0eb...daf97`. See
[`successor_active_set_recalibration_audit_20260815.md`](research/daily_coarse_graining/successor_active_set_recalibration_audit_20260815.md).

The bounded repeated active-set smoke completed at commit `b730731`. It
reconstructs the admitted first update and applies one new recalibrated norm-
`0.002` update. At the following state, the recalibrated direction remains
common across all 40 objectives with minimum relative derivative `0.0105908`.
Norms `0.001`-`0.002` regress at least one objective, while norm `0.0005`
decreases all 40 but misses the per-step `1e-4` useful-effect gate for anchor
and BGB medians. No hard or nonfinite failures occur. The two-update endpoint
improves all `6/6` non-test 7/30-day cells and every primary aggregate, but the
combined improvement is only `0.01822%`, below the `0.5%` promotion threshold.
Report SHA256 is `04c35474...613d7`. See
[`successor_repeated_active_set_smoke_20260815.md`](research/daily_coarse_graining/successor_repeated_active_set_smoke_20260815.md).

The active design task is a cumulative-microstep gate. It will recalibrate the
active set after every safe norm-`0.0005` microstep, preserve all-40
nonincrease at each microstep, and apply the unchanged four-component
`1e-4` effect gate to the cumulative macro-update. The four-microstep maximum,
early success stop, and unsafe-step stop are frozen under contract SHA256
`be46303e...2c545a`. Local admission passes 74 related tests and all static,
contract, and launcher checks. The free-`gln01` execution completed at commit
`b3d2f5f` and rejects the mechanism. The first microstep is safe and applied.
After it, the next recalibrated direction remains first-order common with
minimum relative derivative `0.0105646`, but its norm-`0.0005` finite step
regresses up to `28.7222%`, dominated by BGB tendency across multiple groups.
The step is not applied; no validation values or parameters are retained.
Report SHA256 is `7316749e...b674aa`. See
[`successor_cumulative_microstep_smoke_20260815.md`](research/daily_coarse_graining/successor_cumulative_microstep_smoke_20260815.md).

The active task is a zero-retention parameter-geometry attribution at this
stopped state. It must use existing architecture ownership to locate
direction-norm and relative-update concentration, then test a bounded set of
block projections only after they preserve all-40 first-order descent. No
checkpoint, architecture change, paid run, or further raw-direction
subdivision is authorized. The seven-block leave-one-out contract SHA256 is
`fa046011...c87153`; local admission passes 77 related tests plus static,
contract, and launcher checks. The free-`gln01` execution completed at commit
`c51dd87`. The output decoder contains `96.1526%` of squared direction norm.
Excluding it breaks common descent with minimum relative derivative
`-0.285257`; excluding each other block preserves first-order descent but
leaves the full `28.72%` finite-step regression essentially unchanged. No
top-level block is safely attributed. Report SHA256 is
`a7a170c7...c1a003`. See
[`successor_parameter_geometry_attribution_20260815.md`](research/daily_coarse_graining/successor_parameter_geometry_attribution_20260815.md).

The active design task is a four-leaf decoder attribution for the first and
second decoder kernels and biases. It must preserve the same stopped state,
all-40 first-order prefilter, and one norm-`0.0005` finite-step gate. A failure
of every leaf exclusion closes raw parameter-space direct steps and opens a
function/output-space trust-region design; it does not justify changing the
Attention architecture. The contract SHA256 is `08783a37...3cf721`; local
admission passes 79 related tests plus static, contract, and launcher checks.
The free-`gln01` execution completed at commit `39cd8ef` and finds no safe
leaf exclusion. The second decoder kernel carries `68.4533%` of full squared
direction norm; the second bias has the largest relative step at `0.645335%`
of its parameter norm. Both are required for common descent. Removing either
first decoder leaf preserves first-order descent but leaves the `28.72%`
finite regression. Report SHA256 is `513e4337...9b6c4d`. See
[`successor_decoder_leaf_attribution_20260816.md`](research/daily_coarse_graining/successor_decoder_leaf_attribution_20260816.md).

Raw Euclidean parameter-space direct steps are closed. The active design task
is now frozen as an output-locality audit comparing the 30-day per-field JVP
and exact norm-`0.0005` change at the same stopped state. All 181 fields use
their accepted train/train scales and day-start-delta representation. A
three-seed output VJP decomposes the NPP/AGB/BGB finite mismatch into decoded
output nonlinearity and downstream conservative-trajectory nonlinearity,
without per-field intervention rollouts. The contract SHA256 is
`edaebad5...7e9852`; 97 directly related local tests plus static, contract, and
launcher checks pass. See
[`successor_output_locality_audit_20260816.md`](research/daily_coarse_graining/successor_output_locality_audit_20260816.md).

The free-`gln01` execution completed at commit `2a29c2c`; report SHA256 is
`93ebbff8...1cfa85`. All ten groups, output-tape replay, JVP/VJP chain-rule,
state, budget, and nonfinite gates pass. Decoded-output nonlinearity accounts
for `96.08%` of NPP and `79.79%` of BGB finite mismatch.
`daily.fast_maintenance_respiration` carries `94.29%` of absolute NPP and
`95.11%` of BGB output-nonlinearity contribution. Its exact normalized RMS is
`0.08206` versus first-order RMS `0.000414`, and its maximum absolute change is
`4.71565` output scales. GPP contributes with the opposite sign and offsets
the regression.

The maintenance-local normalized output trust region is frozen under contract
SHA256 `84b96d4f...e35b72`. It uses the raw failed proposal as matched control,
the preceding safe microstep's exact maintenance change times `1.25` as the
cap, and one shared maintenance-displacement gradient for correction norms
`0.000125`, `0.00025`, and `0.0005`. GPP is explicitly uncapped. Local
admission passes 99 related tests plus static, contract, and launcher checks.
The first four-point-year reverse implementation exited before any candidate
or report after a `39,772,984,576`-byte XLA allocation request. The exact
gradient mean is now accumulated over 40 one-point-year reverse microbatches,
with a zero-displacement memory preflight before stopped-state reconstruction.
The free-`gln01` execution completed at commit `629238a`; formal report SHA256
is `64c68938...ce98c5`. All evidence checks pass with zero hard failures and
zero nonfinite values. The safe-reference maintenance RMS/maximum are
`0.0004166623`/`0.00402458`, defining caps `0.0005208279`/`0.00503073`; the
raw control reproduces RMS `0.08206115`, maximum `4.71565188`, and worst loss
regression `28.7222%`.

All three single-gradient corrections fail. Norm `0.00025` is the informative
result: maintenance RMS falls to `0.00363294` and the NPP/AGB/BGB medians all
improve, but RMS remains `6.98x` above its cap, maximum remains `15.33x` above
its cap, the anchor regresses `0.0177%`, and the worst loss regresses
`0.2733%`. The fixed maintenance gradient is useful but cannot jointly restore
the frozen constraints. The single-gradient projection is closed; do not
subdivide it. The active task is to freeze a bounded nonlinear constrained
solver that relinearizes violated constraints in a low-dimensional parameter
subspace and verifies each correction with exact finite evaluation. No
architecture change, optimizer update, checkpoint, validation read, sealed
test, or paid run is authorized. See
[`successor_maintenance_output_projection_protocol_20260816.md`](research/daily_coarse_graining/successor_maintenance_output_projection_protocol_20260816.md).

The successor low-dimensional nonlinear projection is frozen under contract
SHA256 `bb0f0718...d67ba07`. Its matched control is the informative
norm-`0.00025` correction. The candidate subspace has at most eight ordered
directions spanning raw progress, raw and relinearized maintenance repair,
the worst maintenance scalar, aggregate loss/effect repair, and two leading
individual loss violations. The subspace is fixed before at most four
sequential relinearizations and eight exact trials. Exact acceptance retains
the prior all-40, NPP/AGB/BGB, maintenance, state, budget, and nonfinite gates
and adds total-displacement bounds `[0.000125, 0.00075]`. Local implementation
and admission pass 95 related tests plus all static, contract-hash, and
launcher checks. The active operation is one free-`gln01` zero-retention
execution. See
[`successor_low_dimensional_nonlinear_projection_protocol_20260816.md`](research/daily_coarse_graining/successor_low_dimensional_nonlinear_projection_protocol_20260816.md).

The free execution completed at commit `85f1633` in `4905.58 s`, with report
SHA256 `4af8ddaf...5447ce`. Evidence is valid, the eight-dimensional basis and
matched-control reconstruction pass, and all hard/nonfinite counts are zero.
The matched point still violates 12 constraints. The bounded hard/slack solve
returned no admissible nonzero step, so no exact candidate trial was opened.
Fixed low-dimensional post-training projection is closed.

The Attention-only mixed-point annealed continuation completed as approved
`gnall` job `14569673` at commit `f702696` in `00:35:11`. All 250 updates and
1,000 unique point-years completed with zero hard failures. Median training
loss fell 4.7% from the first to last 50-update window and maintenance
aggregate validation RMSE improved 22.3%. The research-primary
NPP/AGB/BGB/total-live composite nevertheless became 6.98% worse; all four
primary aggregates and all six 7/30-day cells regressed. The final state is
retained but not promotion eligible, and update 1000 remains a diagnostic
reference only. See
[`successor_attention_mixed_annealed_continuation_result_20260816.md`](research/daily_coarse_graining/successor_attention_mixed_annealed_continuation_result_20260816.md).

The user rejected further local ranking around the weak update-1000 model as
the main research strategy. It is diagnostic evidence, not an incumbent that
defines success. The route-local-decoder A/B is superseded before
implementation.

The user approved `axis_aligned_process_expert_operator_v1`. The audit
behind it records 23,540 train/train point-years and 8,591,565 days versus only
500 point-years and 8,000 one-day anchors in the first fit. It also identifies
a deeper architecture bottleneck: all source/process Attention finishes at one
query per field before scalar coordinates are expanded, so large carbon
tensors cannot condition one output coordinate on its matching local input
depth/pool/PFT state. The proposal preserves dynamic axes through seven
process-owned expert paths and axis-local heads, targets 35-50 million
parameters, trains complete v6 one-day epochs before a 7/30-day curriculum,
and uses only absolute scientific gates. It has no matched control during
baseline qualification. The refined proposal keeps neural algebra and AdamW
in float32 but casts head outputs into the existing float64 constrained
physics, retained-tail, loss, and budget path. It also stops successively at
the structural, one-point-year, 64-point-year, 1%/10% epoch, one-day, 7-day,
30-day, and 365-day gates; every expert is scored independently so aggregate
loss cannot hide an unlearned process. See
[`strong_baseline_axis_aligned_process_expert_review_v1.md`](research/daily_coarse_graining/strong_baseline_axis_aligned_process_expert_review_v1.md).

Its contract is frozen at SHA256 `baf5791f...a96351d`. The executable has
49,750,681 float32 parameters, seven independent experts, source-restricted
forcing Attention, one shared factorized axial block, axis-aligned
static/process-static/state inputs, per-label heads, and an
equal-expert/equal-label loss. Fifteen local tests pass exact
schema and capacity checks, eager/JIT full forward, finite loss and JVP/VJP,
full/compact PFT mapping, forcing-source isolation, inactive-PFT isolation,
and a synthetic bound one-day conservative update with zero water/carbon
budget residual. The combined strong-baseline and relevant successor/parameter
routing regression run passes 39 tests.
See
[`strong_baseline_local_acceptance_20260816.md`](research/daily_coarse_graining/strong_baseline_local_acceptance_20260816.md).

The real-v6 zero-update admission passed. The subsequent one-point-year run at
commit `4b849fb` used all 364 valid train/train days for 100 epochs and 4,600
finite unclipped AdamW updates. Its report SHA256 is
`ada70c37...b4b399b`. GPP (`0.03391`), maintenance (`0.00697`), NPP
(`0.04041`), AGB tendency (`1.80e-5`), and BGB tendency (`3.60e-5`) meet their
declared NRMSE thresholds, with zero nonfinite, negative-inventory, or budget
hard counts. Complete process loss fell `97.60%`, below the `99%` gate, and
four of seven experts missed their `95%` reduction gates. The stock-scale
AGB/BGB metric was already below threshold at initialization, so it is not a
discriminating one-day gate. The qualification failed, retained no checkpoint,
and blocks the 64-point-year and paid stages. See
[`strong_baseline_one_point_qualification_result_20260816.md`](research/daily_coarse_graining/strong_baseline_one_point_qualification_result_20260816.md).

The bounded one-point loss-floor diagnostic contract and local implementation
are complete. The runner preserves the original 100-epoch recipe, adds at most
25 exact all-364-day updates through 46 size-8 gradient microbatches, reports
per-expert/per-label/per-field loss floors, loads the accepted train-only
one-day AGB/BGB tendency scales, and writes only a diagnostic parameter state.
Scientific rejection now exits normally and remains distinct from execution
failure. The related local suite passes 94 tests plus static checks.

The real-v6 preflight and full run then completed on free `gln01` at commit
`04b0c2e`. The final total reduction is `95.4744%`; the 25 exact-gradient
updates improve their starting loss by only `3.3112%` and do not cross the
original gates. Corrected GPP, maintenance, and AGB NRMSE pass; BGB
(`0.05064`) and NPP (`0.05694`) fail narrowly. All hard counts are zero. Five
labels account for `91.06%` of the final process loss, led by
`fast_owner.diffuco` and `fast_owner.driver` at `76.18%` together. Report
SHA256 is `2f338354...7ad5661`. See
[`strong_baseline_one_point_loss_floor_diagnostic_result_20260817.md`](research/daily_coarse_graining/strong_baseline_one_point_loss_floor_diagnostic_result_20260817.md).

The active task is a source/representation audit of the dominant endpoint
labels before any new architecture is implemented. No repeated stochastic
fit, paid job, 64-point-year fit, full epoch, 7/30/365-day screen, long-held
point, Gate D2, sealed test, or product acceptance is authorized.

That bounded audit is complete and subsequently corrected after tracing the
actual output transform and loss path. The original `39,019/260,988` claim was
wrong because it tested signed deltas against absolute endpoint ranges. The
corrected inverse-decoder audit finds zero hard range violations, 24,532 exact
0/1 boundary endpoints, and only 34 interior targets requiring absolute raw
above 20. True-v6 boundary calls still verify exact reconstruction of `z0m`,
`roughheight*`, `drysoil_frac`, `gtemp`, `stempdiag`, `tdeep`, and `hsdeep`;
`z0h` remains unresolved, while coupled energy primitives and canopy storage
remain learned. See
[`dominant_endpoint_source_representation_audit_20260817.md`](research/daily_coarse_graining/dominant_endpoint_source_representation_audit_20260817.md).

The exact-reconstruction claim now passes a 20-point-year non-test panel:
eight spatial-train points and two spatial-validation points in 1961 and 2006.
All eight admitted field families pass every point-year, with maximum error
`2.22e-15` or lower. `z0h` fails `20/20` and remains learned. Report SHA256 is
`0b906b0c...7925f74`.

The same panel closes all ten thermal coefficients from the complete daily
boundary with zero failed point-years under `1e-10 + 1e-12*abs(target)`;
report SHA256 is `adbdb831...6e579e8`. They are now verified exact derived
fields. A separate DOC panel corrects the capture semantics: both advective
and diffusive interface labels store nonnegative magnitudes, with direction
owned by water-flow sign or concentration ordering. The endpoint-trapezoid
DOC diffusion proxy reaches correlation `0.9833`, NRMSE `0.2119`, and residual
variance fraction `0.0450` overall and passes all four non-test slices. It is a
physical baseline plus learned residual, not an exact retained process. Report
SHA256 is `e200f2d7...8b69cbd`.

The user approved the complete `149/16/1/15` learned-primary, exact-derived,
physical-baseline/residual, and auxiliary/status classification. The formal
mechanistic-output implementation retains the axis-aligned seven-expert
Attention backbone, adds coupled surface-energy routing and capacity-aware
canopy storage, keeps `z0h` learned, and exposes all 181 fields after the
conservative update. It has 49,744,537 float32 parameters and 165 neural
fields/39,153 neural scalar outputs.

Eight local tests pass schema/capacity, eager/JIT primitive and post-update
forward, DOC identity/nonnegativity/gradient, thermal-kernel identity, complete
loss, all-parameter VJP/JVP, bound one-day state, and water/carbon budgets. The
bounded real-v6 `gln01` admission also passed. A fresh one-epoch 364-day smoke
then completed 46 finite AdamW updates; all prediction/state/budget/inventory
hard counts were zero and no checkpoint was written. Its report SHA256 is
`725130c4122801f56b9f22c6a0d7b6c2cf0794978d4586eb4b13f6c5bc73d304`.
The first formal 100-epoch run reduces total loss `99.1046%` and passes all
five primary-science and hard gates. It fails only the independent expert
reduction gates for canopy GPP (`93.9165%`), hydrology/snow (`83.0539%`),
litter/POC/DOC (`80.3503%`), and surface energy (`92.2124%`). Its report SHA256
is `b72ac45f...58daa0b`. A second detailed run localizes the residual to
hydrology endpoints, canopy debt, coupled energy endpoints, and DOC/litter,
and exposes sensitivity to the unclipped optimization trajectory. Its report
SHA256 is `86e7f64e...c7b539`. See
[`mechanistic_output_one_point_qualification_result_20260817.md`](research/daily_coarse_graining/mechanistic_output_one_point_qualification_result_20260817.md).

The 20-point-year non-test hydrology/canopy boundary audit is complete. Final
`nroot` is history dependent and its one-boundary reconstruction passes only
`5/20` point-years. With true final `nroot`, final `us`, `humrelv`, and
`humrel` pass `20/20` at maximum error `2.22e-16`; `evap_bare_lim` is exact
from tile `evap_bare_lim_ns`. `temp_hydro` fails every point-year with maximum
error `9.88667 K` and remains learned. The panel includes 101 under-`mcr`
events and therefore covers that active mask behavior.

Canopy storage stays within capacity on all 7,290 endpoints. Signed canopy
debt is positive on `41.51%`, and daily `vevapwet` includes negative
condensation values, so the old rare-debt wording and wet-evaporation hard
scale are rejected. See
[`hydrology_canopy_boundary_audit_result_20260817.md`](research/daily_coarse_graining/hydrology_canopy_boundary_audit_result_20260817.md).

The user approved and the code now implements the hydrology/canopy v2 output
mechanism. The frozen v2 contract learns 161 fields/39,006 scalar coordinates,
derives 20 fields exactly, and contains 49,743,001 float32 parameters. Final
`nroot` is a support-aware sparse simplex coordinate; final `us`, `humrelv`,
`humrel`, and grid `evap_bare_lim` are materialized after the conservative
update. Canopy uses one signed final endpoint plus storage/zero/debt status
logits and a unique deterministic conversion to the original conservative
transfers. The old four-transfer MSE is replaced by normalized endpoint MSE
and categorical status loss.

The v2 local gate passes 60 focused and related regression tests, including
eager/JIT forward, complete loss, Attention-path VJP/JVP, endpoint conversion,
one-day state, and water/carbon budgets. Ruff, compileall, and diff checks also
pass. The subsequent zero-update real-v6 `gln01` admission also passes. Its
complete 49,743,001-element gradient is finite with L2 norm `208.5265`; all
181 outputs and declared next-state values are finite, and water/carbon budget
residuals are exactly zero. JAX peak bytes in use are 1.441 GB despite the
roughly 25 GiB allocator reservation visible in `nvidia-smi`. Report SHA256 is
`cd376e48...ccd0d8`.

The one-epoch training smoke then passes all preregistered gates. All 46
batch-8 AdamW updates are finite, complete process loss decreases `7.3818%`,
all seven experts improve, masks/counts are unchanged, and all state/budget/
inventory hard counts remain zero. GPP, maintenance, and NPP NRMSE improve to
`0.1390`, `0.2301`, and `0.1253`. Report SHA256 is
`61de8c4c...a48b5c0`.

The formal 100-epoch/4,600-update run then completes with no execution failure.
Total loss falls from `4.5707906` to `0.0498893` (`98.9085%`). All five
primary NRMSE and all hard gates pass, but hydrology/snow (`93.7780%`),
canopy/GPP (`93.4923%`), surface energy (`89.3930%`), and litter/POC/DOC
(`80.2522%`) miss their independent `95%` reduction gates.

The following zero-training attainability audit passes all 364 days. Maximum
learned-field normalized MSE is `2.0612e-9`, maximum derived exact-field
normalized MSE is `9.8733e-22`, and DOC plus every hard count are zero. The
thermal source panel finds no physical-tolerance source failure; finite
conservative arithmetic in `mc/mcl` explains the old coefficient-level
physical-tolerance failures.

The subsequent isolated diagnostics complete all 4,600 updates per expert.
Surface energy reaches `97.9259%` aggregate reduction and passes every one of
its 11 dominant-field gates. Litter/POC/DOC reaches only `91.8014%`; 12 of 28
dominant fields fail, led by litter-above incoming amount, coupled OK_LEAK
auxiliary endpoints, DOC incoming amount, adsorption, and advective transfer.
Coverage, count identity, state, budget, finite-value, and nonnegative-inventory
gates pass, and no checkpoint is written. The active gate is an architecture
discussion for that litter/POC/DOC mapping. Another network, complete training
run, 64-point-year, full-v6, rollout, Gate D2, sealed-test, and product stages
remain closed. See
[`mechanistic_output_v2_isolated_expert_memorization_result_20260818.md`](research/daily_coarse_graining/mechanistic_output_v2_isolated_expert_memorization_result_20260818.md).

## First Formal Neural Candidate

`axial_process_operator_v1` and its exactly parameter-matched
`factorized_process_control_v1` are implemented at 5,464,321 float64
parameters. The candidate uses native-record forcing attention and four
scientific-domain axial process blocks; the control replaces interactions
with token-local transforms and masked pooling while retaining the same
parameter tree. Neither arm inherits a rejected model or checkpoint.

Local regression and bounded real-v6 `gln01` forward, loss, JVP/VJP, PFT
permutation, inactive-slot, conservative budget, retained-tail, and
100-update tiny-fit gates pass. The retained tail uses native interval
closed-form reductions and does not construct 48 forcing records or execute a
48-step state scan.

The first formal matched seed pair completed. The axial candidate job
`14507225_0` reached its best score `0.8219892971` at update 8750; the
factorized control job `14508831_1` reached `0.8433795294` at update 9000.
The candidate improved the overall runner macro-MSE score by `2.54%`, temporal
by `5.08%`, and spatial by `1.83%`, while regressing the joint slice by
`4.41%`. Both trained stably with nearly identical final training loss.

The user stopped the experiment for practical equivalence after this first
pair. Seeds `20260808/20260809` and the pair's 7/30-day rollout will not be
run; neither arm is promoted. Because the original contract named NRMSE while
the runner reported macro MSE and did not define three-seed aggregation, this
is not a formal equivalence claim or a complete-protocol rejection. See
[`first_neural_process_head_first_pair_decision_20260808.md`](research/daily_coarse_graining/first_neural_process_head_first_pair_decision_20260808.md).

The full train/train process statistics artifact remains accepted. A
vectorized, deterministic, partition-mergeable fitter
covers all 101 named process outputs plus semantic water/carbon/thermal input
scales, with PFT-shared moments and hash-bound loading. Approved `cnmix` job
`14506391` completed all eight fit steps over 8,591,565 rows and 23,540 source
shards. The batch failed after 31:22 only because its merge CLI passed one
unsupported argument; commit `9e8d4e9` fixed that wiring, after which the
retained partials merged and independently reloaded on `test01` without a
rescan. The final JSON SHA256 is
`9bc2cab43b3279e39a0fc2876ca298b2b371d6890986d44ef035859ca925eaab` and
the NPZ SHA256 is
`addf957afbd8338f4aa250155dcd299d568c1070b644e30279f9bf0ea37f9bf8`.

The final `gln01` no-checkpoint training preflight passed at commit `492f2c0`.
Both arms completed 100 real changing train/train updates at about
`0.170-0.171 s/update`, with finite loss, gradient norm, and clip scale. Each
of the temporal, spatial, and joint non-test validation slices completed its
bounded four-batch probe; sealed test data remained unread. The report SHA256
is `127ef3289af776d5c5bb994f9129f5d220aa8b8574b18c50e7e8a77aba739277`.
The source-backed long-horizon objective foundation and ownership-aware
parameter boundary are now implemented. The latter separates named
`vcmax25`/maintenance fast factors from retained `alloc_min`/`residence_time`
and also routes `vcmax25` to retained daily `vmax_step`. The closed first A/B
did not consume its packed fast-process parameter columns and cannot support
parameter-conditioned or Gate-D2 claims. See
[`daily_operator_parameter_routing_v1.md`](research/daily_coarse_graining/daily_operator_parameter_routing_v1.md).
The user selected source-routed field-level attention against an exactly
matched source-graph MLP control. The v1 contract is frozen at SHA256
`bb8bb349...92b5c`; both implemented arms contain exactly `8,679,681`
parameters and share the 20-family source registry, 101 field queries,
process graph, decoder, parameter routing, and loss boundary. Local eager/JIT
forward, loss/JVP/VJP, PFT permutation, inactive isolation, fast/retained
parameter gradients, one-day conservation, synthetic free rollout, 5+7
trajectory extraction, and bounded train/train coefficient calibration pass.
The inference audit reports zero violations. See
[`successor_process_head_local_foundation_20260809.md`](research/daily_coarse_graining/successor_process_head_local_foundation_20260809.md).
The real-v6 lifecycle smoke is not accepted. On Day 2 it loses 37
Teacher-defined values from the active vegetation slot. Initialization and
baseline probes do not repair the failure, and replacing all 101 named outputs
with Teacher values still reconstructs only `87/160` canonical fast-target
leaves and loses the same 37 next-state masks. The failure is therefore in the
named-output-to-fast-boundary adapter, not evidence against either attention or
the matched graph control.

Commit `e705b68` adds a 160-leaf ownership ledger. It identifies 14 leaves/113
columns where daily-integrated typed fluxes are written into same-named
instantaneous state, and 106 leaves/1,204 columns still carried from day start.
A read-only `test01` scan of 18 existing v6 annual shards and 6,564 transitions
shows 62 of those carried leaves, totaling 1,007 columns, actually change
within the day. Gate C2 had required non-inventory fast fields to remain
explicit endpoint reconstruction targets; the current 101-output schema does
not implement that layer. See
[`successor_fast_boundary_coverage_20260809.md`](research/daily_coarse_graining/successor_fast_boundary_coverage_20260809.md).

The revised successor fast-owner reconstruction schema is accepted at machine
contract SHA256
`4db5f828442212c58903096c36015f9d769c19aeaaa4eb9a4326fbea2fdfc4c9`.
It assigns all 160 leaves/2,855 columns exactly once to 21 existing endpoints,
19 derivations, 30 direct mirrors, 11 guarded/static carries, and 79 new neural
value heads. The added neural width is 839 values plus two `rveget` status
logits; the 14 typed-integral/state collisions are explicitly forbidden. The
contract and full 79-row field table are in
[`successor_fast_owner_reconstruction_contract_v1.md`](research/daily_coarse_graining/successor_fast_owner_reconstruction_contract_v1.md).

The non-neural JAX materializer is now wired as an opt-in stage inside the
canonical retained-tail adapter. Local daily-operator/successor regression is
`79/79`. Free-`gln01` detached commit `4648a00` reached `160/160` exact leaves
for cold Day 2, ordinary Day 3, and restart 1962 Day 1, with exact masks,
discrete state, and `rveget` status plus passing retained lifecycle and
water/carbon budgets. Its report SHA256 is
`f3ae02d08c950bbf7269c8ba618b3fb34e4ed88d14c3d264a9751cd9f360e084`.
No training, checkpoint, Teacher regeneration, or historical v5 data was used.

The user approved the identical candidate/control decoder extension. It adds
79 value queries and one `rveget` logit query while preserving the original
101 fields. Both matched arms now contain exactly `8,722,689` parameters and
share a 181-field/48-label schema, physical constraints, normalization,
dynamic mask loss, and materializer. Sixty-five fast-state targets are
day-end-minus-safe-day-start deltas; 14 daily/OK_LEAK targets are absolute.
Loss group, process domain, and source route are explicit independent
metadata. The attention candidate and fixed-graph control both pass eager/JIT
forward, loss/JVP/VJP, inactive-PFT, one-day conservation, 160-leaf
materialization, and inference-boundary audits. Full local evidence is in
[`successor_fast_owner_decoder_local_acceptance_20260809.md`](research/daily_coarse_graining/successor_fast_owner_decoder_local_acceptance_20260809.md).

The added-head train/train statistics artifact is accepted. Approved `cnmix`
job `14515519` completed eight partitions and merged 8,591,565 train/train
rows from 23,540 source shards in 8 minutes 57 seconds. A fresh `test01`
process independently reloaded 79 added fields with zero never-defined
columns and finite moments, then assembled complete 181-field normalization
and 180-field value-loss scales. `rveget` was defined for 4,771,338 of
17,183,130 eligible slots. The final JSON SHA256 is
`b2e61a171156071d404d12b833130f2d0e98064564d1c3e007e16f0210734719`
and the NPZ SHA256 is
`890d1fd6110912a835c760b4fe58e9320481a929797858cad5b70b1bf03eb99f`.
See
[`successor_fast_owner_statistics_acceptance_20260809.md`](research/daily_coarse_graining/successor_fast_owner_statistics_acceptance_20260809.md).
The free `gln01` no-checkpoint GPU probe is also accepted at commit `7c555d1`.
Both matched arms produced 181 finite constrained fields and finite fitted
losses; all 155 parameter leaves and 8,722,689 gradient elements were finite
and nonzero. All 78 PFT-axis outputs were exactly zero in the disabled slot.
The canonical transition passed with water/carbon residual magnitudes below
`1.1e-14`. The report SHA256 is
`5bb1f885367b3fc5183216c476a6bbe8d37122cb698d990bd5ca346006c0aaa7`.
See
[`successor_fast_owner_gpu_probe_acceptance_20260809.md`](research/daily_coarse_graining/successor_fast_owner_gpu_probe_acceptance_20260809.md).
The free bounded-small-fit preflight is accepted at commit `d1ae419`. Both
real-v6 Teacher replay days pass all state, mask, representation, discrete and
budget gates. The candidate coefficient preflight has finite nonzero gradients
and bounded coefficients; both matched arms complete one update with all
8,722,689 gradient elements finite, all 155 parameter leaves nonzero, finite
updated parameters and zero rollout hard failures. The report SHA256 is
`1e79bc76b97098ecf0725d94edfe144363420dba18792f05105e8f0707d543b6`.
The earlier production runner and two-update probe proved same-process
cross-shard reuse but spent 86.57 seconds per hot entry on host preparation.
Its 20-hour paid request is rejected and its Slurm script is removed.

Commit `1c63494` replaces per-update GPU-host preparation with hash-bound CPU
artifacts. The accepted two-entry manifest totals 66,868,764 bytes and binds
the source HEAD, contract, schedule, trace signature and per-file SHA256. The
fixed-seed cold `gln01` report passes both arms and all hard gates. Its hot
switched entry loads in 0.235 seconds; replay, attention and control take
1.311, 1.138 and 0.879 seconds. Report SHA256 is
`481581d1648699e140fa12c1b6a8e342f154ecb2216473f01a6916bd95c31f60`.

Cross-process persistent cache reuse is not accepted. A new process with the
same fixed hash seed and completed cache still recompiled `jit_train_step` and
was stopped by its 10-minute hard limit. See
[`successor_small_fit_prepared_batch_acceptance_20260810.md`](research/daily_coarse_graining/successor_small_fit_prepared_batch_acceptance_20260810.md).

Commit `507cb61` then froze the four accepted train/train objective
coefficients behind a source-report and first-artifact identity contract. A
fresh two-entry manifest binds the new HEAD while retaining byte-identical
artifact payloads. The cold `gln01` probe completed in 1,442.46 seconds, down
465.92 seconds (24.4%) from the 1,908.38-second baseline. Coefficient loading
took 0.018 seconds with no recalibration. Final zero-rate evaluations reused
the compiled training steps in 1.055/0.894 seconds and added no executable.
Both remaining cold train-step compile/updates took about 581 seconds; hot
attention/control updates remained 1.208/0.894 seconds. All gradients were
finite and all hard gates were zero. Report SHA256 is
`a9ab43d855a4c035cbc52f978717de9737f801ef349981a19b50e22d2056bdc9`;
see
[`successor_small_fit_compile_reduction_acceptance_20260810.md`](research/daily_coarse_graining/successor_small_fit_compile_reduction_acceptance_20260810.md).

Commit `50bc99a` implements a four-worker resumable CPU preparation boundary
and a fail-closed finalizer. The two-worker `test01` smoke passes with
byte-identical artifact hashes, about 4.8 GB peak RSS per worker, approximately
70 seconds of artifact work per worker and exact merged coverage. Its final
manifest SHA256 is
`f73c714a7f9e24699f7775b5c23f221a7eda4e1b1fdb81621f927bede26b0f31`.
The bounded resource plan requests four one-core `cnmix` tasks for four hours
(CNY 1.12 ceiling), then only after manifest admission requests 16 CPU cores
and one `gnall` GPU for two hours (additive CNY 6.64 ceiling). See
[`successor_small_fit_production_resource_plan_20260810.md`](research/daily_coarse_graining/successor_small_fit_production_resource_plan_20260810.md).

Approved `cnmix` job `14517346` completed in `01:58:09`; all four preparation
workers and the finalizer exited zero. The final product contains exactly 500
artifacts totaling 16,717,190,876 bytes. A separate `test01` process rehashed
every artifact and passed exact coverage, size, SHA256, path, HEAD, schedule
and policy checks. Manifest SHA256 is
`0182b7f7a6d5eae09025a336f26bc67ba5fb54706c3529a846d3885c16d80462`.
See
[`successor_small_fit_prepared_500_acceptance_20260810.md`](research/daily_coarse_graining/successor_small_fit_prepared_500_acceptance_20260810.md).

The pre-existing runner would have discarded both trained parameter trees at
process exit, making its named later 7/30-day screen impossible without
retraining. A checkpoint handoff amendment now preserves the immutable
producer identity at `d493699` while recording a distinct consumer commit.
Only a complete, passing 500-update fit may atomically publish both arms,
including parameters, optimizer state, hashes and tree schemas; failed and
submaximal runs publish nothing. Local contract, round-trip, tamper, incomplete
fit and injected second-arm failure tests pass `23/23`. See
[`successor_small_fit_checkpoint_handoff_v1.md`](research/daily_coarse_graining/successor_small_fit_checkpoint_handoff_v1.md).

The free `gln01` GPU-array serialization/reload preflight passed at consumer
commit `c15cd86`. Both arm trees originated on GPU, survived atomic host
persistence and verified reload exactly, and were removed with the temporary
directory. The persistent report SHA256 is
`a923d6371473bc19bef9ea89e3a390a4798b02ac13e23782d2520ab22616cf5a`.
It records zero training updates and no Teacher/v5 access; see
[`successor_checkpoint_handoff_gpu_probe_acceptance_20260810.md`](research/daily_coarse_graining/successor_checkpoint_handoff_gpu_probe_acceptance_20260810.md).

Approved paid fit job `14517771` started but failed closed after `00:02:52`
with exit code `1:0`, before the training loop and before any checkpoint
publication. The output directory is empty. The loader incorrectly required
the two-update coefficient-source schedule hash to equal the complete
500-update consuming schedule hash even though their coefficient-defining
first artifact is byte-identical. The schedule-ownership repair and a new
CPU-only exact-production admission probe pass locally; no coefficient,
objective, data or gate changed. See
[`successor_small_fit_paid_admission_failure_20260811.md`](research/daily_coarse_graining/successor_small_fit_paid_admission_failure_20260811.md).

While that job waits, the complete successor 7/30-day acceptance package was
frozen before any fit result. Contract SHA256 is
`99ec6eb0...0b6fc07`. It hash-admits both successful 500-update checkpoints,
selects a deterministic non-test panel, prepares host artifacts on free CPU
test nodes, runs same-boundary matched 7/30-day free rollouts, and applies
biomass/productivity, bias and exact hard gates. Focused local tests pass
`12/12`. At package-freeze time no rollout screen had run and no production
checkpoint existed.
See
[`successor_7_30_rollout_screen_package_20260810.md`](research/daily_coarse_graining/successor_7_30_rollout_screen_package_20260810.md).

The repaired zero-update production-asset admission has now passed on
`test01` at commit `0ed9891` against the exact 500-entry manifest, formal
schedule, first artifact, coefficient source and checkpoint handoff. Report
SHA256 is `50d090e6...18ea089`; no training, checkpoint, v5, sealed-test or
Teacher-regeneration path was used. See
[`successor_production_fit_asset_admission_20260811.md`](research/daily_coarse_graining/successor_production_fit_asset_admission_20260811.md).

The replacement fit is complete. Paid job `14520379` resumed from update 155
and completed both arms through update 500 with every update applied, finite
gradients, and zero hard counts. Its final process exit was a runner-only JIT
reuse misclassification: recovery host arrays legitimately created one second
device specialization at update 157, followed by zero new executables through
update 500. The complete update-500 recovery was retained.

Commit `d834d5d` accepts only that bounded post-recovery specialization and
supports finalization from a completed recovery without another training
update. The free-`gln01` finalizer published both checkpoints atomically. Fit
report SHA256 is `ea4b66cd...a259a`; checkpoint manifest SHA256 is
`f8eed186...75e40`. A separate `test01` reader rehashed and loaded both
checkpoints, verifying 500 updates, 8,722,689 parameters and 17,445,379
optimizer elements per arm, exact tree schemas, identities and finite values.
See
[`successor_small_fit_500_checkpoint_acceptance_20260811.md`](research/daily_coarse_graining/successor_small_fit_500_checkpoint_acceptance_20260811.md).

The frozen three-window 7/30-day free smoke passed at evaluation commit
`753db81`. Both arms completed one temporal, spatial, and joint non-test
window, all hard counts were zero, and the report SHA256 is
`966313db...458e22`. The smoke is explicitly not promotion eligible and no arm
is classified. Its single temporal window is substantially worse for
attention, while the spatial and joint windows are closer; the frozen formal
panel must determine whether that signal generalizes. See
[`successor_7_30_free_smoke_acceptance_20260811.md`](research/daily_coarse_graining/successor_7_30_free_smoke_acceptance_20260811.md).

The deterministic 96-window formal panel is prepared and independently
rehash-verified on free `test01`. It contains 24 artifacts totaling
`212,275,046` bytes. Prepared-manifest SHA256 is `6f4c4a30...23d0074` and
window-plan file SHA256 is `71a1a1ac...f397965`. See
[`successor_7_30_formal_prepared_acceptance_20260811.md`](research/daily_coarse_graining/successor_7_30_formal_prepared_acceptance_20260811.md).

Approved `gnall` job `14520704` exhausted its 45-minute limit before its first
formal artifact completed and wrote no report. The evaluator was blocked in
shared-filesystem synchronization while writing many tiny JAX persistent-cache
entries under `/WORK`; the GPU held the model but remained idle. This is an
infrastructure failure and provides no scientific classification. The local
launcher now uses process-local JIT caching only; see
[`successor_7_30_formal_cache_timeout_20260811.md`](research/daily_coarse_graining/successor_7_30_formal_cache_timeout_20260811.md).

The repair completed the full frozen formal screen on free `gln01` in
`312.730` seconds, making another paid submission unnecessary. The report
SHA256 is `7e8dc04f...a5b9d52b`. Attention is rejected: all six primary
composites are `60.61-272.25%` worse than control, 46/72 key-RMSE gates and
42/72 bias gates fail, while all 30 hard gates pass. NPP, maintenance
respiration, AGB/BGB, and total live biomass are the repeated dominant
failures. See
[`successor_7_30_formal_screen_result_20260812.md`](research/daily_coarse_graining/successor_7_30_formal_screen_result_20260812.md).

The bounded no-training attribution is complete. The formal fit consumed 500
of 23,540 available train/train point-years and was still improving at update
500; every update in both arms exceeded the global clip threshold. The old
rollout reduction also made each MSE component identical to its bias-squared
component. Commit `70c2f62` repairs bias reduction across rollout windows.

A 16-batch, two-arm coefficient probe found candidate coefficients spanning
more than 25 orders of magnitude. Free CPU activity inspection ruled out
inactive vegetation or empty Teacher targets on the near-zero-gradient
batches. A one-day `gln01` path probe then found order-one gradients at the
named-output boundary but parameter gradients around `1e-24` to `1e-20`:
initialized GPP and maintenance outputs had entered the shared nonnegative
decoder's negative saturation tail. Across all 16 batches, both arms have the
same mix of near-zero and extreme-positive initial outputs. This shared output
conditioning also explains persistent gradient clipping; it is not evidence
against attention specifically. See
[`successor_training_sufficiency_loss_gradient_audit_20260812.md`](research/daily_coarse_graining/successor_training_sufficiency_loss_gradient_audit_20260812.md).

The rejected attention checkpoint remains rejected and the control is not
promoted. The user approved a matched repair that changes only the common
final decoder kernel initialization to `1e-3` of its former scale, retaining
zero bias, both bodies, parameter count, routing, retained tail and operator
boundary. Implementation and 59 related local tests pass. The free `gln01`
16-batch re-audit passes at `0ef543b`: near-zero active GPP and maintenance
cases are eliminated, GPP >10x cases are eliminated, maintenance >10x cases
fall from 41/60 to 3/60, and the minimum parameter-gradient norm rises from
about `1e-26` to `0.6077`, with the exact parameter count unchanged.

Commit `b800db9` closes the follow-on asset gate without another GPU probe.
The immutable v2 report's 32 raw measurements are recalculated once under the
v3 lower-10%-quantile rule, producing coefficients `0.088284572329923`,
`0.13244689760336273`, `0.019877774130136576`, and
`0.029825749684563974`; all state/budget hard counts remain zero. Free
`test01` froze and independently reloaded those coefficients plus a 48-window
temporal/spatial/joint validation panel with eight distinct landpoints per
slice, nonoverlapping within-shard windows, zero exact overlap with the old
96-window panel, and no sealed-test use. See
[`successor_staged_training_assets_acceptance_20260812.md`](research/daily_coarse_graining/successor_staged_training_assets_acceptance_20260812.md).
Commit `9b4a1ce` completes the free `gln01` staged-runner smoke. Both matched
arms applied two finite-gradient updates, and one fixed validation artifact
from each temporal/spatial/joint slice passed for both arms with zero state,
mask, discrete, nonnegative-stock, or budget hard failures. An independent
`test01` process re-read the report. Wall time was `26:35.37`, maximum host RSS
was `11,648,856` KiB, observed GPU allocation peaked near `21.9` GiB, and one
XLA train-step compile took `5:57.59`. Report SHA256 is
`67b4ba0748e4f2fdc918de68587fa5ea4834271152940aba011c60fd06839f4c`.
This is pipeline evidence, not evidence that either architecture is accurate
or promotable. See
[`successor_staged_training_gpu_smoke_acceptance_20260812.md`](research/daily_coarse_graining/successor_staged_training_gpu_smoke_acceptance_20260812.md).

The first staged milestone is now accepted at production commit `9856dae`.
Paid `cnmix` job `14524758` prepared and finalized exactly 500 gzip artifacts
in `03:26:53`; an independent `test01` process decompressed and rehashed every
artifact. Manifest SHA256 is
`19d0cfd699c2535a101edbe245fffc23f49a6c514f565bb069b60e5683355375`.
Paid `gnall` job `14525122` then completed both matched arms through update 500
and the full frozen 48-window validation in `00:52:51`, with every update
applied and all training/validation hard counts zero. Report SHA256 is
`5d2ee5b19143f0d5662e6cad0818b5d82de91f859b84140ad33942da52a5019e`.

The control validation composite is `1.0457081708` and attention is
`1.0523784009`; attention is `0.64%` worse overall and slightly worse in all
six slice/horizon cells. Neither arm is promoted. The checkpoint
`improved=true` flags only establish the first staged baseline, not improvement
over zero training. Pre-clip gradient norm still exceeds one on 482/500
attention and 474/500 control updates. Independent `test01` reload verified
the recovery, best checkpoint, both 8,722,689-parameter trees, all update
records, and all hard counts. See
[`successor_staged_500_milestone_acceptance_20260812.md`](research/daily_coarse_graining/successor_staged_500_milestone_acceptance_20260812.md).

The second staged milestone is accepted at production commit `ea21f8a`.
Preparation job `14527233` timed out after four hours with 56 resumable
artifacts; free-node reproductions showed that its extreme per-sample delays
were execution anomalies and reproduced two artifacts bit for bit. Approved
`cnall` continuation `14532991` reused those artifacts and finished the 500
incremental `501-1000` products in `01:47:12`. A separate `test01` process
rebuilt the frozen schedule and rehashed and deserialized all 500 artifacts.
Manifest SHA256 is
`fe39325a3c18d6a7240e0fe5c2c23c14324b5abd9a88377a1ba380112196f076`.

Paid `gnall` job `14535746` resumed the accepted update-500 recovery and
completed update 1000 plus the unchanged 48-window panel in `00:52:54`. Report,
recovery, and best-checkpoint SHA256 values are respectively
`e379e3e7...bd68ce1`, `d49222d4...7cae7`, and `8a3967e3...f50d0a`.
Independent `test01` loading verifies both complete 8,722,689-parameter trees,
the optimizer state, all 500 incremental records, finite scalar values, and
zero training/validation hard counts.

Control improves from `1.0457081708` to `1.0060782200` (`3.79%`); attention
improves from `1.0523784009` to `1.0145714634` (`3.59%`). All six fixed
slice/horizon cells improve for each arm. Control remains `0.84%` better
overall, so neither arm is promoted and attention has not established a
decisive advantage. Incremental clipping falls to 402/500 control and 290/500
attention updates, but gradient spikes remain. Temporal 30-day normalized
GPP/NPP RMSE remains about `2.27/2.34-2.36`, while total-live-biomass RMSE is
about `0.061`; productivity remains the main scientific error. See
[`successor_staged_1000_milestone_acceptance_20260813.md`](research/daily_coarse_graining/successor_staged_1000_milestone_acceptance_20260813.md).

The approved update-2000 continuation completed under GPU job `14540920`.
Control improved only `0.31%`, from `1.0060782200` to `1.00298315`, below the
frozen 1% gate. Attention worsened `3.08%`, from `1.0145714634` to
`1.04584784`; its maintenance-respiration errors were especially unstable.
All hard gates passed, but the best checkpoint remains update 1000 and the
unchanged schedule must not continue to update 5000.

A free-`gln01` equal-sample gradient probe at commit `14b9420` then compared
`1 x (32 anchors + 4 rollouts)` against
`4 x (8 anchors + 1 rollout)`. Both arms reproduced the ordinary loss and
full gradient to about `1e-17`, with zero hard failures and no sampled
gradient above the clipping threshold. Mixed point-year batches changed the
gradient direction but did not consistently reduce cross-batch conflict.
Report SHA256 is
`eec3fccc372ad72f1818c86f7867c1eb63fc39d8a463bfe7b883196f3e88925a`;
see
[`successor_batch_diversity_probe_20260814.md`](research/daily_coarse_graining/successor_batch_diversity_probe_20260814.md).

The 100-update equal-sample A/B completed on free `gln01` at commit `aa602ab`
in `4130.87` seconds, including the unchanged 48-window validation panel.
Report SHA256 is `6c3f16ff...161154b`; all hard counts are zero and no
checkpoint or recovery was written. Attention mixed sampling scores
`1.00714567`, 4.35% better than attention single at `1.05300199` and 0.73%
better than the update-1000 baseline. Control mixed scores `1.00539728`,
0.25% worse than control single at `1.00288890`. The preregistered both-arm
gate fails, so mixed sampling is not adopted and no paid continuation is
authorized.

The report-only loss attribution is complete. Across six cells,
control-single changes maintenance/NPP/total-live-biomass RMSE by
`-12.94%/+2.39%/+2.38%`; attention-single changes them by
`-30.32%/+14.53%/+14.50%`. Attention-mixed reverses the pattern to
`+12.32%/-2.68%/-2.66%`. The signs are consistent across all six cells for
the principal tradeoffs. See
[`successor_loss_conflict_attribution_20260814.md`](research/daily_coarse_graining/successor_loss_conflict_attribution_20260814.md).

The attention-specific zero-update train/train gradient audit completed on
free `gln01` at commit `4f35e6a` in `779.454` seconds. Report SHA256 is
`a476b332624a83515555e5532229ca51b265c6ff7eab1c349156c1a3d1410b49`;
all hard and nonfinite-gradient counts are zero, and no update, checkpoint,
validation value, or sealed-test value was used. Maintenance has stable
negative gradient cosine with NPP, AGB, and total live biomass in at least
three of four single groups. Mixed sampling mitigates none of those pairs and
makes all four median maintenance pairings less aligned, so the proposed
Attention mixed-conflict mechanism is rejected.

The frozen objective also yields median Attention gradients around `1e-1` for
NPP and `1e-2` for maintenance, versus only `1e-6` to `1e-7` for AGB, BGB,
and total live biomass. This four-to-six-order imbalance supports a loss-
objective design review, not another unchanged continuation. The active task
is to specify a train-only biomass increment/tendency mechanism, maintenance
guard, matched current-loss Attention control, graph-control role, and bounded
preflight/short-fit gates before implementation or training. See
[`successor_per_output_gradient_audit_20260814.md`](research/daily_coarse_graining/successor_per_output_gradient_audit_20260814.md).
The discussion draft is
[`successor_loss_objective_design_review_v1.md`](research/daily_coarse_graining/successor_loss_objective_design_review_v1.md).
It explicitly does not reopen the rejected all-state `process_increment_v2`:
the proposed signal is limited to source-partitioned AGB/BGB tendencies after
the conservative operator, with unchanged one-day process supervision.
The 365-day rollout, long-held points, Gate D2, paid training, and product
acceptance remain closed.

## Stable Teacher

The canonical user-facing PFT14 Teacher core is on `main` at `5910ba9`.
The corresponding research-branch core commit is `10030eb`; their complete
`jax_orchidee/` trees are byte-identical. The branches remain separate only
because the research branch also contains experimental dataset and neural
code.

Gate A cross-branch parity is complete. Seven detached-worktree cases cover
cold start, later day, restart, two 365-day cold runs, a 365-day restart run,
three landpoints, and a true 1961-to-1962 year handoff. Six cases have no
differing state or modelout leaf. One low-stock case activates the research
branch's source `min_stomate` correction on day 274; its first fields,
magnitude, continuous-only behavior, and source-extracted Fortran Oracle are
hash-bound in the historical admission evidence. The accepted six-file core
was then synchronized into `main`, so the current drift gate requires raw
parity and has no source-correction exception. See
[`teacher-branch-parity-acceptance.md`](teacher-branch-parity-acceptance.md).

Current evidence:

- cold-start, restart, 365-day, and cross-landpoint executable-reuse gates
  pass;
- seven previously unused landpoints completed 1961-2010 validation;
- their largest 2010 AGB/BGB/GPP/NPP relative error is `8.5e-5`;
- the full 669-landpoint acceptance run remains a release gate.

The project must not claim equivalence for other PFTs or unsupported
ORCHIDEE configurations.

Gate B1 is complete. The versioned PFT catalog records
stable semantic IDs, canonical Fortran PFT/MTC identities, traits, parameter
row ownership, and explicit process capabilities. Runtime parameter loading,
modelout selection, state checkpoints, and newly created SECHIBA/STOMATE
restart files retain this identity. Restart reads now parse stable metadata or
require an explicit legacy paper layout, then remap all declared PFT axes by
stable ID. The paper layout still activates only bare soil and
`mangrove_pft14`; PFT2-PFT13 are `structural_only` and fail closed if activated.
Generic orchestration now resolves PFT-indexed run.def rows, PFT14 capability
slots, daily allocation inputs, and production/reference modelout selection
through stable identities or explicitly named legacy layouts. State/layout
axis mismatches fail closed. A compact bare-soil-plus-PFT14 lifecycle now
passes cold start, one complete day, three-file restart write/readback, and a
restart-backed continuation day while retaining a two-slot PFT axis.

Gate B2 is complete. The frozen machine contract
`manifests/coarse_graining/daily_neural_pft_interface_v1.json` defines the
leading variable PFT axis, named parameter and trait channels, exact mask
semantics, forbidden identity shortcuts, and physical-parameter ownership.
Tests cover PFT permutation equivariance, inactive-slot isolation, named
channel completeness, and variable legal `n_pft`. `VCMAX25` and the
maintenance-temperature polynomial remain explicit fast-process factors;
`ALLOC_MIN` and `RESIDENCE_TIME` remain in exact daily processes. Constant
parameters cannot be claimed as learnably invertible without controlled
Teacher perturbations. This closes structural extensibility only; PFT2-PFT13
remain scientifically unsupported.

## Daily Coarse-Graining Research

The `research/daily-coarse-graining` branch contains the complete Teacher plus
research-only capture, dataset, and neural-surrogate code. The production
Teacher defaults are unchanged because all capture hooks default to off.

### Active architecture decision

The final surrogate target is now a true conservative daily process operator:

```text
day-start state + native 6-hour forcing + parameters/static conditions
  -> daily fluxes, transfer fractions, and bounded tendencies
  -> one constrained water/carbon/energy update
  -> retained source-backed daily processes
  -> next-day state
```

Final inference must not reconstruct 48 interpolated forcing steps or execute
a 48-step state scan. The accepted 96-day `OK_LEAK` driver capture and exact
48-step replay remain Teacher/Oracle evidence, not the final neural boundary.
Direct neural ownership of `carbon_32l`, DOC, or `deepC_peat` remains rejected.

PFT14 is the current training and acceptance scope, but the architecture must
use shared PFT-axis processing conditioned on source-backed traits, parameters,
fractions, and masks. Supporting another PFT later will require Teacher branch
coverage and training data, but must not require redesigning the neural
interface. See
[`research/daily_coarse_graining/conservative_daily_process_operator_v1.md`](research/daily_coarse_graining/conservative_daily_process_operator_v1.md).

Gate C is complete. Its source-backed inventory classifies 50 required daily
labels against the real v5 Teacher contract: 10 are present, 11 are exactly
derivable, and 29 are supplied by three daily-reduced compiled capture
families. Capture is default-off and stores no 48-step trajectory. The
true-label constrained updater represents ordinary inventories with bounded
outgoing fractions and nonnegative inputs; source-signed `qsintveg` is
represented as nonnegative storage minus nonnegative evaporation debt.

The Gate C2 lifecycle matrix passes cold-start continuation, an ordinary
later day, and a 1962 restart-year day. All 349 day-end state leaves, 26
modelout fields, and four modelout leaves pass; maximum state error is
`2.78e-17`. Maximum water, carbon, and flux-side energy residuals are
`4.70e-13`, `8.05e-9`, and `6.98e-9`, respectively. Defined masks and inactive
PFT slots are explicit, no post-hoc clipping is used, and the accepted exact
three-file restart roundtrip composes with the restart-day constrained replay.
Evidence is described in
[`gate_c2_constrained_replay_v1.md`](research/daily_coarse_graining/gate_c2_constrained_replay_v1.md).
The six prerequisite sequence is now `6/6`.

Gate E1 is **accepted**. The
`research/daily_coarse_graining/daily_operator/` package consumes a masked
variable-length native forcing sequence, retains explicit PFT inputs, predicts
constrained process quantities, applies one pure-JAX conservative
water/carbon/thermal update, and composes through the reusable canonical
retained-tail adapter. `DailyOperatorInput` contains only canonical day-start
state and allowed forcing/PFT/static/annual/calendar inputs; the unique
`CanonicalDailyOperatorInputAssembler` derives every overlapping grouped,
litter/lignin, mask, and process-static view. Its fail-closed cache gate rejects
all tested contradictory copies.

Thirty E1 unit tests and the 70-test B2/C/D1/registry/E1 suite pass. The real
adapter, not the identity fixture, passes eager/JIT agreement to `5.69e-14`, a
finite forward JVP, and a finite reverse VJP. A real-Teacher Oracle matrix
passes cold continuation, ordinary later day, restart-year, and restart
composition through `daily_operator_transition`. Every per-case canonical
continuous/discrete next state passes with maximum error below `3.94e-13`;
maximum water, carbon, and thermal flux residuals are `4.70e-13`, `8.69e-9`,
and `6.99e-9`. The graph contains no callback and no length-48 scan. All 47
frozen labels have exactly one owner binding.

The frozen data-readiness matrix is `10` labels already present, `8` exactly
derivable, and `29` requiring 66 full-coverage typed fields. Gate E2's active
design now generates those fields and the base Markov rows in the same
continuous Teacher execution. The formal sixth dataset release
`pft14-daily-teacher-v6` binds Teacher source SHA256
`ffae1ec0...457223`, configuration, PFT catalog, Markov contract, coherent
typed contract, and producer sources. A
point-year is complete only when its base shard, lossless typed shard plus
exact masks, year-end checkpoint, and common metadata all exist and hash
verify. The release SHA256 is
`1affefd9b3166005cdd868477ba25e68b682c7394438e5a79c4fcf087c8496a8`;
the unified dataset schema is `daily_teacher_dataset_manifest_v6`, the typed
packaging contract is v3, and the underlying Markov state contract remains v5.

A real three-day 1961 cold-start instrumentation A/B passed: capture-on and
capture-off compiled base arrays and final state were elementwise equal,
discrete state was exact, and all 66 typed fields passed the fixed compiled
production gate `atol=1e-8, rtol=1e-10`. The complete v5/v6 compatibility and
data/operator regression is `102 passed`; Ruff and Python compilation pass.
The six-point 1961 production
pilot passed on Explore1000 as job `14486923`: all six workers exited zero,
and unified aggregation plus the formal coherent reader accepted six shards,
2,184 transitions, all exact masks, and all year-end checkpoint metadata. The
dataset manifest SHA256 is
`a7e44cc536a1c1bae4f707a5097d379c8019fe1e1fbe146fad48084ef0042d88`.
Peak process RSS was 31.4 GiB; base, typed, and checkpoint assets total
28,221,594 bytes. The provisional full storage projection is 146.5 GiB.
This cold-year pilot did not measure a later hot year or execute a real
1961-to-1962 chain. That gap is now closed by Explore1000 job `14490208`, a
ten-landpoint, two-year v6 reuse benchmark at commit `19087b1`. All five
workers and `20/20` point-years passed hash verification and unified
aggregation. First-process cold entries averaged 2148.84 seconds total;
same-point hot capture averaged 82.73 seconds; after switching landpoints,
capture averaged 91.07 seconds with no compiled-cache growth; and the second
point's hot year averaged 81.97 seconds. Peak RSS was about 31.2 GiB per
worker, and five staggered workers completed safely on one node. The accepted
global shape is 100 workers on 20 nodes, with about 8.6-10 hours projected
wall time and about CNY 60 expected CPU charge. See
[`teacher_v6_reuse_benchmark_20260806.md`](research/daily_coarse_graining/teacher_v6_reuse_benchmark_20260806.md).
The pilot details remain in
[`coherent_teacher_pilot_20260806.md`](research/daily_coarse_graining/coherent_teacher_pilot_20260806.md).

Full v6 production and admission are now complete. The accepted aggregate has
669 landpoints, 50 noleap years, 33,450 point-year shards, and 12,208,581 daily
transitions. Its dataset-manifest SHA256 is
`5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99`.
All base and typed shards, metadata, input identities, checkpoints, and
year-to-year links passed aggregation. Finalizer job `14500801` completed in
`02:17:57` with exit code zero. Target representation has zero persistence
mismatches; statistics use 8,591,565 train/train samples only; and a real
batch-8 JAX smoke produced finite predictions, loss, and gradients. The
acceptance-report SHA256 is
`8784ed42dc64ef77c9642f613eb2e5940635edc6e744a1af4ea8634293baa8e3`.
See
[`teacher_v6_full_production_acceptance_20260807.md`](research/daily_coarse_graining/teacher_v6_full_production_acceptance_20260807.md).

Explore1000 runtime cleanup subsequently archived the 100 historical v5 worker
directories as individually verified tar files while leaving the formal v6
product unchanged. The operation reduced the project runtime by 153,803 files
and 21.475 GiB. Historical v5 scripts must restore the required worker tar
before reading its old loose-file path; v5 remains prohibited as an input or
join source for formal v6 training. Archive paths, hashes, restoration, and
post-cleanup v6 evidence are recorded in
[`storage_cleanup_20260807.md`](research/daily_coarse_graining/storage_cleanup_20260807.md).

The former immutable-parent Gate E2 attempts below are historical negative
evidence. They explain why old v5 state rows cannot be joined to labels from a
new Teacher execution. Job `14481274_0` stopped in preflight
because the historical generation-plan configuration predates the current PFT
catalog; execution is now correctly bound to the current repository
configuration while the historical plan owns only the immutable input
inventory. Replacement job `14481314_0` passed preflight and cold start but
stopped after the first captured transition: current-Teacher Day 2 end state
is not bit-exact to the parent shard generated at Teacher commit `7397d1e`.
It ran for `00:10:02`, peaked at about 9.22 GB RSS, and wrote no sidecar. The
bounded diagnostic subsequently classified the difference in job
`14481448_0`: only `cn_ind` and `ind` differ, both by one ULP; the global
maximum absolute and relative errors are `2.842170943040401e-14` and
`2.063503050227843e-16`, and all discrete state is exact. This is safely
inside Gate A's existing `atol=1e-12, rtol=1e-12` policy and is not a
scientific Teacher regression. The historical design then reused that
declared continuous policy, retained exact discrete comparisons, and kept
sidecar encoding itself bit-exact. Its planned task-0 continuation was later
abandoned in favor of coherent regeneration.
The first full task-0 submission, `14482172_0`, stopped before Python execution
because the cluster's older Bash treats an empty array expansion as unbound
under `set -u`. The launcher now uses explicit diagnostic and full-generation
branches and has a regression guard against reintroducing the empty-array
dispatch. No Teacher transition or new sidecar was produced by that job.
Replacement `14482331_0` entered Teacher execution and stopped at Day 21. The
only examples outside the Gate A combined tolerance were three mirrors of
`soilflx`, with absolute error `4.5773e-11` and relative error `1.3124e-12`;
all discrete state remained exact. This is accumulated free-run roundoff from
joining a current-Teacher trajectory to an immutable old-Teacher parent, not
an admissible sidecar transition policy. The tolerance was not widened.
That teacher-forced workaround was not promoted. Historical v5 remains
experiment evidence only; the coherent release regenerates both sides of the
training record together. See
[`coherent_teacher_dataset_generation_v1.md`](research/daily_coarse_graining/coherent_teacher_dataset_generation_v1.md).

Completed infrastructure:

- Daily Fast-Day Teacher Contract v5:
  `S[d] (3,854 continuous values plus 12 exact discrete values) + native
  6-hour forcing[d] + P -> B_fast[d] (2,855 values)`, followed by retained
  source-backed daily season/STOMATE to produce `S[d+1]`;
- only the true cross-day subset of `sechiba_finalize_state`, including
  `leaf_ci`, is carried in `S[d]`; the 93-field finalize packet is no longer a
  duplicated learned output;
- canonical PFT14 state trajectories with PFT1/PFT14 axis compaction,
  exact discrete state, deterministic mirror reconstruction and year-start
  `nroot` handling;
- exact native-forcing window reconstruction of all 48 Teacher input steps;
- named condition slices for 84 parameter values, 735 physical/static
  landpoint values, annual CO2, year, and day index;
- a machine-audited ownership ledger covering every compiled complete-day
  Teacher argument;
- a hash-verifying training reader that enforces frozen splits, excludes
  landpoint identity, computes train-only streaming statistics, excludes
  NaN/inf and ORCHIDEE `+/-1e20` sentinels, and uses bounded prefetch;
- a persistence-centered residual target representation with exact
  restoration of persistent undefined values and a dedicated classifier for
  source-dynamic `diffuco.rveget` defined/undefined transitions;
- pure-JAX gradient and checkpoint plumbing;
- a hash-bound post-generation acceptance gate that requires a complete v4
  aggregate, audits target representation, fits sentinel-aware train-only
  statistics, and passes a real-batch finite forward/loss/gradient smoke
  before GPU training;
- compiled Teacher training capture with reduced host transfers;
- restartable, atomic landpoint-year NPZ shard generation with provenance,
  frozen spatial/temporal splits, resume, and aggregate validation;
- a frozen 669-point production specification and a first five-point,
  1961-2010 architecture-development subset with two train, two validation,
  and one test landpoint;
- Linux CPU and V100 compatibility tests.

Current scientific status:

- the neural path is technically connected but is not a validated daily
  surrogate;
- the contract-v4 nine-point V100 baselines established
  nontrivial one-step learnability: its equal-split validation score was
  0.812 versus 1.064 for the matched persistence baseline, but spatial and
  joint scores remained 0.855 and 1.438;
- that run exposed an absolute-state `rveget` undefined classifier which
  damaged a stronger persistence prior; the accepted follow-up predicts only
  defined/undefined flips and reduced temporal/spatial/joint classification
  errors to `18/0/0`;
- a fresh 15-epoch run reached its best equal-split score of `0.772` at epoch
  10, improving temporal/spatial/joint scores over matched persistence by
  `67.0%/28.0%/18.0%`; additional epochs did not resolve the dominant spatial
  DIFFUCO/ENERBIL and HYDROL errors;
- no user-facing `daily-surrogate` run mode exists;
- the first seven-day v4 free rollout reached `0.921` normalized state RMSE,
  while teacher-forced daily error stayed near `0.11-0.15`; this identifies
  recursive state-distribution drift rather than a retained-tail handoff error;
- contract v4 was then found to omit the fast-owned cross-day `leaf_ci` output.
  Contract v5 appends it losslessly from stored `S[d+1]`, so no Teacher rerun
  is required, but all v4 statistics and checkpoints are superseded;
- the pure-JAX retained-tail transition passes a real Day 2 forward and
  reverse-mode gate: maximum next-state error `5.68e-14`, zero mask/discrete
  mismatch, and finite gradients for all 2,847 defined target values;
- a real three-day `lax.scan` gate also passes: maximum trajectory error
  `1.71e-13`, zero mask/discrete mismatch, and finite gradients for all 8,541
  defined day-target values;
- the complete nine-point v4 asset has been losslessly migrated to v5 without
  rerunning Teacher: 450/450 landpoint-year shards and about 1.4 GiB passed
  source/output hash verification;
- the v5 dataset acceptance gate passed on 96,354 train/train transitions with
  zero target-representation persistence mismatches and finite forward, loss,
  and gradients for the 1,963,369-parameter canonical model;
- a real V100 one-window multistep smoke passed at research commit `f68ea59`:
  one horizon-1 update completed in 104.84 seconds including first compile,
  with loss `0.0495625`, gradient norm `0.389538`, and a finite 31.4 MB
  checkpoint; this also closed canonical generation-plan hashing and external
  server data-root wiring in the multistep launchers;
- the first complete v5 V100 curriculum finished on `gln01` in 35:03. Its
  ten-epoch one-step initialization selected epoch 10 at validation score
  `0.680439`; the subsequent `1:64,3:64,7:64` stages remained finite;
- on the joint spatial-and-temporal validation window at landpoint
  `215.0-119.0`, year 2005, days 100-106, the seven-day free-rollout final
  state RMSE improved from `0.473270` for the one-step checkpoint to
  `0.429058` after multistep training, with zero defined-status or discrete
  mismatches. Teacher-state feedback kept all daily RMSE values below
  `0.086930` and ended at `0.068236`, so recursive state-distribution drift,
  rather than an invalid retained-tail handoff, remains the dominant error;
- a complete field-level drift diagnostic at research commit `18471aa`
  reproduced every daily rollout RMSE exactly. The current multistep state
  loss is uniform over 3,854 scalar values: slowproc/STOMATE owns 2,386 values
  and 61.91% nominal weight, but already contributes 76.89% of the first-day
  realized Huber state loss. Biomass alone contributes 36.39% on Day 1 and its
  normalized RMSE grows from `0.660380` to `4.841768` by Day 7; growth and
  maintenance respiration, NPP, litter partitioning, and surface energy state
  are also early divergence leaders. This rules out the simplistic diagnosis
  that important carbon state is merely invisible to the loss. The next
  objective must combine process-balanced weighting with explicit recursive
  stability/state-change supervision, then be tested under a controlled A/B;
- a four-way split diagnosis at research commit `825935b` compared
  train/train, train/validation-year, validation-point/train-year, and joint
  validation rollouts on the same Day 100-106 window. Day-7 RMSE was
  `0.145471`, `0.134233`, `0.519886`, and `0.429058`, respectively. The
  controlled temporal holdout did not degrade the training landpoint, while
  the spatial holdout failed even in a training year. Spatial-condition
  coverage is therefore a demonstrated limitation of the current six
  training landpoints. Recursive objective failure also remains: train/train
  `litterpart` grows from `0.541` to `3.652`, proving that the global `<=0.29`
  gate can hide a scientifically important low-dimensional failure. Do not
  add more years of the same points; first run one process-balanced/state-
  increment objective A/B, then add a bounded, source-selected spatial pilot;
- the bounded `process_increment_v2` equal-budget A/B completed and is
  rejected. On train/train Day 7, global normalized state RMSE changed from
  `0.145471` to `0.146456`, biomass from `0.014259` to `0.016716`, and
  litterpart from `3.651536` to `4.008877`. The candidate retained zero
  defined-status and discrete mismatches but failed all required seen-condition
  improvement gates. `canonical_multistep_v1` remains the baseline; do not
  continue tuning process/increment loss weights;
- the counterfactual Teacher-at-model-state gate passed at research commit
  `87e1132`: exact Teacher re-entry completed at model-generated Days 2, 4,
  and 7 with zero defined-status or discrete mismatch. Mean matched operator
  error was `0.085390`, mean state sensitivity was `0.073494`, and their ratio
  was `1.161865`. This supports stop-gradient pushforward with same-state
  on-policy Teacher labels;
- the real detached-prefix pushforward smoke passed at research commit
  `c3e2d79`. After a three-day model prefix, one exact same-state Teacher query
  and final-transition update produced finite loss `0.0836542`, gradient norm
  `0.1333402`, and a finite checkpoint;
- the corresponding 192-update, batch-4 `on_policy_pushforward_v1` A/B is
  complete and rejected. On train/train Day 7, global RMSE improved from
  `0.145471` to `0.092764` and litterpart from `3.651536` to `0.854632`, but
  NPP and growth respiration regressed about 9.6%. Spatial-validation global
  RMSE worsened from `0.519886` to `0.550696` and joint validation from
  `0.429058` to `0.459871`; spatial litterpart also worsened by more than 60%.
  All mask/discrete gates passed, but the predeclared named-state and spatial
  gates did not. Keep `canonical_multistep_v1`; do not tune this candidate or
  inspect the sealed test split;
- deterministic replay of that run's single extreme batch proved a separate
  physical-domain defect in neural rollout reconstruction: the unconstrained
  network produced negative `carbon_32l`, `DOC`, and `deepC_peat` stocks before
  exact Teacher re-entry, causing NaNs and a `7.27e8` soil-carbon artifact.
  Commit `5414d8c` now applies the source-backed nonnegative-stock projection
  at the shared NumPy/JAX physical reconstruction boundary. The exact four
  replayed losses changed from `0.0997/0.5658/0.0984/562889` to
  `0.0999/0.0946/0.0970/0.1165`, with defined-status mismatches falling from
  980 to zero;
- the frozen four-way seven-day baseline was rerun after that semantic fix.
  Day-7 RMSE is now `0.144765`, `0.133391`, `0.519286`, and `0.428353` for
  train/train, train/validation-year, validation-point/train-year, and joint
  validation. The small changes leave the scientific diagnosis intact:
  spatial generalization remains dominant and train/train `litterpart` still
  reaches `3.651537`;
- the matched `structured_process_film_v1` architecture A/B is complete and
  rejected. Relative to an identically continued flat control, structured
  Day-7 RMSE improves 4.36% and 4.24% on the two spatial-validation cases but
  regresses 7.68% and 11.32% at the seen landpoint. Its maximum useful
  parameter/static validation permutation effect is only `5.26e-5`, below the
  predeclared `1e-4` condition-use threshold; parameter permutation slightly
  improves loss, showing a wrongly learned association rather than a dead
  channel. The flat continuation itself worsens all four frozen global scores,
  so neither checkpoint is promoted. See
  `structured_architecture_ab_result_20260726.md`;
- the reusable 669-point Teacher data-product admission now passes at the
  contract level. The versioned machine gate binds the frozen 535/67/67
  spatial split and 1961-2010 temporal split to Markov Contract v5, verifies
  all state/target/diagnostic/forcing/condition inventories, and computes
  33,450 point-years with 12,208,581 transitions. Its final mode additionally
  requires all 33,450 aggregate shards and exact split assignments. The next
  data operation is therefore full 669-point baseline generation after
  staging/plan/resource approval, not another architecture-specific spatial
  pilot. Parameter perturbations and new forcing products remain separate
  parent-bound extension datasets;
- the first real bounded 669 production probe generated and hash-verified the
  complete `001.0-071.0:1961` v5 shard at contract SHA256
  `6813065b...9d442`. It took 3084.5 seconds for the entry and reached about
  29.5 GiB peak process RSS. Slurm's exit code 2 was an orchestration defect:
  the old CLI reported an intentionally partial `--max-new-entries` worker as
  failed after successfully writing its resumable manifest. That exit
  semantics now has a regression fix. The `07fe2f9` probe remains immutable
  admission evidence; full production must use one fresh output root bound to
  the accepted post-fix commit rather than mixing commit identities;
- the post-fix admission at commit `7397d1e` passed: Slurm task `14390213_0`
  completed with exit code zero in 42:43, the entry took 2381.0 seconds and
  reached about 29.5 GiB peak process RSS. Its NPZ SHA256 exactly matches the
  original probe, proving the orchestration-only change did not alter
  scientific arrays. This five-worker output remains immutable admission
  evidence and is not mixed into the later 100-worker aggregate;
- the accepted full-production topology is now 20 `cnall` nodes with five
  one-CPU workers per node, for 100 concurrent workers. The launcher commit is
  `74bd4eb`, while the separately validated Teacher worktree remains fixed at
  `7397d1e`. Five-minute within-node startup staggering reduces overlapping
  cold-compilation memory peaks. The validated `w100` plan SHA256 is
  `9ba2caf5...5fa17e`; 69 workers own 350 point-years each and 31 own 300 each;
- the first two-node admission job `14391204` exposed default `srun`
  first-task wait semantics, not a Teacher failure. The explicit `--wait=0`
  fix was then accepted by rerun `14391639`: all ten ranks exited zero in
  55:46, all 11 accumulated point-years and hashes passed, no active lock or
  manifest error remained, and worker 0 closed the 1961-to-1962 restart chain.
  Cold workers peaked near 31 GB RSS and the resumed worker near 26.2 GB;
- consecutive-entry calibration `14391864` passed in 40:03. Its first fresh
  process entry took 1743.348 seconds, while the next three years in the same
  process took 158.455, 158.915, and 160.493 seconds. Mean steady-state
  production is therefore 159.288 seconds per point-year, including capture,
  compression, hashes, and checkpoint output, or about 0.436 seconds per
  simulated day. The post-run hash audit found 15 valid point-years and no
  active lock or manifest error;
- production recovery and consumption are prepared: a progress auditor
  identifies running/partial/missing/invalid workers, arbitrary worker IDs can
  be resumed without changing the canonical 100-worker assignment, and final
  aggregation now requires complete worker inventories, no active lock, all
  shard/metadata/checkpoint hashes, and exact checkpoint chains;
- the complete-data training protocol is frozen at
  `manifests/coarse_graining/daily_teacher_669_training_protocol.json`.
  It specifies exact-once balanced landpoint/year/day streaming with at most
  eight open shards, train/train-only normalization, four non-test
  model-selection slices, sealed final testing, and one-step through
  complete-chain promotion gates. A local real-shard streaming preflight
  passed; it is an I/O gate, not neural accuracy evidence;
- the literature-backed post-admission architecture decision is frozen in
  `long_rollout_architecture_review_20260727.md`: first compare a matched flat
  control with a process- and axis-aware operator, then add mixed-horizon and
  explicit tendency-bias training only if the architecture passes. Long-term
  acceptance targets bounded physical drift and unbiased budgets, not an
  impossible promise of mathematically zero accumulated error;
- the provisional seven-day free-rollout gate of `<=0.29` therefore did not
  pass. No v5 30-, 365-day, or 50-year neural rollout has passed;
- generated labels remain `provisional_teacher` until the 669-point Teacher
  acceptance gate is complete.
- the first five-epoch V100 run is rejected as architecture evidence because
  it used contract v3 and statistics v1, which treated `rveget=1e20` as an
  ordinary regression value and duplicated finalize mirrors in `B_fast`;
- the accepted five-point v3 Teacher dataset remains provenance evidence, but
  it and its v1 statistics/checkpoint must not be reused for v4 training.
- old v1/v2 shards are audit evidence only and must not be expanded into the
  current v3 dataset.

## CPU and GPU Decision

One-landpoint V100 Teacher capture reached about `0.263 s/day` hot for a
28-day block, with about `922.6 s` cold compilation. This does not establish
that GPU is faster than CPU because the retained CPU measurement was not a
matching same-process hot benchmark. It does establish that one landpoint
does not provide enough parallelism to justify GPU Teacher generation.

Current policy:

- use persistent CPU workers for Teacher shard generation unless a measured
  batched-landpoint GPU implementation changes the result;
- use the accepted seven-day complete-day block for CPU Teacher generation;
  the 28-day capture result is a historical V100-specific experiment;
- use GPU for neural-network training;
- keep inference backend-neutral and benchmark CPU latency versus batched GPU
  throughput after the network architecture is frozen;
- do not describe the research branch as a delivered GPU model.

The project-owned Explore1000 `orcjax_gpu` environment passed its real-V100
runtime gate on 2026-07-23. JAX 0.4.38 selected backend `gpu` and executed a
finite JIT loss and gradient after explicit PJRT plugin discovery. The
canonical verification and training entry points now share that fail-fast
initialization and reject CPU fallback. This validates the training runtime;
it is not evidence that the neural surrogate itself is scientifically valid.

## Repository and Release Architecture

The intended user-facing release is one codebase with two independent runtime
choices:

- execution backend: `cpu` or `gpu`;
- transition implementation: `teacher_half_hour` or `neural_daily`.

CPU and GPU support must not become permanent model forks. They share the
same state, forcing, parameter, restart, output, and acceptance contracts;
only batching, compilation, and device execution may differ. The stable
`main` branch contains accepted user-facing implementations. The
`research/daily-coarse-graining` branch remains a reproducible development
line for dataset generation, network training, experiments, and promotion
evidence, not a separate product.

A released neural checkpoint cannot be the sole training artifact. Its
provenance must bind the Teacher commit, dataset and split manifests, state
and input schema hashes, train-only normalization statistics, architecture,
optimizer and schedule, random seeds, environment lock, checkpoint hash, and
rollout acceptance report. Reproduction code and compact manifests belong in
Git; large datasets and weights remain external assets with stable identifiers
and SHA256 hashes. Accepted training and inference code is merged into
`main`, while ongoing experiments continue on the research branch.

Canonical-Teacher physical-parameter gradient Gate D1 is accepted. Its matrix
covers seven source-backed process pairs, seven complete Day-2 pairs, three
two-day propagation pairs, four restart-year pairs, and two source-extracted
Fortran finite-difference pairs. JAX forward and reverse AD agree with centered
`h`/`h2` Richardson finite differences under the frozen 1% active-pair policy;
inactive and threshold-adjacent cases are classified separately. The gate
found and repaired dormant-branch reverse-AD pollution across half-hour state
and a traced `altmax` host cache conversion at the yearly HYDROL first step.
The accepted evidence and exact scope are documented in
[`gate_d1_physical_parameter_gradients_v1.md`](research/daily_coarse_graining/gate_d1_physical_parameter_gradients_v1.md).

Gate D1 does not validate physical-parameter gradients of a future daily
surrogate. Before neural parameter calibration, sensitivities, or Jacobian
claims, Gate D2 must compare surrogate AD against canonical Teacher/Fortran
finite differences over frozen parameter ranges and held-out combinations.
Network-weight and cross-day state gradients remain training checks and cannot
satisfy Gate D2.

The source-driven physical-parameter candidate registry is accepted. It binds
34 parameter families and 91 scalar components to hashed Fortran source and
classifies them as 9 inversion candidates, 17 sensitivity-only families, 2
discrete controls, and 6 currently non-identifiable families. The first wave
remains `VCMAX25`, `MAINT_RESP_SLOPE_C`, `ALLOC_MIN`, and `RESIDENCE_TIME`.
Five second-wave PFT14 families require priors, controlled perturbations, and
new gradient evidence before promotion. Registry membership does not expand
the public parameter interface. See
[`physical_parameter_candidate_registry_v1.md`](research/daily_coarse_graining/physical_parameter_candidate_registry_v1.md).

## Teacher Production Evidence

Explore1000 admission measurements established the current production policy:

- a first eight-day landpoint compiled and captured in 1,406.85 seconds;
- a second landpoint in the same process captured in 3.35 seconds and took
  40.15 seconds including preparation, proving executable reuse across
  equal-shape PFT14 landpoints;
- a cold 1961 full year took 2,395.32 seconds;
- the following in-process 1962 hot year took 102.30 seconds to capture and
  139.20 seconds including compression, hashing, checkpoints, and shared IO;
- peak RSS reached about 29.4 GiB, so the initial total concurrency cap is five
  persistent workers per approximately 187 GiB node;
- persistent compilation cache did not eliminate compilation in a new Slurm
  process, so production relies on same-process reuse rather than cross-job
  cache reuse.

The first bounded production target was the frozen five-point, 1961-2010 v3
dataset: 250 landpoint-year shards and 91,245 daily transitions. Its first
submission was rejected before acceptance because the production planner
incorrectly inserted Gregorian leap days into the paper model's fixed 365-day
noleap calendar. The planner, plan validator, and training calendar features
now enforce noleap semantics. The clean replacement completed and passed its
aggregate gate on 2026-07-23: all five 1961-2010 chains, 250 shards, 91,245
transitions, source/input hashes, checkpoint links, splits, and the single
Markov contract are valid. Its operational evidence is recorded in
[`research/daily_coarse_graining/HANDOFF.md`](research/daily_coarse_graining/HANDOFF.md).

## Historical Neural Milestones

The material below records completed v3-v5 dataset and rejected-model
evidence. It is not the active work queue. Gate E1 is accepted; the coherent
Gate E2 coherent production and formal v6 dataset admission have passed. The
first formal neural pair is closed without promotion. The long-horizon and
parameter-routing foundations are complete. The successor options analysis is
complete, the pair is selected, and the local synthetic foundation is
accepted. The real-v6 `gln01` smoke is the active stop point before fitting.

The full 669-point baseline is generated and admitted. Finalization job
`14400343` completed in 4:08:54 with exit code zero. Its fail-closed gates
accepted all 33,450 point-years and 12,208,581 transitions, found zero target
persistence mismatches, fitted statistics from the 8,591,565 train/train
samples only, and passed a finite real-batch JAX forward/loss/gradient smoke.
The dataset manifest SHA256 is
`89ea2f24bad8f4b39129937f4106285dd53936c35d8dc610810f865b02d4e508`;
The statistics JSON SHA256 is
`c9ed1c6ea4cfca6c0da9504a36d440a2b388375f01acf8bc78edc7645a4ffb37`;
the statistics NPZ SHA256 is
`33865286021c39bbdb6511a115da55143789e857c31e701c7417ff1a9c50cdbd`;
the acceptance report SHA256 is
`0b13da5c7f8664f9b3b8fe70f6b8dc8e3aca513ecbf15d4b3d879a07d663b596`.

The matched `canonical_flat_v1` versus `axis_process_coupled_v1` architecture
A/B is complete and passed. Both arms used the same admitted samples, update
budget, optimizer, schedule, seed, physical reconstruction, and canonical
objective. Teacher shards were not regenerated and the sealed test split was
not inspected.

The architecture screen was frozen in
`canonical_669_axis_process_architecture_ab.json`, canonical SHA256
`39b94a28e51dba67017ac3f06973b7f4fbbfa4a560b249eec49420f4baed8099`.
It uses one exact-once train/train epoch, batch size 256, learning rate
`1e-4`, seed `20260728`, full temporal/spatial/joint one-step validation, and
no test samples. The production trainer now supports both registered
architectures through the same loss and balanced reader, performs one shared
asset verification for the two-arm run, and avoids per-batch host loss
synchronization.

The prepared real-shard V100 compile/update smoke passed on `gln01` at commit
`0992178`. It hash-verified the train/train shard `001.0-071.0:1961`, used
zero-based sample indices 0-255 as one batch of 256, and consumed no
validation or test sample.
Both arms retained float32 parameters and finite losses. Flat compile/hot
updates took `2.2506/0.00862 s`; axis/process compile/hot updates took
`8.7199/0.00757 s`. The report SHA256 is
`3975eb93000b355b51c565f421f04fdc19342032360c1610e1823dce607f3d16`.

Full job `14403674` completed in `03:00:49` with exit code zero from clean
commit `4c1fe0c`. The final report SHA256 is
`1995fb7f9284bde56c8ac5a9cd411d1bf08fc94f795a0ab1fdad69b42814c9a7`.
Axis/process passed every predeclared gate and produced decision
`advance_axis_process_to_rollout_stability_experiment`. Its
temporal/spatial/joint score ratios relative to flat are
`0.943530/0.960916/0.952866`; the maximum process-family ratio is `0.990947`,
the parameter ratio is `0.995249`, and test evaluation is false. The selected
axis/process checkpoint SHA256 is
`a119999606b063ac9f9ed47e4d1bb664cdfe32b9459e7e7ee24e96a0e944db2e`.

The production screen now performs the full acceptance/hash preflight once,
runs the flat and axis/process arms concurrently on two isolated V100 devices,
and then applies the unchanged aggregate gates. The original single-process
runner remains a sequential fallback. A real-shard two-process smoke passed on
both free `gln01` V100s at commit `ef3ddda`: both reports retained float32
parameters and reproduced identical compile/hot losses for both architectures.
The GPU 0/GPU 1 report SHA256 values are
`ec66dac4be68cc0afc12dac5cd59efc2b3b6932377c999fa6ccc6665ed9a680a`
and
`4001937f2d22916755d9646cef7227cdb26496b5a6375246d8349070a0322351`.
The launcher is restart-safe under the same output root: an existing preflight
must match every frozen input exactly, completed arm checkpoints/reports are
resumed without another epoch, incomplete arms rerun, and worker logs append.
The 12-hour Slurm limit is retained as a safety ceiling.

The conditional post-architecture rollout-stability experiment was frozen
before the parent result in
[`canonical_669_rollout_stability_protocol.json`](../manifests/coarse_graining/canonical_669_rollout_stability_protocol.json),
Its v1 canonical SHA256 was
`370011f6d8edc447bd0ebf037249df1a18f3bfb30071204001366a2aecde310b`.
Its auditor binds the admitted 669-point assets and architecture-screen hash,
keeps the test split sealed, and rejects drift in the matched arms,
`1/3/7/30`-day sampling, loss inventory, train-only coefficient calibration,
hard constraints, screening thresholds, promotion ladder, or stop rules.
Experiment B runs only if the parent decision is exactly
`advance_axis_process_to_rollout_stability_experiment`; a rejected architecture
returns to a named architecture hypothesis rather than silently applying the
stability objective to the flat model.

The Experiment B training implementation and real-shard GPU gate now pass.
Horizon `1/3/7/30` protocol shapes completed on a V100 with two distinct train
landpoints sharing one executable, finite first and second updates, zero hard
counts, and exact checkpoint resume. Horizon-7 and rematerialized horizon-30
hot updates take about `0.10 s` and `0.20 s`, with peak allocator use below
`0.55 GiB`.

Host preparation was then reduced without changing any measured loss,
component value, gradient norm, or hard count. Each shard is loaded once,
runtime/static state is cached by landpoint, and the five forcing leaves
actually consumed after the learned fast-day boundary are reconstructed in
one vectorized batch operation. Steady forcing preparation fell from seconds
to about `0.015 s`; a complete warm-cache horizon-7 preparation is `0.245 s`.
A 16-landpoint cache gate measured about `7.8 MiB` RSS growth per added
landpoint, making the 535-train-landpoint cache safe on a gnall node. Details
are in
[`rollout_stability_smoke_20260729.md`](research/daily_coarse_graining/rollout_stability_smoke_20260729.md).

Preflight, calibration, execution, and arm checkpoint identities now bind the
training Git HEAD in addition to protocol, data, schedule, model, and
environment.

The first paid Experiment B attempt, job `14410820`, failed after `00:07:47`
during train-only coefficient-calibration ordinal 4, before either training
arm started and before any checkpoint was written. The exact batch was
landpoint `087.0-105.0`, year 1961, horizon 1, batch size 256. It contained
three and only three defined-status errors, all in the PFT14 column of
`diffuco_previous_step_state.rveget`, on Days 2, 294, and 358. There were zero
discrete mismatches, nonfinite defined values, or negative source-constrained
carbon stocks. The sealed test split was not used. The diagnostic SHA256 is
`f24eeef2f937c2f828b4b7449cdadbaa5d60ef8cf581d186564eb622e6751a2e`.

This exposed a protocol contradiction rather than a new retained-tail defect:
`rveget` defined/undefined is an explicitly declared learned classification
target, but v1 also treated every classification mistake as an exact-zero
optimizer veto. Protocol v2, canonical SHA256
`a0ecfd4c89ff3c5691f153168f3ba80939582774689b92ff04b0cf77e699cb40`,
separates declared dynamic `rveget` status errors from unexpected structural
status errors. Declared errors remain supervised, counted, reported, and
screened against the matched control; they no longer veto an optimizer update.
Unexpected status errors, discrete mismatches, nonfinite defined values, and
negative source-constrained carbon stocks remain exact hard failures.

The required free `gln01` gate passed at commit `13f09c9`. It reproduced
calibration ordinal 4 with exactly three declared dynamic errors, zero
unexpected/discrete/nonfinite/negative-stock hard counts, finite loss and
gradient norm, and `update_applied=true`. The sealed test split was not used.
The smoke report SHA256 is
`66dc952abef5804eabe43ca8a9e9144828caba9649452557ded274f2bf8146a5`.

A later paid attempt reached mixed-horizon update 226 before failing on two
rollout samples with nonfinite gradients. Source-owner isolation proved that
the failure came from the NPP negative-stock add-back landing one ULP above
`min_stomate`, which activated a strict leaf-fraction division gate. The same
pathology was reproduced from the original Fortran procedure under both
`gfortran -O0` and `-O3`. Commit `f9554c6` preserves the source carbon budget
while pinning the corrected stock exactly to the intended threshold.

After that stabilization, both failing samples had finite gradients, the
complete update-226 audit passed 64/64 rollout samples with all hard counts
zero, and the candidate update was applied. A resumable sequential prefix gate
then passed all 256 mixed-horizon updates at commit `452b743`, including an
exact checkpoint restart at update 128 and coverage of horizons 1, 3, 7, and
30. The sealed test split remains unused.

Formal Experiment B rerun `14419242` completed the full 8,192-update control
but failed closed at candidate update 302. Exact checkpoint replay and
horizon-prefix bisection localized the first bad transition to 1973 Day 355.
The forward model remained finite; reverse-mode AD failed in the STOMATE
leaf-age turnover quotient at a `3.7058e-311` leaf stock. Commit `90113c0`
preserves the source-equivalent quotient primal and masks only quotient
tangents below the float64 representable-gradient boundary. The complete
update-302 audit then passed with zero bad samples, finite independent
component gradients, all hard counts zero, and an applied update.

An identity-checked diagnostic checkpoint fork at commit `b8c8dee` resumed the
formal update-256 state and passed every mixed-horizon update through 1024.
All 768 post-fix updates were applied; horizon counts were `431/317/182/94`
for 1/3/7/30 days, all exact hard counts stayed zero, and the sealed test split
was not used. The failed formal root remains rejected. See
[`rollout_turnover_ratio_gradient_20260730.md`](research/daily_coarse_graining/rollout_turnover_ratio_gradient_20260730.md).

The clean formal Experiment B rerun is complete. Slurm job `14425151` ran on
one V100 from training commit
`d6e43cc79f590e494f79fe54adba4ac2c28779b6`, completed in `04:16:53`, and
returned exit code zero. Both matched arms completed all 8,192 updates with
identical horizon counts `3277/2458/1638/819` for 1/3/7/30 days and did not
access the sealed test split. The immutable output root is
`runtime/outputs/training/canonical-669-rollout-stability-v2-d6e43cc`.
Control/candidate checkpoint SHA256 values are
`8a099c85f8c869398ffe2203b0fa1d558735b0f819ac606d112d452db2fb2ec1`
and
`4f36bb39839f8fba7a9cbd0283610bced3bd45c0688891bcdb4d051917320300`;
their training-report SHA256 values are
`6ca7985edd5b5b61beecd534a7f341a7173a6ef549b39ddccca3b93be2237069`
and
`ffd25be74ab739d15c45f66d04fc79d304e4877821b91f23f12d85da5749184c`.
This proves successful matched training, not promotion.

Commit `a9d2390` adds the fail-closed full-data screening evaluator. Local
validation passes 64 tests, Ruff, `py_compile`, shell syntax, and diff checks.
Real V100 smokes cover temporal, spatial, and joint model-selection shards,
large batches `512/512/256`, and a bit-exact 30-day versus `15+15`
restart-split probe with maximum difference zero. The evaluator covers every
valid 1/7/30-day teacher-forced and free-rollout window, the complete state,
process families, tendency bias, 12 named science fields, dynamic status and
hard constraints; incomplete evidence cannot be classified.

The formal matched screen completed as Slurm job `14430377` in `04:21:26`
with exit code zero. All 4,754 point-year references and every valid
teacher-forced/free 1/7/30-day window were evaluated; the report SHA256 is
`1387448b0b0831661daaaffc8985ef22602540e0c7e91fea7bfe96c085828bb0`.
The candidate is rejected: 114 of 223 relative gates fail. All one-step global
no-regression gates and all three 30-day global rollout gates pass, with
30-day temporal/spatial/joint error ratios `0.9181/0.7539/0.8047`. However,
none of the 7-day global gates reaches the required 5% improvement, 46/54
tendency-bias gates fail, and 57/108 named science-field gates fail. NPP,
growth and maintenance respiration, biomass, LAI, `carbon_32l`, and
`deepC_peat` are the repeated failures.

All exact hard and structural constraints pass, including zero unexpected
status/discrete/nonfinite/negative-stock errors and a bit-exact 30-day
`15+15` restart split. This is therefore an objective/architecture failure,
not a numerical-integrity or orchestration failure. Do not run confirmation
seeds, inspect the sealed test split, or begin 365-day validation for this
candidate. See
[`rollout_stability_screening_20260731.md`](research/daily_coarse_graining/rollout_stability_screening_20260731.md).

Experiment C is now frozen as a bounded causal-carbon hypothesis, but has not
yet run on a real GPU shard. It keeps the Experiment B one-step control as a
frozen 1,954,041-parameter parent and attaches an exact-zero
76,877-parameter adapter. The adapter can alter only 141 PFT14 columns:
same-day GPP, 12-part maintenance respiration, 32-layer carbon, and deep peat
carbon. The other 2,714 fast-day columns and the dynamic undefined head remain
parent-owned and bit-exact. The objective directly supervises this causal
`B_fast` interface, then separately supervises downstream carbon state,
signed flux bias, and stock-tendency bias; reset day-end `gpp_daily` is
explicitly excluded as a proxy for same-day GPP. DOC and litter are
no-regression guards.

The protocol SHA256 is
`35ede512bbc5f55881317cb700d7512d71993391c65f36ed8a5de087c4d4c3e7`.
Local architecture/objective/protocol regression and a complete synthetic
two-day retained-transition reverse-mode path pass. Paid matched training is
forbidden until an eight-update, real train/train-shard V100 feasibility gate
passes all 1/3/7-day, frozen-parent, protected-column, gradient, and hard-state
checks. This preparation does not establish that Experiment C improves
rollout quality. See
[`causal_carbon_adapter_experiment_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_experiment_20260731.md).

The Experiment C real-shard feasibility gate subsequently passed at commit
`79258fd`. Eight matched train/train updates covered 1/3/7-day horizons on a
free `gln01` V100. Every update was applied with finite gradients; all
unexpected/dynamic status, discrete, nonfinite, and negative-stock counts were
zero. The frozen 1,954,041-parameter parent stayed outside the optimizer, all
2,714 protected columns and the dynamic undefined head remained bit-exact,
and the maximum protected-column difference was `0.0`. Report SHA256 is
`ca8cacdccb42f20b94b5cd64796118cb170f7fbc7813c036cbc6b1edc8aa7fa8`.
The smoke checkpoints are not training parents. Formal train-only
independent-component gradient calibration must precede a fresh matched A/B
from the exact-zero adapter. See
[`causal_carbon_adapter_feasibility_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_feasibility_20260731.md).

The formal Experiment C lifecycle completed as paid job `14434127` at training
commit `8d24fd0`. It ran for `02:33:11` on one `gnall` V100 and exited zero.
All 48 deterministic train/train calibration records and both matched
4,096-update arms completed. Each arm has exact horizon counts
`2048/1229/819` for 1/3/7 days. Both final parent-invariance checks report
bit-exact protected columns and dynamic undefined head with maximum protected
absolute difference `0.0`; neither arm used sealed test data. Evidence hashes:

```text
control checkpoint:
4ee68de505f0163a62231e0310369882d46a0b72663b87d2a394c0e4ffeef720
control training report:
7400144a2f7143d5591e7a1bf2b49cfcd0c50d55385cee71dcd481c554e2170c
candidate checkpoint:
ea7193453e279bd94eb55984c629a0f530d0275466c1cd0daccf142754497892
candidate training report:
0a358a32f9b156c2d2434a0d11b6e0a8a35ceae47f3f6f0fdd7f3b78d7dc0eaf
```

This closes formal training execution only; it does not establish that the
candidate improves model-selection rollout quality.

The post-training screen was frozen before any candidate validation output was
available. It evaluates all temporal, spatial, and joint model-selection
windows at Day 1 teacher-forced and Day 7/30 free rollout. Classification
requires all 165 relative, 108 exact-zero, and six structural gates to pass;
the sealed test remains unread. After training, run a one-shard-per-slice free
`gln01` smoke before requesting the paid all-sample screen. The smoke
subsequently completed on GPU 1 in `286.46` seconds at evaluation commit
`375b8d2`. It covered one complete temporal, spatial, and joint shard for both
arms, with all unexpected-status, discrete, nonfinite, and negative-stock
counts zero. Both restart-split probes passed, and sealed test data remained
unread. The top-level smoke report SHA256 is
`f05d88d97ec93152487f22697e50361de705a6e1223964971bdfccdcd777b7cb`.
As preregistered, this limited smoke is not promotion-eligible; it authorizes
the paid all-sample screen.

The first paid submission, job `14436672`, exited after two seconds before
starting the evaluator because the Slurm wrapper directly executed a
non-executable checkout file. Commit `7091e1d` changes only that invocation to
explicit `bash` and adds a regression test; it does not alter the evaluator or
frozen gates. Replacement job `14436716` completed all 4,754 references in
`04:25:50` with exit code zero from the immutable `7091e1d` checkout.

Experiment C is rejected with decision
`stop_and_attribute_declared_screening_failure`. All 108 hard and six
structural gates passed, but 73/165 scientific relative gates failed:
11/12 causal-interface, 46/63 primary-state, 3/27 flux-bias, and 13/36
stock-tendency gates. All nine global terminal-state and all 18 litter/DOC
guard gates passed. The dominant failure is direct damage to the carbon-stock
boundary: `carbon_32l` interface error worsened by `7.33-9.32x` and
`deepC_peat` by `5.66-7.45x` across all three model-selection slices. The
matched report SHA256 is
`2b636981a5aeb73479c354558c09ed99cf398559c7f315bd56774f7f9822b5dd`.
Do not run confirmation seeds, 365-day validation, complete-chain validation,
or sealed test for this candidate. See
[`causal_carbon_adapter_screening_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_screening_20260731.md)
and
[`causal_carbon_adapter_screening_protocol_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_screening_protocol_20260731.md).

A single post-rejection attribution is now frozen before any successor is
built. It evaluates the rejected checkpoint with only
`carbon_stock_interface` set to exact zero while retaining the trained flux
adapter. The one-shard-per-slice smoke must reproduce the prior control,
recover `carbon_32l` and `deepC_peat`, retain at least 24/27 flux-bias gates,
and keep every hard and litter/DOC guard gate. It does not retrain or revive
Experiment C. See
[`causal_carbon_adapter_stock_ablation_protocol_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_stock_ablation_protocol_20260731.md).

That one-shard-per-slice ablation is complete at commit `7c53f79` and fails
its predeclared gate. It exactly preserves the original candidate's direct
flux interface, and its 27 downstream flux-bias metrics change by only about
`-0.23%` to `+0.31%`. It strongly repairs temporal and joint carbon-stock
error, but spatial `carbon_32l` and `deepC_peat` remain `13.71x` and `16.20x`
worse than the matched control. Thus neither independent stock residuals nor
an unconditional return to parent-owned stocks is acceptable. The next
candidate requires source-backed learned transfers with a conservative stock
update. No paid training is authorized yet. See
[`causal_carbon_adapter_stock_ablation_result_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_stock_ablation_result_20260731.md).

The subsequent source audit closes the ownership ambiguity. Contract v5's
1,376 compact `OK_LEAK` values contain 1,222 independent inventory values,
64 derived `deepC_peat` values, and 90 ratio or overlapping tracer/partition
values. `deepC_peat` is reset from `carbon_32l` before decomposition and must
not be a neural stock owner. Existing shards provide endpoint supervision but
not aggregate transfer labels or the 13 half-hour driver series required by
the exact source-backed scan. At that historical decision point, a bounded
successor predicted those drivers and executed the existing 48-step
`_paper_compiled_ok_leak_fold`. Its capture and replay evidence remains valid,
but the 2026-08-02 true-daily decision supersedes it as the final surrogate
design. See
[`carbon_budget_ownership_audit_20260731.md`](research/daily_coarse_graining/carbon_budget_ownership_audit_20260731.md).

The bounded train-only auxiliary asset and persisted replay gate are now
complete. The accepted plan selects 16 days from each of six training
landpoints across years and process conditions. All `96/96` target-day
captures pass the exact capture interface with no failed record; the
hash-bound reader verifies every source shard, state/target row, report, and
driver NPZ. Replaying each persisted 13-driver sequence through the current
exact 48-step scan reproduces all 14 `ok_leak.*` endpoints within `1e-12` for
`96/96` days. The global maximum absolute error is `4.5401182813264995e-14`;
six records are fully bit-exact and 77 have 13/14 bit-exact endpoints. The
remaining differences are float64 evaluation-order noise below the frozen
tolerance, not relaxed acceptance. The sealed test split was not used.

This closes capture-interface and persisted-scan replay sufficiency. It does
not assert equivalence of a reconstructed full outer seven-day Teacher block:
historical shards were produced in seven-day compiled blocks, and one-day
outer replay is not the accepted verification boundary. The next action is to
close conservation, feasible-perturbation/nonnegative-stock, real
forward/reverse finite-gradient, and bounded tiny-fit parent-no-regression
micro-gates before any paid neural A/B. See
[`ok_leak_auxiliary_capture_replay_20260801.md`](research/daily_coarse_graining/ok_leak_auxiliary_capture_replay_20260801.md).

The source-backed combined carbon-inventory diagnostic is now connected to a
diagnostic-only full-step output of the same 48-step scan. The production
return boundary remains unchanged. A real persisted-driver smoke for
`001.0-071.0`, 1961 Day 2 passes all 48 steps: all values are finite, all five
independent stock families are nonnegative, the diagnostic final carry is
bit-exact to ordinary replay, and maximum absolute closure is `2.9293e-9`.
The closure policy uses the source `min_stomate=1e-8` verdict from
`constantes_var.f90`; relative closure (`1.1884e-8` at the worst low-stock
step) is retained as a diagnostic and is not an invented source gate. Job
`14446767` completed in `00:06:38` with exit code zero. Feasible perturbation,
real forward/reverse gradient, and bounded tiny-fit gates remain.

A deterministic bounded selection tool is now ready for the existing
server-side v5 shards. It admits only the six spatial-train/temporal-train
landpoints and selects 16 days per point: cold-start anchor, low/high extrema
of six reachable dynamic process metrics and rank-space farthest fill.
`shumdiag_peat` and `runoff2peat` remain exact-zero contract fields because
the audited HYDROL-local SAVE branch is statically
`peat_hydro=F/branch_peat=F`; changing forcing or landpoint cannot activate
it. `PERMA_PEAT` carbon redistribution remains in the exact scan. Static
`fpeat` is report-only. The intended 96
days require about 44.58 MiB of uncompressed driver labels. Source shard,
day-start state, and endpoint hashes are mandatory, and sealed references are
filtered before file access. The first real plan
`500979ab...d9166a3` is rejected because its old static-`fpeat` proxy was zero.
The second `f7967d70...09d0f92` plan is also rejected because it required an
unreachable peat-hydrology branch and ranked tied zeros by date. Neither is a
capture asset. The accepted replacement is frozen at
`manifests/coarse_graining/ok_leak_auxiliary_capture_96day_v1.json`, canonical
SHA256 `8ebe5345...3b922c7`. It passes all 19 independent checks over 96 unique
train-only days, including exact selected-row state and target hashes. The
complete capture and replay result is recorded in
[`ok_leak_auxiliary_capture_replay_20260801.md`](research/daily_coarse_graining/ok_leak_auxiliary_capture_replay_20260801.md).
See also
[`ok_leak_auxiliary_capture_selection_protocol_20260731.md`](research/daily_coarse_graining/ok_leak_auxiliary_capture_selection_protocol_20260731.md).

The restartable one-process capture and persisted-replay runners are now
validated on all 96 accepted records. They verify the source generation plan,
stage each day atomically, and resume only hash-identical passed records.
Capture-interface failures and persisted-replay failures are both zero.

The 2026-08-14 gradient-balanced Attention smoke is complete and rejected.
At commit `e22a235`, a free-`gln01` run continued the accepted update-1000
Attention state for 10 matched updates. The candidate used a per-update
minimum-norm convex combination of unit gradients for one-day anchor, NPP,
AGB tendency, and BGB tendency; the control retained the staged-v3 objective.
All 10 candidate raw directions pass the `0.05` minimum-cosine gate, candidate
clipping is `0/10`, and all hard failures are zero. On the frozen six-artifact
7/30-day subset, however, the candidate worsens the four-key NPP/AGB/BGB/live
biomass composite by `8.39%`, worsens every key, and improves `0/6` cells.
Maintenance improves `30.80%`. The report SHA256 is
`d84a6a764efdbd3b4d8cb74d5ddd0965c301b1ea162671a3db1b7a465bb83834`.

This rejects raw-gradient minimum-norm combination followed by inherited
AdamW state; it does not reject the Attention architecture. The conditional
100-update A/B is closed. The next active task is to freeze a zero-update
realized-step audit of whether AdamW momentum/preconditioning preserves the
raw common-descent direction. No further optimizer update, paid job, sealed
test, or architecture change is authorized by this result. See
[`successor_gradient_balanced_smoke_20260814.md`](research/daily_coarse_graining/successor_gradient_balanced_smoke_20260814.md).

The follow-up zero-update realized-step audit completed on free `gln01` at
commit `6cec975`. The raw minimum-norm direction passes `4/4` groups. After
the actual inherited-AdamW transform, minimum component cosines fall to
`0.00039-0.00464` and no group passes the `0.01` direction gate. Reset AdamW
also passes `0/4`; two groups have a negative component cosine. Exact virtual
all-component loss checks pass `3/4` inherited groups and only `2/4` reset
groups. Hard and nonfinite counts are zero, and no proposed state is retained.
The report SHA256 is
`4b987902e4baf40e5223ae9201be416ff8f61f64080309056302ba947bf47037`.

This rejects both inherited and reset AdamW as realizations of the global
common-descent direction. The active gate is a zero-update shared
direct-direction trust-region audit; no optimizer-reset smoke or further
training is authorized. See
[`successor_realized_step_audit_20260815.md`](research/daily_coarse_graining/successor_realized_step_audit_20260815.md).

The direct-direction trust-region audit then completed on free `gln01` at
commit `55c30a3`. With AdamW and all coordinate-wise optimizer transforms
removed, global step norm `0.001` decreases all four component losses in all
four train/train groups. Median relative changes are `-0.0116%` for the
one-day anchor, `-0.1652%` for NPP, `-0.3081%` for AGB tendency, and
`-0.0317%` for BGB tendency. Step norm `0.003` is rejected because one group
regresses by `19.43%`; all larger tested norms are also outside the shared
trust region. Hard and nonfinite counts are zero, no update is retained, and
the report SHA256 is
`ccc4e1fd29ddebf2d55976aec0282c62a14c814c6b148e1d862f9513379e1fbd`.

This is positive local evidence for directly realizing the common-descent
direction, not a promotion result. The active gate is a no-checkpoint
10-update direct-direction smoke at fixed global norm `0.001`, followed by
the accepted non-test 7/30-day subset. See
[`successor_direct_trust_region_audit_20260815.md`](research/daily_coarse_graining/successor_direct_trust_region_audit_20260815.md).

The 10-update direct-direction smoke completed on free `gln01` at commit
`621cf57` and is rejected. All ten proposed updates pass finite and process
hard gates, and every component has a negative median relative change, but
only `6/10` updates decrease every component. The maximum single
batch-component regression is `19.11%`. On the accepted non-test 7/30-day
subset, the NPP/AGB/BGB/total-live composite worsens `1.4348%`, every primary
key worsens, and `0/6` cells improve. Maintenance-respiration RMSE improves
`5.7566%`. No final parameters or checkpoint are retained. Report SHA256 is
`48e8e7da98a5b2c52e045af5653be5aea0bb85f970cb34b6f7ac40a65b50abf7`.

Removing AdamW therefore fixes direction realization but not direction
transfer. The active gate is a zero-update robust multi-batch audit using ten
mixed batches and all 40 batch-component gradients at the frozen update-1000
parameters. See
[`successor_direct_direction_smoke_20260815.md`](research/daily_coarse_graining/successor_direct_direction_smoke_20260815.md).

The accumulated multi-batch audit completed on free `gln01` at commit
`ab7f41a`. One effective update now covers 40 point-years, 320 one-day anchors,
and 40 30-day windows. The direction is clear for the four group-mean
objectives (minimum cosine `0.2736`) and sharply reduces instability: at global
norm `0.0003`, the worst group-component regression is `0.0323%` rather than
the per-batch smoke's `19.11%`. However, only `4/10` groups improve every
component, so no frozen step passes. Norm `0.001` is nonlinear and regresses up
to `28.89%`. Hard and nonfinite counts are zero, and the report SHA256 is
`0601fcd6bc9f51b638bb2de14087be152320e65fa6430e5ddcab254990c0d847`.

This confirms that effective batch size matters but ordinary averaging is not
enough. The active gate is a zero-update group-aware 40-objective common
direction audit. See
[`successor_multibatch_direction_audit_20260815.md`](research/daily_coarse_graining/successor_multibatch_direction_audit_20260815.md).

The group-aware 40-objective audit then completed on free `gln01` at commit
`b0ebda9`. A common direction exists with minimum constraint cosine `0.050356`.
Every tested norm through `0.0003` improves all four components in all ten
groups, and all hard/nonfinite counts are zero. The formal gate is not passed
because the one-day anchor and BGB tendency median decreases at `0.0003` are
`1.6217e-5` and `5.1778e-5`, below the required `1e-4`; NPP and AGB tendency
already pass. No parameters or checkpoint are retained. Report SHA256 is
`f9c55f15dbd4744097dc5bc5089a35447d2cc63fcad41a00650622e3d809a4e3`.

This solves the observed direction-sign conflict and leaves a bounded
finite-step question. The active gate is a zero-update extension at norms
`0.0005, 0.001, 0.0015, 0.002`, frozen before execution. No architecture
change, training continuation, validation read, or sealed-test access is
authorized. See
[`successor_group_aware_direction_audit_20260815.md`](research/daily_coarse_graining/successor_group_aware_direction_audit_20260815.md).

The bounded finite-step extension then completed on free `gln01` at commit
`f0b2235`. Norm `0.002` passes all ten groups and all four component median
effect gates. The one-day anchor, NPP, AGB tendency, and BGB tendency medians
are `-1.2176e-4`, `-1.0193e-3`, `-1.3361e-3`, and `-3.4462e-4`; no individual
component regresses. Hard/nonfinite counts are zero and no parameters are
retained. Report SHA256 is
`a8b448488a25e924a36dc356a0bdba6f6cc213c6509e7e21ac3e4af0635edf9a`.

The active gate is a five-update frozen-scalarization smoke that realizes the
14 positive constraints through eight group-level reverse passes per update.
It must preserve all ten groups locally and improve the established non-test
7/30-day NPP/biomass composite before any rotating-group training design is
opened. See
[`successor_group_aware_step_extension_20260815.md`](research/daily_coarse_graining/successor_group_aware_step_extension_20260815.md).

The parent-frozen scalarization smoke then completed on free `gln01` at
commit `1bbb163`. Update 1 reproduces the accepted direction and improves all
40 constraints. Update 2 at fixed norm `0.002` regresses multiple constraints,
worst group-5 NPP by `0.0861%`, so it is not applied. The one accepted update
improves all six independent 7/30-day cells and each NPP/AGB/BGB/total-live
aggregate, but only by `0.00922%` in the combined primary score. Report SHA256
is `1aeb93ae49e29496c43dfd9589f7e87f6e6be20c214a23a6d266aaca57360ca6`.

The active gate is a bounded backtracking smoke at norms
`0.002, 0.001, 0.0005`. It keeps the eight-VJP realization, stops if no useful
step preserves all 40 constraints, and retains no checkpoint. See
[`successor_group_aware_scalarization_smoke_20260815.md`](research/daily_coarse_graining/successor_group_aware_scalarization_smoke_20260815.md).

The matched backtracking smoke completed on free `gln01` at commit `4118c72`.
At the second proposal, worst regressions scale from `8.6109e-4` at norm
`0.002` to `4.3088e-4` at `0.001` and `2.1552e-4` at `0.0005`. No second
update is applied, and all hard/nonfinite counts remain zero. This closes
smaller fixed-weight steps and requires direction recalibration. Report
SHA256 is
`c2691824b6548be05337620ffb91cab1994b58bd83293a56968d89574a074261`.

The zero-retention exact 40-gradient audit at the state after the first
admitted update completed with minimum cosine `0.0486966`, passing norm
`0.002`, and support overlap `12/14`. Its efficient active-set realization
then passes with 14 component reverse passes and one all-group JVP round. See
[`successor_second_state_recalibration_audit_20260815.md`](research/daily_coarse_graining/successor_second_state_recalibration_audit_20260815.md).
