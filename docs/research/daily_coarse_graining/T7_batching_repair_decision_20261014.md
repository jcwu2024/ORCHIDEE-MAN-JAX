# T7 batching repair decision

Updated: 2026-10-14. **No-go for a local B4 repair candidate.** This bounded
saved-array/source/HLO review selects no prototype and makes no new model run.
The accepted early affine pass is real, but does not locate or repair the later
raw/first-day/T7 discrepancies. B1 is a compared route, not derivative truth.

## Confirmed evidence and unresolved questions

All numerical gates retain `rtol=1e-5, atol=1e-6`; no scale, precision, loss,
sampling, recurrence, capacity or parameter change is selected.

| Boundary / class | Confirmed evidence | Unconfirmed / candidate consequence |
|---|---|---|
| Early summary affine | Accepted GPU12 measurements/91 finite arrays/4 small compiles; independent true-operand FP64 oracle maximum error/allowance0.102922. Cross-route primal/tangent/input-gradient errors0; loss/kernel/bias ratios0.000451/0.000087/0.001580. | Covers original early `(4,257,4) @ (4,384)` summary_input, sample3/raw434 only. Does not exclude later reader64/attention/fusion, or certify original T7. No early-affine repair target. |
| Raw neural finite cross-route | New NumPy pass over all24 original PKLs reproduces each cycle's active tangent failures:434=1769 coordinates/91 fields,2382=10/10,2621=36/10,200=21/14. Worst ratios459.3841/14.5231/19.5719/12.1981. Four primal coordinates fail in DOC incoming_amount/outgoing_fraction (two each), worst1.276019. All sample3 values finite,161 masks exact; inactive failures0. | Failing raw prefix precedes physical transforms, but no first differing internal operation or independent derivative reference is known. Raw test has no VJP. |
| First-day finite cross-route | Canonical public step, all215 input leaves matched, natural scale/no /4. Each cycle's next-state tangent failures200/16/14/16 for434/2382/2621/200; worst ratios30.7390/9.4982/4.4387/11.4262. Defined finite failures persist with control200. Finite primal worst0.593506, six observation tangent values pass,20 numeric aux leaves/3883 values plus3 None exact. | Occurs before later recurrence/loss; does not prove a common cause with raw/T7, establish derivative truth, or waive nonfinite admission. |
| Full T7 finite cross-route | B1 native loss JVPs6.1448252/11.4858093/0.01472492/0.82920219; B1/4 versus B4 mean ratios23.77419/1.16314/0.04686/0.01174. Eight of16 cross-route checks fail (434/2382 each cycle). Full-state endpoint cross-route comparisons fail5/3854 coordinates (33/434/515/2382/2621). | B1/4 establishes matching mean-loss convention, not reference correctness. Four-direction tests do not cover all initial-state or parameter directions. |
| Full T7 same-route JVP/VJP | Own checks pass20/64 at native scale,24/64 at mean scale. Native B1begin2621 difference2.6162691e-6 exceeds allowance1.1472230e-6; its /4 check passes, while B1end/4 still fails2621. Full state self-repeat gates also fail. | Route alignment alone is insufficient admission. Do not hide native failures with /4, or assert that cross-route roundoff explains own AD inconsistency. |
| State20 strict nonfinite | First-day primal `diffuco_previous_step_state.roughheight_pft[0,0]` is NaN: one B1/four full-B4 entries per saved result; defined mask false, tangent zero. Fortran `condveg_z0cdrag_dyn`, condveg.f90:1519–1720 assigns1715–1717 inside jv loop beginning2; JAX condveg.py:364 initializesNaN,366–418 loop. Downstream source use: diffuco.f90::diffuco_aero:1067–1068. | Source-undefined classification does not waive the strict nonfinite gate. It does not explain the finite defined failures. A source/interface decision is needed before any state admission change. |
| Repeats and remaining gates | Raw/first-day16 nontrivial repeats exact;12 B4 other-sample direction checks zero. Full T7:24 nontrivial JVP repeats exact;64/64 primal/aux bridges pass,14 physical/mask gates finite zero,355 FP32 parameter leaves/66,123,713 coordinates self-repeat pass. Saved reverse B4/B1 ranges434/2382/2621 are separated, with gap/B1 width7.2098/3.3015/3.0196. | Exact forward JVP repeats and finite reverse variability coexist. Ranges are descriptive, not a causal/statistical bound. Full cross-parameter admission for all four B1 windows is not established. |

## Bounded attribution: two mechanisms only

**1. Batch-shape-dependent contractions/reductions/fusion.** Original
`strong_baseline_architecture.py::_linear:887` is FP32 value@kernel+bias.
The executed latent-reader overlay uses `_prepare_field:205–230` for coordinate,
axis/local projections and `_decode_reader_rows:233–276` for native reader blocks,
readout and restored masks. Shared attention score/value contractions are at
`shared_transformer.py:518/543`. Actual raw optimized HLO changes reader64
`[128,64]×[64,64]` algorithm18 in B1 to `[512,64]×[64,64]` algorithm19 in B4;
B1 also has algorithm-1 variants. Attention shapes/algorithms change too.

This is an eligible execution mechanism, not attribution. Saved raw434 failures
include typed.ok_leak20 fields/1095 coordinates and inventory.carbon6/538; these
are name families. A bounded inert extraction of the hash-bound original context
recovers all161 actual field/expert assignments. Raw434 failures by expert are
hydrology_snow47 fields/81 coordinates, litter_poc_doc28/1646, soil_thermal3/7,
surface_energy3/4, daily_interface_finalize7/28 and canopy_gpp3/3; totals91/1769.
The four DOC primal failures belong to litter_poc_doc. This closes static layout
mapping, but true later64 operands remain missing. No boundary intervention or operand-bound local
oracle connects a contraction to those failures, and separate-prefix HLO need
not match the full-day/T7 executable. Replacing one GEMM or mapping a whole
prefix is unsupported; potential batching gain lost by such changes is unmeasured.

**2. Nonlinear normalization/extrema and shared-path AD accumulation.** Original
`_masked_summary:957–973` computes mean/std/min/max with physical masks;
`_standardize:976–981` and overlay `_normalized_local_state:800–826` feed field,
axis/local paths into the reader. Saved434 scale1.702165729e-7 and multiple
field/axis/local memberships are sensitivity clues. The four selected directions
are all active/defined/non-tail; none lies on an initial tied field/axis extremum.
2621 and200 are unique field minima. Later recurrence ties, actual path
contributions and the effect of FP32 fan-in remain unknown. No supported
intervention separates normalization/AD from mechanism1. Changing scale,
precision, custom reverse, loss or tolerance would not be a function-preserving
execution repair within this task.

Original source verification used the c599236bcf57b745e9bafed678298010f4c3b58f
Git source, the nine-member raw overlay archive
f3dbcc7f62aed4f3afe478ae3f8f434d3d8a7efd1dba292c6c2aea62e1c2fd86,
and both HLO identity receipts. Current overlay bytes were independently checked
against original execution hashes: shared_latent_reader `02f44b…5eb20` and
shared_transformer `aad25a…c9b96`. Current HEAD was not substituted for execution
identity. Worker graph discovery/coverage calls returned Transport closed;
the parent successfully confirmed main generation2026-10-10T08:46:50Z and
ready/known-excluded daily source coverage. Exact source fallback supports these
bounded claims, not exhaustive graph coverage.

## Single advancement recommendation and regression boundary

The supervisor should pause B4 local kernel repair and next decide a bounded
**native AD consistency and valid-state admission baseline for this same fixed
model**, aimed at allowing reliable throughput comparisons. This differs from
route alignment: same-route native JVP/VJP and state20 already fail, so making
B1/B4 agree would leave unresolved admission gates. Start specification from
the saved public-loss JVPs, full VJP endpoints, parameter/state repeat arrays,
input identities and state20 source evidence; identify an independent consistency
criterion for the existing loss/directions without declaring either route truth.
The source/interface treatment of undefined state20 needs explicit scientific
discussion; neither filling NaN nor excluding it is selected here.

Minimum evidence for that decision is the native and mean-scale own-AD table,
full parameter/initial-state coverage limits, exact finite/mask state interface,
and a bounded proposed independent consistency method with cost and stopping
rule. Execution or changed science/resources needs separate supervisor/user
scope. This recommendation starts no diagnostic chain or training. If a local
kernel route is later reopened, true reader64 operand/boundary JVP/VJP capture
and a function-preserving intervention are missing evidence, not the automatic
next task. Do not build the backbone just to capture64→64 here.

Any future original-model regression retains66,123,713 parameters and fixed
TRAIN742/275/482/299; full public loss/aux/physics/masks, complete native parameter
and initial-state gradients, independent derivative consistency, separate
native/mean-scale gates, strict state20 nonfinite admission, FP32 neural/FP64
physics and24GiB. Reuse unchanged references/inputs/executables while recognizing
B1 is not truth. Record compilation, peak memory, time-to-valid-measurement,
steady-state throughput and lost batching gain separately. No concrete repaired
math/selector exists, so no genuine fully specified T7 GPU proposal is supplied.
Historical B4 lowering435s/backend7595s alone defeats a superficial20-minute
rerun plan. Historical10800s/CNY8.28 is not present authorization.

## Local artifacts and checks

New evidence directory: `outputs/T7-batching-repair-decision-20261014/`.
[Raw evidence](../../../outputs/T7-batching-repair-decision-20261014/raw_evidence.v1.json)
and its script/receipt performed one restricted NumPy-only pass
over24 saved raw PKLs in0.721075s, matched accepted artifact hashes, shapes,
topology/counts/masks/repeats/isolation and verified original evidence unchanged.
[Source/HLO evidence](../../../outputs/T7-batching-repair-decision-20261014/source-hlo-evidence.json)
verified9 archive members,2 original overlay hashes and4 actual saved HLO
representative instructions.
[Field/expert evidence](../../../outputs/T7-batching-repair-decision-20261014/field_expert_map.v1.json)
recovers the161-entry ledger and seven-expert mapping, stops at inert layout
BUILD before JAX reconstruction, and retains unchanged context/raw/audit hashes.
[Cross-stage evidence](../../../outputs/T7-batching-repair-decision-20261014/cross-stage-evidence.json)
reuses existing analysis JSON/result documents, not a new full-T7 PKL audit
(full endpoints are not local). It retains native/mean/state20 distinctions;
state20 has15 nonfinite entries per direction/three cycles and60 across all24
first-day calls. JSON and key assertions passed.
Two-route saved-HLO headers/source identities and local document links pass.
The HLO identity receipts contain no historical gzip byte digest; current
file hashes are recorded without claiming a historical byte-hash match.

No model forward/JAX numerical tracing, executable loading, compilation, remote compute, GPU, Teacher,
optimizer update, launcher work, cleanup, scope reopening or external-chat
message occurred. Bulk artifacts remain outside Git. The five entry documents
point to this decision; historical identities, failures and closed scopes remain.
