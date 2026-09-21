# Shared decoder: preserve capacity while reducing memory

2026-09-21. Planning/source audit after the4eeb168 admission memory rejection.
No replacement architecture or new training is claimed in this document.

User follow-up: implement and measure steps1+2 first. Do not add step3 if
the real daily plus T7/T30 training fits with sufficient margin. Architectural
capacity and potentially active outputs remain unchanged.

The user correctly challenged interpreting `auxiliary_supervision` as evidence
of low long-term importance. That registry is a declared category, not a
measured sensitivity ranking or proof that a smaller neural decoder suffices.
Distinguish direct use of a predicted quantity, effects through shared-parameter
training, and effects across recurrent states. Keep the approved161 fields,
75,578,497 parameters, output depth/width and all potentially active coordinates
while first investigating execution changes that preserve the function.

## What the auxiliary classification actually establishes

A bounded independent Sol/high audit enumerated all66 fields from the canonical
schema:28 water,24 carbon and14 energy. The category is assigned from frozen
label lists by `target_oriented_contract.resolve_target_output_hierarchy`, not
from an empirical sensitivity score. The earlier contract discussion explicitly
reserved head-capacity/reconstruction decisions for architecture review.

There is a current-boundary rationale for the supervision/diagnostic role:
the conservative updater consumes inventory endpoints and thermal changes;
it does not consume the typed auxiliary flux records. Temporary copies of
some typed fields into fast-state slots are overwritten by canonical endpoint
owners/mirrors before the retained tail runs. The science extractor reads GPP
and the resulting continuous state. This is a statement about the current
canonical path, not about the ecological importance of the physical processes.

Their training influence remains: all66 fields enter typed supervision, with
auxiliary label weight0.25 in the existing necessary-process loss. Those losses
update shared representations and can help or interfere with science learning.
Neither the category nor zero direct next-state use demonstrates that192/128
lightweight heads are sufficient. The bounded historical gradient audit found
zero direct flux-objective gradients on separate auxiliary heads of an older
83,755,173-parameter model; it did not test auxiliary-loss ablation, shared-weight
effects, head-width alternatives or long-horizon counterfactual sensitivity.
No such accepted capacity demonstration was identified in the inspected source
and named reports; this is not an exhaustive repository-wide absence claim.

Evidence: `target_oriented_contract.py:86,312,613`,
`neural_process_head.py:247,448`, `conservative_update.py:86`, `model.py:82`,
`mechanistic_process_head.py:254`, `canonical_retained_tail.py:355`,
`successor_training.py:409`, and the historical
`target_oriented_daily_operator_contract_v1.md:50` /
`target_oriented_flux_gradient_variance_audit_result_20260826.md:76`.
Paths above are within `daily_operator/` or this research-doc directory.
Graph project/coverage calls returned Transport closed; the review used exact
source and owner manifests. No66-field dynamic intervention experiment ran.

## Findings available before another GPU compilation

GPU admission estimated36.17GiB live gradient memory, of which35.61GiB was
temporary memory; no optimizer update executed. The previous architecture
used different decoders for different output categories. That explains a
change in work performed, but it is not evidence that all auxiliary coordinates
need less capacity. Individual buffer attribution is still incomplete.

The current `_field_active_and_pft_owner` already makes output coordinates
outside `layout.active_pft_indices` permanently inactive on full-PFT axes.
`_decode_field` nevertheless builds and decodes their queries before returning
zero. A metadata-only local count, using canonical fixture metadata and patched
legacy initializers, gives:

| Existing category | Fields | Output coordinates | Eligible under static PFT layout |
| --- | ---: | ---: | ---: |
| Direct science | 2 | 26 | 26 |
| State/budget | 93 | 3,682 | 3,298 |
| Auxiliary supervision | 66 | 35,298 | 5,226 |
| Total | 161 | 39,006 | 8,550 |

The current compact PFT indices are0 and13. Therefore30,456 output coordinates
are structurally masked irrespective of parameter values for this layout.
This does not classify any potentially active physical process as unimportant.
It is also not a prediction of78% less peak memory: compiler liveness, input
encoding, retained physics and backward work must be measured separately.
Reproduction helper: `outputs/audit_shared_decoder_static_pft.py`; no Teacher
data, checkpoint or old neural weights were read/initialized. Full deployment
schema/counts must agree before using this gather plan in production.

## Ordered execution changes

1. **Static gather/decode/scatter.** Generate queries only for coordinates
   allowed by the declared PFT layout, run the unchanged decoder, then scatter
   into the complete field-shaped zero array. Keep original field/axis indices,
   active-PFT ordering and local-state mappings. Runtime masks still apply to
   every eligible coordinate. Do not exclude a coordinate because its current
   observed value, PFT fraction, flux, gradient or error happens to be small.
   Recompute the gather plan from the layout when supported PFTs change.
2. **Reuse identical key/value projections.** Within each decoder layer and
   ownership stream, project common family/process tokens once, and project
   the unchanged raw-carbon bank once for the applicable GPP path. Keep query-
   specific axis/local tokens and masks. Avoid physically expanding common
   projected banks across output coordinates. All bank segments must share
   the original single softmax normalization: separate segment softmaxes would
   change the model. Caches exist only within one differentiable invocation,
   not across parameter updates or rollout days; no stop-gradient is allowed.
3. **Native checkpoint/recompute.** If needed after the first two changes,
   apply native JAX rematerialization around coordinate chunks/field decoding
   so reverse mode reconstructs intermediates instead of retaining every
   chunk. Include coordinate-feature generation in the boundary when liveness
   evidence requires it. Preserve the ordinary scalar loss and automatic
   differentiation; do not introduce another custom backward implementation.

All output queries currently act independently given their input bank: decoder
attention has one query per coordinate, layer normalization and FFNs operate
on feature dimensions, and there is no output-output attention. This is the
structural basis for the static gather/scatter proposal. Linear K/V projection
and per-key normalization commute with reuse of an identical token; they do
not justify merging distinct tokens, masks or ownership streams.

Rematerialization can increase work. Reusing projections can reduce work.
Measure both memory and warm execution time, rather than declaring success
merely because a graph fits or counting parameter savings that do not exist.

## Evidence required to adopt an execution variant

- Same parameter tree/initial values,161 output names/shapes, valid coordinate
  representations, source masks, physical transforms and scalar loss.
- Bounded original-versus-optimized forward and per-group loss comparison; parameter,
  continuous-state and physical-parameter gradient comparison using native AD.
  Include inactive/all-masked rows, PFT permutation, chunk boundaries and GPP
  raw-carbon routing. Use justified FP32 tolerances; do not require bitwise
  equality or loosen tolerances to pass.
  Compare carbon science responses/derivatives separately on the bounded
  reference so the current enormous state anchor cannot hide their discrepancy
  inside an apparently small relative error of the total scalar/gradient.
- Original small-config comparisons are feasible locally; the rejected full
  production gradient was never executed and cannot be called a reference.
  Use actual deployed forward evidence and a bounded native JVP/reverse check
  for the optimized full graph.
- Compile on the actual free GPU, record live-memory estimate and actual
  allocator peak separately from JAX reservation, and complete two full updates
  plus physical checks. Then verify the changed joint T7/T30 execution. Do not
  conclude scientific learning from this technical comparison.

The very large state-change loss remains a separate blocker needing field/
scale attribution. Do not hide it by changing weights while testing execution
equivalence. Any scientifically motivated loss amendment gets its own declared
scope and relevant gradient/update admission.

If these execution changes cannot fit or have unacceptable runtime, review a
new architecture explicitly. A lower-capacity auxiliary route, latent bottleneck,
or output removal changes the hypothesis and requires scientific validation;
none is automatically authorized by the historical category label.

## Implemented execution and local evidence

`shared_transformer.py` now has three execution policies: `original` for
bounded reference checks, `gather_only`, and default `gather_reuse`. These
are execution settings, not new parameter trees. Shared K/V projections remain
inside the differentiable invocation; the decoder combines all score segments
before a single softmax and contracts shared value banks without tiling them
over queries. There is no decoder rematerialization in this implementation.

The admission runner reports the deployed eligible count and retains its24GiB
gradient-memory gate. Forward-only state diagnostics attribute the existing
state anchor to groups, canonical fields and largest coordinates; they do not
change the scalar or add another training objective. The bias decomposition
retains cross-field terms, so a field's signed bias contribution may be negative.
Local evidence:

- The canonical static plan retains8550/39006 coordinates across161 fields.
- Original/gather-only/gather-reuse comparisons pass on a bounded four-field,
  two-decoder-block model: GPP, maintenance, snow and a full-PFT thermal field
  with nontrivial local-state indices. Checks include padded chunks, runtime
  all-PFT masking, permutation and raw-carbon routing.
- Separate four-field Jacobians agree for neural parameters and continuous
  inputs; physical-parameter inputs retain zero neural-decoder sensitivity
  (explicit physical factors remain outside the raw decoder). The final
  four-objective Jacobian check passed in137.75seconds. Forward tolerances are
  rtol2e-5/atol2e-6, gradient rtol5e-4/atol2e-5, reflecting FP32 reordered
  reductions; they were not loosened after a failure.
- All161 physical output arrays agree with original execution in the canonical
  small-width head fixture. Four physical-head tests passed in178.72seconds.
- Trajectory/runtime tests (15) and state-diagnostic/shared-training tests (7)
  pass. The canonical diagnostic owner map covers all seven state groups; the
  host decomposition reproduces masked MSE plus squared mean bias, including
  opposite-sign cross-field terms. Reporting does not change the scalar loss.
- Ruff and whitespace checks pass. Sol/high implemented the execution module
  and bounded derivative tests; the lead reviewed/integrated and checked the
  physical-head and diagnostic paths. Graph transport remains unavailable.

The full original production gradient still has no numerical reference: its
previous compilation was rejected for memory. Local bounded equivalence does
not certify full production gradients or scientific learning. Optimized actual
memory/speed and numerical admission are pending the free-node run.

## Deployed admission

Immutable code094dfd2 is deployed at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-decoder-094dfd2`.
CPU/GPU target-shell dry runs and shell syntax passed. Free test02 shape
admission COMPLETED0 in417.41seconds, confirming75,578,497 trainable parameters,
161 fields/39006 outputs,8550 static eligible coordinates, active PFT0/13 and
rematerialization false. This is shape evidence, not a numerical gradient.
The final dedicated local suite passed3 tests in158.45seconds.

Free gln01 GPU0 wrapper10651/Python11462 is RUNNING with60min main-test ceiling.
The synthetic GPU environment check PASSED70.95s (JAX0.4.38).
Logs and JSON use the prefix
`runtime/outputs/diagnostics/shared-decoder-094dfd2-gpu`. Existing24GiB
compiled-gradient admission limit is unchanged. Do not duplicate the run.
Full numerical gradients, two updates, actual memory/time and subsequent
T7/T30 admission remain pending. Local CPU report:
`outputs/shared-decoder-094dfd2-cpu.json`.
