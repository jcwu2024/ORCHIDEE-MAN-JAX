# Proposed bounded state-gradient reference diagnostic

2026-10-02. **Authorized and executed; closed with a backend-compilation
timeout, without a numerical verdict.** Actual preparation, lowering, budgets
and saved-array verification are authoritative in the
[FP64 result](shared_state_reference_fp64_results_20261002.md). The proposal
and preparation record below describe the approved scope and its evidence
before execution; their pending checks are resolved or classified there.
This proposal is a diagnostic scientific choice, not a training method or
automatic admission of any closed execution candidate.

## Authorized decision

Allow one original-public-T7 reference experiment with the complete neural
path evaluated in FP64, alongside the already saved original FP32 evidence.
Physical calculations remain the canonical FP64 calculations. Use the same
architecture, masks, normalization/calibration values, update0 parameter
values, artifact742/window0, prepared inputs and public loss. Converting saved
FP32 values to FP64 does not introduce different weights or new data.

The motivation is measured uncertainty: the same original FP32 reverse binary
fails state self comparisons, its four-direction native JVP check is not
strictly consistent with reverse AD, and deterministic FP32 expert5 execution
is stable but1.8875x slower and fails old-reference parity. These observations
do not prove either old derivative wrong or authorize weaker acceptance.

## Bounded implementation and measurement

1. Audit the active neural dtype boundaries, including feature normalization,
   attention/norm/FFN/readout and derivative accumulation. Preserve all formulas,
   outputs and source masks; expose any remaining FP32 boundary explicitly.
   No claim of a full FP64 neural reference until this audit passes.
2. Run the small actual-environment preflight, then compile one native public
   loss graph returning the complete initial-state VJP plus a JVP for a dynamic
   direction. All64,884,481 parameters and all case inputs remain dynamic;
   only the3854 initial states are differentiated for this diagnostic.
   Parameter-gradient and optimizer admission remain separate existing gates.
3. Evaluate33,199,428 and control200 at unchanged physical values, initially
   and over seven repeats. Save loss/auxiliary, all3854 state derivatives,
   directional values, binaries, executed source and actual stage exits.
4. Compare native forward/reverse consistency and within-arm stability using
   the unchanged `rtol=1e-5, atol=1e-6`; require finite results, zero hard
   physical counts, declared source masks, undefined-state zero derivatives
   and estimated/observed active memory below24GiB. Report the FP32/FP64
   primal/derivative differences without declaring either model equivalent.

Use one free V100 only. Proposed bounds: preflight180s, lowering300s,
backend1800s, model process2400s; cumulative one-GPU exposure at most2640s
(0.7333 card-hours), including preflight, failed starts and grace. No new paid
submission is included.
Any paid fallback requires a separate exact resource/command/cost proposal.
Remote writes stay under jcwu's runtime tree; preserve the immutable checkout
and existing prepared-data identities. A budget or ordinary code failure is
an implementation issue; preserve completed stages. Pre-compilation repairs
must stay within the cumulative exposure bound. Stop after a failed expensive
model compilation or numerical diagnostic; no automatic recompilation, dtype
sweep or larger allocation is included.

The diagnostic ends with a judgment about the reliability of these reference
derivatives and an evidence-based next optimization decision. Passing it does
not adopt FP64 training, loosen the current execution gates or reopen a
rejected candidate automatically. Failure is reported without a scale sweep
or expansion into an unrestricted gradient project.

Evidence: [original JVP](shared_original_state_jvp_results_20261002.md),
[complete slicing](shared_original_dual_slice_update_results_20261002.md), and
[deterministic runtime](shared_deterministic_runtime_results_20261002.md).

## Pre-execution source audit and preparation, 2026-10-02

**Pre-execution snapshot; superseded by the linked results for actual execution
status.** The following pending-check statements describe preparation before
the authorized run, rather than its current status.

**Implementable, with bounded diagnostic value; not a speed intervention and
not yet admitted for full-model execution.** The exact parameter scope is
64,884,481 coordinates in300 leaves,519,075,848 parameter bytes in FP64
(259,537,924 in FP32). Parameters remain dynamic arguments; this experiment
differentiates only the3854 FP64 initial states. It cannot replace the complete
parameter/state/optimizer acceptance gate used for an execution optimization.

Source truth was read from immutable checkout
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1`
at `c599236bcf57b745e9bafed678298010f4c3b58f`, plus its admitted staged modules
under `runtime/transfer/shared-combined-recurrent-20260926/`. Local graph Verify
generation is2026-10-02T04:30:36Z; relevant canonical files are
`metadata_changed/no_recorded_issue`, so actual source reads supplement the
graph. Local staged decoder/backbone paths are absent and were not treated as
source truth. Small source snapshots are under
`outputs/diagnostics/shared-state-reference-precision-audit-20261002/source/`.

The audited hard FP32 boundaries are:

| Actual source, relative to daily_operator or staged directory | Boundary requiring a diagnostic FP64 override |
| --- | --- |
| `input_conditioning.py`,12–52 | `standardize_continuous` default dtype, unconditioned normalization and final conditioned cast |
| `strong_baseline_architecture.py`,54–58 | Both `NeuralPrecisionPolicy` dtype getters; changing weights alone leaves these FP32 |
| same,886–979 | Linear input cast; layer-norm input/mean/variance; stable head L2; cosine temperature; summary inputs/counts; feature mean/scale/value casts |
| same,1003–1007,1137,1227,1283–1295 | Forcing time/duration/predecessor/valid features, discrete/static inputs, PFT traits/fractions and calendar fields |
| `shared_transformer.py`,305,495,672–676 | Attention temperatures and source-family pooling count/value/accumulation dtype |
| same,787–807,929–965,1029–1041,1123,1187 | Local normalization, empty/axis buffers, valid-feature casts, carbon feature casts and inactive decoder output |
| staged `shared_backbone_projection_reuse.py`,48 | Backbone cosine temperature |
| staged `shared_decoder_expert_execution.py`,141 | Empty eligible-field output dtype |

FFN and readout use the same `_linear` helper, so its FP64 input, FP64 weights
and FP64 bias are all required. Attention softmax/value aggregation and
projection accumulation inherit their operands' dtype once these boundaries
are corrected. `shared_decoder_dense.py` and staged
`shared_decoder_macro_packed.py` have no explicit FP32 cast; their imported
neural helper aliases still require rebinding. `state_features.py` is already
FP64. `shared_process_head.py` physical output transforms are already FP64
and are left intact. Initialization routines are not rerun: existing FP32
weight values are exactly widened, with no random reseeding or new calibration.

Prepared dedicated files:

- `scripts/hpc/shared_neural_fp64_diagnostic.py`: a temporary neural-only
  function overlay. It changes only explicit `jnp.float32` dtype references
  in the eight audited modules, updates the two dtype getters, rebinds imported
  function aliases and restores all bindings afterwards. Existing serialized
  class identities and physical source logic are preserved.
- `scripts/hpc/probe_shared_state_reference_fp64.py`: a draft original-public
  T7 combined state VJP/dynamic JVP harness. It checks exact parameter count,
  update0/artifact742 identity, saved weight values, finite outputs,14 hard
  counts, undefined-state exact-zero derivatives and24GiB memory. It saves
  lowering and fails before backend compilation if any FP32 tensor remains in
  the full public lowering. It does not compute parameter gradients.
- `scripts/hpc/run_shared_state_reference_fp64.sh`: prepared launcher,
  preflight180s and model2400s, same container/driver bindings and original
  LLVM setting; one free GPU1 only.

**Actually checked:** Python syntax; target Bash syntax and actual
`--dry-run preflight`/`--dry-run model` argument expansion; small local ORCJAX
CPU checks against the downloaded canonical/staged source. Linear, layer norm,
L2 normalization, standardization, continuous-tail conditioning, masked
summary, FFN, normalized attention and native FFN input gradient return finite
FP64 arrays of the expected small shapes. Imported aliases, class identities
and restoration checks pass. Evidence is `cpu-policy-check.json` and
`target-shell-dry-run.txt` in the audit output directory.

**Not yet checked or executed at this pre-execution snapshot:** real prepared-input assembly, full tokenizer and
decoder shape/mask equivalence, full-public JAX lowering, new GPU preflight,
FP64 backend compilation, real forward/AD values, memory or repeat stability.
The harness is a reviewable draft, not a claim of full-model readiness.
Before expensive compilation, it still needs explicit complete prepared-tree
shape/dtype/mask identity checks, declared-mask-count comparison against the
original, and a check that the full VJP does not depend on the dynamic JVP
direction. Source replacement alone is not full dtype admission. These checks
are dependent implementation after precision authorization, not evidence
already obtained.

Reviewable static command (runs no precision model):

```powershell
conda run -n ORCJAX python outputs/diagnostics/shared-state-reference-precision-audit-20261002/check_policy_cpu.py
```

Target shell command already exercised without model execution:

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-state-reference-fp64-20261002/run_shared_state_reference_fp64.sh --dry-run model
```

After authorization and the remaining assembly checks, the prepared launcher
entry would be the same command without `--dry-run`, preceded by its
`preflight` stage. At this pre-execution snapshot it had not run; the completed
execution and backend timeout are now in the linked result. The user authorized the explicit
scientific choice to spend one bounded free-node experiment on changed neural
precision; no paid job, training or new default is requested.

## Decision value and stopping rule

The300s lowering/1800s backend/2400s model limits are defensible **caps**, not
runtime predictions. The original state JVP took81.110s lowering and279.696s
backend; the proposed combined state reverse/forward graph is different and
has no measured FP64 compile multiplier. V100 supports FP64, but extra bytes,
backward activations and longer compilation may exhaust the cap or24GiB.
Neither input bytes nor synthetic CPU passes guarantee GPU memory admission.
The whole-model timer overrides the sum of stage budgets; preflight plus
model timeouts/graces total at most approximately2580s/.7167 GPU card-hours,
excluding launcher setup. No larger fallback is part of this choice.

If FP64 restores strict forward/reverse consistency and repeat stability,
the supported conclusion is that this bounded source-equivalent computation
has a more internally consistent derivative at these inputs. The next
scientific decision is whether to adopt a **separately specified diagnostic
reference contract** for future comparisons, including how FP32 execution is
judged against it. That decision requires discussion; it does not retrospectively
certify a closed candidate or silently replace the old strict gate. Preserve
old/new losses and gradients, then return to complete-update speed work.

If FP64 still disagrees, is unstable, retains FP32, fails physics/masks,
exceeds memory or times out, close this single diagnostic with its precise
failure category. Do not continue into higher precision, extra directions,
FD steps, custom AD, compiler sweeps or T30. Return to the speed workstream
with the reference uncertainty stated explicitly; further scientific
adjudication would need a new concrete decision.

Even a pass cannot prove a physical right derivative at the selected zero
domain boundaries, validate hidden branch decisions or the full source state
manifold, prove either original FP32 AD direction wrong, establish global
Fortran/process coverage, or yield adopted training throughput. There is no
specific currently unclosed original-precision route established by this
audit that would bypass those gates. The smaller deterministic expert5 route
has already failed strict parity and speed and is not repeated. Consequently,
this FP64 option should be chosen only for the stated reference decision,
not presented as progress toward the10x throughput target.

Lead review repaired early launcher exit recording: create the diagnostic
directory and install the exit trap before checkout/occupancy validation,
while keeping existing-output guards first. Local Bash syntax passes;
arguments are unchanged. Upload this local logging repair before an authorized
execution. Earlier target dry-runs do not certify a GPU/model run.
