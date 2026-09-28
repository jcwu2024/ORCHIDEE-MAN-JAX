# Bounded broader-neural precision screen

Date: 2026-09-27. This is a numerical/performance diagnostic, not production
precision admission, training or final scientific acceptance.

## Declared comparison

One actual daily row (row 0 of saved artifact 742 micro4) uses the canonical
`make_shared_joint_daily_objective`: `process_day / 3 + (flux + stock) / 9`.
Native AD differentiates all 64,884,481 parameters and the continuous input
state. Both processes use the same immutable `c599236` source checkout,
update-0 diagnostic checkpoint, expert128 decoder and backbone source-reuse
execution. No data, sampling, objective, architecture or capacity changes.

Only the `_linear` and `_ffn` aliases in `shared_transformer`,
`shared_backbone_execution`, `shared_backbone_projection_reuse` and
`shared_decoder_macro_packed` change dot policy to native
`F16_F16_F32`, with FP32 parameter storage. The original strong-baseline
helpers, tokenizer/carbon encoders and all physical computation are untouched.
Norms, GELU, softmax, attention scores/value sums and controls remain FP32.

The strict `rtol=1e-5`, `atol=1e-6` comparison is retained and reported
separately. The independently declared provisional engineering screen is:

- Finite loss, outputs and both gradient trees; physical hard counts zero;
  raw and state diagnostic masks unchanged; structurally undefined continuous
  input gradients exact zero.
- Loss relative difference at most `1e-3` (near-zero reference absolute
  difference at most `1e-6`).
- Each semantic parameter group and each continuous contract state field:
  gradient relative L2 at most `1e-2` and cosine at least `0.9999`.
  Reference norm at most `1e-12` instead uses maximum absolute difference
  at most `1e-6`, without misleading ratio/cosine.
- Output discrepancies divided by existing TRAIN scales are descriptive only;
  absent, incompatible and invalid scales are identified without invented gates.
  Zero fractions and newly zero/nonzero coordinates expose underflow; accidental
  reference zeros are not treated as structurally inactive coordinates.

A dynamic FP32 loss multiplier is an argument, default 1. The canonical scalar
is multiplied before native AD and reported loss/gradients divided afterward.
Only multiplier 1 is authorized for this comparison; no scaling sweep.

## Execution and evidence

Free Explore1000 `gln01` actual host `ibc13b02n04`, one V100-SXM2-32GB (GPU0).
Before launch both GPUs were empty/idle; host load was `0.00/0.01/0.05`.
The job recorded CPU affinity `0-111` and launch load `0.08/0.03/0.06`;
this is not a whole-node reservation. No paid allocation was submitted.

The deployed runtime interface check passed on JAX 0.4.38 in 69.43s; it
reads no Teacher data and does not certify the real neural model. Each real
process has a 600s deadline and lowering/backend compile together a 300s
deadline. The completed FP32 executable/result must be saved before candidate
compilation; it is never automatically recomputed.

Local checks: ten focused tests in
`tests/test_shared_neural_precision_probe.py` passed with
`conda run -n ORCJAX python -m pytest -q -p no:cacheprovider tests/test_shared_neural_precision_probe.py`.
Undefined-name lint passed. Both reference and candidate launcher branches
were exercised with actual target-shell `--dry-run` expansion.

The first reference launcher (PID 95415, `00:30:02+0800`) exited before backend
compile because the retained-tail static configuration expected
`data/forcing/cruncep_twodeg_1991.nc`. The existing 851,746,768-byte forcing
asset was present at `runtime/data/forcing/cruncep_twodeg_1991.nc`. The retry
used a container-only, read-only `runtime/data -> data` bind; no new data,
global path change or physical logic substitution. The unchanged successful
runtime preflight was reused.

Fresh retry `r1` (wrapper PID 98046, `00:34:36+0800`) lowered in
**106.553965s**, then hit the hard combined lowering/backend compile limit of
**300s**. The timeout report elapsed was **312.666877s** including process
initialization; exit status **124**. Roughly 193.45s of the declared compile
budget remained after lowering. There was no completed executable or FP32
result. **Candidate was not launched.** This is a compilation-resource limit,
not numerical rejection or a passed mixed-precision screen. No loss, physical
gate result, state/parameter gradient discrepancy, underflow result or warm
runtime was measured by this probe.

All task processes finished and GPU processes were empty afterward. One GPU
was exposed throughout; the two launcher intervals totaled approximately
448s / **0.1244 GPU-card-hours of wall exposure**, including preflight,
initialization and failure handling (log timestamp resolution 1s). This is
not measured kernel time and incurred no paid allocation.

Both attempt logs/reports and the successful preflight report are preserved
on Explore1000 and copied locally under `outputs/diagnostics/`:

- `shared-neural-runtime-20260927.json`
- `shared-neural-reference-20260927.{json,log}` (missing input path; the JSON
  records the last reached stage, the log records the exception and exit 1)
- `shared-neural-reference-20260927-r1.{json,log}` (compile timeout)

That initial outcome did not authorize cap extension, long recurrent graphs,
loss scaling trials or paid jobs. The subsequent authorized continuation below
preserves the same numerical contract; it remains unmeasured on the real model.

## Native staged continuation

The canonical objective is factored into three separately jitted functions,
with native `jax.value_and_grad(..., argnums=(0, 1), has_aux=True)` outside an
outer jit. They use existing owners, not independent loss/cotangent formulas:

1. Canonical input assembly, shared context and learned physical outputs.
2. Canonical dynamic retained binding, conservative update and retained tail.
   A decoded-output proxy uses the original `prediction_from_learned_outputs`
   and `materialize_post_update` methods. The latter explicitly discards the
   context and parameter arguments; expensive context assembly occurs once.
3. The unchanged `make_shared_joint_daily_objective`, with its transition
   result supplied dynamically. Direct parameter/state loss dependencies remain
   in the signature, alongside all neural-mediated and physical state paths.

Native AD composes these calls; no hand VJP, fixed observation cotangent,
stop-gradient, shortened gradient or physics substitution. Six named native
linearized-forward/transpose compile units are expected. Each unit retains a
300s cap and each process a 600s cap; these are new bounded execution units,
not an extension of the original monolithic compile.

The first staged attempt, PID 102175 (`01:20:39+0800`), hit the 300s cap on
the **neural linearized-forward** unit, before physical/loss compilation.
Saved StableHLO has 132,110 operations, 1,999 `dot_general` operations and
eight while operations. Its main result is **18,964 leaves / 3,858,177,208
nominal bytes** (3.59 GiB), including large decoder-shaped residual tensors.
This is a nominal IR interface size, not a measured live GPU peak; it does
not establish that eight-block backbone unrolling dominates compilation.

The approved next diagnostic uses native `jax.checkpoint` at the neural
boundary before jit, with full native recomputation/derivatives. Before its
backend call, the recorded result interface decreased to **2,531 leaves /
7,974,606 nominal bytes** (7.61 MiB), a 99.79% nominal-byte reduction.

The remat attempt, PID 104491 (`01:35:48+0800`), completed five compilations:

| Unit | Compile/cache lookup seconds |
| --- | ---: |
| Neural forward | 81.609094 |
| Physical forward | 85.766880 |
| Loss forward | 10.172333 |
| Loss transpose | 3.316134 |
| Physical transpose | 242.712263 |

The 600s outer launcher cap then stopped preparation of the final neural
transpose. Physical transpose completed within its unit cap; its IR has
31,128 operations / 3.05 MB versus forward 27,915 operations / 3.04 MB.
These lookup durations are not complete cold-start time: tracing, artifact
persistence, dispatch and initial execution add costs. No complete gradients,
validated physical gates or warm model timings were saved.

### Cache correction

**Five completed compilations did not mean five persisted executables.**
The first cache-only resume, PID 106911 (`01:48:03+0800`), correctly stopped
before backend compilation at the first required cache miss (66.040s).
The original cache directory had no executable cache files, only 224 KB of
autotune entries. JAX 0.4.38 initializes its persistent cache at most once;
the probe configured its directory after physical-package initialization and
the tiny runtime check had already compiled. Initializing with path `None`
left executable caching disabled. Saved IR cannot substitute for a serialized
executable. Earlier progress messages overstated persistence; this correction
is the authoritative status. No completed scientific reference was lost,
because none had yet been obtained, but compilation work must be reconstructed.

The harness now configures the cache before helper/JAX/model compilation,
including launcher environment values and the script's first setup action.
A regression test enforces that ordering. Two independent free-node processes
using a fresh small admission cache verified actual executable reuse:

| Tiny native-stage process | Backend compile calls | Executable cache files | Seconds |
| --- | ---: | ---: | ---: |
| Cold | 13 | 13 | 1.414320 |
| Fresh-process replay | 0 | 13 | 0.497893 |

Returned loss, outputs, masks and parameter/state gradients match exactly
across processes and each matches the tiny unsplit native-AD calculation.
This is runtime/cache plumbing evidence, not actual-model numerical acceptance.
Reports are `shared-neural-reference-20260927-cache-smoke3.json` and
`shared-neural-reference-20260927-cache-smoke4.json` under local and remote
`outputs/diagnostics/`. All task processes stopped at that point. Another
expensive model compile had not run at the time of the cache verification.

### Revised bounded reconstruction

After measuring the five-unit cold cost and verifying real cross-process
cache reuse, the lead explicitly authorized one reconstruction of the lost
compiled units and completion of the remaining native neural transpose.
Reference and candidate processes now have a **1,200s** whole-process limit.
The five previously measured units remain capped at **300s each**; the
first-ever final neural transpose (ordinal 5 and neural-stage identity) has a
**600s** cap. This is a deliberate bounded free-node diagnostic budget, not a
paid job or extension of an unchanged monolithic graph. Recompiling the five
lost units is necessary harness-bug rework, not claimed reuse or a speed win.
The candidate must use the same checkpointed boundary, unchanged objective,
data and numerical contract, multiplier 1, and shared physical/loss cache.
Completed unit cache hits and backend calls are recorded. No repeated 69s
public runtime preflight is needed. The numerical comparison requires a full
validated FP32 reference and candidate.

### Final bounded outcome and structural follow-up

The cache-fixed reference, PID 109538 (`02:00:34+0800`), stopped at the final
neural transpose's **600s unit cap**, not the outer 1,200s cap. Its recorded
process elapsed time is **1,188.471561s**. Five units completed with lookup
durations 82.652869 / 82.524797 / 10.381833 / 3.432671 / 242.180955s and
backend durations 81.240075 / 81.138155 / 10.103606 / 3.341675 / 239.303867s.
The five named executable cache files now exist: neural forward, physical
forward, loss forward, loss transpose and physical transpose, totaling
15,763,738 bytes. They remain under the declared runtime diagnostics cache
`shared-neural-reference-20260927-staged.stages-cache`. The final neural
transpose has saved IR but no completed executable. Its old observer counts
backend calls only on return, so its recorded zero does **not** prove backend
compilation never started. Future reports emit backend entry and increment
the call count on entry; no graph rerun was performed for this logging repair.

There is no full gradient/reference pickle, physical/numerical acceptance or
warm full-result timing. The candidate was not launched. Both task PIDs ended,
and the subsequent GPU process query was empty. This latest process alone
accounts for about **0.330 GPU-card-hours** of free one-V100 wall exposure;
earlier attempts and cache-repair work remain separate costs, not a cold-start
speedup. No retry or cap extension follows. Production stays FP32.

Raw report/log and final IR are preserved as
`shared-neural-reference-20260927-cache-fixed.{json,log,stage-05.hlo.pkl}`.
The read-only structured MLIR inspector ran with deployed JAX 0.4.38 on gln01
without model execution/compilation; its JSON is
`shared-neural-reference-20260927-cache-fixed.reverse-structure.json`.

| Reverse IR measurement | Value |
| --- | ---: |
| Text bytes / operations / functions | 22,182,748 / 216,415 / 853 |
| Main operations / dots | 197,494 / 5,164 |
| Whole-module dots / while regions | 5,704 / 16 |
| Dots containing FP32 384x384 / 768x768 matrix types | 3,972 / 144 |
| Zero-row dot shapes | 360 |
| Main result leaves / nominal bytes | 301 / 259,568,756 |
| Main argument leaves / nominal bytes | 2,671 / 267,504,356 |

The 16 while regions have width384 decoder operands; their bodies total
14,225 directly nested operations (excluding out-of-region callees). The most
repeated dot shapes are small-row width384 projections: one-row projection
1,016 instances, one-row outer-product 508, two-row projection 344. These are
static graph counts, not GPU MAC counts, live memory, kernel attribution or
proof of backend compilation time. The 3,972 count is 69.6% of dot operations,
not FLOPs or runtime; 144 is not all backbone operations. Saved IR omitted
source locations, so exact operation-to-source attribution is unavailable.
Zero-row counts do not authorize removal of potentially active outputs.

Direct source identifies the next exact-function follow-up:
`shared_decoder_expert_execution.py:28-200` builds dynamic tokens and calls
`shared_transformer.py::_field_dynamic_projection_cache:424-448` separately
for each field. The two shared coordinate blocks perform axis/local K/V
projection: eight dots per field. `_decoder_dynamic_plan:810-900` already
assigns global unique axis identities `(expert, slot, role, position)` and
local-state identities; `_decoder_dynamic_tokens:904-977` can materialize the
full bank without field selections. A bounded candidate can materialize
and project that union once, then gather the existing per-field banks.
This targets repeated decoder setup/AD accumulation rather than shortening
attention or scanning an assumed eight-block compile culprit. The active
structural route's weak attention-concatenation result is a different change;
it does not establish rejection of union dynamic-projection reuse.

Before integration, count actual union-bank rows/bytes and preserve
pre-projection bias, undefined-local sentinel, inactive fallback, expert/slot
identity, GPP carbon route, full output restoration and all native parameter/
state gradients. Local union indices are bounded by the 3,854 state entries
plus sentinel; actual axis-bank size and temporary memory require a layout
count. This candidate has not been implemented, tested or timed and is not
an architecture/capacity or production-precision decision.

The earlier raw reports, logs, StableHLO modules and structural summaries are
preserved. A future continuation must account for all cold attempts rather
than calling a short replay duration the full cold cost. Production remains
FP32; checkpointing is a diagnostic execution choice, not production adoption.

Any future warm timings at this boundary include full returned parameter/state
gradients and diagnostic outputs, unlike the production interface. Forcing,
day inputs and targets are captured for artifact742 row0 of prepared micro4;
this is a fixed-row diagnostic, not generic dynamic-sample production timing.
Such timings are not complete optimizer
updates, not evidence of recurrent speedup, and not certification of the 10x
same-workload throughput target. No long recurrent compile is authorized by
this diagnostic. Production stays FP32.
