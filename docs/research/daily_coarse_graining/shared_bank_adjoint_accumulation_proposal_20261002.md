# Bank-adjoint accumulation precision proposal — 2026-10-02

Status: isolated implementation, source/CPU checks and actual test01 command
preparation completed on **2026-10-03 Asia/Shanghai**. The user explicitly approved
this narrow precision diagnostic on that date. The original3600s cumulative
allowance is now exhausted: the sole backend completed and its binary was saved,
then the route timed out in uninstrumented postcompile export/proof handling.
Candidate numerical/state/memory/speed acceptance remains unmeasured. See
`shared_bank_adjoint_accumulation_results_20261003.md` and the concrete pending
480s same-binary route in `shared_bank_adjoint_saved_binary_proposal_20261003.md`.
This is a new narrow accumulation
hypothesis. It does not reopen R4/V4/G2, adopt a reference, or authorize training.
Repeated-index accumulation is not an established cause of the earlier state
discrepancies, and there is no speed forecast.

## Concrete precision boundary

Keep FP32 projection computation and its weights/biases. In the actual macro
`_decode_mapped_rows`, exactly widen the already projected FP32 `axis_key`,
`axis_value`, `local_key` and `local_value` banks to FP64 **before** creating or
executing the original chunk map. These four widened arrays remain its lexical
closure. Immediately after each indexed bank selection in
`_run_coordinate_blocks_reusing_projections`, cast the selected values to FP32
before the unchanged concatenate, active-row bias replacement, attention,
normalization and FFN operations. Family/process/carbon projections retain FP32.
Keep row order, original chunk128, padding, gathers, masks and all physical code.

The proposed wrapper uses `_DecoderBlockProjection._replace` on those four
fields at the outer macro boundary; the four inner gathers become, for example,
`projected.axis_key[axis_indices].astype(jnp.float32)` and
`projected.local_key[local_indices,None,:].astype(jnp.float32)`. The two value
gathers use the same immediate cast. The implementation is an isolated
execution overlay with original-helper identity and source-hash guards, not
a production-module change. No custom VJP/JVP is needed.

For finite FP32 banks/tangents, widening is exact and the gathered cast round
trip restores the same FP32 values and tangent values. The chosen numerical
change is native reverse accumulation: local gather transposes scatter-add
FP64 contributions to bank64, the outer map accumulates bank64 cotangents,
and the transpose of the outer widening finally returns FP32 to the original
projection. Projection AD and other upstream/downstream neural arithmetic stay
FP32; physics remains FP64. This changes the reverse numerical method and
therefore requires a scientific choice before real execution.

Widening a bank only inside each chunk does not suffice: its local reverse sum
is narrowed before the outer map reduces FP32 closure cotangents. Casting only
selected values after an FP32 gather is still narrower and leaves its scatter
FP32. The full-bank FP64 boundary must enclose the original map.

## Completed feasibility evidence

`outputs/diagnostics/shared-bank-adjoint-accumulation-preparation-20261002/`
contains `probe_bank_adjoint.py`, saved NumPy arrays, native StableHLO and
optimized CPU HLO, `cpu-evidence.json`, `canonical-cpu/`,
`saved-IR-scope-audit.json` and independent `source-review.json`.

Local ORCJAX JAX0.10.1 and one necessary test01 canonical JAX0.4.38 CPU run
tested repeated-index cancellation within one chunk, across single-row chunks,
and across three three-row chunks. Both versions preserve bit-equal original
primal, forward JVP and query VJP in every case. The independent oracle uses
NumPy FP64 indexed summation of the original FP32 local cotangent contributions:

| Case | FP64 bank oracle | Original max error: local / canonical | Inside-chunk widening error | Outside-map widening error |
|---|---|---:|---:|---:|
| Within chunk |`[2,2]` at bank row0|2 /2|0|0|
| Across chunks |`[2,2]` at bank row0|2 /2|2|0|
| Mixed three chunks |`[6,6]` at bank row0|4 /6|6|0|

The outside-map native VJP recovers the independent FP64 sum exactly and
returns FP32 bank/query cotangents in both versions. No custom AD, real model,
Teacher data, or GPU was used. These deliberately ill-conditioned toys prove
the mechanism can survive compilation; they do not quantify its model benefit.

Saved mixed-case StableHLO has an FP64 `scatter` and FP64 bank carry in the
transposed outer `while`, with FP32 local query buffers. Optimized local HLO
retains the FP64 scatter instruction. Canonical CPU lowers scatter into an
inner FP64 update loop; its enclosing chunk map still carries FP64 and adds
the chunk cotangent in FP64, then converts to FP32 before the original FP32
projection transpose. The scope-aware audit explicitly distinguishes that
inner scatter loop from the outer map. An arbitrary FP64 while alone does not
prove cross-chunk accumulation. Both inside-chunk variants keep an FP32 outer
carry despite any inner FP64 scatter loop.

This is CPU feasibility. A GPU diagnostic must inspect its actual optimized
HLO or equivalent executable evidence for retained FP64 gather transpose and
outer bank accumulation. If GPU compilation removes either boundary, classify
the precision route as ineffective in that backend and stop; do not claim an
implemented FP64 accumulation method from source casts alone.

## Source and deployment provenance

Verify graph project `E-ORCHIDEE-MAN-JAX`, generation
`2026-10-02T04:30:36Z`, reports no recorded issue for the three known paths but
metadata changed. Exact source fallback was read; no graph-completeness claim
is made. Main spans are `shared_transformer.py::_DecoderBlockProjection:350–360`,
`_field_dynamic_projection_cache:424–448`,
`_run_coordinate_blocks_reusing_projections:537–610` and
`shared_decoder_macro_packed.py::_decode_mapped_rows:25–76`.
The closed gather-hoist helper was inspected only for source context; none of
its reordered logic is copied or reopened.

Read-only test01 inspection confirmed current actual execution-source hashes:
checkout `shared-split-width-v1` shared transformer
`8af78fbdf969c97deb553da1ad2ec90fa4d817d2cdb79e3cfff0a9ad4dd89b9d`, and
staged `runtime/transfer/shared-combined-recurrent-20260926/`
`shared_decoder_macro_packed.py`
`689baa917ab518e87eae809af22900979d04ebed06825f8ff12f83f200e7158d`.
They match the audited local files. The macro is staged; it is absent from
the immutable checkout, so deployment must guard this actual staged identity.
The current expert adapter imports that macro route. This preparation changes
neural AD accumulation only and implements no new Fortran physical process.

## Proposed single free V100 T7 diagnostic

After approval of this precision choice, build **one candidate full
parameter-plus-initial-state AD** graph for the unchanged original public T7
objective: update0, artifact742/window0, original64,884,481 parameters/300 leaves,
3854 initial states, days/targets, masks, checkpoint policy and loss. No optimizer
update is proposed. Use the saved original FP32 full-AD result/executable
`shared-recurrent-checkpoint-policy-20260930-v2-T7-reference.json`,
`.reference.pkl` and `.reference.executable.pkl` under the existing remote
diagnostic root. Validate identities/ABI and replay that binary without compiling
another reference. Original self-drift remains diagnostic, never a state-gate waiver.

Return complete loss/auxiliary/parameter/state arrays and estimated memory,
physical/source masks and actual undefined-state gradients. Compare initial
candidate and three bounded interleaved warm pairs against matched original
full AD. Preserve every strict parameter and state comparison (`rtol=1e-5`,
`atol=1e-6`), all values finite, all14 hard counts zero, unchanged source masks,
21 undefined-state gradients exactly zero, and estimated/observed active memory
≤24GiB. Groupwise summaries are descriptive and do not replace strict gates.
Run all three pairs after valid hard admission even if a numerical comparison
fails; hard finite/physical/mask/undefined-zero/memory failure stops immediately.
Keep complete failed pairs and whole-process peaks without inventing arm peaks.

Reuse saved original FP32 JVP evidence from
`shared-original-state-jvp-20261002/model.directional-results.pkl` and its report
at basis coordinates33,199,428,200. Compare candidate state VJP coordinates to
these independent original forward derivatives, including known original JVP/VJP
disagreement. No candidate JVP compilation is needed for this bounded proposal.
Improvement or non-improvement here is a supplemental diagnostic, not proof of
cause and not permission to substitute a new self-consistent reference for the
original strict full-AD gate. The separate full-neural-FP64 pass remains context
only. Reuse completed inputs/references; do not sample new Teacher data.

The diagnostic measures unchanged complete T7 parameter/state derivative
workload with all three synchronized pair timings. Report original/candidate
medians and the existing1.05 speed screen separately from correctness. This
is not complete optimizer-update throughput and carries no speed prediction.

## Budget and stopping rule

Suggested single free GPU1 V100 on idle gln01, canonical JAX0.4.38,
OMP4/OpenBLAS1 and existing nonparallel LLVM compilation flag. No paid fallback,
GPU-precision flag sweep, automatic second backend or old-candidate retry.
Record node, affinity/load, visible GPU contexts and admission occupancy.
Run the required small actual-runtime check once before the new AD boundary.

One candidate backend cap **2800s**, lower cap300s, cumulative whole route
**3600s including preflight/setup, reference replay, comparisons/three pairs
and10s TERM/KILL grace**, at most1 free GPU card-hour/CNY0. All stage timeouts
are intersected with the shared remaining cap; the per-stage maxima are not
additional allowance. The backend exceeds the suggested2400s because the
saved original T7 full-AD backend actually took **2544.640612s**, with
lower174.487069s. A2400s cap would already censor that baseline. A2800s cap
provides only bounded headroom, not a compile-time guarantee; no silent
extension is allowed if the candidate needs more.

Compilation timeout/resource failure is incomplete compatibility evidence;
removed FP64 accumulation is an implementation failure; strict numerical or
speed failure is rejection. A successful diagnostic still does not adopt a
reference, train, execute T30/full updates, or reopen any closed variant.

## Concrete staged route and readiness

The reviewable implementation consists of
`scripts/hpc/shared_bank_adjoint_accumulation.py`,
`scripts/hpc/probe_shared_bank_adjoint_T7.py`,
`scripts/hpc/run_shared_bank_adjoint_T7.py` and
`scripts/hpc/run_shared_bank_adjoint_T7.sh`. The exact four sources are staged
under `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/`
`shared-bank-adjoint-accumulation-20261002`, with their source snapshots and
matching deployed SHA256 values recorded in the preparation evidence.
The declared immutable checkout remains `runtime/checkouts/shared-split-width-v1`
at `c599236bcf57b745e9bafed678298010f4c3b58f`. Guard the actual checkout transformer
and the separately staged original macro paths/hashes listed above.

The exact proposed command, **only after approval**, is:

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-bank-adjoint-accumulation-20261002/run_shared_bank_adjoint_T7.sh --gpu 1
```

Its unique output is
`runtime/outputs/diagnostics/shared-bank-adjoint-accumulation-T7-20261002`.
The launcher requires the free gln01 node, an empty selected GPU, no paid Slurm
allocation, and the exact immutable HEAD. The original public objective and
saved executable are reused with the existing staged asset/AD helpers. The
canonical GPU environment, foundationpose container and one nonparallel LLVM
flag are specified by the saved dry-run; the old operand-upcaster flag is absent.
Setup, preflight, sole model subprocess and kill grace share the 3600s cap.
Launch exceptions, timeout logging failures and a passed preflight followed by
a failed model launch all retain nonzero exits and actual timing records.

The preparation directory contains `runner-budget-checks.json`/`.log` (8 checks:
seven hard-stop/budget/exit checks plus pure mask-helper interface),
`overlay-checks.log` (7 CPU tests), `final-hlo-fixture-checks.log`,
`actual-format-direct-scatter-fixture.hlo`, `staged-interface-checks.json`/`.log`,
`dry-run-gpu0.log`, `dry-run-gpu1.log`, invalid-budget/GPU logs,
`assets-metadata.json`, timestamp records and `readiness.json`.
On test01, actual `bash -n` and both shell dry-run expansions passed; invalid
arguments exited2. Source guards were exercised against the actual deployed
split roots. All preparation read only source, JSON and file metadata; it made
no GPU context, model lowering/backend call or prepared-pickle/Teacher read.
Metadata readiness does **not** establish actual deserialize/input ABI acceptance.

The model runner reuses only the AST-extracted pure `mask_inventory` helper from
the existing staged FP64 diagnostic, source SHA256
`34da692a6d10f42a19a1ca63cd38b2a089a3db7af3237484f629ccdbc43ce843`,
function lines49–74. It does not import or execute that precision route.
The unchanged112-mask inventory spans the complete case, layout, normalization,
state scales, template and contract; case leaves alone are insufficient.

The optimized-HLO audit requires field-marked **reverse FP64 scatter-add** and
the matching FP64 outer carry tuple index. It resolves parameters separately
within called fusion computations and accepts either an explicit native FP64
add returned at that index or a native FP64 scatter that directly updates the
carried bank. A positive fixture uses genuine saved optimized CPU HLO with only
symbol suffixes and synthetic scope metadata added. Distinct bank shapes may
be covered across distinct marked macro loops; repeated application shapes are
deduplicated. Forward gather output shape is a selected/chunk shape, and a
forward FP64 gather may legally optimize to FP32; neither is a required reverse
accumulation proof.

This is a structural proof per field/deduplicated bank shape, not an exhaustive
GPU-IR proof of every repeated same-shaped bank occurrence in every decoder
block. Keep the per-block application metadata and exact source/alias guards;
the unchanged full parameter/state arrays and strict gates remain required.

This bounded text parser still returns **incomplete** for unsupported formats,
tuple-root fusions, renamed/missing scopes, changed carry representations or
lowered scatter implementations. Incompleteness is not evidence that XLA
removed the precision boundary. The runner saves the actual serialized candidate
and compressed optimized HLO, then stops before model evaluation if automatic
proof is incomplete. A targeted offline saved-executable/source audit can resolve
that limitation before any subsequent numerical evaluation; it cannot waive
strict numerical, physical, mask, memory or speed gates or silently compile a
second candidate. Preparation itself made no GPU call. The subsequently approved
execution has ended at its cap as described above; no candidate values, reference
adoption or training occurred. Additional GPU time needs a separate approval
because the cumulative allowance is exhausted, without repeating the precision
choice decision.
