# Cross-expert non-GPP decoder macro-row screen

Latest17:03CST: both reference-only probes completed/exit0 with20groups each.
Scalar trace/lower177.58s plus compile679.88s; micro4 205.26s plus680.94s.
Consolidated reference artifacts35,142,067/96,732,246bytes. These preparation
times are not a monolithic warm-execution baseline. Free expert128 micro4
candidate96056 began17:03:39 onGPU1,20min bound, restoring the saved micro4
reference. Only expert128 runs in this first candidate screen to reduce time
to a useful result; packed128 remains available for a later justified control.
Output `shared-macro-expert128-20260925-micro4.{json,log}`. Actual target-shell
resume/variant argument expansion checked; no fresh reference compilation.

Latest16:41CST: partitioned native-AD reference implemented and checked locally:
four new tests and six existing decoder tests pass, Ruff/compile checks clean,
and a tiny CLI reference-only/save/resume smoke passes. Fields from expert5
appear across partitions in scalar/micro4 parity tests. The objective uses
global-row normalization and sums all native shared-parameter/input gradients.
Completed pieces are saved atomically in one artifact; trace/lower and backend
compile are reported separately, with a stage saved before each operation.
Reference timings must not be used as the original monolithic baseline.

Free scalar/micro4 reference-only probes83546/83547 began16:41:11 on gln01
GPUs0/1, each20min ceiling; both GPUs empty at dispatch. Isolated overlay
`runtime/transfer/shared-macro-partitioned-20260925`; report/log/artifact stem
`runtime/outputs/diagnostics/shared-macro-partitioned-20260925-micro{1,4}`.
Shell syntax and actual target-shell argument expansion pass. These prepare
the correctness oracle only; candidate speed measurement follows a completed
reference. No whole-model compile, paid submission or scientific training.

Status: local scalar and micro4 CPU numerical probes pass, 2026-09-25.
This is an unpromoted diagnostic fragment. The accepted decoder, model
parameters, training runner, masks, outputs and loss are unchanged. No
successful deployed GPU or full-model compile was run. The first actual-layout
GPU attempts below stopped before compilation because of filesystem waiting.

Completed: both resumed probes485/486 exited124 at20minutes during
`reference64_trace_lower_compile`. Micro4 emitted XLA's slow-compilation alarm.
Neither saved a reference result nor executed a candidate; no GPU speed or
numerical candidate verdict. Do not repeat this monolithic reference compile.
Next construct the same synthetic objective's reference outputs/native
gradients in bounded independent pieces, preserving summed shared-parameter
contributions. Such modular-oracle timing is not the original monolithic
performance reference and must not inflate a candidate speedup claim.

September25 15:36:23CST: resumed free scalar/micro4 screens485/486 on GPUs0/1,
each20min bound, after a bounded JAX import/CPU-device check completed. They
progressed to `reference64_trace_lower_compile`; Python PIDs610/611. Outputs
use `shared-split-macro-packed-20260925-micro{1,4}-resume2.{json,log}`.
The comparison is reference64 versus packed128 and expert128; no speed
claim until measured results and full output/gradient parity complete.

## Source-layout scope

`outputs/shared-decoder-packing-audit-20260923.json` records 161 decoder
fields and 8,550 eligible coordinates. Excluding the sole raw-carbon-bank
field (`daily.fast_gpp`, two coordinates) leaves 160 non-carbon fields,
8,548 rows and 264 field-local 64-row chunks. A cross-field map at 128,
256 or 512 rows would schedule 67, 34 or 17 chunks, respectively. It
does not omit any eligible coordinates. Padding every row to the maximum
five axis slots raises axis-slot work from 32,595 to 42,740, or 31.1%.
These counts are layout arithmetic, not a throughput prediction.

## Bounded implementation

`shared_decoder_macro_packed.py` keeps one field-local dynamic axis/local
K/V bank per field, offsets its gather indices into concatenated banks,
and retains each row's query, masks and active flag. Family/process
projection caches are selected by the row's expert index. The existing
`_attention_from_projected_segments` implements attention, and the
result is sliced back into the original per-field output lengths. The
reference preserves independent field-local 64-row maps through the
production `_run_coordinate_blocks_reusing_projections` helper. Both
paths use native JAX reverse AD and the same two-block readout. Carbon
banks are rejected, not silently ignored.

`probe_shared_decoder_macro_packed.py` exercises eight synthetic fields
across all seven experts, 0-5 axis slots, mixed active/attention masks,
one 896-entry local bank, and two fields sharing an expert but not their
local bank. It is deliberately not the full 160-field layout. Scalar
and four-distinct-sample microbatch modes compare the entire loss/output
and parameter/payload-gradient pytrees at unchanged `rtol=1e-5`,
`atol=1e-6`. All three candidate chunks (16, 32 and 64) passed, with
zero failed coordinates:

| Local CPU mode | Output/loss coordinates | Gradient coordinates | Maximum absolute gradient difference |
| --- | ---: | ---: | ---: |
| Scalar | 47 | 25,825 | 3.43e-7 |
| Micro4 | 185 | 82,753 | 3.13e-7 |

The maxima above are across the three chunk sizes. Reports are
`outputs/shared-decoder-macro-packed-scalar-local.json` and
`outputs/shared-decoder-macro-packed-micro4-local.json`. Four focused
unit tests and Ruff pass in `ORCJAX` on Windows with JAX 0.10.1 CPU.

The one-repeat timing samples are diagnostic only. On micro4, the
reference's compiler temporary memory was 4.24 MB, while packed16,
packed32 and packed64 required 12.25, 16.30 and 7.37 MB; their warm
times were 11.48, 9.93 and 9.34 ms versus 8.19 ms reference. Thus this
fixture does not show a CPU speed or memory gain. The per-row gather
replicates family/process banks and the production helper re-normalizes
their identical projected keys per row, potentially increasing AD
intermediates. Pre-normalizing shared keys once would require a separately
verified attention implementation because the production helper always
normalizes internally; this screen does not make that change.

No width-384 deployed-V100, actual-layout, daily/T7/T30 gradient,
physical or complete-update result exists for this candidate. The
prior same-expert Q1 and five-field carbon-group screens remain separate
evidence. This fragment is not an execution recommendation or training
admission.

## Actual-layout staged screen

The source-only exporter now writes
`outputs/shared-decoder-macro-layout-20260925.json` (175,378 bytes) from
`_load_static_layout` and `_decoder_dynamic_plan`. It records all 160
non-GPP field names/indices, expert identities, eligible row counts,
inverse axis/local gathers, field-local bank sizes, and one boolean per
local-bank entry indicating whether its source state index is mapped
(`>=0`). The JSON loader checks schema, family/expert counts, field
order, 8,548-row total, bank bounds, and masks. The actual-layout runtime
uses only this JSON, never the unit fixture, contract, or source manifest.
Its values and active/attention masks remain synthetic FP32 values;
unmapped local entries are always masked off.

The runtime accepts `--layout-metadata-file` and defaults to reference64
then packed128/256/512 for that layout. `--variants packed128 ...` can
bound the candidate set. The result report is updated before lowering,
after compile, and after each comparison. After reference execution,
one `.reference.pkl` artifact beside the JSON report stores the complete
loss/output and parameter/payload gradients. A later invocation with
`--resume-reference` checks width, microbatch and layout digest before
using it; this is for recovery after a candidate compile timeout, not
cross-configuration comparison. Tiny-fixture mode remains the default.

The new JSON-only regression exports the metadata locally, patches the
static builder to fail, and checks runtime loading plus 160-field
synthetic fixture construction. The changed tiny CLI path passes. No
actual-layout or width-384 compilation was run locally; deployed GPU
results are pending.

## Same-expert alternative

`expert_decode_fields` groups the non-carbon fields by their seven expert
identities, retaining field-local axis/local K/V banks and offset gathers.
It pads axis slots only to each expert group's maximum and calls the
existing `_decode_mapped_rows` / production coordinate-block helper with
that expert's shared family/process projection cache. Thus it does not
gather or renormalize the common bank per query as the cross-expert
candidate does. No attention arithmetic is copied or changed. Outputs
are restored to original field order and lengths; carbon banks remain
explicitly unsupported.

From the exported actual layout, expert-group 128/256 maps would use
71/38 chunks, versus 264 field-local64 chunks and 67/34 global
cross-expert chunks. Within-expert axis-slot padding raises the 32,595
real slots to 39,680 (21.7%); global five-slot padding uses 42,740
(31.1%). The largest expert, litter/DOC, owns 42 fields and 7,542 of
8,548 rows. These are static work counts only; compilation, memory,
warm speed, and full-model effects are unmeasured.

The tiny heterogeneous fixture passes `--variants expert128 expert256`
with scalar and micro4 native AD at the same `1e-5/1e-6` gate. Each
mode has zero failures over 47 scalar or 185 micro4 output/loss
coordinates and 25,825 scalar or 82,753 micro4 parameter/payload
gradient coordinates. The maximum absolute gradient difference is
5.63e-7 scalar and 3.06e-7 micro4. Reports are
`outputs/shared-decoder-macro-expert-scalar-local.json` and
`outputs/shared-decoder-macro-expert-micro4-local.json`. Every tiny
expert group fits in one chunk, so this proves numerical routing and
gradient parity, not a 128-versus-256 scheduling difference. The probe
records a numerical rejection per candidate and continues to later
candidates, while a compiler-memory limit still stops the run. The
deployed actual-layout screen remains pending.

## Deployed attempts and current blocker

Free gln01 scalar/micro4 launchers96824/96825 began around11:39CST on
September25, after both cards were released from the filesystem-blocked dual
attempt. Both target-shell command expansions had passed dry-run. By11:44,
Python96900/96960 each had zero CPU use and waited in `ptlrpc_set_wait` before
JAX initialization; both GPUs remained at0MiB. Even a `stat` of the staged
probe from free test02 produced no file result within its10second timeout.
This broadens the evidence beyond Teacher inventory access; it does not
establish the server-side cause of the Lustre delay. The lead sent TERM to
both verified owned Python processes. No reference or candidate compile,
timing, numerical comparison or optimizer update was completed.

Logs remain at
`runtime/outputs/diagnostics/shared-split-macro-packed-20260925-micro{1,4}.log`.
No new paid work was submitted. Next first confirm a small metadata/read check
on free resources returns promptly, then resume these bounded screens and the
saved-executable dual comparison. The local launchers now add30seconds of
kill-after grace to their finite20/30minute caps; retransfer them before reuse.
