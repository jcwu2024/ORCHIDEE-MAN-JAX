# Cross-expert non-GPP decoder macro-row screen

Status: local scalar and micro4 CPU numerical probes pass, 2026-09-25.
This is an unpromoted diagnostic fragment. The accepted decoder, model
parameters, training runner, masks, outputs and loss are unchanged. No
deployed GPU or full-model compile was run.

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
