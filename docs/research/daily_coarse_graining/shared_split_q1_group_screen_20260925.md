# Split decoder within-expert Q1 grouping screen

Status: bounded scalar and micro4 actual-helper GPU screens passed,
2026-09-25. The production model, capacity, outputs, masks, losses, data and
training runner are unchanged. No full-model compile or optimizer update was
performed.

## Audited scope and mechanism

The static exact-shape audit in `outputs/shared-decoder-packing-audit-20260923.json`
has one 45-field signature with exactly one eligible coordinate, one chunk,
zero axis slots/bank entries, one local-bank entry and no carbon bank. Of
those fields, 22 belong to `hydrology_snow` (field indices 8, 9, 14, 15, 40,
42, 44, 45, 46, 53, 56, 61, 118, 119, 124, 125, 129, 133, 135, 138,
141, 142). The accepted layout has 20 source families. This is the only
group exercised; `daily.fast_gpp` and the raw-carbon bank are excluded.

The reference uses the production `reference_decode_fields`, with 22
independent calls to `_run_coordinate_blocks_reusing_projections`, each field
retaining its own query, mask and local K/V projection. The candidate
concatenates those 22 prepared one-row fields into one call of that same
production two-block helper and shared readout, then splits the 22 scalar
outputs. Both paths reuse the one expert's family/process projection cache.
Each field-local K/V is still projected once inside native AD and kept
distinct. Query/FFN/readout matrix work can become larger GEMMs; no
coordinate is omitted and no mask is relaxed. The candidate helper requires
a same-expert builder and is not valid for arbitrary fields with different
shared banks.

This is different from the rejected width768 eight-field outer `lax.map`:
the actual split-width decoder helper and the real 22-field Q1 shape are
used, and there is no runtime outer map or cross-expert bank selection. It
also differs from block-major dense packing: attention itself is grouped
within one shared expert, not kept as 22 separate attention calls while
packing only FFN/readout. The earlier failures still warn against broad
field batching and memory extrapolation.

## Local checks

The diagnostic uses FP32 payloads, width384, two coordinate blocks, FFN1536,
eight heads and the full common readout. It includes nonzero K/V biases,
inactive rows, heterogeneous family/local masks, one shared expert bank and
separate field-local tokens. The strict existing execution parity gate is
rtol 1e-5/atol 1e-6 for output, loss and every parameter/input gradient.

| Variant | Primal JAXPR equations | Lower s | Compile/load s | Warm ms | Compiler temporary bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 22 unrolled decodes | 9,712 | 4.232 | 30.013 | 28.713 | 3,342,468 |
| One grouped 22-row decode | 909 | 0.411 | 1.927 | 24.204 | 1,031,328 |

All 23 output/loss and 3,672,705 parameter/payload-gradient coordinates
pass, with zero tolerance failures. Grouping reduces this component's primal
equation count 90.6%, lowering time 90.3%, warm CPU time 15.7% and compiler
temporary bytes 69.1%. The tiny width16 regression and width384 CPU probe
both pass. Source: `outputs/shared-decoder-q1-group-local.json`; two focused
test and Ruff pass. CPU JAX is 0.10.1, not deployed V100/JAX 0.4.38.

Free `gln01` GPU1 scalar probe 53850 also passed on deployed JAX 0.4.38/V100:

| Variant | Primal JAXPR equations | Lower s | Compile/load s | Warm ms | Compiler temporary bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 22 unrolled decodes | 9,758 | 6.059 | 62.142 | 7.0721 | 43,861,520 |
| One grouped 22-row decode | 933 | 0.571 | 7.394 | 2.6624 | 3,496,976 |

All 23 output/loss and 3,672,705 parameter/payload-gradient coordinates
pass strict parity; maximum absolute differences are 4.17e-7 and 4.77e-7.
The grouped fragment saves 4.41 ms (62.35%) and 92.0% compiler temporary
bytes on this scalar GPU component. Sequential compile/load is shorter by
54.75 s; these are component compilations, not full-model cache/startup
measurements. Copied report: `outputs/shared-split-q1-group-20260925.json`.

Free `gln01` GPU1 micro4 probe 54879 also passed on deployed JAX 0.4.38/V100:

| Variant | Primal JAXPR equations | Lower s | Compile/load s | Warm ms | Compiler temporary bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 22 unrolled decodes, micro4 | 10,341 | 6.565 | 69.978 | 6.9952 | 13,350,872 |
| One grouped decode, micro4 | 1,054 | 0.825 | 7.636 | 2.8363 | 8,052,192 |

All 89 output/loss and 3,747,585 parameter plus four-sample payload-gradient
coordinates pass strict parity with zero failures. Maximum absolute
differences are 5.66e-7 and 1.34e-7. Grouping saves 4.16 ms (59.45%) and
39.69% compiler temporary bytes in this fragment. Both variants have equal
14,990,340 compiler argument bytes. Copied report:
`outputs/shared-split-q1-group-20260925-micro4.json`.

These component results do not predict full-model cold compilation or
throughput. Field feature assembly, the 22 production eligible-coordinate
scatters, other field groups, GPP carbon attention, tokenizer, physical
updater, recurrent state and all non-Q1 branches are excluded. The large
saved full-model HLO cannot be attributed to this group from the local
equation count. The absolute 4.16 ms micro4 fragment saving is small relative
to the hundreds of milliseconds in a complete daily micro4 reverse. No
full-model cold compile, daily/T7/T30 gradient, complete-update timing, or
science result has been measured for a grouped model. This is a candidate
for targeted integration planning, not production adoption or a reason to
start learning.

Although micro4 passes, this Q1 group mainly tests graph replication and
tiny calls, not the decoder's largest arithmetic work. The same static
audit identifies a separate `litter_poc_doc` group of five fields with 448
eligible coordinates each, seven 64-query chunks per field, four axis slots,
42 distinct axis-bank entries and one local-bank entry. A later bounded
actual-helper 448x5 test, or a source-equivalent 128/256 query-chunk test,
could probe larger GEMMs and the costlier repeated work. Neither is
implemented or performance-admitted here. Do not launch a full-model compile
solely because this 22-scalar-field fragment is favorable.

## GPU execution record

The two Python files were staged under `runtime/transfer/`, leaving the
accepted `c599236` checkout immutable. Free `gln01` GPU1 used the accepted
`foundationpose` container through
`runtime/transfer/launch_shared_q1_group.sh` with `MICROBATCH=1`
(PID 53850) and `MICROBATCH=4` (PID 54879); both completed exit 0 within
their 10-minute limits. The micro4 launch from `gln01` was:

```bash
MICROBATCH=4 bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/launch_shared_q1_group.sh
```

The updated probe preserves scalar `--microbatch 1` as its default. For
micro4 it uses four distinct FP32 sample payload trees, vmaps the same
one-sample objective, and averages the four losses; all shared-parameter
and per-sample payload gradients are compared. A tiny width16 micro4 CPU
preflight passes 89 output/loss and 11,009 gradient coordinates, although
its grouped compiler temporary bytes are 7.2% higher than reference; the
actual V100 micro4 result above resolves that fragment's memory comparison
in favor of grouping. The GPU probe exits nonzero if either strict numerical
comparison fails. Its
report separates compilation/lowering, compiler temporary/argument bytes,
and warm component timing. A fragment win would only warrant a targeted
integration plan and saved-reference daily/T7/T30 gradients, physical and
complete-update checks, not immediate full-model adoption or learning.
