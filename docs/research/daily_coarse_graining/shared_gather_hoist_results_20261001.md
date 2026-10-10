# Original FP32 expert5 gather-only execution diagnostic

Status: **closed at the real component, without observed speed benefit**.
The intended gather/scatter relocation was confirmed in compiled HLO. Forward
values pass the original strict gate; native parameter and raw-input VJPs fail
in the initial and every one of seven alternating pairs. This specific
candidate is not worth expanding to the expensive public four-sample graph.
No public loss, physical constraints, four-sample3854 state AD, optimizer,
training or paid job was executed. This does not reject broader indexing or
gradient strategies. The separate approved FFN-only screen also remains
state-gradient rejected; see the current precision report.

## Exact treatment and source evidence

The original expert5 path has7542 rows,59 original128-row chunks and ten padding
rows to7552. The candidate pads the axis/local indices first, then gathers raw
axis K/V and local K/V for both coordinate blocks before the same `lax.map`.
Concatenation, active-row `where`/bias replacement, every learned projection,
attention/softmax, normalization, FFN and public readout remain inside each
original128-row body. Padding keeps the original index0 selection and inactive
bias substitution. All neural formulas/dtypes remain FP32; physical FP64 is
untouched by the source transformation and was not exercised by this component.

Source provenance is `shared_decoder_macro_packed._decode_mapped_rows:25-76`,
`shared_transformer._run_coordinate_blocks_reusing_projections:537-610` and
`_attention_from_projected_segments:451-534`. The isolated adapter is
`scripts/hpc/shared_chunk_gather_hoist_execution.py`. It patches only the actual
macro `_decode_mapped_rows` for7542-row expert5 and restores it on every exit.
Other routes fall through to the original helper. Source helper identities and
actual paths/hashes are recorded. The component entry independently verifies
original `st`/strong-baseline linear, FFN and layer-norm aliases, original
384/eight-head/two-block scope and FP32 parameter/raw-input dtypes. It never
patches any helper. The initial review identified this direct-entry guard as
missing; it was added before the first GPU execution, without a failed model run.

G2 changes map batching, V4 replaces the complete chunk map with vmap, and R4
moves the full coordinate runner while retaining learned128 shapes. None is the
standalone gather-only transformation tested here; those closed sweeps were not
reopened. Graph project/coverage transport remained closed, with no confirmed
generation. Verify-tier bounded exact source/HLO/capture fallback is recorded;
no graph completeness claim is made. The independent Luna/high audits are in
`gather-hoist-feasibility-20261001.*` and `compiled-transform-audit-20261001.*`.

## Actual runtime, inputs and checks

The immutable checkout is `runtime/checkouts/shared-split-width-v1` at
`c599236bcf57b745e9bafed678298010f4c3b58f`, with JAX/JAXlib0.4.38 in the accepted
canonical GPU venv inside foundationpose Singularity. Free gln01 GPU0 is one
V100-SXM2-32GB on ibc13b02n04, OMP4/OpenBLAS1. Before each invocation both GPUs
had0MiB/0% and no other GPU compute process; component-start host load was
.10/.09/.07 and affinity0-111. These observations are not node exclusivity or
a causal explanation of timing. Node timestamps and `perf_counter` durations
are kept separately.

An actual deployed runtime preflight passed, with no Teacher read or real-model
claim. CPU checks used the complete canonical two-block decoder including
readout, repeated indices, inactive rows and padding; output was bitwise equal,
all1946 parameter/input/projection-bank AD coordinates passed unchanged
rtol1e-5/atol1e-6, maximum difference0. This is small CPU equivalence evidence.
Target `bash -n` and actual argument expansion for preflight/component passed.
Output guards precede the EXIT trap and cover initial/binary/repeated sidecars.

The GPU component uses the unchanged artifact742/update0 real expert5 sample0
raw-bank capture, original sin(arange(output_size)*.11) JAX cotangent, original
two-block parameters, masks and indices. Both saved original forward and native
VJP executables are loaded and hashed, with **zero original graph recompiles**.
Only candidate forward/VJP binaries compile. The original coordinate-only
capture excludes public readout, so both compared component arms exclude it;
the public adapter retains readout inside the original map. Scope is
2,896,128 output coordinates,3,548,928 decoder parameters and21,579,648 raw
input/bank derivative coordinates. These are not the64,884,481 full-model
parameters or15,416 public four-sample state derivatives.

## Numerical and timing results

| Initial strict comparison, rtol1e-5/atol1e-6 | Failed coordinates | Maximum absolute difference |
| --- | ---: | ---: |
| Output |0/2,896,128 |1.311302e-6 |
| Native decoder parameter VJP |720,360/3,548,928 |1.726151e-4 |
| Native raw-bank/input VJP |1420/21,579,648 |1.636147e-5 |

All seven paired output checks pass; parameter failures range719,652–720,430
and raw-input failures1390–1405. Local verification of all16 retained initial/
repeated trees confirms their complete scope counts, FP32 dtypes and finite
values. Initial gradient trees in both arms have nonzero derivatives. The original binary's self bridge to
its older preserved result is bitwise for output and passes raw-input VJP;
parameter self comparison has1624 failures. Within-arm and full raw repeats are
retained as diagnostics, with no noise waiver or enlarged tolerance.

| Warm median / memory | Original | Gather-only | Original/candidate ratio |
| --- | ---: | ---: | ---: |
| Forward seconds |.024373395 |.024384730 |.999535193 |
| Complete native VJP seconds |.099313354 |.099659441 |.996527304 |
| Compiler estimated live bytes |2,260,028,960 |2,366,028,560 |— |

The candidate's median VJP is about .35% slower, without evidence of a
statistically significant regression. No stable usable gain is observed.
Estimated live bytes increase106,000,000 (4.69%); cumulative observed active
peak is2,667,639,808bytes, below the12GiB component limit. Candidate lowering
and backend times are in the summary; backend forward14.110018s and
VJP8.437068s, internal total49.044226s.

VJP times form two clusters near .099 and .118s. In repeats1–6 the first arm
is slower regardless of candidate identity; repeat0 is an exception. Thus the
overall medians require an order caveat. Using the saved times without another
GPU run, all-seven first/second-position ratios are .988194/.992044. The
balanced repeats1–6 give .990672/.992640, with first medians
.117244836/.118348772 and second medians .098724352/.099456351. Both
positions show no candidate benefit. Causes such as thermal state, CPU work or
transfer effects were not measured and are not assigned.

## What compiled HLO confirms

The original map body has eight raw gathers across two blocks, with axis bank
`[1,906,384]` and local bank `[1,2476,384]`. The candidate moves all eight
gathers to entry scope over all padded rows. Original reverse scatters use
per-chunk axis/local index vectors `[640,2]`/`[128,2]`; the candidate's eight
scatters are at entry scope with full vectors `[37760,2]`/`[7552,2]`.
This is the requested transformation, rather than compiler movement back into
the loop. No repair/recompile or additional component GPU probe is justified.

Original reverse carries contain four axis and four local bank-shaped fields.
They disappear from the new reverse loop, but tensor carry slots fall only
237→235. Four full-row axis arrays `[59,1,128,5,384]` and four local selections
within the larger collection of `[59,1,128,1,384]` arrays replace much of that
state. Whole-row raw gathered payload is about265.5MiB across both blocks,
versus2.25MiB/block for each original chunk. Scatter and residual storage are
relocated; native reverse work is not eliminated. Timing causality has not
been isolated, so this structure explains what changed, not a measured time
breakdown.

Both optimized native-VJP HLOs have247 `dot_general` **metadata annotations**
and retain128-row/384-width learned-body dimensions. This is not a device dot
count or proof that algorithms, layouts, reductions or bitwise values are
unchanged. Floating-point execution and scatter accumulation may change under
the transformation; strict actual results above take precedence over algebraic
equivalence or shape preservation.

## Costs, retained artifacts and disposition

| Free GPU invocation | Launcher seconds | Assigned card-hours | Exit/status |
| --- | ---: | ---: | --- |
| Runtime preflight |41.696509 |.011582363711 |0/passed |
| Real component initial plus seven pairs |51.499566 |.014305434889 |1/parity_failed |
| Total |93.196075 |.025887798600 |Paid0 |

Budgets were300s preflight wrapper,900s component wrapper,890s component
process and180s per candidate backend. Accounting includes host/import/compile/
comparison work with one assigned GPU, not utilization. The prepared daily
full-AD source/launcher was not staged or executed after the component result.
There is no full-update throughput, scientific or production admission.

All26 remote artifacts are downloaded and size verified under
`outputs/diagnostics/shared-gather-hoist-20261001/`: initial full output/VJP
trees for both arms,14 full repeated output/VJP trees, candidate forward/native
VJP executables, optimized HLO, logs, preflight/results and EXIT records.
`artifact-manifest.json` gives actual paths, sizes and copy status;
`component-summary.json` gives metrics, order groups, source hashes and costs;
`launcher-accounting.json` provides the dedicated execution ledger. Exact
executed overlay files are retained in `executed-snapshot/`. Baseline executable
identities and SHA256 remain recorded in the measured component JSON.

The narrow gather-only candidate is closed for absent observed benefit and
strict derivative rejection. No new FP32 sweep or public graph compilation
follows from this result. The separately approved FFN-only screen completed
on2026-10-02 and failed47 sample-state fields; it grants no full-update admission.
