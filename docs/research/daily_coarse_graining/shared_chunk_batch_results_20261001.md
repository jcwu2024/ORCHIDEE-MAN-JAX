# Original FP32 expert chunk batching

Status: whole-chunk vmap is4.546855x faster at the component; a learned-shape
repair is1.945904x faster and passes the component output gate. Both fail
the unchanged synthetic-cotangent parameter/input gates. The original
four-sample public daily-loss repair passes all strict numerical/physical
checks and gives1.403892x speed. T7 loss/parameter gradients pass but the strict
initial-state gate fails, including a persistent difference beyond observed
self noise. This closes R4 recurrent progression; its accepted daily assets remain.
No production adoption, recurrent acceptance or training has occurred.

## Bounded scope and prior evidence

The named September23 dense-stage report tested width768, four heterogeneous
fields of64/61/58/64 rows, concatenation after field-local attention and
different tolerances. Its packed-all path is not whole-expert chunk vmap.
The September26 FFN fragment's vmap is over distinct samples. Targeted source
inspection found the original macro-packed coordinate helper and dense-stage
chunk helper still use native `lax.map`. This is bounded source evidence;
graph transport remains unavailable, so no fresh exhaustive graph claim is made.

The October1 treatment uses the preserved real raw-bank capture, sample0,
expert5,7542 rows at width384/eight heads/two original coordinate blocks.
It preserves59 chunks of128 rows, including10 padding rows, original masks,
projection reuse, FP32 arithmetic, native AD and the original unscaled
JAX `sin(arange(output_size)*0.11)` cotangent. It replaces only the chunk-axis
mapping primitive with native vmap. The leading sample-axis vmap stays in place.

Original mathematical helpers are reused from
`research/daily_coarse_graining/daily_operator/shared_transformer.py`:
`_project_key_value`375-376, `_attention_from_projected_segments`438-534,
`_run_coordinate_blocks_reusing_projections`537-610. Packing follows
`shared_decoder_macro_packed._decode_mapped_rows`32-76. No physical process
or Fortran-derived behavior is reimplemented by the diagnostic.

## Same-runtime component result

Original legacy JAX/JAXlib0.4.38, original min-group5/LLVM parallelism false,
one free V100 on gln01/ibc13b02n04, OMP4. The saved original threshold-default
forward and VJP binaries are loaded, with identities and SHA256 recorded;
the original component is not recompiled. Only vmap component binaries compile.
Original libcuda API12000 and libcudart API12010 are checked from loaded paths.
An actual deployed public runtime preflight passed before the changed graph.

| Scope | Original map | Whole vmap | Ratio original/candidate |
| --- | ---: | ---: | ---: |
| Median warm forward | 0.024642522s | 0.008357555s | 2.948532x |
| Median warm native full VJP | 0.102475636s | 0.022537695s | 4.546855x |
| Compiler estimated live bytes | 2,260,028,960 | 1,015,002,072 | 55.09% lower |

Seven warm pairs alternate ordering. Vmap lowering0.912118s; backend
forward2.966089s and native VJP6.919636s. Two-binary cumulative allocator
peak2,633,146,880 bytes is separate from per-executable live estimates.
Both initial outputs/VJPs are finite. All3,548,928 decoder parameter
coordinates have nonzero derivatives;4,186,752 of21,579,648 input coordinates
have nonzero derivatives, with masks and inactive coordinates retained.

Fixed comparison remains `rtol=1e-5, atol=1e-6`.

| Initial comparison | Coordinates | Failed | Maximum absolute error |
| --- | ---: | ---: | ---: |
| Output | 2,896,128 | 1,040 | 2.682209e-6 |
| All decoder parameter VJP | 3,548,928 | 1,505,605 | 3.962517e-4 |
| All captured raw-bank/input VJP | 21,579,648 | 4,087 | 3.099442e-5 |

All seven cross-arm output comparisons have1040 failures. Cross-arm parameter
failures range1,505,541-1,505,978; input failures4075-4102. Original self
parameter failures1350-1792; candidate self2076-2509. Both arms' self output
comparisons are bitwise equal and self input comparisons pass. Comparing
replayed original baseline with its preserved old result gives bitwise-equal
output, passing input VJP, and1679 parameter failures; it preserves the
previously known original scatter variation rather than certifying self stability.

The complete repeated parameter trees identify1,484,463 coordinates failing
every matched pair, and1,476,100 whose two seven-repeat intervals are disjoint
by more than the largest unchanged strict allowance. All four FFN matrix
gradient leaves are internally bitwise stable in both arms;1,065,694 of their
coordinates fail every pair. These differences exceed self scatter variation.
The synthetic component gate fails; its self failure is not a waiver.
Actual public-loss evidence, if collected, is a separate cotangent/scientific
diagnostic and cannot retroactively make this component gate pass.

## Causal evidence and next repair

The optimized HLO maps query/output/FFN dot row dimension128 to7552 under
vmap. FFN examples are `[128,384]@[384,1536]` versus
`[7552,384]@[384,1536]`. Shared-family score GEMM changes
`[8,20,48]@[8,48,128]` to `[8,20,48]@[8,48,7552]`;
attention-value GEMM similarly changes128 to7552. Shared raw-bank projection
remains `[3402,384]@[384,1536]`. Native learned-matrix gradients change
contracting dimension128 with original loop-carried accumulation/beta1 to
one dimension7552 contraction/beta0. Head normalization and layer norm
retain48/384 feature reductions but row layouts change. This establishes
changed numerical execution; it does not attribute every error to one kernel.

The cheap causal screen compares real first-query projection, head
normalization and family-score stages on identical original-map stage inputs.
Screen outputs can change compiler fusion and are labelled accordingly.
The causal screen completed on identical real original-map inputs per stage:
query projection fails9993/2,899,968 padded outputs, maximum1.907349e-5;
head normalization and family score are bitwise equal. Real weights are
captured constants in this stage screen, so full-graph algorithm identity is
not inferred from it. It isolates a changed first-query learned GEMM as one
forward source; other changed stages remain relevant.

The first repair reuses the canonical block runner over7552 padded rows;
only learned `_linear` and parameterized `_layer_norm` calls keep native
128-row maps. All remaining coordinate operations are batched. Native AD
retains128-row learned-matrix contractions with loop accumulation/beta1,
as verified from optimized HLO. No parameter replication or custom AD is
introduced. Learned dot shapes alone do not preserve every algorithm:
first-query output layout changes `{0,1}` to `{1,0}`, and selected algorithm
18 to-1. These are concrete residual numerical execution differences.

| Repair component | Original | Preserve128 | Ratio |
| --- | ---: | ---: | ---: |
| Median forward | 0.024291605s | 0.019124132s | 1.270207x |
| Median native full VJP | 0.101790551s | 0.052310171s | 1.945904x |
| Estimated live bytes | 2,260,028,960 | 1,219,793,080 | 46.03% lower |

The repair output passes all2,896,128 coordinates with max1.311302e-6.
Initial parameter VJP fails718009 coordinates/max1.225471e-4; raw/input
VJP fails1379/max1.424551e-5. All initial trees remain finite/meaningful;
the unchanged strict parameter/input and self gates fail. Repeat interval
analysis identifies697512 parameter coordinates failing every pair and
690208 disjoint ranges beyond the largest strict allowance, including510438
internally bitwise-stable FFN matrix coordinates. None is waived.

Because the repair still gives useful component speed, a distinct actual
public-loss screen completed: original four daily samples, all64,884,481
parameters,160 loss/auxiliary coordinates,11 physical hard counts and24GiB
memory checks. It reuses the admitted original executable/results and public
loss utilities; it patches only the7542-row expert5 route, including its
original readout. Other routes use the original adapter. Budgets:2100s Python
process,240s lower,1500s backend (external process2110s); based on the earlier
123s lowering/1008s backend original-four normalized-key diagnostic.
All numerical outcomes and seven alternating warm pairs are preserved.
T7/T30 full state AD and complete-update gates
remain separate; the failed synthetic component gate stays recorded.

The daily adapter records exactly one traced expert5 route at7542 rows,
7552 padded rows/59 chunks/128. The outer native sample vmap retains the
four original distinct samples. The model diagnostic additionally includes
the readout: its canonical `_linear` calls have first dimension7552 and
are therefore restored to128-row maps by the same learned-shape wrapper.
This readout scope was absent from the component; it is explicitly new
model evidence rather than inferred from component tests.

| Public daily check | Result |
| --- | --- |
| Loss | Pass; max1.040928e-8 |
|160 auxiliary coordinates | Pass; max2.479168e-7 |
|64,884,481 parameter gradients/300 leaves | Pass; max1.341105e-7 |
| Finite/meaningful gradients | Pass;64,862,209 nonzero; norm10.394024386 |
|11 physical hard counts | All zero |
|24GiB process memory gate | Pass; peak11,362,754,816 bytes |

Daily original median0.374887692s, repair0.267034534s, ratio1.403892170x
from seven alternating synchronized pairs. Candidate estimated live memory
8,268,431,164 bytes versus original11,344,311,092 bytes. Process peak is
cumulative across both binaries; it is not a per-candidate observed peak.
Lowering113.112480s/backend737.001633s/Python884.637010s, exit0.
The replayed baseline has bitwise-equal saved loss/auxiliary and passing
all parameter gradients(max2.980232e-8), with zero baseline model recompiles.

T7 completed after the passed real daily gate, using its durable original
dual-AD reference and original full3854-coordinate initial state. Budgets
360s lower/5400s backend/7200s Python process (external7210s), based on the
prior original T7 default2545s and normalized candidate3364s backend costs.
Default native checkpointing, original FP32/FP64 boundaries and strict
tolerances are retained. A separate T30 initial-state reference has not
been established by the named current T30 parameter-only admission; it
must be verified/built if T7 actually passes, rather than assumed from
the old two-process GPU dispatch labelled dual.

Prepared `original-T30-reference` and `shape-preserved-T30` launcher modes
retain the same 360/5400/7200 second budgets. They require the completed
T7 report to pass the initial parameter/state comparison and all seven
full derivative paired replays before loading inputs. Self replays must be
recorded and remain noise diagnostics; baseline self drift alone is not a
candidate rejection. The generic recurrent driver now includes all paired
state replays in its final status; a separate saved-binary replay supplies
complete repeated parameter evidence before adoption. The completed R4
T7 used its preserved earlier snapshot with repeated state comparisons
and only an initial complete parameter comparison. A saved-binary replay must
supply seven complete parameter/state pairs before admission; its initial-only
final status is insufficient. The R4 state gate failed, so these R4 replay/T30
modes remain unlaunched. Target-shell syntax and the prepared replay/T30
dry-run expansions passed. The replay has a 900 second Python/910 second
external budget and forbids model compilation. It retains every state vector,
complete per-leaf parameter metrics and full result trees for any failed pair.
No T30 process is launched.

## R4 T7: strict state failure

The actual T7 initial comparison passes all 20 loss/auxiliary coordinates
(max 2.417004e-7) and all 64,884,481 parameter gradients (max 4.470348e-8,
relative L2 3.438340e-7). The initial state comparison fails 4 of 3,854
coordinates; all seven paired replays fail, with counts 4,4,5,4,4,4,3.
All gradients are finite and meaningful (64,820,737 nonzero parameter
coordinates and 3,833 nonzero state coordinates), all 14 physical hard
counts are zero, and the cumulative process peak is 3,507,481,856 bytes,
below 24 GiB. The native checkpoint, FP32/FP64 boundaries and tolerance
remain unchanged.

Raw state evidence identifies five coordinates across the initial and
repeated failures:

| State index | Failed paired replays | Within-arm diagnosis |
| --- | ---: | --- |
| 33 | 7/7 | Both arms fail their self comparisons; repeat ranges overlap |
| 34 | 5/7 | Original self passes; candidate self fails two repeats |
| 199 | 7/7 | Both self comparisons pass; disjoint ranges beyond strict allowance |
| 509 | 2/7 | Original self passes; candidate self fails one repeat |
| 1024 | 7/7 | Both self comparisons pass; range gap is below largest strict allowance |

At index 199, original repeats lie in
[-0.049182433647,-0.049182237463], candidate repeats in
[-0.049187068832,-0.049186806173]. Their gap is 4.372526e-6, exceeding
the largest unchanged strict allowance 1.491824e-6. This independent
repeated evidence rules out baseline self noise as a complete explanation
of the cross-version failure. Self comparisons remain diagnostics rather
than an added adoption gate. No cross-version failure is waived.

The initial maximum error among failed coordinates is 6.678700e-5.
The whole-state maximum 24.811663 occurs on passing larger coordinates;
it must not be interpreted as the failed-coordinate error.
Exact values and every state vector are retained in
`shared-command-buffer-loops-shape-preserved-T7-state-intervals-20261001.json`
and its source artifacts.

Median synchronized execution is 1.572075 s original versus 1.218308 s R4,
or 1.290376x. Candidate estimated live memory is 3,051,610,373 bytes versus
3,488,289,277 bytes original. Lowering takes 157.449688 s, backend
compilation 2481.478690 s, Python process 2834.822079 s. The launcher exits
1 after 2848.256905 s (0.791182473501 card-hours). The original baseline
model is not compiled. Saved original replay loss/auxiliary is bitwise
equal and all parameter gradients pass; one baseline state coordinate
drifts across processes and is explicitly preserved independently of the
fresh same-process comparison used to judge R4.

No R4 T30, seven-full-parameter adoption replay or complete update is
launched after this failure. The independent stronger V4 daily result
has its own report and subsequent T7 execution; it does not clear R4's gate.

The new shape guard checks exact declared query/FFN/readout shapes and rejects
projected banks whose first dimension collides with7552. It is a local scope
hardening after the daily process started; the measured daily snapshot is
preserved separately and predates this guard. It does not change its graph.

## Artifacts and resources

Source: `scripts/hpc/probe_shared_command_buffer_loops.py`,
`scripts/hpc/probe_shared_chunk_batch_stages.py`,
`scripts/hpc/shared_chunk_batch_execution.py`,
`scripts/hpc/probe_shared_normalized_keys_daily.py` (shared original-loss runner),
`scripts/hpc/run_shared_command_buffer_loops.sh`, and
`scripts/hpc/analyze_shared_command_buffer_runtime.py`.
Executed snapshots and local results are in
`outputs/diagnostics/shared-command-buffer-loops-20261001/`.
Remote diagnostics use
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/`.

Main component JSON:
`shared-command-buffer-loops-chunk-vmap-default-20261001-vmap2.json`;
interval analysis:
`shared-command-buffer-loops-chunk-vmap-interval-analysis-20261001.json`.
Full initial output/parameter/input trees, candidate binaries/HLO, fourteen
raw repeated parameter trees, launcher sidecars and logs are preserved.
The `chunk-vmap.executed.py/.sh` snapshots identify actual successful code.

The initial diagnostic shape guard mistakenly omitted the query singleton
axis and exited1 before component compilation; corrected shape is
`(7542,1,384)`. Its JSON/log/sidecar are retained. Preflight launcher30.743039s,
failed guard5.190450s, successful component46.354981s:82.288470s total,
0.022857908342 card-hours across three one-GPU process launches, paid jobs0.
These include startup and candidate compilation; the old baseline's historical
compilation is not charged again. Additional completed causal-screen/repair
records: screen9.464702s/0.002629083991 card-hours; repair31.901106s/
0.008861418433 card-hours. Total five completed launches123.654279s/
0.034348410766 card-hours, paid0. Daily launcher891.194946s/
0.247554151548 card-hours. Six completed launches total1014.849224s/
0.281902562314 card-hours. Completed R4 T7 adds 2848.256905 s/
0.791182473501 card-hours. The failed/successful saved daily HLO exports add
2.395319 s/0.000665366318 and 22.592005 s/0.006275556816 card-hours.
Nine completed launches total 3888.093452 s/1.080025958949 card-hours, paid 0.
Independent V4 daily and its running T7 have separate records; they are not
included in this R4/component/HLO aggregate.

The saved daily executable was exported with zero model calls or model
compilations. The initial exporter applied its guard before prepared-data
module imports and rejected a tiny import-time constant compilation; its
2.395 second failure is retained. The corrected exporter initializes module
identities before guarding deserialization and succeeds. Its compressed HLO
is 9,111,167 bytes, with decompressed SHA256
`86e959be83a6a54c149b4be2b88f6a3903efb3aaa46b1aa322ba9e19c327c25a`.
Six candidate readout-shaped GEMMs are preserved in the export JSON; the
384-to-256 projection has M=512 (four samples times 128 rows). This is
direct compiled readout-shape evidence;
these shape filters alone do not attribute a GEMM to a unique expert.
The R4 T7 HLO is preserved remotely as 3,418,204,276 bytes and transferred as
a 381,081,571-byte gzip file; its remote original remains intact.

Compiler dump archive `shared-command-buffer-loops-20261001-vmap-compiler-dumps.tar.gz`
was successfully listed with3587 entries. The five completed preflight,
failed-shape, whole-vmap, stage-screen and repair dump directories were then
combined in the verified4681-member archive
`runtime/transfer/shared-command-buffer-loops-20261001/archives/shared-chunk-batch-completed-dumps-20261001.tar.gz`.
Every member name/type and complete regular-file SHA256 was checked against
the closed loose files before removal;4676 files and5 directories were removed.
Archive5352894 bytes/SHA256
`fe5517614571259414f8d8b21584d5c9a0ec6abf6fbd345e17d5de3c9c36ef5e`.
Its compact `.index.json` preserves directory counts, verification and exact
restoration command, and is copied locally with the archive. The current
model compiles have no compiler dump directory and are unaffected.
Local syntax checks and actual target-shell `--dry-run` expansion passed.
Both free GPUs were0MiB/0% with no compute PIDs after this component run.
