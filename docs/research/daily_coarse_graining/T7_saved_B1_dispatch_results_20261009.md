# Saved B1 T7 synchronization comparison, completed2026-10-09

The approved900-second free single-V100 diagnostic completed all three pairs.
Dispatching four saved B1 calls before their dependent merge gave warm medians
**5.035198020→4.916826092s**, **1.024074866x/+2.4075% throughput** (~2.35%
less wall time). All loss, auxiliary and full parameter-gradient cross/self
comparisons passed. Strict initial-state cross comparisons failed1/1/2
coordinates and self repeats failed in both modes, so status is
`numerical_failed`; the route is unadopted. There were no model compilations,
optimizer updates, paid submissions or science/precision/architecture changes.

## Matched work and actual timings

The same fixed saved66,123,713-parameter latent32x384/private-reader64x2 B1
native parameter/state AD executable was used for both modes. TRAIN artifacts
742/275/482/299 retain their original four independent7-day recurrences and
all five argument arrays, physical masks and source-guarded turnover. The four
windows are1991-08-14–20/1970-06-28–07-04/1971-04-16–22/1967-07-22–28
in the365-day noleap model calendar; identities/provenance reuse the accepted
four-window asset and CPU readiness reports. No Teacher data was regenerated.
Precision stays FP32 neural/FP64 physics, and public checkpointed loss/full AD
is unchanged. The existing merge computes the sample mean loss/parameter
gradient, unreduced per-sample auxiliary values and state stack divided by4.

`block_each` waits after every saved B1 call and after merge. `dispatch_four`
queues all four calls then merges their dependencies and waits once. Both
drain completed inputs before timing and include final merge completion; host
transfer, saving and strict comparisons follow timing. Dispatch-only sample
durations are not GPU latency, and no separate CUDA-stream concurrency is
claimed. No B4 executable was loaded or executed in this comparison.

| Pair, order | block_each s | dispatch_four s | Throughput ratio | Strict state cross failures |
|---|---:|---:|---:|---:|
|1,block→dispatch|5.027461188|4.916826092|1.022501324|1|
|2,dispatch→block|5.035198020|5.023113830|1.002405717|1|
|3,block→dispatch|5.055273611|4.912678568|1.029025926|2|

All three directions favor dispatch, but paired gains vary0.2406–2.9026%.
Three fixed-weight observations do not demonstrate a stable changing-update
improvement or predict other windows/loads. Median throughput is0.794408→
0.813533 trajectories/s or5.560854→5.694731 model-days/s. This small measured
saving can be retained as execution evidence toward cumulative speed work;
its importance is not determined by whether it reaches1.5x. It does not justify
integration while the strict state gate fails, and another GPU timing run just
to refine this small gain is not selected. The larger B4 approximately2.555x
result remains separate: this synchronization experiment does not explain or
independently decompose that gain.

## Complete gates and retained state failures

Each of three cross and six self comparisons passed loss1, auxiliary76 and
all66,123,713 parameter coordinates/355 leaves at rtol1e-5/atol1e-6. All
parameter/state/auxiliary values were finite. Every full initial-state gradient
has shape4x3854, with nonzero counts3833/3834/3834/3834. Every mode/warmup/
pair preserved all14 per-window physical/mask hard counts as0. Complete raw
arrays and timings were saved before gates; no state failures were waived.

| Cross | Sample, coordinate | Binding | Dispatch actual | Block reference | Absolute difference | Allowed |
|---|---|---|---:|---:|---:|---:|
|1|1,509|hydrol snowgrain[0,2]|0.22365938127040863|0.22365500032901764|4.380941391e-6|3.236550003e-6|
|2|3,434|hydrol profil_froz_hydro_ns[0,5,3]|1.5362132692243904|1.5361923479940742|2.092123032e-5|1.636192348e-5|
|3|3,434|hydrol profil_froz_hydro_ns[0,5,3]|1.5362058931495994|1.5362515354063362|4.564225674e-5|1.636251535e-5|
|3|3,2621|slowproc_stomate deepC_peat[0,3,1]|0.003680352121591568|0.003678705543279648|1.646578312e-6|1.036787055e-6|

Against each mode's own warmup, block_each state failures are2/3/1 and
dispatch_four2/3/3. Their complete coordinates/values/bindings are in the saved
report, including sample2 coordinate514 in block pair2. Existing B1 state
nondeterminism and B4 failures stay separate evidence; this round neither
identifies the numerical cause nor certifies scientific correctness. Retain the
cheap helper for later complete-workload verification; return the main work to
large graphs/batching and bounded source/derivative analysis of their retained
state failures before selecting integration or a new GPU comparison. No new
resource grant follows from this report.

## Runtime, memory and settled exit

The exact saved B1 sidecar, all four CUDA ABIs and unchanged runtime preflight
passed. B1 load194.613712s; four cold calls19.281523/1.281306/1.249241/1.245733s; the
small reduction helper alone compiled once in3.980968s. Model compile counts
B1=0/B4=0. Setup/load and first-call latency are excluded from warm speed.
Journal reached `completed` at elapsed369.752953s, preserving every pair/report.

Recorded process JAX allocation peak3,071,994,112B (~2.861GiB) and sampled
card-total peak6327MiB pass24GiB. These are cumulative process/card peaks,
not isolated mode or compiler workspace peaks. Known unchanged B1 compiler
live estimate1,192,605,605B is a different quantity. Dispatch keeps four full
return trees until merge; no general overlapping-workspace memory bound was
inferred. Own node wasibc13b02n04, visibleGPU1, OMP4/OpenBLAS1, CPU affinity
0–55; it was not a four-core Slurm allocation. Runtime host load42.08/41.98/
41.96 is an observed snapshot, not evidence for earlier experiments' contention.

The final `__main__` full-report stdout print encountered a NumPy int64 JSON
serialization error **after all scientific arrays, comparisons and the complete
report were saved**. Actual guardian exit1 and the original traceback are
retained. The report says `execution_completed:true`, `numerical_failed`,
expected numerical exit3. A compact scalar stdout receipt replaced that print;
the CPU regression includes representative np.int64 metadata, and all9 tests
pass. The repaired receipt was also serialized from the actual saved report
on CPU. No GPU rerun or ledger rewrite was used to polish the exit status.

The sole independent900-second session13fc2018-0e69-4706-aa47-da7ffee61f69
settled **389.092703195 occupied GPU seconds**, **0.108081306 card-hours/CNY0**,
remaining510.907296805s unused. No old balance was borrowed. Both own guardian
87348 and model87404 were absent, GPU1 0MiB/0% and selected compute-process
query empty at remote2026-10-09T16:45:01+0800. Remote clock ran about8 minutes
behind the client; do not mix timestamps. The scope is closed; its unused
balance does not authorize another run.

## Artifacts and preparation evidence

User's literal approval “批准。” is in local`T7-dispatch-approval-20261009.json`
and remote`runtime/outputs/diagnostics/T7-dispatch-20261009.approval.json`.
The original false template is preserved as historical preparation. New code
is staged under`runtime/transfer/T7-dispatch-20261009`; old paid overlay and
immutablec599236 checkout stayed unchanged. Target Bash syntax/real argv,
Python parser, bootstrap/new-ledger binding and saved-B1/readiness identities
were verified ontest01 CPU before occupation, with no model eval_shape rerun.

Small local evidence under`outputs/diagnostics/shared-latent-reader-20261004/`:
`T7-dispatch-20261009.saved-dispatch.json`, `.saved-dispatch.pair{1,2,3}.strict-comparison.json`,
`.occupation.json`, `.summary.json`, `.launcher.log`, `.final-release.txt`,
`.target-CPU-preflight.json`, `.target-shell-dry-run.txt`,
`.repaired-stdout-CPU-receipt.json` and NVLink version-only trace. Memory/process
CSV use`latent-reader-T7-20261005-dispatch-20261009.memory.csv` because the
unchanged monitor admits that filename family; the ledger/result scope is new.
All12 approximately253MiB full-result pickles remain remote under the new stem;
no bulk gradients were downloaded. Graph tool transport remained closed;
parent Verify evidence and exact-source fallback support this bounded work,
not a fresh/exhaustive graph-coverage claim.
