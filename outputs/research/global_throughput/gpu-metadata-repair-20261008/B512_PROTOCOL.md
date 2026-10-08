B512-only protocol is implemented as `--continue-512` with explicit sealed B128
and B256 directories, five core and five wrapper collections, and a 2700-second
window. The two continuation flags are mutually exclusive. B128/B256 remain
historical origins and never become destination stages, measurements or compiles.
Only B512 can appear in `completed_shapes`; comparison retains B128/B256/B512.

The importer preserves original identity, timing, provenance and execution origin,
requires the original sealed SHA and byte length, 515 strict passed gates, 760
full output leaves with 512 lanes, and checks complete tree/shape/native dtypes
against the unchanged FULL64 reference. It rejects changed runtime/compiler,
science/inputs/reference, workload, gates, original code origin, timings, output
schema or B256 compiler memory. The existing narrow loadavg tuple/list exception
preserves all three telemetry values; no general coercion was added.

B512 device admission uses imported B256 actual compiler temporary bytes
(1,151,609,352 bytes), scaled to B512, plus existing resident/full-output overlap
and 2 GiB slack. Old B64 actual RSS remains the host estimate origin, with the
existing 8 GiB host slack. The preset 30-minute remaining-window gate remains.
These estimates are planning guards; they cannot guarantee completion or peak
memory. A denied B512-only run reports failure, with no successful B512 claim.

`scripts/hpc/slurm_global_gpu_capacity_512.sh` declares one GPU, four auxiliary
CPUs and 45 minutes. It resolves the shared launcher through declared `CODE_DIR`
even after Slurm copies the wrapper to its spool. After separate explicit approval,
job15657822 was submitted exactly once from gln01 through192.168.11.2. Actual allocated GPU identity admission and selected-card
diagnostics remain shared with the existing launcher.

Local ORCJAX validation: 56 capacity tests passed. Tests exercise sealed-B256
rejections, native schema, loadavg drift, no baseline measurements/compiles,
B256 memory reuse, resource denial, complete two-block collections, command-line
protocol rejection, and actual Bash/Python dry-run expansion from both checkout
and copied Slurm spool. Ruff passes for the changed benchmark and tests.

`b512-baseline-admission.json` records successful local admission of both actual
sealed historical checkpoints against the actual FULL64 reference, with no
compilation or model execution. B128's locally flattened evidence filenames are
presented through two unchanged hard links in `sealed-b128-origin-view`; no
second checkpoint data copy was created. The current GPU runtime and revised
B512 executable still require their separately planned bounded validation.

The immutable metadata-v1 checkout is now deployed at accepted SHA
`ad6c0e1faa7e22b590627844e9cb0ad492e9deb3`, clean before/after all 9 declared
checkout interface checks. Actual CLI help/B512 dry-run, both Bash syntax checks
and direct/copied-spool wrapper expansion pass; direct and spool commands agree.
JAX was not imported and no model preparation, lowering, compilation, invocation
or GPU initialization occurred. See [deployment](deployment-receipt.json) and
[interface receipt](deployment-interface-receipt.json). Its actual Linux source
SHA is `b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8`,
with active exact certificate and canonical baseline `11b6db89...`.
Total conservative canonical wall is 367.0582004s within the 600s cap. These
interfaces supersede the earlier streamed old-ROOT preparation; real CPU
before/after lowering, local 103 tests, and unresolved GPU execution retain
their separate evidence scopes. Paid approval and submission are recorded in
[submission receipt](submission-receipt.txt); the [initial status](job-15657822-initial-status.txt)
at2026-10-08 12:51:09UTC is PENDING (Priority), with no B512 timing. Queue time is unbilled.

The exact source compatibility certificate distinguishes the full actual source
hash from the normalized baseline science identity. It admits only the approved
byte-pinned immutable PFT-mask execution helper repair; every other scientific
source file remains frozen to the baseline digest. No general AST normalization
or relaxed importer equality is used. New capacity manifests retain
`actual_science_source_sha256` and `execution_source_compatibility`; the report and
dry-run proposal also expose the complete execution source identity. Historical
baseline source identities and execution origins remain their original values.
The aggregate certificate recognizes exactly the canonical LF baseline hash
`11b6db89...` and audited Windows checkout hash `4151db25...`; it applies no
generic line-ending normalization. DIFFUCO itself retains the exact approved
module hash. Unapproved module line endings, helper/body/formula changes and an
altered aggregate from any other source file fail admission. Reverification with
the current production `scientific_source_digest()` API admits both sealed
origins and confirms its canonical science SHA matches their original identity.

The separately approved and submitted allocation is one node, one task, one V100, four auxiliary CPUs and 45 minutes
in `gnall`, with the immutable declared checkout and a new output directory.
Worst-case GPU card-hours are 0.75 and auxiliary CPU-hours are 3.0. At the current
planning prices of 2.2 CNY/GPU-hour and 0.05/0.07 CNY/CPU-hour, the full-window
cost envelopes are 1.80 CNY primary and 1.86 CNY conservative. These are full-window
estimates, not scheduler billing or consumed wall time. Job15657822 has no requeue;
no automatic retry is authorized. ReqNodeList is `ibc13b03n02`. The actual
2026-10-08 12:48:06UTC [submission snapshot](submission-node-snapshot.txt) records
CPUAlloc112/112, CPULoad22.74 and1/8 GPUs allocated; no idle eligible gnall node
was observed. The earlier11:21 proposal snapshot is historical. Neither snapshot
reserves the node or implies whole-node exclusivity. Actual node/CPU affinity/load
and visible GPU processes remain startup evidence to collect.

Actual capacity destination is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-capacity-512-15657822/capacity`;
actual log is `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_gpu_capacity_512_15657822.txt`.
