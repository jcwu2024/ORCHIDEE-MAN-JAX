# Fixed1024 capacity protocol: preparation only

The explicit `--fixed-1024 --b512-evidence-dir ...` mode declares
`GPU-fixed1024-repeated-FULL64-B512-B1024-capacity-v1`. It measures a new B512
collection (two chunks) followed by a resource-admitted B1024 collection (one
chunk) in the same process and job. Both perform the same14336 lane-days:
ordered64 original FULL64 points repeated16 times, continuous Days2–8 then9–15,
48 half-hour steps/day, full native outputs and unchanged physical masks and
`atol=1e-8`, `rtol=1e-10`. This is1024 lanes, not1024 independent points or
independent ensemble-member inputs. No B64/B128/B256 executable is compiled.

Historical fixed512 constants, CLI defaults, identity and stage names remain
unchanged. The new names are `capacity_B512_fixed1024_v1` and
`capacity_B1024_fixed1024_v1`. Each stage saves a full checkpoint, SHA256,
manifest, report and IO evidence atomically before proceeding. It performs a
full warmup, five synchronized complete core collections and five synchronized
complete prepared-wrapper collections. There are1024 per-lane comparisons
against original FULL64 and three repeat/wrapper gates,1027 gates per stage,
with760 native output leaves. Missing or tolerance-failed evidence cannot
produce an accepted performance stage.

Old B512 job15657822, code `ad6c0e1faa7e22b590627844e9cb0ad492e9deb3`, remains
the sealed numerical/schema/compiler-memory/RSS origin only: checkpoint
220430830 bytes, SHA256
`1c44d84ad73f3a04f6d21522a0b35011d157e3a1dbc5b155dc93833a44912a2a`.
Its timing is excluded from the new curve; old timing multiplied by two is
never a new measurement. The new importer verifies exact source identity,
native schema, original code/job, complete gates, original five-sample timing
consistency and report/manifest/envelope equality, then returns an origin
without timing. [Actual saved-origin admission](historical-b512-admission.json)
exercises this on the unchanged saved checkpoint and original FULL64 schema.
The existing independently admitted
`gpu-sweetspot-preparation-20261006/target-input-admission.json` retains its
original identity; preparation does not repeat the FULL64 input audit.

Both shapes require at least2700 seconds remaining in the proposed5400-second
window. This45-minute budget guard is a planning threshold, not a completion
guarantee. B1024 additionally requires a newly sealed B512 stage with
`strict_numerical_passed=True` and all1027 gates passed. A numerical failure
retains B512 evidence but skips B1024 explicitly. Neither resource decision
uses performance improvement as a condition.

Device admission uses the maximum historical/current actual compiler temporary
buffer analysis scaled by candidate shape, plus twice resident inputs, three
full-output buffers and2GiB free-card slack. Outer admission uses an explicitly
conservative raw repeated-lane byte upper bound (shared static leaves may be
counted repeatedly); inner preflight uses actual pure-helper host-stack bytes.
Both are planning estimates, not measured VRAM peaks. Host admission uses the
larger of historical B512 peak×2 or current B512 process peak×2, plus8GiB;
process RSS is never interpreted as GPU memory. The initial historical host
planning requirement is47583821824 bytes. The actual stack and selected-card
snapshot at execution determine admission; [resource estimates](resource-estimate.json)
are projections, not measurements or reservations.

The proposed wrapper is `scripts/hpc/slurm_global_gpu_capacity_1024.sh`:
gnall,1 node,1 task,4 auxiliary Slurm CPUs,1 V100,90-minute hard wall, no requeue.
It resolves the shared launcher from declared `CODE_DIR`, including a Slurm
spooled copy. Deployed clean immutable checkout is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-fixed1024-v1`;
single-use planned run root is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1`.
Capacity evidence is its `/capacity` child; scheduler log is
`runtime/logs/global_gpu_capacity_1024_%j.txt`. The exact invocation, working
directory, node basis, resource/cost decision and pinned execution SHA
belong to [the separate paid proposal](B1024_PAID_PROPOSAL.md). Preparation
has deployed and verified the pinned checkout; it submits no job and old budgets remain closed.

[Target Bash interface evidence](canonical-launcher-interfaces.json) validates
actual runtime argument expansion and identical direct/spooled commands in a
diagnostic fixture with only the PROJECT literal relocated under preparation.
Local Python dry-run also verifies the new contract without JAX or input reads.
[Actual deployed interface evidence](deployment-interface-receipt.json) now verifies the exact clean target on test02:10 subprocess checks plus collector code-pin admission, with direct/spooled commands equal and no PROJECT relocation. Existing source identity/certificate and four committed interface hashes pass; no JAX import, model work or GPU initialization. The fixture remains historical evidence.

The new `collect_fixed1024_result.py` requires an explicit reviewed
`--expected-code-sha`, exact Linux full-source identity and certificate,
approved job/node/allocation envelope and observed startup GPU UUID/global ID/driver. It cannot
silently reuse the ad6/fixed512 collector contract. Terminal accounting uses
one complete raw9 parent row plus retained child rows; cost counts the parent
once. A complete strict B512 stage can survive a later B1024 failure or parent
TIMEOUT; that parent failure/cost remains explicit and B1024 stays withheld.
No terminal parent means no final cost. Whole-job cost, warm medians/ranges,
startup, checkpoint IO and CPU saved-array collection are separate; absent full
phase setup/validation evidence means no K estimate. No CPU ratio, Fortran,
annual/669-point/50-year, general science,10×, training or AD acceptance follows.

[Canonical successive-shape evidence](canonical-helper-probe.json) covers only
the actual DIFFUCO insertion helper on CPU JAX0.4.38:512/1024 each public/internal
10/10, five native outputs exactly equal independent NumPy, immutable host mask
and unchanged first signature. It performs no physical-model preparation,
lowering, compilation, invocation or GPU initialization. Real fixed1024 GPU
capacity, timing and memory acceptance remain unmeasured.

Execution code is pinned to `bd9e4c04af68d77e79b984bfc8aa064977f893d3` in
[final-code-pin.json](final-code-pin.json). The verified incremental bundle
requires the preserved accepted `ad6c0e1` Git history. Subsequent documentation
commits are not the execution revision. Captured shell raw SHA values bind to
the committed LF shell blobs through the fixture's explicit CRLF-to-LF step;
the only semantic fixture edit is PROJECT relocation. This shell handling does
not alter the exact scientific-source certificate. [Deployment](deployment-receipt.json) and actual target entry/collector interfaces pass; paid approval and real fixed1024 results remain pending.

After separately approved execution, collect the raw9 sacct record on a login
endpoint and validate saved arrays on `test02`. The future numeric JOB_ID,
GPU_UUID, GLOBAL_GPU_ID and DRIVER_VERSION come from that job's actual startup
and verified gpu-identity.json; they are not new resource approval requests.
COLLECTOR_CPU_SET contains four currently allowed CPU IDs on test02; select it
from that process's actual affinity before collection. The approved node must
match terminal/startup evidence. These are templates,
not commands executed during preparation:

```bash
# Login endpoint: scheduler query only; CODE_DIR and RUN_ROOT are proposal paths.
/rmprog/slurm/v22.05.7/bin/sacct -n -P -j "$JOB_ID" --format=JobIDRaw,JobName%64,Partition%64,State%64,ElapsedRaw,AllocCPUS,AllocTRES%512,NodeList%128,ExitCode > "$RUN_ROOT/sacct-raw9.txt"

# test02: canonical CPU Python, affinity limited to four available CPUs,
# OMP/OPENBLAS/MKL=1; no model invocation or GPU initialization.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c "$COLLECTOR_CPU_SET" /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python -B "$CODE_DIR/outputs/research/global_throughput/gpu-capacity-1024-preparation-20261008/collect_fixed1024_result.py" \
  --code-dir "$CODE_DIR" --expected-code-sha bd9e4c04af68d77e79b984bfc8aa064977f893d3 \
  --stage-dir "$RUN_ROOT/capacity" --sacct-file "$RUN_ROOT/sacct-raw9.txt" \
  --admission /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/target-input-admission.json \
  --b512-evidence-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-capacity-512-15657822/capacity \
  --job-id "$JOB_ID" --job-name orchjax-gpu-cap1024 --node ibc13b03n05 \
  --gpu-uuid "$GPU_UUID" --slurm-global-gpu-id "$GLOBAL_GPU_ID" --driver-version "$DRIVER_VERSION" \
  --approved-nodes 1 --approved-tasks 1 --approved-cpus 4 --approved-gpus 1 --approved-wall-seconds 5400 \
  --output "$RUN_ROOT/collected-fixed1024.json"
```
