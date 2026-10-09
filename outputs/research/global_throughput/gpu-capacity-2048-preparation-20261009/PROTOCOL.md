# Fixed2048 fresh-process capacity protocol

2026-10-09 Asia/Shanghai; contract accepted for single job15671732, now terminal and budget closed. New explicit contract
`GPU-fixed2048-repeated-FULL64-B1024-B2048-fresh-process-capacity-v1`.
The next single-card comparison measures fresh B1024×2 chunks and conditional
B2048×1 chunk, both the same ordered2048 lanes from64 genuine FULL64 points
repeated32 times (28672 lane-days), continuous Days2–8 then9–15,48steps/day.
Native dtypes, full outputs/masks and fixed atol1e-8/rtol1e-10 remain unchanged.
Each stage has2051 gates (2048 per-lane FULL64 comparisons plus reset/permutation/
replacement),760 native output leaves and5 synchronized core+5 wrapper samples.
Historical job15662711 B1024 is accepted numeric/schema/compiler/RSS provenance;
its old fixed1024 timing is never doubled or imported as a new2048 measurement.

The two shapes use symmetric fresh GPU child processes in one Slurm parent,
on the same node, selected physical GPU UUID, runtime and allocated resources.
Both retain the default allocator and `PREALLOCATE=false`. Cache clearing/gc
is not treated as GPU pool release; no guard subtracts allocator reservation
from physical free memory. Parent coordinator uses canonical CPU Python and
initializes no JAX/GPU context. Child1 must exit successfully AND have a complete
atomic strict stage before candidate dispatch. A distinct PID/start boundary is
allowed only under this new explicit contract; common job/node/GPU/resources/
runtime/science/input/workload/allocator/allocation-start identity is independently
checked by the collector. Candidate worker start must follow control dispatch exit.
Missing/corrupt/failed/timeout candidate preserves completed control and withholds
a two-shape curve. No retry/requeue/extension/cancellation is authorized.

Coordinator output is `capacity/orchestration.json`; each child owns
`capacity/B1024` or `capacity/B2048` with its report/manifest/full checkpoint.
This prevents report overwrite or manifest races. Atomic checkpoint serialization/
write, input reload/admission, per-child startup and dispatch/exit windows are
recorded separately; warm boundaries remain the inherited full collection.
No executable/timing is inherited between processes. Fresh process lifetime
is an execution change, not a physics or allocator change.

The proposed finite envelope is gnall1node/1task,4 auxiliary Slurm CPUs,
1Tesla V10032GB,7200s/120min/no-requeue/no-exclusive. Both stages need at least
3600s remaining in the SAME allocation window before starting. That threshold is
not a completion guarantee. Candidate admission is time/memory/strict-control
only, without a speedup condition. Check actual pure-helper host stacks, full output
bytes, newest admitted actual compiler temporary analysis and current selected-card
free snapshot before device_put as the whole-stage allocation budget; retain2GiB GPU and8GiB host slack. After compilation, update the unchanged whole-stage formula using actual nonnegative compiler analysis and compare it to that SAME original budget. Record separately labeled post-transfer/postcompile snapshots, require unchanged selected-card UUID/total, and fail closed if actual analysis is missing or the updated requirement exceeds the original budget. Postfree is telemetry, never relabeled untouched budget or credited allocator reservation; this avoids double-counting already placed resident inputs. Compiler
buffers and free snapshots are not measured peak VRAM. A conservative denial is
not proof that hardware cannot run B2048.

[Resource arithmetic](resource-estimate.json) projects B2048 requirement plus
slack29675924096B (27.64GiB), below34359738368B physical by4683814272B.
Historical host2×peak+8GiB is64321847296B versus observed315860516864B available.
These are planning extrapolations; runtime exact-stack values and resource guards
are authoritative. Expanded new run/validation samples and two fresh compiles/
reloads/imports justify proposing120min; potential roughly2×B2048 compile is an
uncertain planning scenario, not a measured runtime promise.

The immutable checkout and single-use run root are
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-fixed2048-v1`
and `.../runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed2048-v1`.
Slurm spools the new2048 wrapper only, resolving helpers from declared CODE_DIR.
The separate [paid proposal](B2048_PAID_PROPOSAL.md) supplies final execution pin,
node basis, concrete command/cwd and CNY4.80 primary/CNY4.96 conservative ceilings.
Those rates assume additive GPU/CPU billing, unverified. Preparation authorizes
no sbatch, paid spend, automation, model lower/compile/run or new scientific inputs.

After separately authorized execution, raw9 terminal sacct parent/child rows are
preserved. Cost counts the parent once, retaining but not summing child rows; queue
time is unbilled. The strict collector runs saved arrays only on test02 canonical
CPU env, ≤4 allowed CPUs,OMP/BLAS1,Python `-B`. It checks exact code SHA, checkpoint
SHA/bytes, original admission/FULL64/native schema/source certificate, every lane,
all2051 gates and runtime/GPU/process origins. Only full accepted stages publish
medians/full ranges and warm cost per common28672 lane-days; startup/compile/first
call/reload/dispatch/checkpointIO/fulljob cost remain separate. No K estimate without
complete setup attribution, no CPU/GPU ratio/Fortran/annual669-point50-year/10×/
trainingAD/general scientific acceptance. Old accepted budgets/results stay closed
and unchanged; B1024 currently has the lowest measured warm unit cost, with no
plateau observed yet, while B2048 is unmeasured.
