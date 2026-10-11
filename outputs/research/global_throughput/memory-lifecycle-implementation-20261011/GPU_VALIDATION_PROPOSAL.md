# GPU Memory-Lifecycle Validation Proposal

Updated 2026-10-11. The dedicated local preparation is complete; see the
[current result](../gpu-lifecycle-preparation-20261011/CURRENT_RESULT.md) and its
verification records. This proposal does not authorize cluster access, deployment,
GPU execution, submission, or reuse of any closed budget. No server checkout or
runtime artifact has been created by this preparation.

## Fixed comparison

Contract: `GPU-B1024-phased-memory-lifecycle-new-science-v1`.
Process strategy: `sequential_fresh_process_per_strategy`.
Only stage: `capacity_B1024_lifecycle_v1`; B1024 x 2 chunks.

Within one future allocation, run two serial fresh processes from one immutable
code SHA with identical science, inputs, runtime/compiler environment, node,
selected GPU UUID/total, allocator, resource allocation and sampling order:

| Order | Strategy | Core roots during wrapper samples |
| --- | --- | --- |
| 1 | `phased_resident_control` | Retained |
| 2 | `phased_release` | Cleared before wrapper sampling |

Both arms run five core samples followed by five wrapper samples.
`common_identity` excludes only `measurement_strategy` and
`wrapper_resident_input_policy`; it retains and matches
`sampling_order=five_core_then_five_wrapper`. Historical interleaved samples are
separate evidence. The dedicated runner has no fixed2048 or B2048 option.

Each arm repeats 64 genuine FULL64 inputs 32 times: 2048 effective lanes,
28672 lane-days, Days2-8 followed continuously by Days9-15, 48 half-hour steps/day.
Preserve full physics, all native outputs/masks/dtypes, `atol=1e-8, rtol=1e-10`,
exact discrete fields, and only the declared bare-soil roughheight NaN mask.
JAX x64 does not imply all arithmetic is FP64. This capacity load does not
establish 669 heterogeneous points or independent ensemble-member acceptance.

## Scientific version and strict admission

The prior [carbon synchronization](../carbon-core-sync-20261011/CURRENT_RESULT.md)
is a prerequisite: NPP and leaf changes were independently cherry-picked, with
37 lightweight checks and 354 bounded CPU returned-leaf comparisons accepted.
The prior 199 lifecycle checks concern execution logic. Those checks are reused,
not rerun or represented as new complete-model/GPU acceptance.

The new scientific content pin is
`1fdacde59d86f696f384b56d69292c8f6fa8d30b`, canonical scientific SHA256
`8c8b1dfd0b207e96bbb81c2d3eb85ae6edccb9ed3c6b77fb470dabc58c5fa77d`.
`global_lifecycle_science.py` admits exact tracked scientific blobs or their exact
declared LF-to-CRLF representation, rejects extra executable scientific files and
path escapes, and requires the expected HEAD plus a clean checkout for execution.
The old `global_source_compatibility.py` and its DIFFUCO-only execution-equivalence
claim remain unchanged. The carbon fixes are scientific changes.

The separate old-reference admission is
`new-science-versus-frozen-old-FULL64-strict-comparison-v1`.
Frozen `B4_group_references` SHA256 is
`a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6`;
old scientific SHA256 is
`11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`.
This permits a strict comparison; it does not claim source equivalence, change
old goldens, or transfer historical GPU acceptance to the new core.

The runner independently checks the actual initial 760-leaf native schema and
2048 FULL64 comparisons before any warm sample. A finite tolerance mismatch
raises immediately and preserves `failed_initial_arrays.pkl`; the new path does
not inherit the old cross-backend exploratory-continuation behavior. It saves
actual initial/reset/permutation/replacement arrays and both initial chunks'
first-output/second-input state handoffs. Warm samples use the unchanged original
executable and `benchmark_global_batch.run_two`.

After each child exits zero, a separate CPU/NumPy validator reopens its sealed
arrays through the restricted NumPy/builtin unpickler. It recomputes all 2051
gates, schema/dtype/masks, 2048 reference comparisons, reset/permutation/replacement,
exact actual state handoffs, source/runtime/allocation identity, phase order,
five-sample timing summaries and memory evidence. Missing, partial or inconsistent
proof fails closed. The first child must exit, seal and independently pass before
the second dispatch. Require at least 3600 seconds remaining immediately before
that dispatch. Only after both independently pass may the parent publish core
and wrapper median throughput ratios, release/control, for this same job.

Any child, array, source/runtime/UUID, memory or time failure stops the sequence.
Preserve successful and failed artifacts. No retry, requeue, extension, precision
or tolerance change, allocator switch, new shape, or scale-up is part of this plan.
If either arm fails, publish no ratio or memory-benefit claim.

The independent validator decodes the exact four-field inert `ModeloutResult`
DTO without JAX/model import. Only after runtime admission does the GPU runner
restore the exact production NamedTuple class for JAX treedef compatibility,
preserving every leaf object. The existing local frozen-reference format and
container bridge checks both passed; neither executes a model or proves new-core
FULL64 values.

## Memory and measurement scope

Retain the guard `2R + 3O + scaled_temp + 2 GiB` against the original preallocation
physical-free budget. Record the actual compiled executable analysis and updated
guard before execution; postcompile free memory remains telemetry. Old B1024
metadata supplies frozen compiler/RSS bounds; its checkpoint is only streamed for
hash/bytes and is not loaded for historical arrays or timings.

Persist phase starts/completions, sample indices, failure tracebacks, code/process
origin, allocation/node/CPU information, selected-card identity, logical bytes,
compiler analysis, guard decision, and collection-boundary evidence. Core-root
ownership immediately before/after the wrapper transition and each wrapper sample
start/end include host RSS and same-card physical snapshots, outside warm timers.
The wrapper timer continues to include host stacking, H2D, both continuous blocks,
D2H and synchronization. Save all five core/wrapper samples, medians and full ranges.
Setup, lowering/compilation, initial collection, reference/prepared reload,
validation materialization, checkpoint IO and parent wall/cost remain separate.

Discrete physical snapshots are not peak VRAM. The implementation has no validated
peak sampler, full per-chunk/per-block/transfer memory trace, or reliable allocator
live/reserved sampler. Logical root release does not prove physical driver release.
Report peak VRAM unavailable and observed snapshot scopes explicitly. Do not reuse
the historical B2048 9,161,696,464-byte request as a peak estimate.

## Candidate checkout and canonical command

Proposed future code cwd, not deployed:

`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-lifecycle-v1`

The final full executable candidate SHA and locally resolved static argv are in
`../gpu-lifecycle-preparation-20261011/EXECUTION_CANDIDATE.json`, generated after the
preparation commit/push. That local receipt avoids a self-referential commit.
Before submission, copy the exact approved SHA into `EXPECTED_CODE_SHA`; both
launcher and source admission enforce it. Server Git 1.8.3.1 commands use `cwd`
or `cd CODE_DIR && git`, `--porcelain`, and no `git -C`/worktree requirement.

Dedicated parent: `scripts/hpc/slurm_global_gpu_lifecycle.sh`.
CPU parent: canonical `.venvs/orcjax_cpu/bin/python -B` and
`scripts/hpc/global_gpu_lifecycle_coordinator.py`.
CUDA child: canonical `.venvs/orcjax_gpu/bin/python`,
`scripts/hpc/global_batch_gpu_entry.py`, and `/apps/soft/sif/foundationpose`.
Set both host and Singularity `ORCHJAX_GLOBAL_BATCH_ENTRY` selectors to
`benchmark_global_gpu_lifecycle.py`; the runner filename is not an extra argument.
Set x64 true, PREALLOCATE false, leave the allocator default, use one allocated
visible GPU, four requested Slurm CPUs and a common allocation-window epoch.
Propagate Slurm/TRES, UUID/driver, affinity/thread and project-root metadata.

The child argv, substituted by the coordinator for each arm, is:

```bash
/rmprog/slurm/v22.05.7/bin/srun --ntasks=1 --cpus-per-task=4 --cpu-bind=cores \
  /usr/bin/time -v singularity exec --nv --bind "$BINDS" --pwd "$CODE_DIR" \
  /apps/soft/sif/foundationpose \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python \
  "$CODE_DIR/scripts/hpc/global_batch_gpu_entry.py" \
  --prepared-input /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl \
  --full-reference-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-round4/model \
  --memory-baseline-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/global-throughput-v2-2b14506/gpu-b64/candidate-first \
  --b1024-evidence-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1/capacity \
  --config "$CODE_DIR/configs/orchidee_man_250919.yaml" \
  --asset-root /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/round4-inputs/selected64-assets-v2 \
  --output-dir "$RUN_ROOT/$MEASUREMENT_STRATEGY/capacity/B1024" \
  --warm-repeats 5 --budget-seconds 7200 --stage-shape 1024 \
  --measurement-strategy "$MEASUREMENT_STRATEGY" --expected-code-sha "$EXPECTED_CODE_SHA"
```

`BINDS` is resolved on the actual approved node from canonical host libcuda/NVML
paths and `/WORK:/WORK`. The local Bash dry-run uses an explicit unresolved-driver
placeholder. Passing local expansion into the actual coordinator dry-run validates
argument handling and static planning, not those host drivers, CUDA or server Bash.
Local paths with spaces, code pin rejection and the entry env protocol are checked.
Windows fixtures map local project paths and the mocked container Python token;
the future POSIX output plan is separately checked without local-drive rewriting.

Future single-use output root:

`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-20261011-${SLURM_JOB_ID}`

The launcher accepts a separately specified unique `RUN_ROOT` under the same
`gpu-lifecycle-*` prefix. It rejects reused roots and overlapping source/output
paths. Each arm writes `<root>/<strategy>/capacity/B1024`; the parent keeps journal,
independent acceptance receipts, child logs, allocation metadata and terminal proof.

## Proposed submission and cost: unapproved

| Field | Proposed value |
| --- | --- |
| Partition / tasks | `gnall`, 1 node, 1 task |
| Resources | 1 V10032GB, 4 Slurm CPUs |
| Wall limit | 120 minutes, `--no-requeue`, no `--exclusive` |
| Work directory | Immutable lifecycle candidate directory above |
| Command | Dedicated launcher below, exact SHA supplied from candidate receipt |
| Output | New single-use root and two arm directories above |
| Slurm stdout / stderr | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_gpu_lifecycle_%j.txt` for both |
| Node | Pending fresh scheduler occupancy/CPU-load inspection; prefer an idle node and select explicitly where practical |
| Primary worst case | 4 CPUs x 2 h x CNY0.05 + 1 GPU x 2 h x CNY2.20 = CNY4.80 |
| Conservative worst case | 4 CPUs x 2 h x CNY0.07 + 1 GPU x 2 h x CNY2.20 = CNY4.96 |

The cost assumes additive rates; it is not a verified bill or approved allocation.
A previous B1024 child took about 2287 seconds: two such walls would be about
4574 seconds, leaving 2626 seconds of 7200 for overhead. New-science compilation,
extra array validation and runtime variation can consume that margin; completion
is not guaranteed. The second-arm 3600-second remaining-time check stays binding.

Reviewable future submission shape, after separate explicit authorization:

```bash
export CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-lifecycle-v1
export EXPECTED_CODE_SHA=<exact-40-character-candidate-SHA>
[[ -d /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs && -w /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs ]] || {
  echo "shared Slurm log directory must exist and be writable before submission" >&2
  exit 2
}
/rmprog/slurm/v22.05.7/bin/sbatch --partition=gnall --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --gres=gpu:1 --time=02:00:00 --no-requeue \
  --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_gpu_lifecycle_%j.txt \
  --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_gpu_lifecycle_%j.txt \
  --chdir="$CODE_DIR" --nodelist=<freshly-reviewed-node> \
  "$CODE_DIR/scripts/hpc/slurm_global_gpu_lifecycle.sh"
```

The node, driver bind paths and new job ID are future allocation values. No
connection/transfer or deployment is authorized by this document. Approval must
name the immutable SHA, cwd, complete command, output, resources, finite walltime
and worst-case cost. Closed job15671732 and earlier balances remain closed.
The launcher uses the same absolute shared Slurm log paths. Verify that parent
directory on the target before submission: Slurm opens logs before script startup,
so the script's later `mkdir` cannot create it in time. This target check has not
been performed locally.

## Remaining acceptance

Complete local source/runner/validator and launcher/coordinator checks are recorded
with actual commands and results in the current preparation report. The passed old
FULL64 format check on the existing local frozen asset proves only
unpickler/schema/comparator compatibility and immutable-reference admission.
Local source dry-run, fake arrays/processes, and Bash expansion do not certify
new-core FULL64 values, canonical JAX0.4.38/CUDA execution, allocator recovery,
physical memory reduction or throughput improvement. Those remain the future
approved two-arm diagnostic's outcomes. No B1280, B1536, B2048, training,
architecture change or main merge is part of this stage.
