# GPU Memory-Lifecycle Validation Proposal

Status: proposal only, prepared 2026-10-11. No GPU job, cluster connection, launcher, collector, or runtime artifact was created or run for this proposal. Submission, deployment, and use of any previous job balance are not authorized here.

## Decision Under Test

Compare two fresh-process B1024 runs over the same 2048 effective lanes, with the same saved inputs, code pin, Slurm parent allocation, selected GPU UUID, canonical runtime, allocator, numerical contract, and sampling order:

1. `phased_resident_control`: run five core samples, then five prepared-wrapper samples while retaining the core device input roots.
2. `phased_release`: use the same five-core-then-five-wrapper order, clear the core device input roots immediately before wrapper sampling, then let each wrapper collection stack, transfer, execute both continuous blocks, copy outputs to host, and synchronize as before.

This isolates the effect of the resident-input ownership policy within the phased schedule. It does not compare either arm causally with historical interleaved timings. `legacy_interleaved` remains the default code behavior and a historical reference only; it is not a third arm.

The implementation names the strategies `legacy_interleaved`, `phased_resident_control`, and `phased_release`. Its contract marks phased runs as `five_core_then_five_wrapper` and distinguishes `retain_core_inputs` from `release_core_inputs`. Keep that strategy and policy in every saved identity so results cannot be mixed. At lifecycle baseline `277aa35793e5b6dd3a1d436a32b4cbc97a305106`, local Python dry-run and stage-fixture checks set `candidate_shapes=[1024]`; this is pre-core-sync execution-logic evidence. The strategy remains restricted to `--fixed-2048 --stage-shape 1024` and does not select a B2048 child. The synchronized core is currently refused by the source-identity guard described below, so the old dry-run result does not certify a new-core execution path.

## Workload And Acceptance

Both future arms must run the same synchronized scientific core from one immutable, accepted code pin. Local synchronization, the bounded CPU production-adapter audit, and its documentation are complete; see the [carbon-core synchronization result](../carbon-core-sync-20261011/CURRENT_RESULT.md). The final GPU execution pin is still pending independent scientific-source admission and the future two-arm launcher/validator preparation; the tested scientific state `ef271cef8551f47f82cccf234949d021e958917e` is not declared ready for execution. The development sequence uses lifecycle baseline `277aa35793e5b6dd3a1d436a32b4cbc97a305106`, followed by the NPP change from `e317fa81b97b8b8c959f2e7dfdfba525bb54dec8` (cherry-pick `c6a88afd41120dfd6434bdf5d442b72b58b34ccd`) and the leaf change from `2a3e303845154576c6f4b5c0b8155106e5bcc70b` (cherry-pick `ef271cef8551f47f82cccf234949d021e958917e`). These identify development lineage, not a final execution pin.

Historical B1024/B2048 numerical acceptance applies only to the old code pins `bd9e4c04af68d77e79b984bfc8aa064977f893d3` and `8aa2e05aa24b65dc4d30a27d4ee857bba6312221`; it does not certify the synchronized scientific core at a newer pin. The frozen FULL64 outputs are comparison goldens only. The new code must execute the original strict tolerances and every acceptance gate against them; do not modify the golden outputs or relax tolerances to obtain a pass. The 199 pre-sync local checks validate execution logic. The new 37 regressions and 354/354 CPU returned-leaf comparisons validate the declared small production-adapter boundary only: 348 records are exact, all 57 stock and 84 discrete/text records are exact, and six leaf_frac records have maximum absolute difference 3.3306690738754696e-16 within the original tolerances. Neither evidence set establishes new-core FULL64 or canonical GPU acceptance.

Each arm uses 64 genuine FULL64 points repeated 32 times, B1024 x 2 chunks, 2048 effective lanes, and 28,672 lane-days. Preserve lane order and origins, both continuous windows (Days 2-8 then 9-15), 48 half-hour steps per day, all native outputs and masks, native dtypes, `atol=1e-8`, `rtol=1e-10`, and exact discrete fields. JAX x64 is enabled, but this does not make every arithmetic operation FP64. This repeated FULL64 capacity load is not 669 heterogeneous points or independent ensemble members.

For each saved arm, require the complete stage to be sealed and the strategy-aware saved-array check to pass before launching the next arm. The acceptance contract is at least 2051 passing gates, 760 native output leaves with expected shapes, dtypes, and masks, and 2048 saved-output/FULL64 comparisons, plus the existing reset, permutation, and replacement isolation checks. Keep all five synchronized core and all five wrapper samples and report median and full range. A partial stage, missing proof, or a mismatch stops the sequence.

The current fixed2048 collector was built for its B1024/B2048 result contract. Its compatibility with strategy-tagged B1024-only arms has not been established. The current source-identity guard actively blocks the new core: at `ef271cef8551f47f82cccf234949d021e958917e`, `global_source_compatibility.source_identity(Path.cwd())` (invoked through ORCJAX `python-B`) exited with status 1 and raised `ValueError: science source changed outside certified execution-only cache repair` at helper line 53. The existing certificate admits only the old science source plus the prior DIFFUCO repair. This is correct version isolation, distinct from numerical failure; the guard remains unchanged. Do not put the carbon-process changes into the old execution-only equivalence claim. Future preparation must establish an independent scientific-source identity for the new core and explicit strict comparison admission against the old frozen FULL64 reference. This proposal establishes neither. Do not infer that an old collector or B1024 checkpoint automatically passes new-core admission. Also prepare the smallest strategy-aware saved-array validator that checks the requirements above and rejects absent or mismatched `measurement_strategy`, `sampling_order`, and `wrapper_resident_input_policy`. The first arm must pass these checks before dispatching the second. Do not claim that the existing collector already supports this workflow.

## Memory Evidence

Persist the following evidence before the corresponding risky operation, and flush each phase record to the report and event log before proceeding. These are validation targets; the current implementation's coverage is narrower, as noted below.

- Before device placement: code and strategy identity, process PID/start epoch, Slurm job/node, selected physical GPU UUID and total/free bytes, workload and logical input/output byte counts, allocator policy, and the original preallocation budget.
- After lowering/compilation and before the updated guard or executable call: actual compiler argument/output/temp/alias fields when available, compiler-analysis scope, updated whole-stage estimate, original free budget, slack, selected UUID, postcompile free-memory snapshot, and the guard decision. Failure to obtain required analysis or to match UUID/total fails closed.
- During each collection: phase start and completion, `sample_index`, boundary name, whether core input roots remain owned, current host RSS, and selected-card physical free/used telemetry where available. Record transfer, initial collection, every core sample, every wrapper sample, D2H/synchronization, permutation/replacement gates, and any exception traceback.
- Keep host RSS and GPU telemetry in separate fields. Record allocator logical/live, reserved, and active values only when the deployed runtime exposes them reliably; retain their original labels and do not treat pool reservation as physical usage or as a peak.
- Do not describe a discrete GPU snapshot as peak VRAM. Unless a validated peak sampler is actually present and its interval/coverage is saved, report peak VRAM as unavailable. A polling maximum, if later added, must be labeled as sampled peak with its sampling interval and any gaps.

The accepted code change currently persists collection phase starts/completions and sample indices with Recorder RSS, plus physical GPU snapshots at post-transfer, postcompile, and post-measurement boundaries. It does not persist per-chunk, first-block, second-block, H2D, or D2H boundary telemetry; it does not take per-sample GPU snapshots and does not include a peak sampler. Treat those as future instrumentation gaps if needed for interpretation. Any added phase evidence must be flushed outside warm timing intervals so measurement boundaries stay unchanged. Do not imply that the current patch records those finer boundaries.

Keep the existing resource guard unchanged: `2R + 3O + scaled_temp + 2 GiB` must fit against the original preallocation physical-free budget, with the postcompile free snapshot retained as telemetry rather than substituted for that original budget. This is a conservative whole-stage estimate, not a measured live set or peak. Record the exact same measurements for both arms. The release arm may reduce live logical references; it is not assumed to return that memory to the driver or increase physical free memory. `gc.collect()`, cache clearing, and `PREALLOCATE=false` do not prove allocator release. The historical B2048 request of 9,161,696,464 bytes is not a peak estimate and must not be reused to infer this comparison's memory.

## Measurements And Stopping

Report for each arm: lower/compile/first-full-collection times; prepared-input and reference reload; host stack and transfer time; all five core and five wrapper samples, medians and ranges; checkpoint serialization and atomic-write time; phase-boundary RSS; physical GPU snapshots; and the complete parent wall time and allocation cost. Keep the warm wrapper boundary identical in both arms, including host stacking, H2D transfer, both-block `run_two`, D2H, and synchronization. Report setup and IO costs separately from warm timings.

Stop without launching the second arm if the first arm fails to seal or pass saved-array acceptance, if its exit is nonzero, if the next-arm resource/time guard fails, or if the selected UUID/runtime identity changes. Stop on any scientific, schema, mask, state-continuity, or fixed-tolerance failure. Preserve the completed control and failure evidence. Do not retry, requeue, extend, alter precision/tolerance, switch allocator, or escalate batch size. If both arms do not complete and pass, publish no ratio and no VRAM benefit claim. No new B2048 stage is part of this proposal.

## Runtime And Child Command

Use the Explore1000 canonical CUDA entry and driver-bind profile from `scripts/hpc/slurm_global_gpu_capacity.sh`: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python`, `scripts/hpc/global_batch_gpu_entry.py`, and `/apps/soft/sif/foundationpose`. The entry must receive the benchmark arguments directly so JAX/CUDA initialization remains canonical. Keep one visible allocated GPU, four allocated Slurm CPUs, `JAX_ENABLE_X64=true`, `ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_gpu_capacity.py`, and `XLA_PYTHON_CLIENT_PREALLOCATE=false`; leave the default allocator unchanged. Preserve the same allocation-window epoch, GPU UUID, driver, affinity/resource metadata, and environment proof for both children.

Proposed but not created execution checkout:

`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-lifecycle-v1`

Its immutable commit SHA is pending final code-owner review and acceptance. Proposed single-use output root:

`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-20261011-<new-job>`

The exact child command shape, with allocation-specific values passed by the future parent launcher, is:

```bash
"$SLURM_BIN/srun" --ntasks=1 --cpus-per-task=4 --cpu-bind=cores \
  /usr/bin/time -v singularity exec --nv --bind "$BINDS" --pwd "$CODE_DIR" \
  /apps/soft/sif/foundationpose \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python \
  "$CODE_DIR/scripts/hpc/global_batch_gpu_entry.py" \
  --prepared-input /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl \
  --full-reference-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-round4/model \
  --memory-baseline-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/global-throughput-v2-2b14506/gpu-b64/candidate-first \
  --config "$CODE_DIR/configs/orchidee_man_250919.yaml" \
  --asset-root /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/round4-inputs/selected64-assets-v2 \
  --output-dir "$ARM_OUTPUT/capacity/B1024" \
  --warm-repeats 5 --budget-seconds 7200 \
  --fixed-2048 \
  --b1024-evidence-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1/capacity \
  --stage-shape 1024 \
  --measurement-strategy "$MEASUREMENT_STRATEGY"
```

Set `MEASUREMENT_STRATEGY` and `ARM_OUTPUT` once per child to the matching pair:

| Child order | Strategy | Output subdirectory |
| --- | --- | --- |
| 1 | `phased_resident_control` | `phased_resident_control` |
| 2 | `phased_release` | `phased_release` |

The exact environment exports, CUDA/NVML host driver bind paths, selected-GPU identity receipt, and `SLURM_*` propagation must follow the accepted GPU launcher. Both children run serially inside the same parent allocation; the first must exit and validate before the second begins. This command is a recipe for a future launcher, not a ready-to-submit script.

## Proposed Allocation And Cost

| Field | Proposal |
| --- | --- |
| Partition | `gnall` |
| Resources | 1 node, 1 task, 4 Slurm CPUs, 1 V100 32 GB GPU |
| Wall limit | 120 minutes, finite; `--no-requeue`; no `--exclusive` |
| Work directory | Proposed immutable checkout above, under `/WORK/liwei_work/jcwu/` |
| Output | New single-use arm directories under the proposed runtime output root |
| Node selection | Unknown until a future approved submission; inspect current scheduler occupancy and CPU load, prefer and select an idle node where practical, and record actual node/load/processes at job start |
| Primary worst-case estimate | CPU: 4 CPUs x 2 h x CNY 0.05 = CNY 0.40; GPU: 1 GPU x 2 h x CNY 2.20 = CNY 4.40; additive total CNY 4.80 |
| Conservative estimate | CPU: 4 CPUs x 2 h x CNY 0.07 = CNY 0.56; GPU CNY 4.40; additive total CNY 4.96 |

This is a proposed cap, not an approved budget or verified bill. It assumes additive CPU/GPU rates and excludes no separate billing policy. The historical B1024 child wall of about 2,287 seconds per run gives a planning total of about 4,574 seconds for two children, leaving about 2,626 seconds inside a 7,200-second parent limit for startup and orchestration; this extrapolation is not a completion guarantee. Require at least 3,600 seconds remaining before dispatching arm two. Do not reuse the closed job15671732 or earlier job budgets.

Submission-layer command is not yet available: neither existing launcher is suitable. `slurm_global_gpu_capacity_2048.sh` invokes the coordinator that can dispatch B2048, and `slurm_global_gpu_capacity.sh` routes `FIXED_2048=1` through that same coordinator. Prepare a dedicated serial two-arm parent launcher, including phase/receipt collection, identity checks, saved-array acceptance between children, failure stop behavior, and unique output paths. Then exercise the actual target Bash argument expansion with `--dry-run` and review the exact resolved command before any future approval. The submission shape will be:

```text
sbatch --partition=gnall --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
  --time=02:00:00 --no-requeue [--nodelist=<node selected from fresh occupancy/load evidence>] \
  <dedicated two-arm launcher prepared and dry-run validated>
```

The bracketed nodelist is selected from live evidence; the launcher placeholder is intentionally not an invented path. The child command above is source-checked against the current CLI, but this round did not create or validate a two-arm submission launcher or a compatible saved-array collector. Before any submission, prepare those pieces, exercise the actual target Bash argument expansion with `--dry-run`, and state exact resources, finite wall time, worst-case cost, working directory, commit SHA, full expanded command, and output root in the approved block. Do not route this proposal through the historical B2048 launcher or coordinator.

## Readiness And Limits

The strategy implementation, local execution-logic validation, sequential core cherry-picks, and bounded CPU audit have been reviewed and accepted locally. Ordinary Git commit/push receipts identify the delivered development state; they do not declare a GPU execution pin. Historical B1024/B2048 numerical acceptance does not transfer to this core, so its strict golden comparisons and full gates remain outstanding. Independent new-core scientific-source identity and old-reference strict comparison admission, the dedicated two-arm launcher, and strategy-aware saved-array validation still need preparation; the launcher must pass target-shell dry-run argument expansion. Preserve the old DIFFUCO-only execution certificate. Therefore, this proposal is not ready to submit. At submission planning time, inspect current scheduler node occupancy/CPU load and select an appropriate node. Keep the current shared source checkouts and running jobs unchanged.

The saved evidence supports this bounded comparison proposal only. It does not establish that clearing Python roots releases physical VRAM, lower peak memory, or make B2048 viable. Any future canonical GPU run must report observed values with their scope and retain all failure evidence. This proposal records the plan; it is not evidence of canonical numeric or GPU validation. The code owner reported 199 local checks passed in 22.70 seconds across five related files, plus Ruff and diff checks; this proposal task did not rerun them. Those local fake-logic checks are not canonical GPU or numerical validation. This subtask ran no tests or model execution.
