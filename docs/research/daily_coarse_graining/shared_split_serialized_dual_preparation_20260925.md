# Saved-Executable Dual-Process Throughput Diagnostic

Status: compact replay80665 now PASSED/exit0. Full results in
shared_split_dual_compact_results_20260925.md:14.851->11.193s daily32 and
20.391->11.336s daily64, with full changing-update/reference parity.
The preparation and failed-attempt chronology below is historical.

Latest: paid15386166 also timed out124 after28min6s on idle ibc13b03n04,
with zero updates; daily/T7 binaries loaded, T30 incomplete. See
`shared_split_dual_idle_node_proposal_20260925.md` for measured stage times.
Next prepare compact exact replay arguments once on free CPU and reuse the
declared fresh update0 checkpoint, avoiding full bootstrap on each GPU replay.

## Compact preparation follow-up

Deployed preparation98568 PASSED/exit0:201.4025s measured preparation,
130.37CPU-seconds,one~50MiB file, artifacts742/275. Both exact method/source
validation and actual window preparation completed. Free compact dual80665
started16:13:54CST on gln01,30min bound, logs/results
`shared-split-dual-compact-20260925.{log,json}`. The initial stages bypassed
full runtime bootstrap and loaded daily before proceeding toT7; no complete
update comparison at this snapshot. GPU0/1 were empty at dispatch; host
load1.92/11.09/9.72, so report the actual environment with eventual timings.

Implemented in the same diagnostic entry point; production learning and
gradient math are unchanged. `--prepare-input-bundle --input-bundle PATH`
runs on free CPU, uses the original declared source/method, validates the two
existing real TRAIN artifacts742/275, prepares the exact32-row/T7/T30 argument
trees, and writes one consolidated pickle. It references the original profile's
fresh update0 checkpoint; it does not regenerate Teacher data or initialize
another model. A completed matching bundle is reused without rebuilding it.

GPU `--input-bundle PATH` uses those saved pytrees and update0 through the
existing probe's comparison/timing hooks. It skips the coherent-index scan,
canonical-example and retained-template/head reconstruction and fresh
initializer. Two changing updates still compare all gradients, parameters,
AdamW state and original reference losses/checkpoint;32/64 fixed-weight
complete timings remain unchanged. The returned diagnostic checkpoint is
not formal training. CPU-prepared inputs still require deployed GPU parity;
local tests cannot certify that result.

Ten focused tests pass, including the saved native reverse, changing inputs,
private worker protocol, and mocked two-payload compact replay/checkpoint.
Ruff passes. Target-shell CPU and GPU `--dry-run` expansions pass.
Preparation launcher98568 started15:48:44CST on test02 with a30min limit.
Files under `runtime/outputs/diagnostics`:

- `shared-split-dual-inputs-c599236-preparation.json` and `.log`.
- `shared-split-dual-inputs-c599236.pkl` (written only after completion).

Overlay files are isolated under `runtime/transfer/shared-dual-compact-20260925`.
Launcher variables: `PREPARE_INPUT_BUNDLE=1`, `INPUT_BUNDLE`, `ENTRY_FILE` and
`OUTPUT`; replay omits preparation mode and retains the physical GPU selectors.
The source checkout remains c599236. Stage reports now include CPU and wall
time; saved-executable loads separately record file read and deserialization.

Recovery through cln01 started13:24CST as launcher104560/Python104627,
report/log `shared-split-serialized-dual-20260925-resume1.{json,log}`.
Small script reads succeeded, but Python imports and bootstrap still waited
in `ptlrpc_set_wait`/`cl_sync_io_wait`. It exited124 at30minutes, confirmed
13:55 with no remaining Python/GPU process. No optimizer update or full-model
gradient binary load completed. A different idle-node paid comparison is
prepared in `shared_split_dual_idle_node_proposal_20260925.md`, not submitted.

## Boundary

The immutable `c599236` method and three trusted JAX 0.4.38 V100 executable
files remain the source of daily micro4, T7 and T30 gradients. The direct-load
two-changing-artifact replay already passed all-parameter and full-AdamW
comparison at `rtol=1e-5, atol=1e-6`. The new diagnostic does not lower,
trace or compile a full-model gradient and does not alter the architecture,
loss, masks, optimizer, training batch recipe or saved reference.

The saved GPU executable embeds logical device 0. A tiny actual JAX 0.4.38
retarget test found that `default_device` and explicit deserialization
assignment do not move its native replica to a second visible GPU. The
diagnostic therefore runs two isolated processes: the parent sees physical
GPU0 as logical device 0 and loads daily/T7/T30; a persistent child sees
physical GPU1 as logical device 0 and loads the same trusted T7/T30 binaries.
The child starts with its CUDA selector set before importing JAX. It uses only
inherited private pipes, with no listening port or per-update files. Standard
output is reserved for framed pickle messages and ordinary child logging goes
to standard error.

For each measured dual proposal, the parent transfers the full current
parameter tree and recurrent cases to the child while it evaluates the full
daily microbatch mean gradient. The child returns both full recurrent
gradients. The parent applies the unchanged sum, physical/numerical checks,
gradient clipping and AdamW once. Timed dual wall includes host/device and
pipe transfers, both recurrent executions, daily execution, gradient merge
and optimizer update; no transfer is hidden outside the timed proposal.

## Checks and Scope

`probe_shared_serialized_dual.py` hooks the existing learning runner's
compile and update boundaries. It compares two actual, changing 32-row
technical updates against serial execution at every full gradient,
parameter and AdamW coordinate, plus losses and optimizer scalars. It checks
the final diagnostic checkpoint against the trusted serialized-replay
reference. The runner's technical checkpoint is not formal training.

The subsequent fixed-weight benchmark times at least three interleaved,
blocked serial and dual complete proposals each for 32 rows and a genuine
64-row daily batch formed from the two distinct 32-row artifacts. The 64-row
case uses one recurrent T7/T30 case and is a throughput proposal, not the
declared training recipe. Distinct artifact IDs do not prove every daily row
is unique, and no unique-coverage claim is made. Reports include daily rows
per second, full-update medians and GPU-hour/estimated-charge ratios, plus
complete-tree parity at the same initial weights.

The deployed small proof compiled only synthetic recurrent gradients on the
parent logical GPU0, serialized them, loaded them in the child logical GPU0
on physical GPU1 and passed two changed parameter/data RPC rounds with full
synthetic gradient and loss comparison. It read no Teacher data and did not
load the full model. Evidence:
`runtime/outputs/diagnostics/shared-serialized-dual-worker-smoke.json`.
The worker's tiny recurrent/total times were 0.8766/0.8848 seconds on the
first round and 0.00416/0.00921 seconds on the second. These are protocol
checks, not full-model throughput measurements.

The first full saved-executable benchmark ran through
`scripts/hpc/run_shared_serialized_dual.sh`; its report and log are
`runtime/outputs/diagnostics/shared-split-serialized-dual-20260925.json`
and `runtime/outputs/diagnostics/shared-split-serialized-dual-20260925.log`.
Its throughput, transfer overhead and cost outcome remain pending.

Launcher PID95004, Python PID95059, began11:21CST on free gln01
(hostname ibc13b02n04),30min bound, both GPUs initially idle. At11:34 it
remained in bootstrap, with only28seconds cumulative Python CPU time and
the main thread waiting in `ptlrpc_set_wait`. A bounded5second strace
observed `lstat` of the canonical v6 typed-shard path under
`workers/worker-099-of-100/typed/spatial-train_temporal-train/`.
The concurrent175KB overlay transfer also took several minutes. These are
observations of filesystem metadata waiting during this startup, not evidence
of slow model compilation or a GPU contention explanation for earlier runs.
The26.34GB visible GPU0 allocation is the configured JAX reserved pool, not
measured live model memory. No full gradient binary had loaded at this check.

At~11:37, after more than16minutes without leaving bootstrap, the lead sent
TERM to the verified owned Python95059. Launcher exited143 and Python exited;
no optimizer updates or full-gradient loads were lost. Do not report this as
an AD, compilation, numerical or dual-execution failure. The existing JSON
retains its last running stage; this documented termination supersedes it.
No paid job or identical immediate rerun followed. Free GPUs were instead
assigned to the staged scalar/micro4 actual-layout decoder screens, which do
not load the Teacher inventory. The narrow index-path optimization being
tested locally can reduce repeated metadata calls but cannot cure an
individual stalled Lustre call.

## Code and Local Verification

- `scripts/hpc/probe_shared_serialized_dual.py`: complete-update benchmark,
  immutable-checkout diagnostic overlays and synthetic worker smoke mode.
- `scripts/hpc/probe_shared_serialized_dual_worker.py`: private recurrent
  worker that loads the unchanged trusted T7/T30 executables.
- `tests/unit/test_shared_serialized_dual.py`: artifact, pipe, proposal
  timing and CPU-sized full-gradient/optimizer behavior.
- Targeted local suite: 9 passed, including existing serialized replay and
  two-CPU-device microbatch integration. Ruff and Python compilation pass.

The deployed full benchmark is the only evidence that can establish whether
the process split materially improves complete-update sample throughput.
