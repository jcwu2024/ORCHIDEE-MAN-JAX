# CUDA 12.3 capture runtime: supported, WHILE candidate rejected

The isolated CUDA12.3 library pair passes the actual deployed GPU runtime
preflight and enables previously unavailable capture. The original real
expert5 micro1 decoder nevertheless runs **slower** with WHILE capture, and
its native parameter/input VJPs fail strict parity by much more than baseline
self noise. Do not upgrade WHILE to daily loss, T7/T30 or optimizer updates.
The new library pair is a verified diagnostic asset, not an adopted production
runtime or a measured full-model speed improvement.

## Runtime identity and actual capability

The official minimal package pair was verified and extracted independently:
`cuda-compat-12-3-545.23.08-1` plus `cuda-cudart-12-3-12.3.101-1`, total
40,878,973 compressed bytes. See the separate manifest and CPU preparation
evidence in `outputs/diagnostics/shared-cuda-compatibility-20261001`.
Canonical environments, SIF and system libraries remain unchanged.

This execution uses process-only LD_LIBRARY_PATH/absolute LD_PRELOAD from
`runtime/toolchains/cuda-runtime-compat-12.3.101-driver-545.23.08`.
Before and after actual GPU initialization, `/proc/self/maps` and API queries
identify only `usr/local/cuda-12.3/compat/libcuda.so.545.23.08` and
`usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudart.so.12.3.101`, both
API12030. The kernel driver reported by nvidia-smi remains525.85.12; a system
driver replacement is not claimed. JAX/JAXlib remain0.4.38, source checkout
`c599236bcf57b745e9bafed678298010f4c3b58f`.

The candidate calls canonical `check_daily_training_runtime.py` under these
actual GPU libraries once: passed, Python29.080339s, launcher34.014457s.
FP32/FP64 native AD, microbatch accumulation and AdamW synthetic interfaces
pass. This reads no Teacher data and is not scientific/model admission.
Subsequent processes verify the same libraries against the completed preflight.

Small59-iteration native map/VJP support arms both exit0 with no capability
type-removal messages. The WHILE arm's optimized HLO has a command-buffer call
containing both original loops; default has none. All483,360 synthetic
output/parameter/input coordinates are bitwise identical. The real default
arm then also captures GEMMs in its command buffers, consistent with restored
CUBLAS capability. Actual PTX and thunk evidence are preserved; enum presence
or CPU API queries alone were not used to assert GPU support.

## Real component scope and matched execution

Both arms reuse the captured raw inputs from update0/artifact742, recompute all
bank K/V projections, and return the complete native VJP. Expert5 micro1 is
7,542 actual rows with original chunk128/padding, two384-wide blocks/eight
heads, original FFN/attention/normalization placement, FP32 and `lax.map` AD.
Full parameter scope is3,548,928, raw-bank/input VJP scope21,579,648, output
scope2,896,128. The original JAX `sin(0.11*i)` cotangent is unscaled over all
outputs; it is a synthetic diagnostic. Public loss/physical science/model
updates were not executed.

Treatment adds WHILE to default `FUSION,CUBLAS,CUSTOM_CALL,CUDNN`. Minimum
graph group remains5; LLVM parallelism remains false. No threshold2, unroll,
normalized-bank, precision, architecture, loss or sampling change is mixed in.
Repeated enum options require independent compile processes in JAX0.4.38.
Both executables were compiled in the same candidate runtime, then loaded in
a third process with the same exact library and input identities. The original
cotangent helper is prepared before replay; component compiler lookups are
prohibited during seven alternating pairs. Thus zero component recompilation
is claimed, not zero elementary-helper compilation during process startup.
All seven full parameter-VJP vectors per arm are saved in both individual-arm
measurement and paired replay; initial complete output/input/parameter trees
are saved as well.

| Metric | New-runtime default | New-runtime +WHILE |
|---|---:|---:|
| Lowering seconds, forward + VJP | 1.485584 | 1.476649 |
| Forward backend seconds | 2.332848 | 1.912726 |
| Native VJP backend seconds | 7.584215 | 7.324975 |
| First native VJP seconds | 0.221888 | 0.217617 |
| Seven alternating warm forward median seconds | 0.024538584 | 0.027833715 |
| Seven alternating warm native VJP median seconds | 0.101039125 | 0.112683052 |
| Estimated live bytes | 2,949,780,640 | 2,291,923,744 |
| Native-VJP optimized HLO command-buffer calls | 5 | 1 |

The one WHILE-arm command buffer contains the outer native loops; call-count
reduction does not mean fewer mathematical operations. HLO/PTX/thunk sequences
differ. Forward ratio is0.881614x, native VJP0.896667x: approximately13.4% and
11.5% slower, respectively. Live memory falls22.30%; cumulative replay peak
device allocation is3,280,753,152 bytes. Neither memory improvement nor actual
capture supplies a throughput gain. The older original-runtime component
timings remain separate; no old-runtime executable was loaded in the new
environment and presented as an old-runtime wall measurement.

## Strict errors and independent repeat evidence

Every output comparison passes (initial maximum absolute error1.311303e-6),
but complete parameter and raw-bank/input VJPs fail `rtol=1e-5, atol=1e-6`.
Initial parameter failures735,295, maximum error1.783371e-4; initial input
failures1,500, maximum error1.440942e-5. Parameter failures over seven pairs:
**[735855,735101,735876,735502,735008,735391,735423]**. Input failures:
**[1489,1497,1487,1500,1498,1504,1499]**.

Default self comparisons fail1,617–1,894 parameter coordinates; WHILE self
fails1,544–1,868. Both arms' input/output self comparisons pass. NumPy-only
analysis of the full saved parameter vectors finds716,166 coordinates fail
every pair, with709,458 having disjoint observed ranges separated by more
than the largest unchanged strict allowance. In particular, both arms' four
FFN matrix-gradient leaves are internally bitwise stable, while521,148 of
their coordinates fail every pair. This establishes treatment differences
beyond the observed scatter noise; it does not rely on increasing tolerance
or accepting noisy parameter admission. Actual optimized kernels differ;
mathematical source reuse was not represented as bitwise compiled equivalence.

## Resources, exits and assets

All GPU phases use one idle free V100 (`gpu_count=1`), OMP4, no paid jobs.
Six started processes have complete launcher start/finish/exit records:
preflight34.014457s, support2.014161s/3.818838s, real components22.783674s/
22.080151s, paired replay19.484157s. Their summed launcher exposure is
**104.195437s =0.028943176866 GPU card-hours**, including startup/compilation.
CPU preparation is separate. A first immediate WHILE attempt exited2 in
the occupancy guard before GPU invocation; its failed observation was not
recorded then, so no retrospective cause is asserted. It has GPUcount0 and
card-hours0. Both failure and successful retry sidecars are retained. The
launcher now records observed memory/utilization, phase and actual startup
state, and preserves every attempt record. Final GPUs have0MiB/0% and no
compute PIDs; all execution and transfer sessions completed.

Scripts are `probe_shared_command_buffer_loops.py`,
`run_shared_command_buffer_loops.sh` and the NumPy-only
`analyze_shared_command_buffer_runtime.py`. AST, target bash syntax, actual
new-environment dry-run and real early-guard/no-GPU recording were verified;
actual runtime/component/replay provide the numerical and execution checks.
No new standalone static tests mirror these wrappers.

Local evidence is `outputs/diagnostics/shared-command-buffer-loops-20261001`,
with stems `shared-command-buffer-loops-runtime-{preflight,support,component,replay}-...-20261001-compat`.
The replay interval-analysis JSON preserves per-leaf range evidence/examples;
launcher-accounting JSON links all actual exits and resource records. The
6.1MiB compiler-dump archive retains4,080 entries including before/after
command-buffer scheduling, PTX, LLVM, memory and thunk evidence. Remote
artifacts and overlay remain inside project runtime. Original WHILE capability
and threshold5→2 evidence are preserved separately. No current entry docs,
core defaults, canonical environment or scientific protocol was changed by
this owner.

The next operation should use this negative real evidence when selecting a
distinct supported compute route. This result supplies no reason to repeat
WHILE on micro4 or compile the full model for it, and no same-FP32 path to a
tenfold training gain is established.
