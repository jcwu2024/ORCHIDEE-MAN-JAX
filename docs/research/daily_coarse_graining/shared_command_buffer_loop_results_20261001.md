# Command-buffer WHILE capability result, October 1, 2026

The deployed JAX/JAXlib 0.4.38 CUDA compiler explicitly removes WHILE capture
on the free V100 runtime. Its direct `command_buffer_scheduling.cc:825` log
reports GPU toolkit **12.1.0** and driver **12.0.0**. The same capability filter
removes CUBLAS, CUDNN and CUSTOM_CALL. This closes the proposed WHILE flag route
on the current runtime; no expert5 or model compilation was started for it.

The default command types read from live DebugOptions are
`FUSION,CUBLAS,CUSTOM_CALL,CUDNN`. The treatment adds only `WHILE`, with LLVM
module parallelism false in both arms. Per-compilation options reject the
comma-separated repeated enum (`not a valid enum value`), so two independent
processes used global XLA flags. Each compiled a nonlinear 59-iteration
`lax.map` and its original native parameter/input reverse AD, FP32, unscaled
`sin(0.11*i)` cotangent. The initial metadata diagnostic had a borrowed-object
lifetime error; retaining its parent fixes it. The failed startup logs are
preserved. These failures were diagnostic-code errors, not GPU admission.

Both arms compiled and ran within the 180-second support cap. Their optimized
HLO is byte-identical, as are the dumped PTX and thunk sequences. Both outer
loops remain `kWhile`; adding WHILE does not capture them. A third process
loaded the saved executables and prohibited compiler lookups during seven
alternating warm pairs. All **483,360** output/parameter/input coordinates are
bitwise equal in every pair and self replay. There is no mathematical or
execution candidate to upgrade to an actual loss. Replay timing is recorded
but not a speed claim: diagnostic verbose logging was enabled, and the two
executables are identical.

Direct binary capability evidence is
`outputs/diagnostics/shared-command-buffer-loops-20261001/shared-command-buffer-loops-support-while-20261001-capability-confirmed.log`.
The same directory preserves every JSON/log/result/executable/HLO. Runtime
overlay is `runtime/transfer/shared-command-buffer-loops-20261001`; its XLA
dump directories preserve before/after command-buffer scheduling, PTX,
LLVM, memory and thunk evidence. Immutable source checkout remains
`c599236bcf57b745e9bafed678298010f4c3b58f`; no core/default changes or paid jobs.
Every GPU diagnostic used one visible GPU (`gpu_count=1`, paid jobs0).
The older launcher did not record complete start/finish wall exposure for all
capability/startup failures, so a total launcher card-hour figure is unavailable;
it is not reconstructed from incomplete timestamps or warm intervals. New
runtime launchers now save start/finish, exit and full wall/card-hours explicitly.

The next bounded candidate is lowering `xla_gpu_graph_min_graph_size` from
its live default **5** to **2**, preserving the remaining FUSION capture
capability. Short fusion groups between GEMM boundaries may then be captured
without changing formulas, normalization, expert128 padding, native AD,
FP32, architecture, sampling or loss. This is a distinct unmeasured hypothesis;
no previous named threshold test was found in the bounded current-doc/script
search. First require actual HLO capture evidence and the full real expert5
decoder parameter/raw-bank/input VJP comparison. Neither the unsupported WHILE
flag nor the old rejected normalized/unroll2 variants are included.
