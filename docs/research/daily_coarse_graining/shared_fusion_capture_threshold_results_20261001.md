# FUSION capture threshold: real expert5 result, October 1, 2026

Reducing `xla_gpu_graph_min_graph_size` from **5** to **2** is supported and
changes actual GPU command-buffer execution, but the real expert5 micro1
component has **no warm benefit**, uses **33.37% more live memory**, and fails
the unchanged strict parameter-VJP gate. Do not upgrade it to micro4, actual
loss, T7/T30 or full optimizer updates. No core/default/runtime change is
adopted.

This candidate preserves original FP32 formulas, normalization placement,
two decoder384 blocks, eight heads, expert128 padding (7,542 actual rows,
59 chunks), original `lax.map`, and native reverse AD. The saved original
raw banks are reprojected, including every decoder parameter and raw-bank/input
derivative. The cotangent is the original JAX `sin(0.11*i)`, unscaled, over
every output coordinate; this is not a science loss. LLVM parallelism is false
for both same-process arms. Only the minimum command-buffer group size differs.
Both arms retain the deployed default command types; WHILE is not requested.
Public loss, physical/science checks and model/optimizer updates were not run.

The small support check first demonstrates real `to_apply=%command_buffer`
calls after lowering the threshold (default0, candidate3) with bitwise-equal
483,360 output/parameter/input coordinates. Its initial call-count parser
looked for `calls=` instead of `to_apply=` and reported0 in both JSON rows;
the preserved HLO supplies the actual counts, and the parser is corrected.
The real component's saved HLO and native thunk dumps then show additional
capture: **20 → 34** command-buffer calls in optimized native-VJP HLO, with
different thunk sequences. PTX files also differ; kernel identity is not claimed.

| Actual expert5 micro1 metric | Default5 | Threshold2 |
|---|---:|---:|
| Lowering seconds, forward + native VJP | 1.459318 | 0.003131 |
| Forward backend compilation seconds | 2.564801 | 1.526769 |
| Native VJP backend compilation seconds | 7.374270 | 6.722212 |
| First native VJP seconds | 0.204254 | 0.219419 |
| Seven-pair warm forward median seconds | 0.025250474 | 0.025346315 |
| Seven-pair warm native VJP median seconds | 0.100610364 | 0.101691518 |
| Estimated live bytes | 2,260,028,960 | 3,014,278,560 |

The forward ratio is **0.996219x**, native VJP **0.989368x**. Candidate lowering
reuses the same traced function; its short lower time is not an independent
startup comparison. Total Python process is **37.904658s**, launcher exit0,
one idle free V100, OMP4. Two-binary peak device allocation is **3,315,889,920**
bytes; this is distinct from each executable's estimated live bytes.
`gpu_count=1`, paid jobs0. The Python37.904658s corresponds to0.010529072
card-hours as a **partial process measurement**, including its compilation;
it excludes launcher startup and is not total launcher exposure. The original
launcher did not record a complete finish timestamp; full launcher card-hours
are unavailable. Its preserved log records exit0. Future runtime launches save
explicit complete wall/card-hours sidecars.

All **2,896,128** output coordinates are bitwise equal. All **21,579,648**
captured raw-bank/input derivative coordinates pass `rtol=1e-5, atol=1e-6`
in the initial comparison and every pair. The complete **3,548,928** decoder
parameter derivative coordinates fail: initial1,674, seven pairs
**[1935,1557,1729,1547,1591,1489,1549]**. The initial maximum absolute
parameter error is1.4305115e-5; raw-bank/input error is8.6426735e-7.
Default self comparisons fail1,473–1,778 parameter coordinates; candidate
self comparisons fail1,592–1,853. This reproduces known original native
expert5 scatter accumulation noise, but does not waive candidate failures or
establish numerical equivalence. Even independent of that unresolved gate,
the measured time and memory reject this candidate.
Both initial complete trees are finite, with all3,548,928 parameter coordinates
nonzero and4,186,752 nonzero raw-bank/input derivative coordinates; the local
`.finite-analysis.json` records this independently from saved results.

Source evidence uses the canonical `shared_transformer` projection and block
runner, plus original expert128 diagnostic packing. The graph service remained
`Transport closed`; source fallback was bounded, with no exhaustive graph claim.
The immutable deployed checkout is
`c599236bcf57b745e9bafed678298010f4c3b58f`. No Teacher data preparation or model
architecture/sampling/loss change occurred.

Implemented entry points are `scripts/hpc/probe_shared_command_buffer_loops.py`
and `scripts/hpc/run_shared_command_buffer_loops.sh`. Syntax/AST and actual
target-shell `--dry-run threshold-support`/`threshold-component` pass. The
local asset directory is `outputs/diagnostics/shared-command-buffer-loops-20261001`;
the decisive stem is
`shared-command-buffer-loops-threshold-component-default-20261001` and includes
JSON, log, complete result trees, forward/native-VJP binaries and optimized HLO.
Matching remote artifacts and compiler dumps remain under project runtime.

The initial JSON mistakenly inherited the synthetic WHILE `scope` label.
Its preserved `.original-scope.json` records the executed report; the main
JSON's explicit `metadata_correction` changes only that label to the real
expert128 component scope. Numerical/timing data are unchanged and no rerun
was performed. The `.executed.py`/`.executed.sh` snapshots preserve the actual
component driver and launcher. Initial complete result trees are saved for
both arms; seven-pair error summaries and self comparisons are retained,
but full repeated parameter vectors were not saved in this run.

The adjacent capability diagnostic establishes why only FUSION capture is
available: the current compiler removes CUBLAS/CUDNN/CUSTOM_CALL/WHILE for
toolkit12.1/driver12.0. This threshold result does not measure a newer compatible
runtime's capture capabilities. Any such next candidate first needs a bounded
runtime compatibility check; do not sweep more arbitrary flags or recompile
the old normalized/unroll2 candidates on this evidence.
