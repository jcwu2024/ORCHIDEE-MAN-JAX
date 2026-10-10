# Original precision deterministic execution check

2026-10-02: **the current runtime supports deterministic GPU operations, but
direct adoption is closed by the real expert5 result.** Deterministic expert5
outputs and complete native VJPs are bitwise stable across eight evaluations.
They fail the unchanged strict comparison against the original saved binary,
and synchronized execution is approximately1.8875x slower. No precision,
scientific gate, reference, training method or default changes.

## Actual compatibility

Free gln01, actual host `ibc13b02n04`, GPU1 V100-SXM2-32GB, driver525.85.12,
canonical JAX0.4.38 and existing Singularity/driver bindings. Only this GPU is
used. Fresh occupancy checks and launcher records show both GPUs empty before
the three stages; this is a snapshot, not a reservation. CPU affinity0–111,
OMP4/OpenBLAS1. Preflight/primitive/expert launch loads are respectively
0.03/0.05/0.28,0.11/0.08/0.27 and0.01/0.03/0.13.

Canonical immutable checkout preflight
`scripts/hpc/check_daily_training_runtime.py` passes with
`--xla_gpu_enable_llvm_module_compilation_parallelism=false --xla_gpu_deterministic_ops=true`.
Its synthetic public AD, float0, state-coupled fractions, same-shape reuse,
weighted accumulation and AdamW checks pass; it reads no Teacher/model data.

Three additional small synthetic probes use dynamic unsorted4096 indices into
97 buckets, FP32 inputs/parameters and FP64 scalar accumulation. Duplicate
scatter-add, gather native VJP and a native checkpointed T7 scan with full
parameter/state VJP all compile and execute. Each result is finite, nonvacuous
and bitwise stable across eight calls. Backend times are0.557907,0.554667 and
1.321885s. This establishes bounded primitive compatibility, not full-model
determinism or gradient correctness.

## Real original expert5 check

The original real expert5 capture is reused unchanged: sample0/micro1,
query[7542,1,384], full59chunks128,384 width/eight heads/two decoder blocks,
all3,548,928 decoder parameters and21,579,648 captured input coordinates.
The input VJP scopes are query2,896,128, dynamic17,376,768 and raw banks1,306,752.
FP32 neural precision and the original JAX-generated `sin(arange*0.11)`
cotangent on every output coordinate remain unchanged. Captured masks/indices
and raw bank tensors are the same objects in both calls. This component has
no physical updater, backbone, public science loss or full-network update.

The original native VJP executable is deserialized once from
`shared-command-buffer-loops-threshold-component-default-20261001.default.native_vjp.executable.pkl`.
It is the original expert128 `lax.map` execution. The outer micro1 sample map
is retained exactly as in that baseline; no closed chunk-vmap or other
candidate runs. Original model compilations0. Saved identity, route,
microbatch, architecture, input shape, cotangent and compiler options are
checked. The original fresh/saved output and query/dynamic VJPs match bitwise;
raw banks pass strict comparison. Its parameter VJP replay has1758 strict
failures, preserved as original drift rather than waived.

One new combined output/native VJP graph calls the same original component
source. **JAX's actual compiler API accepts the explicit
`xla_gpu_deterministic_ops=True` option.** All remaining numerical compiler
settings match the saved default: graph minimum size5, LLVM parallelism false,
FUSION/CUBLAS/CUSTOM_CALL/CUDNN command-buffer types. Process flags remain the
original numerical flags; only the new executable receives the deterministic
override. Historical dump-only flags/path are recorded from the original
report and are not reused. Thus saved-original execution is not relabelled as
deterministic merely because of a process environment flag.

The first output/VJP from each arm and seven pairs alternate evaluation order.
Each call synchronizes the entire output and full parameter/input VJP. All16
results are finite. The complete arrays are preserved for the two initial
results and all seven pairs; no partial gradient is used to classify a pass.
Strict allowance remains `1e-6 + 1e-5*abs(original_reference_coordinate)`.

| Comparison | Decoder parameter failures / 3548928 | Raw-bank failures / 1306752 | Output/query/dynamic failures |
| --- | ---: | ---: | ---: |
| First new/original | 749403 | 1561 | 0 / 0 / 0 |
| Pair0 | 749059 | 1569 | 0 / 0 / 0 |
| Pair1 | 749187 | 1569 | 0 / 0 / 0 |
| Pair2 | 749838 | 1564 | 0 / 0 / 0 |
| Pair3 | 748889 | 1567 | 0 / 0 / 0 |
| Pair4 | 749701 | 1556 | 0 / 0 / 0 |
| Pair5 | 749413 | 1559 | 0 / 0 / 0 |
| Pair6 | 749378 | 1570 | 0 / 0 / 0 |

Initial maximum errors are1.275539e-4 for parameters and1.469254e-5 for raw
banks. The output strict bridge passes with maximum1.430511e-6, but is not
bitwise identical. Every deterministic self comparison has zero difference
in every scope. Original parameter self comparisons have respectively
1625,1530,1678,1718,1694,1797 and1557 strict failures. Original raw-bank self
differences stay below the strict allowance; query/dynamic/output self values
are bitwise identical.

These observations show a concrete deterministic execution path for this
component and stronger repeat stability than its original binary. They do
not establish that the deterministic derivative is scientifically correct,
explain the earlier slicing/JVP discrepancies causally, or authorize replacing
the reference. The new/original gate remains failed. No public T7/T30 graph,
training, Teacher regeneration or paid job is started.

## Runtime, memory and evidence

| Stage | Python seconds | Launcher seconds | One-GPU card-hours | Actual exit |
| --- | ---: | ---: | ---: | ---: |
| New deterministic canonical preflight | 33.135296 | 40.078857 | 0.011133015818 | 0 |
| Synthetic index/scan primitives | 3.873302 | 4.784033 | 0.001328898138 | 0 |
| Real expert5 paired native VJP | 49.310289 | 51.965571 | 0.014434880879 | 0 |
| Total | — | 96.828461 | **0.026896794835** | — |

Real lowering1.443105s/backend23.347234s, within120s/240s bounds; external
process600s/internal590s. All lowering, backend, executable-save, initial-save
and pair-save phases persist separately. Preflight and primitive external
bounds are180s/120s. Target Bash syntax and actual dry-run expansions pass.

Seven synchronized warm VJP medians are original0.099106913s and
deterministic0.187065205s; original/new ratio0.529799x. This is a component
diagnostic, not accepted training throughput. The deterministic option is
slower in this measured scope. Estimated live memory is2,260,028,960 original
and2,260,030,368 new bytes; observed whole-process peak2,620,563,968bytes is
below24GiB. The synthetic primitive peak is4,787,456bytes.

Local compact artifacts:
`outputs/diagnostics/shared-deterministic-runtime-20261002/`, including all
three final reports, logs, actual exits, executed sources/launchers, synthetic
StableHLO/output arrays and artifact manifest. Full expert output/VJP pairs,
serialized deterministic executable and lowered/compiled HLO remain remote
under the matching directory in `runtime/outputs/diagnostics/`; the nine raw
result files total1,793,619,232bytes. Matching transfer directory is
`runtime/transfer/shared-deterministic-runtime-20261002`.

Independent CPU-only saved-array verification on test01 (24.868022s, no model
execution) recomputes the first comparison, all seven pair comparisons and
all14 self comparisons. Every recorded strict failure count and maximum error
matches. Complete-result fingerprints have eight unique original results and
one deterministic result. `expert5.artifact-validation.json` preserves this
check; the small verifier is kept with the local artifacts.

Source checkout stays immutable at
`c599236bcf57b745e9bafed678298010f4c3b58f`. New sources are
`probe_shared_deterministic_runtime.py`, `run_shared_deterministic_runtime.sh`,
`probe_shared_deterministic_expert5.py` and `run_shared_deterministic_expert5.sh`
under `scripts/hpc/`. Current entry documents remain the lead's responsibility.
