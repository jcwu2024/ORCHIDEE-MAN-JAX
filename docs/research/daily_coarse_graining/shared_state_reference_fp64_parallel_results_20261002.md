# Original-public T7 full-neural-FP64 parallel4 diagnostic: 32-call pass

2026-10-02. **The separately approved free run produced one executable and
all32 saved results. The actual-GPU harness and independent NumPy-only
saved-array verifier passed the unchanged strict gates.** The new family used
2783.953854s (46.399231min,0.773320515023 card-hours), below its4500s cap.
This establishes bounded internally consistent initial-state derivatives for
the declared original T7 case. It does not adopt a new reference or precision
default, admit a closed candidate, or establish training throughput.

The [approved proposal](shared_state_reference_fp64_parallel_compile_proposal_20261002.md)
is fulfilled. The closed [first FP64 run](shared_state_reference_fp64_results_20261002.md)
and its outputs remain preserved. No further GPU execution is part of this
result; unused allowance does not authorize another expensive compilation.

## Scope, graph and physical admission

Execution used the immutable `shared-split-width-v1` checkout,
HEAD`c599236bcf57b745e9bafed678298010f4c3b58f`, original update0,
artifact742/window0/T7. All64,884,481 weight values in300 leaves were exactly
widened to FP64 and passed dynamically. Differentiation covered only the3854
initial states. The four dynamic basis directions33,199,428,200 each received
eight calls, with the declared alternating order and unchanged public loss.
No parameter AD or real optimizer update was performed.

The268 input leaves,15 calibration/context arrays and112 source/science/PFT
boolean masks matched the prior admitted preparation. Serialized prepared-data
class identities and exact floating-value widening were retained. The
executed `combined` AST and frozen neural policy matched the prior executed
FP64 source. Physical formulas and masks were preserved, including the exact
R4 threshold`2**-23`; the source-backed constant bridge and Fortran provenance
are documented in the linked first-run report and proposal.

Lowering took228.479364s and contained zero FP16/BF16/FP32 tensors. Its saved
StableHLO was **raw-byte identical** to the prior admitted FP64 graph:
SHA256`87870eb2b081e27c2bd6b9e6ab13d3367f7f9154162a42f7ca94fdd0f4812a26`.
No textual normalization was needed. The backend therefore evaluated the same
admitted computation with the newly requested compiler configuration.

All32 results were finite, with all14 hard physical counts zero, declared-mask
count14, all112 saved source masks unchanged, and exactly zero derivatives at
all21 undefined source-state coordinates. The independent CPU verifier checked
the saved full tuples and mask arrays against their identities and recorded
`status=passed`, `call_count=32`, `rtol=1e-5`, `atol=1e-6`.

## Derivative consistency and repeat drift

The normalized error below is
`absolute_error / (1e-6 + 1e-5*abs(VJP))`; the strict gate passes at≤1.
FP64 values are the first saved call per direction; maxima cover all eight
calls. The full ranges are retained in`terminal-summary.json`.

| Direction | FP64 JVP | FP64 VJP | Maximum absolute JVP/VJP error | Maximum normalized error |
| --- | ---: | ---: | ---: | ---: |
|33|-0.19848743570550928|-0.19848743571754932|1.213615319351e-11|4.065884101396e-6|
|199|-0.05423567693724193|-0.05423567693723560|6.890321646580e-15|4.467398064706e-9|
|428|1.2590262625061581|1.2590262625062438|9.281464485866e-14|6.829496045757e-9|
|200|973.9010815123311|973.9010815122923|3.899458533851e-11|4.003546410644e-9|

All full3854-state VJP comparisons and complete same-direction result repeats
passed the strict tolerance. Independent saved-array summaries give:

| Comparison | Maximum absolute state-VJP drift | Maximum normalized drift |
| --- | ---: | ---: |
|Each direction against its first call|1.601874828339e-7|7.828610626315e-8|
|All directions/repeats against the first overall call|8.381903171539e-8|7.372042409481e-8|
|Largest pairwise range across all32 calls|1.639127731323e-7|1.252540505244e-7|

For the pairwise row, normalization uses the first overall call's coordinate
magnitude. These are small accepted numerical drifts, rather than bitwise
identity. Each direction's JVP was unchanged across its eight calls; the full
VJP had the small drift quantified above. Bitwise equality was not an admission
requirement.

## Differences from saved original FP32

FP64 loss was0.08667498215554777, versus saved FP32
0.08665354178956992: change`+2.144036597784e-5`, or`+0.024742631%`.
The loss/aux comparison exceeded the same tolerance at5 of20 scalar
coordinates, with maximum absolute difference0.00042679383607813204.
The full state gradient exceeded it at584 of3854 coordinates, with maximum
absolute difference14328.010894123465 at state1785: FP64
-13981726.45395039 versus FP32 -13996054.464844514. Those comparisons report
precision sensitivity; matching FP32 was expressly not the new diagnostic's
admission gate.

| Direction | Saved FP32 JVP | Saved FP32 VJP | New FP64 JVP | New FP64 VJP |
| --- | ---: | ---: | ---: | ---: |
|33|0.36843998782004184|0.3686206638813019|-0.19848743570550928|-0.19848743571754932|
|199|-0.04918365310913531|-0.049182327107955906|-0.05423567693724193|-0.05423567693723560|
|428|1.2265334746571595|1.2264883582101902|1.2590262625061581|1.2590262625062438|
|200|972.1697290384398|972.1667741504964|973.9010815123311|973.9010815122923|

Direction33 changes sign between the declared precision computations. The
independent evidence establishes FP64 JVP/VJP consistency within this bounded
case. It does not by itself prove a mathematical or scientific reference,
identify the source of FP32/FP64 sensitivity, or certify parameter gradients,
other cases, T30, training accuracy or all Fortran branches. A future reference
or precision adoption would need a separate scientific decision. No such
adoption is necessary to close this approved diagnostic.

## Runtime, memory and actual budget

One physical GPU1 V100 on free`gln01` /`ibc13b02n04` was used with the canonical
GPU environment/container, JAX0.4.38, driver525.85.12, OMP4/OpenBLAS1 and CPU
affinity0–111. Both GPUs were idle immediately before launch. Model-start
load was0.32/0.11/0.08; the earlier fresh snapshot was0.00/0.01/0.05. These
observations do not establish a causal contention effect.

| GPU stage | Launcher seconds | Card-hours | Outcome |
| --- | ---: | ---: | --- |
|Canonical synthetic runtime preflight|90.357491493|0.025099303193|Passed; Python79.989187s|
|FP64 lower/backend/serialization/32 calls|2693.596362591|0.748221211831|Exit0, completed|
|**New-family cumulative**|**2783.953854084**|**0.773320515023**|Below4500s/1.25card-hours|

The preflight is a synthetic runtime/interface check, including its canonical
synthetic optimizer case. It is not real-model or scientific acceptance. The
model backend completed in2328.484519s, below3600s; Python total was2676.345874s,
below its dynamically shortened4369s process limit. Exactly one expensive
attempt and one successful model compilation were recorded. Paid cost wasCNY0;
no`sbatch`, paid fallback, training, T30, FD or precision sweep occurred.

Estimated active live memory was5,946,893,925bytes /5.538477GiB; observed JAX
active peak was5,991,222,016bytes /5.579760GiB. Both are strictly below24GiB.
The allocator's peak pool was10.001953GiB, a separately reported quantity.
Earlier~2540MiB context occupancy during compilation was not used as memory
admission evidence.

All32 measured executable calls had median2.939246s and range2.930598–3.392285s.
Excluding the first four-call cycle, the28 repeated calls had median2.939246s
and range2.930598–2.996492s. Timing includes execution, synchronization and host
result transfer. This state-only FP64 combined VJP/JVP diagnostic is a different
workload from parameter-plus-state AD or a complete optimizer update; these
times cannot establish training throughput or be compared as an equivalent
full-AD speed ratio.

Node19:25:02 terminal inspection found PID50543 gone, both GPUs0MiB/0%, and no
GPU compute processes. Its context was released. The original closed family's
0.647810510861card-hours remain separate from this new authorization.

## Linker evidence and compilation limits

The model log contains23 wrapper invocations: two`--version` calls,20
preparation/helper device links at epoch1790937586.520998442–1790937588.719067942,
and one public-backend device link. The captured`stage=compiling` report had
mtime epoch1790937823 (18:43:43), at main elapsed238.497027s. Whole-second
mtime resolution makes its start boundary1790937823–1790937824 uncertain;
none of the observed calls falls near that boundary.

The final call started at epoch1790940019.316587416 /19:20:19.316587+08:00,
well after compile start. Adding the measured2328.484519s backend duration
places compilation return at about19:22:31–32 (main elapsed≈2566.981546s).
The executable's mtime19:22:45 supplies an independent postcompile bound.
Only`lowered.compile` was active in this interval; thus the23rd record is
attributable to the public`jit_combined` backend. It invoked the matching
nvlink12.6.85 with`-arch=sm_70`, four input cubins and one output path:

```text
/tmp/tempfile-ibc13b02n04-95b4ae4f37163e5-50543-65cd9b7a9d14a.cubin
/tmp/tempfile-ibc13b02n04-95b4ae4f37163e5-50543-65cd9b7a9dcdb.cubin
/tmp/tempfile-ibc13b02n04-95b4ae4f37163e5-50543-65cd9b7a9e898.cubin
/tmp/tempfile-ibc13b02n04-95b4ae4f37163e5-50543-65cd9b7a9f27b.cubin
-o /tmp/tempfile-ibc13b02n04-95b4ae4f37163e5-50543-65cd9b7a9fd7c
```

LLVM module parallelism and force4 were requested, with LLVM parallelism also
passed explicitly to`compile(compiler_options=...)`. The logged four inputs
demonstrate actual device linking within that backend. They do not measure
four concurrent threads or a4× acceleration. The prior serial FP64 backend
was censored at1800s and never completed, while this requested-parallel run
completed in2328.484519s. There is no uncensored equivalent serial baseline,
so a compiler speed ratio cannot be calculated.

## Saved artifacts and reproducible CPU verification

Local compact evidence is under
`outputs/diagnostics/shared-state-reference-fp64-parallel-20261002/`:
actual terminal model/preflight JSONs and exits, executed source/policy/launcher
snapshots, logs,32 complete result pickles, input identity and mask arrays,
preparation checks, runtime observations and final linker-phase evidence.
`cpu-independent-verification.json` records the NumPy-only gate result;
`terminal-summary.json` retains the metrics above;
`artifact-manifest.json` records actual paths, sizes and SHA256 values.

The remote-only executable is250,679,801bytes,
SHA256`253e432eefc0e63642faff07a85a85436ccd945ff60165134bf4b02a7230e22c`.
The raw-identical StableHLO is63,413,105bytes. Both remain under the same
new family in the project's remote runtime outputs; neither was regenerated.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
conda run -n ORCJAX python outputs/diagnostics/shared-state-reference-fp64-parallel-20261002/verify_saved_arrays.py
conda run -n ORCJAX python outputs/diagnostics/shared-state-reference-fp64-parallel-20261002/summarize_terminal.py
```

These commands read saved artifacts, import no JAX, and launch no GPU work.
Strict gates, precision defaults, adopted references and closed candidate
decisions are unchanged. The speed workstream can resume with this bounded
FP64 consistency evidence and the reported precision sensitivity explicit.
