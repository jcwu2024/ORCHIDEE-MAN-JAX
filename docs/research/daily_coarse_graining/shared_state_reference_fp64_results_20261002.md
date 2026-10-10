# Original-public T7 full-neural-FP64 diagnostic: backend timeout

2026-10-02. **The authorized diagnostic reached its 1800s backend-compilation
cap before producing a binary. No real-model forward result, state VJP or JVP
was evaluated.** This is a compilation-budget limitation, with no numerical
verdict about FP64 derivative consistency or stability. The ordinary dtype
admission issue encountered first was repaired successfully.

This closes the single bounded execution authorized in the
[precision proposal](shared_state_reference_precision_proposal_20261002.md).
It establishes neither a new diagnostic reference nor a speed gain. Existing
precision defaults and closed candidate decisions remain in force. No second
expensive compilation, parameter AD, real optimizer update, training, T30, FD,
precision sweep, paid job or `sbatch` was performed.

## Executed scope and admission evidence

The immutable checkout was `shared-split-width-v1` at
`c599236bcf57b745e9bafed678298010f4c3b58f`. The exact original update0,
artifact742/window0/T7 identity matched the saved original-FP32 evidence,
including source-profile SHA, checkpoint, method and staged decoder/backbone
hashes. All 64,884,481 weight values in 300 leaves were exactly widened to
FP64 and supplied dynamically. The diagnostic differentiated only the 3854
initial states; directions33,199,428,200 were dynamic, with 32 calls planned
and **zero calls executed**.

Preparation checked the complete JAX input treedef/class identities, shapes,
dtypes and exact floating-value widening over268 leaves. Fifteen FP32
calibration/context arrays were exactly widened without reconstructing
serialized dataclasses. The112 boolean mask arrays from the case, layout,
normalization, retained template, state scales and contract were unchanged.
Source provenance and audited neural boundaries are in the proposal; no
physical process function was copied, recompiled or replaced.

The canonical actual-GPU synthetic runtime preflight passed with JAX0.4.38,
one visible V100 and driver525.85.12. It took69.927946s inside Python and
77.857677s at the launcher. This checks runtime interfaces, including its
synthetic optimizer case; it is not real-model or scientific acceptance.

The first full lowering took229.803695s and contained42 FP32 tensor
occurrences, all scalar R4 epsilon constants subsequently promoted to FP64.
It stopped before any expensive backend compile. Exact canonical source reads
identified `jax_orchidee/stomate/season.py` line26, uses304/310/576, and
`permafrost.py` line25, epsilon defaults57/142/170. The diagnostic temporarily
widened that scalar representation and the three already-bound epsilon
defaults. Its physical value remained exactly
`1.1920928955078125e-7 = 2**-23 = 0x1.0000000000000p-23`, rather than FP64
machine epsilon. Source FP32 bits were`0x34000000`; exact FP64 bits were
`0x3e80000000000000`. Formulas and numeric thresholds were unchanged.

The small CPU boundary/restoration check passed: masks immediately below,
at and above the threshold were identical, and both globals and all three
keyword-default dictionaries were restored on context exit. Actual canonical
source snapshots, hashes, locations, values and hex evidence are saved in
`outputs/diagnostics/shared-state-reference-precision-audit-20261002/cpu-exact-constant-check.json`.

The repaired full public lowering took201.938510s and passed the strict
zero-FP16/BF16/FP32-tensor check. It therefore admitted the full dtype lowering
to the single expensive backend attempt. This is graph admission evidence,
not numerical derivative evidence.

## Budget and terminal outcome

One free GPU1 on `gln01` / `ibc13b02n04` was used. Both GPUs were idle before
each launch. The launches retained canonical container/driver bindings,
OMP4/OpenBLAS1 and disabled LLVM module-compilation parallelism. The first
occupancy snapshot had load0.00/0.01/0.05; the repaired launch had
0.01/0.22/0.20. No contention claim is inferred from later snapshots.

| GPU stage | Launcher seconds | Card-hours | Outcome |
| --- | ---: | ---: | --- |
| Canonical synthetic preflight |77.857677|0.021627132429|Passed |
| First FP64 lowering |243.019452|0.067505403294|Ordinary dtype admission failure; no backend attempt |
| Repaired lowering + backend |2011.240710|0.558677975138|Zero FP32 lowering; backend timer fired |
| **Cumulative** |**2332.117839**|**0.647810510861**|Below2640s cumulative authorization |

The repaired launch's external timeout was dynamically limited to2299s,
with internal process limit2279s after accounting for the earlier320.877129s
and a grace/setup margin. Its backend timer fired at1800s; terminal JSON
records `status=timeout`, `timeout_scope=backend`, `limit_seconds=1800`,
Python elapsed2009.343844s. The container/launcher returned exit1, which is
preserved alongside that explicit phase report. `model_compile_attempted=1`
and the harness's precompile `model_compiles=1` are attempt markers: **zero
successful model compilations and zero serialized executable binaries** were
produced. The worker process was gone and both GPUs showed0MiB/0% with no
compute processes after termination.

The remaining cumulative allowance does not authorize another expensive
compile. No automatic expansion was attempted.

## Independent saved-array check and scientific limits

The NumPy-only verifier read the saved identity/mask arrays, original-FP32
identity and both attempts' initial state. It independently checked all112
mask shapes, booleans, hashes and values; the initial3854-state array and
masks were identical between attempts. The saved source state has21 undefined
coordinates under the canonical finite/`abs<0.5e20` rule.

There are no saved FP64 model-result arrays to verify. The independent report
therefore records `call_count=0`, `status=model_numerics_not_admitted` and null
numerical gates. New finite values,14 hard physical counts, declared-mask
count14, undefined-state exact-zero derivatives, estimated/observed active
memory below24GiB, JVP/VJP parity, full-VJP direction invariance, eight-call
repeat stability and FP32/FP64 primal/gradient differences are **unmeasured**.
The source-mask check is not a substitute for those runtime gates. Strict
`rtol=1e-5, atol=1e-6` was preserved throughout.

The evidence supports successful exact preparation and full FP64 public
lowering within this scope. It does not support or refute the numerical
reliability of FP64 derivatives, prove an original-FP32 derivative wrong,
admit a closed execution candidate, establish global Fortran branch coverage
or demonstrate complete-update throughput. Return to the speed workstream
with this reference uncertainty explicit; any further reference experiment
requires a separately bounded scientific decision.

## Artifacts

Local compact artifacts are under
`outputs/diagnostics/shared-state-reference-fp64-20261002/`: both model JSONs,
actual exits, executed harness/policy/launcher sources, logs, identity pickles,
mask NPZs, the preflight and `cpu-independent-verification.json`, with its
NumPy-only `verify_saved_arrays.py`. `artifact-manifest.json` records their
locations/sizes and the remote-only graphs. The first and repaired StableHLO
files remain in the same family under the project's remote runtime outputs,
63,417,461 and63,413,105 bytes respectively. No completed stages were
overwritten. There are no 32-call result pickles or executable to collect.
