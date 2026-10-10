# V4S/V7 same-input complete-update diagnostic

This is the sole bounded execution diagnostic following the saved-array A/B
review. It preserves the failed strict initial-state derivative evidence and
tests practical complete-update behavior. It grants no scientific or production
adoption and retains zero training checkpoints.

The completed CPU offline review retains daily dual-AD `1.671871x` and T7
`1.742263x`. Loss, full parameter gradients and physical checks pass. Official
AdamW applied separately to saved micro4 and T7 gradients from the real update0
checkpoint also passes strict updated-parameter and optimizer comparisons. Those
are fixed-weight, clipped component proposals, not the declared mixed update.
Daily state failures remain27 initially/34 union/20 every pair, with18 disjoint
repeat ranges beyond tolerance. T7 retains four persistent coordinates33,199,
385,428. Source review finds no added custom AD, stop-gradient or changed
mathematical decoder/mask/readout formula. Internal recurrent cotangents were
not observed. Original JVP/VJP disagreement, inconclusive FD and changed-FP64
self-consistency do not supply FP32 truth or waive any failed gate.

## Executable workload

Use immutable `c599236bcf57b745e9bafed678298010f4c3b58f`, the actual update0
parameters/AdamW state and existing prepared artifact742 inputs. The candidate
uses saved V4S daily dual and V7 T7 binaries. The reference uses their original
daily/T7 dual binaries. Both use the same original T30 dual binary. No model AD
graph is compiled or modified. Neural FP32, physical FP64, native checkpointing,
all300 numeric parameter leaves/64,884,481 coordinates, static None ABI, source
masks, loss, sampling, and AdamW rules are retained.

Exactly three predeclared pairs produce six full proposals, in orders
reference/candidate, candidate/reference, reference/candidate. Within **every**
pair both arms receive identical parameters and AdamW state. Only the reference
proposal advances the common input to the next pair. No independent candidate
trajectory, bootstrap proposal, warm model call or additional pair is allowed.

Each proposal executes the official daily32 equal-row mean from eight distinct
dynamic micro4 slices (leaf slicer), plus public T7 and T30, combines their full
parameter gradients without extra averaging, and applies official clipped
AdamW once. This gives48 daily micro4 calls, six T7 and six shared-original T30
calls,60 model calls total. Initial-state gradients are complete diagnostic
outputs and do not enter AdamW. Same-binary T30 comparisons within a pair are a
same-input repeat control; cross-pair weights change and are not self-drift.

## Preservation and checks

Before classification save each complete combined gradient, loss/auxiliary,
updated parameters, complete optimizer and daily/T7/T30 initial-state gradients.
Preserve six raw proposal files (approximately6.23GB total; few filesystem
entries), compact hard records, exact failed state coordinates, full parameter
comparison records, defined/undefined masks and sources. The raw proposal is a
discarded diagnostic result, not a training checkpoint. Record canonical
`input_assembly._defined`52–54, prepared source/mask identity, undefined derivative
zero checks, branch counts and native serialized ABI.

Compare A loss/auxiliary, all parameter gradients, actual updated parameters,
AdamW state, raw norm and clip scale, and B all initial-state gradients using
unchanged `rtol=1e-5, atol=1e-6`. Strict A or B discrepancies remain rejection
evidence and do not suppress later pairs when hard ABI, finite values, zero
physical counters, source-mask/undefined checks,24GiB memory and budget pass.
A hard/resource failure stops after preserving available raw results. Historical
failed reports retain their status; diagnostic loaders validate genuine source,
method/checkpoint/data, A and physical/ABI scope without forging admission.

Record every arm/branch first-call marker, actual shared T30 executable identity,
binary loading/setup, helper compilation, complete synchronized proposal time,
host transfer, and preservation/comparison separately. Pair0 contains cold
effects and does not certify speed stability. Report the last two warm pair
times, ratios and ranges; no invented tight stability gate or extra probe.
Interpret A and B separately; a practical A pass does not certify internal
recurrent sensitivity or erase B failures.

## Exact command and budget

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-v4-same-input-update-20261003/run_shared_v4_same_input_update.sh --gpu 1
```

One free gln01 GPU on `ibc13b02n04`, outside paid Slurm; CNY0. Whole dispatch
1800s including10s kill grace, maximum0.5 GPU card-hour. The first origin is
recorded by Bash before Python imports. This single allowance includes
deserialization (existing original loads13.8/94.9/95.6s plus approximately100s
candidate load), helper setup, six proposals (approximately75s basis), raw IO,
comparison and ordinary pre-model interface repair. It uses the measured
approximately400s original full-family baseline; no arbitrary fragmented stage
budgets or new model compilation budget is added. Resume retains the exact first
origin and is allowed only before any model call or raw proposal. No paid
fallback, precision/architecture change, HLO decompression or bulk hashing.

New source files are `probe_shared_v4_same_input_update.py`,
`run_shared_v4_same_input_update.py` and `.sh`; the shared offline tree comparator
is reused. Preparation checks include CPU metadata/source/runtime evidence,
static None ABI controls, exact six-proposal/reference-only protocol controls,
hard-stop negative control, Bash syntax and actual full/resume dry-run expansion.
Actual results live under
`runtime/outputs/diagnostics/shared-v4-same-input-update-20261003`.
