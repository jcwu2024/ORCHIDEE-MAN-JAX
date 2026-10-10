# Expert5 Whole-Chunk Vmap Daily-Loss Screen

Status: the parameter-only daily screen passed on 2026-10-01; full daily
parameter-plus-initial-state AD was rejected on strict state-gradient parity.
This is exploratory execution evidence only. The original screens ran no
optimizer update; the 2026-10-03 offline review below adds actual official
AdamW component proposals without adopting the failed state-gradient route.

## Scope and implementation

The treatment changes only the chunk-axis mapping primitive for the exact
expert5 route with 7,542 rows: canonical 128-row packing produces 59 chunks
and 10 trailing padded rows, then `jax.vmap` evaluates the original chunk
function. Padding, original masks and indices, decoder projections, both
384-wide/eight-head coordinate blocks, readout and active-coordinate masking
are retained. The result flattens the 59-by-128 values and restores the
original 7,542-row shape. The outer four-sample vmap remains unchanged.

The canonical chunk function is the one in
`research/daily_coarse_graining/daily_operator/shared_decoder_macro_packed.py`
(`_decode_mapped_rows`, lines 25-76): it calls
`shared_transformer._run_coordinate_blocks_reusing_projections`, the two
readout `_linear` calls, and its active mask. The independent adapter in
`scripts/hpc/shared_chunk_vmap_execution.py` reproduces that packing and
readout, replacing only `jax.lax.map(decode_chunk, values)` with
`jax.vmap(decode_chunk)(values)`. Other expert shapes fall through to the
canonical helper unchanged. It adds no custom AD, parameter replication,
normalized-key or union bank, learned-shape map, alternate loop, or new loss.

The unique wrapper reuses the saved public-loss runner and admission utilities;
it only tags the machine-readable report with the V4 variant and source hashes.
The exact source difference is recorded by the SHA256 fields in the result JSON.
Neural precision remains FP32, physical precision remains FP64, and the
unchanged comparison tolerance is `rtol=1e-5`, `atol=1e-6`.

## Numerical result

The screen uses the original four distinct daily samples, update-0 checkpoint,
the unchanged public daily objective, and native `jax.value_and_grad` over all
64,884,481 trainable parameter coordinates. The decoder readout is included.
The preserved context identity records sample rows `0, 1, 2, 3` and their
individual hashes in the result JSON. No optimizer update was performed.

| Comparison or check | Result |
| --- | ---: |
| Public daily scalar loss | Pass; 1 coordinate, max abs `1.481e-8` |
| Auxiliary output | Pass; 160 coordinates, max abs `2.899e-7` |
| All parameter gradients | Pass; 64,884,481 coordinates, max abs `1.341e-7`, relative L2 `3.705e-7` |
| Nonzero parameter-gradient coordinates | 64,862,209 of 64,884,481 |
| Hard physical counters | 11 counters, all zero |
| Candidate finite/physical admission | Pass |

This public-loss result does not rerun or clear the separate synthetic
component test: its whole-vmap VJP comparison still fails the unchanged strict
gate. The public-loss screen does not return state gradients, establish T7 or
T30 parity, measure a complete training update, or certify full-network
training acceptance.

## Runtime and memory

The node scheduler snapshot was `ibc13b02n04 State=IDLE`, `CPUAlloc=0`, empty
`AllocTRES`, and no queued/running Slurm jobs on that node. Immediately before
launch GPU1 was 0 MiB and 0% with no compute PID. GPU0 already had another
process and was not used. At process start the host load was `1.00, 1.02, 0.96`
and CPU affinity was `0-111`. The run used one free V100, OMP=4, the immutable
`shared-split-width-v1` checkout at `c599236bcf57b745e9bafed678298010f4c3b58f`,
the existing JAX 0.4.38 GPU environment, and the already-passed legacy-runtime
preflight. No Slurm job or paid resource was used.

| Measurement | Canonical field baseline | Whole-chunk vmap |
| --- | ---: | ---: |
| Candidate lower time | Existing binary reused | 101.564 s |
| Candidate backend compile | Existing binary reused | 719.661 s |
| First result call | 14.331 s | 0.393 s |
| Seven-pair warm median | 0.375510 s | 0.221913 s |
| Estimated live bytes | 11,344,311,092 | 7,445,705,660 |

The resulting field-over-candidate warm median ratio is `1.692144x`. The seven
alternating measurements were:

- Field: `0.378884, 0.375888, 0.373242, 0.376294, 0.373342, 0.375510, 0.373383` s.
- Candidate: `0.221628, 0.223036, 0.221336, 0.222895, 0.221704, 0.222855, 0.221913` s.

Candidate observed peak device allocation was `11,362,754,816` bytes
(`10.836 GiB`), below the unchanged 24-GiB gate. The launcher wall time was
`873.501 s`, or `0.242639` GPU card-hours. This bounded screen improves on the
previous shape-preserved public-loss candidate median (`0.267035 s`, `1.40389x`
versus field) while retaining the prior synthetic failure. These timings do
not meet or establish the separate order-of-magnitude full-training target;
the 719.7-second candidate compile remains a material cost.

## Execution records

The target-shell `--dry-run` expansion passed before execution. Command:

```bash
RUN_ID=20261001-v2 bash \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-chunk-vmap-daily-20261001/run_shared_chunk_vmap_daily.sh \
  expert5-micro4
```

The v2 run exited 0 with `status=passed`, `stage=completed`, and total runner
elapsed time `867.820 s`. The v1 attempt is also preserved; it exited before
lowering because the adapter imported the macro helper before the candidate
module registered it. The adapter was corrected to resolve the canonical
module at patch entry, and v2 used a new output stem.

Local machine-readable report:
`outputs/diagnostics/shared-chunk-vmap-daily-20261001/` contains v1 and v2 JSON
status reports and launcher records. V2 JSON SHA256 is
`75e90934cfe7e7e176b270d9441bdb494ef18272c3131bde217f953941102e91`.

Remote full artifacts remain under
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/` with stem
`shared-chunk-vmap-daily-expert5-micro4-20261001-v2`: the reference result,
candidate result (including the full parameter-gradient tree), serialized
candidate executable, JSON report and launcher record are retained. No
additional long validation or production training was run.

## Daily Dual-AD Supplement

Status: **V4 daily dual-AD parity failed** on the same original four rows.
Loss, auxiliary outputs, the complete parameter gradient and physical gates
pass; the explicit initial-state gradient fails the unchanged strict tolerance
and rejects strict V4 daily dual admission. The original screen ran no
optimizer update. The later A/B review preserves that rejection while testing
discarded optimizer proposals; production adoption remains absent.

The saved executable ABI is `(parameters, initial_states, daily_samples)`.
`initial_states` is the original `daily_samples.inputs.continuous_state`, shape
`(4, 3854)`. The helper replaces only that leaf in the dynamic sample batch,
then calls the unchanged `make_microbatch_objective` wrapping
`make_shared_joint_daily_objective`. Native `value_and_grad` differentiates
only `argnums=(0, 1)`; every forcing, target, mask and other sample leaf stays
dynamic. The result is `((loss, auxiliary), (parameter_gradients,
state_gradients[4,3854]))`. The state gradient is not an optimizer input. The
binary signature records 891 input leaves and 462 output leaves, allowing the
same ABI to represent distinct four-row slices without capturing one fixed
sample pack.

The original-field reference compiled the admitted expert128 adapter with
backbone source reuse, without V4 vmap. Its loss, auxiliary values and all
64,884,481 parameter-gradient coordinates bridge to both the saved V4
parameter-only field executable and result. The candidate adds only the V4
whole-chunk-axis vmap adapter. A fresh in-process replay of the saved dual
reference executable is the candidate comparison baseline; saved-reference
state self-drift is diagnostic and does not gate admission.

| Gate | Result |
| --- | ---: |
| Loss | Pass; max abs `3.7652e-8` |
| Auxiliary | Pass; 160 coordinates, max abs `2.3766e-7` |
| Parameter gradient | Pass; all 64,884,481 coordinates, max abs `1.4901e-7`, relative L2 `3.4855e-7` |
| Initial-state gradient | **Fail**; 27 of 15,416 coordinates exceed strict elementwise tolerance; relative L2 `1.3387e-6` |
| Finite/nonzero gradients | Pass; 64,862,209 parameter and 15,334 state coordinates nonzero |
| Physical hard counts | Pass; all 11 counters zero |

The maximum raw absolute initial-state-gradient difference is sample 1/state
coordinate 1785: candidate `-628365.73466`, field `-628364.53498`, absolute
error `1.1996839`. Its elementwise tolerance is `6.2836583`, so this coordinate
passes the relative tolerance and is not the largest failing coordinate. The
largest failing coordinate is sample 0/state coordinate 1883: candidate
`11842.33414`, field `11842.50814`, absolute error `0.1739963` against an
allowed `0.1184261`. In seven alternating full comparisons, loss,
auxiliary and all parameter gradients passed all seven pairs; state gradients
failed all seven. Per-pair state failure counts were `26, 29, 27, 25, 28, 26,
26`, with 34 unique failing sample/state coordinates and 20 failing in every
pair. The repeat-persistent sample 3/state 1785 coordinate differed by
`0.828-1.285` (field range `44365.524-44365.981`, candidate range
`44364.467-44364.981`). The 2026-10-03 exact recomputation corrects the old
claim that this coordinate was self-stable in both arms: it is absent from
the coordinates passing every self comparison in both arms. State self-drift
also remains present:
candidate repeat counts versus its first result were `3, 8, 4, 4, 5, 2, 6`;
field repeats versus the fresh replay were `10, 7, 4, 11, 9, 7, 6`. The
unchanged strict tolerance was not relaxed.

Warm medians were `0.382541 s` field and `0.228810 s` candidate (`1.671871x`),
but this speed result is diagnostic only because state parity failed. Estimated
live memory was `11,348,361,292` bytes for the field reference and
`7,436,589,716` bytes for V4. Observed peak was `11,366,804,992` bytes, below
the 24-GiB gate. Reference compile/lower/runner times were `859.428/102.997/
993.370 s`; candidate times were `845.197/112.827/1057.050 s`. Launcher
records report `0.278075` reference and `0.297203` candidate GPU card-hours.

The deployed synthetic runtime preflight passed on GPU (`cuda:0`) in 23.3 s,
with no Teacher data read. The dynamic-sample helper test also passed in the
local `ORCJAX` environment, proving distinct forcing/target packs change loss
and gradients through the same compiled three-argument ABI. Target-shell
`bash -n` and `--dry-run reference` / `--dry-run candidate` passed. The
candidate JSON records `status=parity_failed`; its retained launcher record
has `exit_status=0` because the original probe entry point did not propagate
the report status. The local probe entry point is corrected to return nonzero
for any completed non-pass report; the executed remote source snapshot and
exit-0 launcher record remain preserved. No model graph was rerun for this
entry-point correction.

Remote full artifacts are under `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/`
with stem `shared-chunk-vmap-daily-dual-v4-20261001`: preflight JSON,
reference/candidate JSON and launcher records, both serialized executables,
both full result trees, the fresh field replay, and all fourteen repeated raw
state-gradient vectors are retained. Local reports and raw diagnostic trees
are under `outputs/diagnostics/shared-chunk-vmap-daily-dual-v4-20261001/`.
The executed remote dual probe SHA256 is
`460f4172e94af9a6de0420508f7d265c5d9b7ac9b6b689f8c7c2c8c9aa0b3d80`; helper
SHA256 is `b1d26466425748f5801184420aafdfdd0d99af25cc25b4616a65b2d4ebf36f52`.
The field expert adapter, backbone adapter and V4 adapter hashes are recorded
in each JSON report. This dual result does not waive the separate synthetic
component failure or establish T7, T30 or complete-update admission.

Graph lookup was unavailable (`Transport closed`); this report relies on the
bounded source paths above and the immutable executed snapshot. It makes no
exhaustive branch-coverage claim.

## Saved-array A/B review, 2026-10-03

A means public loss, full parameter gradients, actual AdamW proposals and
physical execution. B means complete initial-state derivatives, repeat
stability and internal recurrent cotangents. A and B remain separate claims;
passing A does not clear failed B or establish internal recurrent sensitivity.
The exact saved-tree comparison preserves structure, shape, dtype and the
two matching static `None` parameter nodes, alongside all 300 numeric leaves
and 64,884,481 coordinates. Lightweight controls reject `None` mismatch,
missing keys, shape changes and a strict numerical discrepancy.

Recomputation retains the loss/160 auxiliary/full parameter pass and all seven
full daily paired passes, max parameter error `1.4901161193847656e-7`. It
retains state failures `27` initially, `[26,29,27,25,28,26,26]` across repeats,
34 coordinates in the union and 20 failing every pair. **18** coordinates,
rather than all 20, have disjoint repeat intervals separated beyond the largest
strict allowance. Both-arm self-pass examples include sample0/state177
(gap `4.202127456665039e-6`, allowance `2.1963654309511185e-6`) and
sample1/state428 (gap `3.6858615966472996e-5`, allowance
`1.0420004660950411e-5`). The original `1.671871x` dual speed is retained.

On free CPU test01, official immutable-source `fresh_start_training.make_update`
467–472 / `target_oriented_training.target_oriented_adamw_update` 371–426
applied the two saved daily micro4 gradients separately to the **real**
update0 parameters and AdamW state. These are fixed-weight component proposals,
not daily32/T7/T30 mixed updates. All strict comparisons passed: updated
parameters max abs `3.03611159324646e-7`, complete optimizer max abs
`1.3969838619232178e-9`, and raw norm/clip checks. The optimizer is
beta1 `.9`, beta2 `.999`, clip1, epsilon `1e-8`, learning rate `1e-5`, decay0;
the actual step advances 0→1. It contains 601 numeric leaves/129,768,963
coordinates and four matching static `None` nodes. Four complete daily/T7
component proposals were preserved remotely before comparison. CPU JAX0.4.38
compiled only the official update helper (`13.775395 s`); there were zero model
evaluations, model AD compiles, GPU calls or retained training updates.

The actual saved prepared states establish masks through canonical
`input_assembly._defined` 52–54 (`isfinite & abs<0.5e20`); no zero-gradient
inference defines the masks. All 82 undefined daily input coordinates have
exact zero derivatives in both first outputs and all fourteen repeated state
arrays; all arrays are finite. The existing 112 source-mask inventory and the
same source/profile/method/checkpoint/artifact742 identity bridge are retained.
Internal rollout masks and recurrent cotangents were not observed. Original
FP32 JVP/VJP disagreement, inconclusive finite differences and self-consistent
FP64 results with changed primal/neural arithmetic do not define FP32 truth,
admit V4, or waive the strict failures.

Compact evidence: `outputs/diagnostics/shared-v4-ab-offline-decision-20261003/`
contains exact tree/interval results, controls, `official-adamw-summary.json`
and `independent-evidence.json`. Full official component proposals and declared
mask arrays remain in the corresponding remote diagnostic directory. The new
same-input six-proposal contract is specified in
`shared_v4_same_input_update_proposal_20261003.md`; its authorized free-GPU
execution uses saved binaries with zero new model AD compilation. Its completed
result follows.

## Complete same-input update diagnostic, 2026-10-03

The authorized ordinary free-GPU run completed exactly three pairs/six
discarded full proposals, with orders reference/candidate, candidate/reference,
reference/candidate. Both arms used identical parameters and AdamW state
within each pair; only the reference advanced the next pair's common input.
Each proposal used all 32 distinct prepared daily rows as eight micro4 leaf
slices, public T7 and T30, the official gradient mix and clipped AdamW. Saved
V4S daily/V7 T7 candidates were compared with original daily/T7 binaries;
both arms used the identical original T30 object. All 60 model calls
(48 daily, six T7, six T30) completed without model AD compilation. No
bootstrap, warm model call, extra pair, retry or training checkpoint occurred.

**A fails overall:** the first two pairs pass all A checks, but pair2's
combined full parameter gradient fails at 977 coordinates across 12 leaves
(`max_abs=1.5232712030410767e-5`, relative L2 `2.396795741362668e-5`).
The saved component-gradient decomposition is unavailable, so the discrepancy
is not attributed to daily, T7, T30, a particular compiler behavior or a
specific internal cotangent. No extra model calls occurred in this closed
1800-second window. Under standing same-method free-diagnostic authorization,
the independent900-second branch diagnostic is specified in
`shared_v4_branch_attribution_proposal_20261003.md`.
All three pairs still pass updated-parameter, complete AdamW, public
loss/auxiliary and norm/clip comparisons. Those passes do not waive the
combined-gradient failure.

| Pair / common input optimizer step | Combined gradient failures | Updated parameter max abs | Optimizer max abs | Daily32 state failures | T7 state failures | Shared T30 state failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / 0 | 0 | `1.0430813e-7` | `9.3132257e-10` | 231 | 8 | 0 |
| 1 / 1 | 0 | `5.9604645e-8` | `1.8626451e-9` | 227 | 2 | 2 |
| 2 / 2 | **977** | `1.5832484e-7` | `1.4202669e-7` | 156 | 123 | 0 |

B fails every pair over complete daily state derivatives (shape8×4×3854,
123,328 coordinates), T7 and T30 derivatives. Shared-original T30 pair1
fails at505 and1911 despite the same binary and inputs. This is within-pair
repeat evidence; changing reference weights across pairs are not self drift.
Unobserved internal recurrent cotangents remain unresolved.

All six proposals pass hard ABI, finite, physical, undefined-state derivative,
mask, optimizer-step and 24-GiB checks. Daily11/T7 14/T30 14 physical counters
are zero. Declared undefined input counts are658 daily32 and21 for each
recurrent case, with exact zero derivatives in every proposal. Peak cumulative
device allocation is `13,231,106,048` bytes. The final report is
`diagnostic_strict_discrepancy_not_adopted`; launcher exit0 means diagnostic
completion, not numerical or scientific admission.

| Complete synchronized proposal time | Reference | Candidate |
| --- | ---: | ---: |
| Pair0, cold | `64.112146 s` | `17.262032 s` |
| Pair1, warm | `13.132996 s` | `11.113097 s` |
| Pair2, warm | `13.073024 s` | `11.259284 s` |

The two warm median times give `1.171355900x`; paired ratios are
`1.181758449x` and `1.161088415x`. Stability is unresolved from two
predeclared observations. These complete timings preserve the daily32/T7/T30
sampling and AdamW workload, and exclude separately recorded host transfer
(`0.344–0.699 s`) and preservation/classification. Raw IO, hard checks and
comparison were not independently timed; the family total includes them.
Pair0 remains separate: 143 helper backend compilations occurred in its
reference proposal and four in its candidate proposal, with zero helper
backend compilations in the last four proposals. The actual shared T30 first
call belongs to reference pair0; candidate pair0 reused that executable after
its first call, so its apparent cold timing cannot establish a speed claim.

Binary load/ABI times were original daily/T7/T30 `14.403494/95.727225/
95.021264 s`, candidate daily/T7 `12.766452/96.651626 s`. Startup before the
first proposal was `382.386428 s`. Node `ibc13b02n04`, free GPU1, CPU affinity
0–111, host load0.00/0.01/0.05 and no visible GPU processes were recorded
before dispatch. The whole first-Bash-origin family took `578.381442 s`
(`0.160661512` one-GPU card-hours), inside1800s including10s grace; CNY0,
paid jobs0, retained training updates0, production adoption false.

Compact actual evidence is in
`outputs/diagnostics/shared-v4-same-input-update-20261003/`: `model.json`,
three pair records with full per-leaf A and exact failed B coordinates,
six hard records, `actual-exit.json`, declared masks, `decision.json` and
`raw-preservation-index.json`. Six complete raw proposals, each
1,039,261,317 bytes/6,235,567,902 bytes total, were saved remotely before
classification under the corresponding diagnostic `model/` directory.
No large proposal was downloaded. The original fixed-weight daily/T7
speed and offline AdamW evidence remain scoped to their measurements; this
new run does not establish practical A admission, B admission, or the10x
training target.

An independent free test01 CPU/NumPy re-read of all six actual raw proposals
took34.981565s and exactly reproduced every A/B failure count, including
977 failures across12 leaves in pair2. It retained exact failed parameter
indices, values, allowances and parameter-block paths in
`pair-2-failed-parameter-indices.json`; independent comparisons and source
evidence are in `independent-verification.json` and three
`pair-*-independent-comparison.json` files. The actual snapshot passes the
same current tuple to both arms and advances reference only; immutable
functional AdamW returns new trees, and recorded call order/steps agree.
Raw files do not contain both arm input arrays or next-pair inputs, so no
independent byte comparison of those input trees is claimed. `GPU-cleanup.json`
records releasedGPU1, empty contexts and0MiB/0% after this family ended.

Existing warm model markers separate the branch costs without added calls:
candidate T30 median6.727974s is about60.1% of the11.186190s warm complete
proposal median. Pair1 reference daily/T7/T30 times are3.137706/1.757644/
6.726707s, candidate1.913576/0.895035/6.724304s; pair2 reference3.027732/
1.581994/6.730749s, candidate1.810584/0.928017/6.731644s. The remainder
includes instrumentation, slicing, synchronization and gradient helpers;
it is not a pure AdamW measurement and does not attribute the977 failures.

## Fixed-input branch discriminator, 2026-10-03

Under standing same-method free-diagnostic authorization, the independent
900s window completed38 saved-binary calls at the exact old pair2 common
parameters from `pair-1-reference.full-proposal.pkl`, with no update or new
model AD compile. Two alternating repeats used both arms' canonical daily32
mean (eight distinct micro4 leaf slices), T7, and two original T30 calls.
Every raw call was saved before hard/strict checks, followed by complete
daily32 means and exact canonical device mixed gradients. The historical
mixed-gradient977 and A/B rejection remain preserved.

Fresh evidence localizes the parameter-gradient discrepancy to **T7 at this
fixed input**, while daily32 retains full-parameter parity. It does not
identify an internal AD cause or recover absent old component gradients.

| Comparison | Repeat0 parameter failures | Repeat1 parameter failures | State failures |
| --- | ---: | ---: | --- |
| Daily32 candidate vs reference | 0 | 0 | 159 / 157 |
| T7 candidate vs reference | **1357** | **1348** | 124 / 124 |
| Daily32 within reference / candidate | 0 | 0 | 8 / 11 |
| T7 within reference / candidate | 0 | 0 | 0 / 0 |
| Original T30 repeat1 vs repeat0 | 0 | — | 1 |

T7 crosses fail across17 leaves, max abs `1.5135854482650757e-5` and
`1.5109777450561523e-5`;968/965 of the retained old977 coordinates fail in
the T7 component comparison. Daily parameter max abs is `1.7881393e-7`,
and T30 repeat parameter max abs `1.1175871e-8`. All branch loss/auxiliary
comparisons pass. Every arm's daily/T7 parameter self repeat passes.

Canonical `combine_daily_mean` reconstruction using the same T30 for both
arms fails at960/954 coordinates, all belonging to the retained old977.
Assigning T30 repeat0 to reference and repeat1 to candidate with repeat0
daily/T7 still fails at960. Fresh same-arm mixtures pass against the retained
old pair2 mixtures (max abs at most `8.940696716308594e-8`). This fresh
evidence implicates T7 without claiming exact reproduction of977 or a unique
forensic decomposition. Approximate comparison is not transitive; same-arm
fresh/retained passes do not clear cross-arm failures.

All42 hard records (38 native calls plus four daily means), parameter300/
64,884,481/two-None ABI, complete state shapes, physical counters and
undefined derivative zeros pass. Post-helper cumulative peak is
`12,201,233,152` bytes (11.363284GiB), below24GiB. The window finished in476.695342s/
0.132415373 one-free-GPU card-hours/CNY0, inside900s including10s grace,
with143 helper backend compilations and zero model AD compiles. GPU1 was
released; `GPU-cleanup.json` records empty contexts and0MiB/0%.

Local compact evidence is in
`outputs/diagnostics/shared-v4-branch-attribution-20261003/`: full per-leaf
branch/mixed comparisons, restricted old977 coordinate values,42 hard records,
`decision.json`, source/launcher accounting and cleanup. The remote `model/`
directory preserves38 complete native result trees, four daily32 means and
six canonical mixed gradients:48 files/12,466,655,086 bytes. No raw arrays
were bulk-downloaded. This is completed attribution evidence with zero
updates/training/adoption; no candidate T30 or additional experiment followed.

Final free test01 CPU verification independently re-read all48 raw files:
ten complete components and six mixed gradients, plus32 daily micro4
results for reconstruction. Full parameter/state/loss/auxiliary cross and
self results reproduce the recorded counts. All four daily mean parameter
trees reconstruct **bitwise exactly** through source-ordered dtype-preserving
sequential addition/division8; all six mixed parameter trees reconstruct
**bitwise exactly** through NumPy `(daily_mean+T7)+T30` without FP64 promotion.
This is measured exactness on these retained arrays, not a general compiler
equivalence claim. Exact failed parameter/state indices and source-block
paths are preserved in the independent records. A CPU auxiliary NamedTuple
path traversal was repaired with its first source retained; completed
component comparisons were reused, and no model/GPU call or tolerance
change occurred. Local `independent-verification.json` and
`independent-*-reconstruction.json` document completion; the repair run took
45.540999s, with its limited error record retained separately.

Stop V7 as a direct replacement and stop candidate T30 expansion. Retain the
daily32 full-parameter passes, update0 component/AdamW evidence and measured
complete-update1.171356x throughput within their scope. No unbounded internal
cause investigation or additional GPU run is required for this rejection.
