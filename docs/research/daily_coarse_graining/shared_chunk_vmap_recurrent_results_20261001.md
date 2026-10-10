# Expert5 whole-chunk vmap recurrent AD gate

Status: V4 T7 finished `numerical_failed` on free GPU0. Persistent strict
initial-state derivative differences reject strict recurrent admission.
The 2026-10-03 A/B review and discarded complete-update diagnostic preserve
that failure; no recurrent admission, complete-update acceptance or training
is claimed.
The original-four V4 daily parameter-AD screen independently passed at
1.692144x warm speed; the synthetic component failure remains preserved.
R4's separate T7 failed strict state comparisons, including a persistent
coordinate difference, and is closed for recurrent progression.

## Scope and source

This run reuses `scripts/hpc/shared_chunk_vmap_execution.py`, the exact V2
adapter admitted by the daily screen. Its SHA256 is
`8a3cc2940a23ced4b5e1c3f0ed7093a29005851771db2f95f784d35e4c399a51`.
Only expert5's 7,542-row chunk route changes: the original 59 chunks of
128 rows, ten padded rows, coordinate blocks, readout and source masks are
evaluated through native `jax.vmap`. Other expert shapes retain their
canonical helper. There is no R4 learned-shape wrapper, normalized bank,
custom AD, parameter replication or physical-process rewrite.

Canonical provenance is
`shared_decoder_macro_packed._decode_mapped_rows` lines 25–76 and
`shared_transformer._run_coordinate_blocks_reusing_projections` lines
537–610, with original `_linear`/normalization/FFN helpers in
`strong_baseline_architecture.py` lines 886–899. The public recurrent
objective, native default checkpoint policy, complete initial state,
FP32 neural and FP64 physical boundaries are unchanged.

The run uses the declared update-0 checkpoint and artifact 742 T7 window
from `shared-split-dual-inputs-c599236.pkl`, all 64,884,481 parameter
coordinates/300 leaves, and all 3,854 initial-state coordinates. Immutable
checkout: `runtime/checkouts/shared-split-width-v1`, commit
`c599236bcf57b745e9bafed678298010f4c3b58f`. Graph transport remains
unavailable; source claims are bounded to these exact paths and snapshots.

## Runtime and gates

Node `ibc13b02n04`, free V100 GPU0, canonical existing GPU environment,
JAX/JAXlib 0.4.38, original CUDA libraries, OMP=4, LLVM parallelism false,
original command-buffer threshold. The accepted public runtime preflight
is reused. GPU1 belongs to the independent V4 daily parameter/state-AD
diagnostic and is not used by this run.

Budgets are 360 seconds lowering, 5,400 seconds backend compilation,
7,200 seconds Python process and 7,210 seconds external launcher. These
reuse the source-backed original/normalized T7 compile envelope; they do
not authorize paid jobs. Start: node timestamp 2026-10-01 07:36:13 CST.
Python PID 18720 and timeout PID 18689 have ended; launcher exit status is 1.

The original T7 dual-AD serialized executable is reused with zero model
compilations. Saved loss/20 auxiliary values replay bitwise; all parameter
gradients pass (max 1.490116e-8). One saved original state coordinate drifts
across processes, and this is recorded independently. Candidate comparisons
use a fresh same-process baseline; baseline self noise is not itself a
candidate rejection or a reason to waive a candidate-versus-baseline fail.

Candidate lowering took 146.183800 seconds and backend compilation took
2,527.957784 seconds. Python completed after 2,939.055230 seconds. All 20
loss/auxiliary coordinates pass (max absolute difference 1.464838e-7), as do
all 64,884,481 parameter coordinates (max 9.685755e-8, relative L2
6.576704e-7). The 14 physical hard counts are zero. Gradients are finite and
meaningful: parameter norm 3.864171 with 64,820,737 nonzero coordinates,
and 3,833 nonzero initial-state coordinates. Peak cumulative process
memory is 3,507,481,856 bytes; candidate estimated live memory is
2,730,459,773 bytes against baseline 3,488,289,277 bytes.

Initial state fails at 6 of 3,854 coordinates. Seven alternating paired
state comparisons fail at `[7,7,7,5,7,5,6]` coordinates. Original and
candidate self-failure counts are respectively `[1,1,1,0,1,2,1]` and
`[1,1,1,2,1,1,1]`; these self comparisons remain diagnostics. Warm median
time is 1.578531723 seconds original versus 0.906023597 seconds candidate,
or 1.742263x. This timing does not admit the failed route.

### Raw state interval diagnosis

The failed-coordinate union is `32,33,34,199,385,416,422,428,509`.
Coordinates 33, 199, 385 and 428 fail every paired comparison and have
disjoint seven-repeat ranges beyond the largest strict allowance:

| Coordinate | Range gap | Largest strict allowance | Self behavior |
|---|---:|---:|---|
| 33 | 1.362264e-4 | 4.686329e-6 | Both arms have self noise |
| 199 | 2.684650e-6 | 1.491825e-6 | Both arms pass every self comparison |
| 385 | 2.205372e-5 | 8.240995e-6 | Both arms pass every self comparison |
| 428 | 2.645503e-5 | 1.326488e-5 | Both arms pass every self comparison |

For example, coordinate 199 has original range
`[-0.049182530946,-0.049182183346]` and candidate range
`[-0.049185274225,-0.049185215596]`. These stable, disjoint differences
prevent attributing the result entirely to baseline self noise. The initial
maximum error among **failed** coordinates is 1.589656e-4; the global
maximum 67.595623 is from passing large coordinates and does not describe
the failed-coordinate error. The raw interval artifact retains every
repeat, exact coordinate, range and pair/self result. There is no basis for
a noise-only deterministic execution screen of this result. T30 and the
supplemental seven full-parameter replay are not launched.

Admission requires unchanged `rtol=1e-5`, `atol=1e-6` for the initial full
loss/auxiliary/parameter/state comparison, all seven alternating paired
state comparisons, 14 zero physical hard counts, meaningful finite full
gradients and the 24-GiB memory limit. All self records and raw state
vectors remain diagnostic evidence. The current driver includes strict
paired state records in its final status; it does not mistake initial
parity alone for recurrent acceptance.

If initial and all seven paired state comparisons pass, a saved-binary
replay with zero model compilation must also pass seven complete
parameter/state/loss/auxiliary pairs. It saves per-leaf parameter metrics,
every state vector and complete failed result trees when needed. Only
after these gates may a bounded original T30 dual reference be reused or
built and the V4 T30 candidate tested. Prepared target-shell dry-run
expansions pass, but no T30 process has launched.

Full-update admission preparation names V4 explicitly and requires the
supplemental full paired derivative evidence. Metadata contract checks
confirm that actual failed R4 T7 is rejected, synthetic noisy self records
are allowed only with strict cross comparisons passing, and a single
strict paired state failure or missing full paired evidence is rejected.
These synthetic checks supply no model or scientific acceptance. This was
the historical strict admission contract. The later explicitly authorized
same-input complete-update diagnostic preserves failed A/B evidence and
continues only while hard checks pass; it grants no adoption.

Future recurrent runs make optimized HLO export explicit through
`--export-hlo`; default execution saves the serialized program, numerical
results and timing metadata without producing GiB text. Requested export
uses gzip after the numerical checks/timings. This logging efficiency change
is local preparation only: the executed V4 snapshot remains unchanged.
Its existing HLO is compressed and preserved separately after completion.

## Artifacts and accounting

Public runner and launcher overlay:
`runtime/transfer/shared-command-buffer-loops-20261001`.
Unchanged V4 adapter:
`runtime/transfer/shared-chunk-vmap-daily-20261001/shared_chunk_vmap_execution.py`.
Remote output/log stem:
`runtime/outputs/diagnostics/shared-command-buffer-loops-whole-vmap-T7-default-20261001-v4`.
Executed snapshots, current JSON, dry-run expansions and metadata checks:
`outputs/diagnostics/shared-chunk-vmap-recurrent-20261001/`.

All initial result trees and 14 raw repeat state vectors are retained.
The candidate serialized executable is 212,822,886 bytes, SHA256
`d32765ee553fe19e89912f1a00c0d59a30f1f3616a61356c12881cc2c6031c0f`.
Optimized HLO is preserved as a 751,037,444-byte gzip file; a complete local
readback verifies gzip CRC, 6,506,535,254 decompressed bytes and SHA256
`8bb3ad8edc6093fa6c6228d2f804fabe1a6b51a3eb994173d78a40986a244ca8`.
The remote original remains intact. Verification record:
`shared-chunk-vmap-preserved-model-hlo-verification-20261001.json` in the
local artifact directory. Preservation required zero model compilations.

Completed launcher time is 2,961.714980 seconds / 0.822698606 card-hours.
GPU count is one, paid jobs zero, retained training updates zero. Original historical
compilation is not charged again; completed V4 daily and R4/component/HLO
costs retain their separate records. The failed V4 recurrent timing is
separate from full training throughput and the still unmet 10x research
target. The independently accepted daily 1.692144x evidence remains scoped
to its daily gate; recurrent rejection does not erase that result.

## Saved-array A/B review, 2026-10-03

The saved first comparison retains A loss/19 auxiliary/full-parameter passes:
all 300 numeric parameter leaves/64,884,481 coordinates and two matching
static `None` nodes, max parameter error `9.685754776000977e-8`. All 14
physical counters remain zero. Seven T7 repeats saved state arrays only;
they are **not** seven complete parameter/loss/auxiliary pairs. B retains the
initial six failures, repeat counts `[7,7,7,5,7,5,6]`, nine-coordinate union
and the four persistent, disjoint coordinates33,199,385,428 above. The
`1.742263x` saved dual timing remains diagnostic.

Official AdamW on free CPU test01 applied the original and candidate saved
T7 gradients separately to actual update0 weights and moments, using unchanged
`make_update` 467–472 / `target_oriented_adamw_update` 371–426 in immutable
c599236. The updated parameters, complete optimizer, raw norm and clipping
all pass strict comparison: max abs `6.016343832015991e-7` for parameters and
`2.444721758365631e-9` for optimizer. Actual step is 0→1, with beta1 `.9`,
beta2 `.999`, clip1, epsilon `1e-8`, learning rate `1e-5`, decay0. Complete
optimizer ABI is 601 numeric leaves/129,768,963 coordinates plus four `None`
nodes. These are fixed-weight T7 component proposals, not official mixed
daily32/T7/T30 proposals or a training trajectory. All raw proposals were
saved before comparison. No new model evaluation or model AD compile was
performed; CPU JAX0.4.38 compiled only the official optimizer helper.

The actual prepared T7 input has 21 undefined coordinates by canonical
`input_assembly._defined` 52–54 (`isfinite & abs<0.5e20`). Derivatives at all
21 coordinates are exactly zero in both first outputs and all fourteen
repeat arrays; all derivatives are finite. This uses declared masks, without
inferring undefined state from zero derivatives. Source/profile/method/
checkpoint/artifact742/T7 identity bridges connect existing independent
records and retain the 112 source-mask inventory. They do not observe hidden
recurrent masks or internal adjoints.

Independent original FP32 JVP33/428 is repeat-stable yet fails VJP comparison;
199 passes first and fails at the final borderline repeat, while control200
passes. Existing finite differences leave unresolved bands/null/inconclusive
evidence. FP64 JVP/VJP is self-consistent but changes neural arithmetic and
primal values, including coordinate33's sign. It does not supply unchanged
FP32 truth, certify V4, or waive B failures. Practical A evidence and unresolved
internal recurrent sensitivity remain distinct.

Machine-readable controls, exact tree/interval comparisons, official optimizer
and independent identity/mask records are in
`outputs/diagnostics/shared-v4-ab-offline-decision-20261003/`. Full official
proposals and declared mask arrays remain in the same remote family. The sole
new execution contract is `shared_v4_same_input_update_proposal_20261003.md`:
three same-input pairs, reference-only advancement, saved V4S daily/V7 T7
and shared original T30 binaries, strict A/B failures retained, zero model AD
compiles and zero training/adoption. Actual completed results follow.

## Complete same-input update diagnostic, 2026-10-03

The one authorized ordinary free-GPU diagnostic completed all three
predeclared pairs/six full proposals. Within every pair the candidate and
reference received identical parameters/AdamW state; only reference advanced
the next common input. The official daily32 mean (eight distinct micro4
leaf slices), public T7/T30 mix and clipped AdamW were retained. Candidate
used the saved V4S daily and V7 T7 binaries, reference original daily/T7,
and both arms the same original T30 binary. Exactly48 daily/six T7/six T30
model calls occurred, with no model AD compilation, extra warm call, retry,
training or adoption.

**Overall A and B fail.** A passes pair0 and pair1. Pair2's combined full
parameter gradient fails at977 coordinates across12 leaves (max abs
`1.5232712030410767e-5`, relative L2 `2.396795741362668e-5`). Updated
parameters, complete AdamW, loss/auxiliary and norm/clip pass all three
pairs. Combined-gradient failures remain failures despite optimizer passes;
component gradient decomposition is not retained and no causal attribution
is established. Largest parameter difference lies in
`shared_transformer/readout_second/kernel`; the per-leaf record preserves
all12 affected leaves without inferring a specific internal cause.

| Pair | A | Daily32 B failed coordinates | T7 B failed coordinates | Shared original T30 B failed coordinates |
| --- | --- | ---: | ---: | ---: |
| 0 | Pass | 231 | 8 | 0 |
| 1 | Pass | 227 | 2 | 2 |
| 2 | **Fail** | 156 | 123 | 0 |

The two T30 pair1 failures are coordinates505 and1911, using the identical
binary and identical input within that pair. Their absolute differences are
`2.130866050720215e-6` and `8.067810057355018e-4`, exceeding allowances
`1.6957443803548812e-6` and `4.5540311028292033e-4`. This control exposes
strict repeat differences and neither clears candidate T7 failures nor
attributes them to the candidate. Cross-pair weights change and are not a
self-repeat comparison. Internal recurrent cotangents remain unobserved.

All six hard records pass finite/ABI/physical/source-mask/undefined derivative/
optimizer-step and memory checks. Every physical counter is zero (daily11,
T7/T30 14 each); all21 undefined recurrent coordinates and658 daily32
coordinates have exact zero derivatives. Peak device bytes13,231,106,048
remain below24GiB. Six complete raw proposals were preserved before
classification (6,235,567,902 bytes total), with no bulk download.

Cold pair0 times are reference64.112146s/candidate17.262032s and remain
separate. Warm reference times13.132996/13.073024s and candidate11.113097/
11.259284s give median ratio**1.171355900x**, pair ratios1.181758449/
1.161088415x. These are complete same-workload synchronized proposals,
including the official optimizer, with zero helper backend compilations in
the four warm proposals. Two observations do not establish speed stability.
Host transfer is recorded separately; raw IO/hard-check/comparison times
were not split. Whole dispatch578.381442s includes startup, execution and
preservation/comparison: one freeGPU1,0.160661512 card-hours,CNY0, inside
1800s including10s grace. First Bash origin precedes Python imports; no
fragmented retry occurred. Node load0.00/0.01/0.05, CPU affinity0–111 and
empty GPU contexts were recorded on ibc13b02n04.

Final status `diagnostic_strict_discrepancy_not_adopted`, exit0 records
successful diagnostic completion while preserving strict rejection. Actual
records are in `outputs/diagnostics/shared-v4-same-input-update-20261003/`,
including `decision.json`, full per-leaf A comparisons, exact B coordinate
evidence, masks, all six hard records, launcher accounting and raw-preservation
index. Full proposals remain in its remote `model/` directory. The companion
daily report gives complete binary-load/first-call/cold-effect details.
Historical1.742263x fixed-weight T7 timing and offline official AdamW passes
remain valid within their scope; this changing-common-input test grants no
practical A or recurrent B admission and does not meet the10x training target.

Independent free test01 CPU re-reading of all six raw proposals reproduced
all three A/B results exactly in34.981565s. Exact977 failed parameter indices
and12 source parameter blocks, complete independent comparisons and executed
snapshot/functional AdamW evidence are retained locally in
`pair-2-failed-parameter-indices.json`, `pair-*-independent-comparison.json`
and `independent-verification.json` in the actual diagnostic directory.
Both arm inputs are bound to the same current tuple by the executed source
and recorded order/steps. Raw inputs were not separately captured, so an
independent input-array byte comparison is not claimed. ReleasedGPU1 is
confirmed by local `GPU-cleanup.json` (empty contexts,0MiB,0%). Existing
warm markers show unchanged candidate T30 at median6.727974s, approximately
60.1% of11.186190s complete warm median. The two daily/T7 gains and T30's
share do not explain the977 gradient failures. Under standing same-method
free-diagnostic authorization, the independent900s fixed-input branch
diagnostic is described in
`shared_v4_branch_attribution_proposal_20261003.md`.

## Fixed-input branch discriminator, 2026-10-03

The independent900s free-GPU window completed exactly38 calls using actual
old pair1-reference updated parameters (the old pair2 common input), original
prepared daily32/T7/T30 data, unchanged arithmetic and the same five saved
binaries. Two repeats alternated arm order and retained full parameter/state
derivatives and loss/auxiliary before checks. There were zero optimizer
updates, new model AD compiles, warm calls, retries or adoption.

**Fresh full-parameter disagreement localizes to T7 at this input.** Daily32
passes full parameter comparisons twice; original T30 full parameter repeats
pass. T7 candidate/reference fails1357/1348 coordinates across17 leaves,
max abs `1.5135854482650757e-5` / `1.5109777450561523e-5`, relative L2
`3.6076070470900985e-4` / `3.602665174451603e-4`. Both arms' T7 full
parameter self repeats pass (max abs reference `4.4703484e-8`, candidate
`2.0489097e-8`), as do both daily parameter self repeats. All branch
loss/auxiliary comparisons pass. T7 cross states fail124 coordinates in
each repeat while both T7 state self repeats pass; daily state crosses
fail159/157 and self repeats8/11. Original T30 parameter self comparison
passes (max abs `1.1175871e-8`) while one state coordinate fails.

The T7 component fails968/965 of the retained old977 parameter coordinates;
daily and T30 component comparisons fail none of those coordinates. Exact
canonical daily accumulation and `(daily_mean+T7)+T30` device reconstruction
give960/954 fresh shared-T30 mixed failures, all inside old977. Assigning
T30 repeat0 to reference and repeat1 to candidate leaves960 failures with
repeat0 daily/T7. Fresh same-arm mixtures pass against retained old pair2
mixed gradients, max abs at most `8.9406967e-8`. Thus the fixed-input fresh
parameter failure is reproduced through T7 and is not removed by the tested
shared/assigned T30 control. The exact old977 count is not reproduced,
historical branch arrays remain unavailable, and no internal AD mechanism
or exhaustive recurrent cause is established. Passing same-arm approximate
comparisons does not imply transitive cross-arm parity.

All42 hard records pass: complete300-leaf/64,884,481/two-None parameter
ABI, complete micro4/daily32/recurrent state shapes, physical/source-mask/
undefined derivative and finite checks. Post-helper device peak
12,201,233,152bytes (11.363284GiB) passes24GiB. Family time476.695342s, one freeGPU1,
0.132415373 card-hours,CNY0;900s first-Bash-origin including10s grace was
respected. Helper backend calls143, model AD compiles0. Final status
`branch_diagnostic_completed_not_adopted`; GPUrelease record confirms
empty contexts and0MiB/0%.

Complete local branch/mixed per-leaf and old977-coordinate evidence,42 hard
records, launcher accounting, `decision.json`, cleanup and preservation index
are in `outputs/diagnostics/shared-v4-branch-attribution-20261003/`. Remote
full arrays are38 native results plus four daily32 means and six canonical
mixed gradients,48 files/12,466,655,086 bytes; no bulk download occurred.
No training, candidate T30, architecture/precision change or further experiment
was launched. Update0 T7 parameter/AdamW passes and its1.742263x timing
remain valid at that original input; they do not admit this advanced common
input or erase the complete-update A/B rejection.

Final independent free test01 CPU verification re-read all48 raw files,
covering all ten complete components, six mixed gradients and32 micro4
calls for daily reconstruction. Every full-parameter/state/loss/auxiliary
cross/self result agrees with the deployed report, with exact failed indices
and parameter blocks preserved. Four canonical daily means and six source-
ordered dtype-preserving NumPy `(daily_mean+T7)+T30` parameter trees match
the saved device outputs **bitwise exactly**. No FP64 host sum, model call,
GPU call, hash sweep or raw bulk download was used. This verifies the retained
interface/arithmetic evidence, not general bitwise equality across other
inputs or compilers. An auxiliary NamedTuple path repair in the CPU-only
reconstructor retains first-source/error evidence and reuses completed
component comparisons; no executed model algorithm or tolerance changed.
The repair run completed in45.540999s. `independent-verification.json`,
independent full branch/mixed comparisons and four-mean/six-mix reconstruction
records are available in the local actual diagnostic directory.

Stop V7 direct replacement and candidate T30 expansion. Retain scoped daily
A and original update0/T7 parameter/AdamW passes and speed evidence, plus the
measured1.171356x complete-update ratio. The fresh T7 full-parameter failure
at the advanced common input is sufficient rejection; no unbounded internal
AD investigation or further GPU run follows.
