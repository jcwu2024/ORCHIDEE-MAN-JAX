# Selected shared latent reader diagnostic

<!-- independent-t7-batch-current:start -->
## Four-window T7 batching, final outcome, 2026-10-09

The earlier independent7200-second free attempt's sole B4 backend compilation
exceeded its bounded deadline and produced no saved executable or identity
sidecar. That free-scope outcome was window-limited, not a numerical, memory or
batching-speed result; B4 cross parity, batching speed and the1.5x screen
remained unmeasured in that free attempt. Saved-only recovery completed the
serial denominator within that same grant. Its loss/auxiliary/parameter repeats
pass, but initial-state repeats fail; the serial report retains
`numerical_failed`. No optimizer update, architecture/precision/loss/sampling
change, daily/T30 graph or adoption occurred in the free scope. Its closed
grant could not fund another B4 compilation.

**Paid replacement result (2026-10-09):** the approved [replacement proposal](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-B4-paid-followup-replacement-proposal.md)
was submitted once as job15672280 and ran on `ibc13b03n01` with1 node,8 CPUs,
1 V100 and64GiB. Runtime preflight passed in71.84s without a real-model check
or Teacher-data read. The single B4 executable and identity sidecar were saved;
compile count is B1=0/B4=1. Lowering took435.176s, backend compilation
7595.274s, and serialization/save29.302s. Three alternating full-AD pairs
completed within the approved window.

The stable warm median is **5.025831s serial → 1.967053s B4**, or **2.555005x**
(60.86% lower latency; 155.50% higher throughput) for this fixed-weight,
four-window T7 mean-gradient diagnostic. Pair speedups were3.188x/2.556x/
2.545x; the first serial pair was slower (6.270s) than the later serial pairs
(5.026/5.011s), so the three-pair median is the reported comparison. This does
not measure a changing-weight complete optimizer update.

Strict tolerances remained `rtol=1e-5, atol=1e-6`. Loss, all66,123,713
parameter-gradient coordinates, auxiliary outputs, finiteness and physical/
mask hard checks passed. State-gradient parity failed in9/15,416,9/15,416 and
11/15,416 coordinates across pairs1–3; same-implementation repeat checks also
failed1–4 state coordinates in each measurement. The worst cross-implementation
case was sample3 coordinate434, `hydrol_previous_step_state.profil_froz_hydro_ns`:
pair1 absolute error1.906285e-4 against allowance1.636190e-5 (11.65x);
pair2 1.433173e-4 (8.76x); pair3 1.566017e-4 (9.57x). Other failures include
snow grain, snow depth/liquid water and STOMATE carbon coordinates. These are
material strict state failures; do not classify them as harmless roundoff or
relax the gate. Full failed-coordinate values and ranges are in the three
strict-comparison JSON files. Across the paired values, sample3 coordinates
434,2382 and2621 retain separated B4/B1 value ranges. This supports a measured
cross-implementation difference at those coordinates, but does not identify
the cause or establish scientific harm.

The largest sampled card memory was9939MiB, below the24GiB gate; the result's
JAX observed peak was3,912,754,432 bytes (3.644GiB). The fixed-weight 1.5x
engineering screen passed, but status is `numerical_failed`, adoption and
scientific acceptance are false, and no optimizer/training update ran. The
runner exited1 because strict state gates failed; Slurm records job15672280
`FAILED`, exit1:0, elapsed02:22:02. The final GPU snapshot showed0MiB,0%
utilization and no processes.

The paid ledger records8518.111 GPU/helper seconds and a runner-elapsed
allocation-rate estimate ofCNY6.5314. Slurm's final elapsed02:22:02 (8522s)
gives an allocation-rate estimate ofCNY6.533533; including the prior failed-job
reserve of120s/CNY0.092 gives an estimated cumulativeCNY6.625533 (CNY6.63
rounded) against the approvedCNY8.28 ceiling. Explore1000 center billing is
not independently available, so these are estimates, not a final invoice.
The original failed job15666049 and
all its records remain separate. No retry, requeue, second compile, new graph,
gate change or training update is authorized by this completed diagnostic.

The recovered compact evidence is under
[`outputs/diagnostics/shared-latent-reader-20261004/t7-b4-replacement-15672280/`](../../../outputs/diagnostics/shared-latent-reader-20261004/t7-b4-replacement-15672280/).
Full result arrays and the saved executable remain preserved under the remote
runtime output stem; the deployment, approval, command, complete chronology,
and release evidence are in the [readiness record](../../../outputs/diagnostics/shared-latent-reader-20261004/latent-reader-T7-batch-20261009-paid.replacement-readiness.md)
and [submission receipt](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-B4-paid-replacement.submission.txt).
The fixed-weight diagnostic does not change the formal update's daily32/eight
micro4 + one T7 and one T30 data count or loss mix.

### Inputs and unchanged mathematical boundary

All four windows come from existing prepared TRAIN spatial/temporal artifacts,
each first7-day window with its original day-start3854-coordinate state. Dates
use the declared365-day noleap calendar; row bounds below are zero-based and
stop-exclusive. Teacher day keys are pairwise disjoint. No Teacher data was
regenerated or newly prepared, and no validation/test window or duplicate copy
was introduced. The first two windows match the retained B1 input bundle byte
for byte; complete hashes and five-argument geometry are in assets-readiness.

| Sample | Artifact | Landpoint | Year, rows | Model dates |
|---|---:|---|---|---|
|0|742|171.0-095.0|1991,225:232|1991-08-14–20|
|1|275|293.0-093.0|1970,178:185|1970-06-28–07-04|
|2|482|297.0-081.0|1971,105:112|1971-04-16–22|
|3|299|345.0-069.0|1967,202:209|1967-07-22–28|

Both paths use the same saved66,123,713 latent32x384/private-reader64x2
parameters, field64 execution, guarded source turnover, FP32 neural/FP64
physics, full recurrence, public checkpointed objective and all physical masks.
B4 computes `value_and_grad(mean(vmap(public objective)),argnums=(0,1))`, with
one shared355-leaf parameter gradient, four state gradients scaled1/4 and
unreduced per-sample auxiliary outputs. The serial denominator includes four
saved B1 complete parameter/state AD calls followed by device-side loss and
parameter mean, auxiliary stack and state stack/4, all synchronized. CPU shape
readiness passed: eval_shape466.753019s, elapsed521.953334s. Short independent
CPU fixtures verify mean/state1/4/full parameter AD and masks; runner unit tests
pass7/7, with asset-owner checks retained separately. All four saved B1 ABIs
were verified on the actual CUDA runtime.

### Serial measurement and strict numerical result

Recovery loaded the existing B1 executable in186.612037s and compiled only the
small device merge helper. Cold individual calls14.677812/1.251077/1.288856/
1.245033s and cold four-window18.512022s are startup evidence, not the warm
denominator. Three full four-window warm measurements, including merge, are
5.247294180/5.024243169/5.036806351s: median **5.036806351s**, equivalent to
**0.794154018 trajectories/s** or **5.559078124 model-days/s** on one V100.
Host transfer/preservation and strict comparisons occur after speed timing.
These are serial self measurements; no alternating B1/B4 pair exists.

Per-window cold losses are0.02362536586305439,0.05020227304469185,
0.016085619908642164 and0.24504535941259684. Cold merged and all three warm
losses are0.0837396545572463. Each repeat passes loss1, auxiliary76 and all
66,123,713 parameter coordinates at unchanged rtol1e-5/atol1e-6. Every value is
finite; full state AD has shape4x3854, nonzero counts3833/3834/3834/3834 and
all14 hard physical/mask counts0 per sample. Strict state repeats versus the
preserved cold merged result fail1/5/4 coordinates out of15,416. Sample index
below follows the table above; coordinate is the full state index, after1/4.

| Repeat | Sample, coordinate | Binding | Absolute difference | Allowed |
|---|---|---|---:|---:|
|warm1|1,509|hydrol snowgrain[0,2]|3.769994e-6|3.236544e-6|
|warm2|0,508|hydrol snowgrain[0,1]|6.496906e-6|5.249587e-6|
|warm2|1,509|hydrol snowgrain[0,2]|5.677342e-6|3.236544e-6|
|warm2|3,33|diffuco snowgrain[0,0]|1.394749e-5|1.303102e-5|
|warm2|3,434|hydrol profil_froz_hydro_ns[0,5,3]|3.816187e-5|1.636197e-5|
|warm2|3,2621|slowproc_stomate deepC_peat[0,3,1]|1.065433e-6|1.036781e-6|
|warm3|3,33|diffuco snowgrain[0,0]|1.549721e-5|1.303102e-5|
|warm3|3,182|enerbil snowdz[0,2]|1.396053e-6|1.060116e-6|
|warm3|3,434|hydrol profil_froz_hydro_ns[0,5,3]|7.063895e-5|1.636197e-5|
|warm3|3,2621|slowproc_stomate deepC_peat[0,3,1]|1.139939e-6|1.036781e-6|

Complete actual/reference values, full leaf comparisons and failure ranges are
saved in the serial report; every full array was preserved before its gates.
Existing original/candidate state-repeat failures from earlier component work
remain separate and are not waived. No scientific accuracy or AD admission is
inferred from finite loss, passed parameter repeats or the serial timing.

### Compile phase and memory limits

Original B4 job entered lowering at elapsed229.290004s and backend-budget screen
at618.358762s (stage interval389.068758s, not a saved independent stopwatch).
The backend gate compared5868.641311 available seconds against historical B1
5543.315643s and protected600s for completion. Backend started618.373414s;
deadline report elapsed6487.594409s, phase `backend_B4`, `incomplete_timeout`.
Actual stage interval5869.220995s gives only a lower bound on required compile
time; it does not supply the minimum additional allowance or completion ETA.
No executable, sidecar or reusable lowered artifact was written. The retained
`executable.pkl.compile-attempt.json` prevents silent retry in this scope.

The exact readiness identity already used JIT compiler options
`xla_gpu_enable_llvm_module_compilation_parallelism=true` and
`xla_gpu_force_compilation_parallelism=4`; these were not merely missing launcher
flags. Original process environment was gone before inspection, so its full
XLA_FLAGS is not independently known. Recovery live environment had no matching
XLA_FLAGS line; that is not retroactive evidence about the original process.
NVLink trace contains only a version probe, no cubin-link completion evidence.
The last original snapshot recorded RSS43,590,364KiB,103% cumulative CPU and
GPU1 4277MiB/0%; this does not isolate the internal compiler bottleneck. Original
start host load0.28/0.45/0.24 and recovery start42.07/42.18/42.29 are separate
snapshots; later load cannot establish past contention.

Known B1 compiled argument264,820,773B/output264,528,792B/temp663,256,040B,
alias0 give planned live1,192,605,605B. B4 input271,357,572B/output264,618,556B
and4xB1-temp heuristic3,189,000,288B are planning only; B4 compiler workspace/
tape peak, actual compiled memory and runtime memory are unmeasured. Serial
recovery JAX allocation peak2,282,799,872B (~2.126GiB) and sampled card-total
NVML peak6327MiB both pass24GiB. Original timeout sampled card peak4277MiB
does not prove eventual B4 memory admission. Card-total, allocator and
compiler-planned numbers measure different quantities.

### Settled resources and reproducible artifacts

The same independent ledger contains preflight shell failure0.040477s,
successful preflight80.029821s, original compile occupation6492.521513s/exit124
and serial-saved recovery322.373034s/exit0. Total **6894.964845410 GPU seconds**
(1.915268013 free card-hours/CNY0), remaining305.035154590s unused. Both own
guardian/model sets were absent at final release check; GPU1 was0MiB/0% and
the compute-process query empty at remote2026-10-09T01:38:11+0800. Own log
watcher was closed. No unrelated process was terminated. The free scope is
closed; no old budget or new paid allocation was borrowed.

Remote stem is `runtime/outputs/diagnostics/latent-reader-T7-batch-20261008`
under `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`. Compact local evidence under
`outputs/diagnostics/shared-latent-reader-20261004/` includes suffixes
`.assets-readiness.json`, `.cpu-readiness.json`, `.runtime-preflight.json`,
`.compile-and-compare.json`, `.serial-saved.json`, `.occupation.json`,
`.executable.pkl.compile-attempt.json`, `.final-release.txt`, launcher/NVLink
logs and actual target-shell dry-run receipt. Memory/process CSV names start
`latent-reader-T7-20261005-batch-20261008-{compile-and-compare,serial-saved}`.
Full cold individual/merged and warm1–3 NumPy result pickles remain remote
under the stem; the saved B1 graph remains the paid-replacement candidate asset.
A small test01 CPU-only pickle inspection produced `.serial-saved.scalars.json`
with zero model/GPU calls. Code checkout remains immutable `c599236` with the
declared latent overlay; no running code was changed during either occupation.
<!-- independent-t7-batch-current:end -->

<!-- saved-t30-hotspot-current:start -->
## Saved T30 hotspot diagnosis completed, 2026-10-08

The short authorized loop obtained actual GPU traces for both corrected saved
T30 graphs. It produced no model optimization patch or new speed gain. The only
specific equivalent candidate considered, removing structural padding from the
axial tokenizer pointwise FFN, is not selected for implementation: its identified
GEMM bucket is too small to justify another compilation/validation loop on the
current evidence. Daily/full-update preparation remains deferred. Its structure
review recommendation was followed by the separately authorized independent
T7 batching experiment reported above; NEXT_STEPS controls the next action.
This trace grants no permission to reduce widths or a measured speed forecast.

### Actual execution and trace measurements

Both architectures reuse the existing artifact742/day69–98 input, their own
unchanged saved weights, corrected turnover, FP32 neural/FP64 physics and native
complete parameter/initial-state AD. Each executed one cold, three unprofiled
warm and one profiled call: ten complete AD calls, zero full-model or local
module compilations and zero weight/optimizer updates. Unprofiled warm times are
original6.683036/6.671471/6.680751s and candidate5.208632/5.204718/5.211331s.
These reproduce the prior approximately1.28x component result; they are not a
new optimization gain or an alternating-pair complete-update comparison.

The percentages below use **summed durations of captured device events**;
they do not partition a full update or unprofiled wall time. Device interval
union, span and gaps are separate quantities.

| Captured trace metric | Original | Candidate |
|---|---:|---:|
| Device events |709023|618608|
| Device-event span s |4.075940|3.194918|
| Active interval union s |3.590773|1.858875|
| Gaps inside that span s |0.485167|1.336043|
| Fused kernels, summed-device-time share |55.36%|58.78%|
| GEMM, summed-device-time share |37.38%|35.34%|
| Transfer/set, summed-device-time share |4.98%|4.90%|
| Events at most10 microseconds, count share |90.40%|98.92%|
| GEMM device events |69288|87503|
| Profile main-call annotation s |7.508983|6.855661|

Candidate representative FP32 shapes include M64/N64/K64 (16721 events,
0.104626s), M64/N256/K64 (12492,0.077944s), M64/N64/K256
(8304,0.068456s), batched B8/M8/N64/K32 (8154,0.046925s) and
B8/M32/N64/K8 (8155,0.038695s). The shared large M1373/N12288/K768
bucket has30 events/0.053088s. These support many narrow reader/attention
operations mixed across forward, AD and recomputation. Device event counts
are not unique HLO invocation or host launch counts.

The candidate cuts arithmetic substantially while preserving fine execution
granularity: total event count falls only12.75%, GEMM event count rises26.29%,
and nearly99% of captured events last at most10 microseconds. Short operations
and observed gaps are a plausible reason arithmetic savings do not translate
proportionally to wall time, but this trace does not establish an exclusive
launch-overhead fraction or a causal full-update decomposition.

Coverage limits remain material. CUPTI reports that per-thread activity buffers
are unavailable and collection overhead may be large; it does **not** report
confirmed event loss. TensorFlow profiler hooks are unavailable, while GPU trace
collection succeeded; no environment installation was made. The device span is
shorter than the main-call annotation, so complete timing coverage is unproved.
Explicit host launch API matching found0 events: that share was not obtained.
Source hints leave53.88%/49.64% of summed device time unmapped. The
`strong_baseline_architecture.py` hints account for37.21%/38.20%, largely at
the generic `_linear` line887, and cannot all be assigned to tokenizer/backbone.
Neither pure reverse versus recomputation nor retained-physics exclusive time
has been isolated. Fusion/source hints cannot close those gaps.

### Correct field64 arithmetic ledger and local decision

The actual candidate executes161 field maps/265 chunks,9046 padded reader rows
and8550 eligible rows per sample/day. Its source-visible forward dense/dot MACs
are23,917,041,024 versus original expert12852,551,472,768. The candidate's
tokenizer7,317,450,624, input/source-reuse backbone13,738,905,600 and
bridge332,955,648 sum to21,389,311,872 MACs/89.43%. T30 forward totals
717,511,230,720 MACs. This excludes norm/GELU/softmax, gather/scatter,
physics/loss, AD/recomputation and startup; **89.43% is not a measured time
share**. The old expert128 candidate estimate is not the field64 ledger.
Common expert latents and bank K/V are already cached in source. Different
experts' live axis values cannot be merged merely because their shapes match;
source placement alone also does not prove absence of compiled AD repetition.

The one local candidate is `_factorized_axial_tokens` pointwise FFN at
`strong_baseline_architecture.py` lines1085–1088. Its fixed layout has4032 rows
(7 experts x18 axes x32 extent),980 physical rows and3052 structural padding
rows. Gathering all physical rows before the row-local FFN and restoring the
original shape/mask is mathematically equivalent; dynamic false-mask physical
coordinates must still be retained. Changed contractions would require strict
local forward/full parameter/physical-state AD validation before adoption.
No such patch or local GPU probe was run.

The identified4032/384/1536 FP32 GEMM bucket is134 events/0.047243543s/
1.315685% of captured original device time and68 events/0.023895653s/
1.285489% for the candidate. Making that **entire identified bucket** free
would give approximately1.0130x captured-device-time speed; scaling only its
arithmetic by the75.6944% padding share gives approximately1.0098x before
gather/scatter overhead. These are conditional bucket estimates, not the full
FFN, T30 wall-time or complete-update ceiling; fusion coverage is incomplete.
They do not justify selecting this small change as the next substantial route.
Historical failed grouping/command-buffer routes are not reopened.

### Numerical outcome, resources and reproducible artifacts

Both profile self repeats pass full forward/gates and all parameter gradients
(original64,884,481; candidate66,123,713). Finite values, all14 hard counts0
and24GiB gates pass. Strict initial-state repeats still fail original1 coordinate
(505/snowdz) and candidate3 (33/snowgrain,200/canopy2ground,2455/carbon_32l).
The recorded JAX allocation peaks are3.267193/1.131028GiB in their respective
processes; these are not card-total or compiler-planned memory. No gate is
relaxed, and no state/science/model/complete-update admission follows.

The independent2700-second free single-V100 ledger records four occupations:
24.106043s (overlay IO bootstrap failure before calls),609.827196s
(full-HLO export stopped with own-wrapper TERM),660.206035s (original trace
saved; candidate HLO index deadline before candidate calls),469.646036s
(saved-only candidate trace/index completed, exit0). Ordinary repairs and
all loading/export/cleanup are included. Total1763.785310 GPU seconds =
29.396422 minutes/0.489940 free card-hours/CNY0; remaining936.214690 seconds
are unused. The final selected GPU0 snapshot was0MiB/0%/no compute PID;
own GPU processes were released. No Slurm submission occurred. This completed
scope is closed; its remainder does not authorize a new experiment.

Code: `scripts/hpc/profile_shared_latent_saved_T30.py` reuses saved graph ABI,
validation and the no-model-compile guard; `analyze_shared_execution_trace.py`
adds optional saved HLO source/shape hints. The existing budget helper is reused
with an isolated scope. Launcher branches passed actual target-Bash dry-run.
Local compact evidence is in `outputs/diagnostics/shared-latent-reader-20261004/`:
`latent-T30-hotspot-20261008.occupation.json`, `.profile-v2.json`,
`.profile-v3.json`, `.profile-v2.original.trace-summary.json` and
`.profile-v3.candidate.trace-summary.json`, plus the executed launcher and
summary tool. Source evidence is in
`outputs/diagnostics/shared-latent-source-ledger-20261008/{field64_source_ledger.json,axis_ffn_scope.md}`.
Raw traces, partial/full HLO metadata indices, logs and full arrays remain under
remote `runtime/outputs/diagnostics/latent-T30-hotspot-20261008*`; successful
T30 binaries remain preserved. Graph coverage had metadata-changed freshness;
the source ledger used bounded direct-source fallback, not exhaustive coverage.

The current evidence supports stopping the axial-padding probe and latent
daily/full-update expansion. It does not prove large equivalent improvements
impossible. Actual wall-time attribution remains incomplete, so MACs alone
justify neither a width choice nor a promised speed gain. The subsequent
independent batching result and current NEXT_STEPS supersede this loop's
forward-looking recommendation.
<!-- saved-t30-hotspot-current:end -->

<!-- corrected-t30-current:start -->
## Current T30: matched comparison completed; strict state failures retained

The authorized artifact742/day69–98 T30 original and latent32/private-reader64
graphs are both saved with exact identity sidecars and complete first-call
results. Both first complete AD calls have finite loss/parameter/state values
and zero hard counts. The saved-only comparison completed8 full AD calls and
3 alternating same-GPU1 warm pairs after the node13:40:26Z idle check. Original
warm median **6.650839463s**, candidate **5.189474726s**, ratio **1.281601668×**
(+28.16% component throughput;1.461364737s less warm call time). It made0/0
model compilations. All forward/gate and parameter repeats pass, but strict
state repeats fail original2/3 and candidate3/3; status is `numerical_failed`.
No architecture/scientific admission or complete-update speed follows.
Neither cold call contributes to the warm ratio.

Across the three warm calls, original sample SD is0.097600114s (CV1.45536%)
and candidate sample SD is0.003583814s (CV0.06906%). Pair ratios are
1.314921160,1.281236635 and1.280737054; their median is1.281236635,
mean1.292298283, sample SD0.01959358 and CV1.51618%.
Three pairs describe this bounded component measurement, not population
uncertainty or complete-update throughput.

Public T30 losses read from saved full cold results on test01 CPU with IO only
are original **0.07021391662495689** and candidate **0.10376825476327928**.
The diagnostic original update0 and candidate fresh weight trees differ;
these scalars are not an accuracy or cross-architecture parity gate.

The fresh grant is `59a57e60-91e1-434a-9787-7189dc602e93`, signed local UTC
2026-10-06T10:41:07.987Z from the user's current delegation in source thread
`019fdc7a-5f6e-7620-a3be-28c855800369`:15600 cumulative GPU seconds, protected
compile12000/matched3600 pools, sequential single-card occupations no longer
than7200 seconds, plus ordinary repairs within those pools. No optimizer
update, formal training or paid execution is part of this scope.

Both graphs use FP32 neural/FP64 physics, the public native checkpointed
objective with `argnums=(0,1)`, guarded turnover `f6240cf5…`, the original
width768/eight-block backbone and identical case digest
`9f91e4642bf065f8a9ffc14417fa06387aacd6464b978130f6139a36c7889cfe`.
Original uses expert128/field dynamic-projection/source reuse; candidate uses
field/chunk64,32×384 latents and private width64/two-block readers. They retain
all legal/private/GPP outputs, physical and science masks, the updater/tail,
declared weights, sampling and unchanged tolerances.

| Architecture | Lower s | Backend s | Probe elapsed s | Compile first AD s | Matched load s | Matched first AD s | Compiler planned live bytes | Compile JAX peak bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original |175.894229|2487.809690|2694.608052|7.073615|96.880020|20.903494|3490684815|3508121600|
| candidate |298.930043|5853.291522|6212.564805|8.772806|175.910771|5.649935|1195593871|1214432000|

These elapsed figures include the probe's asset/graph/result work; the complete
occupation timings below also include runtime preflight and process cleanup.
Compiler-planned live memory and JAX allocation peaks each pass the24GiB gate.
NVML card-total peaks are a separate scope: original compile6555MiB over1359
samples, candidate compile5767MiB over3050 samples, and matched8275MiB over212
samples. Candidate compile includes external GPU allocation after the process
appeared and must not be described as candidate-only memory. Matched holds
both graphs. Its selected-GPU process samples contain probe PID50083 (183 rows)
and non-report PID49820 (14 rows, ownership unverified from the CSV), with no
known external PID31585 sampled; initial and final
selected-GPU snapshots are idle. This sampled evidence does not prove continuous
absence between samples. Matched JAX
cumulative peak3774566912 bytes belongs to the process holding both graphs,
not either model alone. The candidate compile first call is explicitly
**contended** and neither cold call supplies a steady-state denominator.

All eight matched full results and both compile first results have finite full
parameter gradients (original64,884,481/300 leaves; candidate66,123,713/355
leaves), finite3854-coordinate state gradients and3833 nonzero state entries.
All14 hard counts are0 for each. T30 strict self repeats use unchanged
`rtol=1e-5, atol=1e-6`; all full forward/gate and parameter comparisons pass.
The state failures below remain failures, with full results and exact ranges
saved. Existing T7 and historical daily strict failures remain preserved below.

| Pair/order | Original warm s | Candidate warm s | Original state failed coordinates | Candidate state failed coordinates |
|---|---:|---:|---|---|
|1 original→candidate|6.818932695|5.185811061|none|33,2455|
|2 candidate→original|6.648945136|5.189474726|505|33,200|
|3 original→candidate|6.650839463|5.192978092|505,1833|2455|

Coordinate33 is diffuco snowgrain[0,0];200 hydrol canopy2ground[0,1];505 hydrol
snowdz[0,1];1833 slowproc DOC[0,1,4,0,2,0];2455 slowproc carbon_32l[0,2,1,9].
DOC is included in all complete state comparisons: original pair3 has one DOC
failure and all other DOC comparisons have0. These point/window results do not
establish all-branch coverage, trained accuracy or long-horizon science.

### Occupation accounting and external release reconciliation

| Phase | Measured GPU s | Nominal elapsed+60 s | Conservative charge GPU s | Execution outcome |
|---|---:|---:|---:|---|
| original compile |2785.439559|2845.439559|2845.439559|probe and launcher0; own release confirmed|
| candidate compile |6259.534116|6319.534116|7200.000000|probe0; launcher125 only at final external-PID release check|
| matched |431.701096|491.701096|491.701096|8calls/3pairs; launcher0 and own release confirmed|

After all three phases, measured occupation is9476.674772 GPU seconds;
conservative charge10537.140655 GPU seconds, one GPU per sequential occupation.
This is2.632410 measured /2.926984 conservatively charged card-hours.
Compile remaining1954.560441 seconds; matched remaining3108.298904 seconds;
total grant remaining5062.859345 seconds. Matched used491.701096 of its3600 pool.
No refund, grant reset or borrowing occurred.
The synthetic runtime checks passed on the deployed GPU runtime and read no
Teacher data; they are neither real training updates nor scientific admission.
Node and workstation timestamp sources are retained separately because the
node clock was about8 minutes behind the signing clock; elapsed limits use
monotonic durations.

GPU1 was idle when each occupation began. Its monitor first records external
PID31585 at epoch1791287591, about1927 seconds after candidate occupation start.
That observation does not by itself explain earlier compile latency. At exit,
the candidate's raw occupation is `interrupted`, with raw ledger exit0,
`gpu_release_confirmed=false`, `own_process_release_confirmed=false`, and the
unaltered end snapshot; launcher terminal output separately reports125.
The executed guardian's process-group termination check is true. Fixed-PG
scans of25563 and25582 show zero members, every earlier owned process identity
is absent, and external31585 retains its boot/start-tick identity and distinct
UID. Its presence means the card is **not idle**, while our processes are gone.

The accepted ordinary reconciliation preserves raw copies of both ledgers,
changes only candidate status to `reconciled`, stores charge7200 and adds the
own-process proof/launcher125 annotation. Raw release flags and end snapshot
remain unchanged. CPU actual-schema validation proves the raw interrupted
rows block a next occupation and reconciled rows admit a simulated3600-second
matched reservation with compile charge unchanged; no model runs in that test.
At the fresh post-reconciliation node snapshot13:29:12Z, GPU0 had4135MiB/98%
with external PIDs31586/44532, and GPU1 had1509MiB/93% with external31585.
At that snapshot no matched occupation was launched. GPU1 became idle before
the later13:40:26Z check and completed matched occupation. No external process
was touched.

### Preserved artifacts and next executable boundary

Small evidence is under `outputs/diagnostics/shared-latent-reader-20261004/`:
the two `latent-reader-T30-20261006-compile-{original,candidate}-30610006-0000-4000-8000-00000000000{1,2}.json`
reports, identity sidecars, preflights, launcher and memory/process logs;
`latent-reader-T30-20261006.compile-candidate.raw-budget.json`,
`.compile-candidate.reconciled-budget.json`, `.reconciliation.applied.json`,
`.reconciliation.cpu-validation.json` and fixed-PG process evidence.
The exact bookkeeping script is `reconcile_latent_T30_external_release.py`.
Final small receipts are
[matched result](../../../outputs/diagnostics/shared-latent-reader-20261004/latent-reader-T30-20261006-matched-30610006-0000-4000-8000-000000000003.json),
[final budget](../../../outputs/diagnostics/shared-latent-reader-20261004/latent-reader-T30-20261006.final-budget.json),
[compact summary](../../../outputs/diagnostics/shared-latent-reader-20261004/latent-reader-T30-20261006.compact-summary.json),
and `latent-reader-T30-20261006.actual-losses.json`.
Retained executed-source copies are
`probe_shared_latent_reader_recurrent_pair.executed-58c73984.py`,
`shared_latent_recurrent_phase_budget.executed-98ecc0ae.py`, and
`run_shared_latent_reader_recurrent_pair.executed-30ce715e.sh`; their full SHAs
match the actual phase/previous deployment receipts.
Large executables and complete AD arrays remain in remote project-owned
`runtime/outputs/diagnostics`; neither successful graph may be recompiled.

The existing [T30 command package](shared_latent_reader_T30_followup_proposal_20261006.md)
loaded actual UUID0001/0002 graphs for matched UUID0003 on GPU1. This measured
T30 component ratio complements T7 28.24% and historical daily18.81%; those
gains cannot be added into a complete-update speed claim. The historical daily
measurement predates guarded turnover. Any critical-path estimate must retain
that limitation and separate scheduling/IPC/optimizer overhead; unchanged
changing-weight complete-update throughput and its1.5× engineering screen
remain unmeasured.

The user paused the round on2026-10-07 for reassessment. Successful graphs and
all results are retained, along with incomplete full-update source preparation.
CPU readiness validates five existing graphs; candidate corrected daily is
missing. Final probe deployment, launcher deployment, target-Bash dry-run and
GPU ABI checks are unfinished; no new model/update execution was started.
Execution owners report no active preparation processes or new queued jobs.

Logged T30 lowering plus backend compilation totals8815.925484s (2.448868h).
Compile occupations total9044.973675s; the saved-only load/comparison occupation
is431.701096s (7.195min), making9476.674772s (2.632410h) total measured GPU
occupation. These are occupied times, not pure GPU compute. External allocation
was observed during candidate compilation; its impact on elapsed time is
unknown. The reported roughly six-hour round is not a synchronized timing
ledger: remaining wait, reconciliation and later source preparation cannot be
reliably separated from existing evidence. No new timing instrumentation ran.

The measured component gain and65.75% compiler-planned live-memory reduction
remain engineering evidence. Strict state failures remain open, and memory
savings do not establish scientific adoption. Transparent
planning uses `max(parent,T30)+H`: historical daily proxy4.59045/3.76647s and
these T30 medians give1.2816× at `H=0`; identical nonnegative overhead lowers
that estimate. This is not measured complete-update speed and does not meet
the1.5× engineering screen. No new GPU graph, optimizer update or training
execution is authorized by that preparation decision.
<!-- corrected-t30-current:end -->

<!-- corrected-t7-current:start -->
## Current outcome — corrected T7

The completed corrected T7 component has 3 synchronized alternating same-card pairs: original median **1.571275 s**, candidate **1.225237 s**, original/candidate ratio **1.282425×**. It remains blocked by the preserved technical/numerical failures below.

Saved actual T7 losses on the identical artifact742 input:

| Architecture | Public loss |
|---|---:|
| original | 0.0866535292151 |
| candidate | 0.0236253658631 |

These scalars were read from preserved full results on test01 with CPU IO only. The original diagnostic update0 and candidate fresh initialization are different weight trees. Their loss difference does not establish trained accuracy or scientific acceptance; cross-architecture parity remains outside the gates.

The first candidate attempt stopped at the harness binding guard before lowering, with original/candidate model compile counts both zero. The already installed raw function was exactly the staged candidate function; the legacy original-adapter context rejected that legitimate binding. The corrected candidate context requires identity with that exact function, retains it and temporarily installs/restores the historical source-reuse backbone. CPU mocks verified normal/exception restoration and rejection of unknown bindings. This was an interface failure; no candidate numerical or model-compile failure is inferred. The original failure receipt remains `E:\ORCHIDEE-MAN-JAX\outputs\diagnostics\shared-latent-reader-20261004\latent-reader-T7-20261005-compile-candidate.json`; any recovery report must be selected explicitly.

Original executes the existing expert128 decoder/source-reuse route: non-GPP chunks128, canonical GPP chunks64, and field dynamic-projection reuse. The candidate executes canonical latent-reader field chunks64; these are distinct execution routes.

Both retain the width768/eight-block backbone and original physical outputs, masks, sampling and public native-checkpointed T7 loss. Candidate streams use 32×384 latents and private width64/two-block readers. Original has 64,884,481 FP32 parameters/300 leaves at the fixed diagnostic update0; candidate uses the exact saved fresh 66,123,713-parameter/355-leaf whole-tree initialization. Both use guarded turnover function f6240cf5…, FP32 neural arithmetic, FP64 physics and full native parameter/initial-state AD. Cross-architecture numerical parity is not a gate.

A T7 component ratio of 1.5 is an engineering observation, without a scientific accuracy gate. The unchanged-workload complete-update 1.5× engineering screen remains **unmeasured/unmet**; neither this component nor historical daily timing proves it. No scientific acceptance, full-update adoption, optimizer updates, additional paid authorization or automatic T30 execution follows.

### Cold costs and saved graph recovery

| Architecture | Lower s | Compile s | Matched load s | Compile-stage first AD s | Matched first AD s |
|---|---:|---:|---:|---:|---:|
| original | 176.017106 | 2502.203159 | 89.936903 | 1.984161 | 37.401993 |
| candidate | 324.935682 | 5543.315643 | 169.773898 | 3.815041 | 1.673573 |

Recorded matched-stage compile counts: `{"candidate": 0, "original": 0}`. Each successful graph is serialized with mathematical identity before validation; recovery loads that graph under the compilation guard. Complete results are saved before validation. Host transfer, serialization, comparison and cold costs are outside synchronized forward+AD timing.

Original compile/cold cost carries a contention caveat: after its idle launch, the execution owner observed external PID79464 on selected GPU1. The idle snapshot was not a reservation. The named `latent-reader-T7-20261005.compile-original-contention-snapshot.txt` retains the occupancy evidence; this does not establish the duration, effect or cause of a compile slowdown. Matched pairs require a fresh idle-card check before launch.

### Matched complete T7 calls

| Pair | Order | Original s | Candidate s | Original/candidate |
|---|---|---:|---:|---:|
| 1 | original → candidate | 1.715457 | 1.225237 | 1.400102 |
| 2 | candidate → original | 1.570988 | 1.223382 | 1.284136 |
| 3 | original → candidate | 1.571275 | 1.235978 | 1.271280 |

Three-pair timing dispersion (sample standard deviation and coefficient of variation): `{"original": {"sample_standard_deviation_seconds": 0.08332651889535737, "coefficient_of_variation": 0.05146026217360262, "minimum_seconds": 1.570988480001688, "maximum_seconds": 1.7154571115970612}, "candidate": {"sample_standard_deviation_seconds": 0.0068003588757305205, "coefficient_of_variation": 0.0055368532153652235, "minimum_seconds": 1.2233820483088493, "maximum_seconds": 1.2359782010316849}}`. Cold-first calls are excluded; this small sample does not certify full-update throughput.

Observed lower+backend costs: original `2678.220265` seconds, candidate `5868.251324` seconds. These were different occupations/environments, so they are not a controlled cold-start cause comparison. The candidate compilation was more expensive; successful binary reuse across later runs is a condition of useful throughput. The warm component ratio excludes first compilation/load and does not certify startup-inclusive efficiency.

Warm median saving per T7 call: `0.346037` seconds. T30 and complete-update acceptance must separately record cold/load, warm timing and card-hours.

Exact executed T7 harness bytes are retained at `outputs\diagnostics\shared-latent-reader-20261004\probe_shared_latent_reader_T7_pair.executed-f584d1da.py` and `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-latent-reader-20261004/probe_shared_latent_reader_T7_pair.executed-f584d1da.py`, SHA256 `f584d1da168c7b226f6f1870ba0713933fec28a155aa7bc61163c7721324b9f0`. The newer explicit-horizon wrapper has CPU-verified identical T7 mathematical identities; historical approval, binary and full arrays remain preserved.

### Strict repeat evidence

Unchanged coordinatewise tolerances are `rtol=1e-5, atol=1e-6`; finite/nonzero full gradients, physical/science masks and the <24 GiB compiled/observed gates are separate. DOC failures stay failures and are not removed from all-state acceptance. All complete result arrays remain in the remote artifacts listed in the compact summary.


**original / pair1**: full result `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-replacement-matched.original.pair1.result.pkl`.
Forward/parameter/state comparison counts: `{"forward_and_gates": {"coordinates": 20, "failed": 0, "nonfinite": 0, "passed": true}, "all_parameter_gradients": {"coordinates": 64884481, "failed": 0, "nonfinite": 0, "passed": true}, "all_initial_state_gradients": {"coordinates": 3854, "failed": 1, "nonfinite": 0, "passed": false}}`.
Reference complete arrays: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-replacement-matched.original.cold-first.result.pkl`. Complete C-order failed parameter/forward half-open ranges with actual/reference/error/allowance bounds: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-replacement-matched.original.pair1.strict-failure-ranges.json`; the compact summary preserves every range and leaf path/shape.

| Coordinate | Bound field/local offset | Actual | Reference | Absolute error | Allowance |
|---:|---|---:|---:|---:|---:|
| 33 | diffuco_previous_step_state.snowgrain[0] | 0.368348240852 | 0.368362426758 | 1.41859054565e-05 | 4.68362426758e-06 |

**candidate / pair1**: full result `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-replacement-matched.candidate.pair1.result.pkl`.
Forward/parameter/state comparison counts: `{"forward_and_gates": {"coordinates": 20, "failed": 0, "nonfinite": 0, "passed": true}, "all_parameter_gradients": {"coordinates": 66123713, "failed": 0, "nonfinite": 0, "passed": true}, "all_initial_state_gradients": {"coordinates": 3854, "failed": 1, "nonfinite": 0, "passed": false}}`.
Reference complete arrays: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-replacement-matched.candidate.cold-first.result.pkl`. Complete C-order failed parameter/forward half-open ranges with actual/reference/error/allowance bounds: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-replacement-matched.candidate.pair1.strict-failure-ranges.json`; the compact summary preserves every range and leaf path/shape.

| Coordinate | Bound field/local offset | Actual | Reference | Absolute error | Allowance |
|---:|---|---:|---:|---:|---:|
| 508 | hydrol_previous_step_state.snowgrain[1] | 1.69982933998 | 1.69984996319 | 2.06232070923e-05 | 1.79984996319e-05 |

**original / pair2**: full result `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-replacement-matched.original.pair2.result.pkl`.
Forward/parameter/state comparison counts: `{"forward_and_gates": {"coordinates": 20, "failed": 0, "nonfinite": 0, "passed": true}, "all_parameter_gradients": {"coordinates": 64884481, "failed": 0, "nonfinite": 0, "passed": true}, "all_initial_state_gradients": {"coordinates": 3854, "failed": 1, "nonfinite": 0, "passed": false}}`.
Reference complete arrays: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-replacement-matched.original.cold-first.result.pkl`. Complete C-order failed parameter/forward half-open ranges with actual/reference/error/allowance bounds: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-replacement-matched.original.pair2.strict-failure-ranges.json`; the compact summary preserves every range and leaf path/shape.

| Coordinate | Bound field/local offset | Actual | Reference | Absolute error | Allowance |
|---:|---|---:|---:|---:|---:|
| 33 | diffuco_previous_step_state.snowgrain[0] | 0.368352800608 | 0.368362426758 | 9.62615013123e-06 | 4.68362426758e-06 |

**original / pair3**: full result `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-replacement-matched.original.pair3.result.pkl`.
Forward/parameter/state comparison counts: `{"forward_and_gates": {"coordinates": 20, "failed": 0, "nonfinite": 0, "passed": true}, "all_parameter_gradients": {"coordinates": 64884481, "failed": 0, "nonfinite": 0, "passed": true}, "all_initial_state_gradients": {"coordinates": 3854, "failed": 1, "nonfinite": 0, "passed": false}}`.
Reference complete arrays: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-replacement-matched.original.cold-first.result.pkl`. Complete C-order failed parameter/forward half-open ranges with actual/reference/error/allowance bounds: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-replacement-matched.original.pair3.strict-failure-ranges.json`; the compact summary preserves every range and leaf path/shape.

| Coordinate | Bound field/local offset | Actual | Reference | Absolute error | Allowance |
|---:|---|---:|---:|---:|---:|
| 33 | diffuco_previous_step_state.snowgrain[0] | 0.368453115225 | 0.368362426758 | 9.06884670258e-05 | 4.68362426758e-06 |

### Resource and independent budget evidence

original compiled memory: `{"alias_size_in_bytes": 0, "argument_size_in_bytes": 259863845, "estimated_live_bytes": 3488290989, "output_size_in_bytes": 259571424, "temp_size_in_bytes": 2968855720}`.
candidate compiled memory: `{"alias_size_in_bytes": 0, "argument_size_in_bytes": 264820773, "estimated_live_bytes": 1192605605, "output_size_in_bytes": 264528792, "temp_size_in_bytes": 663256040}`.

Recorded JAX allocator peak: `3773142528` bytes. NVML peak and process release come from the independent settlement receipt, without equating allocator planning with device observation.

This **independent 7,200 GPU-second grant** records charged `5680.508494503796` GPU-seconds, remaining `1519.491505496204`, card-hours `1.5779190262510545`, all sessions settled `True` and NVML peak `1003.0` MiB. These charges and balances are separate from all historical daily/leaf allowances.

The earlier T7 ledger remains preserved at `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005.gpu-budget.json`: 3 settled sessions charged 2808.652223 GPU-seconds. Its unused balance is excluded from this new migration grant; retries share the new cumulative ceiling.

The **separate paid allocation** has a 7,200-second wall and ¥5.52 approved ceiling; it draws no funding from the free grants. Job `15580715` scheduler state `FAILED`; helper `resource_failed`; helper settled `True`. Pending/queued status does not establish an allocated GPU, completed model graph or matched pass.

Full helper occupied GPU time, including setup/preflight/compile/load/timing/persistence/cleanup: `26.612438` seconds. Helper-only cost estimate: ¥`0.020403`. Scheduler allocation elapsed: `28` seconds; scheduler billed duration: `unavailable` seconds; actual accounting cost: ¥`unavailable`. Helper time and scheduler billing are distinct receipts.

Queue duration: `49869.0` seconds, excluded from GPU charge. Paid helper receipt `outputs\diagnostics\shared-latent-reader-20261004\latent-reader-T7-20261005-paid.gpu-budget.first-attempt.json`; compact scheduler receipt `outputs\diagnostics\shared-latent-reader-20261004\latent-reader-T7-20261005-paid.accounting.json`. Missing or unsettled accounting remains unavailable; the approval ceiling is not a measured charge.

Paid terminal execution error: `PaidResourceError: assigned physical GPU is occupied or availability unverified before candidate-model; no model work allowed`. The environment preflight completed and its child was released; candidate model work never started. The second idle check recorded no PID/0 MiB but utilization5%, with final utilization2%. This does not establish the driver-level cause. The repaired launcher now waits at most20s including both queries for utilization to reach exact0 only when no PID and memory≤100MiB; external PID, high memory, query error or deadline still stops execution. No threshold or scientific gate is relaxed.

Current paid scheduler reason: `NonZeroExitCode/PaidResourceError after successful preflight, util5%/0MiB/noPID`; declared-node resource snapshot: `{"host_load": [6.74, 6.81, 6.46], "final_PID_count": 0, "scheduler_elapsed_seconds": 28, "candidate_guard_PID_count": 0, "affinity": "12,13,14,15,68,69,70,71", "source": "latent-reader-T7-20261005-paid.gpu-budget.first-attempt.json", "candidate_guard_memory_mib": 0, "candidate_guard_utilization_percent": 5, "node": "ibc13b03n04", "scope": "actual first allocation, not current resource availability", "final_utilization_percent": 2}`. This is a point-in-time resource observation, not a reservation or a contention explanation. The existing paid backup retains its approved resource/time/cost envelope; no additional submission or larger budget follows.

Original human approval and n07 submission provenance remain retained. The lead's execution judgment moved the same pending job `15580715` from `ibc13b03n07` to `ibc13b03n04` within the same 8 CPU/one V100/64 GiB/two-hour/¥5.52 envelope, after a hold and checked resource snapshot. No new submission occurred. The selected node had CPUAlloc88/112 and load6.50, with GPU2 scheduler-free; it was not completely idle.

The **newly human-approved replacement** is job `15590636`: scheduler `COMPLETED`, helper `finished`, settled `True`. Its separate cap is 7140 seconds / ¥5.474 (display approval ceiling¥5.48). The first failed allocation reserves60 seconds/¥0.046; the cumulative cap remains7200 seconds/¥5.52. This is exactly one replacement, with no funding from either free grant.

Replacement helper GPU seconds: `6403.818634`; scheduler elapsed seconds: `6408`; actual cost: `unavailable`. Separate receipts: `outputs\diagnostics\shared-latent-reader-20261004\latent-reader-T7-20261005-paid-replacement.gpu-budget.json`, `outputs\diagnostics\shared-latent-reader-20261004\latent-reader-T7-20261005-paid-replacement.accounting.json`. Unsettled elapsed time and approval caps are not final charges.

Independent paid CSV NVML peak: `8271` MiB; helper process release `True`. Scheduler-based by-second estimate¥`4.912800`; rounded billed duration and actual monetary charge remain unavailable. Final GPU snapshot had0MiB/no process but25% utilization, so exact idle utilization is not claimed.

The preceding free migration candidate receipt remains `outputs\diagnostics\shared-latent-reader-20261004\latent-reader-T7-20261005-compile-candidate-free-migration-20261006.json`: status `incomplete_timeout`, deadline events `[{"elapsed_seconds": 5669.622747899964, "phase": "backend_candidate", "stage": "deadline_exceeded"}]`. It produced no saved candidate graph or model call; the independent7200-GPU-second settlement remains separate.

### Consequence and next measurement

The bounded warm T7 gain supports preparation of the [corrected T30 follow-up proposal](shared_latent_reader_T30_followup_proposal_20261006.md). It does not meet or measure the complete-update1.5× engineering priority. The four strict state failures remain preserved; no automatic T30 graph, optimizer update or adoption follows. The proposal separately budgets cold compilation/load, matched complete AD and process release, with new human authorization required.

The source audit found the wide latent bank and wide K/V projections outside the field loop, and narrow reader K/V outside the chunk loop; there is no evidence of a repeated wide bank to remove. Canonical candidate field64 is 161 maps / 265 chunks / 9,046 padded coordinates. The earlier expert128 23.883 GMAC estimate must not be used for this field route. Optional cost analysis reads the already saved/loaded executable and causes no new compile or profiler graph.

Historical daily evidence remains below: its 1.188076× micro4 median ratio and strict state-gradient failures on both architectures preceded the guarded turnover correction. Those results are retained evidence, without supplying corrected T7, complete-update or scientific admission.

Current execution owner is `scripts/hpc/probe_shared_latent_reader_T7_pair.py`; this block is refreshed offline by `outputs/diagnostics/shared-latent-reader-20261004/write_latent_T7_results.py`. The sibling `latent-reader-T7-20261005.compact-summary.json` names source receipts, exact strict failures and preserved remote full-array artifacts. Large executables and full AD pickles remain under project-owned remote `runtime/outputs/diagnostics`; this updater reads compact JSON only.

<!-- corrected-t7-current:end -->

## Historical daily evidence retained

The following sections record the completed pre-correction daily diagnostic. Their then-pending T7 statements and budgets are historical; current corrected T7 authority is the block above.

### Frozen scope

Seven expert streams each retain 32×384 latents and two latent blocks, with
private width-64 two-block readers, eight heads, FFN256 and readout64→256→1.
The original 768×8 backbone, encoding/bridge, physical processes, canonical
losses, masks and sampling remain. Neural arithmetic is FP32 and physics FP64;
full native parameter/initial-state AD is used. Canonical execution is
`latent_reader_execution="field"`, configured field chunks of 64: **66,123,713
parameters and 355 FP32 leaves**. Expert128 retains a strict discrepancy and
was not substituted; its 23.883 GMAC estimate does not apply to this field route.

Candidate weights are the exact saved fresh public whole-tree initialization
with the declared seed. Original update0 is only the original-architecture
baseline. The immutable checkout is c599236 with the declared staged overlay;
the old original daily executable filename's V4 is provenance, without a
candidate architecture mixture. Losses differ across architectures by design:
candidate 0.388390814905172, original 0.500934418782458.
Cross-architecture loss parity was not a gate or an accuracy conclusion.

### Daily measurement and strict replay evidence

The longer authorized run launched 2026-10-05 at 10:17:50+0800 on free gln01,
node `ibc13b02n04`, physical GPU1, CPU affinity 0-111. Both cards were idle
(0 MiB/0%, no compute processes); load 0.00/0.01/0.05. Original full-state
reference identity/ABI checks passed. Lowering took 309.986596 s,
backend entry was at 379.893438 s and backend compilation took
**3657.794692 s**. Force4 and the repaired CUDA 12.6.85 linker
were explicit; this is observed compile cost, without a proven speedup or
specific compiler-pass attribution. The 315 MiB executable/identity and 253 MiB
complete first result were saved before postprocessing.

Its first synchronized micro4 warm call took 0.327005427 s. Loss/auxiliary 161
coordinates were exact and all 66,123,713 parameter-gradient coordinates
passed (max_abs 1.1175871e-8). State replay failed 2/15,416 coordinates,
max_abs 0.0856906772, relative_l2 2.4669686e-7, at unchanged
`rtol=1e-5, atol=1e-6`. It exited 1 with `NumericalFailure`.
That failed warm call's full raw state values were not saved; the aggregate
comparison and the complete first result remain. New recovery values below
are compared to that preserved first result and are not claimed to be the
unsaved original failed warm values.

Recovery launched at 11:42:25+0800 on the same node/GPU1 with load 0.00/0.06/0.47,
both cards again idle, affinity 0-111 and a 900-second cap. Identity and executable
ABI were rechecked; candidate/original model compile counts were both 0. Saved
force4 binary options were retained; current default compile options were
explicitly unused. Three candidate warm calls were 0.329725780/0.317064505/
0.317200905 s (median 0.317200905 s). The synchronized matched full parameter/state
daily micro4 calls were:

| Pair | Order | Original seconds | Candidate seconds | Original/candidate |
|---|---|---:|---:|---:|
| 1 | original → candidate | 0.376175579 | 0.316831041 | 1.187307 |
| 2 | candidate → original | 0.377397401 | 0.317654228 | 1.188076 |
| 3 | original → candidate | 0.377629906 | 0.320056960 | 1.179883 |

Median original 0.377397401 s / candidate 0.317654228 s gives 1.188076×.
All counts, masks, gradient boundaries and four daily samples are unchanged;
these are fixed weights and inputs, without recurrent or AdamW update timing.
Host copies, evidence serialization and comparisons occurred outside the
synchronized executable timing. There are only three pairs on one V100.

Recovery first and every candidate/original timed result were validated.
Candidate: 355 gradient leaves,66,123,713 coordinates,66,102,209 nonzero,
norm 5.39158725. Original: 300 leaves,64,884,481 coordinates,64,862,209 nonzero,
norm 10.39402366. Every state gradient is FP64, shape 4×3854, all finite,
15,334 nonzero. All 11 hard counters were 0 for both architectures on every
timed result: budget_failure, carbon_negative, discrete_mismatch, mask_mismatch,
prediction_nonfinite, science_mask_mismatch, science_missing_valid_term,
science_nonfinite_prediction, science_nonfinite_target, state_nonfinite and
water_negative. Loss/auxiliary 161 coordinates were exact in every comparison;
all parameter-gradient coordinates passed (candidate max_abs≤1.4901161e-8,
original≤2.9802322e-8). State-gradient comparisons retained these outcomes:

| New recovery call | Failed state coordinates /15,416 | Max absolute error over full state |
|---|---:|---:|
| recovered-first-vs-preserved-first | 1 | 0.0285635591 |
| candidate-warm-0 | 1 | 0.0856906772 |
| candidate-warm-1 | 0 | 0.114254236 |
| candidate-warm-2 | 1 | 0.0571271182 |
| original-matched-0 | 4 | 0.228508473 |
| candidate-matched-0 | 1 | 0.0571271182 |
| candidate-matched-1 | 2 | 0.114254236 |
| original-matched-1 | 1 | 0.228508473 |
| original-matched-2 | 3 | 0.298287418 |
| candidate-matched-2 | 1 | 0.0571271181 |

Max absolute error is over the whole array; it is not necessarily an error
on a failed coordinate. Coordinatewise tolerance is
`1e-6 + 1e-5 * abs(reference)`, without norm-based acceptance. The recovery
completed diagnostic recording and then exited 1 with the retained strict
`NumericalFailure`; one passing warm state call never clears the prior failure.

#### Actual failed coordinates and scales

The saved prepared context was inspected on CPU only, with no model/backend/
gradient call. Its actual bound contract is `context.head.contract`:
`slowproc_stomate_previous_step_state.DOC` occupies [1327,2223), shape
(1,2,32,2,7,1);`slowproc_stomate_previous_step_state.carbon_32l` occupies
[2286,2478), shape(1,3,2,32). The following leaf offsets are flat coordinates
within those actual contract spans; they are not guessed physical/PFT indices.

| Call | Sample,state coordinate | Leaf/local flat offset | Actual gradient | Reference gradient | Absolute error | Effective tolerance |
|---|---|---|---:|---:|---:|---:|
| recovered-first-vs-preserved-first | 1,1869 | DOC[542] | 13.24932595 | 13.2487888561 | 0.000537093828 | 0.000133487889 |
| candidate-warm-0 | 0,1897 | DOC[570] | 13.7121306574 | 13.7123083031 | 0.000177645697 | 0.000138123083 |
| candidate-warm-2 | 1,1869 | DOC[542] | 13.2482517623 | 13.2487888561 | 0.000537093828 | 0.000133487889 |
| original-matched-0 | 0,1882 | DOC[555] | -39.1774021638 | -39.1757929782 | 0.0016091856 | 0.00039275793 |
| original-matched-0 | 2,1924 | DOC[597] | -0.179458364563 | -0.17946138021 | 3.01564782e-06 | 2.7946138e-06 |
| original-matched-0 | 3,2320 | carbon_32l[34] | -1.34821098906 | -1.34818619353 | 2.47955322e-05 | 1.44818619e-05 |
| original-matched-0 | 3,2386 | carbon_32l[100] | -1.15852155749 | -1.15850737158 | 1.41859055e-05 | 1.25850737e-05 |
| candidate-matched-0 | 1,1869 | DOC[542] | 13.2482517623 | 13.2487888561 | 0.000537093828 | 0.000133487889 |
| candidate-matched-1 | 0,1897 | DOC[570] | 13.7121306574 | 13.7123083031 | 0.000177645697 | 0.000138123083 |
| candidate-matched-1 | 1,1869 | DOC[542] | 13.2482517623 | 13.2487888561 | 0.000537093828 | 0.000133487889 |
| original-matched-1 | 0,1882 | DOC[555] | -39.1771607859 | -39.1757929782 | 0.00136780776 | 0.00039275793 |
| original-matched-2 | 2,1924 | DOC[597] | -0.179457668644 | -0.17946138021 | 3.71156655e-06 | 2.7946138e-06 |
| original-matched-2 | 2,2319 | carbon_32l[33] | -1.94060715675 | -1.94057756305 | 2.95937061e-05 | 2.04057756e-05 |
| original-matched-2 | 3,2386 | carbon_32l[100] | -1.15852310721 | -1.15850737158 | 1.57356262e-05 | 1.25850737e-05 |
| candidate-matched-2 | 1,1869 | DOC[542] | 13.2482517623 | 13.2487888561 | 0.000537093828 | 0.000133487889 |

All failed-coordinate inputs are finite. Representative candidate DOC1869
input 9.22386309e-12 has TRAIN base state scale 4.23340697e-10,
next-state scale 4.23441349e-10 and delta scale 1.20104846e-11; DOC1897
input 3.53508435e-10 has corresponding scales5.11971141e-9,
5.12039277e-9 and 1.76789907e-11. Full input/actual/reference FP64 arrays,
exact failed rows, base scale arrays, grouped physical scales, science loss
scales and identities are retained in all ten mirrored `state-replay.pkl`
files. `daily-replay-coordinate-analysis.json` preserves all row values and
the actual contract mapping; `daily-replay-runtime-contract.json` preserves
the contract/identity inspected from the prepared context. The raw input has
four nonfinite entries at(row0..3,state20), actual roughheight_pft local 0;
these are distinct from the entirely finite gradient arrays and the finite
failed-coordinate inputs. No NaN, scaling, reduction or compiler cause is
assigned by this evidence.

Snapshot model-identity metadata was corrected after root review. The three
`original-matched-*` raw files initially carried candidate identity although
their labels, original-reference comparisons and actual numerical arrays were
original executions. Their active local and remote artifacts now use the
already validated original admission identity (`field_original`); all ten
snapshot files and JSON rows explicitly name executed architecture and replay
scope. Future writes require explicit execution architecture/identity, including
the original identity for any additionally saved failed full parameter result.
This was CPU serialization only: every label, array, summary, input, scale,
loss/auxiliary and comparison payload was preserved with identical before/after
numerical-payload SHA256. The repair receipts record old/new artifact hashes;
no model, GPU, gradient, timing or validation was recomputed and no budget charge
changed. Two targeted metadata tests passed in0.12 s; the existing eight numerical
diagnostic tests were not rerun for this metadata change.

### Memory and native compiler artifacts

| Bytes | Candidate | Original |
|---|---:|---:|
| Compiled arguments |266,259,808|261,302,880|
| Compiled outputs |264,623,604|259,666,236|
| Compiled temporary |4,242,843,824|10,827,392,176|
| Estimated live (alias0) |4,773,727,236|11,348,361,292|

Recovery candidate-result-phase observed peak was 5,366,611,968 B. After both
binaries executed, cumulative process peak was 12,161,682,944 B and cumulative
allocator pool peak 22,546,481,152 B, below 24 GiB (25,769,803,776 B).
These are process observations: the original executable was preloaded for ABI
before candidate execution; they do not certify two-card complete-update peaks.

The long compile preserved one native module `module_0519.jit_objective` with
17 files totaling 2,121,611,291 B, without an all-pass regex: before-HLO
353,818,743 B, optimized-HLO 231,021,335 B, buffer assignment261,135,834 B,
memory report 553,027 B, inlined unoptimized LLVM 1,015,884,111 B, other unoptimized
LLVM 88,164,118 B, four optimized LLVM and four PTX plus backend extras.
The native memory report totals 4,780,413,025 B (4.45 GiB) with temporary≈3.95 GiB;
it is auxiliary compiler planning evidence. Optional snapshots add measured
overhead and establish completed stages, without a pass-cause or speedup claim.
No IR regeneration or second model compile was used for recovery.

### Budget and release

The initial 18,000 occupied GPU-second ceiling was increased by explicit user
approval on 2026-10-05 by 7,200 to 25,200. Approval covered one additional free
daily window and necessary saved-binary recovery, without training, recurrent
execution, another large compile, precision/architecture/loss/sampling changes
or paid fallback. Every historical charge remains in the locked ledger:

The subsequent separately approved original full-state schedule continuation
added1800 to 27000. Its free two-card window charged 897.671305 GPU-seconds
to its own source, without consuming the daily-only extension. This later
original diagnostic does not alter the candidate conclusions above.

| Session id prefix | Stage | Cards | Occupied GPU seconds | Exit |
|---|---|---:|---:|---:|
| 8362cb58 | preflight | 1 | 27.166253 | 0 |
| 7e8dc419 | original-changing | 2 | 958.758946 | 0 |
| ad232805 | T7 | 1 | 20.714380 | 1 |
| 96aa45af | T7 | 1 | 3841.944231 | 124 |
| 6c5076d7 | T7 | 1 | 5577.124218 | 1 |
| b9744e34 | nvlink-smoke | 1 | 3.325437 | 0 |
| 972212c2 | daily | 1 | 14.240683 | 1 |
| c3ee822f | daily | 1 | 2744.802284 | 124 |
| 7b7bd8f9 | daily | 1 | 3428.916094 | 124 |
| 25084d88 | nvlink-smoke | 1 | 19.698289 | 0 |
| 264ea0f1 | nvlink-smoke | 1 | 3.576026 | 0 |
| bef88fe4 | nvlink-smoke | 1 | 3.175281 | 0 |
| a7119fdf | daily | 1 | 4094.435519 | 1 |
| 4136fcab | daily | 1 | 133.952412 | 1 |
| aada8ced | original-full-state-schedule | 2 | 800.818923 | 1 |
| 90f65afc | original-full-state-schedule-extra-1800 | 2 | 897.671305 | 1 |

All 16 sessions are settled. Total charged is **22570.320281 GPU-seconds**
(6.269533 card-hours), overall remaining
4429.679719. The approved additional window used
4228.387930/7200, leaving 2971.612070 within that window. The long
compile charged 4094.435519 s; recovery charged 133.952412 s. The subsequent original
full-state schedule charged 800.818923 GPU-seconds to the separate original 18,000
source, leaving 555.738955 in that source and 199.181077 in its fixed1,000 route.
It did not consume the daily-only additional window. The ledger ceiling
or unused allowance is not permission for another large compile. The final
wrapper 19725 is gone; both GPUs show 0 MiB/0% and no compute processes.
Process-release evidence is mirrored alongside the ledger. No paid jobs ran.

### Supporting evidence and pending gates

The deployed no-Teacher synthetic preflight passed on JAX/JAXLIB 0.4.38/V100,
report elapsed 25.710745 s and occupied 27.166253 GPU-seconds. The accepted
CUDA 12.6.85 nvlink path passed real four-cubin small smoke linking, finite/
nonzero parameter/state AD and strict replay. Earlier T7 compilation timed
out then failed at compiler subprocess startup (`process.Start()`), with no
model binary/numerical verdict. Daily setup repair installed the existing
public `shared_daily_dual_ad` module; bounded 2400/3100-second backend attempts
timed out before this successful long compile. Low-volume compiler logger
settings reached the runtime but emitted no pass lines. Full failed reports,
logs and charges remain, without retrospective tolerance or math changes.

Original architecture changing-weight schedule equivalence passed two actual
AdamW updates per chain (four total), using old daily32|T7+T30 versus new
daily32+T7|T30. All 64,884,481 gradient/parameter coordinates and 129,768,963
optimizer coordinates passed per pair, including subsequent parameter use,
loss/auxiliary and physical gates. Max parameter differences 3.1665e-8/5.9605e-8;
step 0→1→2 and request versions 0→1 were verified. Logical parent/worker peaks
15,065,246,976/3,758,390,016 B passed 24 GiB. Those historical schedule binaries
did not independently certify initial-state AD. Both final parameter/AdamW
diagnostic chains remain preserved remotely.

The selected field route passed a bounded smooth-fixture independent
JVP/VJP/finite-difference gate with six nonzero normalized coordinates,
DOC[1327]=0.25, difference 1.318% at unchanged h=1e-3, rtol 2e-2,atol 2e-5.
Cancelling-direction and expert128 failed evidence remain. Fixture admission
does not certify deployed full state gradients or all derivatives.

Full-state ABI adapters and dual runner are implemented/staged:16 targeted
adapter tests, combined 40 CPU tests and 6 final orchestration guard tests passed.
The default dual diagnostic requires four changing performance pairs plus
independent two-step equivalence chains, all-pair and post-first median ratios
≥ 1.5 and max 16 candidate attempted calls. Its required admissions/manifest
remain unavailable; it was not executed. Eight pure NumPy replay-diagnostic
tests passed, probe AST/help and actual recovery target-shell argument
expansion passed. No expensive runtime/preflight was repeated for logging only.

Hard state/science mask counters differ from raw undefined-output verification,
which remains missing. `declared_mask_change` can be nonzero by design.
Candidate T7/T30, actual raw reader output zeros, changed-weight full-state
dual equivalence/memory/throughput and reproducible fresh scientific training
remain unadmitted. The next scientific or resource choice belongs to the root
research decision; no further automatic run follows this completed recovery.

### Artifacts and owners

Local reports/logs/identity/ledger and all ten small state-replay PKLs are under
`outputs/diagnostics/shared-latent-reader-20261004/`. Large saved binaries and
complete candidate/original results remain under remote project
`runtime/outputs/diagnostics` with the same stems; they were not recopied.
Long stem:`shared-latent-reader-20261004-daily-first-full-ad-force4-long-diag`.
Recovery stem:`shared-latent-reader-20261004-daily-first-full-ad-force4-replay-diagnostics`.
The exact launch/target-shell receipts, native HLO directory, persistent budget
authorization and retained original references provide restoration provenance.

Execution owners are `shared_gpu_budget.py`,`run_shared_latent_reader.sh`,
`probe_shared_latent_reader.py`,`shared_latent_replay_diagnostics.py`,
`shared_full_state_ad_execution.py`, and the prepared full-state dual
runner/private worker under `scripts/hpc/`. Serialized class/module identities
are preserved. Recovery explicitly requires the saved executable, preserved
failed report/first result and exact parameters, rejects new compiler overrides
and keeps failure sticky while completing bounded diagnostics.

Parent Verify graph evidence is project`E-ORCHIDEE-MAN-JAX`, generation
`2026-10-04T14:30:28Z`. These probe/helpers are not tracked and relevant shared
metadata changed; bounded exact-source fallback was used. Actual contract
mapping above uses the serialized runtime object, without an exhaustive graph
or repository completeness claim.

### Authorized paid fallback — job 15580715

The human approved one paid fallback with maximum charge ¥5.52, answering
`批准这一次，最多5.52元（推荐）` to request
`call_ZTJG0vNtMhNi5VjpWs7DPO4h`. The approved receipt is
`latent-reader-T7-20261005-paid.approval.json`; the reusable template retains
`approved:false`. Exactly one job, **15580715**, was submitted at remote
`2026-10-05T23:15:10+08:00`. At the latest scheduler check, remote
`2026-10-05T23:32:32+08:00`, it remains `PENDING / Priority`, with no estimated
start, assigned node or `AllocTRES`; `ElapsedRaw=0`. The paid helper and model
have not started, and charge before allocation is zero. No free candidate runs concurrently with
this job. The runner is `scripts/hpc/shared_latent_T7_paid_budget.py` with
`scripts/hpc/slurm_shared_latent_reader_T7_pair_paid.sh`.

The proposed envelope is one nonexclusive `gnall` node/task, eight Slurm CPUs,
one V100, 64 GiB and two hours. At CPU ¥0.07/core-hour plus GPU ¥2.20/hour,
the worst charge is ¥5.52. Workdir is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1`.
The exact approved submission is `sbatch --nodelist=ibc13b03n07
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-latent-reader-20261004/slurm_shared_latent_reader_T7_pair_paid.sh`.
The absolute launcher entry is staged under the transfer overlay; the immutable
code checkout remains the scientific code root and working directory.
The prepared preference is CPU-idle `ibc13b03n07`; the scheduler snapshot showed
CPUAlloc0/112, load0.01 and StateIDLE, but GresUsed/AllocTRES reported eight GPUs
allocated without a corresponding queued job. GPU availability is **unconfirmed**.
The unapproved template names the exact scheduler snapshot under the overlay:
`latent-reader-T7-20261005-paid.scheduler-detailed-snapshot.txt`. A read-only
SSH attempt to inspect the node closed its connection. Target-shell syntax and
dry-run expansion checks passed on `gln01`; the receipt is
`outputs/diagnostics/shared-latent-reader-20261004/latent-reader-T7-20261005-paid.target-shell-dry-run.txt`.
The approval authorizes this one submission; no automatic second job follows.
At the pre-submission free-card check, remote `2026-10-05T23:09:31+08:00`, both cards still had external
PIDs 104848/104846, approximately 4997/4957 MiB and 100% utilization; no project
T7 process was running. The fixed free candidate launcher remains ready, while
the paid job is the sole selected execution route.

All setup, approved-job environment verification/preflight, graph work, timing and cleanup fit
within the first approved7200 seconds: 180 preflight, 6240 candidate, 600 matched, 60 setup
margin and 120 cleanup reserve. Inner candidate lower/backend/call limits are
420/5700/180 seconds with a 6210-second probe wall; matched probe wall is 570
seconds. The saved corrected original is load-only. Actual JAX 0.4.38 and device
kind must exactly match its sidecar; no original recompile fallback exists.
`--runtime-preflight` can name the completed free receipt. The approved job first
checks its original-report linkage and actual canonical Python/container/code
binding, JAX/JAXlib/NumPy versions, x64, device kind and original driver version.
When those match, it reuses the passed receipt and executes no fresh synthetic
check. A changed or unverifiable environment runs the bounded synthetic check
within the same 180-second cap. Every paid phase samples memory/utilization and
process memory on the single scheduler-selected physical GPU; its CSV paths and
phase PID are recorded in the independent ledger.
Before preflight and again immediately before candidate model work, the parent
host helper checks only the scheduler-assigned physical GPU/UUID. It must have
zero compute PIDs, at most100MiB total card memory and zero utilization. Query
failure or occupancy persists a snapshot and stops as a resource failure; other
users' processes are never modified. Card-total observations remain separate
from the own-architecture 24GiB memory gate.
TERM/INT requests are sticky. The helper stops its own phase process group with
a bounded TERM/KILL sequence, persists phase exit/release evidence and settles
the independent ledger after the final GPU snapshot. Slurm sends batch TERM
120 seconds before the hard wall limit (`--signal=B:TERM@120`). The120-second cleanup
reserve remains inside the7200-second wall; helper occupied time includes its
setup and cleanup, while scheduler startup and billing remain separately labeled.
Any free, adapter-recovery or paid candidate success is also load-only under
the exact graph identity/ABI guard. An incomplete or ambiguous saved candidate
blocks compilation rather than authorizing another graph. Failure preserves
assets and prevents a fabricated matched ratio.

The frozen manifest SHA/path, source overlays, siblings, inputs and mathematical
identity remain intact. Its `paid_jobs:false` records the original free scientific
protocol; a separate explicit paid approval receipt and independent allocation
ledger are the outer execution authority, excluding all free/historical
grants. The helper wall-time cost estimate excludes scheduler allocation startup;
the approved worst scheduler bill remains ¥5.52. No training or T30 is included.
