# Original full-state two-card schedule diagnostic — 2026-10-05

The bounded original-only window completed one real paired changing update,
artifact 742, with both independent public AdamW chains advancing step 0→1.
The report is **strict_failed**, with timing **incomplete**: four pairs were
requested, and no post-first pair fit the conservative remaining-time gate.
All full parameter-gradient, parameter, AdamW and component loss/auxiliary
comparisons passed. Initial-state gradient strict failures remain unresolved.
No production adoption, scientific acceptance or formal training follows.

## Executed scope and environment

Old schedule: parent daily32, worker T7+T30. New schedule: parent daily32+T7,
worker T30. Daily32 contains eight distinct micro4 slices; complete T7 and T30
use the unchanged public losses and native full parameter/initial-state AD.
All three branches consumed parameter version0. The original architecture has
64,884,481 FP32 parameters in 300 leaves, FP32 neural and FP64 physics.
The saved original daily/T7/T30 full-state executables were reused; all four
loaded instances reported model_compiles=0 and model_gradient_compilations=0.

The free gln01 launch at 2026-10-05T12:45:56+0800 ran on ibc13b02n04, parent
physical GPU0 and worker physical GPU1, both V100-SXM2-32 GB. CPU affinity was
0-55. Both GPUs were idle at launch; load was 0.00/0.01/0.05. Immutable checkout:
c599236bcf57b745e9bafed678298010f4c3b58f, with the recorded staged overlay.
External cap 500 seconds, Python wall 480 seconds, save/exit reserve 30 seconds.
No paid job or new model-gradient compilation ran.

## Timing and its limits

| First paired update | Old seconds | New seconds |
|---|---:|---:|
| Complete update, including synchronization and public AdamW |54.703719525|7.818119701|
| Daily32 |26.161991294|4.459329627|
| T7 |22.268866789|1.921879223|
| T30 |6.968147047|6.734809229|
| Branches including IPC |30.764035875|7.721263457|
| Synchronized recurrent branch |29.237590767|6.735278249|
| Worker total |29.600519352|6.941591075|
| Returned gradient device put/sync |0.076815145|0.048527036|
| Component merge/sync |1.629127871|0.011728963|
| AdamW/sync |11.567917785|0.020105598|
| Merge/check/optimizer |23.928236213|0.094999554|
| Parent parameter device_get |0.132821580|0.020429533|
| Request pickle write |0.376340300|0.424021155|
| Response wait/read/unpickle |30.238969911|7.275949690|

The first ratio is 6.997043×. The first old call includes helper warming;
this is not a steady-state throughput ratio or isolated compilation estimate.
Post-first samples and ratio are absent. Pair elapsed including comparison,
download and save was 76.002142221 seconds. Subtracting both complete updates
gives 13.480302995 seconds of combined overhead; separate old compare/save
timestamps were not recorded and are not reconstructed.

Measured full-state executable load/ABI times were parent daily22.540654,
parent T7 89.441947, worker T7 97.218590 and worker T30 95.103411 seconds,
summing 304.304601 seconds. The start-worker event occurred at 116.363414
seconds; adding its two measured loads gives a startup lower bound 308.685415
seconds, not an exact ready timestamp. Final Python elapsed 388.828648 seconds.
Next-pair prediction was 1.25×76.002142+5=100.002678 seconds; it did not fit the
450-second reserve boundary. Resource incompleteness does not identify an
invalid schedule.

## Numerical, physical and memory evidence

Unchanged coordinate tolerance: rtol 1e-5, atol 1e-6; actual=old, reference=new.

| Comparison | Coordinates | Strict failures | Maximum absolute error |
|---|---:|---:|---:|
| Mixed parameter gradient |64,884,481|0|2.980232238769531e-8|
| Updated parameters |64,884,481|0|2.980232238769531e-8|
| Public AdamW tree |129,768,963|0|3.4924596548080444e-10|
| Daily32 initial-state gradient (batch mean) |123,328|34|retained in JSON/full arrays|
| Raw daily micro4 mean derivatives |123,328|46|retained in JSON/full arrays|
| T7 initial-state gradient |3,854|2|retained in JSON/full arrays|
| T30 initial-state gradient |3,854|0|retained in JSON/full arrays|

Component parameter-gradient, loss and auxiliary comparisons also had zero
failures. All state gradients were FP64 and finite; daily had 122,670 nonzero
coordinates and each recurrent branch 3,833. Physical/science hard counters
were all 0; parameters and optimizer were finite. Both schedules had daily loss
0.7295894826464191, T7 0.08665354178956992, T30 0.07021390839213733,
joint 0.8864569328281264.

CPU inspection independently reproduced the fail counts and mapped the
actual prepared-context contract: daily batch-mean failures were biomass 3,
DOC 23 and carbon_32l 8. The daily batch mean is each micro4 mean gradient
divided by 8; this scaling is unrelated to physical-state input standardization
or its precision. T7 failures were both
diffuco_previous_step_state.snowgrain:

| State coordinate/local offset | Old | New | Absolute error | Effective tolerance |
|---|---:|---:|---:|---:|
|33/0|0.36855316162109375|0.3685283958911896|2.4765729904174805e-5|4.685283958911896e-6|
|34/1|-1.5477838516235352|-1.547804594039917|2.0742416381835938e-5|1.6478045940399173e-5|

The coordinate analysis preserves every failed row, full state arrays and
contract spans. No cause is assigned. Prior original daily micro4 self-replay
failures 4/1/3 remain unresolved and are not superseded by this comparison.

Observed parent peak 13,770,521,088 B; worker peak 3,775,201,536 B with T7+T30
resident/executed. Both are below 24 GiB. Allocator pool peaks were respectively
19,325,255,680 B and 5,366,611,968 B. These are observations for this one completed
pair, not admission of another architecture or long training run.

## Preservation, budget and release

Local evidence folder: outputs/diagnostics/shared-latent-reader-20261004.
Primary JSON/log/launch/release: shared-original-full-state-schedule-20261005.*.
Full state artifact: .state-diagnostics.pkl; exact rows:
original-full-state-schedule-state-coordinate-analysis.json. Remote consolidated
.pair-001-gradients.pkl (~2.08 GB) retains both mixed and component full
parameter/state gradients. Independent .old-diagnostic.pkl and
.new-diagnostic.pkl (~743 MiB each) retain their own full parameters/AdamW at
step 1. All owned temporary files completed; the final pending filename is
logging context, not evidence of an incomplete retained artifact.

Unique persistent ledger:
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-latent-reader-20261004.gpu-budget.json.
Session aada8ced-eed4-47b3-b7ea-a436e706fe1a settled at 400.409461 seconds,
two cards, **800.818923 occupied GPU-seconds** (0.222450 card-hours).
At this window's settlement all 15 sessions were settled, total 21672.648975
of 25200, global remaining 3527.351025. The later separately approved
continuation has its own followup_results report and additional-source settlement.
This schedule uses the original18,000 source: baseline 16643.442122 plus this
800.818923 gives 17444.261045, source remaining 555.738955. Its fixed 1,000 route
has 199.181077 remaining. The separate daily-only7,200 extension used 4228.387930
and retains 2971.612070; it cannot fund this schedule.

Wrapper/parent PIDs25504/25557 exited. Initial release12:52:36 showed both 0 MiB,
no processes and transient GPU0 utilization34%. Read-only confirmation12:55:27
showed both 0 MiB/0%, with no compute processes. No additional GPU call followed.

Storage check observed shared filesystem5.1P free, which is not personal quota
headroom. Actual liwei quota was 25,657,708,056 kB over soft25,000,000,000,
hard0, grace about 1week9h50m, files8,773,012/10,000,000. This run preserves
approximately 3.6 GB in a few consolidated files. No scientific data/results
were deleted. Proposed continuation and its separate approval are recorded in
shared_original_full_state_schedule_followup_proposal_20261005.md.
