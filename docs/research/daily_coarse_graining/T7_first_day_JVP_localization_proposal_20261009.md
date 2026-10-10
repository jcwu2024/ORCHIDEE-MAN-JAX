# T7 First-Day Forward-JVP Localization

Updated: 2026-10-10. Source/artifact filenames retain their 20261009 origin.

## Decision and purpose

The completed [B1/B4 T7 diagnostic](T7_B1_state_JVP_completion_results_20261009.md)
found repeatable forward-directional differences at state coordinates 434 and
2382, in addition to reverse-mode inconsistencies. The four-window batch has
measured 2.555005x full-AD component speed, but remains ineligible for complete
updates under the unchanged state gate. A reverse-accumulation intervention
alone is therefore not the next selected explanation or repair.

Compare the canonical **first daily transition** in B1 and B4 using the same
four TRAIN windows 742/275/482/299, weights and first-day inputs. Measure all
next-state values and floating observations, and their native forward JVPs.
This locates whether a difference is already visible before subsequent days
and the trajectory loss. It does not change the training objective, select a
derivative reference, or certify the whole T7 computation.

The saved T7 JVP executables return the loss tangent only; their auxiliary
outputs are primal values through `jax.jvp(..., has_aux=True)`. They cannot
provide per-output or per-day tangents without a new diagnostic graph. Keep
those successful executables and all full VJP arrays intact.

## Fixed computation and comparisons

Use `make_shared_recurrent_objective` with the existing exact candidate head,
source-reuse adapters and guarded turnover overlay; obtain its public
`RolloutLoss.step`. Invoke that canonical transition, rather than implementing
a new process, updater, trajectory loss or observation cotangent.

Extract day zero from each already prepared seven-day case. B1 uses sample 3,
artifact 299; B4 uses `vmap` over all four independent states/discrete inputs
and first-day templates/retained inputs, with one shared parameter tree.
Use the existing four FP64 unit directions at 434/2382/2621/control 200,
isolated to B4 sample 3. Neural arithmetic remains FP32 and physics FP64.

Preserve full next continuous state and every canonical observation. Floating
outputs participate in native JVP; integer, Boolean and static outputs are
retained as auxiliary values and compared exactly. Compare the B1 output with
B4 sample 3 at its **natural per-case scale**, with no factor of four: a daily
transition is not the sample-mean loss. Keep any duplicated state observation
explicit in the result schema.

Run three alternating cycles of four directions for each route: 24 JVP calls.
Save every complete primal/auxiliary/tangent result before comparisons, then
check cross-route values and tangents, own-route repeats, shapes/dtypes, finite
values and all discrete/source-mask outputs. Numeric comparisons retain
`rtol=1e-5, atol=1e-6`; no failure is waived. Record complete per-leaf failure
counts and bounded position samples, with full arrays retained remotely.

Check the canonical learned-output nonfinite count is zero and record budget
residuals. Budget agreement and a finite daily observation are not substitutes
for the full T7 fourteen hard gates. This component does not measure loss,
full parameter VJP, optimizer updates or accepted training throughput.

## Executable and resource plan

Immutable checkout:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1`
at c599236. New isolated overlay:
`runtime/transfer/T7-first-day-jvp-20261009`.
Reuse common exact sources/data under `runtime/transfer/shared-latent-reader-20261004`
and the existing prepared manifest. No Teacher regeneration.

Implementation:

- `scripts/hpc/shared_latent_T7_day_jvp.py`: daily JVP/output comparison helpers.
- `scripts/hpc/probe_shared_latent_T7_day_jvp.py`: exact asset binding, first-day
  extraction, one compile per route, preservation and measurement.
- `scripts/hpc/run_shared_latent_T7_day_jvp.py` and `.sh`: canonical runtime,
  target argument expansion, actual exit status and independent occupation.
- Focused component and launcher tests, plus target CPU input/metadata checks.

Approved **new** free grant: one gln01 V100, OMP4/OpenBLAS1, at most 5400
cumulative GPU seconds (1.5 card-hours), CNY 0, no paid fallback. Entire startup,
actual deployed runtime preflight, lowering/compilation, measurements, ordinary
repairs, saving and exit count against the same cap. Compiled and observed
memory must remain below 24GiB.

Run the small `check_daily_training_runtime.py` on the actual GPU environment
before the changed output/JVP boundary is compiled. Its cap is 240 seconds.
Probe cap is 4800 seconds; the outer cap leaves 360 seconds for exit after the
preflight. Each route has lowering 180, backend 1800, cumulative route 2100
seconds. Two route maxima consume 4200 seconds, leaving 600 probe seconds for
assets, preservation, measurement and comparison. There is one backend attempt
per route and no additional full-T7 graph. These are termination caps, not
predicted runtimes or guaranteed success. The measured full-T7 B1/B4 backend
times of about 993/968 seconds justify avoiding the failed 900-second budget.

Preserve successful binaries and attempts immediately. On numerical failure,
save results and return 3; distinguish resource/runtime errors. The launcher
must retain the actual child exit code and settle its occupation ledger.
Save each successful executable's optimized HLO as one compressed artifact
within the same preservation budget, after the binary. This supports later
CPU inspection of the compiled first-day arithmetic without another GPU load;
it does not establish the cause of the existing full-T7 discrepancy.
No old approval, balance or ledger is reset or borrowed. The completed B1
scope used 1780.743670 seconds; its unused balance is not permission for these
two new graphs. The user's direct 2026-10-10 approval supplies only the new
bounded scope below; CPU preparation and the false template do not supply it.

## How the result changes the next action

If full first-day tangents differ, continue localization inside this daily
transition using its failed outputs and exact saved inputs; do not first compile
another full T7 reverse graph. If primals also differ, retain that distinction
and inspect forward arithmetic alongside differentiation.

If the daily comparison passes, the discrepancy has not been reproduced at
this boundary. Move the next localization to later recurrence or the public
loss using the measured result; this proposal does not automatically authorize
another horizon, graph, precision or reference. A passing first-day component
does not admit batching. The eventual candidate still needs complete loss,
all parameter/state gradients, physics, memory, stable full-AD timings and the
same-workload complete-update gate.

## Execution status

Implementation, parent artifact review and target CPU acceptance are complete:

- 27 focused analytic/launcher tests passed in ORCJAX. They include native JIT
  JVP, natural per-case scale, sample isolation, missing B4 axis rejection,
  exact masks, failure preservation, JSON coordinate types, actual exit-code
  forwarding, scope guards and the 24GiB sampled-memory boundary.
- Real test01 readiness verified all 215 first-day input leaves: B1 sample 3
  exactly equals B4 sample 3, with complete `(3854,)` FP64 state and the four
  original artifact identities. No neural graph was traced.
- Actual gln01 Bash `execute --dry-run`, target Python dry-run, child plan-only
  and rejection of the false approval template passed. No CUDA/model run or
  GPU preflight was performed by those checks.

The [verification record](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.verification.json)
and [real input receipt](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.cpu-readiness.json)
are saved locally and remotely. Parent review corrected B4 axis admission and
NumPy coordinate serialization before deployment. The source archive
`T7-first-day-jvp-20261009.source.tar.gz` has SHA256
`c947e671e61e214234fa0258da6e5e26a34ecc1eb49000229fa58735ed0a74bf`;
all seven members match the final local/deployed sources. Four are the new
component/probe/launchers; three are unchanged prior metadata/path dependencies.
Completed old overlays remain intact.

The user explicitly replied “批准。” on 2026-10-10. The
[separate true approval](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.approved.json)
binds this exact independent free gln01 single-V100 5400-GPU-second/CNY 0 scope,
probe 4800/preflight 240/exit reserve 360, two new graphs each one backend
attempt, 24 JVPs, zero VJPs/updates and no paid fallback. Weights, inputs,
FP32 neural/FP64 physics, natural per-case scale and strict gates are unchanged.
The [false template](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.approval.json)
stays `approved=false`. Execution completed both graphs and all 24 calls,
one compile per route, no retry/VJP/update. The result is `numerical_failed`,
actual launcher/settled ledger exit 3. Independent scope charged
2124.572662062943/5400 GPU seconds/0.5901590727952619 card-hours/CNY 0, within cap;
GPU released, no prior funding reused. Unused 3275.427337937057 seconds grant
no new graph. [Actual results](T7_first_day_JVP_localization_results_20261010.md)
own complete comparisons, runtime/preflight/memory, actual occupation/release,
saved arrays/binaries/HLO and independent CPU evidence. Finite tangent
differences reproduce within the first daily transition for all directions;
raw state 20 nonfinite remains a separate failure. Accepted v3 metadata/source
audit confirms defined failures and exact repeats; no cause, derivative reference,
precision change or adoption follows.

Completed approved command from gln01 (preserve; do not rerun):

```bash
T7_FIRST_DAY_JVP_GPU_INDEX=1 bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-first-day-jvp-20261009/run_shared_latent_T7_day_jvp.sh execute
```

Working directory is the immutable c599236 checkout declared above. The
launcher resolves that checkout and the new overlay explicitly. Result prefix
is `T7-first-day-jvp-20261009` under the existing diagnostic output directory.
The graph service was unavailable (`Transport closed`, generation unknown);
exact source fallback and actual input/runtime-interface checks support this
preparation, without a graph coverage claim or model acceptance.
