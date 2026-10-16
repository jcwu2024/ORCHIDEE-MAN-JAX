# Repaired current-model T7 precision admission: terminal input failure

Updated:2026-10-16. **Job15895454 FAILED1:0 after140 allocated seconds.**
The first lower/trace attempt failed while opening a missing retained-input
forcing asset. This is an input-preparation/runtime failure, not a precision or
batching numerical counterexample. The one authorized attempt is consumed and
this diagnostic line is stopped: no retry, new proposal, GPU run, training or
watcher follows. Subsequent CPU input preparation grants no GPU authorization.

## Terminal execution and accounting

Actual sacct job/batch:10:34:48→10:37:08+08, ibc13b02n01,4CPU/1GPU,
FAILED1:0,140 seconds. The extern step independently COMPLETED0:0/140s and is
not added again. Terminal scontrol was purged according to the collection owner;
no separate terminal scontrol raw capture is saved in this evidence directory.
The actual RUNNING allocation response remains saved in the allocation diagnostic.

Discovery passed3.0113496s. Synthetic preflight passed60.718841s internally
(wrapper65.535891s), with real_model_checked=false. Probe wrapper65.850928s
ended childexit1. Result status execution_failed, stage failure_saved,
elapsed58.981702s; sole_lower_started was29.289971s. The traceback reaches
retained-tail driver bundle construction and fails opening
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/data/forcing/cruncep_twodeg_1991.nc.
Results are empty: completed traces/compiles/calls0, trace attempts1,
compile/call attempts0. Native JVP/VJP, FD, repeats, full3854 comparison and new
physics/precision gates are unmeasured. Strict state20 rawNaN remains a separate
unwaived failure; no new trajectory was audited.

Scheduler140s gives0.0388888889 GPU card-hours,0.1555555556 allocated CPU-hours
and estimated CNY0.0964444444. The supervisor's more precise elapsed clock is
140.217637777s, with its own estimate CNY0.096594372691. These are distinct
accounting clocks, not two charges; actual bill is unknown. Both remain below
approved CNY3.72. Sampled peak memory345MiB is not full-model memory admission.
Supervisor runtime_failed/exit1, within_total_cap=true; ledger closed/retry=false.
Owned groups48313/48392/49486 were released and active groups are empty.
Selected UUID GPU-a6321ffd-0e5a-8be8-1703-86713168c895 release sample has
memory0MiB/utilization0/processes[], release_verified=true. These actual release
records supersede the earlier startup snapshot's absence of release evidence.

## Root cause and three preserved identities

The prepared pickle's template is already a CanonicalRetainedTailAdapter.
Its static_owner carries runtime_year1991, lifecycle_start_tstep48 and
result_day_index2; its driver context points to shared-root config
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/configs/orchidee_man_250919.yaml.
With ORCHIDEE_DATA_ROOT unset, config loading resolves data relative to that
config's project root. That shared data directory contains only README, without
forcing or INPUTDIR_ZZ, so lower-time retained bundle construction fails.
The captured chain is lower→combined_native→public/canonical rollout→retained
tail dynamic_transition→paper_1961_driver_later_day_runtime_result→
_paper_1961_later_day_transition_inputs (compiled-forcing branch,
orchestration line9938)→_paper_1961_step_bundle_from_context(1991,48)→
read_forcing_model_step_cached→_read_forcing_land_cache→xr.open_dataset.
Annual CO2 resolution and static payload construction are companion consumers
covered by the subsequent CPU preparation check.

These three identities are separate and remain unchanged:

| Layer | Preserved identity |
|---|---|
| Actual sampled T7 case | artifact299/window0, landpoint345.0-069.0, year1967/day203–209, dates1967-07-22–28; case SHA2568c0ed07cd295a28a2f0ca0a9ae46b1904fc59328863aa4c89af9e6f83e820448 |
| Retained static owner dependency | runtime_year1991/tstep48; actual domain nbindex1, forcing indices[[145,46]], lon111/lat−3; consumes the1991 forcing file |
| Serialized first-bundle template | year1961/CO2 317.27; a third static template identity |

The1991 asset is a retained static-bundle dependency, not the sampled T7 window's
raw meteorology. Do not replace it with1967 forcing or relabel the1961 template.
The saved run_def path contains landpoints/291.0-087.0; that is a path identity,
not the actual domain coordinates above. Initial-state SHA256 remains
54d01d0f376b0ceed8a843740328dba9d61d2a8989507436bda1a0d16c624f5d.
The bounded field audit below resolves the selected dynamic-input bindings;
broader scientific calibration of retained structural constants is unassessed.

## Isolated asset recovery and CPU preparation boundary

Canonical local1991 NetCDF source:851746768 bytes, SHA256
bf2e53a34b77e55badf4f4527c36a7c1c9da61636d9317b43288080ad28e790d.
Canonical CO2 text:3816 bytes, SHA256
ad5e96976dd89d8e3ddf9cbc57293acdab8d07a33a8e016f3285055a3df02c31.
The files were copied as regular files, without symlinks, to isolated
runtime/transfer/T7-current-model-forcing-asset-recovery-20261016/inputs/
under forcing/ and INPUTDIR_ZZ/. Source provenance establishes canonical local
origin and transferred-byte identity. The original run did not independently
pin raw forcing hashes; these hashes do not retroactively establish that identity.
Original shared-root data, a833f987 checkout and old source cache remain unchanged.

The normal generic-wrapper dry-run now calls the CPU retained-input consumer by
default, via prepare_shared_current_T7_retained_inputs.py and the precision probe.
Explicit --retained-data-root is required; this checks the step-bundle read and
static payload construction used by the later-input branch,
beyond file existence checks. The full later-input transition was not run.

Actual test01 attempt3 passed, exit0, timeout300s, through the ordinary generic
wrapper --stage dry-run at source/import checkout
f2242fc5a7657a3f3061d85a5e1b998663c8f944 in
runtime/checkouts/T7-current-model-retained-input-cpu-20261016.
Outer SSH wall time40.46245599s includes startup and is not a pure-function timer.
The receipt records retained_inputs_consumed: remote regular-file byte sizes/
hashes match the canonical sources, resolved1991 CO2 is354.47ppm, all checked
numeric static-payload fields are finite and missing fields are empty. Existing
lon/lat shape(1,1) and dtypes are preserved. All three identities and original
case/state SHA256 values above are unchanged. This is bounded forcing/CO2 read
and bundle/static-payload validation: no neural or physical transition,
scientific appropriateness assessment, full-model traces/compiles/calls or their
attempts (all0), GPU execution or new state20 trajectory. State20 remains rawNaN.

The separate actual Bash wrapper→supervisor build_plan→runtime-child probe parser
chain check preserves explicit root, wrapper shell exit0; missing root rejects
with parser exit2. Its runtime-child/container fields are metadata only, not
executed. Earlier CPU attempts remain recorded: attempt1 failed the existing
source-cache path constraint; attempt2 failed the added validator's shape
assumption. Corrections changed only the new CPU checkout/argv/validator, without
scientific or input-data changes. Focused local regressions passed19 tests in
3.04s with one existing pytest warning; the previous144-test suite was not rerun.
The CPU checkout also borrows a833f987 Git objects; retain that dependency.
No GPU authorization is restored by these completed preparation checks.

## Effective sampled inputs: bounded source and CPU audit

The subsequent test01 audit exited0 within timeout180s; outer SSH wall time
123.17926955s includes startup. It used actual input assembly,
Adapter._dynamic_day_inputs and driver transition-input construction without
mocking overrides, at source f2242fc5a7657a3f3061d85a5e1b998663c8f944.
Script audit_shared_current_T7_input_binding.py SHA256 is
a336ef37ac9a0acd63106d59c804b2ef4f4a3e068864972cbe8df24ac7010a56.
Seven saved canonical day states supplied previous/token inputs: this is input
coverage, not a predicted recurrent trajectory. Physical/neural calls, JIT,
model traces/compiles/calls and loss evaluation were not run. Neither the19-test
nor144-test suite was rerun for this audit.

| Effective input / boundary | Source and bounded evidence |
|---|---|
| Sample CO2 | All seven days are1967/day203–209 with annual CO2 321.62ppm; annual conditions feed the input token at strong_baseline_architecture.py:1211. Base354.47ppm is not that token value. |
| Daily weather | Six daily weather fields are recomputed from the native sampled records, then deterministically override _retained_tail_fast_target and the daily fold. research/daily_coarse_graining/canonical_retained_tail.py:273–285 installs half-hour/daily-fold/okleak replay; generic scan replacement is not the exclusion argument. |
| Coordinates / landpoint fields | Effective metadata.lalo=[[-21,165]]. Twenty landpoint fields match sampled overrides exactly across seven days,140/140 checks. The base domain111/−3 remains distinguishable metadata. |
| Date / season state | Dynamic day_index feeds science_daynumber, julian_diff, GDD and end-year paths. The1991 runtime year remains only for base bundle reads at this boundary. Prepared day inputs replace old restart/season templates, then named season fields are replaced by prior state; actual forecast flags/transitions were not executed. |
| Base weather/CO2 exclusion | orchestration.py:1578–1623→coupled.py:1330–1422,1229–1283 constructs entry-source weather, but the returned boundary carries only clay/bulk-density/poor-soil plus routing into okleak inputs, and contfrac into output inputs. Entry source is not returned. orchestration.py:8746–8772 reads only veget_max/zsoil/zfsoil/carbon-old/prod10/prod100/contfrac/one-day boundary fields. |
| Active fixed constants | Old-context PFT/config flags/run scalars and vertical grids diaglev/zlt/znt remain active structural/scientific constants, not merely shapes. The shared context/contract declares fixed layout/parameters; every constant was not independently recalibrated. |

Source paths for the abbreviated references above and below:
`orchestration.py` = `jax_orchidee/driver/orchestration.py`;
`coupled.py` = `jax_orchidee/coupled.py`;
`strong_baseline_architecture.py` =
`research/daily_coarse_graining/daily_operator/strong_baseline_architecture.py`;
`state-JVP runner` = `scripts/hpc/probe_shared_latent_T7_state_jvp.py`;
`precision_assets.py` = `scripts/hpc/shared_current_T7_precision_assets.py`;
`shared_latent_T7_raw_neural_jvp.py` =
`scripts/hpc/shared_latent_T7_raw_neural_jvp.py`.

Thus the canonical1991 recovery restores template readability. Source closure
and real input construction establish, for this fixed sampled boundary, that
the1991 weather/354.47ppm base values and1961 template exogenous values do not
feed the retained loss inputs in place of the sampled weather/CO2/coordinates/
day fields. No such exogenous contamination was found and no behavioral repair
is indicated by this audit. This is not broad scientific acceptance or proof
that all structural constants are appropriate; the earlier false appropriateness
flag denoted unassessed scope, not a demonstrated year-mismatch error.

FP32 full-B1 state completion uses the same context/public objective with
sample3=299 (state-JVP runner:54–64,378–394,431–444); the current loader rebuilds
the same public case (precision_assets.py:157–198). The original pair identity
itself names artifact742; sampling[3] supplies299. The raw-neural prefix ignores
retained inputs (shared_latent_T7_raw_neural_jvp.py:35–56), so it supplies no
independent retained-loss exclusion evidence. Historical raw forcing bytes were
not pinned; shared case/context identity does not remove that provenance limit.

## Immutable execution history and evidence

Executed a833f987ac65e12748f50d4ee9cfd05cbc69d839, fresh one-shot approval
SHA2565e769bdf86732c1b56bf7ed34000b95d6117df1b208f5363060d82c38e7aaefb,
proposal af1cce1067b55941243028d4778380195f003a09d8d40fca481293d6ed10c1b6.
One exact-argv submission returned Submitted batch job 15895454. Approved bounds
were1V100/4CPU/5400s/24GiB/nonexclusive/no-requeue/CNY3.72, one compile/52 calls.
The checkout runtime/checkouts/T7-current-model-precision-admission-repaired-20261016
borrows old20261014-v2 Git objects; retain both. The closed old job15866925
launcher failure and consumed grant remain separate history.

Local evidence root:
outputs/T7-current-model-precision-admission-repaired-execution-20261016/.
terminal/ contains accounting-raw.txt, result.v2.json, result.v2.probe.stderr.txt,
result.v2.supervisor.json, result.v2.ledger.json, terminal-summary.json and
fixed-context-metadata.json. Startup/submission captures remain unchanged as
historical observations. The initial CRLF target-check failure was corrected
before submission; target-shell-approval-check.json records the successful check.
cpu-remote-receipts/attempt3/preparation.json records actual successful retained
input consumption; cpu-remote-receipts/argument-chain.json records parser/argv
propagation and its execution limits. Remote CPU outputs remain under
runtime/outputs/diagnostics/T7-current-model-retained-input-cpu-20261016.
cpu-local-tests.txt preserves the actual19-test stdout.
input-binding/remote/input-binding.json preserves the seven-day input audit,
its code/script identity and no-transition scope.
[NEXT_STEPS](../../NEXT_STEPS.md) owns the current stopped diagnostic boundary.
