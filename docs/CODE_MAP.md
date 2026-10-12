# Code Map

Updated: 2026-10-12. Bounded task navigation, not an exhaustive inventory.
First-day/raw-neural JVP execution and CPU acceptance are complete; preserve graphs/HLO/full results.
[NEXT_STEPS.md](NEXT_STEPS.md) controls the handoff/stop and closed-scope boundaries;
[current-status.md](current-status.md) owns measured facts and limits.

## Source and Teacher

| Path | Responsibility |
|---|---|
| `fortran_source/ORCHIDEE/` | Fortran process/branch truth |
| `fortran_run_scripts/paper_250919/`, `reference/`, `traces/`, `docs/source_audits/` | Paper protocol, source audit/reference/trace provenance |
| `jax_orchidee/cli/`, `runners/`, `driver/` | Public Teacher, IO and compiled lifecycle |
| `jax_orchidee/sechiba/`, `stomate/`, `coupled.py`, `ad_primitives.py` | Source-order physics and canonical derivatives |
| `jax_orchidee/parameters/` | Parameter/PFT catalog and stable identities |

Teacher and surrogate research are separate. Reuse coherent v6 fixed Teacher
and prepared shards; no v5 joins/regeneration. Scoped PFT14 closure is not
full669-point Fortran/all-branch acceptance. PFT2–13 remain structural only
and fail closed. A named old result is not current execution authority.

## Public daily model

Paths are under `research/daily_coarse_graining/daily_operator/`.

| Owner | Responsibility |
|---|---|
| `model.py::bind_daily_operator` | Canonical input→head→conservative updater→retained-tail binding |
| `shared_transformer.py`, `shared_latent_reader.py` | Original masked shared transformer and opt-in latent32/reader64; public factory/field execution |
| `shared_process_head.py::SharedTransformerProcessHead` | Typed neural process outputs bound to physical processes |
| `shared_decoder_expert_execution.py`, `shared_decoder_macro_packed.py`, `shared_backbone_execution.py`, `shared_backbone_projection_reuse.py` | Original field/expert execution, identical projection/source reuse and stable serialized identities |
| `shared_persistent_outputs.py`, `shared_cimean.py` | Persistent endpoints and source-backed annual-Ca identity |
| `shared_trajectory_loss.py`, `shared_training.py`, `shared_recurrent_training.py`, `rollout_loss.py` | Shared science terms, daily/joint loss and public native checkpointed rollout AD |

Original:64,884,481 parameters, backbone768x8, coordinate decoder384x2.
Selected diagnostic prototype:66,123,713 parameters, expert32x384 latents,
private reader64x2, canonical field/chunk64. Physical state carries inter-day
memory. Preserve canonical six-hour forcing/binding/updater/tail, potentially
active outputs/masks and complete parameter/state AD; no default adoption.

## Current T7 diagnostic and shared runtime

Paths are under `scripts/hpc/`.

| Owner | Responsibility |
|---|---|
| `shared_transformer_runtime.py`, `canonical_training_runtime.py`, `retained_training_runtime.py` | Prepared inputs/calibration, bootstrap and retained templates |
| `target_oriented_training_prepared.py`, `training_artifact_io.py`, `runtime_diagnostics.py` | Prepared artifact identities, shared host artifacts and diagnostic observations |
| `shared_daily_microbatch.py`, `shared_daily_dual_ad.py`, `shared_full_state_ad_execution.py` | Public sample-mean daily parameter/state AD, saved graph identity/ABI and optimizer views |
| `probe_shared_latent_reader_T7_pair.py` | Corrected original/candidate public objective, exact parameters/context/source identities, guarded turnover and saved T7/T30 full-AD graphs |
| `shared_latent_T7_batch_assets.py`, `shared_latent_T7_batch_ad.py`, `probe_shared_latent_T7_batch.py` | Four distinct TRAIN windows742/275/482/299, five dynamic inputs, `grad(mean(vmap(public checkpointed loss)))`, saved-B1 denominator and complete B4 comparisons |
| `probe_shared_latent_T7_state_jvp.py`, `shared_latent_T7_state_jvp.py` | Completed B1/B4 directions 434/2382/2621/control 200 at sample299; FP32 neural/FP64 physics/state directions; native B1 tangents plus separate /4 versus B4 mean, graph identities/full-VJP endpoints, tangent/primal/hard/self/cross comparisons and retained failures |
| `run_shared_latent_T7_state_jvp.py`, its shell launcher | Actual runtime preflight, isolated overlay resolution, independent approval-bound3600-second occupation, route/load/compile caps, actual memory/exit observations and saved-only VJP loading |
| `shared_latent_T7_day_jvp.py`, `probe_shared_latent_T7_day_jvp.py`, `run_shared_latent_T7_day_jvp.py`, `run_shared_latent_T7_day_jvp.sh` | Completed canonical `RolloutLoss.step` first-day B1/B4 full outputs/native tangents at natural scale/no /4; exact assets, saved binaries/HLO/24 raw results, strict numeric/discrete/mask comparisons, actual exit 3 and settled independent occupation; no retry/VJP/update |
| `shared_latent_T7_raw_neural_jvp.py`, `probe_shared_latent_T7_raw_neural_jvp.py`, `run_shared_latent_T7_raw_neural_jvp.py`, `run_shared_latent_T7_raw_neural_jvp.sh` | Completed canonical neural prefix/all 161 raw fields/exact masks; 24 native JVPs, binaries/HLO; accepted v3-final audit retains active primal/tangent failures, exact repeats/isolation; scope closed/handoff/stop |
| `audit_shared_latent_T7_input_numerics.py`, `probe_shared_latent_T7_batch_numerics.py` | Local bounded rebuild/summary/normalization audit and unchanged affine scalar/vmap/lax.map discriminator; independent FP64 oracle, per-round complete arrays, actual AD pairing and nonfinite JSON/NPZ failure retention; synthetic method evidence, no real-model/GPU acceptance |
| `capture_shared_latent_T7_summary_operands.py` | Thin CPU-only identity-checked original TRAIN window0/day0 loader and canonical rebuild/assembly/normalization/PFT/pool summary boundary; sample3 raw434 preprocessing JVP and seven-key affine NPZ; actual eight-file sources now captured/CPU admitted; Windows restores saved PosixPath as lexical PurePosixPath |
| `run_shared_latent_T7_batch_numerics.py`, its shell launcher | Deployed b676353 NumPy-only capture admission, independent one-shot1200s scope, host UUID/V100/24GiB monitor, preflight/probe owned groups and independent shell deadlines; target CPU shell/private-phase checks accepted, container/PJRT/GPU pending; both cards busy, no GPU/preflight/open scope/ledger reservation |
| `run_shared_latent_T7_batch_numerics_paid.py`, its shell and `slurm_shared_latent_T7_batch_numerics_paid.sh` | Local v2 candidate: Supervisor-owned original-mask runtime discovery, shared startup/preflight deadline, UUID-only CSV/XML/monitoring, actual-job kernel/cgroup association and explicit container contract. Broad numeric-mask association rejects unknown; unknown UUID retains mutex/full charge. Original science/count/memory/release gates unchanged. Historical frozen c8a44c3 job15812634 failed at its first CSV query; consumed v1 scope cannot migrate. No v2 deployment or execution. |
| `shared_affine_gpu_admission.py`, `shared_affine_cuda_identity.py`, `freeze_shared_affine_paid_scope.py` | Persistent atomic cross-scope mutex and bounded free-state admission; v2 runtime count1/PCI→physical driver UUID discovery and repeated preflight/affine binding, total driver enumeration recorded separately. Legacy strict/v1 defaults retained; accepted-commit→false scope/source manifests remain explicit, with no old-approval migration or stale-lock takeover. |
| `shared_gpu_budget.py`, `shared_latent_T7_memory.sh` | Shared persistent reservation/charge/termination helper and read-only NVML sampling; current launcher binds a separate grant, never resets old ledgers |
| `analyze_shared_latent_T7_state_repeats.py` | NumPy-only21-array state extraction and per-session repeat/range evidence; no GPU/model/AD calls |
| `probe_shared_latent_T7_dispatch.py`, `shared_latent_T7_dispatch_ad.py` | Completed saved-B1 synchronization comparison; cheap device-side merge retained for eventual complete-workload verification |
| `prepare_shared_latent_full_update.py`, `probe_shared_latent_dual.py`, its worker | Corrected saved-graph readiness and changing-weight complete-update integration; deferred until current gates/remaining corrected daily graph are resolved |
| `probe_shared_forward_reverse.py`, `analyze_shared_update_trace.py`, `profile_shared_latent_saved_T30.py`, `analyze_shared_execution_trace.py` | Existing forward/additional-AD and saved-T30 execution evidence; source/profile hints do not isolate pure reverse or exclusive module time |

Relevant regression: `tests/unit/test_shared_latent_T7_state_jvp.py` exercises
small independent AD, direction/scaling/bridge/caps, deployed-interface compile
options and launcher admission. First-day helper/launcher tests are
`tests/unit/test_shared_latent_T7_day_jvp.py` and
`tests/unit/test_shared_latent_T7_day_jvp_launcher.py` (27 focused passes).
Actual model/GPU checks remain separate.
The supervisor accepted the [2026-10-11 local preparation](research/daily_coarse_graining/T7_batch_numerics_local_preparation_20261011.md)
as local methods/preparation only; it links input6/matrix6 tests, final CPU v3
evidence and exact missing asset identities.
Its [GPU plan](research/daily_coarse_graining/T7_batch_numerics_GPU_confirmation_plan_20261011.md)
now has fresh recovery input admission and a deployed bounded wrapper;
target CPU shell/private-phase readiness is accepted, while actual
container/PJRT/GPU bootstrap remains pending, without a selected repair.
The [deployment result](research/daily_coarse_graining/T7_batch_numerics_GPU_deployment_results_20261011.md)
owns the separate approved1200s scope's current `approved_waiting_resources`
status, deployment paths/identities and saved CPU receipts. Both cards were
busy at the final refresh; no GPU launch/preflight/open scope/ledger reservation
occurred and the one-shot is unconsumed. No further polling/automatic monitoring.
The [single-GPU terminal result](research/daily_coarse_graining/T7_paid_single_GPU_execution_results_20261012.md)
owns the latest paid status: sole approved n04/4CPU/1V100 job15812634 at
frozen c8a44c3 ended FAILED2:0 after6 allocated seconds. The first
`nvidia-smi -i 5` CSV query returned6/“No devices were found”; no XML,
actual-job cgroup, Supervisor, preflight, CUDA/PJRT or affine12/91 ran.
Numerical status is not_executed; OS opening denial is not assessed.
Scope closed/consumed, mutex absent; live GPU cleanup/peak memory unobserved.
Allocation0.0016666667card-hours, additive estimate CNY0.0041333333; bill unknown.
No retry, mapping query, repair/redeploy, free fallback, monitoring or training.
The [preparation/proposal](research/daily_coarse_graining/T7_paid_single_GPU_preparation_results_20261012.md)
retains historical false scope, exact argv and88 CPU cases; these do not
establish GPU or scientific acceptance.

The [local v2 identity repair](research/daily_coarse_graining/T7_paid_GPU_identity_local_repair_20261012.md)
now owns the candidate method: Supervisor-owned original-mask discovery,
runtime count1/PCI→driver physical UUID, UUID-only NVML and explicit actual-job
association. Numeric masks with broad device access fail mapping_unknown.
V2 explicitly separates runtime visibility from total driver enumeration;
old consumed grants cannot migrate. Local mocks establish no deployed/API,
allocation, memory, numerical or release acceptance. No new execution is authorized.

The [paid v3 terminal result](research/daily_coarse_graining/T7_affine_paid_GPU_recovery_results_20261012.md)
owns historical v3 approval/execution: sole fixed n01 job15806646 at58a2d938 ended
FAILED2:0/7s. CSV/UUID XML succeeds (driver525.85.12/minor0); unchanged wildcard
cgroup-device gate rejects isolation before Supervisor/preflight/CUDA-PJRT/probe.
Raw devices.list was not retained; exact configuration cause remains unproved.
Execution evidence is `outputs/T7-affine-paid-recovery-execution-20261012/`:
complete Slurm/stdout/XML/archive, attempts, false/open/closed scopes, saved-only
summary and closeout. Scope closed/consumed, mutex absent, no GPU cleanup
observation, no retry. Preparation30+10 tests and test01 CPU evidence are retained
in the [v3 proposal](research/daily_coarse_graining/T7_affine_paid_GPU_recovery_proposal_20261012.md).
Old v2 job15782177/f4ae FAILED2:0/8s/no probe/unknown child stderr stays unchanged;
free b676 approval remains paused/unconsumed. No repair/adoption/training follows.

Paid v2 `allocation_epochs` and `allocated_tres` admit22.05.7 scheduler time
and actual allocation detail; `--scheduler-allocation-epochs` is CPU metadata
only before outer deadlines. `tests/fixtures/slurm22/` records trimmed actual
CPU-job schema/provenance; synthetic GPU fixtures do not prove GPU allocation.
New paid tests are `tests/unit/test_shared_latent_T7_batch_numerics_paid.py`
and `tests/unit/test_shared_latent_T7_batch_numerics_paid_shell.py`;
`test_shared_affine_gpu_discovery.py`, `test_shared_affine_single_gpu_contract.py`
and `test_shared_latent_T7_paid_startup_evidence.py` cover local v2 binding,
association, owned lifecycle and failure evidence. They mutate
only isolated temporary fixtures. Actual Slurm allocation/device runtime and
management-chain admission are live gates; the current b676 deployment remains unchanged.
The [GPU preparation report](research/daily_coarse_graining/T7_batch_numerics_GPU_preparation_results_20261011.md)
owns the current launcher/tests/scope/manifest and three-file test incident.
Original capture NPZ/receipt remain missing; new `summary-recovery.v1.*` six arrays
match independent preincident SHA evidence, with fresh complete-file identities.
The unchanged capture source is restored; no whole affine probe was rerun.
The [previous early summary_input package](research/daily_coarse_graining/T7_summary_capture_results_20261011.md)
owns that extractor/fixture inventory and completed eight-file retrieval,
real `(4,257,4)` capture and unchanged CPU probe. Its
`tests/unit/test_shared_latent_T7_summary_operands.py` passes11 cases including
Windows historical-path deserialization; CPU JAX0.10.1 passes12 measurements/
91 saved arrays, independent oracle/AD/isolation and all cross/repeat checks.
The extractor still supports only sample3/raw434. That capture source was
`70619b0a...901072`, execution HEAD `9de13774...b497`; the previous fixture and
missing-source identities remain historical. The earlier method was accepted;
the supervisor accepted that seven-file stage, committed/pushed as `048c963...1738`;
outputs and historical assets remained outside Git.
The early CPU pass does not exclude later/GPU kernels or resolve original T7 gates.
Runtime checkout is c599236 at `runtime/checkouts/shared-split-width-v1`;
unchanged assets are in `runtime/transfer/shared-latent-reader-20261004`.
Named results preserve overlays, ABI identities and exact binary paths.
The B1 completion's separate 3600-second grant is closed. Its NumPy analyzer is
`outputs/diagnostics/shared-latent-reader-20261004/analyze_T7_state_jvp_completion_20261009.py`.
It checks native/scaled arrays, complete numeric/static tree topology, primal
bridges, strict comparisons, actual occupation and NVML. The named completion
report links its compact JSON and CPU receipts. The selected
[first-day results](research/daily_coarse_graining/T7_first_day_JVP_localization_results_20261010.md)
own completed evidence, raw paths, runtime/occupation/release and limitations.
`outputs/diagnostics/shared-latent-reader-20261004/analyze_T7_first_day_jvp_20261009.py`
is the independent NumPy-only saved-array analyzer; seven target guards pass.
The [raw-neural result](research/daily_coarse_graining/T7_raw_neural_JVP_results_20261010.md)
links accepted v3-final audit, raw arrays, binaries/HLO and settled resources.
Its batch-dependent FP32 contractions remain a hypothesis; mapped scalar
execution is discussion-only, unimplemented/unmeasured. Preserve the
[handoff](research/daily_coarse_graining/daily_speed_handoff_20261010.md) and stop.

## Retained older evidence and navigation

Original full-state scheduling/leaf CUDA/JVP source families, V4/V7 learning
and science families, and previous precision/fanout/checkpoint/compile probes
remain bounded historical evidence. Their closed windows and failures are in
named reports, the
[variant registry](research/daily_coarse_graining/shared_execution_variant_registry_20260930.md)
and [history index](research/daily_coarse_graining/HISTORY_INDEX.md).
Consult a named family when needed; do not read the script/report collection
as current context or reopen a closed architecture/precision route implicitly.

Preserve public module boundaries, serialized prepared-data class/module
identities, exact Teacher/scientific arrays and successful graphs. Map edits
do not authorize asset deletion. Follow
[RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md) for current facts and historical
recovery; environment/server rules remain in AGENTS and
[deployment-explore1000.md](deployment-explore1000.md).
