# Current Roadmap

Last updated: 2026-08-10.

This is the authority for what to do next. It deliberately separates the
forward Teacher, the daily surrogate, multi-PFT extensibility, and gradient
validation. Do not start another neural training job until Gates A-C pass.

## Four Separate Claims

The project currently contains four related but distinct goals:

1. **PFT14 forward Teacher**: the JAX half-hour model reproduces the paper-era
   Fortran PFT14 configuration.
2. **True daily surrogate**: a learned model replaces the 48 half-hour state
   transitions with one conservative daily transition.
3. **PFT-extensible modeling stack**: the Teacher can select a declared PFT
   catalog and the neural interface can consume it without redesign.
4. **Scientifically valid gradients**: automatic derivatives with respect to
   physical parameters agree with finite differences.

Passing one claim does not imply the others. In particular, finite gradients
with respect to network weights do not validate physical-parameter gradients,
and a PFT-axis-shaped network does not prove scientific support for a new PFT.

## Current Facts

- `main` at `5910ba9` contains the canonical user-facing PFT14 Teacher core;
  the byte-identical research core is `10030eb`.
- `research/daily-coarse-graining` contains the same complete
  `jax_orchidee/` tree plus research-only code.
- Gate A detached-worktree parity is complete. Seven lifecycle/multilandpoint
  cases are accepted; six are numerically identical and one has a fail-closed,
  Fortran-Oracle-bound `min_stomate` source correction.
- The old `7333b46` versus `a779663` comparison is preserved as historical
  Gate A evidence. The accepted core is now synchronized into `main`, and
  future Teacher changes must keep both branches aligned.
- The existing 669-point v5 Teacher dataset is accepted as a provenance-bound
  data product. Its old fast-day target remains useful evidence, but it is not
  the final daily-surrogate boundary.
- Previous direct-state and causal-carbon neural candidates were rejected.
- Gate E1 is accepted. Its unique canonical input assembler, typed process
  boundary, conservative updater, reusable canonical retained-tail adapter,
  and bounded real-Teacher lifecycle/AD evidence pass locally. Gate E2 now has
  a frozen coherent Teacher data release: base Markov rows, 66 typed fields,
  masks, and year-end checkpoints are emitted by the same continuous Teacher
  execution and share one manifest. The old immutable-parent sidecar proposal
  is rejected. The first candidate/control seed pair later completed and was
  stopped for practical equivalence without promotion. The formal product is
  frozen as dataset manifest v6, typed contract v3, and canonical release v2;
  full 669-landpoint, 1961-2010 generation and admission are complete. No more
  Teacher-data generation is authorized for the successor candidate.
- The current Teacher and data are scientifically scoped to PFT14. Gate B1's
  catalog foundation now gives PFTs stable IDs, source PFT/MTC identities,
  traits, parameter ownership, capabilities, and named execution layouts.
  PFT2-PFT13 remain structurally declared but scientifically unsupported and
  fail closed when active. Restart reads now verify/remap stable identity, and
  production orchestration resolves PFT rows and mangrove execution slots by
  stable ID. A compact bare-soil-plus-PFT14 cold/restart lifecycle now passes.
- Gate B2's neural PFT interface is frozen in
  `daily_neural_pft_interface_v1.json`: named parameter/trait channels,
  exact mask semantics, permutation equivariance, inactive-slot isolation,
  and variable legal `n_pft` are machine-validated. This is structural
  extensibility, not scientific support for PFT2-PFT13.
- Gate C is complete. Its 50-label water/carbon/energy inventory is frozen;
  29 non-identifiable labels are supplied by three daily-reduced compiled
  capture families. True-label constrained replay passes cold continuation,
  ordinary later-day, restart-year, mask, budget, and restart-composition
  gates without a 48-step candidate graph. No 669-point regeneration has
  been authorized yet.
- Gate D1 is complete for the canonical Teacher. Seven process cases, seven
  complete Day-2 pairs, three two-day propagation pairs, four restart-year
  pairs, and two source-extracted Fortran finite-difference pairs pass the
  frozen forward/reverse/finite-difference policy. This does not validate a
  future daily surrogate; that remains Gate D2.
- The source-driven physical-parameter candidate registry is complete. It
  classifies 34 process-owned families (91 scalar components) as 9 inversion
  candidates, 17 sensitivity-only families, 2 discrete controls, and 6
  currently non-identifiable families. The first wave remains the four
  paper-calibrated parameters; no additional public tuning channel was
  enabled.

## Dependency-Ordered Gates

### Gate A: Select One Canonical Teacher - Complete

The isolated-worktree A/B between `main` and the research branch passed its
declared matrix. The accepted decision and evidence hashes are in
[`teacher-branch-parity-acceptance.md`](teacher-branch-parity-acceptance.md).

Acceptance:

- every difference is located at its first day and field;
- default-off diagnostics have no effect;
- AD-safety edits preserve the forward primal;
- any `min_stomate` difference is classified as an intentional Fortran-source
  correction or repaired;
- one commit is recorded as the sole canonical Teacher for all later work.

This gate does not require 669 landpoints. The 669-point run remains the final
forward-release acceptance of the selected Teacher.

### Gate B: Freeze Teacher PFT Registry and Parameter Ownership

Create a machine-readable contract for every state, trait, static parameter,
and tunable physical parameter consumed by either the replaced fast processes
or the retained exact daily processes.

#### B1: Teacher-side PFT selection

Introduce a source-backed PFT catalog and per-run layout. Each entry has a
stable semantic ID, canonical Fortran PFT/MTC identity, traits, parameters,
process capabilities, and active fraction. Array positions are an execution
detail and must not be used as the scientific identity.

Most PFT differences should be data driven. A PFT that only changes existing
parameters and traits should not need a new Python module. Truly distinct
process families, such as crop/STICS, mangrove or peat controls, and phenology
families, use an explicit capability registry with source-backed dispatch.
This is the useful plugin boundary; making every PFT an arbitrary code plugin
would duplicate process logic and weaken Fortran provenance.

The current paper configuration remains one named catalog/layout with bare
soil plus PFT14. Supporting a different number of PFT slots may trigger one
JAX compilation for the new static shape, which is acceptable. Restart,
modelout, state schemas, and manifests must retain stable PFT IDs so removal,
addition, or reordering cannot silently attach state to the wrong PFT.

Teacher acceptance includes:

- parameter and state shapes derive from the selected catalog, not `NVM=14`;
- generic kernels contain no paper-case PFT14 selection;
- one active supported PFT can be removed without changing another PFT;
- two supported PFTs can coexist and preserve independent state;
- reordering execution slots with the same stable IDs preserves remapped
  results;
- source-hard-coded numbered paths remain explicit, guarded capabilities
  rather than being falsely generalized;
- each newly claimed PFT passes its reachable Fortran branch and numerical
  validation before it becomes a Teacher data source.

#### B2: Neural PFT interface

The neural interface must use explicit PFT axes:

```text
pft_state[n_pft, ...]
pft_parameters[n_pft, ...]
pft_traits[n_pft, ...]
pft_fraction[n_pft]
active_pft_mask[n_pft]
```

Each physical parameter must be assigned to exactly one class:

1. retained exact formula;
2. explicit parameterized factor around a learned environmental modifier;
3. learned conditional response supported by controlled Teacher perturbations.

Neural acceptance includes PFT permutation equivariance,
inactive/zero-fraction PFT isolation, variable legal `n_pft` shape, and no
hard-coded PFT14 index in the new model. This makes the architecture
extensible; another PFT still needs its Teacher-side capability closure,
Teacher data, and held-out validation.

### Gate C: Freeze Daily Flux Labels and Prove Non-Neural Replay

Inventory the existing v5 shards against the daily water, carbon, and energy
budget. Classify every required daily flux or transfer as present, exactly
derivable, or missing/non-identifiable. Capture only the last class.

Before building a network, feed true Teacher daily labels through one
constrained daily updater and require accepted reconstruction of the next-day
state, exact discrete behavior, nonnegative inventories, and declared budget
closure. The candidate graph may consume native six-hour forcing records, but
must not reconstruct 48 forcing steps or run a 48-step state scan.

Failure here means the boundary or labels are wrong. It is not a neural
architecture or training problem.

### Gate D: Validate Physical-Parameter Gradients

The paper at <https://arxiv.org/html/2606.07681v1> validates active
output-parameter pairs by comparing automatic differentiation with central
finite differences, using a 1% relative-error acceptance threshold and
reporting inactive parameters separately. Its demonstrated errors are below
`1e-4` for the tested active pairs.

Apply that idea in two distinct stages:

- **D1, canonical Teacher**: compare JAX forward- and reverse-mode derivatives
  with central finite differences of the same canonical Teacher. For selected
  source-critical pairs, also compare with Fortran finite differences. Cover
  smooth active pairs, inactive pairs, threshold-adjacent cases, one day,
  multiday propagation, and restart/year boundaries.
- **D2, accepted surrogate**: before parameter inversion or scientific
  sensitivity claims, compare surrogate AD with canonical Teacher/Fortran
  finite differences over the declared parameter ranges and held-out
  combinations.

Network-weight gradients and cross-day state gradients are separate training
checks. They cannot satisfy D1 or D2.

Gate D1 is accepted in
[`gate_d1_physical_parameter_gradients_v1.md`](research/daily_coarse_graining/gate_d1_physical_parameter_gradients_v1.md).
Gate D2 remains conditional on an accepted daily surrogate.

### Gate E: Implement and Train the Daily Operator

Only after A-C pass, implement the architecture in
`conservative_daily_process_operator_v1.md`:

```text
day-start state + native six-hour forcing + parameters/traits/static data
  -> process-structured daily fluxes and transfer fractions
  -> one conservative state update
  -> retained exact daily processes
  -> next-day state
```

Promote in order: supervised flux fit, one-day state/budget checks, 7- and
30-day free rollout, 365-day validation, long-period held-out validation, and
CPU/GPU performance. Do not revive a rejected prototype merely because its
training loss is easy to optimize. The binding negative-design decisions and
conditions for reopening an old idea are in
[`failed_architecture_lessons.md`](research/daily_coarse_graining/failed_architecture_lessons.md).

### Gate F: Product Acceptance

- Teacher release: complete the 669-landpoint 1961-2010 forward acceptance for
  the canonical Teacher.
- Daily-surrogate release: pass held-out landpoint, year, forcing, and declared
  parameter-range tests; long rollout and restart gates; physical budgets; and
  measured inference cost.
- Multi-PFT scientific support: repeat source closure, data generation, and
  held-out validation for each newly claimed PFT.

## Immediate Work Order

Gates A-D1 and the physical-parameter candidate registry are complete. The
formal architecture/data-readiness review accepts Gate E1. Gate E2 is governed
by
[`coherent_teacher_dataset_generation_v1.md`](research/daily_coarse_graining/coherent_teacher_dataset_generation_v1.md).
The frozen release `pft14-daily-teacher-v6` binds the Teacher source,
configuration, PFT catalog, Markov contract, typed contract, and producer
sources. A real 1961 three-day cold-start A/B proves that enabling typed
capture leaves all ordinary compiled outputs and final state unchanged; all
66 typed fields pass the fixed `atol=1e-8, rtol=1e-10` production gate, and
discrete state is exact. The current v5/v6 compatibility and data/operator
regression passes 102 tests.

The six-point 1961 coherent production pilot passed under job `14486923`.
All six workers exited zero, and the formal reader accepted 2,184 transitions,
six base shards, six typed shards, exact masks, and six year-end checkpoints.
The complete core assets occupy 28,221,594 bytes and peak process RSS was
31.4 GiB. The provisional 669 x 50-year storage projection is 146.5 GiB.
The hash-bound evidence is in
[`coherent_teacher_pilot_20260806.md`](research/daily_coarse_graining/coherent_teacher_pilot_20260806.md).

The full 669-landpoint, 1961-2010 v6 plan is built and verified. Job `14490208`
then accepted the production process shape on ten representative landpoints:
all `20/20` point-years hash-verified, same-process compilation survived a
landpoint switch without a new executable, steady hot capture was about 82
seconds per point-year, and five staggered workers completed safely on one CPU
node. Use 100 workers on 20 nodes, five workers per node, with a 180-second
local startup stagger. The measured projection is about 8.6-10 hours and CNY
60; a 12-hour request caps CPU charge at CNY 84. See
[`teacher_v6_reuse_benchmark_20260806.md`](research/daily_coarse_graining/teacher_v6_reuse_benchmark_20260806.md).

Full v6 production and admission are complete. The accepted dataset contains
669 landpoints, 33,450 point-years, and 12,208,581 transitions; all worker,
shard, typed-label, checkpoint-chain, representation, statistics, and real
gradient gates passed. Its manifest SHA256 is `5fc2c069...dc13aa99`, and the
acceptance report SHA256 is `8784ed42...aa8e3`. The full evidence is in
[`teacher_v6_full_production_acceptance_20260807.md`](research/daily_coarse_graining/teacher_v6_full_production_acceptance_20260807.md).

Do not regenerate this dataset or read and join historical v5 rows. Every
successor candidate must still enter only through the stable
`bind_daily_operator(input_assembler=canonical_assembler,
process_head=candidate_head,
retained_tail=canonical_retained_tail_adapter)` boundary and the predeclared local/small
gates. Training must bind the accepted v6 manifest, statistics, and acceptance
report. The accepted review and full 47-label data matrix are in
[`gate_e1_architecture_data_readiness_review.md`](research/daily_coarse_graining/gate_e1_architecture_data_readiness_review.md);
the executable evidence packet remains
[`gate_e1_daily_operator_skeleton.md`](research/daily_coarse_graining/gate_e1_daily_operator_skeleton.md).

The first-candidate mechanism, matched control, fixed training budget, and
bounded gates were frozen in
[`first_neural_process_head_candidate_v1.md`](research/daily_coarse_graining/first_neural_process_head_candidate_v1.md)
and its machine contract. The declared Transformer and parameter-matched
control are implemented under `daily_operator/`; local regression and the
bounded `gln01` real-v6 forward/loss/JVP/VJP, PFT, inactive-slot, and
conservative construction smoke have passed. The evidence is
[`first_neural_process_head_smoke_acceptance_20260808.md`](research/daily_coarse_graining/first_neural_process_head_smoke_acceptance_20260808.md).

The retained-tail construction-time
48-record forcing expansion has been replaced by source-exact native-record
closed-form daily inputs; cold/ordinary/restart next-state, modelout,
eager/JIT, derivative, and graph gates pass. The declared 100-update
candidate/control tiny-fit has passed with finite losses and reductions of
82.39% and 67.87%, without retaining a checkpoint. The full train/train-only
typed-field statistics with PFT-axis moments shared across active slots are
now frozen and pass identity, scale, merge, and independent-load audits.

The final no-checkpoint `gln01` training preflight also passed. Both arms ran
100 real changing train/train batches with bounded prefetch, finite loss and
gradients, and all three non-test validation slices. Streaming cost was about
`0.170-0.171 s/update`; the probe report SHA256 is
`127ef3289af776d5c5bb994f9129f5d220aa8b8574b18c50e7e8a77aba739277`.
One formal matched seed pair then completed. The axial candidate scored
`0.8219892971` and the factorized control `0.8433795294` on the runner's macro
MSE selection score. The candidate was modestly better overall and on
temporal/spatial slices but worse on the joint slice. The user stopped the
comparison for practical equivalence after this pair. Remaining seeds and the
pair's 7/30-day rollout are cancelled, neither arm is promoted, and no formal
three-seed equivalence or full-protocol rejection is claimed. The controlling
record is
[`first_neural_process_head_first_pair_decision_20260808.md`](research/daily_coarse_graining/first_neural_process_head_first_pair_decision_20260808.md).

The source-backed long-horizon biomass and productivity foundation is now
implemented in
[`long_horizon_biomass_productivity_objective_v1.md`](research/daily_coarse_graining/long_horizon_biomass_productivity_objective_v1.md).
Preserve the one-day named-process anchor and separately evaluate 1/7/30-day
cumulative GPP/NPP/respiration, signed bias, and terminal AGB/BGB/litter/soil
carbon under hard conservation/state gates.

The ownership-aware parameter boundary is also implemented in
[`daily_operator_parameter_routing_v1.md`](research/daily_coarse_graining/daily_operator_parameter_routing_v1.md).
The full six-column PFT matrix is an operator carrier, not a generic neural
feature matrix. `vcmax25` and the three maintenance coefficients reach named
fast-process factors; `alloc_min` and `residence_time` bypass the neural
encoder and reach retained allocation/mortality. `vcmax25` is additionally
routed to the retained daily `vmax_step`. The closed first A/B omitted its
fast parameter dependence and remains ineligible for Gate D2.

The successor architecture-family analysis is complete in
[`successor_process_head_architecture_options_v1.md`](research/daily_coarse_graining/successor_process_head_architecture_options_v1.md).
It proposes `source_routed_process_attention_v2` against an exactly matched
`source_routed_graph_mlp_control_v2`: both remove premature domain/global
pooling, use 101 source-routed field queries, obey the ownership-aware
parameter boundary, and train against the one-day plus 1/7/30-day objective.
The attention arm differs only by content-dependent weighting along the same
admitted source edges. A day-local RNN remains legal but is not preferred for
five forcing records; cross-day neural memory is not proposed.

The user selected the proposed attention/graph pair. Its machine-readable
source adjacency, exact configuration, bounded coefficient rule, small-fit
ceiling, and stop gates are frozen in
`manifests/coarse_graining/successor_process_head_candidate_v1.json` at SHA256
`bb8bb349...92b5c`. The local implementation has exactly `8,679,681`
parameters in each arm and passes eager/JIT forward, loss/JVP/VJP, PFT,
parameter-route, conservative one-day, synthetic free-rollout, 5+7 trajectory,
and train/train coefficient-preflight tests. See
[`successor_process_head_local_foundation_20260809.md`](research/daily_coarse_graining/successor_process_head_local_foundation_20260809.md).

The real-v6 lifecycle smoke exposed a blocking output-boundary omission. The
strict Day-2 case loses 37 Teacher-defined active-vegetation state values, and
replacing all 101 named process outputs with Teacher values does not repair it.
That all-Teacher oracle reconstructs only `87/160` fast-target leaves. A
contract ledger also identifies 14 daily-integral/instantaneous-state name
collisions, while a read-only 18-shard v6 scan finds that 62 currently carried
leaves totaling 1,007 columns actually change within the day. The evidence is
[`successor_fast_boundary_coverage_20260809.md`](research/daily_coarse_graining/successor_fast_boundary_coverage_20260809.md).

A revised canonical fast-owner schema is now accepted in
[`successor_fast_owner_reconstruction_contract_v1.md`](research/daily_coarse_graining/successor_fast_owner_reconstruction_contract_v1.md)
and `manifests/coarse_graining/successor_fast_owner_reconstruction_v1.json` at
SHA256 `4db5f828...fdfc4c9`. It classifies all `160/160` leaves and 2,855
columns exactly once: 21 existing process endpoints, 19 source-exact
derivations, 30 direct mirrors, 11 guarded/static carries, and 79 independent
neural endpoints. The extension adds 839 value scalars plus two `rveget`
defined logits and explicitly removes all 14 typed-integral/state collisions.
The four promoted neural endpoints are final-step maintenance respiration,
daily wind, final STOMATE-entry surface temperature, and restart-recomputed
inundation control; formal v6 evidence disproved their earlier derivation,
mirror, or carry classifications.

The JAX fast-owner materializer is implemented behind the opt-in canonical
retained-tail boundary. Local daily-operator/successor regression is `79/79`.
On free `gln01`, detached commit `4648a00` passed cold Day 2, ordinary Day 3,
and restart 1962 Day 1 at `160/160` exact fast leaves, with exact masks,
discrete state, and `rveget` status plus passing retained lifecycle and
water/carbon budgets. The report is
`runtime/outputs/training/successor-fast-owner-materializer-4648a00/fast_owner_report.json`
at SHA256 `f3ae02d0...f360e084`; it records zero training updates, no checkpoint,
no Teacher regeneration, and no historical v5 use.

The user approved the matched decoder extension. It is now implemented as 80
additional field queries shared by both arms: 79 value heads plus one
`rveget` status-logit head. The complete schema has 181 fields and 48 loss
labels; both arms have exactly `8,722,689` parameters and identical parameter
trees. Loss label, process domain, and source route are separate metadata.
Sixty-five fast-state heads train on day-end-minus-safe-day-start deltas; 14
daily/OK_LEAK heads train on absolute targets. Inference converts deltas to
physical endpoints before the accepted 160-leaf materializer. The value MSE
and eligible-slot `rveget` BCE share the `fast_owner.diffuco` macro group.
Local eager/JIT forward, loss/JVP/VJP, inactive-PFT, normalization, one-day
budget, materializer, and inference-source audits pass. See
[`successor_fast_owner_decoder_local_acceptance_20260809.md`](research/daily_coarse_graining/successor_fast_owner_decoder_local_acceptance_20260809.md).

The added-head train/train statistics gate is accepted. Approved `cnmix` job
`14515519` completed all eight partitions and merged exactly 8,591,565
train/train rows from 23,540 source shards. A fresh `test01` process reloaded
the final artifact with 79 added value fields, zero never-defined columns,
finite moments, complete 181-field normalization, and complete 180-field
value-loss scales. The final JSON SHA256 is
`b2e61a171156071d404d12b833130f2d0e98064564d1c3e007e16f0210734719`;
the NPZ SHA256 is
`890d1fd6110912a835c760b4fe58e9320481a929797858cad5b70b1bf03eb99f`.
See
[`successor_fast_owner_statistics_acceptance_20260809.md`](research/daily_coarse_graining/successor_fast_owner_statistics_acceptance_20260809.md).

The process-statistics implementation is present in
`daily_operator/training_statistics.py`. It vectorizes all 101 named targets
over coherent annual shards, keeps mergeable count/mean/M2 accumulators,
shares applicable output and semantic-domain moments across active PFT slots,
and writes a candidate/data/contract/source-hash-bound JSON/NPZ pair. Local
equivalence, accumulator merge, PFT sharing, bounded-loss-scale, and artifact
tamper tests pass. Approved `cnmix` job `14506391` completed all eight fit
steps over 8,591,565 train/train rows and 23,540 source shards in about 30.5
minutes per partition. Its batch then failed only at the small merge call due
to an API-wiring error; commit `9e8d4e9` corrected the call without rescanning
data, and the retained partials merged and independently reloaded on `test01`.
The final JSON SHA256 is
`9bc2cab43b3279e39a0fc2876ca298b2b371d6890986d44ef035859ca925eaab`;
the NPZ SHA256 is
`addf957afbd8338f4aa250155dcd299d568c1070b644e30279f9bf0ea37f9bf8`.
No GPU training submission is authorized without the separate approval block.
The free-`gln01` no-checkpoint probe is accepted at commit `7c555d1`. Both
arms produced 181 finite constrained fields, finite fitted losses, finite and
nonzero gradients in all 155 parameter leaves, exact inactive-PFT isolation,
and canonical-transition water/carbon residuals below `1.1e-14`. The report
SHA256 is
`5bb1f885367b3fc5183216c476a6bbe8d37122cb698d990bd5ca346006c0aaa7`.
See
[`successor_fast_owner_gpu_probe_acceptance_20260809.md`](research/daily_coarse_graining/successor_fast_owner_gpu_probe_acceptance_20260809.md).

The free bounded-small-fit GPU preflight is accepted at commit `d1ae419`.
Teacher replay passed both sampled days at `160/160`, exact masks, source-exact
undefined representation, exact discrete state, and zero water/carbon budget
residual. The candidate coefficient preflight produced four finite nonzero
gradient norms and bounded coefficients. Both matched arms then completed one
AdamW update with all 8,722,689 gradient elements finite, all 155 parameter
leaves nonzero, finite updated parameters, and zero rollout hard failures.
The report SHA256 is
`1e79bc76b97098ecf0725d94edfe144363420dba18792f05105e8f0707d543b6`;
see
[`successor_small_fit_gpu_probe_acceptance_20260809.md`](research/daily_coarse_graining/successor_small_fit_gpu_probe_acceptance_20260809.md).

The earlier two-update runner proved same-process cross-shard JIT reuse but
also exposed 86.57 seconds of repeated host preparation per hot entry. Its
20-hour paid envelope is rejected and its Slurm script has been removed.

Commit `1c63494` adds a hash-bound CPU prepared-batch boundary. Two `test01`
artifacts totaling 66,868,764 bytes were prepared without regenerating
Teacher data or reading v5. The fixed-seed `gln01` cold probe passed both arms,
all hard gates, and update-to-update reuse. On the switched hot entry,
artifact load was 0.235 seconds, replay 1.311 seconds, attention 1.138 seconds
and control 0.879 seconds. The report SHA256 is
`481581d1648699e140fa12c1b6a8e342f154ecb2216473f01a6916bd95c31f60`.
See
[`successor_small_fit_prepared_batch_acceptance_20260810.md`](research/daily_coarse_graining/successor_small_fit_prepared_batch_acceptance_20260810.md).

Cross-process persistent JAX cache reuse remains rejected: even with
`PYTHONHASHSEED=0`, a new process re-entered slow `jit_train_step` compilation.
Commit `507cb61` closes the avoidable-compilation gate. It freezes the accepted
train/train coefficients behind a hash- and identity-bound contract, removing
the 408.08-second coefficient preflight, and reuses the compiled zero-rate
training step for final evaluation. A fresh cold `gln01` probe completed in
1,442.46 seconds versus 1,908.38 seconds, a 465.92-second (24.4%) reduction.
Both remaining first train-step compile/updates took about 581 seconds; the
switched hot attention/control updates took 1.208/0.894 seconds, and both final
evaluations added zero JIT executables. All hard gates remained zero. The new
report SHA256 is
`a9ab43d855a4c035cbc52f978717de9737f801ef349981a19b50e22d2056bdc9`;
see
[`successor_small_fit_compile_reduction_acceptance_20260810.md`](research/daily_coarse_graining/successor_small_fit_compile_reduction_acceptance_20260810.md).

Commit `50bc99a` implements the resumable two-stage production plan. Four CPU
workers write disjoint partial manifests; a fail-closed finalizer alone may
write the complete manifest after exact 1-500 coverage and artifact admission.
A two-worker `test01` smoke produced byte-identical artifacts and an accepted
manifest at SHA256
`f73c714a7f9e24699f7775b5c23f221a7eda4e1b1fdb81621f927bede26b0f31`.
Each worker peaked near 4.8 GB RSS and prepared its artifact in about 70
seconds. See
[`successor_small_fit_production_resource_plan_20260810.md`](research/daily_coarse_graining/successor_small_fit_production_resource_plan_20260810.md).

Approved `cnmix` job `14517346` completed the CPU stage in `01:58:09` with all
four workers and the finalizer exiting zero. The accepted manifest contains
exactly 500 entries and 16,717,190,876 artifact bytes. An independent
`test01` audit rehashed every file and passed exact size, SHA256, path,
coverage, identity and policy checks. Manifest SHA256 is
`0182b7f7a6d5eae09025a336f26bc67ba5fb54706c3529a846d3885c16d80462`;
see
[`successor_small_fit_prepared_500_acceptance_20260810.md`](research/daily_coarse_graining/successor_small_fit_prepared_500_acceptance_20260810.md).

The 500 artifacts remain immutably attributed to producer HEAD `d493699`.
Because the pre-existing runner discarded trained parameters, checkpoint
handoff contract `a587b8a9...32315b0` now admits that exact producer manifest
to a distinct clean consumer commit and atomically publishes both complete
arm checkpoints only after a passing 500-update fit. Local identity,
round-trip, tree-schema, tamper, incomplete-fit and interrupted-second-arm
tests pass. The serialization-only free-`gln01` preflight also passes at
consumer commit `c15cd86`: both GPU-origin arm trees reload exactly, the
temporary checkpoint is removed, and no Teacher/v5 data or training update is
used. Approved job `14517771` started but failed closed after `00:02:52`,
before any training update or checkpoint publication. The output directory is
empty. The cause is an identity-ownership bug: the coefficient loader required
the accepted two-update source schedule to equal the complete 500-update
consuming schedule, although both use the exact same coefficient-defining
first artifact. The local repair preserves all coefficient values, data,
objective and gates and adds a CPU-only exact-production asset probe.

The conditional 7/30-day acceptance package is complete and frozen before
the fit result. Contract SHA256 is `99ec6eb0...0b6fc07`. It requires exact
dual-checkpoint admission, deterministic temporal/spatial/joint non-test
windows, byte-identical candidate/control panels, cumulative productivity and
terminal biomass/carbon metrics, signed-bias gates and zero exact hard counts.
Host artifacts are prepared on free CPU test nodes before GPU evaluation.
Focused local tests pass `12/12`; no numeric screen has run.

The immediate order is: commit and transfer the repair to a clean worktree;
run the zero-update production-asset admission on `test01`-`test04`; record its
passed report hash; then present a fresh approval block for any replacement
paid fit. Do not resubmit before that free gate passes. After a successful
future fit publishes both checkpoints, prepare the three-window panel on a
free CPU node and run the free `gln01` smoke. The formal 96-window screen,
365-day, long-held-point, Gate D2 and product gates remain closed.

## Completed Task Packet: Gate C

This packet is complete and retained as its acceptance contract. Gates A-C
must not be rerun unless shared Teacher code or a frozen contract changes.
Gate D1, the candidate registry, and Gate E1 are complete. Gate E2 coherent
local implementation, representative six-point production pilot, formal v6
product freeze, global-plan verification, five-worker reuse benchmark, full
production, aggregation, and formal admission are complete. The successor
pair is selected, frozen, and locally implemented. Its synthetic foundation is
accepted, and the revised canonical fast-state owner schema plus non-neural
materializer now pass formal cold/ordinary/restart `160/160` acceptance. The
79-value-head plus mask decoder extension, its full train/train statistics,
and the no-checkpoint real-v6 GPU probe are accepted. The bounded-small-fit
single-update preflight, prepared-batch boundary, same-process hot reuse, and
avoidable-compilation reduction are accepted. Cross-process train-step cache
reuse and the old paid envelope are rejected. The new staged CPU preparation
is accepted. Its checkpoint-capable consumer passes local and free GPU
serialization tests, while the conditional paid fit remains unsubmitted.
Gates D2 and F remain queued.

### Inputs

- canonical Teacher core: `main` at `5910ba9`, byte-identical to research core
  commit `10030eb`;
- true-daily design:
  [`research/daily_coarse_graining/conservative_daily_process_operator_v1.md`](research/daily_coarse_graining/conservative_daily_process_operator_v1.md);
- frozen PFT interface: `manifests/coarse_graining/daily_neural_pft_interface_v1.json`;
- frozen label inventory: `manifests/coarse_graining/daily_flux_label_inventory_v1.json`;
- existing v5 shard contract and dataset manifests;
- source-backed carbon ownership audit and Teacher state/target leaf metadata.

### Implementation steps

1. Implement `water_transfer_daily_v1` as source-resolved daily sums inside
   the complete-day SECHIBA diagnostic fold.
2. Implement `ok_leak_transfer_daily_v1` as resolved daily transfer sums
   inside a diagnostic variant of the compiled OK_LEAK fold.
3. Implement `energy_flux_daily_v1` as source-flux time integrals inside the
   complete-day ENERBIL/THERMOSOIL/explicit-snow diagnostic fold.
4. Prove each capture on a bounded real Teacher day without changing ordinary
   production outputs or storing a 48-step trajectory.
5. Implement a non-neural constrained updater driven by those true labels,
   followed by retained exact daily processes.
6. Prove next-day state reconstruction, budgets, nonnegative inventories,
   exact discrete behavior, cold/later/restart/mask coverage, and restart
   roundtrip without a 48-step reconstruction or state scan.

### Required artifacts

- three daily-reduced diagnostic capture families;
- one non-neural constrained updater and retained-tail composition;
- replay, budget, mask, lifecycle, and restart evidence.

### Definition of done

- every frozen inventory label is supplied by v5, exact derivation, or one of
  the three declared daily captures;
- true Teacher labels reconstruct the accepted next-day canonical state at
  declared field-aware tolerances;
- discrete and defined-status behavior is exact;
- water and carbon budgets close and thermal residual semantics are explicit;
- physical inventories stay nonnegative by construction rather than clipping;
- cold, later-day, restart-year, and active/inactive-mask cases pass;
- targeted regression tests, `git diff --check`, Ruff, and Python compilation
  pass;
- no neural training, 48-step candidate inference, or 669-point regeneration
  is started by this task.
