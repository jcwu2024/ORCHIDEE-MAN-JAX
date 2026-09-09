# Current Roadmap

Last updated: 2026-09-09.

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

The successor pair has now completed all 500 updates under training identity
`9023de2`. Both arms passed the frozen promotion gates with zero hard counts.
Commit `d834d5d` repaired the resumed-process JIT reuse verdict and finalized
the completed recovery without another training update. The accepted fit
report SHA256 is `ea4b66cd...a259a`; the atomically published dual-checkpoint
manifest SHA256 is `f8eed186...75e40`. A separate `test01` reader rehashed and
loaded both parameter and optimizer trees. See
[`successor_small_fit_500_checkpoint_acceptance_20260811.md`](research/daily_coarse_graining/successor_small_fit_500_checkpoint_acceptance_20260811.md).

Do not rerun the 500-update fit. The frozen three-window `gln01` free smoke
passed at evaluation commit `753db81`: both arms completed all temporal,
spatial, and joint windows, every hard count was zero, and the report SHA256
is `966313db...458e22`. The smoke is not promotion eligible and its
`classification` is null. Its single temporal window strongly favors the
control while the spatial and joint windows are closer; this is preserved as
a diagnostic rather than used to change the frozen model or gates. See
[`successor_7_30_free_smoke_acceptance_20260811.md`](research/daily_coarse_graining/successor_7_30_free_smoke_acceptance_20260811.md).

The deterministic 96-window formal panel is now prepared and independently
reloaded on free `test01`. Its 24 hash-bound artifacts total `212,275,046`
bytes; prepared-manifest SHA256 is `6f4c4a30...23d0074`, window-plan file
SHA256 is `71a1a1ac...f397965`, and all temporal/spatial/joint counts and
policies match the frozen contract. See
[`successor_7_30_formal_prepared_acceptance_20260811.md`](research/daily_coarse_graining/successor_7_30_formal_prepared_acceptance_20260811.md).

Approved `gnall` job `14520704` then reached its 45-minute limit before the
first formal artifact completed and published no report. Read-only node
diagnostics found the Python main thread in shared-filesystem
`cl_sync_io_wait`, zero GPU utilization with the model resident in memory,
and many tiny JAX persistent-cache files appearing slowly under `/WORK`.
This is an infrastructure/cache-I/O failure and supplies no model
classification. The local launcher now keeps only its required process-local
JIT cache; focused validation passes 13 tests. See
[`successor_7_30_formal_cache_timeout_20260811.md`](research/daily_coarse_graining/successor_7_30_formal_cache_timeout_20260811.md).

The no-persistent-cache repair then completed the entire frozen formal screen
on free `gln01` in `312.730` seconds, so no replacement paid submission is
needed. Report SHA256 is `7e8dc04f...a5b9d52b`. The attention candidate is
rejected: all six primary-composite gates, 46/72 key-RMSE gates, and 42/72
signed-bias gates fail, while all 30 numerical/state/budget hard gates pass.
Attention repeatedly damages maintenance respiration, NPP, AGB/BGB and total
live biomass relative to the control despite improving GPP in several cells.
See
[`successor_7_30_formal_screen_result_20260812.md`](research/daily_coarse_graining/successor_7_30_formal_screen_result_20260812.md).

The bounded no-training attribution is complete. The 500-update run used only
500 of 23,540 train/train point-years, its final 100 updates were still
improving, and every update in both arms was globally clipped. The old rollout
bias reduction also made MSE and bias squared identical; commit `70c2f62`
repairs that reduction across rollout windows.

The 16-batch coefficient and gradient-path probes then found a more immediate
shared blocker. Updates 12 and 15 have active vegetation, nonzero Teacher GPP,
and order-one named-output gradients, but parameter gradients near `1e-20` or
below. Their initialized GPP and maintenance outputs collapse toward zero. An
all-16 extension also finds large positive overprediction on many other
windows. Attention and control show the same pattern because they share the
output decoder. See
[`successor_training_sufficiency_loss_gradient_audit_20260812.md`](research/daily_coarse_graining/successor_training_sufficiency_loss_gradient_audit_20260812.md).

Do not freeze the old median loss coefficients, promote the control, tune
against the formal 96-window panel, or start paid preparation/training. The
user approved the first matched shared-output repair: both arms retain their
bodies and operator boundary, while the common `decoder_second.kernel` is
initialized at `1e-3` of its former scale and its zero bias is unchanged. The
implementation and 59 related local tests pass. The zero-training 16-batch
free-`gln01` re-audit also passes at commit `0ef543b`: both arms keep exactly
`8,722,689` parameters; active-Teacher GPP near-zero collapses fall from 9/53
to 0/53 and >10x cases from 22-23/53 to 0/53; maintenance collapses fall from
12/60 to 0/60 and >10x cases from 41/60 to 3/60. The minimum measured
parameter-gradient norm is now `0.6077`, versus approximately `1e-26` before.
This admitted loss recalibration but was not evidence of trained accuracy.
That recalibration is now complete under staged-training contract v3. The
existing 32 raw arm/batch measurements produce one shared lower-10%-quantile
coefficient set; all state/budget hard counts are zero. A separate 48-window
panel uses eight distinct landpoints and two nonoverlapping windows in each of
the temporal, spatial, and joint non-test slices, with zero exact overlap with
the rejected formal 96-window screen. Both assets passed independent reload
on free `test01`; see
[`successor_staged_training_assets_acceptance_20260812.md`](research/daily_coarse_graining/successor_staged_training_assets_acceptance_20260812.md).

The staged runner, recovery path, local gates, fixed milestone validation, and
free-node execution smoke now pass. At consumer commit `9b4a1ce`, both arms
applied two finite-gradient updates and passed one temporal, spatial, and joint
validation artifact with zero hard failures. The smoke completed on free
`gln01` in `26:35.37`, used at most about `21.9` GiB observed GPU memory and
`11,648,856` KiB host RSS, and exposed a `5:57.59` XLA train-step compile.
Report SHA256 is
`67b4ba0748e4f2fdc918de68587fa5ea4834271152940aba011c60fd06839f4c`;
see
[`successor_staged_training_gpu_smoke_acceptance_20260812.md`](research/daily_coarse_graining/successor_staged_training_gpu_smoke_acceptance_20260812.md).

The first formal staged milestone is complete. Job `14524758` prepared exactly
500 admitted gzip artifacts; manifest SHA256 is
`19d0cfd699c2535a101edbe245fffc23f49a6c514f565bb069b60e5683355375`.
Job `14525122` completed 500 updates per arm plus the full frozen validation
panel with zero hard failures; report SHA256 is
`5d2ee5b19143f0d5662e6cad0818b5d82de91f859b84140ad33942da52a5019e`.
Independent `test01` processes admitted all 500 training artifacts and later
reloaded both complete parameter trees and every report record. See
[`successor_staged_500_milestone_acceptance_20260812.md`](research/daily_coarse_graining/successor_staged_500_milestone_acceptance_20260812.md).

The second staged milestone is complete and accepted. Incremental preparation
jobs `14527233` and `14532991` produced and independently admitted exactly 500
new artifacts for updates `501-1000`; manifest SHA256 is
`fe39325a3c18d6a7240e0fe5c2c23c14324b5abd9a88377a1ba380112196f076`.
Job `14535746` resumed the accepted update-500 state and completed update 1000
plus the unchanged frozen validation panel with zero hard failures; report
SHA256 is
`e379e3e700ebc93b558ddefa054af36854470b002a8224bf261211aebbd68ce1`.
See
[`successor_staged_1000_milestone_acceptance_20260813.md`](research/daily_coarse_graining/successor_staged_1000_milestone_acceptance_20260813.md).

Control improves from `1.0457081708` to `1.0060782200` (`3.79%`), and
attention improves from `1.0523784009` to `1.0145714634` (`3.59%`). Each arm
passes its own minimum-1% milestone gate, and all six slice/horizon cells
improve. This establishes that increasing the distinct training data used is
currently beneficial. It does not promote either arm: control remains `0.84%`
better overall, while GPP/NPP, especially temporal productivity, remain the
dominant scientific errors.

The approved `1001-2000` stage completed under GPU job `14540920`. Control
improved only `0.31%`, from `1.0060782200` to `1.00298315`, below the frozen
1% gate. Attention worsened `3.08%`, from `1.0145714634` to `1.04584784`.
Control improved all six validation cells slightly; attention worsened all
six. The accepted best checkpoint therefore remains update 1000. Do not
continue unchanged to update 5000.

The free-node, no-checkpoint batch-diversity short fit frozen in
[`successor_batch_diversity_probe_20260814.md`](research/daily_coarse_graining/successor_batch_diversity_probe_20260814.md).
is complete. Its 100-update A/B has zero hard failures. Attention mixed
sampling improves its matched single continuation by 4.35% and improves the
update-1000 baseline by 0.73%, but control mixed sampling is 0.25% worse than
control single. The preregistered both-arm gate therefore fails. Report
SHA256 is `6c3f16ff...161154b`; do not adopt mixed sampling or propose paid
continuation from this result.

The report-only attribution is complete in
[`successor_loss_conflict_attribution_20260814.md`](research/daily_coarse_graining/successor_loss_conflict_attribution_20260814.md).
Across all six validation cells, single sampling improves maintenance
respiration while systematically damaging NPP and live biomass; mixed
sampling reverses or suppresses that tradeoff, especially for attention. The
next controlled experiment category is loss balancing, not sampling or
architecture.

The attention-specific zero-update per-output gradient audit is complete at
commit `4f35e6a`; report SHA256 is `a476b332...d1410b49`. Three of four
maintenance-versus-primary-output pairs meet the stable single-batch conflict
rule, but mixed sampling mitigates zero of them and makes every median less
aligned. The proposed mixed-conflict-mitigation mechanism is rejected. The
same audit finds NPP/maintenance median gradient norms around `1e-1/1e-2`
while AGB, BGB, and total-live-biomass norms are only about `1e-6` to `1e-7`.
See
[`successor_per_output_gradient_audit_20260814.md`](research/daily_coarse_graining/successor_per_output_gradient_audit_20260814.md).

The named AGB/BGB tendency objective foundation and its train-only scale asset
are now accepted. The asset uses 64 unique train/train point-years and 256
rollout windows from existing prepared v6 data, with zero optimizer updates,
exact-zero Teacher/Teacher loss, and a passed scale-aware float64 AGB+BGB
identity gate. Its file SHA256 is `fdc1f416...d4f175`; the acceptance contract
SHA256 is `a0ec1315...2278490`. See
[`successor_named_biomass_tendency_scale_acceptance_20260814.md`](research/daily_coarse_graining/successor_named_biomass_tendency_scale_acceptance_20260814.md).

The zero-update candidate-objective gradient preflight passed on free `gln01`.
The accepted tendency scales raise unit AGB/BGB median gradient norms to
`8.115/1.549` versus NPP `0.1929`. The frozen shared coefficient is
`0.0544047`, giving weighted AGB/NPP and BGB/NPP ratios of `2.289` and `0.437`.
All four complete candidate gradients are finite and positive, and every
state/budget hard count is zero. Report SHA256 is `2023f3cf...b0d7413`; see
[`successor_named_biomass_gradient_preflight_20260814.md`](research/daily_coarse_graining/successor_named_biomass_gradient_preflight_20260814.md).

The bounded named-biomass objective A/B is complete and rejected. The
candidate worsens the four-key NPP/AGB/BGB/total-live composite by `10.97%`,
worsens each key by `3.73-46.98%`, and improves `0/6` slice/horizon cells.
Both arms have zero hard failures and maintenance improves `31.05%`, so this
is a valid negative objective result rather than a failed run. Candidate
gradient clipping occurs on `93/100` updates versus `47/100` for control. The
report SHA256 is `fb6f6f69...f03b89d`; see
[`successor_named_biomass_objective_ab_20260814.md`](research/daily_coarse_graining/successor_named_biomass_objective_ab_20260814.md).

Post-run attribution finds candidate median raw gradient norm `6.587` versus
control `0.927`, candidate clipping on all 50 updates in the second half, and
the named tendency terms supplying a median `73.9%` of its weighted
long-horizon contribution. Candidate one-day anchor is also worse on `73/100`
paired batches. This points to persistent fixed-weight component dominance,
not a biomass signal that remains too small.

The follow-up zero-update gradient-feasibility gate found a raw common descent
direction for one-day anchor, NPP, AGB tendency, and BGB tendency in all four
sampled groups. A frozen 10-update minimum-norm smoke then completed on free
`gln01` at commit `e22a235` with zero hard failures, `10/10` feasible raw
directions, and zero candidate clipping. It is nevertheless rejected: the
four-key validation composite worsens `8.39%`, every NPP/AGB/BGB/total-live
aggregate worsens, and `0/6` slice/horizon cells improve. Maintenance improves
`30.80%`. Report SHA256 is `d84a6a76...bb83834`; see
[`successor_gradient_balanced_smoke_20260814.md`](research/daily_coarse_graining/successor_gradient_balanced_smoke_20260814.md).

Do not run the conditional 100-update gradient-balanced A/B. The active gate
is to freeze a zero-update realized-step audit before implementation. It must
compare the raw common-descent direction with the actual proposed AdamW
parameter displacement produced by inherited optimizer moments, and may
compare a preregistered matched optimizer-state reset. It must retain no
proposed parameters and perform no training continuation. If the realized
step remains locally improving, stop optimizer work and return to
training-batch versus independent-rollout generalization. No architecture
change is authorized by this result; discuss any new network architecture
before implementation.

The zero-update realized-step audit is now complete at commit `6cec975`.
The raw minimum-norm direction passes all four groups, but inherited AdamW
reduces minimum component cosines to `0.00039-0.00464` and passes `0/4`
direction groups. Reset AdamW also passes `0/4`, with two groups having a
negative component cosine. Virtual all-component loss checks pass only `3/4`
inherited groups and `2/4` reset groups. Hard and nonfinite counts remain
zero. Report SHA256 is `4b987902...47037`; see
[`successor_realized_step_audit_20260815.md`](research/daily_coarse_graining/successor_realized_step_audit_20260815.md).

Do not run an optimizer-reset smoke. The direct-direction trust-region audit
is complete at commit `55c30a3`. Global step norm `0.001` decreases all four
component losses in all four groups with zero hard or nonfinite failures.
Step norm `0.003` is already outside the shared trust region because one
group's largest component regression is `19.43%`. Report SHA256 is
`ccc4e1fd...e1fbd`; see
[`successor_direct_trust_region_audit_20260815.md`](research/daily_coarse_graining/successor_direct_trust_region_audit_20260815.md).

The free-`gln01` 10-update direct-direction smoke is complete at commit
`621cf57` and rejected. All updates are finite with zero hard failures, and
all four component medians decrease, but only `6/10` updates decrease every
component; the largest batch-component regression is `19.11%`. On the
accepted non-test 7/30-day subset, the NPP/biomass composite worsens `1.4348%`,
every primary key worsens, and `0/6` cells improve. Maintenance improves
`5.7566%`. Report SHA256 is `48e8e7da...50abf7`; see
[`successor_direct_direction_smoke_20260815.md`](research/daily_coarse_graining/successor_direct_direction_smoke_20260815.md).

The zero-update accumulated multi-batch audit is complete at commit `ab7f41a`.
A tenfold larger effective update batch reduces the worst regression at norm
`0.0003` from `19.11%` to `0.0323%`, while median NPP and biomass components
improve. It still passes only `4/10` groups because equal averaging hides
small point-year conflicts; norm `0.001` becomes nonlinear and regresses up to
`28.89%`. Report SHA256 is `0601fcd6...c0d847`; see
[`successor_multibatch_direction_audit_20260815.md`](research/daily_coarse_graining/successor_multibatch_direction_audit_20260815.md).

The zero-update group-aware audit is complete at commit `b0ebda9`. A strong
common direction exists across all 40 group-component constraints: minimum
cosine is `0.050356`, all four tested steps improve every component in all
`10/10` groups, and hard/nonfinite counts are zero. At norm `0.0003`, NPP and
AGB tendency meet the frozen median-effect threshold, while the one-day anchor
and BGB tendency improve by only `1.6217e-5` and `5.1778e-5`. The formal
finite-step classification therefore fails on magnitude only, not on any
regression. Report SHA256 is `f9c55f15...a4e3`; see
[`successor_group_aware_direction_audit_20260815.md`](research/daily_coarse_graining/successor_group_aware_direction_audit_20260815.md).

The finite-step extension is complete at commit `f0b2235` and passes. Norm
`0.002` improves every component in all `10/10` groups; median relative
changes are `-1.2176e-4` anchor, `-1.0193e-3` NPP, `-1.3361e-3` AGB tendency,
and `-3.4462e-4` BGB tendency. The least negative individual change remains
`-1.8028e-5`. Hard/nonfinite counts are zero and no parameters are retained.
Report SHA256 is `a8b44848...df9a`; see
[`successor_group_aware_step_extension_20260815.md`](research/daily_coarse_graining/successor_group_aware_step_extension_20260815.md).

The five-update frozen-scalarization smoke completed at commit `1bbb163` and
stopped before applying update 2. Update 1 reproduces the accepted direction
to `4.82e-15` relative norm error and improves all 40 constraints. At update
2, fixed norm `0.002` regresses several constraints, worst group-5 NPP by
`0.0861%`. After the one accepted step, all six independent 7/30-day cells and
all four NPP/AGB/BGB/total-live aggregates improve; the combined improvement
is only `0.00922%`, below the frozen `0.5%` gate. Report SHA256 is
`1aeb93ae...60ca6`; see
[`successor_group_aware_scalarization_smoke_20260815.md`](research/daily_coarse_graining/successor_group_aware_scalarization_smoke_20260815.md).

The backtracking smoke completed at commit `4118c72` and rejects parent-frozen
weights. Update 1 again selects `0.002`. At update 2, norms `0.002, 0.001,
0.0005` have worst regressions `8.6109e-4, 4.3088e-4, 2.1552e-4`; the nearly
linear scaling proves the direction itself has ceased to be common. No second
update is applied. Report SHA256 is `c2691824...74261`; see
[`successor_group_aware_backtracking_smoke_20260815.md`](research/daily_coarse_graining/successor_group_aware_backtracking_smoke_20260815.md).

Do not test smaller fixed-weight steps. The zero-retention second-state exact
recalibration audit completed at commit `1902936`. A new common direction
exists with minimum cosine `0.0486966`; norm `0.002` improves all 40 losses
and passes the four component-effect gates. The new simplex support has 13
constraints, 12 overlapping the parent's 14. It reads no validation values
and retains no parameters. Report SHA256 is `c4ddb39b...10e97`; see
[`successor_second_state_recalibration_audit_20260815.md`](research/daily_coarse_graining/successor_second_state_recalibration_audit_20260815.md).

The bounded active-set recalibration audit completed at commit `217876d` and
passes. Re-solving the 14 parent-support gradients makes all 40 relative
directional derivatives positive in one JVP round; the minimum is `0.0107166`
and no constraint must be added. Norm `0.002` passes all ten groups with
component medians closely matching the exact reference. The mechanism uses
14 component reverse passes, `35%` of the exact reference's 40. Report SHA256
is `e2a9e0eb...daf97`; see
[`successor_active_set_recalibration_audit_20260815.md`](research/daily_coarse_graining/successor_active_set_recalibration_audit_20260815.md).

The bounded repeated active-set smoke completed at commit `b730731` and stops
after one new accepted update. The following recalibrated direction still
passes all 40 first-order checks with minimum relative derivative `0.0105908`.
Norms `0.001` through `0.002` enter a strongly nonlinear regression region;
norm `0.0005` decreases all 40 losses but misses the frozen per-step effect
gate only for anchor and BGB medians. The retained two-update endpoint improves
all `6/6` non-test 7/30-day cells and every primary aggregate, but its combined
improvement is only `0.01822%`, below the unchanged `0.5%` promotion threshold.
Report SHA256 is `04c35474...613d7`; see
[`successor_repeated_active_set_smoke_20260815.md`](research/daily_coarse_graining/successor_repeated_active_set_smoke_20260815.md).

Do not test smaller one-shot norms and do not lower the scientific effect
gate. The active design task is a bounded cumulative-microstep mechanism:
recalibrate after each safe norm-`0.0005` microstep, require every microstep to
be nonincreasing across all 40 objectives, and evaluate the existing `1e-4`
component-effect gate on the cumulative macro-update. The mechanism is frozen
at no more than four microsteps under contract SHA256
`be46303e...2c545a`, and local admission passes 74 related tests plus static,
contract, and launcher checks. The free-`gln01` run completed at commit
`b3d2f5f` and rejects the mechanism. Microstep 1 reproduces the safe
norm-`0.0005` step; after it is applied, a freshly recalibrated direction still
has minimum relative derivative `0.0105646`, but the next norm-`0.0005` step
regresses an objective by up to `28.7222%`, led by BGB tendency across several
groups. It is stopped before application. No validation values or parameters
are retained. Report SHA256 is `7316749e...b674aa`; see
[`successor_cumulative_microstep_smoke_20260815.md`](research/daily_coarse_graining/successor_cumulative_microstep_smoke_20260815.md).

Do not subdivide or repeat the same raw Euclidean direction. The active task
is a zero-retention parameter-geometry attribution at the stopped state:
partition the existing parameter tree by owned encoder, interaction, and
decoder blocks; measure direction-norm and relative-update concentration; and
test only a bounded set of source-defined block projections. A projection is
admissible only if it preserves all-40 first-order descent before its
finite-step gate. The seven-block leave-one-out audit is frozen under contract
SHA256 `fa046011...c87153`; local admission passes 77 related tests and all
static, contract, and launcher checks. The free-`gln01` execution completed at
commit `c51dd87` and did not attribute a safe top-level exclusion. The output
decoder carries `96.1526%` of squared direction norm; excluding it breaks
common descent with minimum relative derivative `-0.285257`, while excluding
any other block leaves the `28.72%` finite-step regression essentially
unchanged. Report SHA256 is `a7a170c7...c1a003`; see
[`successor_parameter_geometry_attribution_20260815.md`](research/daily_coarse_graining/successor_parameter_geometry_attribution_20260815.md).

Do not freeze the full decoder or alter Attention. The active design task is
a four-leaf decoder attribution over `decoder_first/second` kernels and
biases, using the same stopped state, all-40 first-order prefilter, and single
norm-`0.0005` finite-step gate. It is frozen under contract SHA256
`08783a37...3cf721`; local admission passes 79 related tests and all static,
contract, and launcher checks. The free-`gln01` audit completed at commit
`39cd8ef` and finds no safe leaf exclusion. The second decoder kernel carries
`68.4533%` of full squared direction norm and its bias has the largest relative
step (`0.645335%` of parameter norm), but excluding either breaks common
descent. Excluding either first decoder leaf preserves descent yet leaves the
`28.72%` finite regression. Report SHA256 is `513e4337...9b6c4d`.
See
[`successor_decoder_leaf_attribution_20260816.md`](research/daily_coarse_graining/successor_decoder_leaf_attribution_20260816.md).

Raw Euclidean parameter-space direct steps are closed. Do not test more
parameter exclusions or scalar subdivisions. The bounded output-locality
audit is now frozen under contract SHA256 `edaebad5...7e9852` and passes 97
directly related local tests plus static, contract, and launcher checks. It
uses the same stopped state and ten groups to compare the 30-day first-order
JVP and exact norm-`0.0005` changes for all 181 decoded fields in accepted
normalized training coordinates. One three-seed output VJP per group
separates decoder-output nonlinearity from downstream conservative-trajectory
nonlinearity; no per-field intervention runs are used. See
[`successor_output_locality_audit_20260816.md`](research/daily_coarse_graining/successor_output_locality_audit_20260816.md).

The free-`gln01` execution completed at commit `2a29c2c`; report SHA256 is
`93ebbff8...1cfa85`. All evidence gates pass. Output nonlinearity explains
`96.08%` of the NPP and `79.79%` of the BGB finite mismatch.
`daily.fast_maintenance_respiration` alone carries `94.29%` of absolute NPP
and `95.11%` of BGB output-nonlinearity contribution. Its exact normalized
RMS change is `0.08206` versus a first-order RMS of `0.000414`, with maximum
absolute change `4.71565`. GPP has the opposite sign and partially offsets
the regression.

The normalized maintenance-local output trust region is now frozen under
contract SHA256 `84b96d4f...e35b72`. The raw failed proposal is the matched
control; the preceding accepted norm-`0.0005` microstep supplies exact RMS and
maximum maintenance caps at `1.25x`. Three corrections at norms `0.000125`,
`0.00025`, and `0.0005` share one gradient of 30-day normalized maintenance
displacement. GPP remains uncapped. Local admission passes 99 related tests
plus Ruff, compilation, Bash, contract, and diff checks.

The first live four-point-year reverse pass was rejected before any candidate
or report when XLA requested `39,772,984,576` bytes on a 32 GB V100. The exact
same scalar mean is now accumulated over 40 one-point-year reverse
microbatches. A one-point-year zero-displacement memory preflight runs before
the expensive stopped-state reconstruction; this changes execution shape, not
the objective, samples, candidates, or gates.

The free-`gln01` execution completed at commit `629238a`; formal report SHA256
is `64c68938...ce98c5`. All evidence checks pass with zero hard failures and
zero nonfinite values. The safe-reference maintenance RMS/maximum are
`0.0004166623`/`0.00402458`, defining caps `0.0005208279`/`0.00503073`. The
raw control reproduces RMS `0.08206115`, maximum `4.71565188`, and a worst
loss regression of `28.7222%`.

No correction passes. Norm `0.000125` leaves maintenance RMS `0.0390105` and
regresses all three long-horizon medians. Norm `0.00025` is the informative
near-feasible result: it improves NPP/AGB/BGB medians by
`0.1628%`/`0.2292%`/`0.0327%`, but maintenance RMS/maximum remain
`0.00363294`/`0.0771433`, the anchor regresses `0.0177%`, and the worst loss
regresses `0.2733%`. Norm `0.0005` overshoots maintenance RMS to `0.00680043`
and still violates the anchor and loss gates. See
[`successor_maintenance_output_projection_protocol_20260816.md`](research/daily_coarse_graining/successor_maintenance_output_projection_protocol_20260816.md).

The single-gradient projection is closed. Do not subdivide the same
correction. The active design task is a bounded nonlinear constrained solver
in a frozen low-dimensional parameter subspace, initialized from the
informative `0.00025` neighborhood. Each bounded iteration must relinearize
the currently violated all-40 losses, anchor, and maintenance RMS/maximum
constraints, then verify the correction by exact finite 30-day evaluation.
Freeze the subspace, iteration and evaluation budgets, feasibility-restoration
rule, acceptance gates, and failure action before implementation or execution.
This remains an optimizer-geometry investigation, not an architecture change
or training authorization.

That design is now frozen as the low-dimensional nonlinear projection
contract, SHA256 `bb0f0718...d67ba07`. Its matched control is the informative
norm-`0.00025` single-gradient result. It constructs at most eight ordered,
scientifically identified parameter directions at that point, freezes the
orthonormal subspace, and permits at most four sequential relinearizations and
eight exact 30-day candidate evaluations. Every iteration relinearizes all 40
losses, the three long-horizon median effects, maintenance RMS, and all 28,800
signed maintenance scalars. Total displacement is bounded to
`[0.000125, 0.00075]`. Local implementation and admission now pass 95 related
tests plus all static, contract-hash, and launcher checks. The only authorized
next operation is one free-`gln01` zero-retention execution of this frozen
audit. See
[`successor_low_dimensional_nonlinear_projection_protocol_20260816.md`](research/daily_coarse_graining/successor_low_dimensional_nonlinear_projection_protocol_20260816.md).

The free execution completed at commit `85f1633` in `4905.58 s`. Its evidence
is valid, all hard/nonfinite counts are zero, the eight-dimensional basis is
orthonormal, and the matched control is reproduced. The matched point still
has 12 violated constraints. The bounded hard/slack solver found no admissible
nonzero step, so no exact candidate trial was opened. Report SHA256 is
`4af8ddaf...5447ce`. Fixed low-dimensional post-training projection is now
closed.

The Attention-only mixed-point annealed continuation is complete. Approved
job `14569673` used all prepared 1001-2000 assets as 250 updates of four
distinct point-years and annealed the learning rate to `3e-5`. All hard gates
were zero. Median training loss fell 4.7% from the first to last 50-update
window and maintenance aggregate validation RMSE improved 22.3%, but the
primary NPP/AGB/BGB/total-live composite became 6.98% worse. All four primary
key aggregates and all six temporal/spatial/joint 7/30-day cells regressed.
The checkpoint is retained for diagnosis and is not promotion eligible. See
[`successor_attention_mixed_annealed_continuation_result_20260816.md`](research/daily_coarse_graining/successor_attention_mixed_annealed_continuation_result_20260816.md).

The user rejected further local ranking around the weak update-1000 model as
the main research strategy. That checkpoint is diagnostic evidence, not an
incumbent that defines success. The route-local-decoder A/B is superseded
before implementation and must not be built or trained as the next candidate.

The user approved the independently capable strong baseline
`axis_aligned_process_expert_operator_v1`. The design preserves dynamic
PFT/depth/pool/part coordinates from source encoding through seven
process-owned expert paths and axis-local family heads. It targets 35-50
million parameters, uses all 8,591,565 train/train days in complete shuffled
one-day epochs, proves small-set memorization before scale, and then applies a
7/30-day rollout curriculum. It has no matched control during baseline
qualification and is judged only by absolute one-day, productivity, biomass,
bias, state, and budget gates. Controls and ablations are deferred until a
baseline passes those gates. Neural algebra/optimizer state is provisionally
float32, while constrained output transforms, the conservative update,
retained tail, scientific loss, and budgets remain float64. Structural,
one-point-year, 64-point-year, 1%/10% full-epoch, one-day, 7-day, 30-day, and
365-day gates stop spending at the first failed capability; all seven expert
losses remain visible independently. See
[`strong_baseline_axis_aligned_process_expert_review_v1.md`](research/daily_coarse_graining/strong_baseline_axis_aligned_process_expert_review_v1.md).

The architecture contract is now frozen at SHA256
`baf5791f...a96351d`. The 49,750,681-parameter float32 implementation,
formal process head, shared factorized axial block, axis/source masks,
expert-balanced loss, eager/JIT forward, representative JVP/VJP,
inactive-PFT isolation, undefined-sentinel isolation, and synthetic bound one-day budget gate pass 15 local
tests. The shared output-transform and fast-parameter-route refactors also
pass the relevant prior-successor regressions. See
[`strong_baseline_local_acceptance_20260816.md`](research/daily_coarse_graining/strong_baseline_local_acceptance_20260816.md).

The zero-update real-v6 admission passed. The first complete one-point-year
qualification then ran all 364 valid train/train days for 100 epochs and 4,600
finite unclipped AdamW updates. GPP, maintenance, NPP, AGB tendency, and BGB
tendency met the declared `0.05` NRMSE bound and all state/budget hard counts
were zero. The complete process loss nevertheless fell only `97.60%` versus
the required `99%`; canopy GPP, hydrology/snow, soil thermal, and
litter/POC/DOC also missed the per-expert `95%` reduction gate. The run is not
qualified and no checkpoint is promotion eligible. In addition, the stock-
scale AGB/BGB metric was already below threshold at random initialization and
is not discriminating one-day evidence. See
[`strong_baseline_one_point_qualification_result_20260816.md`](research/daily_coarse_graining/strong_baseline_one_point_qualification_result_20260816.md).

The bounded one-point loss-floor diagnostic is now frozen and locally
implemented. It reproduces the 100-epoch qualification, then performs at most
25 exact all-364-day gradient updates through 46 size-8 microbatches per
update. It reports every expert, label, value field, and defined-status loss;
loads the accepted train-only one-day AGB/BGB tendency scales; and writes a
parameter-only state that is explicitly ineligible for formal initialization
or promotion. A scientific gate failure exits normally instead of appearing
as a crashed job. The exact-accumulation, architecture, and tendency-asset
regressions pass 94 tests, plus Ruff, formatting, compilation, CLI, Bash, and
diff checks. See
[`strong_baseline_one_point_loss_floor_diagnostic_v1.md`](research/daily_coarse_graining/strong_baseline_one_point_loss_floor_diagnostic_v1.md).

The real-v6 preflight and frozen full diagnostic completed on free `gln01` at
commit `04b0c2e`. All execution, coverage, finite-gradient, state, budget,
count-identity, corrected-scale, and diagnostic-state gates pass. The total
loss fell from `4.22628` to `0.19781` after the minibatch stage and only to
`0.19126` after 25 exact-gradient updates: `95.4744%` reduction, still below
`99%`. Four experts still fail. Corrected GPP, maintenance, and AGB pass, while
BGB (`0.05064`) and NPP (`0.05694`) narrowly fail. Report SHA256 is
`2f338354...7ad5661`; see
[`strong_baseline_one_point_loss_floor_diagnostic_result_20260817.md`](research/daily_coarse_graining/strong_baseline_one_point_loss_floor_diagnostic_result_20260817.md).

Five labels account for `91.06%` of final loss, and `fast_owner.diffuco` plus
`fast_owner.driver` alone account for `76.18%`. The active gate is therefore a
source/representation audit of those dominant endpoint labels. Determine
which high-loss fields can be reconstructed once per day by existing
source-backed JAX kernels, which are necessary learned primitives requiring a
coupled decoder, and which are poorly conditioned targets. Do not implement a
revised network before discussing that architecture with the user. Repeating
the same stochastic recipe, paid training, 64-point-year fit, full epoch,
7/30/365-day screen, long-held points, Gate D2, sealed test, and product gates
remain closed.

The bounded source/representation audit is now complete and corrected. Its
first version incorrectly compared signed endpoint deltas with absolute
physical ranges. The real decoder first adds day start and then constrains the
absolute endpoint; the loss converts it back to a delta. The corrected
inverse-decoder audit finds `0/260,988` hard range violations. It identifies
24,532 exact 0/1 boundary endpoints and 34 interior targets requiring absolute
raw above 20, which are bounded conditioning/status findings rather than
evidence for a global signed decoder.

One-boundary calls against true v6 endpoints verify exact reconstruction of
`z0m` (`1.33e-15` maximum error), `roughheight*` (zero), `drysoil_frac`
(`3.33e-16`), and the final `gtemp`, `stempdiag`, `tdeep`, and `hsdeep`
diagnostics (zero). `z0h` does not yet reproduce and remains unresolved.
Coupled energy endpoints and canopy storage remain learned. See
[`dominant_endpoint_source_representation_audit_20260817.md`](research/daily_coarse_graining/dominant_endpoint_source_representation_audit_20260817.md).

The exact reconstruction was then expanded to 20 non-test point-years: eight
spatial-train points and two spatial-validation points, each in 1961 and 2006.
`z0m`, `roughheight*`, `drysoil_frac`, `gtemp`, `stempdiag`, `tdeep`, and
`hsdeep` pass every cell with maximum error `2.22e-15` or lower. `z0h` fails
all 20 cells with maximum error `0.0137343` and remains a learned endpoint.
The complete true-v6 boundary also reproduces all ten thermal coefficient
fields in every point-year within `1e-10 + 1e-12*abs(target)`. Report SHA256
is `adbdb831...6e579e8`. They are verified exact derived fields rather than
conditional candidates. No test shard was read.

The DOC diffusion correction is separately closed. Its v6 target is the sum
of 48 nonnegative source magnitudes, not a signed interface sum and not one
exact daily kernel result. The endpoint-trapezoid boundary proxy passes its
frozen gate in all four non-test slices and leaves only `4.50%` residual
variance overall. It is accepted as a physical baseline plus learned residual.
Report SHA256 is `e200f2d7...8b69cbd`; see
[`doc_diffusion_boundary_baseline_result_20260817.md`](research/daily_coarse_graining/doc_diffusion_boundary_baseline_result_20260817.md).

The user approved the complete mechanistic-output architecture. Its formal
contract is
`manifests/coarse_graining/strong_baseline_mechanistic_output_v1.json`. The
implementation learns 165 fields and 39,153 scalar values, materializes the
complete 181-field/39,371-value boundary around one float64 conservative
update, then enters the canonical retained exact tail. It retains
axis-preserving process Attention and independent expert capacity, routes the
coupled energy endpoints through one expert, applies capacity-aware canopy
storage, and removes independent supervision of 16 verified derived fields.

Local tests pass, and the bounded real-v6 `gln01` admission passed eager/JIT
forward, complete mechanistic loss, all-parameter gradient, canonical bound
transition, next-state, and budget checks. The subsequent fresh one-epoch,
364-day smoke completed 46 finite updates with every prediction/state/budget/
inventory hard count zero; its report SHA256 is
`725130c4...bc73d304`.

The first 100-epoch one-point run then passes the `99%` total-loss gate, all
five primary-science NRMSE gates, and every hard gate. It fails only the
independent canopy-GPP, hydrology/snow, litter/POC/DOC, and surface-energy
expert-reduction gates. A second detailed run confirms that `humrel/humrelv`,
canopy debt incoming amount, coupled energy endpoints, and DOC/litter fields
dominate the residual, while also showing sensitivity to the unclipped
high-learning-rate trajectory. See
[`mechanistic_output_one_point_qualification_result_20260817.md`](research/daily_coarse_graining/mechanistic_output_one_point_qualification_result_20260817.md).

The hydrology/canopy source-representation audit is complete on 20 non-test
point-years. Final `nroot` cannot be reconstructed from day-start `nroot` and
final moisture in one call: only `5/20` point-years pass. Once final `nroot` is
supplied, however, `us`, `humrelv`, and `humrel` reproduce in `20/20` with
maximum error `2.22e-16`; grid `evap_bare_lim` is also exact from learned
tile-level `evap_bare_lim_ns`. `temp_hydro` fails `20/20` and remains learned.

Canopy positive storage respects its physical capacity in every one of 7,290
daily endpoints, but debt is positive on `41.51%`, so it is not a rare
outlier. The captured daily `vevapwet` is signed and includes condensation; it
is not a valid hard debt scale. See
[`hydrology_canopy_boundary_audit_result_20260817.md`](research/daily_coarse_graining/hydrology_canopy_boundary_audit_result_20260817.md).

The user approved the hydrology/canopy v2 output mechanism, now frozen in
`manifests/coarse_graining/strong_baseline_mechanistic_output_v2.json`. It
learns 161 fields/39,006 scalar coordinates, materializes 20 exact fields, and
contains 49,743,001 parameters. Final `nroot` is learned on its day-start
support; final `us`, `humrelv`, `humrel`, and `evap_bare_lim` are reconstructed
after the conservative update. One signed canopy endpoint and three status
logits replace independent supervision of the four storage/debt transfers,
which are still materialized uniquely for the conservative updater.

The local gate passes 60 focused and related regression tests plus Ruff,
compileall, and diff checks. The subsequent zero-update real-v6 `gln01`
admission also passes complete eager/JIT forward, loss, the full
49,743,001-element parameter-tree gradient, canonical retained tail, one-day
state, and exact-zero water/carbon budgets. Report SHA256 is
`cd376e48...ccd0d8`.

The one-epoch training smoke passes. It completes all 46 finite updates,
reduces complete process loss `7.3818%`, improves all seven experts, preserves
all masks/counts, and has zero prediction/state/budget/inventory hard counts.
GPP, maintenance, and NPP NRMSE improve from `0.3963`, `0.6725`, and `1.0865`
to `0.1390`, `0.2301`, and `0.1253`. Report SHA256 is
`61de8c4c...a48b5c0`.

The formal 100-epoch/4,600-update one-point run completed. Total loss falls
`98.9085%`, missing the `99%` gate by `0.0915` percentage points. All five
primary NRMSE gates and all hard gates pass. Maintenance, soil thermal, and
daily interface/finalize pass their expert gates; hydrology/snow (`93.7780%`),
canopy/GPP (`93.4923%`), surface energy (`89.3930%`), and litter/POC/DOC
(`80.2522%`) do not. Hydrology improves by more than ten percentage points
over v1, while litter is effectively plateaued. No checkpoint is retained.

The zero-training output-coordinate attainability audit is accepted on all
364 existing v6 rows. Maximum learned-field normalized MSE is `2.0612e-9`,
maximum derived exact-field normalized MSE is `9.8733e-22`, the DOC fixed-
point residual is zero, and every hard count is zero. A separate primitive-
source alignment proves all thermal reconstruction inputs within physical
tolerance; the largest source error is `2.6733e-13` in `mc/mcl`. The older
fixed physical tolerance on Teacher-derived thermal coefficients measured
roundoff amplification rather than a mapping failure and is retained only as
a sensitivity diagnostic. Reports SHA256 are `292e8a03...a0897` for the full
audit and `ffe388fb...55abce` for source alignment.

The two fresh isolated one-expert diagnostics are complete. Surface energy
falls `97.9259%`, misses only the `99%` aggregate gate, and passes all 11
dominant-field gates. Litter/POC/DOC falls `91.8014%`; 12 of 28 dominant fields
fail, concentrated in coupled OK_LEAK auxiliary endpoints, litter/DOC incoming
amounts, adsorption, and advective transfer. Both runs complete all 4,600
updates with nonzero selected block/head gradients and updates, unchanged
counts, zero hard failures, and no checkpoint. Joint equal-expert interference
is therefore not the whole failure mechanism.

The user approved `hierarchical_coordinate_query_attention_v3`, and its local
admission is complete. The 58,736,698-parameter candidate removes the rejected
litter global expert blocks and shared litter label heads, uses four
field-balanced coordinate-query stages, and preserves the accepted
mechanistic-output v2 boundary, conservative update, explicit parameter
ownership, and retained exact tail. Local forward/JIT, official loss and
gradient, complete bound-day state/budget, inactive-PFT, and strict PFT
permutation gates pass. The candidate also uses well-conditioned local
parameterizations for the zero-centered nonnegative and bounded litter fields
whose old transforms starved gradients. See
[`hierarchical_coordinate_query_attention_v3_design_20260819.md`](research/daily_coarse_graining/hierarchical_coordinate_query_attention_v3_design_20260819.md)
and
[`hierarchical_coordinate_query_attention_v3_local_acceptance_20260819.md`](research/daily_coarse_graining/hierarchical_coordinate_query_attention_v3_local_acceptance_20260819.md).

The memory-repaired smoke and the approved 100-epoch isolated diagnostic have
now completed. Job `14619803` reduces litter/POC/DOC loss from `0.1895335` to
`0.00269069` (`98.5804%`) and passes all execution, coverage, state, budget,
inventory, and finite-value gates. It improves the old mapping substantially
and passes `25/28` dominant fields, but it misses the frozen `99%` aggregate
gate. DOC incoming amount, free-to-adsorbed transfer, and advective transfer
remain below their field gates, and no checkpoint is retained.

The follow-up target audit motivated and the user approved
`coordinate_coupled_doc_transition_attention_v4`. Its local implementation is
complete at `59,971,257` parameters. Every DOC inventory coordinate now reads
five upstream transfer representations aligned by physical PFT identity,
layer, pool, and element. A joint signed transition head materializes the
mutually exclusive incoming/outgoing endpoints before the unchanged
conservative update. Focused v4, runner, unchanged v3, and unchanged
mechanistic suites pass `48` tests; static checks pass.

Paid job `14627588` completed the v4 100-epoch/4,600-update diagnostic without
an execution failure. Aggregate loss falls `99.1353%`, but the candidate
passes only `24/28` dominant fields. Coordinate-aligned tokens repair DOC
incoming substantially, from v3's `-0.3085%` reduction to `90.5487%`, while
the joint signed decoder turns the previously passing DOC outgoing fraction
into a `-962.3373%` reduction. Free-to-adsorbed and advective transfer also
remain below their absolute gates. No checkpoint was written.

The bounded zero-training DOC endpoint geometry audit is complete. All active
joint targets have finite inverses and the incoming/outgoing median normalized
Jacobian ratio is `0.8857`, so v4 did not fail because of an inverse bug or a
broad derivative-scale mismatch. It failed the conditioning gate on 110 small
active targets, while incoming supplies `97.6209%` of the shared fields'
combined zero-predictor squared error. The one scalar is therefore trained
primarily to repair incoming and can sacrifice outgoing. Returning to v3's
separate transforms is insufficient: they have 563 small-Jacobian active
targets. Explicit status plus branch-native magnitude has complete finite
coverage and is the accepted basis for v5.

The revised
`coordinate_coupled_doc_status_branch_magnitude_attention_v5` mechanism is
frozen, implemented, and locally accepted at `59,973,309` parameters. It keeps
the complete v4 coordinate-aligned Attention backbone and replaces only the
shared signed endpoint scalar with three status logits plus independent
incoming/outgoing magnitude coordinates. Its branch-balanced official loss
removes the old pooled DOC endpoint contribution rather than duplicating it.
Focused eager/JIT forward, loss, full-gradient, canonical binding, one-day
state/budget, inactive-PFT, and permutation gates pass. The formal training
entry and launcher select V5 explicitly and leave v3/v4 behavior unchanged.

Paid job `14632388` completed the V5 10-epoch/460-update early screen normally
in `01:14:07`. Epoch 5 passed, all gradients and values were finite, coverage
passed, counts were unchanged, and every state, budget, and inventory hard
count was zero. The epoch-10 absolute gate failed: litter/POC/DOC reduction was
`83.1178%` rather than `90%`, and DOC incoming/outgoing reductions were
`29.6589%`/`26.1806%` rather than `50%`. Free-to-adsorbed and advective
transfer improved `16.6932%` and `58.9182%`. No checkpoint was written.

V5 removes v4's catastrophic endpoint trade and its status/incoming-latent/
outgoing-latent losses fall `75.4655%`/`78.4438%`/`54.2036%`, but the
materialized physical endpoint term falls only `29.2932%`. The frozen result
therefore rejects this soft-status-times-magnitude materialization and loss
balance without rejecting the retained coordinate-aligned Attention source
bank.

The subsequent target/output audit is complete. The 161 learned fields are now
partitioned into 2 direct-primary fields (GPP and maintenance respiration), 93
state/budget-essential fields, and 66 low-weight auxiliary process fields; 20
additional boundary fields remain exact/diagnostic. The new objective uses a
one-day state anchor, independent-basis 7-day and 30-day science losses, and a
0.1-weight necessary-process term. It reports and absolutely gates GPP,
maintenance/growth/total autotrophic respiration, NPP, heterotrophic
respiration, NEP/NEE, AGB/BGB/total live biomass, nonliving ecosystem carbon,
and ecosystem carbon without duplicating algebraic identities in the loss.
The frozen machine contract SHA256 is `7c9857c0...86bfce5c`.

The target-oriented successor architecture is frozen and locally implemented
as `target_oriented_hierarchical_coordinate_attention_v1`. It has
`83,755,173` parameters, separate primary/state/auxiliary output capacity,
five cross-domain science tokens, and a fresh constrained output path that
does not inherit the rejected v4/v5 endpoint heads. Its focused local gate
passes `10/10`, including full-boundary eager/JIT, complete loss/gradient,
PFT isolation/permutation, canonical binding, and one-day budgets.

The approved zero-update real-v6 GPU admission has passed on one free `gln01`
V100 at commit `7c6371c`. All 83,755,173 gradient elements and all seven
required architecture families are finite; the complete forward and next-day
state are finite on every declared-defined value; water and carbon budget
residuals are zero. Hot JIT replay is `0.06197 s`. JAX peak live device memory
is approximately 2.49 GiB; its approximately 25 GiB allocator reservation is
not actual tensor use. No update or checkpoint occurred. Report SHA256 is
`a92e9415...ad13803b`.

The first bounded real-v6 training design is frozen as
`target_oriented_bounded_training_v1`, SHA256 `91faf011...ea25b`. Every
effective update uses 28 distinct point-years/landpoints from one rotating
runtime year: 32 one-day samples,
eight seven-day windows, and four thirty-day windows, totaling 208 Teacher-day
usages. Microbatches `4/1/1` use rematerialized rollout and exact sufficient
statistics, so memory reduction does not change full-batch MSE or signed-bias
loss. Focused loss and gradient equivalence tests pass. AdamW uses 250-update
warmup and 250/1,000/3,000 milestones; a train-only eight-batch p90 calibration
replaces the arbitrary fixed gradient-norm threshold.

The eight real-v6 batches and first free-V100 full-step preflight are complete.
The run used about 8.15 GiB peak live memory and about 35.65 seconds per hot
effective step, but all eight gradients were NaN. The failure is localized to
the one-day state anchor rather than the Attention body: a near-constant DOC
coordinate with scale `7.96e-12` produced loss around `1e14`, was duplicated
through the complete `other` state view, and exposed a masked-NaN backward
path. The only complete-output nonfinite is exact/diagnostic
`roughheight_pft`; every learned output is finite. All observed state-status
mismatches belong to declared dynamic `rveget`; water/carbon budgets,
discrete state, and negative-stock counts remain zero.

The frozen state-anchor amendment uses mutually exclusive residual `other`
coordinates, native-unit fallback for the 46/3,854 empirical scales at or
below `1e-6`, safe masked arithmetic, learned-only nonfinite counts, and
report-only declared-dynamic `rveget` mismatches. The amended representative
probe completed under job `14644566`: the approximately `1e14` DOC domination
is gone and all learned outputs, states, budgets, discrete state, and stocks
remain valid, but the one-day/process, seven-day, and thirty-day gradients all
contain NaNs. The subsequent zero-variance masked-summary repair is a valid AD
repair, but job `14650340` proved that it does not affect this parameter-
gradient path: the loss, hard counts, and all three NaN gradients were
unchanged because the repaired summary is on the non-differentiated input
side.

A real-v6 learned-output-boundary VJP then localized the nonfinite derivative
to exactly one of 39,006 scalar coordinates:
`inventory.water.qsintveg.storage.incoming_amount` for bare soil. The old
signed-endpoint reconstruction evaluated `removed / source` at
`source == removed == 0` before a forward `where` hid the value. The safe-
denominator repair at commit `c8cb497` leaves forward physics unchanged.
Free-`gln01` verification shows finite boundary gradients for every coordinate
and every state group; commit `e8294b0` additionally traces the real boundary
cotangent through the complete Attention model, with all 83,755,173 parameter
gradient values finite and finite nonzero gradients in every connected model
family. Auxiliary heads correctly remain zero because this diagnostic uses
only the state anchor.

The complete zero-update real-v6 representative batch contains 32 one-day
samples, eight 7-day windows, and four 30-day windows. It is the smallest
existing probe that simultaneously verifies all three objective-gradient
sources and the optimizer proposal. The first repaired execution,
paid job `14651381`, timed out after `01:30:21` before completing the first
batch and therefore wrote no JSON report or scientific gradient result. The
log shows finite-path XLA compilation, including a 20-minute backward compile,
rather than an exception or hard-gate failure. The shared persistent cache now
contains the completed one-day/forward and subsequent backward executables.
The exact same probe then completed under job `14652590` at commit `50c2c14`
in `00:53:12`. Report SHA256 `aa195412...33ac861` has status `passed`: all
three source gradients and all 83,755,173 combined gradient values are finite;
all seven connected parameter families have finite nonzero combined
gradients; the optimizer proposal is finite with zero nonfinite parameter or
state values; and all budget, discrete, stock, unexpected-mask, prediction,
and state hard-failure counts are zero. No update or checkpoint was retained.

The representative batch also exposes an initialization-conditioning fact.
The one-day/process gradient norm is `5.2020e6`, while the 7-day and 30-day
science norms are `1.4514` and `0.8519`. The long-horizon sources dominate the
primary science-head gradients but reach the shared encoder, refinement,
cross-attention, mixer, and state heads at much smaller scale than the one-day
anchor. This does not imply that AdamW will ignore the primary heads, but it
means raw global norm alone cannot establish useful long-horizon learning.
The frozen eight-batch zero-update preflight completed under job `14659152` at
commit `bca102c` in `00:27:00`. Report SHA256
`7bddd58b...a688de5` has status `passed`: every combined gradient and all seven
connected parameter families are finite and nonzero in all eight batches; the
frozen p90 clipping threshold is `5323655.8449`; the AdamW proposal is finite;
peak live JAX memory is 9.60 GB; and every physical hard-failure count is zero.
The seven cache-hot effective steps have median `36.8113` seconds. This
corrects the earlier 2.5-hour projection, which wrongly treated cold
single-batch compilation as recurring execution.

Across all eight runtime years, 7/30-day gradients strongly update the direct
primary science heads but remain roughly seven orders of magnitude below the
one-day state anchor in shared representation families at initialization.
This does not reject the architecture or objective: the long-horizon paths are
finite and connected, and AdamW does not map raw global norms directly to
step ratios. It does require the first retained stage to report whether
long-horizon loss and shared-family influence improve during warmup.

The formal target-oriented runner and free-GPU admission are complete at
commit `e363482`. The accepted prepared manifest covers updates `1..250`, the
fixed validation manifest contains 96 non-test windows, and the one-update
admission completed with finite parameters/AdamW state, no clipping, zero
contract-defined hard failures, recovery write/reload, and all six 7/30-day
cells before and after the update. Admission uses one window per cell and is
not a scientific promotion screen.

Paid `gnall` job `14664525` completed all 250 updates and both complete
96-window validation panels at commit `e363482`. Every contract-defined hard
failure is zero and only `3/250` updates were clipped. The science macro fell
from `0.964059` to `0.689483`, a `28.4812%` relative improvement that passes
the preregistered `5%` pacing rule but not the absolute 7/30-day gate. AGB,
BGB, and total live biomass pass all six cells; GPP/NPP improve materially but
remain above threshold; 30-day ecosystem and nonliving carbon barely improve,
and maintenance respiration is inconsistent.

The explicit update-251-to-1000 continuation and local qualification are now
implemented. The runner restores the exact update-250 parameters and AdamW
state, consumes partial prepared ranges at absolute update indices, preserves
the frozen schedule/objective/clip threshold/96-window panel, and distinguishes
the stage checkpoint from recovery and initialization. Focused continuation,
checkpoint, and range-manifest tests pass `8/8`; the extended target-oriented
group passes `37/38`, with the sole failure an unchanged historical launcher
wall-time assertion unrelated to this path.

Free-node continuation admission now passes. Update 251 loads the exact parent
parameters and AdamW step 250, uses the correct absolute learning rate, writes
and reloads recovery, has zero physical hard failures, and improves the matched
six-window admission macro by `9.1095%`. Because that panel has one window per
cell, the improvement is engineering evidence rather than promotion.

The exact 251-1000 partial range is accepted. Four free-CPU chunks merged
into 750 unique contiguous artifacts totaling 2,523,051,773 bytes. Every file
hash and payload passed, the full-1000 schedule identity is frozen, and the
merged manifest SHA256 is `8205140d...8fc31`. Teacher regeneration, historical
v5 use, and sealed-test read are all false.

Paid job `14674142` completed the exact update-251-to-1000 continuation at
commit `433d286`. The frozen science macro improves from `0.68948347` to
`0.34085437`, a further `50.5638%`; all physical hard gates pass and clipping
is `0/750` in this stage. The absolute gate still fails. The important change
is that 7/30-day ecosystem and nonliving carbon now pass every slice, while
GPP, NPP, maintenance respiration, growth respiration, and autotrophic
respiration remain the blocking targets.

The zero-update train-only flux-gradient audit completed on free `gln01` at
commit `c5a72a4`, covering 32 predeclared 30-day windows and every one of the
`83,755,173` parameters. Its decision is
`mixed_batch_and_objective_geometry`. GPP supports a larger effective batch
and NPP is already coherent at four windows, but maintenance and total
autotrophic respiration remain opposed between disjoint 16-window estimates.
Batch enlargement alone is therefore rejected.

The component-resolved flux-basis amendment and its free-`gln01` zero-update
geometry audit are complete while retaining the current Attention
architecture and exact update-1000 state. All five declared sources are
finite and nonzero over all `83,755,173` parameters in all eight groups, and
physical hard failures are zero. The successful report SHA256 is
`414aa543...ff2a05`.

The audit rejects a raw averaged continuation. The 7-day flux-gradient norm
ranges from `0.515` to `1713.736`, its median cross-batch cosine is `-0.4739`,
and `53.6%` of batch pairs are opposed. The 30-day flux source has median
cosine `+0.0199` with `50%` negative pairs. Update 465 dominates the mean, so
its apparent `+0.9723` alignment with the one-day gradient is not a stable
training admission. The already successful stock objectives remain protection
constraints.

The protected virtual-step audit completed on free `gln01` at commit
`04c2b23`; report SHA256 is `6a9ed9c9...f1fba`. Its normalized common
direction is real and well conditioned: the independently verified minimum
constraint cosine is `0.04353145`, versus the frozen `0.001` gate. However,
none of the seven finite norms passes. At the smallest norm `0.0001`, exact
7-day component-flux loss has median change `+0.6359%` and maximum change
`+15.7217%`; one-day state/process loss has median `+11.5593%`; and the worst
stock score increases `+33.1712%`. All physical hard and nonfinite counts are
zero, but the flux, state, and stock protection gates fail. No parameter,
checkpoint, or recovery was retained.

This closes direct realization of the raw component-resolved
MSE-plus-squared-bias loss at update 1000. It does not reject the Attention
architecture. The train/train, zero-training residual-tail and sampling-unit
audit then completed on free `gln01` at commit `b80494f`; report SHA256 is
`e1c094e2...27d9e`. All 64 7-day and 32 30-day window gradients are finite
over all `83,755,173` parameters and physical hard failures are zero.

The decisive result is Jacobian amplification rather than a simple high-loss
tail. The largest 7-day gradient norm is `22489.30`, contributes `92.07%` of
the sum of all window gradient norms, and is 54 times the second largest. Its
loss is only `0.05285`, approximately rank 12 of 64 and about `2.12%` of total
loss. The corresponding maintenance-respiration score is `0.27647`; the
largest 30-day gradient window has nearly the same maintenance score
`0.27649`, while loss-matched controls have far smaller gradients.

Frozen soft-L1/log1p influence functions do not materially repair the 7-day
geometry: minimum cosine remains approximately `-0.97` and at least `35.9%`
of windows remain opposed. The best 30-day log1p candidate reduces the
negative fraction from `28.1%` to `18.8%`, but its minimum cosine remains
`-0.874`. The machine action retains bounded influence as a possible later
component; it does not establish robust-loss-only training.

The focused gradient-path attribution then completed on free `gln01` at
commit `53269af`; report SHA256 is `a2e9a885...ffa870d`. All `24/24` full
gradients are finite over all `83,755,173` parameters and all physical hard
failures are zero. Both extreme pairs classify as
`rollout_recurrence_dominant`. Their maintenance free-to-Teacher-state
gradient ratios are `243184.83` at 7 days and `49.38` at 30 days, and the
largest state-head contribution in both cases is the predicted
`hydrol_previous_step_state.snowheat` tendency. The matched controls behave
differently: their Teacher-state maintenance gradients are larger than their
free gradients. This is therefore a scenario-dependent snowheat feedback,
not a generic benefit of Teacher forcing or evidence against Attention.

The snowheat-only causal intervention then completed on free `gln01` at commit
`159f672`; report SHA256 is `e2a6934b...a05e0f`. All `8/8` full gradients are
finite over all `83,755,173` parameters, free scores reproduce exactly, and
all physical hard failures are zero. The 7-day extreme maintenance gradient
falls by `10416.39`, causally confirming snowheat recurrence there. At 30
days, the snowheat-head gradient falls by `78.96`, but the full gradient falls
only by `1.56`; snowtemp becomes the dominant remaining state head with norm
`272.77` of a total `276.37`. Controls remain near their original magnitudes.

The coupled snowheat-plus-snowtemp intervention completed on free `gln01` at
commit `975ae9b`; report SHA256 is `c4b80543...303810`. Both complete gradients
are finite and all physical hard failures are zero. For the 30-day extreme,
the snowtemp-head norm falls `494.88` times and the full gradient falls `5.23`
times, leaving the snowheat head at `40.71` and the direct maintenance head at
`33.43`. The frozen tenfold full-gradient threshold is not met, so the result
is mixed and Attention remains open.

The mirror-complete snow-thermal intervention completed on free `gln01` at
commit `51bb538`; report SHA256 is `5714d30d...4422168`. Both complete
gradients are finite and all physical hard failures are zero. The snowheat
head becomes exactly zero in both windows, proving that the previous residual
used the omitted `diffuco` mirror. The extreme full gradient falls only
`1.5693` times because the direct maintenance head remains `33.4668`; the
matched control falls `2.0219` times and retains a `15.5237` maintenance head.
The frozen classification is mixed and Attention remains open.

The persisted-maintenance intervention then completed on free `gln01` at
commit `d08d6eb`; report SHA256 is `a95ae129...f809b9`. Both gradients cover
all `83,755,173` parameters, all five state slices resolve, and all physical
hard failures are zero. Cutting prior-day `resp_maint` changes the extreme
full gradient from `33.69151431` to `33.69151394` and its maintenance head
from `33.46682558` to `33.46682500`. The matched control is likewise
unchanged. The frozen classification is
`maintenance_state_recurrence_not_explanatory`.

Do not remove `resp_maint` from the maintenance input route on this evidence,
and do not continue arbitrary state-by-state Teacher carry. The remaining
gradient is the direct 30-day maintenance-output objective. The frozen
`target_oriented_maintenance_head_conditioning_audit_v1` has now completed on
free `gln01`; report SHA256 is `85ab83a1...d27757`.

All 16 finite, nonzero 769-parameter head gradients admit a common direction
with minimum cosine `0.056523`. No evaluated finite step passes every frozen
protection gate, so the formal classification is
`finite_head_step_not_protected`. This is not a no-learning result. At head
norm `0.0018462`, median 7/30-day maintenance scores improve by
`1.4415%/0.8188%`, with NPP, AGB, BGB, and total-live-biomass improvements.
The formal failure comes from small positive median changes in 7-day GPP,
growth-respiration, and ecosystem-carbon scores. The larger `0.0055387` norm
also violates several `1%` any-group protection limits.

The source-shaped maintenance architecture discussion and local
implementation are now complete under contract SHA256
`3c4d0c085c767f14b001837ccefb76aca291a56b8fd9d3a5e33279602af57fc4`.
The candidate preserves the update-1000 coordinate-aligned Attention body and
all non-maintenance heads, removes only the 769-parameter direct maintenance
projection, and adds two `768 -> GELU(256) -> 1 -> tanh` source-group branches.
Its exact parameter count is `84,148,646`; both final layers start at zero.
Maintenance is materialized from day-start biomass/LAI, retained-static
`coeff_maint_zero`, named slope parameters, native daily air temperature, and
root temperature from predicted daily `tsoil`. The tree/non-tree reserve
coordinate joins the air/root group respectively. No 48-record reconstruction
or scan is present.

The focused local suite passes all six tests: contract/tree ownership, eager
and JIT forward, finite complete loss, finite nonzero gradients in both new
branches and reused Attention, nonzero-correction PFT permutation equivariance,
bare/inactive exact zero, canonical binding, and water/carbon budgets. Ten
related parent-Attention and parameter-routing numerical tests also pass. The
only failure in that related file is an unrelated pre-existing launcher
assertion expecting `01:00:00` while its Slurm script declares `03:00:00`.

The zero-update real-v6 GPU admission passed on free `gln01` at commit
`cf1fead`; report SHA256 is `b5684fb4...b5119b46`. All 84,148,646 gradient
elements are finite, both physical correction branches and every retained
Attention family have nonzero gradients, all 160 non-maintenance outputs are
exactly equal to the parent, and the canonical one-day water/carbon residuals
are zero. JAX peak live allocation is about 2.55 GiB.

The frozen train/train screen and its zero-update panel are now complete at
commit `250986f`; report SHA256 is `6e7302fe...e094291`. Before training, the
source formula already reduces median maintenance score by `73.6076%` at 7
days and `87.3457%` at 30 days. It therefore clears the required 20% margin by
a large amount. It is not yet protected: 7-day median growth respiration,
NPP, AGB, BGB, and total-live scores regress, and the worst group regression
is `340.382%`. All hard and nonfinite counts are zero.

The free one-update branch-only admission is complete at commit `0ed926b`;
report SHA256 is `c2743cbc...34ea59`. Eight disjoint calibration batches give
a p90 branch-gradient clip of `0.1973867`. Artifact 252 then produces a finite
raw branch norm of `0.2293467`; fresh AdamW update 1 uses clip scale `0.86065`,
changes both new branches, leaves all 83,754,404 retained coordinates exactly
unchanged, and reduces its effective-batch loss by `0.03366%`. Hard failures
are zero before and after. The proposal was discarded and no checkpoint or
recovery was written.

The direct branch-subtree autodiff admission is complete at commit `d965bea`;
report SHA256 is `5852d996...e58538`. Its complete forward still uses all
84,148,646 parameters, while reverse mode differentiates only the 394,242 new
branch parameters. The direct gradient has zero failing coordinates against
the branch extracted from the complete gradient; all three gradient sources,
the objective, hard counts, fresh AdamW proposal, and post-proposal objective
also match. Hot effective-batch time falls from `36.85 s` to `25.18 s`
(`1.46x` speedup), and JAX peak live GPU memory falls from about 7.64 GiB to
2.24 GiB (`70.67%` reduction). Physical gates pass. No training update,
checkpoint, recovery, validation read, or sealed-test read occurred.

This closed the engineering prerequisite. The paid runner, branch-only
checkpoint/recovery, exact `252..351` schedule, final eight-group screen, and
single-V100 Slurm wrapper were locally qualified at commit `9224398`. The user
approved the exact envelope and Slurm job `14701638` completed all 100 updates
in `00:58:51` at commit `7fdce65`; report SHA256 is
`a5143c06...b2d4b22`.

The engineering run passes: only 394,242 branch parameters changed, all
83,754,404 retained coordinates remained exactly fixed, clipping occurred on
`4/100` updates, and all final-screen physical hard failures are zero. The
maintenance reduction rises from `73.61%/87.35%` to `90.76%/96.50%` at 7/30
days. Scientific promotion nevertheless fails. Median 7-day growth
respiration, NPP, AGB, BGB, and total-live-biomass scores regress relative to
the fixed parent, and no 7-day group plus only one 30-day group passes all
absolute keys. The update-100 checkpoint is diagnostic only, and unchanged
branch-only continuation is closed.

The zero-training source-coupling attribution completed on free `gln01` at
commit `4c22a4c`; report SHA256 is `e2a0b90f...26a0d6b`. All hard failures are
zero and the raw carbon-bias identities close to `5.68e-14`. In free rollout,
update 100 improves maintenance by `62.93%/71.74%` at 7/30 days but worsens
growth respiration, NPP, AGB, BGB, and total live biomass. Under matched
Teacher day starts, every one of those protected regressions disappears and
becomes a `7%`-`9%` improvement; GPP is exactly invariant. Maintenance itself
instead worsens by `680%/792%`, making autotrophic respiration worse.

The branch correction therefore depends strongly on the recursively predicted
state distribution. Immediate joint flux expansion is not supported, and
unchanged branch-only continuation remains closed. The zero-training
source-owner state bridge completed on free `gln01` at commit `1c862a4`;
report SHA256 is `7defdf47...642d5c`. All hard failures are zero and the raw
carbon identities close to `3.02e-14`.

Teacher-carrying only `biomass`, `lai`, `veget_max`, and `t2m_longterm`
reverses `4/10` frozen cells: all three 7-day biomass measures and 30-day AGB.
Growth respiration and NPP remain materially worse at both horizons, while
30-day BGB and total live biomass improve relative to free rollout but do not
reverse. The accepted classification is
`source_owner_state_drift_contributes_but_not_sufficient`.

The successor mechanism is now frozen and locally accepted in
[`target_oriented_distribution_aware_maintenance_v1.md`](research/daily_coarse_graining/target_oriented_distribution_aware_maintenance_v1.md)
at contract SHA256 `f167943c...1a37d2d`. It keeps the update-1000 Attention
parent, preserves all 24 PFT/part maintenance coordinates, adds a two-block
maintenance-specific Attention adapter over the full canonical context, and
materializes maintenance through the existing exact source formula. The
complete candidate has `93,504,933` parameters; only the `9,750,529` adapter
parameters are first-stage trainable. The focused local candidate and
predecessor suites pass all 11 unique tests, including eager/JIT forward,
complete loss, adapter gradients, exact non-maintenance parent equality, PFT
permutation, canonical binding, and water/carbon budgets.

The zero-update real-v6 admission passed on free `gln01` at commit `54ada54`;
report SHA256 is `376d8016...c66ece5`. It covers all `9,750,529` adapter
gradient elements, every adapter leaf is nonzero, all 160 non-maintenance
outputs exactly equal the parent, and canonical water/carbon residuals are
zero. JAX peak live allocation is about `1.37 GB`; the roughly 25 GiB visible
to `nvidia-smi` was the allocator reservation.

The zero-update calibration completed on free `gln01` at commit `b980cde`;
report SHA256 is `bb55388f...addf7a474`. All eight free and Teacher-guard
gradients cover all `9,750,529` adapter coordinates, every adapter leaf is
nonzero, and all physical hard failures are zero. The median gradient norms
are `0.3181163` and `13.9532423`; the raw ratio `0.0227987` therefore freezes
the preregistered bounded Teacher-guard coefficient at exactly `0.1`.

The separate 250-update adapter-only training contract is frozen at SHA256
`0c60924e...210051`. It uses 250
prepared train/train artifacts selected from `252..511` after excluding the
calibration/evaluation updates, a
fresh AdamW state, 25-update warmup, `1e-4` peak learning rate, `3e-5` final
rate, and clip threshold `2.692620468992339`. The threshold is the p90 of the
existing calibration triangle upper bounds, so no calibration rerun is
needed. The parent Attention and all non-adapter parameters remain frozen.

The ephemeral artifact-252 admission passed on free `gln01` at commit
`5ea0033`; report SHA256 is `a32b2c17...c7734e`. Free, guard, and combined
gradients cover all adapter coordinates and all leaves are nonzero. The
combined norm `2.0161371` is below the frozen clip threshold, all adapter
leaves change, all `83,754,404` parent coordinates remain exactly fixed, the
same-batch combined objective decreases, and pre/post hard failures are zero.
No update or checkpoint was retained.

The exact 250-update runner, compact adapter-only recovery/checkpoint,
update-0/update-250 train/train screen, and one-V100 Slurm wrapper were locally
qualified at commit `ad5c11b`. Paid job `14856147` then completed all 250
updates at commit `90c6636` in `03:38:12`; report SHA256 is
`7f6d5686...9257`. All retained parent coordinates remain exactly unchanged,
no update is clipped, and every hard failure count is zero. Median maintenance
reduction reaches `90.7403%/95.7084%` at 7/30 days, and maintenance passes
both aggregate absolute gates. Complete scientific promotion fails: GPP,
growth respiration, and NPP remain above their absolute gates, while the
frozen protected no-regression gate also fails. The diagnostic checkpoint
SHA256 is `baae0c13...7da`; unchanged adapter-only continuation is closed.

The zero-training Teacher-GPP causal bridge on the same eight train/train
groups is complete at commit `5480f51`. It substituted only matched Teacher
fast GPP before the retained exact daily carbon chain while preserving the
candidate's own maintenance, state recurrence, and all other outputs. See
[`target_oriented_distribution_aware_training_result_20260829.md`](research/daily_coarse_graining/target_oriented_distribution_aware_training_result_20260829.md).

The bridge contract is now frozen at SHA256 `ce2eefb1...28c5647` and the
forward-only implementation is locally qualified. It replaces GPP inside the
typed `ProcessPrediction`, before the canonical retained tail, while carrying
the model's predicted state between days. The focused and related suite passes
29 distinct tests, including a complete local learned-output/typed-flux/state/
budget day; Ruff, Python compilation, Bash syntax, CLI import, and diff checks
pass. The accepted free-`gln01` execution then completed in `934.80 s`; report
SHA256 is `fa9bc445...531c52`. Both executed modes have zero hard failures.
Teacher GPP makes GPP exact, reduces median growth-respiration score by
`99.6757%/99.5925%` and NPP by `98.6264%/98.8312%` at 7/30 days, and makes all
nine required aggregate keys pass. Teacher-day-start instead severely worsens
the learned fluxes. The frozen classification is
`gpp_boundary_supported_as_next_scope`. See
[`target_oriented_distribution_aware_gpp_bridge_v1.md`](research/daily_coarse_graining/target_oriented_distribution_aware_gpp_bridge_v1.md).

The user-approved structured GPP architecture is frozen at contract SHA256
`ce211fa7...620ea` and its local implementation is accepted. It preserves the
coordinate-aligned Attention parent and the distribution-aware maintenance
adapter, removes only the 769-parameter GPP projection, and adds a
12,879,363-parameter day-local 45-token Transformer with potential,
limitation, and bounded-correction branches. The complete candidate has
`106,383,527` parameters. Focused local tests pass exact ownership/count,
eager/JIT forward, complete loss and gradients, non-GPP parent equality,
parameter isolation, inactive/bare/PFT permutation, and a complete canonical
state/budget day. See
[`target_oriented_structured_gpp_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_v1.md)
and
[`target_oriented_structured_gpp_local_acceptance_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_local_acceptance_20260829.md).

The free-`gln01` real-v6 zero-update admission passed at commit `dbc1373`;
report SHA256 is `7a5d4310...04c488`. The GPU backend loaded the frozen
update-250 distribution-aware checkpoint, preserved all parent parameter
objects, matched all 160 non-GPP outputs bit for bit, covered all 12,879,363
specialist gradient elements, and produced nonzero gradients in all 129
specialist leaves. The canonical daily transition has all 181 outputs, finite
declared state, and exact-zero water/carbon residuals. JIT replay is `0.064 s`
and peak live allocation is `1.47 GB`. See
[`target_oriented_structured_gpp_gpu_admission_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_gpu_admission_20260829.md).

The separate structured-GPP training preflight passed on free `gln01` at
commit `d852e69`; report SHA256 is `d531f8dc...2cbccc`. All eight direct-GPP
gradients cover all 12,879,363 specialist parameters and 129 leaves. Their p90
norm is `17.25686156`. One discarded AdamW proposal reduces same-artifact GPP
loss, preserves all parent arrays, and keeps the daily physical hard gate at
zero. The real 7/30-day forward paths pass without compiling horizon gradients.
See
[`target_oriented_structured_gpp_training_preflight_result_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_training_preflight_result_20260829.md).

Paid job `14873550` completed the frozen warm start at commit `1fb8916` with
exit code zero. Its disjoint endpoint panel improved `8/8`, and aggregate
direct-GPP loss fell from `1.11241519` to `0.74803793` (`32.7555%`). All
physical hard failures are zero, all parent parameters remain exactly fixed,
and report SHA256 is `c0e23be4...500f`. The retained specialist checkpoint is
`ebef94aa...3f9` and is not promotion eligible. See
[`target_oriented_structured_gpp_warm_start_result_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_warm_start_result_20260829.md).

The train-only 7/30-day joint-gradient calibration passed on free `gln01` at
commit `7c547d5`; report SHA256 is `6ff2d03e...389c0`. All source and combined
gradients cover the complete specialist, all physical hard gates pass, and all
pairwise source cosines are positive. The frozen 7/30 coefficients are
`5.065197169020285` and `4.594433600967992`; the combined-gradient p90 clip is
`46.802861793665286`. See
[`target_oriented_structured_gpp_joint_gradient_calibration_result_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_joint_gradient_calibration_result_20260829.md).

The discarded joint-update preflight was frozen at
contract SHA256 `5d255e49...0c5d5`. On train/train artifact 512, load the
accepted warm-start specialist and inherited AdamW state at step 250, combine
the three exact gradients with the frozen coefficients, apply the frozen clip
and one ephemeral step 251, then discard it. Require the negative realized
parameter step to have positive cosine with all three source gradients; all
three same-artifact losses must strictly decrease; pre/post one-day, 7-day,
and 30-day physical hard counts must be zero; the parent and checkpoint must
remain unchanged. Write no checkpoint or recovery. A pass opens only a
separately frozen bounded joint-training contract and paid-resource request.
Do not begin paid joint training, validation, 365 days, Gate D2, sealed test,
or product work before this gate is accepted. See
[`target_oriented_structured_gpp_joint_update_preflight_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_joint_update_preflight_v1.md).

That preflight is now complete and rejected at commit `e9aa41e`; report SHA256
is `c63d2d4e...ccfc6`. Direct GPP, 7-day science, weighted science, and the
complete joint objective decrease, but 30-day science increases `0.2558%`.
The raw 7-versus-30 gradient cosine is `-0.9134`, and the inherited AdamW
negative step has cosine `-0.0167` with the 30-day gradient. All physical,
finite-value, parameter-ownership, and checkpoint gates pass, and no proposal
is retained. See
[`target_oriented_structured_gpp_joint_update_preflight_result_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_joint_update_preflight_result_20260829.md).

Do not relax the completed gate or begin paid single-artifact joint training.
The zero-update train-only gradient heterogeneity audit completed at commit
`95322f8`; report SHA256 is `15d993a2...80eea`. Batch sizes 1, 2, 4, and 8
pass `7/8`, `3/4`, `2/2`, and `1/1` groups respectively. Batch size 4 is the
minimum consistent effective batch. Its two groups have minimum
source-to-combined cosines `0.7992` and `0.7543`, including the group that
contains the rejected single artifact 512. All gradients, objectives,
physical gates, parent identity, and checkpoint identity pass; no optimizer
or update was created. See
[`target_oriented_structured_gpp_joint_gradient_heterogeneity_result_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_joint_gradient_heterogeneity_result_20260829.md).

The discarded batch-four AdamW preflight then passed at commit `42c0e4e`;
report SHA256 is `6a574c67...a0ac1`. Both independent groups start from the
same step-250 state, retain positive realized-step cosine with every aggregate
source, and reduce their mean direct-GPP, 7-day, and 30-day losses. The first
group changes those losses by `-0.1694%/-0.1079%/-0.0897%`; the second by
`-0.2086%/-0.1038%/-0.1222%`. All gradient, finite-value, physical, parent,
and checkpoint gates pass. No proposal is retained. See
[`target_oriented_structured_gpp_joint_batch4_preflight_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_joint_batch4_preflight_result_20260830.md).

The bounded batch-four joint-training contract and paid runner are now frozen
and locally qualified. Contract SHA256 is `168d44bd...98832`; its PCG64
four-epoch schedule SHA256 is `770bae61...5f70`. Each effective update averages
four independent direct, 7-day, and 30-day gradient triples before one inherited
AdamW update. The bounded maximum is 248 joint updates, with atomic recovery
every ten updates and at each epoch boundary. The eight-artifact train-only
panel must strictly improve all three mean sources after an epoch; a failed
epoch writes no checkpoint and stops, while the accepted non-test 96-window
panel remains unread until final model selection. Local contract, recovery,
negative-design, checkpoint, static, CLI, and shell checks pass. See
[`target_oriented_structured_gpp_joint_training_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_joint_training_v1.md).

Paid job `14878155` completed the unchanged bounded contract at commit
`8115588`. All 248 batch-four updates and four train-only epoch panels passed;
step 498 is selected. The formal independent 96-window screen then completed
on free `gln01` at commit `d41cd0c`. All physical hard failures are zero, but
the absolute science gate fails every slice/horizon cell. Science macro NRMSE
is `0.4613409`, versus `0.3408544` for the inherited step-1000 Attention
checkpoint on the same panel. GPP fails `6/6` cells with negative bias and is
`49.3998%` worse on mean NRMSE; maintenance improves `59.3330%` and passes
`5/6`. Biomass and ecosystem-carbon stock keys pass `6/6`, but this does not
override the GPP/NPP/respiration failures. See
[`target_oriented_structured_gpp_joint_training_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_joint_training_result_20260830.md)
and
[`target_oriented_structured_gpp_short_horizon_screen_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_short_horizon_screen_result_20260830.md).

The target-blind coverage audit is complete at commit `db9763b`; report SHA256
is `a53330a6...0528b`. It reconstructs exactly 103,399 unique exposed days,
`1.203494%` of the 8,591,565 accepted train/train transitions. Warm start used
51,863 unique days; joint training used 51,878 unique days and repeated them
over four epochs. Coverage spans all 535 train landpoints and 44 years but only
10,761 of 23,540 point-years. A SHA256-ranked monitor proposal reserves 128
previously unseen point-years on distinct landpoints, samples 2,048 monitor
days, and leaves 8,521,436 first-row-eligible days for future broad training.
See
[`target_oriented_structured_gpp_data_coverage_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_data_coverage_result_20260830.md).

The direct-only streaming probe is accepted at commit `c28a667`; report SHA256
is `06731a56...10761`. Preparing 208 days takes `0.7304 s` rather than the
complete-example path's `84.2489 s`, a `115.34x` improvement and about 284.76
host days/s. The shared-card 208-day exact forward/gradient path sustains 56.36
days/s and the combined hot path 47.05 days/s; this is a conservative result
because an unrelated process already used about 27% GPU. All loss statistics
and all 12,879,363 specialist gradient elements are finite. The path reads no
typed sidecar and constructs no retained-tail, other-target, next-state, or
7/30-day payload. See
[`target_oriented_structured_gpp_streaming_throughput_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_streaming_throughput_result_20260830.md).

The broad-data schedule and free-GPU preflight contract is frozen as
[`target_oriented_structured_gpp_broad_preflight_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_broad_preflight_v1.md),
machine SHA256 `f0e50031...096a3`. Keep the coordinate-aligned Attention parent
and structured-GPP mechanism unchanged. The proposed maximum is 5,000 updates
and 1,040,000 previously unseen days; every 208-day update has 52 distinct
landpoints and four days per point-year. The accepted 128-point-year monitor is
reserved in full and the 103,399 prior-exposure days are excluded.

The target-blind schedule and free-GPU preflight are accepted at commit
`853afa3`. Schedule report SHA256 is `7de9e840...9e252`, schedule SHA256 is
`b4295558...d6b8b`, and GPU report SHA256 is `0962b18d...39b0d`. The full
schedule has exactly 1,040,000 unique days, all 535 landpoints, 23,412
non-monitor point-years, all 44 years, and zero overlap with prior exposure or
monitor point-years. Sixteen global batch gradients were finite across all 129
specialist leaves; raw norms have median 12.8578, p95 20.9879, and maximum
23.0190. Four independent `4e-6` AdamW steps all reduced their own batch loss
and passed retained-parent and physical gates. See
[`target_oriented_structured_gpp_broad_preflight_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_broad_preflight_result_20260830.md).

The active contract is
[`target_oriented_structured_gpp_broad_training_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_broad_training_v1.md),
machine SHA256 `c54233c1...340e1`. The resumable runner and two-update free
`gln01` admission passed at commit `1573eb9`; report SHA256 is
`94a43cbe...cc70` and recovery SHA256 is `eed4aca8...2e92`. Both updates were
finite and nonzero, all retained parent parameters were exactly unchanged, and
all physical hard-failure counts were zero. See
[`target_oriented_structured_gpp_broad_training_admission_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_broad_training_admission_result_20260830.md).

Paid job `14881105` at commit `af5f426` exited before Python after two allocated
seconds because old Bash rejected an empty optional argument array under
`set -u`. It performed no training and wrote no checkpoint or report. Commit
`c24f21b` removes the array and always passes an explicit update count. Bash,
Ruff, 11 focused tests, and a one-second free production-branch launch smoke
pass the former failure point. See
[`target_oriented_structured_gpp_broad_training_launch_failure_20260831.md`](research/daily_coarse_graining/target_oriented_structured_gpp_broad_training_launch_failure_20260831.md).

Corrected job `14883081` completed all 1,000 updates at commit `f040679`; report
SHA256 is `5cfab5a6...bee9`. The fixed monitor loss fell from `0.82876149` to
`0.08593347`, an 89.63% reduction, and 125/128 point-years improved. Aggregate
normalized bias improved from `-0.43192778` to `-0.09223345`, all physical hard
counts were zero, and all 93,504,164 retained parent coordinates were exactly
unchanged. The accepted selected checkpoint is update 1,000, SHA256
`3b173b89...4ec0`. See
[`target_oriented_structured_gpp_broad_training_result_20260831.md`](research/daily_coarse_graining/target_oriented_structured_gpp_broad_training_result_20260831.md).

The fresh target-blind independent screen is now frozen at contract SHA256
`894a0b2d...e6a7`. It keeps 16 windows in each temporal/spatial/joint and
7/30-day cell, but excludes every point-year consumed by the earlier panel
within the corresponding slice before metadata-only ranking. It reports
Teacher-state daily GPP separately from free-rollout GPP, respiration, NPP,
AGB, BGB, and ecosystem-carbon behavior. Promotion remains the existing
absolute 13-key 7/30-day gate with zero physical hard failures; no weak control
comparison is required.

The fresh-screen preparation, six-window admission, and formal 96-window
free-`gln01` execution are complete at commit `c422c37`; formal report SHA256
is `a9b9f6dc...98b`. All physical hard failures are zero, but the absolute
science gate fails. Teacher-state daily GPP NRMSE is `0.44496`; free-rollout
GPP fails all six cells at `0.73409` to `1.00839`, and growth respiration,
NPP, NEP, and NEE also fail all six. Biomass and ecosystem-carbon stocks still
pass all cells. The broad step-1000 checkpoint is not promoted.

The zero-training attribution is complete. Its hash-bound CPU finalizer
reproduces the ordinary fresh report within `6.94e-18`; every decomposition,
carbon-identity, finite-value, and physical hard gate passes. Teacher-state
cumulative GPP fails `5/6` cells, and added cross-day state feedback is
material in `6/6`. The diagnostic Teacher-GPP bridge makes growth respiration
and NPP pass all six cells, with median acceptance-score reductions of
`99.6822%` and `99.6525%`. The formal classification is
`direct_and_feedback_gpp_with_sufficient_downstream_cause`. See
[`target_oriented_structured_gpp_fresh_attribution_result_20260831.md`](research/daily_coarse_graining/target_oriented_structured_gpp_fresh_attribution_result_20260831.md).

The train-only state-feedback attribution is complete at commit `15171cd`;
report SHA256 is `1b2d1820...41c2a19b`. Carrying only
`slowproc_stomate_previous_step_state` captures a median `96.0598%` of the
7-day and `81.9454%` of the 30-day cumulative-GPP free-to-full-Teacher score
gap. It is positive in all six update/horizon cases and is the only component
that passes the frozen materiality rule. Water captures `9.1288%/5.1239%` and
does not pass, but remains a monitored secondary path because all three
7-day groups improve. All physical hard failures are zero, no optimizer or
checkpoint was created, and no fresh target, sealed test, v5 row, regenerated
Teacher datum, 48-record forcing reconstruction, or subdaily scan was used.
See
[`target_oriented_structured_gpp_state_feedback_attribution_result_20260831.md`](research/daily_coarse_graining/target_oriented_structured_gpp_state_feedback_attribution_result_20260831.md).

The user approved the first coupled GPP/carbon-state architecture. Its machine
contract is frozen at SHA256 `c7f81436...e7e74d`, and the local zero-training
implementation passes exact 2,386-coordinate/107-field layout, parameter
budget, eager/JIT forward, complete direct loss, adapter gradients, parameter
route, PFT permutation/inactive isolation, single-day state, and water/carbon
budget gates. The implementation retains the current structured-GPP
Attention, newly controls only `daily.fast_gpp`, and conditions it through two
PFT-query cross-attention blocks over day-start
`slowproc_stomate_previous_step_state` coordinates. It does not directly
predict STOMATE state.

The zero-update real-v6 GPU admission passed under job `14897461` at commit
`989a457`; report SHA256 is `cf8eb1d2...e623`. All 15,960,195 first-stage
trainable gradients are finite and every parameter leaf is nonzero. The
complete next-day state is finite, water and carbon budget residuals are zero,
all non-GPP parent outputs remain bit-exact, and cache-hot forward time is
`0.057055 s/day`. Peak live JAX device allocation is about 1.55 GB. No
optimizer or checkpoint was created. See
[`target_oriented_coupled_gpp_carbon_feedback_gpu_admission_20260901.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_carbon_feedback_gpu_admission_20260901.md).

The 13-day zero-training reachability audit passed under Explore1000 job
`14899147` at commit `b730b99`; report SHA256 is `dc53765e...d020`. Every
float64 `(2386, 2)` Jacobian is finite, every bare-soil derivative is exactly
zero, and the exact-nonzero PFT14 union contains 41 coordinates in 16 source
fields. Thirty-nine coordinates exceed the normalized reporting threshold;
the two smaller exact-nonzero coordinates remain part of the authoritative
union. All next-state and water/carbon budget gates pass. See
[`target_oriented_gpp_state_reachability_result_20260901.md`](research/daily_coarse_graining/target_oriented_gpp_state_reachability_result_20260901.md).

The field-grouped one-day state anchor and four-source objective calibration
passed under job `14899511` at commit `5582348`; report SHA256 is
`4f209b77...b5138`. All four source gradients are finite and nonzero over all
15,960,195 trainable parameters and all 190 leaves on each of prepared updates
`352-357`, `359`, and `360`. All physical hard failures are zero. The resolved
coefficients are direct GPP `1.0`, reachable state `0.6524275555272667`,
7-day science `0.4072260558556481`, and 30-day science
`0.28249951035415655`; the clip threshold is `12.808193184117403`. The
within-artifact combined gradient nevertheless conflicts with 7-day science
in `3/8` artifacts and 30-day science in `4/8`, so calibration alone does not
authorize training. See
[`target_oriented_coupled_gpp_objective_calibration_result_20260901.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_objective_calibration_result_20260901.md).

The discarded optimizer-proposal preflight completed as job `14902356` at
commit `b54c502` in `00:26:32`; report SHA256 is `dd43a960...d146ab`. All
gradients, the optimizer, both trainable-family steps, physical gates, fixed
parameters, and checkpoint integrity pass. Every source gradient aligns
positively with the realized negative parameter step, proving a common
first-order direction. The finite `4e-6` step is rejected: reachable state
falls `30.0427%`, but direct GPP rises `11.9170%`, 7-day science rises
`0.7175%`, and 30-day science rises `0.0550%`. The weighted total falls
`1.1798%`, which cannot override the per-source gates. The raw gradient norm
is below the clip threshold, so clipping is not causal. See
[`target_oriented_coupled_gpp_optimizer_preflight_result_20260901.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_optimizer_preflight_result_20260901.md).

The exact-half-rate `2e-6` proposal passed under job `14902878` at commit
`510a059` in `00:23:34`; report SHA256 is `6abb472d...f751c5`. Direct GPP,
reachable state, 7-day science, 30-day science, and the weighted total decrease
`4.08665%/16.3007%/0.003126%/0.039625%/4.37440%`. All direction, finite-value,
parameter-coverage, physical, fixed-parameter, and checkpoint-integrity gates
pass. The proposal and fresh optimizer were discarded; no checkpoint or
training update was written. See
[`target_oriented_coupled_gpp_optimizer_backoff_result_20260901.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_optimizer_backoff_result_20260901.md).

The target-blind schedule and first bounded-training contract are now frozen
at SHA256 `5378fe99...699cca`. Eight optimizer updates use 64 unique training
artifacts in deterministic eight-artifact macro batches; the calibration and
fixed eight-artifact train-only monitor panels are disjoint from training.
Every proposal is evaluated before retention, recovery is atomic after every
accepted update, and only passing update-4/update-8 train-only monitors can
write a non-promotable checkpoint. The corrected production-equivalent
free-`gln01` admission passes at commit `885693a`: all four source losses
decrease, all gradients and physical gates pass, and temporary recovery
round-trips without retaining the update. Paid job `14903328` then retained
update 1 and stopped at the update-2 finite-step gate. Update 2 improved the
weighted objective `0.686293%`, reachable state `6.17157%`, 7-day science
`1.00842%`, and 30-day science `0.550934%`, but direct GPP regressed
`2.63633%`; the proposal was discarded. No clipping, numerical, physical,
fixed-parameter, or source-checkpoint failure occurred. The stage did not
reach update 4, so it wrote no checkpoint. The active gate is a zero-training
optimizer-geometry and bounded-step audit at the exact update-1 recovery state.
It must distinguish inherited-AdamW direction rotation from finite-step
curvature before any objective change or continuation. Fresh validation,
continued training, 365-day rollout, Gate D2, sealed test, and product
acceptance stay closed. See
[`target_oriented_coupled_gpp_bounded_training_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_bounded_training_v1.md).
Admission evidence is in
[`target_oriented_coupled_gpp_bounded_training_admission_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_bounded_training_admission_20260902.md).
The completed result is
[`target_oriented_coupled_gpp_bounded_training_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_bounded_training_result_20260902.md).

The zero-training update-2 geometry audit is implemented. Its first free run
at commit `cc86e44` completed the full gradient and inherited scale-one
forward, then stopped on an unnecessarily fatal `1e-10` parent-reference
comparison. No proposal or recovery was modified. The corrected contract at
SHA256 `7585cc5b...79f198`
retains only the frozen scientific rejection outcome as an execution gate and
reports numeric differences without aborting. It loads the exact update-1
recovery, recomputes the rejected update-2 macro gradient once, compares
inherited and zero-moment AdamW displacements, and evaluates each on the fixed
`1, 1/2, 1/4, 1/8, 1/16` scale grid. Every proposal is discarded. The active
gate is the corrected contract freeze and free-`gln01` rerun. See
[`target_oriented_coupled_gpp_update2_geometry_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_update2_geometry_audit_v1.md).

The corrected audit completed at commit `b7697f5`; report SHA256 is
`09209273...b60956`. The inherited AdamW direction has direct-GPP cosine
`-0.00493710`, and none of its five scales passes. Reset AdamW has positive
cosine with all four source gradients; its full-scale proposal improves direct
GPP, state, 7-day, 30-day, and weighted losses by
`4.04281%/2.91495%/0.609099%/0.345835%/1.86634%` and passes every hard gate.
Full scale and `1/16` pass while intermediate scales fail, so scalar
backtracking is not supported. The active gate is to freeze and locally
implement a fresh-moments-per-macro-update bounded-training amendment, then
run one discarded consecutive-update free-GPU admission. Paid continuation
and fresh validation remain closed. See
[`target_oriented_coupled_gpp_update2_geometry_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_update2_geometry_result_20260902.md).

That amendment is now frozen and locally qualified at contract SHA256
`a8c362fe...31eda11`. It keeps the architecture, four-source objective,
learning rate, clip threshold, prepared schedule, and finite-step gates
unchanged. Updates 2 and 3 each start with zero AdamW moments and optimizer
step one; accepted parameters pass only in memory between them, and all state
is discarded after reporting. The active gate is the two-consecutive-update
free-`gln01` admission. Both updates must pass before a new paid contract may
be proposed. See
[`target_oriented_coupled_gpp_fresh_moment_admission_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_fresh_moment_admission_v1.md).

The free admission then completed at commit `eb4eb19`; report SHA256 is
`3e07fb8a...15f834`. Update 2 passed with all four source improvements, but
update 3 failed: GPP and 7-day science regressed `0.758192%` and `0.0372167%`
while state, 30-day science, and the weighted objective improved
`4.32042%/1.01398%/0.766949%`. All gradients and physical gates passed, clip
scale was `1.0`, and no state was retained. The update-3 weighted raw gradient
already has negative cosine with 7-day and 30-day gradients (`-0.274594` and
`-0.0310940`), so fresh moments are not a sufficient training rule. The
active gate is a zero-retention update-3 common-direction and finite-step
audit at the state after the admitted update 2. Paid continuation and fresh
validation remain closed. See
[`target_oriented_coupled_gpp_fresh_moment_admission_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_fresh_moment_admission_result_20260902.md).

The update-3 common-direction audit is now frozen and locally qualified at
contract SHA256 `956940a1...55cf09`. It reconstructs accepted fresh update 2
in memory, recomputes the four update-3 gradients, solves the tested
minimum-norm simplex problem, and compares direct versus fresh-AdamW
realizations at matched displacement scales `1,1/2,1/4,1/8,1/16`. Numeric
reproduction differences are reporting-only. Every proposal is discarded and
no validation value is read. The active gate is one free-`gln01` execution.
A common finite step opens only a three-to-five-update free recalibration
smoke; failure redirects work to the long-horizon-primary objective hierarchy,
not another architecture change or paid continuation. See
[`target_oriented_coupled_gpp_update3_common_direction_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_update3_common_direction_audit_v1.md).

The free audit then passed at commit `e704749` in `4,521.701` seconds; report
SHA256 is `642086af...ec8d`. The minimum-norm direction has source cosines
`0.226096/0.226096/0.226096/0.248111` for GPP, state, 7-day, and 30-day loss.
All five direct scales `1,1/2,1/4,1/8,1/16` improve every source and pass all
physical gates; at full scale the improvements are
`0.9890%/6.5908%/3.7807%/6.5117%`. Fresh AdamW is nonmonotonic and fails at
`1/2` and `1/4`, so it is not the primary realization. The active gate is to
freeze and locally qualify a three-to-five-update no-checkpoint free-`gln01`
smoke that recomputes and directly realizes the common direction after every
accepted macro update. Paid continuation and fresh validation remain closed.
See
[`target_oriented_coupled_gpp_update3_common_direction_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_update3_common_direction_result_20260902.md).

The three-update direct-recalibration smoke is now frozen and locally
qualified at contract SHA256 `bb79be4c...76bb4`. It reconstructs fresh update
2, then recomputes the four-source minimum-norm direction before each of
scheduled updates 3-5 and directly realizes fixed global parameter norm
`0.00775897335105`. Each full step must improve all four sources and pass all
physical gates; the first rejection stops the run. It writes only a report and
retains no state. Thirty-six focused and related tests plus static checks pass.
The active gate is one free-`gln01` execution. Paid training and fresh
validation remain closed. See
[`target_oriented_coupled_gpp_direct_recalibration_smoke_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_direct_recalibration_smoke_v1.md).

The corrected free smoke then completed at commit `a32db38` in `2,552.884`
seconds; report SHA256 is `0c2d3e21...2fd2`. Updates 3 and 4 pass with all four
sources improving, but update 5 rejects the fixed full step: state, 7-day, and
30-day losses improve `28.6952%/9.9005%/10.6083%`, while GPP regresses
`58.1509%` and weighted loss regresses `9.1686%`. Its minimum source cosine is
still strong at `0.495394`, so the failure is finite curvature, not a missing
common direction. A local GPP quadratic diagnostic places the improvement
crossing near scale `0.254` and minimum near `0.127`. The active gate is a
zero-retention update-5 audit on scales `1,1/2,1/4,1/8,1/16,1/32`. Paid
training and fresh validation remain closed. See
[`target_oriented_coupled_gpp_direct_recalibration_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_direct_recalibration_result_20260902.md).

The update-5 bounded-scale audit is now frozen and locally qualified at
contract SHA256 `835a8ee8...69cbd`. It reconstructs accepted direct updates 3
and 4 plus the rejected update-5 full step in one process, then evaluates
scales `1,1/2,1/4,1/8,1/16,1/32` on all eight update-5 artifacts. The full
record is reused and every smaller scale receives a real finite-step and
physical evaluation. It retains no state and reads no validation value.
Thirty-nine related tests plus static checks pass. The active gate is one
free-`gln01` execution; paid training remains closed. See
[`target_oriented_coupled_gpp_update5_scale_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_update5_scale_audit_v1.md).

The free update-5 scale audit then completed at commit `25ec5bc` in
`2,696.660` seconds; report SHA256 is `2199a0b7...5081`. Scales `1/4`, `1/8`,
`1/16`, and `1/32` pass all four source and physical gates. The maximum
accepted scale `1/4` improves GPP/state/7-day/30-day by
`0.3069%/10.4394%/2.5014%/2.6021%` and the weighted objective by `2.9544%`.
Scale `1/2` is rejected because GPP worsens `9.3058%` despite an aggregate
improvement. This confirms a bounded trust region rather than an architecture
failure. The active gate is to freeze and locally qualify a no-checkpoint
scale-selecting direct-recalibration smoke that recomputes the common
direction at every update and applies only the largest passing frozen scale.
Paid training and fresh validation remain closed. See
[`target_oriented_coupled_gpp_update5_scale_audit_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_update5_scale_audit_result_20260902.md).

The scale-selecting continuation is now frozen and locally qualified at
contract SHA256 `3b7109fc...8a5`. It reconstructs update 5 at selected scale
`1/4`, then tests new updates 6 and 7. Every update recomputes the four-source
common direction and evaluates `1,1/2,1/4,1/8,1/16,1/32` from largest to
smallest, applying only the first passing scale and retaining nothing. All 42
coupled-GPP/reachability tests and static checks pass. The active gate
is one free-`gln01` execution with a 120-minute timeout. Paid training and
fresh validation remain closed. See
[`target_oriented_coupled_gpp_scale_selecting_recalibration_smoke_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_scale_selecting_recalibration_smoke_v1.md).

The free scale-selecting execution then passed at commit `aaeea61` in
`3,402.216` seconds; report SHA256 is `c8ae64e5...42a0b8`. Updates 5-7 select
scales `1/4`, `1`, and `1/2`. Their weighted four-source objectives improve
`2.9544%`, `3.5294%`, and `3.6991%`, respectively; every individual source and
physical gate also passes. No parameter or checkpoint is retained. See
[`target_oriented_coupled_gpp_scale_selecting_recalibration_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_scale_selecting_recalibration_result_20260902.md).

The active gate is the frozen no-checkpoint non-test 7/30-day transfer audit at
contract SHA256 `110d3b4a...ab571d`. It compares the state before update 5
with the selected state after update 7 on the eight held-aside train/train
monitor artifacts: 64 seven-day and 32 thirty-day windows. Both horizon mean
losses must improve; per-artifact unanimity is report-only. The panel did not
participate in gradients or scale selection. One free `gln01` execution is
allowed with a 120-minute timeout. Paid training and fresh validation remain
closed. See
[`target_oriented_coupled_gpp_scale_selected_transfer_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_scale_selected_transfer_audit_v1.md).

The free transfer execution then completed at commit `67a4d40` in
`3,532.066` seconds; report SHA256 is `b77e92fe...f625`. The preregistered
primary gate passes: independent 7-day and 30-day component-resolved science
losses improve `8.3089%` and `13.4337%`, with `6/8` and `8/8` artifacts
improving, and all physical gates pass. Reachable one-day state improves
`18.2362%`. Direct teacher-state GPP, however, regresses `39.2867%` and
improves on only `1/8` artifacts. See
[`target_oriented_coupled_gpp_scale_selected_transfer_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_scale_selected_transfer_result_20260902.md).

The persistent 100-update scale-selecting runner passed free `gln01` admission
at commit `7d57887` in `4,669.897` seconds. It reconstructed update 7, ran the
detailed eight-artifact monitor, selected scale `1` for one discarded update,
and removed its temporary recovery. Initial cumulative GPP scores remain poor
at `0.417840` for 7 days and `0.497822` for 30 days, so this accepts the runner,
not the model's absolute science.

The first paid launch, job `14911154`, was a zero-training Bash failure and was
corrected at `64185d8`. Corrected job `14911228` then completed normally in
`02:18:50` and stopped at the frozen update-10 scientific gate after ten
accepted updates. Every update decreased all four sources on its own batch,
and the fixed monitor improved reachable state `15.1638%`, 7-day science
`29.5626%`, and 30-day science `31.8176%`. Cumulative GPP improved `30.2653%`
and `33.9401%` at 7/30 days. Direct teacher-state GPP, however, worsened
`24.1562%` and improved on only `1/8` monitor artifacts. No checkpoint was
selected; the retained recovery is attribution-only.

The bounded zero-training direct-GPP generalization audit completed on free
`gln01` at commit `52c60e7` in `7,117.794` seconds; report SHA256 is
`23327c3c...51cd9`. The first adjacent monitor regression is persistent update
2. That step improves its scheduled-group direct GPP by `5.8362%` but worsens
the fixed monitor by `2.2197%`. The scheduled and monitor mean GPP gradients
are strongly aligned at cosine `0.973434`, while the applied four-source
direction has monitor cosine only `0.024148`; artifact-level conflicts and
finite-step nonlinearity remain. Across ten updates monitor direct GPP worsens
`27.6762%`. All engineering gates pass and no state is retained. See
[`target_oriented_coupled_gpp_direct_generalization_result_20260903.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_direct_generalization_result_20260903.md).

The persistent-update-2 direct-step transfer audit completed on free `gln01`
at commit `48eb1ae` in `3,935.020` seconds; report SHA256 is
`df2593b...966c94`. The train-only selector chose scale `1/4`: all scheduled
sources improve by `1.0764-3.6708%`, and the four disjoint guard-group direct
GPP means improve by `0.2811-1.7612%`. The fixed monitor, which was excluded
from selection, nevertheless worsens by `0.5107%`, with `4/8` artifacts
improving. The classification is `train_only_scale_fails_monitor`; no state
was retained. See
[`target_oriented_coupled_gpp_direct_step_transfer_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_direct_step_transfer_result_20260904.md).

The zero-retention multi-group direct-GPP direction audit completed on free
`gln01` at commit `c5c0a56` in `4,874.724` seconds; report SHA256 is
`cf10f5e...663577`. A common eight-constraint direction exists with minimum
cosine `0.150436`, and all six frozen scales improve every scheduled source
and all four train-only guard-group GPP means. The train-only selector chooses
scale `1/2`. The fixed monitor nevertheless worsens by `0.4731%`, with `4/8`
artifacts improving. The classification is
`multi_group_direction_fails_monitor`; all engineering and scheduled physical
gates pass and no state is retained. Group-mean direction construction is
therefore closed as the next training rule. See
[`target_oriented_coupled_gpp_multi_group_direction_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_multi_group_direction_result_20260904.md).

The artifact-aware direction audit completed on free `gln01` at commit
`77d5a45` in `3,630.564` seconds; report SHA256 is `67bde6d8...09cbfe`. A
43-constraint common direction exists with minimum cosine `0.0422923`. The
first scale `1/2` improves all four scheduled sources, all `40/40` train-only
direct-GPP artifacts, and their combined mean by `3.4706%`. The fixed monitor,
read only after selection, improves by `2.7567%`, with `8/8` artifacts
improving. The classification is
`artifact_aware_direction_transfers_to_monitor`; all engineering and physical
gates pass and no state is retained. See
[`target_oriented_coupled_gpp_artifact_aware_direction_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_artifact_aware_direction_result_20260904.md).

The corrected three-update continuity smoke completed on free `gln01` at
commit `c345e64` in `5,467.939` seconds; report SHA256 is
`aa326bb4...52d69`. Updates 2, 3, and 4 select scales `1/32`, `1/16`, and
`1/2`; each improves all `40/40` train-only direct-GPP artifacts and all four
scheduled sources while passing every physical gate. The fixed-monitor mean,
read only after all choices are final, remains improved after every step and
finishes `6.2224%` below its initial value. The classification is
`artifact_aware_three_update_continuity_passes_monitor`. See
[`target_oriented_coupled_gpp_artifact_aware_continuity_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_artifact_aware_continuity_result_20260904.md).

The artifact-aware persistent run completed on free `gln01` at commit
`714aa0d` in `5,434.218` seconds. Logical update 2 selected scale `1/2` and
was retained; logical update 3 found a common first-order direction but no
tested scale passed all train-only gates. The run stopped at the scientific
gate with recovery retained exactly through update 2 and no checkpoint. The
report SHA256 is `ded2f4a5...7aa3a`; the rejected proposal SHA256 is
`a8a983ca...de728`.

The exact-zero follow-up completed on free `gln01` at commit `ca59ed5` in
`557.468` seconds. Exact-zero forward reproduces all eight update-3 seven-day
baselines exactly. The historical `1/64` displacement worsens the aggregate
by `0.5775%`, entirely because artifact `550` worsens by `6.1243%`; the other
seven artifacts improve.

The bounded artifact-550 trace then completed at commit `e436b96` in
`1,218.624` seconds; report SHA256 is `cdca61cd...1f618e`. The fourth of eight
seven-day windows contains the entire material regression. There is no
discrete or defined-mask change. Both mirrored `snowheat` coordinates first
cross `0.001` training scale on Day 2 and reach `0.00417` on Day 4; GPP then
jumps on Day 5 from `2.76734` to `4.86990`, carrying
`+1.51385` NPP and `+0.79464/+0.71857` AGB/BGB into later days. The source
router confirms that both snowheat mirrors can reach all three structured-GPP
branches, but temporal precedence and reachability do not establish
causality. See
[`target_oriented_coupled_gpp_localized_jump_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_localized_jump_result_20260904.md).

The matched snowheat intervention completed on free `gln01` at commit
`67fae75` in `890.625` seconds; report SHA256 is `a19c6230...a15a2e23`.
Control, treatment, and matched trajectories reproduce their engineering
references and have zero physical hard failures. Matching both snowheat
mirrors changes neither the Day-5 GPP jump nor the selected-window objective:
the retained fractions are `1.0` and `0.999999999997151`. The classification
is `snowheat_path_not_explanatory_for_day5_jump`. The potential factor carries
virtually the entire GPP ratio, while limitation and correction are nearly
unchanged. See
[`target_oriented_coupled_gpp_snowheat_causal_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_snowheat_causal_result_20260904.md).

The Day-5 crossed attribution completed on free `gln01` at commit `1c7dc9b`
in `915.314` seconds; report SHA256 is `c014ac09...7780c5`. All engineering
gates pass. The four cells are `CC=CT=2.7673363042` and
`TC=TT=4.8698997235`: parameters explain `100%` of the `+2.1025634193` GPP
jump, while carried state and parameter-state interaction explain exactly
zero. Every state-owner intervention is null. At both states,
`structured_gpp` alone is `100%` sufficient and necessary, while
`coupled_gpp_carbon_feedback` has zero effect and zero interaction. The
parameter path stays near `2.7673` through fraction `1/2` and jumps to
`4.8699` at `3/4`; both state paths are constant. See
[`target_oriented_coupled_gpp_day5_crossed_attribution_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_day5_crossed_attribution_result_20260904.md).

The user-approved bounded diagnostic frozen in
[`target_oriented_coupled_gpp_structured_precision_attribution_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_structured_precision_attribution_v1.md)
completed on free `gln01` at commit `587c3fd` in `935.288` seconds; report
SHA256 is `40c617ba...f5b693`, and every engineering gate passes. Of
12,879,363 weights, 8,974,854 change; 27.85% of changed coordinates move one
ULP, while 72.15% are distinct intermediate float32 values at fraction
`1/2`. Raising only the endpoint-weight interpolation to float64 leaves the
same full GPP jump in the same `10/16` to `11/16` interval, with endpoint-jump
ratio `0.9999997364`. This rejects float32 endpoint quantization as the sampled
cause. `potential_blocks` alone is `0.9999997732` sufficient and
`0.9999056398` necessary; every other top-level subtree is immaterial. See
[`target_oriented_coupled_gpp_structured_precision_attribution_result_20260905.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_structured_precision_attribution_result_20260905.md).

The first internal-localization contract used the whole-structured-tree
`10/16 -> 11/16` interval. Its free execution at `aae98e3` showed that the
isolated potential-block jump instead occurs at `11/16 -> 12/16`; endpoints
reproduced, but component attribution on the plateau was invalid. That v1
result is engineering-inconclusive and recorded in
[`target_oriented_coupled_gpp_potential_internal_attribution_result_20260905.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_potential_internal_attribution_result_20260905.md).

The corrected v2 diagnostic then passed on free `gln01` at commit `56ec8f2`
in `896.484` seconds; report SHA256 is `f781e4a9...07e0a3`. Full float64
retains the sharp jump with endpoint-effect ratio `0.9999999709`. Block 0 is
`1.0000001583` sufficient and `0.9999992086` necessary; block 1 parameters
are immaterial. Within block 0, the Q and K projections are each sufficient
but neither individually necessary, indicating redundant threshold crossing.
Head 7 switches from bank index 205 (`litterpart`, weight `0.459139`, entropy
`1.418627`) to index 153 (`DOC`, weight `1`, entropy approximately zero); its
float32 maximum logit rises from `8.1263` to `7796.5381`, and float64 makes
the same switch. See
[`target_oriented_coupled_gpp_potential_internal_attribution_v2_result_20260905.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_potential_internal_attribution_v2_result_20260905.md).

The zero-training head/token causal audit is now frozen in
[`target_oriented_coupled_gpp_attention_head_token_attribution_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_attention_head_token_attribution_v1.md).
On the same `11/16 -> 12/16` trees it tests paired block-0 per-head Q/K output
slices, a head-7 Q-only/K-only/Q+K crossed panel, exact DOC/litterpart logits,
and matched global versus head-7-only token masks. The free `gln01` run passed
at commit `7788d6b` in `904.470` seconds; report SHA256 is
`2c8a28d8...d94836`. Only block-0 head 7 is sufficient and necessary. A
head-7-only DOC mask leaves the lower endpoint unchanged and removes the
entire upper jump, while a litterpart mask has zero effect. Q-only and K-only
are each sufficient but neither alone is necessary; the paired Q+K slice is
necessary. See
[`target_oriented_coupled_gpp_attention_head_token_attribution_result_20260905.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_attention_head_token_attribution_result_20260905.md).

The user-approved Q/K-conditioned candidate is now frozen and implemented.
Only the two `structured_gpp.potential_blocks` attention layers change: every
projected Q/K vector is normalized per head and scored with one bounded
learnable temperature per block/head. The 16 temperatures initialize at
`sqrt(48)` and are bounded below `12`; no head or token is hard-coded. The
candidate has `109,464,375` total and `15,960,211` first-stage trainable
parameters. The matched reference uses the identical normalized Q/K path with
fixed `sqrt(48)` temperatures and is not a promotion baseline.

The candidate, coupled-parent, and structured-parent local suite passes all
`17` tests in `804.14` seconds, including forward/loss/gradient, initial
candidate/reference agreement, inherited Q/K/V/output gradient reach, parent
regression, PFT isolation/permutation, canonical single-day state, and
water/carbon budgets. Static, compile, shell, and diff checks pass. See
[`target_oriented_coupled_gpp_conditioned_attention_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_conditioned_attention_v1.md)
and
[`target_oriented_coupled_gpp_conditioned_attention_local_acceptance_20260906.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_conditioned_attention_local_acceptance_20260906.md).

The free-`gln01` real-v6 admission passed at commit `812e851` in `1753.274`
seconds. The parent jump from `2.7673221604` to `4.8699012434` is reproduced;
the candidate and fixed reference both give `5.3164683410` at both endpoints,
so the jump fraction is exactly zero and the maximum absolute conditioned
logit is `4.240962`. The real-v6 complete loss is finite, all `15,960,211`
trainable gradient coordinates are finite, both temperature blocks and both
potential-block Q/K/V/output paths have nonzero gradient, every declared
output/state value is finite where defined, and both water/carbon residuals
are zero. No optimizer or checkpoint was created. Report SHA256 is
`080f6b9d...233263`. See
[`target_oriented_coupled_gpp_conditioned_attention_gpu_admission_20260906.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_conditioned_attention_gpu_admission_20260906.md).

The zero jump proves the intended conditioning repair but the common
`5.3164683410` endpoint is not an accuracy result. The discarded update-3
preflight is now frozen and locally qualified at contract SHA256
`b29e92b5...cb97d27d`. It starts from the identity-exact accepted update-2
recovery, adds the conditioned trainable temperatures, and uses 43 constraints:
40 individual direct-GPP artifact gradients plus scheduled state, 7-day, and
30-day gradients. Its full displacement is calibrated from the new
architecture's relative gradient geometry and then tested at descending scales
through `1/64`; the old parent step norm is not inherited. Scale selection is
train-only. Only afterward are the eight fixed monitor artifacts loaded once
each and evaluated as initial/candidate pairs. Every proposal is discarded.

The corrected free-`gln01` discarded update-3 preflight completed at commit
`2adb422` in `6,410.496` seconds; report SHA256 is
`0aa30a0e...e9fe9b42a`. The 43-constraint common direction has minimum cosine
`0.3265653`, versus `0.0422923` for the parent architecture's earlier
artifact-aware audit. The candidate's own calibration selects parameter-delta
L2 `0.0005178796`, and the first multiplier `1` passes. All `40/40`
train-only artifacts improve, their direct-GPP mean falls `0.9204%`, and the
scheduled state, 7-day, and 30-day sources fall `0.7218%`, `0.6061%`, and
`2.5039%`.

The independently excluded fixed monitor also passes: all `8/8` direct-GPP
artifacts improve, mean direct GPP falls `0.9373%`, reachable state falls
`1.0276%`, and 7/30-day component-resolved science falls `0.7332%/0.6816%`.
The required science macro improves `0.7579%`, every physical gate passes, and
no state is retained. Absolute GPP, NPP, growth-respiration, and autotrophic-
respiration gates remain open at both horizons. The accepted classification is
`conditioned_discarded_update_transfers_but_absolute_science_remains_open`.
See
[`target_oriented_coupled_gpp_conditioned_attention_discarded_update_result_20260906.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_conditioned_attention_discarded_update_result_20260906.md).

The completed three-update persistent smoke was frozen under contract
SHA256 `ac65c5be...e5c9aff`. Logical updates 3-5 use 120 unique train-only
artifacts and recompute the 43-constraint direction plus architecture-aware
step calibration after every accepted state. Scale selection remains
train-only. A candidate is retained only when both its step-local monitor
change and its cumulative change from the immutable start pass; monitor
failure discards the candidate and preserves the preceding recovery. The run
is resumable, free-`gln01` only, and bounded to 240 minutes. See
[`target_oriented_coupled_gpp_conditioned_attention_persistent_training_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_conditioned_attention_persistent_training_v1.md).

Local qualification now passes 57 related tests in `374.82` seconds plus Ruff,
Python compilation, both shell syntax checks, manifest/result loading, actual
parser-branch coverage, CLI help, and whitespace checks. The immediate task is
now the update-4 derivative diagnostic described below. Paid execution,
fresh validation, 365-day rollout, Gate D2, sealed test, and product acceptance
remain closed. See
[`target_oriented_coupled_gpp_conditioned_attention_persistent_training_local_acceptance_20260906.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_conditioned_attention_persistent_training_local_acceptance_20260906.md).

Run `45ab263` completed in 3162.530 seconds, retained update 3, and stopped
at update 4 because its 30-day loss increased at every tested scale. All
40 direct-GPP artifacts, scheduled state, and 7-day losses improved; physical
and engineering gates passed. The active gate is a free, zero-retention
day/state derivative localization at saved update 3, artifact 442/window 0.
The completed `d9c4bd4` audit reproduces all source norms and forward losses;
seven of eight artifacts agree with finite differences, while 442 predicts
-0.019983 but actually changes by +0.0000907. Its first window has gradient
norm 1097.9 versus about 2 for the other windows; raw float64 accumulation
does not fix the discrepancy. The completed `2258c18` window audit reproduces
the reverse derivative -1097.9184, but forward tangent +2.6295 and finite
difference +2.6274 agree against it. The identical-scalar remat comparison
reproduced this discrepancy; the no-remat reverse failed because its temporary
buffers require 43.12 GiB on a 32 GiB GPU. Do not rerun that configuration.
The full 30-day streamed adjoint `0056141` completed in 1722.781 seconds:
reverse +2.62694, forward +2.62710, and finite differences +2.63857/+2.63083
agree in sign, with zero baseline hard failures. Saved-trace replay `4468c92`
completed in 502.559 seconds: the largest discrepancy is snowheat at 5.43e-6
relative; GPP differences are at most 5.376e-7 and downstream linearized effects
about 1e-9 to 1e-8. Discarded four-window step `c726446` passed in 890.713
seconds: predicted/actual changes -0.000882413/-0.000882278 at norm 1e-4,
both step magnitudes descend in all windows, and derivative errors are below
0.15%. The discarded original-panel update-4 integration has now completed.
Its actual training-side update passes at scale 1, with direct GPP, state,
seven-day and thirty-day losses all decreasing. Overall integration is not
admitted: seven of 64 seven-day observation-score comparisons fail; all 32
thirty-day comparisons pass. The active task is to locate those seven
differences, keeping the architecture, loss and tolerances unchanged, before
any fixed-monitor check or retained update. See the result and bounded next
diagnostic plan in
[`conditioned_attention_streamed_integration_20260908.md`](research/daily_coarse_graining/conditioned_attention_streamed_integration_20260908.md).
Local qualification passed 25 tests. Free `gln01` execution at `1e6c521`
completed in 2131.7955 seconds; wrapper PID `39230` has exited. The complete
remote report is
`runtime/outputs/diagnostics/conditioned-streamed-integration-1e6c521.json`.
Do not restart the completed run or use the partial local SCP copy. Start
with source comparison and, if needed, a focused zero-update replay of a
failing and a passing seven-day window. Accepted recovery remains update 3.
Source comparison found the same extraction and loss formulas. The active
bounded replay is now specified in
[`conditioned_attention_seven_score_20260908.md`](research/daily_coarse_graining/conditioned_attention_seven_score_20260908.md):
artifact 322/windows 5 and 1, original eight-window totals, compact/verbose
trajectory comparisons and directional finite differences, no update.
Free `gln01` replay `c530f90` completed in 1153.55476 seconds. Compact
trajectories reproduce production scores exactly; verbose recording differs.
Masks/counts agree, and both windows' production finite-difference checks
agree within 0.345%. The active follow-up is the same two-window test with a
compact adjoint tape containing full recurrent state and science observations
only, plus per-day source-mapped state differences. See the fixed follow-up
section in the above contract. Production training remains unchanged.
The compact follow-up completed at `d31e93c` in 615.589 seconds and is not
admitted: window 5 score drift remains -7.37018e-6 and its smaller-displacement
derivative errors exceed 1%. Window 1 passes; physical failures are zero.
The source-mapped same-input thermal/recurrent diagnostic completed in
1328.8 seconds on free gln01 at `ff9310c`; see
[`conditioned_attention_thermal_source_20260908.md`](research/daily_coarse_graining/conditioned_attention_thermal_source_20260908.md).
Same-input one-day execution reproduces the initial snowheat discrepancy;
snow-only incoming differences reproduce almost all Day-4 amplification.
An existing-artifact read on free test01 shows both Teacher windows are
snow-free, with zero snowheat throughout, while the candidate accumulates
roughly 30--35 million by Day 7. The source-backed warm snow-free correction
is now implemented, default off, with the Attention backbone unchanged.
Ten local regime/state/budget/thermal-AD tests pass. A free-CPU read of 528
existing v6 day positions agrees exactly for nine snow endpoint families;
this panel contains no real active-snow examples. The free-GPU paired 7/30-day
screen completed at `4d85256` in 1347.72 seconds. All three repaired windows
have exact source snow states, zero hard failures and passing
production/recorded scores. Science is mixed: composite changes are -2.79%,
+238.69% and +93.22%; GPP drives the two regressions. This is forward
consistency admission only, not an accuracy improvement or model promotion.
See [`conditioned_attention_snow_free_boundary_20260908.md`](research/daily_coarse_graining/conditioned_attention_snow_free_boundary_20260908.md).
The corrected-head streamed/JVP/FD validation now passes at `e6a00b2` in
1402.18 seconds: all three windows, eight direction/epsilon pairs and six
negative-gradient descent checks pass. Maximum derivative discrepancy is
0.0650%, all physical/mask/score checks pass. The frozen gate and evidence are
[`conditioned_snow_free_gradient_20260908.md`](research/daily_coarse_graining/conditioned_snow_free_gradient_20260908.md):
the process completed; do not restart it. These are linearized window scores,
not a full multi-artifact update or final accuracy improvement.
The corrected-head full train-only discarded-step integration is complete under
[`conditioned_snow_free_integration_20260909.md`](research/daily_coarse_graining/conditioned_snow_free_integration_20260909.md).
Run `90f6403` completed in 3363.25 seconds and is admitted: all 96 score
comparisons pass, the first scale passes, all 40 direct-GPP artifacts improve,
and scheduled 7/30-day science losses decrease 0.660%/0.583%. Physical gates
pass. Absolute GPP/NPP/growth/autotrophic-respiration gates remain open.
Paired fixed-monitor transfer is now completed and admitted under
[`conditioned_snow_free_monitor_20260909.md`](research/daily_coarse_graining/conditioned_snow_free_monitor_20260909.md).
Run `6ed00b0` completes in 1121.38 seconds. Candidate reconstruction, training
score reproduction and monitor transfer all pass. All eight monitor artifacts
improve direct GPP (mean -0.89965%); all 18 science-key/horizon losses improve
(macro -0.63715%). Physical gates pass, but absolute GPP/NPP/growth/autotrophic
respiration gates remain open. This is a previously exposed development
monitor, not independent validation or proof of sustained training progress.
The candidate is saved at `runtime/outputs/diagnostics/conditioned-snow-free-monitor-6ed00b0.candidate.pkl`.
Do not reconstruct it again. The original uncorrected recovery remains update 3;
the corrected continuation now has its own accepted update-14 recovery below.
The completed continuation froze this saved
candidate, checked reload/persistence, and specified predetermined training panels,
newly recomputed directions, corrected monitor baselines, cumulative progress,
absolute accuracy and resource bounds before execution. Do not reuse old
uncorrected recovery/monitor identities. The user approved implementation,
free-node checks and submission to gnall. Final acceptance remains closed.
The current execution plan is:
[`conditioned_snow_free_continuation_plan_20260909.md`](research/daily_coarse_graining/conditioned_snow_free_continuation_plan_20260909.md).
It freezes ten new updates (logical 5--14), 400 distinct existing artifact
IDs, unchanged admitted update rules, per-update automatic monitor/recovery,
and summaries after 2/5/10 new steps. Schedule separation was checked locally.
Approved paid envelope is one V100 + 8 Slurm CPUs, 6 hours maximum (CNY 16.56
additive maximum). Local checks and the saved-candidate free-GPU admission
pass (518.64 seconds, 408 readable artifacts, matching corrected monitor,
all hard gates pass). **gnall job 14979295 completed all ten updates** in
02:49:07, with recovery confirmed at logical update 14. Direct monitor loss
falls 9.35793% and science macro falls 7.34134%; all steps and 960 gradient-score
comparisons pass, with one daily-reverse compilation. Last-five-step direct
loss falls a further 5.47676%; no observed plateau. GPP/NPP/growth/autotrophic
respiration absolute gates still fail. See the
[result review](research/daily_coarse_graining/conditioned_snow_free_continuation_result_20260909.md).
The next-stage implementation now follows
[conditioned_continuation_v2_plan_20260909.md](research/daily_coarse_graining/conditioned_continuation_v2_plan_20260909.md):
20 updates (15--34) from update 14, unchanged architecture/update rules, explicit
800 training-artifact exposures / 678 distinct IDs, and paired independent
24-window temporal/spatial/joint validation. Bounded progress reporting passes
free-CPU measurement (late event IO 0.4603s -> 0.00244s). Local checks pass;
free CPU preparation is in progress. Complete free-GPU reload/monitor/independent
baseline admission before requesting the new one-V100/8-CPU/8h paid envelope
(CNY 22.08 maximum). No further paid submission is authorized yet.
The remote checkout no longer needs to remain pinned for this completed job.
See the
[execution record](research/daily_coarse_graining/conditioned_snow_free_continuation_execution_20260909.md)
for paths and resume semantics.
The preceding completed contract is
[`conditioned_attention_streamed_step_20260908.md`](research/daily_coarse_graining/conditioned_attention_streamed_step_20260908.md).
This does not yet establish the production reverse failure's root cause. See
[`conditioned_attention_streamed_adjoint_20260907.md`](research/daily_coarse_graining/conditioned_attention_streamed_adjoint_20260907.md).
The preceding comparison is recorded in
[`conditioned_attention_derivative_modes_20260907.md`](research/daily_coarse_graining/conditioned_attention_derivative_modes_20260907.md).
The prior local-state summary included an undefined NaN and cannot name the
causal process; its diagnostic mask is corrected without changing the model.
Do not restart training yet. Prior evidence is in
[`conditioned_attention_update4_derivative_20260906.md`](research/daily_coarse_graining/conditioned_attention_update4_derivative_20260906.md).

The persistent protocol remains recorded in
[`target_oriented_coupled_gpp_artifact_aware_persistent_training_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_artifact_aware_persistent_training_v1.md).
The first free launch at `d9f1238` exposed an incorrectly transcribed parent
result SHA and stopped before GPU allocation. The corrected contract uses the
already frozen and server-confirmed `764b14e...81168`; the failed preflight
wrote no training state or result.
The completed smoke contract is
[`target_oriented_coupled_gpp_artifact_aware_continuity_smoke_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_artifact_aware_continuity_smoke_v1.md).
The completed one-step audit contract is
[`target_oriented_coupled_gpp_artifact_aware_direction_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_artifact_aware_direction_audit_v1.md).
The completed predecessor contract is
[`target_oriented_coupled_gpp_multi_group_direction_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_multi_group_direction_audit_v1.md).
The predecessor training evidence remains in
[`target_oriented_coupled_gpp_persistent_scale_training_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_persistent_scale_training_v1.md)
and
[`target_oriented_coupled_gpp_persistent_scale_training_admission_result_20260903.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_persistent_scale_training_admission_result_20260903.md),
with the launch correction in
[`target_oriented_coupled_gpp_persistent_scale_training_launch_failure_20260903.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_persistent_scale_training_launch_failure_20260903.md)
and the scientific result in
[`target_oriented_coupled_gpp_persistent_scale_training_result_20260903.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_persistent_scale_training_result_20260903.md),
with the completed attribution protocol in
[`target_oriented_coupled_gpp_direct_generalization_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_direct_generalization_audit_v1.md).
The completed transfer contract is
[`target_oriented_coupled_gpp_direct_step_transfer_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_direct_step_transfer_audit_v1.md).

The first free execution at `e755bb1` stopped after bootstrap because compact
direct-generalization replay records lack the persistent result's
`finite_step_accepted` field. It wrote no output or state. Commit `4953939`
now reads the accepted persistent records and validates replay metadata before
GPU work; the completed scientific contract was unchanged.

Job `14899347` remains an engineering-failure record: its order-sensitive
dictionary guard stopped before objective evaluation. The exact-key-set JIT
repair was exercised by the successful calibration job and is regression
tested. No result or checkpoint came from the failed job.

The approved design is
[`target_oriented_coupled_gpp_carbon_feedback_architecture_review_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_carbon_feedback_architecture_review_v1.md).
It recommends coordinate-level cross-attention over the selected day-start
slow-process carbon state while keeping `daily.fast_gpp` as the only newly
controlled output. The accepted local evidence is
[`target_oriented_coupled_gpp_carbon_feedback_local_acceptance_20260831.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_carbon_feedback_local_acceptance_20260831.md).
This is a frozen architecture contract, not a training contract.

Do not continue the unchanged specialist, tune from fresh targets, or open
365-day rollout, Gate D2, sealed test, or product acceptance. The completed
scope contract is
[`target_oriented_structured_gpp_state_feedback_attribution_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_state_feedback_attribution_v1.md).
The completed attribution contracts remain
[`target_oriented_structured_gpp_fresh_screen_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_fresh_screen_v1.md)
and
[`target_oriented_structured_gpp_fresh_attribution_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_fresh_attribution_v1.md).

The completed zero-update calibration protocol is recorded in
[`target_oriented_distribution_aware_guard_calibration_v1.md`](research/daily_coarse_graining/target_oriented_distribution_aware_guard_calibration_v1.md)
at contract SHA256 `c1019409...e7179d7`, with accepted result in
[`target_oriented_distribution_aware_guard_calibration_result_20260828.md`](research/daily_coarse_graining/target_oriented_distribution_aware_guard_calibration_result_20260828.md).
The active training contract and one-update boundary are in
[`target_oriented_distribution_aware_bounded_training_v1.md`](research/daily_coarse_graining/target_oriented_distribution_aware_bounded_training_v1.md).
The accepted optimizer admission is in
[`target_oriented_distribution_aware_one_update_admission_20260828.md`](research/daily_coarse_graining/target_oriented_distribution_aware_one_update_admission_20260828.md).
See
[`target_oriented_daily_operator_contract_v1.md`](research/daily_coarse_graining/target_oriented_daily_operator_contract_v1.md),
[`target_oriented_attention_v1.md`](research/daily_coarse_graining/target_oriented_attention_v1.md),
[`target_oriented_source_shaped_maintenance_v1.md`](research/daily_coarse_graining/target_oriented_source_shaped_maintenance_v1.md),
[`target_oriented_source_shaped_maintenance_gpu_admission_20260827.md`](research/daily_coarse_graining/target_oriented_source_shaped_maintenance_gpu_admission_20260827.md),
[`target_oriented_source_shaped_train_train_screen_v1.md`](research/daily_coarse_graining/target_oriented_source_shaped_train_train_screen_v1.md),
[`target_oriented_source_shaped_zero_panel_result_20260827.md`](research/daily_coarse_graining/target_oriented_source_shaped_zero_panel_result_20260827.md),
[`target_oriented_source_shaped_one_update_admission_20260827.md`](research/daily_coarse_graining/target_oriented_source_shaped_one_update_admission_20260827.md),
[`target_oriented_source_shaped_branch_autodiff_admission_20260827.md`](research/daily_coarse_graining/target_oriented_source_shaped_branch_autodiff_admission_20260827.md),
[`target_oriented_source_shaped_branch_training_admission_20260827.md`](research/daily_coarse_graining/target_oriented_source_shaped_branch_training_admission_20260827.md),
[`target_oriented_source_shaped_branch_training_result_20260827.md`](research/daily_coarse_graining/target_oriented_source_shaped_branch_training_result_20260827.md),
[`target_oriented_source_shaped_coupling_attribution_v1.md`](research/daily_coarse_graining/target_oriented_source_shaped_coupling_attribution_v1.md),
[`target_oriented_source_shaped_coupling_attribution_result_20260828.md`](research/daily_coarse_graining/target_oriented_source_shaped_coupling_attribution_result_20260828.md),
[`target_oriented_source_owner_state_bridge_v1.md`](research/daily_coarse_graining/target_oriented_source_owner_state_bridge_v1.md),
[`target_oriented_source_owner_state_bridge_result_20260828.md`](research/daily_coarse_graining/target_oriented_source_owner_state_bridge_result_20260828.md),
[`target_oriented_distribution_aware_maintenance_v1.md`](research/daily_coarse_graining/target_oriented_distribution_aware_maintenance_v1.md),
[`target_oriented_distribution_aware_maintenance_gpu_admission_20260828.md`](research/daily_coarse_graining/target_oriented_distribution_aware_maintenance_gpu_admission_20260828.md),
[`target_oriented_distribution_aware_guard_calibration_v1.md`](research/daily_coarse_graining/target_oriented_distribution_aware_guard_calibration_v1.md),
[`target_oriented_attention_gpu_admission_20260821.md`](research/daily_coarse_graining/target_oriented_attention_gpu_admission_20260821.md),
[`target_oriented_bounded_training_v1.md`](research/daily_coarse_graining/target_oriented_bounded_training_v1.md),
[`target_oriented_state_anchor_amendment_20260821.md`](research/daily_coarse_graining/target_oriented_state_anchor_amendment_20260821.md),
[`target_oriented_representative_gradient_probe_20260823.md`](research/daily_coarse_graining/target_oriented_representative_gradient_probe_20260823.md),
[`target_oriented_full_step_preflight_acceptance_20260824.md`](research/daily_coarse_graining/target_oriented_full_step_preflight_acceptance_20260824.md),
[`target_oriented_formal_training_admission_20260825.md`](research/daily_coarse_graining/target_oriented_formal_training_admission_20260825.md),
[`target_oriented_formal_training_250_result_20260825.md`](research/daily_coarse_graining/target_oriented_formal_training_250_result_20260825.md),
[`target_oriented_update_1000_continuation_v1.md`](research/daily_coarse_graining/target_oriented_update_1000_continuation_v1.md),
[`target_oriented_update_251_admission_20260825.md`](research/daily_coarse_graining/target_oriented_update_251_admission_20260825.md),
[`target_oriented_update_1000_preparation_acceptance_20260825.md`](research/daily_coarse_graining/target_oriented_update_1000_preparation_acceptance_20260825.md),
[`target_oriented_update_250_training_dynamics_diagnosis_20260825.md`](research/daily_coarse_graining/target_oriented_update_250_training_dynamics_diagnosis_20260825.md),
[`target_oriented_formal_training_1000_result_20260826.md`](research/daily_coarse_graining/target_oriented_formal_training_1000_result_20260826.md),
[`target_oriented_flux_gradient_variance_audit_v1.md`](research/daily_coarse_graining/target_oriented_flux_gradient_variance_audit_v1.md),
[`target_oriented_flux_gradient_variance_audit_result_20260826.md`](research/daily_coarse_graining/target_oriented_flux_gradient_variance_audit_result_20260826.md),
[`target_oriented_component_resolved_flux_basis_v1.md`](research/daily_coarse_graining/target_oriented_component_resolved_flux_basis_v1.md),
[`target_oriented_component_resolved_geometry_result_20260826.md`](research/daily_coarse_graining/target_oriented_component_resolved_geometry_result_20260826.md),
[`target_oriented_flux_protected_virtual_step_v1.md`](research/daily_coarse_graining/target_oriented_flux_protected_virtual_step_v1.md),
[`target_oriented_flux_protected_virtual_step_result_20260826.md`](research/daily_coarse_graining/target_oriented_flux_protected_virtual_step_result_20260826.md),
[`target_oriented_flux_residual_tail_audit_v1.md`](research/daily_coarse_graining/target_oriented_flux_residual_tail_audit_v1.md),
[`target_oriented_flux_residual_tail_result_20260826.md`](research/daily_coarse_graining/target_oriented_flux_residual_tail_result_20260826.md),
[`target_oriented_flux_gradient_path_attribution_v1.md`](research/daily_coarse_graining/target_oriented_flux_gradient_path_attribution_v1.md),
[`target_oriented_flux_gradient_path_result_20260826.md`](research/daily_coarse_graining/target_oriented_flux_gradient_path_result_20260826.md),
[`target_oriented_snowheat_recurrence_intervention_v1.md`](research/daily_coarse_graining/target_oriented_snowheat_recurrence_intervention_v1.md),
[`target_oriented_snowheat_recurrence_result_20260827.md`](research/daily_coarse_graining/target_oriented_snowheat_recurrence_result_20260827.md),
[`target_oriented_snow_thermal_recurrence_intervention_v1.md`](research/daily_coarse_graining/target_oriented_snow_thermal_recurrence_intervention_v1.md),
[`target_oriented_snow_thermal_recurrence_result_20260827.md`](research/daily_coarse_graining/target_oriented_snow_thermal_recurrence_result_20260827.md),
[`target_oriented_snow_thermal_alias_recurrence_intervention_v1.md`](research/daily_coarse_graining/target_oriented_snow_thermal_alias_recurrence_intervention_v1.md),
[`target_oriented_snow_thermal_alias_recurrence_result_20260827.md`](research/daily_coarse_graining/target_oriented_snow_thermal_alias_recurrence_result_20260827.md),
[`target_oriented_maintenance_state_recurrence_intervention_v1.md`](research/daily_coarse_graining/target_oriented_maintenance_state_recurrence_intervention_v1.md),
[`target_oriented_maintenance_state_recurrence_result_20260827.md`](research/daily_coarse_graining/target_oriented_maintenance_state_recurrence_result_20260827.md),
[`target_oriented_maintenance_head_conditioning_audit_v1.md`](research/daily_coarse_graining/target_oriented_maintenance_head_conditioning_audit_v1.md),
[`target_oriented_maintenance_head_conditioning_result_20260827.md`](research/daily_coarse_graining/target_oriented_maintenance_head_conditioning_result_20260827.md),
[`coordinate_coupled_doc_transition_v4_memorization_result_20260820.md`](research/daily_coarse_graining/coordinate_coupled_doc_transition_v4_memorization_result_20260820.md)
and
[`doc_endpoint_target_loss_geometry_audit_result_20260820.md`](research/daily_coarse_graining/doc_endpoint_target_loss_geometry_audit_result_20260820.md),
[`doc_status_branch_magnitude_attention_v5_design_20260820.md`](research/daily_coarse_graining/doc_status_branch_magnitude_attention_v5_design_20260820.md),
and
[`doc_status_branch_magnitude_attention_v5_early_screen_result_20260821.md`](research/daily_coarse_graining/doc_status_branch_magnitude_attention_v5_early_screen_result_20260821.md).

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
and the 500-update matched fit are accepted, and the dual checkpoint set has
passed formal reload. The frozen 7/30-day screen rejected the trained
attention checkpoint; the training-sufficiency, loss, and shared-decoder
attribution now controls the next gate. Gates D2 and F remain queued.

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
