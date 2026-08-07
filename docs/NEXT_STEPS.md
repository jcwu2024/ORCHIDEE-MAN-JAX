# Current Roadmap

Last updated: 2026-08-06.

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
  is rejected. Candidate selection remains unstarted; a representative
  six-point-year coherent production pilot passed. The formal product is now
  frozen as dataset manifest v6, typed contract v3, and canonical release v2.
  Formal generation begins with a reusable first batch, not another standalone
  calibration.
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

Do not regenerate this dataset or read and join historical v5 rows. The next
task is to admit the first neural candidate only through the stable
`bind_daily_operator(input_assembler=canonical_assembler,
process_head=candidate_head,
retained_tail=canonical_retained_tail_adapter)` boundary and the predeclared local/small
gates. Training must bind the accepted v6 manifest, statistics, and acceptance
report. The accepted review and full 47-label data matrix are in
[`gate_e1_architecture_data_readiness_review.md`](research/daily_coarse_graining/gate_e1_architecture_data_readiness_review.md);
the executable evidence packet remains
[`gate_e1_daily_operator_skeleton.md`](research/daily_coarse_graining/gate_e1_daily_operator_skeleton.md).

## Completed Task Packet: Gate C

This packet is complete and retained as its acceptance contract. Gates A-C
must not be rerun unless shared Teacher code or a frozen contract changes.
Gate D1, the candidate registry, and Gate E1 are complete. Gate E2 coherent
local implementation, representative six-point production pilot, formal v6
product freeze, global-plan verification, five-worker reuse benchmark, full
production, aggregation, and formal admission are complete. Neural candidate
training is next. Gates D2 and F remain queued.

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
