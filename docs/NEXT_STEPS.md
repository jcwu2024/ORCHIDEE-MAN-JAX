# Current Roadmap

Last updated: 2026-08-03.

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

- `main` at `7eb41be` contains the canonical user-facing PFT14 Teacher core.
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
- The active true-daily architecture is specified but not implemented.
- The current Teacher and data are scientifically scoped to PFT14. Gate B1's
  catalog foundation now gives PFTs stable IDs, source PFT/MTC identities,
  traits, parameter ownership, capabilities, and named execution layouts.
  PFT2-PFT13 remain structurally declared but scientifically unsupported and
  fail closed when active. Restart reads now verify/remap stable identity;
  remaining generic positional selectors and a compact-layout cold/restart
  lifecycle are not yet closed.
- Existing gradient tests establish training plumbing and finite derivatives
  in selected JAX paths. They do not yet establish physical-parameter gradient
  correctness for the complete Teacher or future daily surrogate.

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
training loss is easy to optimize.

### Gate F: Product Acceptance

- Teacher release: complete the 669-landpoint 1961-2010 forward acceptance for
  the canonical Teacher.
- Daily-surrogate release: pass held-out landpoint, year, forcing, and declared
  parameter-range tests; long rollout and restart gates; physical budgets; and
  measured inference cost.
- Multi-PFT scientific support: repeat source closure, data generation, and
  held-out validation for each newly claimed PFT.

## Immediate Work Order

The next technical task is to finish Gate B1, not another GPU experiment. Its
catalog, capability registry, stable parameter-row selection, modelout
selection, and restart read/write metadata are implemented. Remove remaining
generic positional selectors and prove a compact-layout cold/restart
lifecycle. Then freeze B2's neural
PFT-axis and physical-parameter ownership contract. Gate C follows with label
inventory and non-neural replay. D1 can validate physical-parameter gradients
against the selected Teacher while Gate C is completed. Only then should the
new daily neural implementation begin.

## Active Task Packet: Gate B1

This is the only implementation task to start from a new clone. Gate A is
closed and must not be rerun unless shared Teacher code changes. Gates B2-F
remain queued until this packet is closed.

### Inputs

- canonical Teacher core: `main` at `7eb41be` (identical in the research
  branch after merge `dda3627`);
- source-backed target design:
  [`porting/pft14_extensible_design.md`](porting/pft14_extensible_design.md);
- current PFT14 layout, parameter loaders, restart schemas, modelout selection,
  and hard-coded PFT/MTC capability branches;
- existing Fortran source ownership and PFT14 Oracle evidence.

### Implementation steps

1. Introduce a machine-readable PFT catalog with stable semantic IDs,
   canonical Fortran PFT/MTC identities, traits, parameters, capabilities, and
   active fractions.
2. Replace paper-driver ownership of positional PFT identity with a per-run
   layout derived from the catalog; preserve the named paper PFT14 layout.
3. Make restart and modelout metadata retain stable IDs across slot removal,
   addition, and reordering.
4. Move genuinely distinct source process families behind an explicit,
   source-backed capability registry. Keep ordinary trait/parameter changes
   data driven.
5. Add removal, coexistence, and slot-permutation tests before claiming any
   additional PFT as scientifically supported.

Current progress: steps 1-3 are implemented. Restart readers parse stable
NetCDF identity or require the explicit legacy paper layout, remap every
declared SECHIBA/STOMATE PFT axis, and production year handoff writes identity
metadata. Step 4 has a fail-closed capability registry; step 5 has exhaustive
restart-axis identity/remap coverage but not a complete compact-layout model
lifecycle. No additional PFT has been promoted beyond `structural_only`.

### Required artifacts

- one versioned PFT catalog schema and the paper PFT14 catalog instance;
- one capability registry with Fortran provenance;
- stable-ID restart/modelout layout metadata;
- focused tests for removal, coexistence, and permutation;
- an updated ownership inventory consumed by Gate B2.

### Definition of done

- PFT14 reproduces the existing canonical Teacher path under the catalog;
- array shapes and parameter ownership derive from the selected layout rather
  than a global `NVM=14` assumption;
- removing one supported PFT cannot change another PFT's state;
- two supported PFTs coexist with isolated state and fractions;
- slot reordering with stable IDs preserves remapped outputs;
- numbered source branches are explicit capabilities rather than falsely
  generalized behavior;
- adding a new PFT still requires its own source closure and numerical
  validation before scientific support is claimed;
- targeted regression tests, `git diff --check`, Ruff, and Python compilation
  pass;
- no neural training or 669-point production run is started by this task.
