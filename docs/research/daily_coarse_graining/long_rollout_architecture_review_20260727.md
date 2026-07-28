# Long-Rollout Architecture Decision

Date: 2026-07-27

## Decision

The next neural experiment will compare the existing flat daily operator with
one process- and axis-aware operator on the complete admitted 669-landpoint
Teacher dataset. It will not continue tuning the rejected sparse-data
objectives or the first structured FiLM adapter.

The scientific target is not mathematically zero accumulated error. No finite
supervised surrogate can promise that for unseen forcing and parameters. The
promotion target is:

- valid physical state and exact discrete/restart semantics at every day;
- bounded free-rollout drift rather than monotonic error growth;
- small and unbiased annual water and carbon budgets;
- stable seasonal and multiannual distributions;
- useful spatial and temporal generalization;
- accepted 1961-2007 validation chains before one sealed 1961-2010 test.

The complete Teacher dataset must pass production admission before either
network is trained. Partial production shards are not architecture evidence.

## Established Diagnosis

The current `canonical_flat_v1` model compresses:

- 3,854 continuous state values and their finite masks to 128 values;
- five native-forcing records through a 64-wide recurrent encoder;
- 84 parameters, 735 landpoint-static values, annual conditions, and calendar
  values to 64 values;
- all three representations through a 256-wide fused residual trunk.

It has 1,963,369 parameters. This model learns a useful one-day operator, but
the nine-point evidence separates two failures:

1. Recursive state-distribution drift occurs even at seen landpoints.
2. Spatial generalization is substantially worse than temporal
   generalization.

The seven-day four-way diagnosis produced Day-7 normalized state RMSE of
`0.144765`, `0.133391`, `0.519286`, and `0.428353` for train/train,
train/validation-year, validation-point/train-year, and joint validation.
This rules out additional years at the same few points as the primary fix.

The retained source-backed daily transition is not the cause. Real one-day
forward/reverse and three-day scan gates close at approximately `1e-13`, with
exact discrete and defined-status behavior.

## Evidence That Constrains the Design

The following accepted decisions remain in force:

- `process_increment_v2` is rejected. It did not improve the predeclared
  seen-condition gates, so loss-weight searching must not be restarted.
- `on_policy_pushforward_v1` is rejected under nine-point data. It improved
  seen-condition rollout but harmed spatial and named carbon metrics.
- `structured_process_film_v1` is rejected under six training landpoints. It
  modestly improved spatial cases but regressed seen conditions and did not
  learn useful parameter/static dependence.
- Nonnegative projection of `carbon_32l`, `DOC`, and `deepC_peat` is a
  source-backed domain requirement and remains active.
- Exact masks, discrete state, deterministic mirrors, restart behavior, and
  the retained STOMATE tail must not become learned hidden state.
- The four calibrated paper parameters remain explicit inputs so parameter
  optimization remains mathematically available after network training.

The complete 535-landpoint training population changes spatial
identifiability. It justifies a new architecture comparison, but it does not
justify rerunning any rejected experiment unchanged.

## Literature Findings

### Rollout curriculum is useful but insufficient

[NeuralGCM](https://arxiv.org/abs/2311.07222) combines differentiable dynamics
with learned physics, increases rollout length during training, and adds an
explicit bias term. Its long stable integrations support mixed-horizon
training and mean-tendency control here. They do not imply that a land-surface
surrogate can be trained with only long-horizon loss.

[GraphCast](https://arxiv.org/abs/2212.12794) trains autoregressively, extends
the horizon from one to twelve steps, and uses gradient checkpointing. It also
shows a tradeoff between short-lead and longer-lead skill. This argues for
retaining a one-step anchor throughout training instead of replacing it with a
sequence of one-way fine-tuning stages.

### Model-state exposure addresses distribution shift

[Message Passing Neural PDE Solvers](https://arxiv.org/abs/2202.03376)
introduces the pushforward trick: create a model-generated input state, stop
its gradient, and train the next transition. This directly targets
model-state distribution shift at much lower memory cost than full long
backpropagation. The local pushforward result confirms the mechanism, but its
spatial regression means it can only return as a separately controlled
auxiliary arm after the new architecture passes.

[Learning to Simulate Complex Physics with Graph
Networks](https://arxiv.org/abs/2002.09405) uses structured noise to expose a
simulator to rollout-like state errors. Noise is not automatically beneficial:
its scale and covariance determine whether it improves stability or simply
damages one-step accuracy. It is therefore a later bounded A/B, not part of the
first architecture comparison.

### Long climate behavior needs its own gate

[ACE](https://arxiv.org/abs/2310.02074) demonstrates century-scale stability
from a one-step climate emulator, while also showing that runs with similar
validation loss can have materially different long-run climate bias. This is
direct evidence that one-step and seven-day scores cannot promote this model.

[DySLIM](https://proceedings.mlr.press/v235/schiff24b.html) regularizes the
long-run invariant distribution of learned dynamics. For this project, its
most transferable idea is a later loss on seasonal/annual state and budget
statistics, not a replacement for local transition supervision.

[PDE-Refiner](https://arxiv.org/abs/2308.05732) shows that low-amplitude modes
ignored by an aggregate loss can destabilize long rollouts. ORCHIDEE has no
spatial image or Fourier grid in one-landpoint mode, so its refiner
architecture should not be copied. The analogous risk is low-variance carbon
memory and pool fields being overwhelmed by wide state families.

## Candidate: `axis_process_coupled_v1`

The first candidate keeps the existing Markov Contract v5, normalized
persistence baseline, forcing encoder, physical reconstruction, and retained
tail. It changes representation capacity, not model semantics.

### Source-derived state encoders

The 3,854 state values remain exhaustively assigned to the eight audited
process groups:

| Process group | Width |
| --- | ---: |
| STOMATE carbon flux | 114 |
| STOMATE vegetation structure | 56 |
| STOMATE litter turnover | 218 |
| STOMATE carbon storage | 1,875 |
| STOMATE environment/phenology memory | 123 |
| Hydrology | 622 |
| Thermal energy | 653 |
| Surface exchange/finalize | 193 |

Within each group, contract leaf shapes and `axis_names` determine the
encoder:

- scalar, PFT, ordered-memory, ordered-vertical, carbon-pool, and general
  tensor leaves enter distinct contract-derived partitions;
- `axis_process_coupled_v1` applies one independent masked dense encoder to
  each nonempty partition and averages those partition tokens within the
  source process;
- the partition labels preserve the source axis semantics and leave a strict
  extension point for specialized depth-local or factorized encoders, but v1
  does not claim convolutional, positional, or tensor-factorized mixing;
- finite masks are encoded beside values exactly as in the current model.

Each group emits a small set of process tokens. Two lightweight
cross-process residual message-passing blocks combine each process token with
the mean global process token. This is an eight-process graph inside one
landpoint, not a geographic graph, attention layer, or U-Net.

### Persistent conditioning

Parameters, landpoint-static conditions, annual CO2/calendar, and the native
forcing summary remain separate named streams. Parameter and static
conditioning modulate:

1. the relevant process encoders;
2. both cross-process coupling blocks;
3. the corresponding output heads.

Condition use must be measured by held-out permutation/intervention tests.
The architecture is not promoted merely because a conditioning path has
nonzero gradients.

### Family-specific output heads

Each fast-day target family has its own residual/tendency head, connected to
its owner process tokens and the shared coupling tokens. The existing
persistence-centered target representation remains the baseline, including
defined/undefined flip prediction. This protects low-dimensional outputs from
competition with the 1,875-wide carbon-storage state while preserving the
single complete-day interface.

No learned recurrent memory may exist outside `S[d]`. A restart created after
any day must reproduce the same next prediction as an uninterrupted rollout.

## Implementation Status

`axis_process_coupled_v1` is implemented behind the shared architecture
registry but has not been trained or promoted. Its implementation identity is
bound to the complete process/axis/target layout SHA256
`bbe5e694f9bc2e8e4038f955cdbc967464b27bae883e95048ad092d1e734d24d`.
On the real Contract v5 widths:

- the flat control has 1,963,369 trainable parameters;
- the candidate has 1,954,041 parameters, a ratio of `0.99524898`;
- a real-width batch-2 JIT forward returns `(2, 2855)` fast-day values and
  `(2, 2)` dynamic-mask logits;
- reverse mode produces 117 finite gradient leaves;
- synthetic restart, checkpoint identity, masks, conditions, multiday scan,
  and no-hidden-memory gates pass.

These are implementation gates only. They are not accuracy, spatial
generalization, long-rollout, or promotion evidence. The matched A/B manifest
must remain unfrozen until the complete 669-point dataset passes final
production admission.

### Capacity control

The flat control and candidate must be matched to within 5% trainable
parameters, targeted near two million. Widths are chosen once from this budget
before looking at validation results. Compiler wall time, training throughput,
peak GPU memory, and inference cost are reported alongside accuracy.

## Training Protocol

Architecture and stability objectives are separated so failures are
attributable.

### Experiment A: architecture only

Train from scratch on admitted train-point/train-year data:

1. `canonical_flat_v1` control;
2. `axis_process_coupled_v1` candidate.

Both arms use identical samples, updates, optimizer, learning-rate schedule,
seed, physical projection, and the accepted canonical objective. One fixed
screening seed is followed by three confirmation seeds only if the candidate
passes the screening gates. The old sparse-data checkpoint is a historical
baseline, not an initialization for this comparison.

### Experiment B: rollout stability

Only after the architecture candidate passes:

- retain a one-step anchor in every update;
- sample mixed horizons `1/3/7/30` throughout training rather than
  irreversibly fine-tuning from short to long;
- backpropagate through sampled rollouts, using JAX rematerialization for the
  30-day arm;
- add a batch-time mean tendency-bias penalty per process family;
- retain named family losses for biomass, litter, NPP, growth/maintenance/
  heterotrophic respiration, hydrology, thermal state, and carbon stocks;
- add water/carbon budget residuals only where contract diagnostics define
  the complete terms.

A detached pushforward arm may be tested after this mixed-horizon baseline. It
must use the full training population and is accepted only if it improves
spatial as well as seen-condition gates. Structured noise is a separate later
A/B. Neither mechanism is bundled into the architecture comparison.

Annual or multiannual statistics regularization is considered only after
one-step and 30-day operator skill passes. It compares seasonal cycles,
annual-mean tendencies, stock distributions, and budget closure on unsealed
validation chains. It does not backpropagate through a 365-day or 50-year
graph.

### Frozen loss decomposition

The stability experiment uses six distinct controls. They must remain
separately reported rather than being hidden inside one aggregate score:

1. `L_fast`: normalized, mask-aware Huber loss on the complete predicted
   `B_fast[d]` Teacher boundary. This anchors the local daily approximation of
   the replaced half-hour processes.
2. `L_next`: process- and named-field metrics on the complete
   `S[d+1]` produced after the differentiable retained STOMATE transition.
   Gradients therefore teach the neural fast-day operator which interface
   values are needed for correct downstream GPP, NPP, respiration, biomass,
   litter, and soil-carbon behavior.
3. `L_rollout`: mixed `1/3/7/30`-day free-rollout loss. From Day 2 onward the
   model consumes its own state. A one-day anchor remains present in every
   stage so longer-horizon training cannot silently trade away local process
   skill.
4. `L_bias`: process-family batch-time mean tendency error,
   `mean(delta_predicted - delta_teacher)`. This targets small same-sign daily
   errors that ordinary RMSE can tolerate but that integrate into large
   multiannual stock drift.
5. `L_science`: explicitly reported and bounded errors for GPP, NPP,
   growth/maintenance/heterotrophic respiration, biomass and biomass
   tendency, litter, and soil-carbon stocks. Low-biomass cases use both
   absolute and relative metrics.
6. Hard semantic and physical constraints: exact masks and discrete state,
   restart-split equivalence, deterministic mirrors, retained-tail ownership,
   and source-backed nonnegative carbon stocks. These are reconstruction or
   admission rules, not soft penalties.

The conceptual objective is:

```text
L = L_fast + alpha * L_next + beta * L_rollout
    + gamma * L_bias + delta * L_science
```

The coefficients and exact normalized field inventories must be frozen in the
experiment manifest before results are visible. The rejected
`process_increment_v2` result prohibits an open-ended search over process
weights. Complete water or carbon budget penalties may be added only when the
contract exposes every required source term; an incomplete conservation
identity must not be invented.

Fluxes and stocks require different long-run gates. GPP, NPP, and respiration
are evaluated by daily/seasonal skill and monthly or annual integrated bias.
Biomass, litter, and soil carbon are evaluated by absolute state error,
relative error where the denominator is scientifically meaningful,
year-to-year tendency error, and the slope of multiannual drift. A small daily
flux RMSE does not pass if its mean bias drives an unbounded stock trajectory.

## Controlled A/B Gates

The screening matrix uses the frozen temporal, spatial, and joint validation
slices. The test split and 2008-2010 years remain sealed.

| Gate | Required evidence |
| --- | --- |
| Semantics | zero discrete and defined-status mismatch; restart-split prediction equivalence; no invalid physical stock |
| One step | no material regression in global or named process-family error |
| 7 and 30 days | teacher-forced and free rollout on temporal, spatial, and joint slices |
| Spatial use | parameter/static intervention changes the relevant outputs in the correct direction or improves held-out likelihood/error |
| Bias | process-family and annualized mean tendency bias reported, not hidden by aggregate RMSE |
| Efficiency | parameter count within 5%; measured compile, update, memory, and inference cost |

For promotion, the candidate must improve spatial and joint 7/30-day free
rollout without regressing the temporal/seen-condition and named carbon gates.
The result must reproduce across three seeds. Exact numerical thresholds are
written to the experiment manifest before GPU execution and cannot be changed
after results are visible.

After Experiment B, promotion proceeds in order:

1. 365-day free rollout on temporal, spatial, and joint validation;
2. restart-split equivalence within those rollouts;
3. complete 1961-2007 validation chains;
4. model and hyperparameter freeze;
5. one sealed 1961-2010 test evaluation.

Failure at a gate returns to the named architecture or objective hypothesis.
It does not trigger inspection of test landpoints or an unconstrained
hyperparameter search.

## Explicit Non-Choices

- Do not treat the flattened state as an image and apply a U-Net.
- Do not add geographic graph edges: each production sample is a
  source-independent landpoint transition.
- Do not add hidden LSTM/GRU memory across days because it changes restart
  semantics.
- Do not run full 365-day or 50-year backpropagation.
- Do not use a cascade of separate lead-time models as the first solution.
- Do not select architecture or normalization with sealed test data.
- Do not promise exact zero long-term error; require bounded, unbiased,
  physically valid climate behavior instead.

## Implementation Order

1. Complete and admit the 669-point Teacher dataset.
2. Generate contract-derived axis/process metadata and assert exhaustive,
   nonoverlapping state and target ownership.
3. Implement `axis_process_coupled_v1` behind the architecture registry,
   without changing the user-facing Teacher path.
4. Pass synthetic and real-batch forward/reverse, checkpoint identity,
   restart, mask, projection, and JIT gates.
5. Freeze the matched A/B manifest and run Experiment A.
6. Implement the mixed-horizon/bias objective only if Experiment A promotes
   the candidate.
7. Run the validation ladder; expose a user-facing daily surrogate only after
   the sealed release gate passes.
