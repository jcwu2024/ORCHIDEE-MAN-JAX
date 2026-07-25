# Autoregressive Daily Surrogate Training Review

Date: 2026-07-25

## Decision Context

The bounded `process_increment_v2` A/B completed on the frozen nine-point v5
dataset and equal update budget. It did not pass the seen-condition gate. On
train/train Day 7, global normalized state RMSE changed from `0.145471` to
`0.146456`, biomass from `0.014259` to `0.016716`, and litter partitioning from
`3.651536` to `4.008877`. The objective is rejected and must not replace
`canonical_multistep_v1`.

This result does not justify another loss-weight search. It requires a review
of how autoregressive simulators are trained under model-induced state
distribution shift.

## What The Current Trainer Does

The current multistep trainer:

1. samples a clean Teacher state at the start of a 1-, 3-, or 7-day window;
2. feeds the neural prediction back as the next day's state;
3. backpropagates through every day in the window;
4. compares each predicted `B_fast` and next state with the archived clean
   Teacher trajectory.

This is genuine free-running truncated backpropagation through time, not
one-step teacher forcing. It is nevertheless incomplete as a treatment of
distribution shift.

After the first day, the model consumes its own state `S_model[d]`, while the
archived fast-day label was produced by the Teacher at `S_teacher[d]`:

```text
current training pair after drift:
    input  = S_model[d]
    label  = B_teacher(S_teacher[d], forcing[d])

scientifically matched operator pair:
    input  = S_model[d]
    label  = B_teacher(S_model[d], forcing[d])
```

The first pair can be used as a trajectory-recovery objective, but it is not a
clean supervision pair for the Teacher transition operator. Combining its
fast-day label with a clean-trajectory next-state loss can impose conflicting
requirements once the model has drifted.

The present curriculum is also small: 64 updates at each of horizons 1, 3,
and 7 with batch size 4. Increasing those numbers without correcting the
training distribution is not an evidence-based next step.

## Established Strategies

### Full autoregressive unrolling

Backpropagate through multiple predictions and score the complete trajectory.
GraphCast and many learned simulators use increasing autoregressive horizons.
This directly optimizes rollout behavior, but memory and gradient difficulty
grow with horizon. It is useful for short horizons, not a practical route to
50-year backpropagation.

Current status: already implemented for 1, 3, and 7 days. The failed A/B shows
that changing scalar weights inside this framework is insufficient.

### Scheduled sampling

Mix Teacher and model states as recurrent inputs. This exposes the model to
some of its own predictions, but the target can remain mismatched to the
sampled input state. It is a useful baseline, not the preferred scientific
solution when the exact Teacher can be queried.

Current status: not implemented. Do not make it the next primary experiment.

### State-noise training

Perturb clean states with realistic rollout-like errors and train the model to
remain accurate or recover. Learned physics simulators use structured noise to
simulate accumulated errors. Independent Gaussian noise on all 3,854 state
values would be physically invalid here; perturbations must respect masks,
nonnegativity, pool structure, and observed error covariance.

Current status: not implemented. It becomes a fallback or complement after a
real model-induced-state experiment.

### Pushforward training

Run the model for a prefix using its own predictions, stop gradients through
that prefix, and train one or a few following transitions. This exposes the
network to model-induced states without retaining a long reverse-mode graph.
It supports much longer exposure horizons than full backpropagation at bounded
memory cost.

Current status: not implemented. This is directly applicable.

### DAgger-style on-policy Teacher queries

Roll out the current model, collect states it actually visits, query the exact
Teacher transition at those same states, and train on the resulting matched
pairs. This addresses covariate shift at its source instead of asking the
network to infer the Teacher action for one state from a label generated at a
different state.

Current status: not implemented. Because the full JAX Teacher exists, this is
the strongest project-specific option, subject to proving that canonical
model states can be safely re-entered into the Teacher fast-day transition.

### Long-run distribution and budget constraints

For chaotic or very long simulations, pointwise tracking alone may be the
wrong final objective. Methods such as DySLIM also constrain long-run
statistics. ORCHIDEE-MAN is seasonally and externally forced, so a stationary
invariant-measure loss is not directly transferable. Appropriate analogues
are seasonal/annual water and carbon budgets, state bounds, carbon-pool
distributions, and event-recovery statistics.

Current status: these belong after local operator learning is sound. They must
not be used to hide daily transition errors.

## Recommended Next Gate

Do not train another network yet. First run a bounded counterfactual Teacher
diagnostic on the existing frozen four-way windows:

1. Roll the accepted v1 network freely to obtain `S_model[d]`.
2. Re-enter selected Day 2-7 model states into the exact JAX Teacher fast-day
   transition under the same forcing and parameters.
3. Obtain `B_teacher(S_model[d])` and the corresponding source-backed next
   state.
4. Compare three quantities:
   - neural `B_fast` versus Teacher `B_fast` at the same model state;
   - Teacher-from-model-state next state versus the clean Teacher trajectory;
   - physical validity and all exact masks/discrete states.

This diagnostic distinguishes two cases:

- **operator-distribution failure**: the Teacher remains valid at model states
  but the network is wrong there. Implement stop-gradient pushforward with
  on-policy Teacher labels.
- **state-validity/stability failure**: small model errors move the system into
  invalid or strongly divergent states. Add explicit physical projection and
  constrained state perturbations before further rollout training.

The diagnostic is intentionally small. It uses existing train and validation
windows, does not touch the sealed test split, and does not generate a new
multi-point dataset.

## Candidate Training Protocol If The Gate Passes

Use a single bounded A/B against frozen v1:

- initialize from the same one-step checkpoint;
- mix clean Teacher pairs with model-induced on-policy pairs;
- use no-gradient prefixes sampled from 0, 1, 3, 7, and later 14 days;
- query the Teacher at the terminal model state for a matched one-step target;
- backpropagate through only the final one or few transitions;
- keep exact retained STOMATE, mask, and discrete-state handling;
- keep the sealed test split untouched;
- evaluate first on the same four seven-day windows, then 30 days only after
  the seen-condition gate passes.

Do not attempt 365-day or 50-year backpropagation. Long trajectories are
validation sequences and sources of no-gradient prefix states; gradients stay
short and bounded.

## References

- Ross, Gordon, and Bagnell (2011), DAgger:
  https://arxiv.org/abs/1011.0686
- Bengio et al. (2015), Scheduled Sampling:
  https://arxiv.org/abs/1506.03099
- Sanchez-Gonzalez et al. (2020), Learning to Simulate Complex Physics with
  Graph Networks: https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html
- Brandstetter, Worrall, and Welling (2022), Message Passing Neural PDE
  Solvers, including the pushforward trick:
  https://arxiv.org/abs/2202.03376
- Lam et al. (2023), GraphCast autoregressive multistep training:
  https://doi.org/10.1126/science.adi2336
- Lippe et al. (2023), PDE-Refiner and temporal rollout strategy analysis:
  https://arxiv.org/abs/2308.05732
- Schiff et al. (2024), DySLIM long-run dynamics:
  https://proceedings.mlr.press/v235/schiff24b.html
