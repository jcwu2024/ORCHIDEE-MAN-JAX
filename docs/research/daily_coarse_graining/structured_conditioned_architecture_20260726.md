# Structured Process-Conditioned Architecture Gate

Date: 2026-07-26

## Motivation

The accepted `canonical_multistep_v1` network has two separately established
limitations. Seven-day error grows recursively even on seen conditions, while
the larger global failure is spatial: validation-point Day-7 RMSE is
`0.519286` in a training year and `0.428353` in a validation year. Condition
interventions also showed that the flat network almost ignores landpoint
conditions: static permutation changed loss by only `2.96e-5`, and the four
paper parameters had no detectable effect.

The failed `process_increment_v2` and `on_policy_pushforward_v1` objectives
must not be tuned further. This gate changes representation capacity while
retaining the accepted `canonical_multistep_v1` objective and the promoted
nonnegative carbon-stock projection.

## Candidate

`structured_process_film_v1` retains the complete trained
`canonical_flat_v1` trunk and adds:

- eight source-derived state adapters for STOMATE carbon flux, vegetation
  structure, litter turnover, carbon storage, environment/phenology memory,
  hydrology, thermal energy, and surface exchange/finalize;
- separate encoders for the four paper-parameter vector, landpoint static
  conditions, and annual/calendar conditions;
- residual FiLM scale and shift paths after both fusion layers, so exogenous
  conditions remain available throughout the operator instead of only at one
  concatenation bottleneck.

All adapter outputs and FiLM heads start at exactly zero. Upgrading the
accepted checkpoint therefore gives exactly the same prediction before the
first update. The old flat condition encoder remains present during this
bounded experiment.

The v5 contract divides all 3,854 continuous states exactly once:

| Process group | Width |
|---|---:|
| STOMATE carbon flux | 114 |
| STOMATE vegetation structure | 56 |
| STOMATE litter turnover | 218 |
| STOMATE carbon storage | 1,875 |
| STOMATE environment/phenology memory | 123 |
| Hydrology | 622 |
| Thermal energy | 653 |
| Surface exchange/finalize | 193 |

The flat and structured models contain 1,963,369 and 2,459,449 trainable
parameters, respectively.

## Implementation Gates

The following local gates passed before GPU training:

- old checkpoints default safely to `canonical_flat_v1`;
- architecture identity and source-derived group hash are checkpoint-bound;
- a structured run is rejected unless an initialization checkpoint is given;
- synthetic JIT predictions are elementwise identical at initialization;
- real v5 3,854-state/2,855-target JIT predictions are elementwise identical;
- a real-width reverse pass has 62 finite gradient leaves, all eight state
  output adapters receive nonzero gradients, and all four FiLM heads receive
  nonzero gradients;
- 73 related unit tests, Ruff, `py_compile`, shell syntax, and diff checks pass;
- rollout, spatial-condition permutation, and counterfactual diagnostics all
  dispatch from checkpoint architecture rather than assuming the flat model.

## Controlled A/B

Do not compare a further-trained candidate only with an untouched checkpoint.
Run two continuation arms from the same accepted baseline checkpoint, with the
same seed, sampled batches, learning rate, and `1:64,3:64,7:64` update budget:

1. `canonical_flat_v1` continuation control;
2. `structured_process_film_v1` candidate.

Both use `canonical_multistep_v1`; neither uses the rejected alternative
objectives. Evaluate both on the frozen four-way Day 100-106 matrix and run the
same spatial-condition permutation diagnostic. The sealed test split remains
untouched.

The prepared Explore1000 entry point is
`scripts/hpc/slurm_structured_architecture_ab.sh`. It runs both arms
sequentially in one allocation, verifies the GPU runtime once, reuses the JAX
compilation cache, and writes separate training, four-way, and condition-use
assets for each arm. Its default request is one V100, four CPU cores, and a
three-hour hard limit; submission still requires an explicit resource/cost
approval.

Promotion requires all masks and discrete states to remain exact, no material
regression on seen-condition global or named carbon states, and a detectable
increase in parameter/static-condition use. Spatial RMSE improvement is the
preferred result. If condition use improves without spatial degradation but
the validation point remains outside the training envelope, proceed only to a
bounded source-selected spatial-data pilot. If condition use remains absent,
reject this architecture before generating more Teacher data.
