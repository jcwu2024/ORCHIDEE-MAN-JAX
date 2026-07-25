# Counterfactual Teacher Re-entry Diagnostic

Date: 2026-07-25

## Question

The accepted daily neural surrogate has small Teacher-forced error but growing
free-rollout drift. The bounded diagnostic asks whether that drift is mainly:

1. neural operator error at states visited by the model; or
2. strong Teacher sensitivity to the model's already shifted state.

It runs the exact JAX Teacher from model-generated Day 2, 4, and 7 states at
train/train landpoint `281.0-095.0`, year 2004, days 100-106. No sealed test
data is used.

## Evidence

- Research commit: `87e1132ce670797a18731a5f478fda9ddee95118`
- Slurm job: `14380170`
- State: `COMPLETED`, exit `0:0`
- Elapsed: 12:08
- Resources: 4 CPUs and one V100
- Peak RSS: 5,901,420 KiB
- Server asset:
  `runtime/outputs/diagnostics/counterfactual_teacher_281_2004_d100_87e1132.json`

The first exact Teacher query took 500.80 seconds including compilation. The
next two took 2.29 and 2.15 seconds.

| Relative day | Neural vs Teacher at same state | Teacher state sensitivity |
| --- | ---: | ---: |
| 2 | 0.081294 | 0.031604 |
| 4 | 0.090372 | 0.065840 |
| 7 | 0.084502 | 0.123036 |
| Mean | 0.085390 | 0.073494 |

The mean operator-to-state-sensitivity ratio is `1.161865`. All Teacher
re-entry calls completed, with zero defined-status and discrete-state
mismatches.

## Decision

The model states remain valid Teacher inputs. State sensitivity becomes
important by Day 7, but matched operator error is larger on average. The next
bounded candidate is therefore stop-gradient pushforward with on-policy
Teacher labels:

- expose the current model to detached prefixes of 0, 1, 3, and 7 days;
- query the exact Teacher at the terminal model state;
- train only the final transition against that same-state Teacher target;
- keep clean samples in the mixture;
- do not extend reverse-mode differentiation to annual or 50-year horizons.

This result does not by itself accept a new neural checkpoint. A real one-
update smoke must first prove the training path, followed by one equal-budget
A/B against the frozen `canonical_multistep_v1` baseline.
