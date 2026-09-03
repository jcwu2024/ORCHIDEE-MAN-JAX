# Direct-GPP Generalization Attribution Result

## Outcome

The free `gln01` audit completed at commit `52c60e7` in `7,117.794`
seconds. Its classification is
`direct_gpp_generalization_attribution_completed`, and every engineering and
evidence-completeness gate passes.

The server report is
`runtime/outputs/diagnostics/target-oriented-coupled-gpp-direct-generalization-52c60e7.json`,
SHA256
`23327c3c4081e283ea934acd03cb2db3c0c04b7ddbf261c430fb17b219e51cd9`.
The update-7 bootstrap took `2,975.771` seconds and did not repeat historical
scale searches.

This result does not show a scheduled-batch versus monitor mean-gradient
reversal. It shows that the realized four-source step protects direct GPP too
weakly, with additional artifact-level conflict and finite-step nonlinearity.

## First Regression

The first adjacent fixed-monitor regression occurs at persistent update 2.
The same realized step changes direct-GPP loss as follows:

| Panel | Before | After | Relative change | Improved artifacts |
| --- | ---: | ---: | ---: | ---: |
| scheduled update-2 group | `0.0538750` | `0.0507308` | `-5.8362%` | `5/8` |
| fixed monitor | `0.0629664` | `0.0643641` | `+2.2197%` | `4/8` |

The scheduled and monitor mean GPP gradients are strongly aligned, with
cosine `0.973434`. The applied four-source descent direction has cosine only
`0.154337` with the scheduled GPP mean and `0.024148` with the monitor GPP
mean. One scheduled artifact and three monitor artifacts have a negative
first-order cosine with that applied direction.

The positive monitor-mean cosine predicts an infinitesimal improvement, yet
the actual scale-`1/2` step worsens the monitor. That is direct evidence of a
finite-step curvature or nonlinearity problem at the realized displacement;
it is not evidence that the two panel means demand opposite parameter
directions.

Across all ten replayed updates, fixed-monitor direct-GPP loss changes from
`0.0666996` to `0.0851595`, a `27.6762%` regression. This reproduces the
direction of the paid persistent result while allowing ordinary independent
GPU replay variation.

## Integrity And Runtime

All 16 artifact gradients are finite and nonzero, all ten persistent updates
replay, same-run panel means agree exactly, and the first regression is found
at the preregistered boundary. The bounded-training parent recovery and all
fixed parameters remain unchanged. No optimizer state, parameter proposal,
recovery, or checkpoint is written.

Only accepted v6 train/train artifacts are used. The run does not regenerate
Teacher data, join v5 data, read fresh validation or sealed test values,
reconstruct 48 forcing records, or execute a 48-step state scan. Reported JAX
allocator peak is `4,496,653,312` bytes.

The final full 19-by-19 gradient Gram caused a separate expensive JIT
compilation. The next audit does not need that matrix and deliberately keeps
only the existing four-source direction computation.

## Decision

Do not change the Attention architecture or continue the retained recovery on
this evidence. First test whether a smaller realization of the already
computed update-2 direction can protect direct GPP simultaneously on four
disjoint future train/train groups. Select a scale using train-only evidence,
then expose exactly that selected scale once to the fixed monitor.

If the train-only scale transfers, the next admissible task is a short
multi-group GPP-protected backtracking smoke. If no shared train-only scale
exists, or the selected scale fails the monitor, direct-GPP constraints must
enter direction construction across multiple groups rather than remaining a
single scheduled-group mean constraint.
