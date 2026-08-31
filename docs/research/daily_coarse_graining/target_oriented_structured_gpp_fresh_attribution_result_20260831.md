# Structured-GPP Fresh Error Attribution Result

Date: 2026-08-31

Status: accepted attribution; broad step-1000 checkpoint remains rejected

Attribution contract SHA256:
`2c93dbe5d31791474c9c02bbf08efd76f831c1f7e99bb5c1458b4cd8f880112a`

Fresh screen report SHA256:
`a9b9f6dcaf9aca349e738f096609e683342286d14d870d2c140afe2a3253798b`

Raw numerical attribution report SHA256:
`07031dbf8b61553dadc631214d1a2eba1a3a329487d10517580a897fee615d4e`

Finalized attribution report SHA256:
`fafd511112e7666335fc6667b179d2fa19c2994a38d6f2e4ffec6455f4f8037e`

Numerical execution HEAD:
`5dad8999aa2444bb6eac4bc71fee1993822015de`

Finalizer HEAD:
`355535b3c5e01d36c891b631bdaf06a60d13a2aa`

## Execution And Correction

The formal free-`gln01` numerical execution completed all `96/96` frozen
windows in about 619 seconds. Ordinary free rollout and Teacher-GPP bridge
both had zero physical hard failures. The numerical report contains all
required sufficient statistics and exact decomposition values.

Its first summary was not the formal scientific result. A carbon-summary loop
used `mode` as its local variable and overwrote the outer `formal` mode. The
raw report consequently emitted `not_run_in_admission`, a null reproduction
difference, and an incorrectly permissive integrity summary. Commit `355535b`
renames the local variable and adds a one-use CPU finalizer bound to the raw
report hash above.

The finalizer read only the two existing JSON reports. It did not load Teacher
data, execute a model forward pass, use a GPU, change a parameter, or write a
checkpoint. It reproduced the earlier fresh-screen metrics to a maximum
absolute difference of `6.94e-18`, then evaluated the frozen classification
and integrity gates.

## Integrity

| Check | Result | Gate |
| --- | ---: | ---: |
| Ordinary free report maximum metric difference | `6.94e-18` | `<= 1e-12` |
| GPP residual identity maximum absolute error | `6.66e-16` | `<= 1e-12` |
| GPP squared-decomposition relative error | `2.18e-16` | `<= 1e-12` |
| Carbon residual identity maximum absolute error | `0` | `<= 1e-10` |
| Ordinary free hard failures | `0` | `0` |
| Teacher-GPP bridge hard failures | `0` | `0` |
| Finite and nonempty reported statistics | passed | required |

The finalized integrity gate passes.

## Six-Cell Attribution

`Direct GPP` is cumulative GPP predicted from the Teacher day-start state.
`Feedback RMS` is the additional free-rollout residual from using the model's
carried state. `Feedback / total` can exceed one when local and feedback errors
partly cancel. Bridge columns replace only GPP with the Teacher value before
the conservative update and retained tail.

| Cell | Direct GPP RMSE / bias | Direct pass | Feedback RMS / total ratio | Growth RMSE free -> bridge | NPP RMSE free -> bridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| temporal 7 | `0.45630 / -0.17348` | no | `0.62831 / 0.68150` | `0.55095 -> 0.01854` | `1.15452 -> 0.05037` |
| temporal 30 | `0.20025 / -0.09439` | no | `0.67843 / 0.92417` | `0.46018 -> 0.03449` | `0.90017 -> 0.07946` |
| spatial 7 | `0.16423 / -0.06297` | yes | `0.91649 / 1.00405` | `0.55036 -> 0.01469` | `1.04308 -> 0.06913` |
| spatial 30 | `0.43005 / -0.14728` | no | `0.72850 / 0.80859` | `0.53743 -> 0.03461` | `1.04556 -> 0.08499` |
| joint 7 | `0.57395 / -0.12436` | no | `0.72129 / 0.71530` | `0.52961 -> 0.02352` | `1.32971 -> 0.07386` |
| joint 30 | `0.56508 / -0.18827` | no | `0.53537 / 0.55661` | `0.49035 -> 0.03241` | `1.21856 -> 0.06238` |

Teacher-state cumulative GPP fails `5/6` cells, so the day-local GPP mapping
is independently insufficient. State feedback is material in `6/6` cells, so
free-rollout failure is not explained by the local head alone.

The Teacher-GPP bridge makes growth respiration and NPP pass all six cells.
Their median frozen acceptance-score reductions are `99.6822%` and
`99.6525%`. The formal classification is:

`direct_and_feedback_gpp_with_sufficient_downstream_cause`

This means exact GPP is sufficient to explain the observed growth-respiration
and NPP failures on this fresh panel. It does not make the bridge a surrogate,
does not prove 365-day behavior, and does not authorize validation-target
tuning or sealed-test access.

## Decision And Next Gate

The broad step-1000 checkpoint remains rejected. The attribution does not
reject coordinate-aligned Attention as a mechanism. It rejects treating the
current fixed-parent structured-GPP specialist, or more unchanged training of
that specialist, as an adequate path to promotion.

The next gate is a design discussion before any architecture or loss edit. A
successor must address both diagnosed terms:

1. improve train-only generalization of the day-local GPP mapping; and
2. constrain the cross-day state channels that feed later GPP, rather than
   optimizing only the direct GPP output.

The design must name the affected output/state scope, train-only objective,
matched diagnostic, bounded admission, and stop rule before implementation or
another training job. The fresh 96-window panel remains evaluation-only.
365-day rollout, Gate D2, sealed test, and product acceptance remain closed.
