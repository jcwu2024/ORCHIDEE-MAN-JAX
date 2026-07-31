# Rollout-Stability Full Screening Result

Date: 2026-07-31

## Decision

Reject `mixed_horizon_stability_v1` and keep
`one_step_continuation_control` as the matched reference for the next
architecture/objective hypothesis. Do not run three confirmation seeds,
inspect the sealed test split, or begin 365-day validation from this
candidate.

This is a completed informative experiment, not an execution failure. Both
arms completed all 8,192 matched updates, the evaluator covered every
predeclared model-selection window, all hard and structural constraints
passed, and Slurm exited zero. The candidate is rejected because it failed
114 of 223 relative screening gates.

## Execution Evidence

- training job: `14425151`, `COMPLETED`, exit `0:0`;
- training commit:
  `d6e43cc79f590e494f79fe54adba4ac2c28779b6`;
- screening job: `14430377`, `COMPLETED`, exit `0:0`;
- screening commit:
  `a9d239096193fd87f17d95f04f0b945835f6cb8d`;
- node and resources: `ibc13b02n01`, one V100 and four CPUs;
- screening elapsed time: `04:21:26`;
- peak process RSS: `6,246,296 KiB`;
- report SHA256:
  `1387448b0b0831661daaaffc8985ef22602540e0c7e91fea7bfe96c085828bb0`;
- sealed test used: false.

At the stated additive rates, elapsed screening cost was approximately CNY
`1.22` CPU plus CNY `9.59` GPU, or CNY `10.81` total.

The evaluator completed all 4,754 point-year references. Per arm it covered:

| Slice | Horizon 1 windows | Horizon 7 windows | Horizon 30 windows |
|---|---:|---:|---:|
| temporal | 585,825 | 576,195 | 539,280 |
| spatial | 1,075,953 | 1,058,265 | 990,461 |
| joint | 73,365 | 72,159 | 67,536 |

## Global Rollout Result

Values are candidate/control error ratios; lower is better.

| Slice | One-step fast | One-step state | 7-day free | 30-day free |
|---|---:|---:|---:|---:|
| temporal | 1.0055 | 1.0014 | 1.0032 | 0.9181 |
| spatial | 1.0027 | 1.0011 | 0.9980 | 0.7539 |
| joint | 1.0024 | 1.0017 | 0.9993 | 0.8047 |

All six one-step global gates pass their no-regression threshold. All three
30-day global state gates pass and improve by about 8.2%, 24.6%, and 19.5%.
However, none of the three 7-day gates reaches the required 5% improvement.
The candidate is therefore not a robust short-to-medium-horizon improvement.

## Failed Process Gates

The 114 failures are:

| Gate family | Failed | Total |
|---|---:|---:|
| one-step global | 0 | 6 |
| one-step process family | 8 | 48 |
| free-rollout global | 3 | 6 |
| tendency bias | 46 | 54 |
| named science fields | 57 | 108 |

The repeated one-step regressions are concentrated in `daily_interface`,
`stomate_carbon_flux`, and, on spatial/joint validation,
`stomate_litter_turnover`.

Global tendency-bias ratios are `2.108/1.826` for temporal,
`2.125/1.683` for spatial, and `3.151/2.560` for joint at horizons 7/30.
Thus the candidate obtains lower 30-day aggregate state RMSE while introducing
substantially worse signed daily drift. This is precisely the failure mode the
tendency-bias gates were designed to detect.

Named science fields fail as follows:

| Field | Failed gates | Total | Candidate/control ratio range |
|---|---:|---:|---:|
| NPP | 9 | 9 | 1.252-1.834 |
| growth respiration | 9 | 9 | 1.340-1.912 |
| maintenance respiration | 9 | 9 | 2.117-4.461 |
| biomass | 8 | 9 | 1.045-1.804 |
| `carbon_32l` | 8 | 9 | 0.577-2.687 |
| `deepC_peat` | 7 | 9 | 0.440-1.587 |
| LAI | 7 | 9 | 0.809-1.523 |

GPP, DOC, height, litter, and heterotrophic respiration pass all nine named
field gates. Their success does not offset the systematic carbon-allocation
and autotrophic-respiration failures.

## Integrity Result

All exact hard constraints pass:

- no unexpected defined-status mismatch;
- no discrete-state mismatch;
- no nonfinite defined state or fast-boundary value;
- no negative source-constrained carbon stock.

All structural constraints pass for both arms:

- 30-day prediction is bit-exact across a `15+15` restart split;
- no hidden cross-day neural memory exists;
- the retained daily tail remains source-backed.

The declared dynamic `rveget` status error rate improves slightly:
candidate `0.0097743`, control `0.0098930`, ratio `0.9880`. This gate passes.

The report field `promotion_eligible=true` means complete evidence was
available for classification. It does not mean the candidate passed.
The authoritative classification is:

```text
status: rejected
decision: stop_and_attribute_declared_screening_failure
```

## Scientific Attribution

The experiment rules out the current mixed-horizon objective as a promotion
path for this checkpoint and architecture. It can reduce aggregate 30-day
state distance, especially on spatial holdouts, but it does so by trading away
signed process balance and key carbon variables. More epochs or confirmation
seeds would test the rejected configuration again and are not justified.

The next research step must be a new, predeclared architecture or objective
hypothesis that directly addresses carbon flux/allocation and unbiased daily
tendencies. It must start from the accepted parent evidence and use the same
sealed-test policy. Do not tune thresholds or weights against this validation
report, and do not treat 365-day rollout as a debugging instrument.

Server evidence root:

```text
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/screening/canonical-669-rollout-stability-screening-a9d2390
```
