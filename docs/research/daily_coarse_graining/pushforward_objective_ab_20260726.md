# On-policy Pushforward Objective A/B

Date: 2026-07-26

## Frozen Experiment

The candidate tests detached model-generated prefixes with exact same-state
Teacher labels. It changes neither the canonical model architecture nor the
accepted nine-point v5 dataset.

- Research commit: `ea6fe2003705d724bb1b1cc0379b35087b9d45bc`
- Initialization: the same accepted one-step checkpoint as the frozen
  `canonical_multistep_v1` baseline
- Curriculum: `0:48,1:48,3:48,7:48`
- Batch size: 4
- Total gradient updates: 192
- Learning rate: `1e-4`
- Seed: `20260725`
- Sealed test split: not used

Training job `14383581` completed with exit code zero in 1:07:56. Peak RSS was
8,646,192 KiB. The checkpoint SHA256 is
`db933ede1f47869c15149caf30ca98e191d6536d2f8ef0348d8b8316a9be87dc`.
The estimated charge was CNY 2.81.

All gradients remained finite, but nonzero-prefix stages contained very large
finite loss outliers. Mean stage losses were `0.0113`, `2.70e20`, `5.43e19`,
and `2.99e15` for prefixes 0, 1, 3, and 7 days. This is warning evidence that
some visited states produce extremely difficult matched transitions; it is not
hidden by the final-step loss.

## Frozen Four-way Evaluation

Evaluation job `14384100` completed with exit code zero in 4:43. All four
cases had zero defined-status and discrete-state mismatches. Estimated charge
was CNY 0.20.

Day-7 normalized RMSE:

| Case | Metric | Baseline | Pushforward | Change |
| --- | --- | ---: | ---: | ---: |
| train/train | global | 0.145471 | 0.092764 | -36.2% |
| train/train | biomass | 0.014259 | 0.014326 | +0.5% |
| train/train | litterpart | 3.651536 | 0.854632 | -76.6% |
| train/train | NPP | 0.435612 | 0.477241 | +9.6% |
| train/train | growth respiration | 0.436102 | 0.477777 | +9.6% |
| train/train | maintenance respiration | 0.125879 | 0.126340 | +0.4% |
| train/validation-year | global | 0.134233 | 0.098483 | -26.6% |
| train/validation-year | biomass | 0.010679 | 0.011807 | +10.6% |
| train/validation-year | litterpart | 3.574573 | 1.093806 | -69.4% |
| validation-point/train-year | global | 0.519886 | 0.550696 | +5.9% |
| validation-point/train-year | litterpart | 3.245493 | 5.287186 | +62.9% |
| joint validation | global | 0.429058 | 0.459871 | +7.2% |
| joint validation | litterpart | 3.004834 | 5.149699 | +71.4% |

The matrix report is under:

```text
runtime/outputs/experiments/canonical-v5-pushforward-ab-ea6fe20/
validation-four-way/matrix_report.json
```

## Decision

Reject this checkpoint as the promoted daily surrogate and retain
`canonical_multistep_v1` as the accepted baseline.

The method has a real positive effect on seen-condition recursive drift,
especially litter partitioning, but it fails the predeclared gate:

1. train/train biomass did not improve;
2. train/train NPP and growth respiration regressed materially;
3. validation-year biomass regressed materially; and
4. spatial global and litterpart errors worsened.

Do not continue this candidate with ad hoc learning-rate, mixture, loss-weight,
or update-count tuning. Do not evaluate the sealed test split. The evidence
should instead inform a separate reassessment of architecture, spatial
conditioning, and a bounded source-selected spatial-data design.
