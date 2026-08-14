# Successor Loss-Conflict Attribution

Date: 2026-08-14

## Scope

This is a report-only attribution from the accepted update-1000 validation
report and the completed 100-update equal-sample batch-diversity report. It
does not train, change coefficients, select against the validation panel, or
change either architecture.

Source reports:

- update-1000 staged report SHA256: `e379e3e7...bd68ce1`;
- batch-diversity short-fit report SHA256:
  `6c3f16ffb8b160bd85b1840955f50d0f422685d323b200ce309fcd89d161154b`.

For each primary output, relative normalized-RMSE change is calculated from
the update-1000 baseline over the six fixed cells formed by temporal,
spatial, and joint slices at 7 and 30 days. Negative values are improvements.

## Cross-Cell Result

| Arm and sampling | GPP | NPP | Maintenance respiration | AGB | BGB | Total live biomass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control single | -1.44% | +2.39% | -12.94% | +3.16% | -1.06% | +2.38% |
| control mixed | -0.00% | -0.09% | -0.51% | -0.04% | -0.39% | -0.09% |
| attention single | -0.10% | +14.53% | -30.32% | +4.48% | +54.90% | +14.50% |
| attention mixed | -0.20% | -2.68% | +12.32% | +0.04% | -10.84% | -2.66% |

The signs are not driven by one outlier validation cell:

- control-single improves maintenance respiration in `6/6` cells but worsens
  NPP, AGB, and total live biomass in `6/6`;
- attention-single improves maintenance respiration in `6/6` but worsens
  NPP, AGB, BGB, and total live biomass in `6/6`;
- attention-mixed improves NPP, BGB, and total live biomass in `6/6`, while
  maintenance respiration improves in only `1/6`.

This is a systematic objective tradeoff. Sampling determines which side of
the tradeoff dominates, and attention is much more sensitive than control.
It is not evidence that four-point batches are universally superior, because
the preregistered control comparison failed. It is also not evidence that a
new architecture is required.

## Objective Mechanism

The frozen long-horizon loss first gives equal macro weight to each of five
flux fields and seven stock fields inside four aggregate components:

```text
cumulative flux MSE
cumulative flux bias squared
terminal stock MSE
terminal stock bias squared
```

Those four aggregates are then weighted against the one-day named-process
anchor by the frozen gradient-ratio coefficients. NPP and maintenance
respiration therefore share the same flux aggregates even when their
parameter-gradient directions oppose. The validation outcome proves a
tradeoff but does not by itself prove which per-output gradients conflict or
whether normalization, the one-day anchor, or a few point-years dominate it.

Gradient clipping is not the cause by itself. It remains frequent in all four
paths: `42/100` attention-single, `47/100` attention-mixed, `62/100`
control-single, and `60/100` control-mixed, with isolated very large spikes.
Clipping limits magnitude but cannot reconcile opposing directions.

## Decision

The next controlled experiment category is **loss balancing**, not another
sampling experiment and not another architecture.

Before changing any coefficient, run one zero-update train/train-only
per-output gradient audit at the accepted update-1000 checkpoint. It must:

- use the existing prepared v6 artifacts and no validation or sealed-test
  values for tuning;
- measure gradient norms and pairwise cosines for GPP, NPP, maintenance
  respiration, AGB, BGB, and total live biomass, separately from the one-day
  anchor;
- report how the frozen normalization and coefficients scale each gradient;
- include both attention and control on the same point-years;
- perform no optimizer update and write no checkpoint.

Only if that audit confirms stable train/train gradient conflict may a
matched loss-balancing mechanism and bounded short-fit gate be proposed.
Possible mechanisms must be discussed before implementation; this result does
not authorize validation-tuned weights, a paid job, another architecture, or
continued staged training.

