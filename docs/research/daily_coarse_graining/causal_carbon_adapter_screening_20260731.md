# Experiment C All-Sample Screening Result

Date: 2026-07-31

## Decision

Experiment C is rejected. Its frozen decision is:

```text
status: rejected
decision: stop_and_attribute_declared_screening_failure
```

Do not run confirmation seeds, 365-day validation, complete validation chains,
or the sealed test split for this candidate. Do not revise its thresholds or
retune its loss weights against this validation result.

## Execution Evidence

Formal screening ran as Slurm job `14436716` from evaluation commit `7091e1d`
against training commit `8d24fd0`. It completed all 4,754 model-selection
references in `04:25:50` with exit code zero:

- temporal: 1,605 references;
- spatial: 2,948 references;
- joint: 201 references.

The evaluated cells were Day 1 teacher-forced and Day 7/30 free rollout. The
sealed test split remained unread.

Evidence hashes:

```text
matched screening report:
2b636981a5aeb73479c354558c09ed99cf398559c7f315bd56774f7f9822b5dd
control arm screening report:
c072dcf9069405936020f77585c6a62356f8ab99919a55c47452f00083efe4c9
candidate arm screening report:
860d0d7f015c0231a5413e8bcdeeaff29eb1d08f9c3359a7aa3379b80ba2bb75
predeclared attribution summary:
22cfbd0d0c641b4eb13d721cbac9feebf2858c5fcb57370e4f815f4a7bc86676
```

## Gate Results

All execution-integrity gates passed:

- hard constraints: `108/108`;
- structural constraints: `6/6`;
- both restart-split probes passed;
- no unexpected status, discrete, nonfinite, or negative-stock failure.

Scientific relative gates failed `73/165`:

| Group | Failed | Total |
| --- | ---: | ---: |
| Causal interface | 11 | 12 |
| Global terminal state | 0 | 9 |
| Primary carbon state | 46 | 63 |
| Flux bias | 3 | 27 |
| Stock-tendency bias | 13 | 36 |
| Litter/DOC guards | 0 | 18 |

Failures were broad rather than confined to one split or horizon:

| Dimension | Failed | Total |
| --- | ---: | ---: |
| Temporal | 22 | 55 |
| Spatial | 25 | 55 |
| Joint | 26 | 55 |
| Day 1 | 28 | 63 |
| Day 7 | 21 | 51 |
| Day 30 | 24 | 51 |

## Scientific Interpretation

The adapter improved many signed flux and tendency biases. Most NPP,
respiration, biomass-tendency, and LAI-tendency gates passed, and all
litter/DOC guards passed. These improvements did not compensate for direct
damage to the learned stock boundary:

- all three `carbon_32l` interface gates worsened by `7.33-9.32x`;
- all three `deepC_peat` interface gates worsened by `5.66-7.45x`;
- all nine `carbon_32l` primary-state gates failed;
- all nine `deepC_peat` primary-state gates failed;
- GPP interface error increased by about `2.5-3.2%` in every split.

All nine global terminal-state gates passed with ratios near one. This is not
evidence of material global improvement: the damaged carbon-stock fields are
small within the 3,854-value aggregate and are exposed only by the required
fieldwise gates.

The result therefore rejects the hypothesis that an unconstrained residual
adapter can safely correct both carbon fluxes and carbon stocks on top of this
already accurate parent. The failure is scientific architecture/objective
quality, not execution, data integrity, restart behavior, or numerical
stability.

## Next Architecture Decision

Do not create another adapter by changing weights or widths. Any later
candidate must be an architecture-level proposal in which learned fluxes or
tendencies drive a budget-conserving stock update, instead of independently
correcting `carbon_32l` and `deepC_peat`. That proposal requires a new frozen
contract and bounded architecture comparison before another full training
screen.
