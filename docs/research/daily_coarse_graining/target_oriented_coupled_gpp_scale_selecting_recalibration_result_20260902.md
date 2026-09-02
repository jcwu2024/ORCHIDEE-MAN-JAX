# Coupled-GPP Scale-Selecting Recalibration Result

Date: 2026-09-02

Status: accepted; non-test 7/30-day transfer audit is next

Executable commit: `aaeea61`

Result SHA256: `c8ae64e5767c1ead579dbb84def10fdab992857abb3a8d676f075ba17c42a0b8`

## Result

The free `gln01` execution completed in `3,402.216` seconds. All three
reconstructed scale-selected direct recalibrations passed their four-source
finite-step and physical gates:

| Update | Tried scales | Selected | GPP | State | 7-day | 30-day | Weighted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | inherited audited grid | 1/4 | -0.3068% | -10.4393% | -2.5015% | -2.6018% | -2.9544% |
| 6 | 1 | 1 | -3.1720% | -4.3370% | -3.0809% | -4.5798% | -3.5294% |
| 7 | 1, 1/2 | 1/2 | -3.5180% | -4.4065% | -3.8337% | -3.2788% | -3.6991% |

Negative percentages are improvements. Update 6 therefore accepts the full
bounded displacement, while update 7 rejects that full displacement and
accepts the half scale. The minimum source cosines for the newly recomputed
updates are `0.143035` and `0.278430`; both exceed the frozen common-direction
threshold.

All proposed values are finite and all physical gates pass. Fixed parameters
remain exactly unchanged. No optimizer state, parameter proposal, recovery,
or checkpoint is retained. No fresh validation, historical v5, sealed test,
Teacher regeneration, 48-record forcing reconstruction, or 48-step state scan
is used. Peak JAX bytes in use are `4,114,680,576`.

## Interpretation And Next Gate

This is positive local optimization evidence: three consecutive batches admit
a common four-objective direction and a real finite step. It is not yet
generalization or persistent-training evidence because every selected scale
was evaluated on the same artifacts that formed its direction.

The next gate reconstructs the selected state in memory and compares it with
the state before update 5 on the frozen eight-artifact non-test monitor. The
64 seven-day and 32 thirty-day windows were not used for gradients or scale
selection. Both horizon mean losses must improve and both physical gates must
pass. Per-artifact unanimity is reported but is not required. Paid training,
fresh validation, 365-day rollout, Gate D2, sealed test, and product acceptance
remain closed.
