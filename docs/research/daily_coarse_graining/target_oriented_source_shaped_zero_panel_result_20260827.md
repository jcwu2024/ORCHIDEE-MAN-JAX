# Target-Oriented Source-Shaped Zero-Panel Result

Date: 2026-08-27

Status: accepted zero-training train/train result; the physical mechanism has
a large maintenance signal but is not yet protected

Report SHA256:
`6e7302fe4131d408ca469a62e3c31d9259fe9efa4a2e713e4242bc549e094291`

Screen contract SHA256:
`a06beb815834142f44939022eea6ebd044e0a096b5d3c443754a9d87fb0f8cee`

Executable commit: `250986f190711e97e22bb8a75d0582418e57bcb0`

## Result

The zero-correction source formula produces a large, consistent maintenance
improvement before any training:

| Horizon | Median maintenance-score reduction | Frozen minimum | Pass |
| --- | ---: | ---: | --- |
| 7 days | `73.6076%` | `20%` | yes |
| 30 days | `87.3457%` | `20%` | yes |

This is the first large improvement produced by the new output mechanism. It
is not a small ranking change around the weak parent.

The complete relative source gate does not yet pass because the zero
correction changes short-horizon carbon allocation too strongly:

| Protected score | 7-day median change | 30-day median change |
| --- | ---: | ---: |
| GPP | `+0.5403%` | approximately `0%` |
| Growth respiration | `+69.9249%` | `-8.6031%` |
| Autotrophic respiration | `-24.2984%` | `-63.8734%` |
| NPP | `+27.9226%` | `-20.7417%` |
| AGB | `+18.3846%` | `-22.4605%` |
| BGB | `+19.0799%` | `-4.9776%` |
| Total live biomass | `+29.1662%` | `-19.1439%` |
| Ecosystem carbon | `-2.8360%` | `-0.9523%` |

The largest any-group protected regression is `340.382%`, in 7-day growth
respiration. The frozen limits are nonpositive median change and at most 1%
for any group, so the formal classification remains
`source_mechanism_not_yet_supported`. Absolute target-oriented acceptance also
does not pass.

All eight groups and both horizons are finite. Prediction/state nonfinite,
unexpected mask mismatch, discrete mismatch, budget failure, and negative
stock hard counts are all zero. Declared-dynamic `rveget` mask changes remain
diagnostic and are not hard failures.

## Interpretation

The result supports continuing the source-shaped mechanism into bounded
learning. The source formula removes most maintenance error, but its fixed
zero correction is too aggressive for some groups and propagates into NPP,
growth respiration, and biomass over seven days. The two state-conditioned
bounded branches were introduced specifically to learn this missing
group/day-dependent correction.

The next experiment therefore trains only the 394,242 new branch parameters.
Freezing all retained parent parameters prevents direct drift in the GPP,
state, and auxiliary heads; those quantities may still change through the
physically intended maintenance-to-state recurrence. The unchanged full
objective supplies the protection signal.

## Execution

The run used one free `gln01` V100, no Slurm, and no charge. It evaluated all
eight frozen train/train groups, 64 seven-day windows and 32 thirty-day
windows, in `1530.89 s` (`25.51 min`). Peak live JAX memory was 2,138,918,656
bytes, about 1.99 GiB. The persistent allocator reserved about 25 GiB, but
those bytes were not live tensors.

Teacher v6 was reused unchanged. Historical v5, validation values, and sealed
test were not read. No optimizer was created, no update was applied, and no
checkpoint was written.

## Next Gate

After explicit user authorization, implement and run one free-`gln01`
branch-only optimizer admission. It will calibrate the branch-gradient clip on
the eight frozen train/train calibration batches, initialize a fresh AdamW
state, apply exactly one update using artifact 252, verify that only the two
new branches changed, and discard the proposal after reporting finite loss,
gradient, step, state, and budget evidence.

Paid 100-update training, validation, 365-day rollout, Gate D2, sealed test,
and product acceptance remain closed.
