# Successor Maintenance-Output Projection Protocol

Date: 2026-08-16  
Status: frozen before execution  
Contract: `successor_maintenance_output_projection_v1.json`  
Contract SHA256: `9f16f83f8f7dac72beeef9b14a510edb722af3fc81b291b50e0e90403367c89c`

## Question

The stopped-state norm-`0.0005` proposal has a valid all-40 first-order
common-descent direction but fails after the finite parameter step. The
output-locality audit attributes most of that mismatch to the decoded
`daily.fast_maintenance_respiration` field. This audit asks whether one
bounded, maintenance-local correction can recover the finite-step loss gate
without changing the Attention architecture or suppressing GPP.

## Matched Arms

All arms use the same update-1000 Attention parent, the same three accepted
direct updates used to reconstruct the stopped state, and the same ten
train/train groups covering 40 point-years and 1,200 daily transitions.

- Safe reference: the immediately preceding accepted norm-`0.0005`
  microstep.
- Raw control: the unmodified stopped-state norm-`0.0005` proposal that
  previously failed the finite-step gate.
- Candidates: start from the raw control and subtract one unit gradient of
  the 30-day normalized maintenance-output displacement at correction norms
  `0.000125`, `0.00025`, and `0.0005`.

The correction objective is the scalar mean of squared normalized
maintenance-output changes over all ten groups. One aggregate gradient is
computed at the raw control and shared by the three candidates. This is a
parameter-metric experiment, not a network architecture change.

## Output Trust Region

The exact RMS and maximum absolute normalized maintenance changes of the safe
reference define the two caps. Each cap is multiplied by `1.25`. A candidate
must satisfy both. The cap is not applied to `daily.fast_gpp`, because the
output-locality audit shows that its finite contribution opposes and partly
cancels the maintenance regression.

## Acceptance

The smallest correction norm is selected only if all conditions hold:

- all 40 group-component losses are nonincreasing within relative tolerance
  `1e-6`;
- median NPP, aboveground-biomass-tendency, and
  belowground-biomass-tendency relative changes are each at most `-1e-5`;
- both exact maintenance-output caps pass;
- at least `25%` of the raw proposal's total parameter displacement remains;
- every state, budget, and nonfinite count is zero.

The safe reference must reproduce its accepted all-40 result. The raw control
must reproduce the finite-step failure and exceed both maintenance caps. The
penalty RMS computed by the differentiable objective must agree with the
direct output statistic within `1e-10`.

## Stop Rules

No optimizer update, parameter set, recovery state, or checkpoint is written.
Validation and sealed-test values are not read. If a candidate passes, the
next task is to freeze a selected-projection screen on the existing non-test
7/30-day subset. If none passes, this one-gradient projection is closed and
the next design must use a genuinely nonlinear constrained solver rather than
additional scalar step subdivision.

Execution is limited to one free `gln01` diagnostic. Paid Slurm work is not
authorized by this protocol.
