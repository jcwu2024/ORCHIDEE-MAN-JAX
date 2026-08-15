# Successor Maintenance-Output Projection Protocol

Date: 2026-08-16  
Status: completed; single-gradient maintenance projection rejected
Contract: `successor_maintenance_output_projection_v1.json`  
Contract SHA256: `84b96d4ffbc86fb8e34303caff953d5a56b7651190294179f15081f1c5e35b72`

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
maintenance-output changes over all ten groups. The exact same mean is
evaluated as 40 one-point-year reverse microbatches and accumulated with equal
weight; every point-year has the same 30-day and output-scalar count. One
aggregate gradient is computed at the raw control and shared by the three
candidates. This is a parameter-metric experiment, not a network architecture
change.

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

## Resource Amendment

The first live implementation vectorized all four point-years in each reverse
pass. It exited before any candidate or report after XLA attempted to allocate
`39,772,984,576` bytes on a 32 GB V100. No parameters, checkpoint, recovery,
or scientific output were written. The implementation now compiles one
zero-displacement, one-point-year reverse pass before the expensive stopped-
state reconstruction. Passing that memory preflight authorizes the 40 exact
microbatch gradients above; it does not change the objective, sample set,
candidate norms, caps, or gates.

## Result

The free-`gln01` execution completed at commit `629238a`. The formal report is
`outputs/successor-maintenance-output-projection-629238a.json`, with SHA256
`64c689388d951c95cbc1339bb5e389e4ff2bae5d20385f655635c1a9a2ce98c5`.
All evidence checks pass, with zero hard failures and zero nonfinite values.
The one-point-year gradient preflight used a reported JAX peak of
`13,869,214,720` bytes; the full audit completed in `3,219.59` seconds.

The safe reference reproduces all 40 nonincreasing losses. Its exact
maintenance RMS and maximum changes are `0.0004166623` and `0.00402458`, so
the frozen caps are `0.0005208279` and `0.00503073`. The raw control reproduces
the failure: maintenance RMS `0.08206115`, maximum `4.71565188`, and worst
group-component loss regression `28.7222%`.

| Correction norm | Maintenance RMS | Maintenance maximum | Effect and loss result |
| --- | ---: | ---: | --- |
| `0.000125` | `0.0390105` | `3.29103` | NPP, AGB, and BGB medians worsen; worst loss regression `12.6934%` |
| `0.00025` | `0.00363294` | `0.0771433` | NPP `-0.1628%`, AGB `-0.2292%`, BGB `-0.0327%`; anchor `+0.0177%`; worst loss regression `0.2733%` |
| `0.0005` | `0.00680043` | `0.0754981` | Overshoots the maintenance minimum; anchor `+0.0400%`; worst loss regression `0.2603%` |

No candidate passes. The `0.00025` correction removes about `95.57%` of the
raw maintenance RMS, but remains `6.98x` above the RMS cap and `15.33x` above
the maximum cap while violating the anchor and some of the 40 loss
constraints. Thus the maintenance gradient is informative, but one fixed
parameter-space direction cannot jointly restore the frozen constraints.

The single-gradient projection is closed. Do not subdivide the same
correction norm. The next task is to freeze a bounded nonlinear constrained
solver that relinearizes violated maintenance, anchor, and all-40 loss
constraints in a low-dimensional parameter subspace and verifies every
accepted correction with exact finite evaluation. This is still an optimizer-
geometry investigation, not an Attention architecture change or a training
authorization.
