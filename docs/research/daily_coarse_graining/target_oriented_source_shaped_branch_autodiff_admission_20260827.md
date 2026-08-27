# Target-Oriented Source-Shaped Branch-Autodiff Admission

Date: 2026-08-27

Status: accepted free-GPU engineering admission; paid 100-update training
requires separate resource and cost approval

Report SHA256:
`5852d996f2baa8afa5e68d0446d30d09bff64b5ac7996836f448a27535e58538`

Executable commit:
`d965bea1ccd9ddd080bd56476ff9999ffa1953b8`

Parent one-update report SHA256:
`c2743cbc3cd563b682176531e7ec2310c292f5fd1f63bbf625c846f79434ea59`

## Result

The direct branch-subtree autodiff path passes its one-batch admission on
prepared train/train artifact 252. The forward path still uses all 84,148,646
candidate parameters, but reverse-mode differentiation is restricted to the
394,242 parameters in the aboveground and belowground source-shaped branches.

The direct gradient and the same branch subtree extracted from the complete
84.1-million-parameter gradient have identical tree structure and zero failing
coordinates under `atol=1e-8`, `rtol=1e-6`. Across all 394,242 coordinates:

| Comparison | Result |
| --- | ---: |
| Maximum absolute gradient difference | `9.3132e-10` |
| Gradient-difference L2 norm | `1.0461e-9` |
| Relative L2 difference | `4.5611e-9` |
| Failing coordinates | `0` |

The 7-day and 30-day source gradients are elementwise identical. The
one-day/process source has maximum absolute difference `5.8208e-11` and zero
failing coordinates. The direct cold-to-hot replay is elementwise identical.

The complete objective and all weighted and unweighted components are equal
between paths. Fresh AdamW receives the same p90 clip threshold
`0.1973866669`, raw norm `0.2293467275`, clip scale `0.8606474996`, learning
rate `3e-5`, and step norm `0.0006801606`. Direct and extracted paths have zero
scalar proposal difference; their proposed branch parameters and optimizer
states have zero failing coordinates. The post-proposal objective matches the
accepted one-update result to maximum absolute difference `1.80e-14`.

Both branches have finite nonzero gradients. Prediction, state, unexpected
mask, discrete-state, budget, and negative-stock hard failures are zero before
and after the discarded proposal. The declared-dynamic `rveget` mask count is
unchanged from the accepted diagnostic policy and is not a frozen hard
failure.

## Resource Evidence

The run used one free `gln01` V100 and no Slurm allocation. Total wall time was
`4092.00 s` (`68.20 min`), dominated by the one-time compilation of the three
new direct reverse graphs. Direct cold time was `3269.58 s` (`54.49 min`).

After compilation, the exact same effective batch took `25.18 s`, including
`7.29 s` forward and `17.68 s` gradient work. Gradient-source times were
`0.94 s`, `5.48 s`, and `11.25 s` for one-day/process, 7-day, and 30-day
sources. The admitted extracted-full-gradient path had a hot median of
`36.85 s`, so direct branch autodiff is `1.46x` faster for the complete hot
effective batch.

JAX peak live device memory is 2,406,170,368 bytes, approximately 2.24 GiB,
versus 8,204,654,080 bytes, approximately 7.64 GiB, for the admitted extracted
path. This is a `70.67%` reduction. The approximately 25 GiB allocator pool
visible in `nvidia-smi` is not the live tensor footprint.

## Data And Policy

Only existing v6 train/train prepared assets and the exact update-1000 parent
were read. Teacher data was not regenerated, historical v5 was not used, and
validation and sealed-test values were not read. No training update was
executed or persisted. One optimizer proposal was evaluated and discarded;
no checkpoint or recovery was written.

The candidate path reconstructs no 48-record forcing sequence and performs no
48-step state scan.

## Interpretation And Next Gate

This closes the direct-autodiff engineering prerequisite. The cheaper path
computes the same branch update as the previously accepted complete-gradient
extraction while materially reducing hot time and live GPU memory. It does not
show that 100 training updates improve the frozen 7/30-day science screen.

The next gate is to prepare and separately authorize one paid V100 run over
prepared train/train artifacts `252..351`, updating only the 394,242 branch
parameters from fresh AdamW state. The run must then evaluate the already
frozen eight-group 7/30-day train/train screen. Validation, 365-day rollout,
Gate D2, sealed test, and product acceptance remain closed.
