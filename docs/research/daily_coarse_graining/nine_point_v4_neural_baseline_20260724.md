# Nine-Point v4 Neural Baseline

Date: 2026-07-24.

## Scope

This is architecture-development evidence, not the frozen ten-point result.
Point `319.0-057.0` was excluded in full because its 1970-2010 Teacher chain
was unavailable. The verified provisional dataset contains nine complete
1961-2010 landpoint chains, 450 shards, and 164,241 transitions. Its split is
six train, one validation, and two sealed test landpoints. The test split was
not evaluated.

The dataset acceptance gate passed with zero target-representation persistence
mismatches. Train/train statistics used 96,354 samples. The first neural run
used 1,953,089 parameters, batch size 256, learning rate 0.001, five epochs,
and seed 20260724. Explore1000 job `14367456` completed on one V100 in 7:13
with about 2.45 GiB peak RSS.

## Result

Training loss fell from 0.041750 to 0.013188. The validation selection score
improved in every epoch and reached its best value, 0.811950, at epoch 5.

| Split | Persistence baseline | Neural epoch 5 | Relative improvement |
| --- | ---: | ---: | ---: |
| temporal | 0.386892 | 0.143387 | 62.9% |
| spatial | 1.125126 | 0.854883 | 24.0% |
| joint | 1.680968 | 1.437579 | 14.5% |
| equal-split selection | 1.064329 | 0.811950 | 23.7% |

This establishes nontrivial one-step learnability. It does not establish a
usable free-running daily surrogate. Spatial and joint errors remain high,
especially for DIFFUCO/ENERBIL and HYDROL.

## Classifier Defect

The first v4 network predicted the absolute defined/undefined state for the
two dynamic `diffuco.rveget` columns. This discarded a strong persistence
prior:

- temporal persistence made 18 classification errors, while the network made
  139;
- spatial and joint persistence were exact, while network accuracy was about
  50% because it predicted unnecessary state changes.

The accepted follow-up representation therefore predicts whether each dynamic
undefined state flips relative to its day-start owner. Its classifier bias is
initialized to a strong no-flip prior. Checkpoint schema v3 makes the old
absolute-state checkpoint incompatible by construction. The continuous 2,815
value target and its statistics are unchanged.

## Decision

The same-budget flip-classifier A/B ran as Explore1000 job `14367477` and
completed in 5:35. It reduced the equal-split selection score from `0.811950`
to `0.793159`. Final absolute undefined-state errors changed from
`139/15993/1079` to `18/0/0` on the temporal/spatial/joint validation splits,
matching the persistence prior. The flip representation is therefore
accepted and the absolute-state checkpoint remains rejected.

A fresh 15-epoch run at the same learning rate and seed then ran as job
`14367514`. It completed in 14:22 with about 3.39 GiB peak RSS. Epoch 10 was
best:

| Split | Persistence baseline | Neural epoch 10 | Relative improvement |
| --- | ---: | ---: | ---: |
| temporal | 0.386892 | 0.127768 | 67.0% |
| spatial | 1.125126 | 0.810324 | 28.0% |
| joint | 1.680968 | 1.379018 | 18.0% |
| equal-split selection | 1.064329 | 0.772370 | 27.4% |

Longer training materially improved temporal skill but only modestly improved
spatial and joint skill. DIFFUCO/ENERBIL and HYDROL spatial errors did not
improve, so further epochs at the same configuration are not the next
experiment. The bounded one-step learnability gate is sufficient to justify
implementing a seven-day validation-only free rollout. The sealed test split
remains untouched, and no production surrogate claim is made.
