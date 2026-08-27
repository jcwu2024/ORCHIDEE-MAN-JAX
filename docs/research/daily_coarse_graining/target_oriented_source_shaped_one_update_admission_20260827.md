# Target-Oriented Source-Shaped One-Update Admission

Date: 2026-08-27

Status: accepted free-GPU engineering admission; paid 100-update training is
not yet authorized

Report SHA256:
`c2743cbc3cd563b682176531e7ec2310c292f5fd1f63bbf625c846f79434ea59`

Executable commit:
`0ed926ba624cf887007542dbf3470b736576abbc`

Screen contract SHA256:
`a06beb815834142f44939022eea6ebd044e0a096b5d3c443754a9d87fb0f8cee`

## Result

The source-shaped candidate passes its bounded one-update admission. Eight
disjoint train/train effective batches produced finite, nonzero gradients in
both new branches. Their complete branch-gradient norms were:

```text
0.0930788, 0.304030, 0.0807769, 0.112726,
0.0540816, 0.0953099, 0.151683, 0.0591464
```

The frozen p90 clip threshold is `0.1973866669`. On training artifact 252, the
raw branch-gradient norm is `0.2293467275`, so the admitted step uses clip
scale `0.8606474996`. Fresh AdamW update 1 uses learning rate `3e-5`.

Both source branches change by finite nonzero amounts:

| Branch | Parameter-step L2 norm |
| --- | ---: |
| Aboveground | `0.0004809576` |
| Belowground | `0.0004809348` |

The combined step norm is `0.0006801606`. All 83,754,404 retained parameter
coordinates remain unchanged; only the 394,242 new branch parameters are
replaced. Optimizer step is exactly 1, and the complete proposed parameter
tree and optimizer state contain no nonfinite value.

The update-252 effective-batch objective changes from `0.2492312576` to
`0.2491473769`, a small `0.03366%` decrease. This decrease is useful as a
direction sanity check, not a scientific result from one warmup step. Before
and after the update, prediction, state, mask, discrete-state, budget, and
negative-stock hard failures are zero. The observed `rveget` declared-dynamic
mask changes remain diagnostic and are excluded from the frozen hard count.

The proposal was discarded after the post-update forward check. No checkpoint
or recovery was written, and no optimizer update was persisted.

## Data And Scope

Calibration used prepared updates `352, 353, 354, 355, 356, 357, 359, 360`.
The ephemeral optimizer update used prepared artifact `252`. These assets are
disjoint from each other and from evaluation updates
`251, 358, 465, 572, 679, 786, 893, 1000`.

All inputs came from the existing v6 train/train prepared assets. Teacher data
was not regenerated, historical v5 was not used, and validation and sealed
test values were not read. Final inference still reconstructs no 48-record
forcing sequence and performs no 48-step state scan.

## Resource Evidence

The run used one free `gln01` V100 and no Slurm allocation. Total wall time was
`5086.38 s` (`84.77 min`). The first effective gradient took `4632.94 s`
because it compiled all three full-parameter reverse graphs. Its one-day,
7-day, and 30-day gradient-source times were `873.15 s`, `1819.97 s`, and
`1821.05 s`. The median of the seven subsequent hot calibration batches was
approximately `36.85 s`; their one-day, 7-day, and 30-day gradient sources
were approximately `4 s`, `8-9 s`, and `16 s`.

JAX peak live device memory was 8,204,654,080 bytes, approximately 7.64 GiB.
The approximately 25 GiB allocator reservation visible in `nvidia-smi` was
not the live tensor footprint.

## Interpretation And Next Gate

This closes optimizer mechanics, gradient clipping, branch isolation, and
single-update physical safety. It does not evaluate the frozen 7/30-day
science panel after training and therefore does not show that the branches
have learned the short-horizon protection required by the screen.

The admission deliberately extracted the branch gradient from the complete
84.1-million-parameter gradient to prove equivalence with the existing full
objective. The measured 77-minute cold gradient makes that implementation
unnecessarily expensive for a 100-update paid run. Before requesting paid
training, differentiate the unchanged objective directly with respect to the
394,242 branch subtree while treating the retained parameters as constants,
then confirm on one existing train/train batch that its branch gradient,
loss, hard counts, and optimizer proposal match this admitted path and record
the new cold/hot resource envelope.

After that engineering equivalence check, the next scientific gate is the
separately authorized 100-update branch-only train/train screen over artifacts
`252..351`, followed by the already frozen eight-group 7/30-day evaluation.
Validation, 365-day rollout, Gate D2, sealed test, and product acceptance
remain closed.
