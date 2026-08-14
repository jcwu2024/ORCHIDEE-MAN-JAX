# Successor Batch-Diversity Probe and Short-Fit Gate

Date: 2026-08-14

## Question

The accepted staged fit updates the network from one point-year at a time,
using 32 one-day anchors and four 30-day rollout windows. This probe asks
whether changing only the point-year diversity of an update is promising
enough to justify a short empirical training comparison.

It does not change the architecture, loss coefficients, total samples per
update, Teacher release, operator boundary, or accepted update-1000 parent.

## Probe Design

The two compared batches are:

```text
single: 1 point-year x (32 anchors + 4 rollout windows)
mixed:  4 point-years x (8 anchors + 1 rollout window)
```

Both contain exactly 32 anchors and four 30-day windows. The mixed loss
retains the joint signed-bias reduction across all four windows. Sixteen
distinct point-years were selected from the accepted update 1001-2000
prepared manifest. No parameter update or checkpoint write occurred.

An equivalence case first partitions one ordinary batch into four contexts,
then requires the ordinary and mixed implementations to agree on loss,
long-horizon components, full parameter gradient, and hard-gate counts.

## Result

The free-`gln01` run completed at commit `14b9420` in `2628.21` seconds.
Report SHA256 is
`eec3fccc372ad72f1818c86f7867c1eb63fc39d8a463bfe7b883196f3e88925a`.

Both equivalence cases pass. Relative full-gradient differences are
`5.01e-17` for attention and `2.39e-18` for control. All numerical, state,
mask, discrete, nonnegative-stock, and budget hard counts are zero.

| Arm | Batch | Gradient norm range | Pairwise negative fraction | Cancellation ratio |
| --- | --- | ---: | ---: | ---: |
| attention | single | 0.0893-0.6801 | 1/6 | 0.5034 |
| attention | mixed | 0.1157-0.7286 | 4/6 | 0.6608 |
| control | single | 0.1543-0.5240 | 0/6 | 0.8518 |
| control | mixed | 0.2083-0.7482 | 3/6 | 0.7524 |

None of the 16 measured gradients exceeds the global clipping threshold of
1.0. This does not contradict frequent clipping during earlier training: it
shows that clipping is batch-dependent and is not universal at the accepted
update-1000 checkpoint.

The mixed batches materially change update directions, but they do not
consistently reduce conflict or cancellation across the sampled groups.
Gradient geometry alone therefore does not admit a training-policy change.

JAX reports estimated compiled device residency of about 12.50 GB for the
attention path and 10.93 GB for control, nearly unchanged between the two
equal-sample layouts. Allocator peak was 13.08 GB; external resident GPU
memory reached about 22.2 GiB. Preallocation was disabled. The difference is
runtime/context/executable/allocator overhead, so external memory remains an
upper bound rather than irreducible live tensors.

## Frozen Short-Fit Gate

One empirical comparison is allowed on free `gln01` before rejecting or
adopting mixed point-year updates:

- start both strategies from the same accepted update-1000 parameters and
  AdamW state;
- run exactly 100 continuation updates at the frozen update 1001-1100
  learning rates;
- keep both architecture arms, loss coefficients, 32 anchors, four rollout
  windows, and 30-day horizon unchanged;
- construct each mixed update from four distinct point-years and rotate which
  member supplies the matched single-point batch;
- evaluate the unchanged 48-window temporal/spatial/joint non-test panel;
- write a report only, with no promotion checkpoint.

The mixed strategy is considered worth a later training proposal only if all
hard counts are zero, it is no worse than the matched single strategy for
both architecture arms, and it improves at least one arm's primary composite
by at least 0.5%. Otherwise stop this sampling branch. This screen cannot
promote an architecture or open 365-day, held-point, Gate D2, sealed-test, or
product gates.

## Short-Fit Result

The free-`gln01` short fit completed at commit `aa602ab` in `4130.87`
seconds. It executed all 100 updates and the complete unchanged 48-window
validation panel. Report SHA256 is
`6c3f16ffb8b160bd85b1840955f50d0f422685d323b200ce309fcd89d161154b`.
No checkpoint or recovery was written, and all training and validation hard
counts are zero.

| Arm | Update-1000 baseline | 100-update single | 100-update mixed | Mixed vs single |
| --- | ---: | ---: | ---: | ---: |
| attention | 1.01457146 | 1.05300199 | 1.00714567 | +4.35% |
| control | 1.00607822 | 1.00288890 | 1.00539728 | -0.25% |

Attention mixed sampling improves all six slice/horizon primary cells over
its matched single continuation and improves the update-1000 composite by
0.73%. Attention single sampling instead worsens the baseline by 3.79%.
Control single and mixed sampling both improve the baseline slightly, but
single is 0.25% better than mixed. The preregistered requirement that mixed
sampling be no worse for both arms therefore fails, and the sampling branch
is rejected for adoption or paid continuation.

Pre-clip gradient counts above 1.0 are `42/100` attention-single, `47/100`
attention-mixed, `62/100` control-single, and `60/100` control-mixed. Large
isolated spikes remain in every path. Point-year diversity changes which arm
is stable, but it does not consistently eliminate clipping or validation
tradeoffs.

The scientific pattern is more useful than the binary gate alone. Single
sampling often improves maintenance respiration while worsening NPP and live
biomass. Mixed sampling largely preserves or improves NPP and biomass but
gives up some maintenance-respiration improvement, especially for control.
The next bounded task is therefore loss-component conflict attribution using
the existing short-fit records and frozen validation outputs. Do not train
again, change architecture, or tune against the validation panel during that
attribution.
