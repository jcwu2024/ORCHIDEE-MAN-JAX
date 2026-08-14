# Successor Named-Biomass Gradient Preflight

Date: 2026-08-14

Status: **passed; coefficients accepted; training still closed**.

## Question

The prior per-output audit found NPP gradients around `1e-1`, maintenance
gradients around `1e-2`, and AGB/BGB gradients around `1e-6-1e-7` under the
old terminal-stock normalization. This preflight asks whether the accepted
train-only AGB/BGB tendency scales produce usable stock gradients without
changing the Attention architecture or performing an optimizer update.

## Frozen Method

The probe loads the accepted update-1000 Attention parameters and 16 existing
train/train point-years from prepared updates `1001-2000`. Four groups each
contain one rollout context from four distinct point-years. It measures NPP,
AGB, and BGB parameter-gradient norms at 1/7/30 days.

NPP retains the accepted cumulative-flux MSE and bias coefficients. AGB and
BGB use equal-horizon, equal-field tendency MSE plus bias squared at unit
weight. One shared stock coefficient is fixed by

```text
npp_median / sqrt(agb_unit_median * bgb_unit_median)
```

The coefficient must lie in `[1e-6, 10]`, and both resulting stock/NPP median
gradient ratios must lie in `[0.1, 10]`. The probe then evaluates the complete
candidate objective, including the unchanged one-day named-process anchor,
four optimized cumulative fluxes, and the two named tendencies. Long-horizon
maintenance and old terminal-stock components do not enter the summed
gradient.

## Result

| Quantity | Value |
| --- | ---: |
| NPP median gradient norm | 0.1928996432968742 |
| AGB unit median gradient norm | 8.115338889968996 |
| BGB unit median gradient norm | 1.5491119023599702 |
| Shared tendency coefficient | 0.05440473497107039 |
| Weighted AGB/NPP ratio | 2.288821554893659 |
| Weighted BGB/NPP ratio | 0.4369060566831568 |

Both ratios pass the frozen interval. All four complete candidate gradients
are finite and positive; their norms range from `0.2317` to `2.1352`. Every
nonfinite, negative-stock, mask, discrete-state, and water/carbon budget hard
count is zero.

The tendency normalization therefore repairs the previously unusable stock
gradient magnitude. This is a gradient-path result, not evidence that a
trained model improves 7/30-day rollout.

## Evidence

- producer commit: `b8ca4d7934df9b1fcf85f8af62ce8ced1722c757`
- free node: `gln01`, GPU 0, Tesla V100 32 GB
- elapsed Python time: `1100.18 s`
- peak JAX device bytes: `12,722,128,640`
- report:
  `outputs/training/successor-named-biomass-gradient-preflight-b8ca4d7/report.json`
- report file SHA256:
  `2023f3cf7b32c4406ee9aae02daccd4b0ae1084e21c6b5ca9de088dc8b0d7413`
- preflight contract SHA256:
  `ec38fc1316c54776c54dd8b9dadd01dcf1e5caf18a4340c466e89244a9442419`
- accepted coefficient contract SHA256:
  `2093c392cc24a59f0b5021b7ecfafade7e3522fd3953014df63d4d85517b1d9f`

The run performed zero optimizer updates, wrote no checkpoint or recovery,
did not regenerate Teacher data or use v5, and did not evaluate validation or
sealed-test values.

## Next Gate

Before implementing or launching the no-checkpoint 100-update objective A/B,
freeze two remaining scientific decision thresholds:

1. the maximum allowed maintenance-respiration regression relative to the
   current-loss Attention control;
2. the required majority among the six NPP/AGB/BGB/total-live-biomass
   slice-horizon cells.

Training remains closed until those thresholds are reviewed.
