# Target-Oriented Update-250 Training-Dynamics Diagnosis (2026-08-25)

## Status

The complete update-1-to-250 record shows real learning, but it also exposes a
specific imbalance: long-horizon gradients strongly reach the direct primary
science heads while remaining much smaller than the one-day objective in the
shared backbone and state heads. This is consistent with substantial GPP/NPP
improvement and almost stationary 30-day nonliving ecosystem carbon.

This diagnosis does not change the frozen architecture, objective, acceptance
gates, or the running update-251-to-1000 continuation. It fixes the analysis
that will be applied immediately when the update-1000 report is available.

## Executable Evidence

The reusable analyzer is
`scripts/hpc/analyze_target_oriented_training_stage.py`. It reads the formal
stage report, separates records belonging to the current continuation stage,
summarizes 50-update blocks, reports state/science loss geometry and gradient
conditioning, compares milestone validation panels, and applies the frozen
absolute and 5% pacing decisions.

The update-250 analysis ran on free CPU node `test01` and wrote:

- JSON:
  `runtime/outputs/diagnostics/target-oriented-training-analysis-u250-433d286/analysis.json`,
  SHA256 `038bd1ab86688cc347897597240e7e4599e1ab5508514bed986ca7ff68f24f68`;
- Markdown:
  `runtime/outputs/diagnostics/target-oriented-training-analysis-u250-433d286/analysis.md`,
  SHA256 `bb588ac16a228ade4c260d517025384a5936f880ed753970d72e27710e1888aa`.

The implementation and its synthetic continuation/gate tests pass with the
existing formal-training tests (`12 passed`); Ruff and `git diff --check` also
pass.

## Training Dynamics

Fifty-update means are:

| Updates | Total loss | One-day state | 7-day science | 30-day science | Necessary process |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1..50` | 256.577 | 255.816 | 0.383836 | 0.336190 | 0.408846 |
| `51..100` | 145.980 | 145.155 | 0.376327 | 0.407792 | 0.415097 |
| `101..150` | 63.277 | 62.596 | 0.301571 | 0.341104 | 0.389785 |
| `151..200` | 42.807 | 42.119 | 0.259375 | 0.391827 | 0.373171 |
| `201..250` | 16.558 | 16.040 | 0.193362 | 0.286605 | 0.381306 |

The one-day anchor falls `93.73%` between the first and last blocks and the
7-day science term falls `49.62%`. The 30-day term falls only `14.75%`, and
necessary process supervision falls `6.74%`. The optimizer is therefore
learning strongly, but not uniformly across objective components.

The one-day state groups mostly fit well. Carbon falls `92.82%`, water
`95.07%`, soil `95.62%`, snow `99.91%`, and the broad `other` group `97.69%`.
Thermal falls only `61.90%` and remains the largest last-block group loss
(`75.49`), followed by `other` (`36.74`). Thus large residual one-day loss is
not evidence that the carbon group itself is unlearned.

## Long-Horizon Geometry

The 7-day flux and stock MSE means fall from `0.4820/0.5425` to
`0.2028/0.3189`. The 30-day flux MSE falls from `0.4373` to `0.2673`, but the
30-day stock MSE rises from `0.4122` to `0.4812` and is non-monotonic across
all five blocks. Because AGB and BGB already validate well, this stock
behavior is dominated by nonliving ecosystem carbon rather than live biomass.

The frozen validation panel confirms the distinction:

- 30-day ecosystem carbon mean NRMSE improves only `0.380%`;
- 30-day nonliving ecosystem carbon improves only `0.344%`;
- 30-day GPP improves `36.94%` and NPP improves `53.45%`;
- 30-day AGB/BGB/total-live biomass improve `49.42%`, `54.12%`, and `52.13%`
  and pass all three spatial/temporal/joint cells;
- 30-day maintenance respiration worsens `9.36%`.

This is not a generic failure to optimize. It is a selective failure of the
slow-stock trajectory despite successful direct productivity and live-biomass
learning.

## Gradient Conditioning

At update 1, the 7-day and 30-day shared-backbone norms are only
`7.78e-4` and `4.14e-4`, versus `1.11e4` from the one-day/process source. By
update 250 they rise to `5.76e-2` and `1.54e-2`, showing that warmup does
increase long-horizon influence and does not disconnect those losses.

The allocation remains highly asymmetric at update 250:

| Source | Shared backbone norm | Primary-head norm | State-head norm |
| --- | ---: | ---: | ---: |
| one-day/process | 179.175 | 0.0137 | 120,504 |
| 7-day science | 0.0576 | 1.8840 | 1.3592 |
| 30-day science | 0.0154 | 0.8289 | 0.0402 |

The direct primary heads are therefore driven mainly by 7/30-day science,
which explains the GPP/NPP progress. The state heads and shared representation
remain dominated by the one-day anchor, which is a plausible cause of weak
30-day slow-carbon correction. The conditioning snapshots use different
effective batches, so their exact ratios are diagnostic rather than a new
quantitative gate; the order-of-magnitude separation is nevertheless clear.

## Update-1000 Decision

The running continuation remains the correct next experiment because update
250 is the end of warmup and the long-horizon shared gradients are increasing.
The update-1000 analyzer will compare initialization, update 250, and update
1000 without changing the frozen decisions:

1. zero hard failures plus the complete absolute 7/30-day pass opens review
   of separately authorized 365-day work;
2. otherwise, zero hard failures and update-1000 macro at most
   `0.6550092954` opens review of updates 1001-3000;
3. otherwise, stop training and attribute the failure.

Separately from those frozen gates, the report must flag whether 30-day
ecosystem and nonliving ecosystem carbon have begun to improve materially from
update 250. If the aggregate 5% pacing rule passes while both slow-carbon keys
remain nearly stationary, another 2,000 ordinary updates are not justified by
the aggregate alone; first attribute state-head weighting, stock-loss scaling,
and slow-carbon output representation. This is a diagnostic interpretation,
not a post-hoc promotion gate.
