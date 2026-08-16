# Strong-Baseline One-Point-Year Qualification Result

Date: 2026-08-16

## Decision

The first full one-point-year qualification of
`axis_aligned_process_expert_operator_v1` is numerically valid but fails the
predeclared complete-process memorization gate. It does not authorize the
64-point-year qualification, paid training, or any 7/30/365-day rollout.

This is not a blanket architecture rejection. All three productivity metrics
and both reported biomass-tendency metrics pass their `0.05` thresholds, and
all state and budget hard counts are zero. The failure is the residual
181-field process loss and four expert-specific reduction gates.

## Execution Identity

- Git HEAD: `4b849fb9ce9d3cf847d1c01c6dcb621feb6f2034`;
- architecture contract SHA256:
  `baf5791f0786f22e05e8db9115eeab9a77687e19250d4d9efcb5181d6a96351d`;
- v6 dataset manifest SHA256:
  `5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99`;
- report:
  `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/strong-baseline-one-point-4b849fb/report.json`;
- report SHA256:
  `ada70c37247c99c9a45130c2b098e3f6d519bd535aed5e2535b44bfccb4b399b`;
- host/device: free `gln01`, one V100 32 GB;
- elapsed time: `2342.17 s` (`39.04 min`), including data assembly and three
  first-use JAX compilations;
- measured JAX peak live allocation: `7,948,430,848` bytes (`7.40 GiB`);
  the approximately 25 GiB `nvidia-smi` value was the JAX allocation pool.

No Teacher data were regenerated, no v5 data were read, no historical
checkpoint was loaded, and no qualification checkpoint was retained.

## Training Boundary

The run started from seed `20260815`, statically bypassed the complete process
graph path, and used all 364 valid Day-2 through Day-365 rows from train/train
point-year `001.0-071.0`, 1961. Every epoch consumed every real row once in a
new order. Batch size was 8; the final padded positions had zero sample
weight. AdamW used 2% linear warmup, cosine decay from `1e-4` to `1e-5`, and
no gradient clipping. Biases and normalization scales were excluded from
weight decay.

All 4,600 updates had finite loss and gradients. The run did not reconstruct
48 half-hour forcing records or execute a 48-step state scan.

## Gate Results

Complete equal-expert/equal-label process loss fell from `4.2262827066` to
`0.1013226346`, a `97.6026%` reduction versus the required `99%`.

| Expert | Initial loss | Final loss | Reduction | Required | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| canopy GPP | 0.925688 | 0.239749 | 74.10% | 95% | fail |
| hydrology/snow | 0.344885 | 0.032757 | 90.50% | 95% | fail |
| surface energy | 0.500362 | 0.006235 | 98.75% | 95% | pass |
| soil thermal | 1.205039 | 0.088309 | 92.67% | 95% | fail |
| maintenance respiration | 0.712820 | 0.0000839 | 99.99% | 95% | pass |
| litter/POC/DOC | 0.148671 | 0.044429 | 70.12% | 95% | fail |
| daily interface/finalize | 25.746513 | 0.297695 | 98.84% | 95% | pass |

The five primary normalized RMSE values all meet the declared `0.05` bound:

| Quantity | Initial | Final | Result |
| --- | ---: | ---: | --- |
| GPP | 0.396280 | 0.033911 | pass |
| maintenance respiration | 0.672558 | 0.006970 | pass |
| NPP | 1.086642 | 0.040409 | pass |
| AGB tendency | 0.000662 | 0.0000180 | pass |
| BGB tendency | 0.000716 | 0.0000360 | pass |

Prediction nonfinite values, next-state nonfinite defined values, negative
water/carbon inventories, and water/carbon budget failures are all zero over
all 364 days.

## Interpretation

The network can learn the one-point-year GPP, maintenance, and NPP mappings to
the requested normalized accuracy while preserving the physical boundary.
It has not demonstrated complete process-head memorization. The final loss is
concentrated in canopy GPP and daily-interface experts in absolute terms,
while canopy GPP and litter/POC/DOC have the weakest relative reductions.
The current report does not retain per-label final loss, so it cannot yet
separate an optimizer noise floor from one or more label/head bottlenecks.

The AGB/BGB result is not strong evidence by itself. The accepted stock-scale
normalization makes the random-initialization one-day tendencies pass by a
large margin. Before this metric controls another gate, its normalization
must be made discriminating using an existing train-only named-tendency scale
or another explicitly reviewed train-only definition; no validation-tuned
scale is allowed.

## Active Next Gate

Do not repeat the same 100-epoch recipe and do not open 64-point-year
training. First freeze and locally admit a bounded one-point loss-floor
diagnostic that:

1. reports every label and field at initialization and at the final state;
2. distinguishes stochastic batch noise from shared-encoder/head capacity by
   adding an exact all-364-day gradient-accumulation tail;
3. retains a diagnostic-only final state so the loss attribution does not
   require repeating the fit;
4. repairs the non-discriminating one-day AGB/BGB normalization before it is
   used as promotion evidence.

Only a rerun that passes the original complete-process reductions, the
repaired primary metrics, finite gradients, and zero hard failures may open
the 64-point-year gate.
