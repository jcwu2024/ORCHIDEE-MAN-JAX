# Experiment C Stock-Adapter Ablation Result

Date: 2026-07-31

## Decision

The bounded ablation fails its predeclared feasibility gate. The exact-zero
stock intervention must not be reused as a candidate, and Experiment C
remains rejected.

This does not show that carbon stocks should be left entirely parent-owned.
It shows why the next candidate must update them through learned,
budget-consistent transfers rather than either independent stock residuals or
an unconditional return to the frozen parent.

## Execution Evidence

The evaluation ran on free `gln01` GPU 1 from commit `7c53f79`. It used the
completed Experiment C checkpoints without retraining and set only
`carbon_stock_interface` output weights and biases to exact zero.

The deterministic one-shard-per-slice smoke completed all three references in
about 145 seconds:

- temporal: 1;
- spatial: 1;
- joint: 1;
- sealed test: unused.

The old and new inventories, expected counts, control-arm metrics, and
control restart probe are exactly equal. The candidate checkpoint SHA256 is
unchanged, and the report records the post-training transform.

Evidence:

```text
output:
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/screening/causal-carbon-stock-ablation-smoke-7c53f79

top-level report:
39413550c372705cf96d198f5392b24db8c25d00087c9eafbc32885a574b2b9c

control arm:
dedad50808dbc38b224a1dbfef4ff1edab1d60d9281b01a6bc7b7b19c523eebd

stock-ablated candidate:
adcb16f7532da8c3e7aad1d956605c22937f8a5706d31310ec9d54b22b3b1f92
```

## Results

All exact hard counts are zero, both restart probes pass, and all 18
litter/DOC guards pass.

The unmodified GPP and maintenance-respiration interface metrics reproduce
the original candidate exactly. Across all 27 downstream flux-bias metrics,
the ablated/original-candidate ratio stays between about `0.9977` and
`1.0031`; the learned flux behavior is therefore effectively preserved.

Carbon-stock behavior separates by split:

| Split | Field | Ablated / control | Ablated / original candidate |
| --- | --- | ---: | ---: |
| temporal | `carbon_32l` | 0.0349 | 0.00384 |
| temporal | `deepC_peat` | 0.0205 | 0.00530 |
| spatial | `carbon_32l` | 13.7136 | 0.70670 |
| spatial | `deepC_peat` | 16.2048 | 0.87772 |
| joint | `carbon_32l` | 0.4418 | 0.03366 |
| joint | `deepC_peat` | 0.4357 | 0.06148 |

The intervention strongly repairs temporal and joint stock errors but does
not recover the spatial stock boundary. Only 9/27 flux-bias gates pass
relative to the strong matched control in this small smoke, below the frozen
24/27 requirement, although the flux metrics themselves are virtually
unchanged from the unablated candidate.

## Attribution

Two conclusions are supported:

1. Experiment C's independent stock residual caused substantial damage.
2. The frozen parent is not an adequate spatial stock solution by itself.

The next bounded work is therefore a source audit of ORCHIDEE carbon-budget
ownership. A successor is admissible only if it predicts source-backed
fluxes/transfers and computes `carbon_32l` and `deepC_peat` through an explicit
conservative update. No new paid training is authorized by this result.
