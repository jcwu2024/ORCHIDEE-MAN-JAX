# Successor Named-Biomass Tendency Scale Acceptance

Date: 2026-08-14

Status: **accepted for zero-update candidate-objective gradient preflight; no
training authorized**.

## Result

The train-only AGB/BGB tendency scale asset passed on free Explore1000 node
`test01` at producer commit `5bd55c703bc392e1ac075cc63326345d04d7d6fe`.
It read the existing prepared v6 updates `1001-2000`, selected 64 unique
train/train point-years and four rollout windows per point-year, and used 256
windows in total. It performed zero optimizer updates and did not regenerate
Teacher data, read historical v5 data, evaluate validation or sealed-test
values, or write a checkpoint.

Accepted scales in `gC m-2` are:

| Horizon | AGB | BGB |
| --- | ---: | ---: |
| 1 day | 1.3522777765283391 | 0.9474426729431713 |
| 7 days | 4.4218978071529 | 1.6402823405932514 |
| 30 days | 15.710052929812397 | 6.016818078490793 |

The one-day scales use the frozen `0.001 * stock_scale` floor. The 7- and
30-day scales use the empirical train-only standard deviation.

## Arithmetic Gate

Teacher-versus-Teacher objective loss is exactly `0.0`. The maximum absolute
`total_live - AGB - BGB` residual is
`2.7284841053187847e-12`. The identity gate uses the scale-aware float64
bound

```text
32 * float64_epsilon
   * max(abs(total_live), abs(AGB) + abs(BGB), 1)
```

rather than a fixed absolute threshold. The maximum residual-to-tolerance
ratio is `0.04562638181444047`, so the observed discrepancy occupies 4.6% of
the arithmetic allowance. This is reduction-order noise, not a relaxed model
accuracy gate.

## Bound Assets

- runtime-relative asset:
  `outputs/training/successor-named-biomass-tendency-scales-5bd55c7/scale_asset.json`
- file SHA256:
  `fdc1f4167112d292e6ca879cad9aa4506c13152738e283724358e09977d4f175`
- internal canonical asset SHA256:
  `eeeff79e24533a27e9dd69fef80e290134ff3f674100c56e60f7d15636f035e0`
- tendency contract SHA256:
  `18335d842b3498ddc57599dfe7c25fdca649452a2d01ff7405afbf2e3b882945`
- acceptance contract:
  `manifests/coarse_graining/successor_named_biomass_tendency_scale_acceptance_v1.json`
- acceptance contract SHA256:
  `a0ec131530e9dd4f72b3f22267ddf88f8af7db9c7262d7d505c7fc5bf2278490`

The strict loader admits only the bound file and canonical hashes, source
contract, exact scales, passed arithmetic gates, and zero-training/data-use
flags.

## Next Gate

Run a zero-update candidate-objective gradient preflight on train/train data.
It must establish finite forward loss and gradients, zero state/budget hard
counts, and median AGB and BGB gradient norms each within `0.1x-10x` of the
NPP median after an explicitly frozen coefficient rule. Do not start the
100-update Attention objective A/B until that preflight passes and the
maintenance no-regression guard and validation-cell majority are frozen.
