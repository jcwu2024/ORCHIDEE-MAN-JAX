# Group-Aware Finite-Step Extension, 2026-08-15

## Decision

Admit global parameter step norm `0.002` for a bounded frozen-scalarization
smoke. The group-aware direction remains common and useful over the extended
grid: all 40 group-component losses decrease, and at `0.002` every component
family reaches the preregistered median effect size.

This passes the finite-step mechanism gate. It does not promote the
Attention checkpoint or justify computing 40 separate gradients on every
production training update.

## Protocol

The frozen extension contract SHA256 is
`e78db1bac583d6adf853eb38207c7c410b5befb1cbb087e01da527d28fd3bda1`.
It recomputes the accepted 40-objective minimum-norm direction at the same
update-1000 parameters and evaluates virtual norms
`0.0005, 0.001, 0.0015, 0.002` on the same ten train/train groups. The grid
was frozen from the prior near-linear anchor threshold estimate of `0.00185`.

The run performs zero optimizer updates, retains no proposed parameters, and
does not read validation or sealed-test values.

## Result

The free-`gln01` run at commit `f0b2235` completed with valid evidence:

- runtime report:
  `outputs/training/successor-group-aware-step-extension-f0b2235/report.json`;
- report SHA256:
  `a8b448488a25e924a36dc356a0bdba6f6cc213c6509e7e21ac3e4af0635edf9a`;
- geometry sidecar SHA256:
  `415a32e181cac84c23dcf6ab4dec952216c481832dcdddbc7813e865158e227b`;
- elapsed Python time: `899.14 s`;
- peak JAX device bytes: `15,530,472,704`;
- hard failures and nonfinite values: zero;
- optimizer updates and retained proposals: zero.

All four step norms improve every component in all `10/10` groups. At the
selected norm `0.002`, the component median relative changes are:

| Component | Median relative change |
|---|---:|
| One-day anchor | `-1.217643e-4` |
| NPP | `-1.019272e-3` |
| AGB tendency | `-1.336053e-3` |
| BGB tendency | `-3.446248e-4` |

The least negative group-component change is `-1.802782e-5`. Therefore the
pass is not produced by medians hiding a regressing group. Formal
classification is `group_aware_finite_step_supported`, with selected norm
`0.002`.

## Efficient Realization

The accepted simplex has 14 positive constraints distributed across eight
groups. For constraint `i`, freeze

```text
scalar coefficient_i = simplex weight_i / parent gradient norm_i
```

The gradient of the resulting weighted scalar loss is exactly the accepted
sum of unit gradients at the parent parameters. Components in the same group
share one vector-Jacobian product, reducing the reverse passes per update
from 40 to eight. All ten groups are still evaluated before and after every
proposed update.

The five-update smoke freezes these coefficients only to test local drift. It
uses the same 40 point-years on each update and therefore is not a final data
sampling or training policy. The update-1000 parent is the matched no-training
control for the accepted independent 7/30-day validation subset.

## Next Gate

The frozen smoke contract is
`successor_group_aware_scalarization_smoke_v1.json`, SHA256
`42423d0b62bd27d632809cfbbd494412213d4fcc777f087e54aef43ae861642a`.
It applies at most five in-memory updates at norm `0.002`, stops before any
update that regresses a group-component loss, retains no checkpoint, and then
evaluates the established six-artifact non-test 7/30-day subset.

Passing would open design of a rotating-group protocol that periodically
recalibrates robust weights. Failure would close reuse of fixed parent
weights; it would not reject the Attention architecture or the existence of
the common direction at update 1000.
