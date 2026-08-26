# Protected Flux Virtual-Step Result (2026-08-26)

## Decision

The zero-update audit is valid and complete, but no tested direct parameter
step protects the already successful state and stock behavior while improving
all required 7/30-day flux targets. The result closes direct realization of
the raw component-resolved MSE-plus-squared-bias loss at the update-1000
Attention state.

A common first-order direction does exist. Its realized minimum cosine with
the 16 group-by-horizon flux gradients is `0.04353145`, well above the frozen
`0.001` gate. The failure occurs when that direction is realized as a finite
float32 model step: even norm `0.0001` regresses some exact flux losses,
one-day state/process loss, and stock metrics by amounts far beyond the
protection limits.

This result does not reject the 83.8-million-parameter Attention architecture.
It rejects unchanged AdamW continuation, the raw amended-loss mean, and the
present normalized common-direction realization. The next gate is a
zero-training residual-tail and sampling-unit audit before any robust loss is
selected or trained.

## Evidence Identity

- branch: `research/daily-coarse-graining`;
- executable commit:
  `04c2b2335c3afdb9842f7f0b4d9e3be89f235ee3`;
- report schema: `target_oriented_flux_protected_virtual_step_audit_v1`;
- remote report:
  `runtime/outputs/diagnostics/target-oriented-flux-protected-virtual-step-04c2b23/report.json`;
- local ignored copy:
  `outputs/diagnostics/target-oriented-flux-protected-virtual-step-04c2b23/report.json`;
- report SHA256:
  `6a9ed9c93141ec1d40f582a6fcfe314b6ade2cc1c52432f36fe57145339f1fba`;
- virtual-step contract SHA256:
  `0781cd758a3c30c9558801beae451301cb66ad794a5ad8aaf47d8f3473a9053b`;
- update-1000 checkpoint SHA256:
  `a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7`;
- prepared manifest SHA256:
  `8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31`.

The free-`gln01` V100 execution took `2095.51 s` (`34:55.51`): baseline
forwards took `507.52 s`, gradient and geometry work took `616.35 s`, and
seven exact virtual-step evaluations took `785.52 s`. JAX peak live device
memory was `3,094,284,800` bytes (`2.88 GiB`), versus an allocator limit of
`25,559,728,128` bytes. No paid resource or Slurm job was used.

## Direction Geometry

All 16 gradients are finite and nonzero over all `83,755,173` parameters.
The minimum-norm simplex uses 13 constraints; the update-358 30-day,
update-786 30-day, and update-893 30-day weights are zero. The largest
normalized weights are:

| Constraint | Weight |
| --- | ---: |
| update 572, 7-day flux | 0.165104 |
| update 465, 30-day flux | 0.155834 |
| update 679, 30-day flux | 0.141900 |
| update 1000, 30-day flux | 0.124187 |
| update 572, 30-day flux | 0.122961 |

The ideal minimum cosine is `0.04353191`; the independently reconstructed
float32 direction gives `0.04353145`, has norm `0.99999999`, and differs from
the geometry norm by only `3.71e-9` relatively. Direction construction is
therefore not the engineering failure.

## Finite-Step Gate

Every step has zero physical hard failures and zero nonfinite values, but all
seven fail the scientific and protection gates. Values below are relative
changes, where positive means worse.

| Step norm | Max independent flux | Max derived flux | One-day median | One-day max | Max stock | Parent-passing absolute keys retained |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0.0001 | +0.300893 | +0.145955 | +0.115593 | +0.178812 | +0.331712 | yes |
| 0.0003 | +0.228976 | +0.086915 | +0.669450 | +1.088574 | +0.128778 | yes |
| 0.0010 | +0.307192 | +0.203007 | +6.156188 | +10.069800 | +0.271475 | yes |
| 0.0030 | +0.627872 | +0.540459 | +53.317800 | +85.406294 | +0.564476 | yes |
| 0.0100 | +1.563187 | +0.826444 | +581.019083 | +928.148395 | +1.041525 | no |
| 0.0300 | +3.658566 | +2.072838 | +5210.144168 | +8288.297455 | +2.536654 | no |
| 0.1000 | +5.038037 | +4.290862 | +58389.132345 | +91577.651674 | +2.598507 | no |

The smallest step already fails every substantive gate:

- its exact 7-day component-flux loss has median change `+0.006359` and
  maximum change `+0.157217` across the eight groups;
- its exact 30-day component-flux loss has median change `-0.0000881`, but
  one group still regresses by `+0.054535`;
- 7-day median GPP, maintenance, and growth-respiration scores regress by
  `+0.006692`, `+0.000855`, and `+0.021488`;
- 30-day median GPP and growth-respiration scores regress by `+0.000164` and
  `+0.0000237`, while maintenance improves by `-0.000521`;
- 7-day and 30-day median NPP improve by only `-0.000110` and `-0.000673`,
  while individual-group NPP regressions reach `+0.145955` and `+0.066083`;
- one-day state/process loss has median `+0.115593` and maximum `+0.178812`,
  against allowed values of zero and `+0.0001`;
- update 893's 7-day aboveground-biomass score regresses by `+0.331712`,
  against the `+0.01` stock cap.

Thus this is not merely a larger-step overshoot. The smallest frozen norm is
already outside a common exact-loss trust region for some groups, and the raw
equal-field squared-error objective permits tradeoffs that violate individual
GPP, respiration, NPP, state, and stock requirements.

## Closed Responses

Do not respond to this result by:

- shrinking or extending the same direct-step grid;
- enlarging the effective batch again;
- continuing AdamW from update 1000;
- changing the Attention architecture;
- reopening a weak candidate/control comparison;
- regenerating Teacher v6, joining historical v5, or reading sealed test;
- opening 365-day rollout, Gate D2, or product acceptance.

No optimizer was invoked, no proposal was retained, and no checkpoint or
recovery was written.

## Next Gate

Before choosing a robust loss, measure the residual distribution at the exact
update-1000 state over the same eight train/train groups. The audit must retain
the physical sampling unit instead of merging all residuals immediately. At
minimum it must report, separately by horizon and flux key:

1. per-window and per-group normalized residual-loss distributions;
2. top-1, top-5, and top-10 contribution shares and effective sample size;
3. whether the update-465 7-day domination comes from a few windows,
   persistent field bias, or a broad regime shift;
4. the effect of bounded candidate influence functions on group and field
   weights without taking a parameter step;
5. whether equal window, equal point-year group, or an explicitly stratified
   unit best preserves rare regimes without allowing squared-error outliers to
   control an update.

Freeze that audit before implementation. Only its train/train result may
select the robust loss and sampling unit for a later zero-update geometry
gate. No new training is authorized by this result.
