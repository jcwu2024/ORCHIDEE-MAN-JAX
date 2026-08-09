# Successor Small-Fit GPU Probe Acceptance

Date: 2026-08-09.

Status: accepted free-`gln01` preflight; no paid fit is authorized by this
record.

## Bound Evidence

- Git commit: `d1ae419937f5129604b3aff6afdd07113561f620`.
- Runner contract SHA256:
  `8317fc74b6ba921b0ed60e774743884ddc81b233c75ce1823f3246211d1640de`.
- Schedule SHA256:
  `f6c0d36f500f1dc47699cd8783c0bbf2379a9d0918751d9f1b99280e6a74103b`.
- Report:
  `runtime/outputs/training/successor-small-fit-probe-d1ae419/probe_report.json`.
- Report SHA256:
  `1e79bc76b97098ecf0725d94edfe144363420dba18792f05105e8f0707d543b6`.
- Backend/device: JAX GPU on `cuda:0`, one Tesla V100-SXM2 32 GB.
- Dataset manifest and acceptance hashes remain the accepted v6 identities.
  Historical v5 was not used and Teacher data were not regenerated.

## Gate Results

The deterministic first schedule entry was landpoint `141.0-073.0`, year
1988, with rollout rows 172 and 38. Both Teacher replay days passed all 160
fast-owner leaves, exact defined status, source-exact undefined
representation, exact discrete state, and complete next-state tolerances.
Water and carbon conservative residuals were both exactly zero under the
cancellation-stable diagnostic.

The candidate-only train/train coefficient preflight produced finite nonzero
gradients and resolved all four coefficients inside `[0.001, 10]`:

| Component | Gradient norm | Coefficient |
| --- | ---: | ---: |
| cumulative flux MSE | 261,321.109 | 0.0268434 |
| cumulative flux bias squared | 261,321.109 | 0.0402651 |
| terminal stock MSE | 2,241,811.471 | 0.00312905 |
| terminal stock bias squared | 2,241,811.471 | 0.00469358 |

The same coefficients were then used unchanged by both arms. Each arm
completed exactly one AdamW update with all 8,722,689 gradient elements
finite and all 155 parameter leaves nonzero. Updated parameters were finite.
All rollout hard-failure counts were zero: budget, defined status, discrete
state, nonnegative declared stocks, and finite defined prediction.

| Arm | Total before | Total after | Anchor before | Anchor after |
| --- | ---: | ---: | ---: | ---: |
| process attention | 117.261 | 77.937 | 110.474 | 77.015 |
| graph MLP control | 123.328 | 81.215 | 115.844 | 80.201 |

This is an execution and trainability acceptance, not a comparative model
decision. One update cannot establish that attention is better than the
matched control.

## Repairs Proven By The Probe

The accepted path preserves the source `NaN` in the bare-soil
`roughheight_pft` slot while keeping its defined mask false; uses an
algebraically equivalent cancellation-stable conservative residual; registers
the CPU backend needed by JAX 0.4.38 scan partial evaluation while retaining
CUDA as the mandatory default; and defines a zero tangent for exact
zero-variance summary standard deviations without changing their forward
values. The last repair removed a real failure in which the forward 30-day
loss was finite but 8,685,057 of 8,722,689 parameter gradients were nonfinite.

## Measured Envelope And Next Boundary

- End-to-end cold probe wall time: 1,807.9 seconds (30.1 minutes).
- Host batch preparation: 40.0 seconds.
- Teacher context and retained adapter: 29.3 seconds.
- Coefficient preflight, including compile: 429.1 seconds.
- Per-arm train compile plus one update: 507.9 seconds attention and 500.2
  seconds control.
- Per-arm post-update compile/forward: 93.1 seconds attention and 92.5 seconds
  control.
- Observed GPU allocation during the largest graphs: about 25.4 GiB.
- No checkpoint was written and the GPU was released at exit.

These cold one-update measurements do not yet establish the 500-update
production wall time. The current accepted script intentionally stops after
one update and therefore is not a paid training command. The next active task
is to implement the contract-bound production runner, prove update-to-update
compile/runtime reuse on a free bounded case, and only then freeze the exact
`gnall` resources, wall time, command, working directory, and worst-case cost
for explicit user approval.
