# Coupled GPP Exact-Zero Displacement Result (2026-09-04)

The free-`gln01` audit at commit `ca59ed5` completed normally in `557.468`
seconds. Its classification is `localized_nonzero_step_discontinuity`.

The exact update-2 recovery reproduces all eight stored update-3 gradient-time
7-day baselines exactly. The aggregate baseline and exact-zero means are both
`0.10814824178512826`; their relative difference is exactly zero. This rules
out a mismatch between the gradient baseline and pure-forward scale evaluator
as the explanation for the rejected update.

At the historical smallest nonzero scale `1/64`, with parameter displacement
norm `0.00012123395861015631`, aggregate 7-day loss is
`0.1087728363298649`, or `0.5775355516%` worse than exact zero. Artifact `550`
(runtime year 1976) moves from `0.08329777949515832` to
`0.088399143814683`, a `6.1242500706%` regression. The other seven artifacts
all improve slightly, so artifact `550` supplies all positive displacement.
Every physical hard failure count remains zero.

The result report is
`runtime/outputs/diagnostics/target-oriented-coupled-gpp-zero-displacement-ca59ed5.json`,
SHA256 `c30f31a95176d118763b96131b80370057f1fa896a0fe20f7c1e99333ca0795d`.
No gradient, parameter update, recovery, or checkpoint was written.

The next gate is a bounded trace of the first rollout divergence within
artifact `550`, using the exact recovered parameters as matched control and
the historical `1/64` update-3 direction as treatment. Do not relax the
7-day gate or change the Attention architecture before that trace.
