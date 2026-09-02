# Coupled-GPP Update-2 Geometry Result

Date: 2026-09-02

Status: accepted moment-rotation diagnosis; no training state retained

The corrected free-`gln01` audit completed at commit `b7697f5` in
`1,870.963` seconds. Report SHA256 is
`0920927313ecba7378128142cce142c9a66bfe5592fe0cf01a37ab4848b60956`.
The parent recovery remained unchanged and no parameter proposal, optimizer
update, recovery, or checkpoint was written.

The first attempt at commit `cc86e44` completed the full update-2 gradient and
the inherited scale-one forward, then stopped on an unnecessarily fatal
`1e-10` parent-reference comparison. The observed rerun differences are only
`2.2e-10` to `7.8e-9`; they are inside the corrected reporting tolerance
`1e-8`. That first attempt was an engineering-policy failure, not a model or
science result, and produced no report or retained state.

## Direction Diagnosis

The inherited AdamW displacement is not a common source-descent direction.
Cosines between each source gradient and `parameters_before -
parameters_proposed` are:

- direct GPP: `-0.00493710`;
- reachable state: `0.162709`;
- 7-day science: `0.0382239`;
- 30-day science: `0.0441446`.

Thus the update-1 moments rotate the realized update-2 displacement slightly
against direct GPP even though the raw direct-GPP gradient has cosine
`0.245048` with the combined raw gradient. All inherited displacement scales
`1, 1/2, 1/4, 1/8, 1/16` fail the frozen every-source finite-step rule. Smaller
steps cannot repair a first-order GPP direction with the wrong sign.

The zero-moment AdamW displacement is a common source-descent direction. Its
four source cosines are `0.195183`, `0.0587320`, `0.0396964`, and `0.0372591`.
At full scale, direct GPP, reachable state, 7-day science, 30-day science, and
the weighted objective improve `4.04281%`, `2.91495%`, `0.609099%`,
`0.345835%`, and `1.86634%`. Every physical hard count is zero. Full scale and
`1/16` pass; the three intermediate scales fail because their 30-day loss
increases. The finite response is therefore non-monotone, and ordinary scalar
backtracking is not supported.

All four source gradients and all 15,960,195 combined gradient values are
finite and every trainable leaf is nonzero. The 93,504,164 fixed parameters
and source checkpoint remain unchanged. Peak live JAX device allocation is
approximately 3.99 GB; the larger `nvidia-smi` value is the allocator reserve.

## Decision

The update-2 rejection is attributed to inherited AdamW moment rotation, not
gradient clipping, model capacity, nonfinite values, physical failure, or a
demonstrated architecture defect. This is positive evidence for the coupled
GPP/carbon-state candidate, but not a science checkpoint.

The next gate is to freeze an optimizer-reset bounded-training amendment. The
simplest supported policy is fresh AdamW moments for every macro update, with
the same four-source objective, `2e-6` learning rate, clip threshold, data
schedule, pre/post finite-step gate, recovery ownership, and milestone
monitor. It must first pass a discarded free-GPU consecutive-update admission.
Paid continuation, fresh 96-window validation, 365-day rollout, Gate D2,
sealed test, and product acceptance remain closed.
