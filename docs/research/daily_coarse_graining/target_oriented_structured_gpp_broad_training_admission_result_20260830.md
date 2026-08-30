# Structured-GPP Broad Training Free Admission Result

Date: 2026-08-30

Status: accepted; paid stage-1 resource approval is the next gate

Executable commit: `1573eb9695f57e55a98e215e12877fdc7d2b2e0e`

Admission report SHA256:
`94a43cbe947c57cff58f3813c18acd8d3f49b8829319a271fe5b85bd2111cc70`

Recovery SHA256:
`eed4aca8436deab002541eaaaec57f6919712e90880a31ca1db051f2691f2e92`

## Scope

The free `gln01` admission loaded the frozen step-498 specialist, rebuilt the
accepted broad schedule, evaluated the fixed 128-point-year and 2,048-day
train-only monitor, and executed exactly two chained broad updates. It consumed
416 unique train/train days. This is runner qualification, not a scientific
training result and not evidence that the model generalizes.

No Teacher data were regenerated, no v5 data were used, and no validation,
sealed-test, consumed 96-window, 7/30-day, or rollout target values were read.
No checkpoint is promotion eligible.

## Result

The initial monitor contained exactly 128 records and 2,048 evaluated values;
its aggregate direct-GPP loss was `0.8287614855`. Update 1 had loss
`1.0086134715`, raw gradient norm `23.019043`, clip scale `0.911761`, and
parameter-step L2 `0.0142237`. Update 2 had loss `0.6253785331`, raw gradient
norm `11.190650`, clip scale `1.0`, and parameter-step L2 `0.00960747`.

Both updates were finite, nonzero, and covered the full specialist. All
93,504,164 retained parent parameters remained exactly unchanged. The initial
one-day physical gate had zero hard failures; its 19 declared-dynamic mask
differences remain the established reported nonfailure count.

The run wrote a recovery bound to optimizer step 2. JAX peak live memory was
about 2.42 GiB even though its allocator reserved substantially more GPU
memory. Total elapsed time was `657.94 s`; the hot second update took `7.77 s`.

## Decision

The runner, direct-only data path, chained AdamW update, monitor, physical gate,
and real recovery write are accepted for the bounded paid stage. Submit no paid
job until the user approves the exact `gnall` resources, five-hour wall time,
working directory, command, and CNY 13.80 worst-case additive estimate.
