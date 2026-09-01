# Coupled-GPP Bounded Training Admission

Date: 2026-09-02

Status: accepted free-GPU production-equivalent runner admission

The corrected `--admission-only` execution completed on free `gln01` GPU 0 at
commit `885693a` in `1,579.743` seconds. Report SHA256 is
`4aaac8caa552f7ff90c037bfaf7020d7522883ea7262aca885372dd7b51b8e23`.

The runner evaluated the fixed eight-artifact train-only monitor, then executed
the first complete scheduled eight-artifact macro proposal. That update covers
1,664 direct-GPP Teacher days, 256 reachable one-day state transitions, 64
seven-day windows, and 32 thirty-day windows. The proposal was evaluated on
all eight artifacts before it was admitted and then discarded.

The direct-GPP, reachable-state, seven-day science, and thirty-day science
macro losses decrease `1.60249%`, `10.41690%`, `1.02810%`, and `1.32569%`.
The weighted objective decreases `2.52874%`, from `0.1877068332` to
`0.1829602144`. The combined gradient norm is `3.48236`, below the frozen
`12.80819` clip threshold, so clip scale is exactly `1.0`.

All four source gradients and the combined gradient are finite over all
15,960,195 trainable values. Every one of 190 parameter leaves is nonzero, and
both trainable families change. All true physical hard counts remain zero;
declared-dynamic mask counts are reported diagnostics and are not failures.
The complete source checkpoint and all 93,504,164 fixed parameters remain
unchanged.

Peak live JAX allocation is approximately 3.80 GiB. After compilation, the
seven hot gradient artifacts take `57.12-57.57` seconds each, and post-proposal
forward checks take approximately `16.65-18.69` seconds each. The runner wrote,
loaded, and verified one temporary recovery, then deleted it. No optimizer
update, parameter state, recovery, or checkpoint was retained.

This accepts the production runner, finite-step gate, and recovery ownership.
It is not a science checkpoint or promotion result. The exact paid envelope
for the first retained eight-update stage is one `gnall` node, eight allocated
CPUs, one V100, 32 GiB scheduler memory, and a five-hour wall limit. At additive
rates its worst-case charge is CNY `2.80` CPU plus CNY `11.00` GPU, CNY `13.80`
total. The user pre-authorized this next training request before admission.
