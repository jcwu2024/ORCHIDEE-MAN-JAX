# Target-Oriented Full-Step Preflight Acceptance

Status: **accepted eight-batch zero-update gradient calibration and physical
preflight; formal training-runner implementation is next**.

Date: 2026-08-24.

## Execution

Approved Explore1000 job `14659152` ran commit
`bca102cd48827e33cdaae3a420f4bce384abb0d7` on one V100 with four CPUs and
32 GiB requested host memory. It completed normally in `00:27:00` and retained
no optimizer update or checkpoint.

The report is
`runtime/outputs/training/target-oriented-full-step-preflight-bca102c/report.json`.
Its SHA256 is
`7bddd58b21221a5c2e625dd08a36292f0e4ce3daa548e69fc298412fda688de5`.
The report has schema `target_oriented_full_step_preflight_v1` and status
`passed`.

## Gradient Calibration

All 83,755,173 combined parameter-gradient values are finite in every batch.
All seven connected parameter families are finite and nonzero in every batch.
The eight unclipped norms are:

```text
5202018.8299
5064208.4001
5607475.5467
4824426.3713
4675236.9935
4930027.9703
4240109.8824
5062018.1212
```

The frozen train/train 90th-percentile clipping threshold is
`5323655.8449`. The finite AdamW proposal uses learning rate `4e-7`, has clip
scale `1.0` on batch eight, parameter-step norm `0.00346267`, and zero
nonfinite parameter or optimizer-state values. The proposal is diagnostic and
was not retained.

The one-day/process, 7-day, and 30-day source gradients are finite and nonzero
where connected in every batch. Their complete norms vary over
`[4.2401e6, 5.6075e6]`, `[0.7387, 6.8844]`, and `[0.5217, 2.9931]`.
The long-horizon gradients consistently dominate the direct primary science
heads but are roughly seven orders of magnitude below the one-day state anchor
in the shared encoder/refinement/attention/mixer families at initialization.
This is a stable conditioning observation across eight runtime years, not a
single-batch anomaly. It does not make the long-horizon heads disconnected,
and AdamW's coordinate-wise normalization means raw norms are not parameter-
step ratios. Formal training must nevertheless report whether long-horizon
loss and shared-family influence improve during warmup.

## Physical And Resource Gates

Every batch has zero budget, discrete-state, negative-stock, unexpected-mask,
prediction-nonfinite, and state-nonfinite failures. All observed mask
mismatches are the declared dynamic `rveget` status. Scaffold-year invariance
is exact.

Peak live JAX device memory is `9,600,627,712` bytes, below the frozen 28 GiB
cap. Slurm measured 17.48 GiB peak host RSS. After the first cache-populating
batch, the seven hot effective steps span `36.7530` to `37.0007` seconds with
median `36.8113` seconds. The complete job used about 0.45 GPU-hours; under the
project's additive rate model its approximate charge is CNY 1.12.

The earlier 2.5-hour eight-batch projection incorrectly treated single-batch
cold compilation time as recurring execution time. The measured hot-step
distribution above replaces that estimate for training resource planning.

## Decision

The repaired target-oriented candidate passes the bounded-training entry
preflight. This is a trainability and physical-validity result, not evidence
that the untrained model meets 7/30-day scientific targets.

The next operation is local implementation and free-node admission of the
formal training runner: exact effective-batch updates, frozen clipping
threshold, recovery records every 50 updates, stage checkpointing, and the
frozen non-test validation panel at initialization and update 250. No paid
training, 365-day rollout, Gate D2, sealed-test access, or product acceptance
is opened by this preflight alone.
