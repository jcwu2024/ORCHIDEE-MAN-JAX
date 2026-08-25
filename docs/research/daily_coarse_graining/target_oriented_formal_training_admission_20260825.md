# Target-Oriented Formal Training Admission and Submission (2026-08-25)

## Status

The formal target-oriented update-1-to-250 admission is accepted and Slurm job
`14664525` has completed. The scientific result is recorded separately in
`target_oriented_formal_training_250_result_20260825.md`.

This report accepts the prepared assets, the retained-update runner, and its
free-GPU admission. It is not the update-250 scientific result.

## Executable Identity

- branch: `research/daily-coarse-graining`;
- executable commit: `e363482a75cbec3d6eaaa46555cf7191f3f55350`;
- accepted 250-update prepared manifest:
  `runtime/outputs/training/target-oriented-prepared-250-9b82b5a/prepared_manifest.json`;
- prepared-manifest SHA256:
  `0b1e2c97419fe48ff9ce286173a2de25a20b5c038ae98b9c48aace33285dfcd3`;
- accepted 96-window validation manifest:
  `runtime/outputs/training/target-oriented-validation-9b82b5a/prepared_manifest.json`;
- validation-manifest SHA256:
  `1c9081ef4d61da93dd94c8b0856549d570c09b99f61550ffef4b0452b6f5186a`.

The prepared manifest has status `passed`, exactly covers updates `1..250`,
contains one retained executable trace signature, and totals 840,481,740
bytes. The validation manifest has status `passed`, contains exactly 96
artifacts totaling 142,113,251 bytes, and passed independent per-file SHA256
verification. Both products declare no v5 use, no sealed-test read, and no v6
Teacher regeneration.

## Admission Repairs

Free-node admission exposed and closed two launcher/runtime defects before
paid submission:

1. CentOS coreutils rejected the compound `timeout` value `5h50m` before
   Python started. Commit `44c2c4d` replaces it with portable `290m`, ten
   minutes inside the five-hour Slurm wall clock.
2. The first complete six-cell initialization reached report finalization but
   failed because JAX reconstructs dictionary pytrees in sorted-key order.
   Commit `e363482` validates registry membership as a set and emits results
   in frozen contract order. A reordered-registry regression test now covers
   that JIT behavior.

Commit `4f77bf7` also binds the formal Slurm launcher explicitly to the
accepted `9b82b5a` prepared assets. The executable repair therefore does not
require regeneration of any training or validation artifact.

## Passing Free-GPU Admission

The passing admission ran on free `gln01` GPU 0 with output
`runtime/outputs/training/target-oriented-training-admission-e363482`.
`report.json` has SHA256
`518761228e3c8880534289b9b238cccc5b5ce410f77d39b2a2c2d6813f4fa377`
and status `completed`.

- backend: GPU (`cuda:0`);
- parameter count: 83,755,173;
- one retained update completed;
- loss: `328.612140988695`;
- unclipped gradient norm: `5202019.0`;
- frozen clip threshold: `5323655.8449`;
- clip scale: `1.0`; the update was not clipped;
- parameter-step norm: `0.0034562365617603064`;
- all contract-defined training hard-failure counts: zero;
- all source/family gradient nonfinite counts: zero;
- recovery was written, reloaded, and verified at update 1;
- six initialization and six post-update 7/30-day cells completed with zero
  contract-defined hard failures;
- cold elapsed time: `5584.1492` seconds, including compilation;
- observed host RSS remained below the requested 32 GiB, and observed GPU
  allocation remained below the V100 32 GB limit.

The six-cell macro changed from `0.7500948009426053` to
`0.7500637163850813`, a relative improvement of only `0.004144%`. This is an
engineering admission observation, not a scientific decision: admission uses
one window per slice/horizon cell and deliberately sets
`full_frozen_panel=false`. Its `classification.status=rejected` therefore
cannot promote or reject the candidate.

Declared-dynamic state-mask mismatches are recorded diagnostically and are
not contract hard failures. Nonfinite predictions/states, unexpected static
mask mismatches, discrete mismatches, budget failures, and negative stocks
were all zero.

## Completed Paid Job

Slurm job `14664525` started on `ibc13b03n02` at
`2026-08-25T02:30:17` and completed with exit code `0:0` after `02:57:58`:

- partition: `gnall`;
- allocation: 1 node, 4 CPUs, 1 V100 GPU, 32 GiB host memory;
- wall-clock limit: 5 hours;
- worst-case additive charge: CPU CNY 1.40 + GPU CNY 11.00 = CNY 12.40;
- work directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`;
- output root:
  `runtime/outputs/training/target-oriented-training-formal-e363482`;
- log: `runtime/logs/target_training_250_14664525.txt`.

Slurm reported `TRES=cpu=4,node=1,billing=4,gres/gpu=1`. The startup log
confirmed commit `e363482`, formal mode, GPU device 0, the accepted prepared
assets, and a V100 32 GB device. The job wrote both recovery generations,
`stage-update-0250.pkl`, `best.pkl`, `progress.json`, and `report.json`.

## Next Operation

The formal report has status `completed`, all 250 updates, zero contract hard
failures, and both full 96-window panels. Its macro improved `28.4812%`, above
the `5%` pacing threshold, but the absolute 7/30-day gate failed. The active
operation is therefore review and implementation of the explicit
update-251-to-1000 checkpoint continuation described by
`target_oriented_formal_training_250_result_20260825.md`. Sealed-test access,
365-day rollout, Gate D2, and product acceptance remain closed.
