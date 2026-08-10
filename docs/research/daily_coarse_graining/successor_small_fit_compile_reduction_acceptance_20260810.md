# Successor Small-Fit Compile-Reduction Acceptance

Date: 2026-08-10.

Status: avoidable same-process coefficient and final-evaluation compilation
removed and accepted; paid production remains unsubmitted pending a new
two-stage resource approval.

## Bound Evidence

- Implementation commit: `507cb61cefa347405189947aff7a60d11da03dad`.
- Frozen coefficient contract SHA256:
  `b32f6cd0102d515b6f556e5e56b170229b27d913af8c0c2172b1e44c1f5f07ad`.
- Coefficient source report SHA256:
  `481581d1648699e140fa12c1b6a8e342f154ecb2216473f01a6916bd95c31f60`.
- New two-entry prepared manifest SHA256:
  `cd69110078083abe78db6bcb18a5fdd811a85f1dbbd01f822363bcd45f22eccb`.
- Cold `gln01` report SHA256:
  `a9ab43d855a4c035cbc52f978717de9737f801ef349981a19b50e22d2056bdc9`.

No paid Slurm job was submitted. Batch preparation ran as a bounded serial CPU
test on `test01`; the cold timing probe ran on free `gln01` GPU 0. Both GPUs
were `0 MiB / 0%` after the probe. Teacher data were not regenerated,
historical v5 was not read, sealed test data were not read, and no checkpoint
was written.

## Frozen Objective

The four train/train gradient-ratio coefficients from the accepted fixed-seed
report are now an identity- and hash-bound input contract:

| Component | Coefficient |
| --- | ---: |
| cumulative flux MSE | 0.026843412276434775 |
| cumulative flux bias squared | 0.04026511841467285 |
| terminal stock MSE | 0.003129054494004748 |
| terminal stock bias squared | 0.004693581741008572 |

The loader verifies the runner, schedule, candidate, fast-owner, first
prepared artifact and source-report identities, plus the train/train split,
component registry, coefficient policy, bounds and finite positive source
gradient evidence. Both arms receive the same coefficients. Recalibration
during fitting and validation-based weight tuning are forbidden.

The new two-entry prepared artifacts bind the new source HEAD but are
byte-identical to the accepted earlier artifacts:

```text
update-0001.pkl  d0b0fdaee6275546f5d740c670846a6d8e0849b0fa0c1d25ea36afe2987349ee
update-0002.pkl  2e86a482be394773d91af90cd8114f319ba405cf487f248270dc3bf177ceab71
total bytes      66,868,764
```

This confirms that the source change altered runner policy, not training
samples or Teacher content.

## Cold/Hot Result

The new cold process completed in 1,442.46 seconds, versus 1,908.38 seconds
for the fixed-seed prepared-batch baseline: a reduction of 465.92 seconds
(24.4%). The coefficient contract loaded in 0.018 seconds and the report sets
`coefficient_preflight_recomputed=false` and
`coefficient_preflight_seconds=0.0`.

| Stage | Attention | Graph control |
| --- | ---: | ---: |
| First cold compile/update | 581.24 s | 580.47 s |
| Switched-point hot update | 1.208 s | 0.894 s |
| Final zero-rate evaluation | 1.055 s | 0.894 s |
| Final-evaluation new JIT executables | 0 | 0 |

The switched hot entry loaded in 0.227 seconds and replayed in 1.352 seconds.
Both final evaluations call the already compiled training step with learning
rate zero and discard its proposed optimizer/parameter result. They therefore
measure the post-fit objective and hard gates without compiling a separate
evaluation graph.

All four updates across the two arms had finite gradients and applied
successfully. Budget residual, negative declared stock, nonfinite defined
prediction, defined-mask mismatch and discrete-state mismatch counts were all
zero. Update-to-update executable reuse passed. The two-update run is a timing
and integrity probe; it does not evaluate the 500-update promotion thresholds.

## Active Gate

The avoidable-compilation gate is closed. Cross-process persistent
`jit_train_step` cache reuse remains rejected, so a production process must
pay the two approximately 581-second cold compilations once and then keep all
500 updates in that same process.

Before any paid submission, define and obtain explicit approval for two
separate bounded stages:

1. resumable CPU preparation of exactly the frozen 500-entry train/train
   schedule under the new HEAD, with measured memory, finite wall time and
   worst-case CPU charge;
2. one-GPU, same-process 500-update fit consuming only the admitted prepared
   manifest, with the measured cold cost included once, a finite wall time and
   worst-case additive CPU/GPU charge.

Do not produce all 500 artifacts on a test node and do not submit `cnmix`,
`cnall` or `gnall` until that resource block is reviewed and explicitly
approved. The 7/30-day screen, 365-day rollout, long-held points, Gate D2 and
product acceptance remain closed.
