# Successor Staged 500-Update Milestone Acceptance

Status: **accepted first staged milestone; continue-to-1000 is eligible but
not yet authorized**.

## Decision

The matched staged-v3 pair completed updates `1-500` and the complete frozen
48-window validation panel. Both arms are numerically stable and conservative,
and both update-500 parameter sets are admitted as the first within-regime
validation baseline.

This milestone does not promote or rank either arm. Its `improved=true` flags
mean that update 500 becomes each arm's first staged best checkpoint because no
earlier staged milestone exists. They do not prove improvement over an
untrained model. The first actual minimum-1% improvement comparison occurs at
update 1000 against the scores recorded here.

## Prepared Input

Paid `cnmix` job `14524758` completed with exit code `0:0` in `03:26:53` using
four one-core workers. The accepted `1-500` product is under
`runtime/outputs/training/successor-staged-prepared-0001-0500-9856dae/`:

- 500 gzip artifacts in exact update order;
- total artifact bytes: `1,661,072,648`;
- manifest SHA256:
  `19d0cfd699c2535a101edbe245fffc23f49a6c514f565bb069b60e5683355375`;
- producer commit: `9856daec8fe05ba2e7a343b62d6ceb0ef35a6ab3`;
- full schedule SHA256:
  `8b14d538516dbad953a30e0be2928d2b1dd2474adf7bf940f4707d177b0d2626`.

Worker peak RSS was approximately 2.8, 4.3, 4.9, and 4.0 GiB. A separate free
`test01` process decompressed all 500 artifacts, recomputed every file hash,
and checked coverage, schedule fields, trace signature, total size, and policy
flags. No Teacher regeneration, historical-v5 read, or sealed-test use
occurred.

## Training Execution

Paid `gnall` job `14525122` completed with exit code `0:0` in `00:52:51` using
four CPUs and one V100. The stage report is under
`runtime/outputs/training/successor-staged-fit-0500-9856dae/`:

- report SHA256:
  `5d2ee5b19143f0d5662e6cad0818b5d82de91f859b84140ad33942da52a5019e`;
- update-500 recovery SHA256:
  `d18e58c99195e32c37d607f43291f911a18162b9a8425688da8ae16710e00936`;
- best-checkpoint SHA256:
  `1c54a43cea7564c2efdc0609a6387d702800fed5ad808c5f3b302315cb0289e8`;
- maximum host RSS: `8,918,104` KiB;
- parameter count: `8,722,689` per arm.

All 500 updates were applied to both arms. Every recorded loss, component,
gradient norm, and parameter value is finite. All training and validation
nonfinite, mask, discrete-state, negative-stock, and budget hard counts are
zero. The candidate does not construct 48 half-hour forcing records or run a
48-step candidate state scan.

Gradient clipping remains active rather than resolved: pre-clip norm exceeded
the threshold on 482/500 attention updates and 474/500 control updates. This is
not a hard failure under staged-v3, but it is evidence that training is not yet
well characterized as converged.

## Validation Result

The update-500 aggregate primary composites are:

| Arm | Composite |
| --- | ---: |
| Graph-MLP control | 1.0457081708 |
| Process attention | 1.0523784009 |

Attention is `0.64%` higher (worse) overall. Its primary composite is also
slightly higher in each of the six slice/horizon cells, by approximately
`0.19-1.12%`. This small consistent difference is retained as evidence but is
not an arm-selection gate.

Both arms have small normalized 7/30-day live-biomass errors on this panel,
while GPP and NPP remain the dominant errors, particularly on the temporal
slice. For example, 30-day temporal normalized RMSE is approximately
`2.35` for GPP/NPP in both arms, whereas total-live-biomass RMSE is about
`0.061`. The scientific objective is therefore not yet accepted despite good
short-horizon stock stability.

A separate `test01` process reloaded the recovery through the production
loader, reloaded the best checkpoint, verified both tree inventories and all
`17,445,378` best-parameter values, and independently checked all 1,000 arm
update records and validation hard counts.

## Cost And Next Gate

Using the documented additive rates, observed execution cost is approximately:

- preparation: `4 * 3.448 h * CNY 0.07 = CNY 0.97`;
- training: `4 * 0.881 h * CNY 0.07 + 0.881 h * CNY 2.20 = CNY 2.18`;
- combined observed estimate: `CNY 3.15`.

The contract makes `501-1000` the next eligible stage. It requires preparing
only that additional 500-update interval, resuming the accepted recovery, and
re-evaluating the same panel. An arm counts as improved only if its composite
falls by at least 1% from the values above with zero hard failures. No
`501-1000` preparation or training is authorized by this acceptance. The
365-day rollout, long-held-point validation, Gate D2, sealed test, and product
gates remain closed.
