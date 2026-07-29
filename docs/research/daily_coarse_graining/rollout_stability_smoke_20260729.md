# Rollout-Stability GPU and Host-Preparation Gate

Date: 2026-07-29

## Decision

The frozen Experiment B implementation is admitted to paid screening
preparation. Real-shard horizon-1, horizon-3, horizon-7, and rematerialized
horizon-30 V100 smokes pass with:

- two distinct train landpoints sharing one executable;
- finite first and second optimizer updates;
- zero nonfinite parameter gradients;
- zero defined-status, discrete-state, nonfinite-defined-value, and negative
  source-nonnegative-stock hard counts;
- exact checkpoint resume;
- no sealed-test access.

No paid Experiment B job had been submitted when this report was written.

## Gradient Stabilization

The original horizon-1 failure was localized with JAX checkify to inactive
Fortran `IF/WHERE` arms that evaluated `0/0` after vectorization. Source-backed
denominator guards were added in season, allocation, plant age, gap mortality,
turnover, and vmax paths. The final targeted regression result was:

```text
191 passed
18 rollout tests passed
```

## GPU Evidence

Accepted protocol shapes:

| Horizon | Anchor batch | Rollout batch | Hot update | Peak allocator |
|---:|---:|---:|---:|---:|
| 1 | 256 | 256 | 0.063 s | not limiting |
| 3 | 256 | 64 | 0.074 s | not limiting |
| 7 | 256 | 32 | 0.096-0.112 s | about 523 MiB |
| 30 remat | 256 | 8 | 0.205 s | about 436 MiB |

The final horizon-7 and horizon-30 reports are:

```text
runtime/outputs/smoke/rollout-stability-hostprep-b913211/horizon7-protocol-shape.json
runtime/outputs/smoke/rollout-stability-hostprep-b913211/horizon30-protocol-shape.json
```

Their losses, component values, gradient norms, and hard counts are identical
to the pre-optimization `4dc016c` reports.

## Host Preparation

Four bounded changes removed repeated host work:

1. load each compressed shard once per update;
2. reuse one runtime per landpoint across updates;
3. reuse first-runtime forcing instead of rebuilding it;
4. reconstruct only the five forcing leaves consumed after the learned
   fast-day boundary, in one vectorized batch operation.

The reduced retained-tail forcing contains `temp_air`, `precip_rain`,
`precip_snow`, `salinity`, and `tide_height`. Every leaf is exactly equal to
the corresponding full Teacher forcing reconstruction. This does not change
the stored native-forcing network input or the Teacher data contract.

For a steady second landpoint, forcing preparation fell from about 6.75 s to
0.015 s. A complete warm-cache horizon-7 update preparation is 0.245 s. New
landpoints still require about 9-11 s for context construction and about 2 s
for retained-tail static assembly.

The bounded 16-landpoint cache report is:

```text
runtime/outputs/smoke/rollout-host-preparation-cache-30f2bfc.json
```

It measured:

- baseline RSS: 1.55 GiB;
- final RSS after 16 cached runtimes: 2.01 GiB;
- steady RSS growth: about 7.8 MiB per additional landpoint;
- transient peak RSS: 4.97 GiB;
- warm-cache preparation: 0.245 s.

A linear 535-train-landpoint estimate adds about 4.2 GiB, so host memory is
not a gnall-node constraint.

## Screening Estimate

Using measured first-landpoint and warm-update costs:

- 128 train-only calibration batches: about 30 minutes;
- 8,192-update control arm: about 35-45 minutes;
- 8,192-update mixed-horizon candidate: about 2.5-3 hours;
- expected sequential total: about 4 hours.

A six-hour wall limit provides recovery margin. The runner checkpoints every
256 updates and binds protocol, data, schedule, model, environment lock, and
training Git HEAD. It refuses calibration, execution, or checkpoint reuse
after code drift.
