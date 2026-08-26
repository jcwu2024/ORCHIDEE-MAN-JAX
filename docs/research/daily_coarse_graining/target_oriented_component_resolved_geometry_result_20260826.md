# Component-Resolved Flux Geometry Result (2026-08-26)

## Decision

The zero-update component-resolved geometry audit passes its engineering
gate, but it rejects an unchanged AdamW continuation and rejects the raw mean
of the eight amended-loss batches as a trustworthy training direction.

All five declared gradient sources are finite, nonzero, and cover every one
of the `83,755,173` Attention parameters in all eight train/train groups.
Physical hard failures are zero. No optimizer was invoked, no parameter or
checkpoint was retained, no Teacher data were regenerated, historical v5 was
not used, and no sealed-test value was read.

The scientific result is that the amended flux gradients are highly
heterogeneous across batches. A separate flux-focused virtual-step gate must
find a finite direction that improves the unresolved 7/30-day flux targets
while treating the already successful one-day state and stock objectives as
protection constraints. Do not continue the old aggregate loss or the amended
loss with unchanged batching and AdamW.

## Executable And Evidence Identity

- branch: `research/daily-coarse-graining`;
- successful executable commit:
  `d7cab74eec6f2f7f0e9546895854551e4669cea7`;
- report schema: `target_oriented_component_resolved_geometry_audit_v1`;
- runtime report:
  `runtime/outputs/diagnostics/target-oriented-component-resolved-geometry-d7cab74/report.json`;
- local ignored copy:
  `outputs/diagnostics/target-oriented-component-resolved-geometry-d7cab74/report.json`;
- report SHA256:
  `414aa5433baeab85ef824de34ad93822a035e4ee4c61375cab09e83d05ff2a05`;
- flux-basis amendment SHA256:
  `6c81e8f56c2c81fc2ac024a71da98f3e875c4ebc429f9e3d1ff4e8e3ef6bfb45`;
- update-1000 checkpoint SHA256:
  `a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7`;
- prepared manifest SHA256:
  `8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31`.

The successful free-`gln01` run took `1861.09 s` (`31:01`). Forward work
took `176.86 s`, gradient work took `1331.48 s`, and the report records peak
live JAX bytes of `12,542,651,648`. The persistent cache was reused.

## Failed First Execution And Repair

The first free run at commit `76b7ddb` completed all eight gradient groups,
then failed only in the final batch-variance summary. It stacked eight full
float64, 83.8M-parameter gradients and requested one additional `5.01 GiB`
device allocation while the model and gradient state were resident. It wrote
no report and retained no model state.

Commits `ebab940` and `d7cab74` replaced that summary with a fixed-size
`1,048,576`-parameter chunked Gram accumulation. The largest explicit chunk
is about 64 MiB for eight float64 vectors. A focused equivalence test proves
that chunked accumulation matches the fully flattened float64 Gram matrix.
The repaired run reused the completed compilation cache and passed.

## Engineering Gate

- groups completed: `8/8`, updates
  `251, 358, 465, 572, 679, 786, 893, 1000`;
- gradient sources per group: `5/5`;
- finite values per source: `83,755,173/83,755,173`;
- nonfinite values: zero;
- summed family coverage: `83,755,173` parameters;
- physical hard-failure count: zero;
- optimizer updates: zero;
- checkpoint writes: zero.

The `751` declared dynamic-state mask mismatches are expected declared-mask
events and are excluded from the hard-failure count. Unexpected mask
mismatches are zero.

## Cross-Batch Geometry

| Gradient source | Norm minimum | Norm median | Norm maximum | Pairwise cosine median | Pairwise cosine minimum | Negative pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| one-day state and process | 6706.69 | 7978.34 | 9541.66 | +0.9975 | +0.9937 | 0.0% |
| 7-day component flux | 0.515 | 12.506 | 1713.736 | -0.4739 | -0.9714 | 53.6% |
| 7-day independent stock | 0.000246 | 0.000701 | 0.033710 | -0.0974 | -0.9290 | 53.6% |
| 30-day component flux | 0.777 | 1.717 | 8.917 | +0.0199 | -0.9472 | 50.0% |
| 30-day independent stock | 0.003959 | 0.004609 | 0.007174 | +0.8843 | +0.6343 | 0.0% |

The one-day gradient is stable. The two flux horizons are not. In particular,
the 7-day flux norm at update 465 is `1713.736`, versus a median of `12.506`.
Five of the other seven group gradients point opposite to at least one large
7-day contribution. The mean 7-day flux gradient is therefore dominated by
one effective batch rather than representing a stable central direction.

## Mean Geometry Is Not A Training Admission

The all-parameter norm of the eight-group mean one-day gradient is
`8066.45`. Mean 7-day and 30-day component-flux norms are `211.02` and
`0.6923`; mean 7-day and 30-day stock norms are only `0.00460` and
`0.00477`.

The sum of the two mean flux gradients has cosine `+0.9723` with the mean
one-day gradient, but cosine `-0.9994` with the mean 7-day stock gradient and
`-0.2920` with the mean 30-day stock gradient. The apparent positive
one-day alignment is largely produced by the update-465 7-day outlier and by
the high-norm state path. It must not be read as evidence that a raw averaged
step will generalize across groups.

This also explains why optimizing all five sources as equal unit objectives
would be the wrong response. Biomass and slow-carbon stocks already pass the
formal 7/30-day screen; they are protection constraints, not targets that
should be allowed to steer the update away from GPP, NPP, and respiration.

## Accepted Next Gate

Freeze a separate zero-update virtual-step contract before implementation.
It must:

1. construct a group-aware common direction from the 16 amended flux
   gradients formed by eight train/train groups times 7/30-day horizons;
2. use normalized gradients so update 465 cannot dominate by magnitude;
3. treat one-day/process, 7-day stock, and 30-day stock losses in every group
   as finite-step protection constraints rather than optimization targets;
4. evaluate a preregistered global parameter-step grid directly, without
   AdamW, momentum, adaptive coordinate scaling, weight decay, or clipping;
5. report the four independent flux components and the derived GPP/NPP/
   respiration/NEP/NEE acceptance metrics after every virtual step;
6. retain no proposed parameters, optimizer state, checkpoint, or recovery.

If no shared flux direction or no finite protected step exists, close this
amended raw-loss realization and redesign the robust loss or sampling unit.
Do not respond by enlarging the batch again, shrinking toward an ineffective
step, changing the Attention architecture, or opening paid training.
