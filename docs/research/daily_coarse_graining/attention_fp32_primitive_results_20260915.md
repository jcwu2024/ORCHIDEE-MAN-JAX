# FP32 Attention: completed bounded V100 comparison

The implicit FP64 score path is a measurable local cost. Explicit FP32 scaling
passes all five source-shaped primitive comparisons and reduces forward plus
parameter/input-gradient median time by 4.81--14.99% on V100. This admits a
real-model precision comparison, not a production precision change or a claimed
whole-update/RTX5090 speedup.

## Execution and scope

- Code: `43e039a`; free `gln01` GPU0, Tesla V100-SXM2-32GB, JAX 0.4.38.
- Completed normally, exit0, 26.85 seconds; ten-minute ceiling, no paid job.
- Entry: `scripts.hpc.probe_attention_fp32 --source-shaped --repeats 51`,
  through `scripts/hpc/run_fresh_start_one_day.sh`.
- Global x64 enabled; highest matrix precision, no TF32/FP16; seed20260915.
- Same synthetic activations, initial parameters and masks in both modes.
  Shapes derive from source fixtures, not captured trained activations or a
  measured distribution of executed calls. No Teacher reads or model updates.
- At least0.5 seconds of alternating workload warmup, then51 synchronized,
  alternating paired repetitions per case. Compilation is excluded from timing.
- Raw result: local `outputs/attention-fp32-v100-source-v2.json`; remote
  `runtime/outputs/diagnostics/attention-fp32-v100-source-v2.json` under the
  canonical project root. The primitive process has exited.

## Timing

Shape notation is `(groups, query tokens, key tokens, width)`; all use8 heads.
Times include the Attention projections as well as softmax/value aggregation.
Reductions are `(old time - new time) / old time`, not throughput percentages.

| Source-shaped family | Shape | Forward old/new, ms | Forward reduction | Forward+grad old/new, ms | Reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| Forcing | (1,45,45,256) | 0.1365 / 0.1333 | 2.31% | 0.3408 / 0.3105 | 8.87% |
| Axial | (126,32,32,384) | 0.9677 / 0.8488 | 12.28% | 2.3159 / 2.0553 | 11.25% |
| Source expert | (1,1,533,384) | 0.2146 / 0.2027 | 5.55% | 0.3539 / 0.3369 | 4.81% |
| Maintenance | (1,24,419,768) | 0.4647 / 0.3843 | 17.30% | 0.9335 / 0.7935 | 14.99% |
| Carbon feedback | (1,2,2386,384) | 0.4698 / 0.4581 | 2.51% | 1.0110 / 0.9322 | 7.79% |

Gradient-call p10--p90 intervals (ms) are old/new respectively:
forcing0.319--0.360/0.293--0.326; axial2.307--2.327/2.048--2.073;
expert0.345--0.359/0.328--0.348; maintenance0.927--0.941/0.786--0.803;
carbon1.004--1.037/0.918--0.960. Small cases still contain launch/runtime noise;
the larger axial/maintenance cases show a clearer separation. These are sample
spreads, not confidence intervals. Do not average the five percentages into a
model-wide gain because actual call frequency, fusion and batch shape differ.

The earlier `76e4b83` representative-shape run completed81.14 seconds and passed
all five numerical comparisons, but its15-repeat timing had initial warmup drift.
Prefer this source-shaped, warmed51-repeat panel for the local timing claim.

## Numerical evidence

All five output/loss comparisons, full parameter/input gradient comparisons and
masked-input checks pass; zero failed gradient coordinates. Output relative L2
differences are1.64e-7--2.74e-7; aggregate gradient relative L2 differences are
1.53e-7--2.95e-7. A singleton query case tests masked keys without masking its
only query; the other cases also test a completely masked query.

Each old forward lowering contains5 FP32 and1 FP64 matrix results, versus6 FP32
and0 FP64 for the candidate. Forward+gradient changes from15 FP32/3 FP64 to
18 FP32/0 FP64. These are static operation counts, not FLOP or time fractions.
Three local unit tests additionally cover dtype promotion, gradients/masks and
native reverse/JVP duality. Tolerances were not changed to obtain a pass.

## Decision and remaining work

1. First candidate: explicit FP32 ordinary Attention scores/softmax/value
   aggregation. The primitive option exists; full-model callers still default
   to legacy precision. Thread one recorded opt-in precision policy through
   the active head configurations, then compare the same trained weights/data.
2. Additional source candidates: FP64 reporting-only full-tree gradient norms
   and FP64 masked neural-family pooling that immediately returns FP32. Their
   isolated performance gains have not been measured. Optimizer clipping uses
   a separate FP32 norm; do not confuse it with reporting norms.
3. Keep physical reconstruction, conservative budgets, retained daily processes
   and sensitive losses/stocks FP64 in the first comparison. Their convertibility
   requires separate evidence, particularly for small stock changes and recurrent
   feedback. No blanket `jax_enable_x64=False` change.
4. Verify actual one-day and T7/T30 loss/gradient/physical/science behavior, then
   measure complete-update wall time. Preserve existing reference/cache evidence;
   new full gradient graphs may incur long cold compilation, unlike this probe.
5. RTX5090 remains unmeasured. Reuse the probe with `--backend gpu` on its actual
   runtime, followed by the full workload. V100 local gains do not predict the
   balance of FP64, memory, kernel-launch and CPU costs on5090.

This is an efficiency result, not improvement in fitted GPP/respiration/biomass.
The separate checked recurrent execution result (14.82% lower update time) must
not be added to these local Attention percentages.
