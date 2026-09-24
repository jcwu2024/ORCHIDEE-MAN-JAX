# Shared split throughput scale: arithmetic scenarios

## Evidence and scope

This is a throughput accounting note, not a training-method change or a model
performance claim. It uses the preserved raw result
`outputs/shared-split-1c70fa4-full-timing.json`: the second of two changing-data
updates measured daily32/micro4 at4.538s, T7 at2.014s, T30 at8.140s,
optimizer at0.025s, and complete update wall time at16.473s. That single warm
observation corresponds to218.5 updates/hour and6,993 daily row draws/hour.
It is not a sustained average. The measured component sum is14.742s; the
remaining1.731s to wall time is left unattributed, not assumed to be IO.

The source data counts supplied for this calculation are8,591,565 TRAIN daily
rows,12,208,581 total daily rows including holdouts, and346,539 rows in the
currently ready daily pool. Holdouts are not part of the TRAIN denominator.
The output preserves counts separately and makes unique date/point/year
coverage unknown; row draws may repeat.

Reproduce the arithmetic with:

```powershell
conda run -n ORCJAX python scripts/dev/estimate_shared_training_scale.py `
  --raw-report outputs/shared-split-1c70fa4-full-timing.json `
  --output outputs/shared-training-scale-20260924.json `
  --train-rows 8591565 --total-rows 12208581 --ready-daily-pool 346539 `
  --planned-updates 500
```

The generated JSON contains all single/dual/three-device arithmetic cases for
daily32/64/128, recurrent-every-2/every-4 average-work cases, and a daily-only
example. Each scenario records TRAIN draw-equivalent days at its selected
arithmetic rate; this is repeated-draw arithmetic, not unique coverage. It
contains no prices or currency estimates.

## What the scale says

1. At the one observed warm baseline,500 updates take2.288 hours of update
   work, before startup, IO, science panels, and variability. They draw16,000
   daily rows:4.62% of the ready pool by count, or0.186% of TRAIN rows. Neither
   fraction establishes unique coverage.
2. Matching the ready-pool count with32 draws/update takes10,830 updates, or
  49.56 warm hours at the observed wall time. Matching the full TRAIN row count
   takes268,487 updates, or1,228.57 warm hours. These are exposure-equivalent
   arithmetic, not time-to-epoch or time-to-unique-coverage guarantees.
3. Ideal two-device placement of daily work against recurrent work has a
  10.179s/update lower bound at batch32; ideal three-device placement of daily,
  T7, and T30 has an8.165s lower bound. For500 updates these imply1.414/1.134
  wall hours but2.827/3.402 allocated GPU-hours, respectively. The bounds
  ignore transfer, synchronization, imbalance, contention, compilation, and
  launch overhead; no dual/three-GPU speed has been measured here.
4. The serial batch64/128 calculations assume daily compute scales linearly
  with rows:21.012s/30.088s per update including the baseline unattributed
  remainder. These cases raise the arithmetic row-draw rate to10,965/15,315
  per hour, but change the sampled minibatch and gradient estimate. They are
  not equivalent to the measured daily32 updates. With microbatch4 retained,
  larger logical batches do not inherently increase activation memory; actual
  admission should check retained inputs/gradient storage and runtime. Gradient
  accumulation earns no speed credit: computing the same32 sample contributions
  still does that work.
5. At batch128, ideal two- and three-device bounds both reach about18.18s/update:
  daily work, rather than recurrent work, is the bottleneck, so a third device
  has no arithmetic benefit in this placement model. At batch64 the third
  device lowers the ideal bound from10.18s to9.10s. This is a candidate for a
  bounded execution test only if communication and GPU-hours make sense.
6. Recurrent-every-2/every-4 cases average9.640s/7.102s per update at batch32;
  the daily-only example is4.563s. These reduce or omit recurrent objectives
  and therefore change the optimization schedule, not just execution
  placement. They are not recommendations or equivalent speedups. Recurrent
  steps remain full-cost spikes in a less-frequent schedule.

## Practical ranking

1. Preserve and budget cold-start cost first: the measured T30 compile stage
  was5,304s (88.40min, about1.47h), roughly64% of the2.288h for500 warm
  updates. This is a substantial but comparable startup cost, not larger than
  the warm-update total. The warm arithmetic is not a job wall-time budget.
  Reuse of unchanged fixed-shape executables must be verified for the actual
  runner.
2. For the unchanged joint objective, first pursue execution/compilation
  improvements that preserve batch32 and all recurrent evaluations. The dual
  ideal bound is a ceiling to test, not an expected result; a third GPU has
  smaller marginal value once daily work dominates.
3. Larger daily batches may improve row draws per hour on the linear model, but
  they alter the minibatch objective. With fixed microbatch4, activation memory
  is not assumed to rise; measure retained-input/storage requirements and
  runtime, and do not equate them with faster daily32 training.
4. Keep recurrent-frequency and daily-only schedules in a separate scientific
  decision. Their arithmetic cannot establish that recurrent supervision or
  its long-term effect is dispensable.

## Limits

Only the warm single-device daily32/micro4 joint update is observed. The script
scales daily time linearly with row count, keeps recurrent component times
fixed, and reports ideal overlap lower bounds without a measured multi-device
execution model. It does not infer unique dates, points, years, row replacement
semantics, scientific adequacy, retained-input or gradient-storage fit for a
changed logical batch, compilation compatibility, long-run throughput, or
paid-resource cost. The fixed-microbatch activation footprint is not assumed to
grow merely because more microbatches are accumulated. Startup and
validation costs remain outside warm update projections. No architecture,
model default, or training schedule is changed by this report.
