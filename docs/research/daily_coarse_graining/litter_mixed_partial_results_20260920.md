# Balanced litter mixed-learning: first attempt and bounded recovery

## First attempt: timeout, not completed scientific assessment

Immutable1391a75 ran on free gln01 GPU1 from2026-09-20 14:51:12CST
until the90min timeout (exit124). Original log is
`runtime/outputs/diagnostics/litter-mixed-admission-v1.log`; the unmodified
last report is preserved as `litter-mixed-admission-v1.first-attempt.json`.
The raw report still says running because timeout interrupted the process;
the process is gone and the shell exit status is authoritative.

Both frozen-source TRAIN calibration batches completed and their full gradient
bundles were saved. The old-loss arm completed20 updates and saved2/20;
the selected-weight arm logged5 updates but saved only2. Updates3-5 in that
arm are not recoverable weights and must be recomputed from saved2.
Neither final daily assessment nor the initial/final16-window recurrent
development assessments ran. No full scientific acceptance is available.

All hard counters across the25 logged mixed updates are zero. Each logged
update also passed finite full-parameter/AdamW and optimizer-step checks.
This establishes bounded joint-update execution, not completed rollout
admission or a scientifically accepted model. Existing budget counters still
measure endpoint bookkeeping, not independent ecosystem carbon-flux closure.

## Timing and recoverable work

| Work | Seconds |
| --- | ---: |
| Daily backward compilation |104.59|
| T7 checked backward compilation |2094.54|
| T30 checked backward compilation |2112.49|
| Old arm warm updates3-20, mean |12.83|
| Selected arm warm updates3-5, mean |12.69|

The two recurrent compilations consumed70.12min. The original90min estimate
was inadequate for compilation, both training arms and all monitoring.
Compilation is the dominant first-attempt wall cost; it must not be described
as slow steady-state optimizer updates. No OOM or numerical exception appears
in the log. Compiled estimated live memory is14.49GiB daily and2.23GiB per
recurrent graph; these are executable estimates, not observed process peak.

At16:27CST both new `jit_objective` persistent cache entries existed in
`runtime/cache/jax/orcjax_gpu/fresh-start-one-day-v1`, approximately24.6MB
each, written15:32 and16:07. Exact-checkout recovery can attempt reuse;
cache existence alone is not evidence of a successful cache hit.

## Coefficient and preliminary learning evidence

The declared TRAIN-only rule selected lambda=0.00005221716892123007.
The weighted new-gradient/old-mixed norm ratios are0.04855 and0.95145 on
the two calibration batches. Their median is0.5 and neither exceeds1.
Old-mixed norms are15.8283/3.2216, unweighted litter norms14717.37/58700.98.
The new term therefore no longer dominates both calibration batches as
weight0.1 did. This limited two-batch calibration is not a general bound
on all future samples or a scientific accuracy guarantee.

At matching training batches for updates2-5, selected-arm unweighted litter
losses are3.50%,7.88%,12.49%,18.04% below the old arm. These are losses
evaluated BEFORE each update, after1-4 prior matched updates; they are not
physical-unit RMSE, fixed-panel improvement or generalization evidence.
T7/T30 losses show small changes in both directions. Do not infer overall
GPP/NPP/biomass improvement from these partial records or compare successive
different batches as a learning curve.

## Reviewed bounded recovery

After inspecting the timeout, saved artifacts and cache entries, started one
free45min recovery on still-idleGPU1 at16:28:10CST. This is completion of
the unchanged experiment, not a changed loss, new coefficient search or paid
extension. Same immutable1391a75 and same output prefix; new log
`litter-mixed-admission-v1.resume1.log`. Wrapper62014, timeout62028,
Python62065; deadline about17:13CST. Metadata/report recovery passed.

Reuse both saved calibration bundles and the old20 checkpoint. Restore the
selected2 checkpoint and perform18 updates to selected20, saving before
monitoring. Then evaluate source4000/old20/selected20 on the identical daily
and full16-window7/30day development panels. Accuracy gate failures remain
reported scientific outcomes, not numerical runtime failures. No automatic
further extension or paid submission. Do not change this running checkout.
