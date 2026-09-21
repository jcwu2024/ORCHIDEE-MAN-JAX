# Shared-v3 daily admission and recurrent CPU preparation

Collected 2026-09-22. Both free-node tasks completed with exit0. No paid
task was submitted and no new GPU task follows from this collection.

## Daily numerical admission

Free gln01 GPU1 wrapper100848, immutable158192f, completed the v3 protocol
in3857.82s (64.30min), plus a separate22.80s synthetic runtime check.
Fresh initialization; all75,578,497 parameters trainable. Complete reverse
gradients are finite; both tokenizer and shared groups have nonzero norms.
295 of298 gradient leaves are nonzero on this sample; this does not mean
every individual parameter has a nonzero gradient.

Native forward-mode directional derivative10.0960211494 agrees with reverse
projection10.0960187860: relative difference2.3409e-7 (0.0000234%). This is
one declared normalized-gradient direction on one daily sample, not an
exhaustive derivative or multi-day acceptance.

Both AdamW updates completed. Initial, post-update and final two-sample
replay hard counters are all zero. Optimizer checkpoint roundtrip is exact
and restored parameters give finite, physically admitted final forwards.

| Quantity | Artifact819 initial | After two updates | Artifact982 initial | After two updates |
| --- | ---: | ---: | ---: | ---: |
| Total daily loss | 0.456601 | 0.362859 | 0.479491 | 0.402201 |
| Carbon flux loss | 0.045846 | 0.020348 | 0.161221 | 0.091530 |
| Carbon stock loss | 0.454038 | 0.128432 | 0.858993 | 0.322473 |
| Process-label loss | 1.304747 | 0.514707 | 0.616835 | 0.275165 |
| State-change anchor | 0.435095 | 1.364884 | 0.219685 | 1.310038 |

Total losses fall20.53%/16.12%, flux groups55.62%/43.23%, stock groups
71.71%/62.46%. The state-change anchor increases, and individual stock
terms do not all improve (e.g. artifact982 litter/canopy terms). Therefore
this demonstrates useful local parameter updates and technical admission,
not uniform improvement, held-out accuracy or stable long-term trajectories.
Do not lower anchor weights or reopen architecture solely from two updates.
The next declared multi-day objective tests the recurrent consequences.

Gradient compile estimate is6.672GiB; measured allocator peak7.522GiB,
distinct from the25.391GiB JAX pool. No OOM occurred. Gradient compilation
took2377.6s; the directional-JVP stage took1201.3s including its first
compilation/execution. Together these stages account for about93% of main
elapsed time. This admission runtime is not per-update training throughput.
The two reported update sections took3.06s/0.918s; the first reused its
saved gradient and neither number is a full mixed-training benchmark.

Local evidence: `outputs/shared-persistent-158192f-admission.json`.
Server prefix: `runtime/outputs/diagnostics/shared-persistent-158192f-admission`.
Completed initial gradient and final diagnostic checkpoint remain on server;
they are not a formal pretrained initialization.

## Recurrent CPU preparation

Free test02 wrapper45145, immutable605c941, completed in785.68s (13.09min).
Existing TRAIN calibration/data reused;75,578,497 parameter shapes match.
Joint-daily gradient tracing, four recurrent forward-shape traces and
first-artifact T7/T30 gradient-shape traces pass.

| Artifact | Horizon | Landpoint | Year | Start row |
| --- | ---: | --- | ---: | ---: |
| 819 | 7 | 083.0-105.0 | 1961 | 180 |
| 819 | 30 | 279.0-095.0 | 1961 | 59 |
| 982 | 7 | 081.0-113.0 | 1963 | 252 |
| 982 | 30 | 303.0-107.0 | 1963 | 108 |

These are technical TRAIN samples, not independent validation. No numerical
full-model forward, gradient or optimizer update was run by this CPU tool.
Local evidence: `outputs/shared-recurrent-605c941-cpu.json`.

## Next action

Finish the bounded GPU admission runner for the already implemented shared
daily/T7/T30 objectives and frozen `shared_recurrent_admission_v1.json`.
Use fresh declared initialization, public native checkpointed AD, four
window forward/physical checks, measured memory, directional checks and two
complete joint updates. Save completed expensive results before subsequent
compilation. Reuse passed CPU/data/daily evidence, without repeating100848.
No paid training or long-horizon scientific acceptance is authorized by
these technical results. Bounded learning and independent panels follow
only after the actual joint objective passes its numerical/runtime gate.
