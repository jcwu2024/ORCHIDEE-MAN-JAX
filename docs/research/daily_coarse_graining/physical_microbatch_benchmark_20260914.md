# Batch64 microbatch8/16 resource probe

User approves implementation and bounded free-node checks. Keep the admitted
physical-v2 objective, Attention, all-network fresh seed20260911, AdamW and
clip1.0 unchanged. This is a batching/throughput check, not a new loss, learning
budget, multiday optimizer admission or paid training authorization.

Each size uses the same64 unique TRAIN rows: concatenate the first two fixed32
draws of the accepted expanded sampler. Microbatch8 and16 are alternative
partitions of those exact rows. Each process compiles the admitted vmapped
reverse objective, executes one full warmup update, saves its mean gradient
and parameter displacement in one diagnostic container, then discards that
update and resets to initial parameters/optimizer. Three sequential timed
updates reuse the same64 rows. All resulting weights are discarded.

Compare loss, complete gradient and actual AdamW displacement at the common
initial state; use the separate CPU comparison script with declared tolerances.
This detects material numerical differences from vmap and accumulation order
without repeating the expensive, unchanged scalar JVP checks. Every actual
update checks finite gradients/parameters/optimizer and physical hard counts.
Do not select by total loss decrease or demand that rounding be bitwise equal.

Report explicit compilation and warmup times separately. Timed updates include
warm-cache reader calls, transfer, synchronized gradient evaluation/accumulation,
physical counts, clipping and optimizer execution. Large host finiteness checks
are separately timed; report both compute and checked-total times. Three updates
provide a bounded throughput estimate, not a sustained global-data benchmark.
Record GPU high-water live allocation separately from JAX reserved pool. The
reported high-water includes warmup and diagnostics, so label it benchmark peak.

JAX pool fraction is0.85 of32GiB; unchanged launchers default to0.65. This leaves
device space for runtime allocations while allowing larger microbatches. Stop
before execution if compiled estimated live exceeds24GiB; actual resource errors
remain resource outcomes, not gradient/science failures. One free GPU/process,
60min cap each, no paid fallback. CPU full-array gradient/optimizer shapes and
actual target-Bash dry runs precede launch. Reuse successful deployed runtime
and physical-v2 AD evidence; no Teacher data preparation or regeneration.

Prefer the faster numerically admitted configuration that leaves adequate
measured headroom for later mixed training. A small/no speed gain favors the
lower-memory option. Microbatch8-vs16 alone does not prove either faster than4;
report that limit. Multi-day windows need their own later batch/memory choice.
After this short probe, return to combined single-day+T7 optimizer admission,
then T30 and broader reproducible learning; do not let batching become an
open-ended infrastructure workstream.
