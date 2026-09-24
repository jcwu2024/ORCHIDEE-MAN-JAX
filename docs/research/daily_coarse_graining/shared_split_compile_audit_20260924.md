# Split shared model: compilation and source-cost audit

Status: source/read-only audit, 2026-09-24. The new cache-reused split profile
is running separately; this note does not claim its result. No architecture,
math, loss, training route, or paid job was changed.

## What is measured

The admitted split model has 64,884,481 trainable parameters (backbone 768x8,
uniform decoder 384x2). The first cold full-update attempt spent 2,849.259s
on daily micro4 and 5,442.468s on T7; T30 was interrupted. Recovery then
observed daily/T7 persistent-cache hits (zero backend compiles) with complete
trace/lower/load stages of 347.6s/572.0s, but cold T30 still took 5,304.27s
for that stage. The completed second update took 16.473s wall: daily32 4.538s,
T7 2.014s, T30 8.140s. These are one warm changing-data observation and
separate startup observations, not sustained 8,591,565-row throughput.
Sources: `shared_split_timing_recovery_20260924.md` and
`outputs/shared-split-1c70fa4-full-timing.cache.json`. The latter records
compiler-boundary times 46.43s/156.84s for hits and 4,817.56s for cold T30;
these are narrower than complete startup stages. Interrupted backend work
cannot be resumed internally.

## Source expansion boundary

`shared_transformer_raw_outputs` loops over all 161 fields in Python, invoking
`_decode_field` for each; each field has its own padded coordinate chunks and
`lax.map` body containing two attention/FFN decoder blocks and readout
(`shared_transformer.py:1332-1418`, `:1141-1329`, `:537-607`). The fields are
not a 161-step runtime `scan`. Native reverse therefore sees a large staged
daily transition; the exact optimized-HLO size and compiler time attributable
to this staging are not established by source inspection alone.

The public recurrent loss places that transition inside a checkpointed
`lax.scan` (`rollout_loss.py:79-92,129-154`); the same objective factory serves
T7 and T30 (`shared_recurrent_training.py:152-240`). The learning runner
constructs one daily micro4 reverse and separate T7/T30 reverse executables
with `jax.value_and_grad(..., has_aux=True)` and `.lower(...).compile()`
(`run_shared_learning.py:327-384`; `shared_daily_microbatch.py:64-96`). Different
horizon shapes require distinct whole-executable cache entries. This is a
source-backed graph boundary, not proof that the decoder alone causes the
88-91-minute cold recurrent compile.

The retained adapter template is constructed from the first prepared artifact's
schedule entry (`shared_transformer_runtime.py:309-323`,
`run_target_oriented_full_step_preflight.py:122-159`) and closed over by
`make_canonical_step`/`bind_daily_operator_dynamic_retained_inputs`
(`canonical_rollout.py:26-54`, `model.py:136-160`). Day-specific retained
inputs are already dynamic. Different bootstrap artifacts are a possible
source of captured-constant/cache-key variation, **not a demonstrated cause**
of the historical admission-versus-learning miss. The completed recovery
proves same-route cache reuse for daily/T7; T30 cross-process reuse remains
unmeasured in the cited result.

## Decision from existing efficiency evidence

Do not retry a whole-field outer map. The width768 synthetic proxy's scalar
map was 58.1% slower with much larger compiler temporaries, while batch4
failed pointwise gradient checks (`shared_execution_cost_profile_20260923.md`).
Do not promote the actual block-major dense packing fragment: at micro4 it
saved only 5.30% component time and raised temporary memory 38.49%, without
a full-model result (`shared_dense_stage_admission_20260923.md`). Pooling
contraction saved under 1ms per microbatch in isolation. These results do not
prove every normalized split-width grouping impossible; they remove the basis
for another expensive full compile of those particular variants.

The highest-leverage **conditional** compilation revision is stable cache
identity across legitimate TRAIN artifact bootstraps: keep genuinely static
owner/config in the closure, but pass any schedule-dependent array leaves of
the retained template as typed dynamic inputs. Preserve exact adapter values,
shapes, masks, forward/gradient semantics and fresh initialization. This is
worth implementing only if the new cached profile plus a bounded two-bootstrap
cache-key comparison shows a miss with otherwise identical graph configuration.
It could prevent repeated multi-hour cold compiles across future segmented
runs; it does not reduce the first cold compile or the measured 16.473s update.
For one uninterrupted run, the current runner already reuses its three fixed
executables across changing samples, so no warm-throughput gain is claimed.

## Bounded measurement and go/no-go

1. Consume the separately running cached profile; require explicit hit events
   for all three components. The offline
   `scripts/dev/summarize_shared_split_profile.py` reads its three optimized
   HLO files and daily-micro4/T7 Chrome traces without JAX or Teacher data.
   It reports instruction/computation counts, GPU categories, summed/union/
   span time, <=10us events, and conservative source buckets. Shared generic
   linear/FFN helper locations and conflicting HLO names remain ambiguous.
   Optimized HLO counts cannot assign compile time to a source module.
2. Compare existing admission/learning bootstrap descriptors and cache-key
   evidence first. Only if a template-dependent miss is reproducible, make a
   small dynamic-leaf prototype and compare two otherwise identical bootstraps:
   output/mask, complete parameter and state gradients, physical counters,
   cache hit, lowering/loading time, actual memory and warm full-update time.
   Hold architecture/loss and all static shapes fixed; stop before backend
   compilation on a miss during this diagnostic. Do not relax tolerances.
3. If template identity is not causal, close that candidate. Use the new
   measured T7/daily source and kernel mix to nominate one execution boundary
   with a bounded actual-helper prototype before any full-model compile.
   The earlier 65,006-event width768 daily trace is useful fragmentation
   evidence, but not a controlled split-model timing attribution.

The full TRAIN release has 8,591,565 eligible rows; 500 updates at daily32
do not traverse it. Compiler amortization and changing-data update throughput
must therefore be budgeted separately from unique-row exposure. This note
does not authorize a long training run or paid allocation.

Evidence tier: task-directed graph search/trace and exact source reads, plus
the named accepted reports. Codebase graph generation 2026-09-24T12:39:36Z
reported no recorded gaps for cited code paths but marked their filesystem
metadata changed; exact source was read before the claims above. This is not
an exhaustive source or compiler audit.
