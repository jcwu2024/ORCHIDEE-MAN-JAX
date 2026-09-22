# micro2/micro4 results and training-speed priority

User decision2026-09-23: prioritize sustainable training throughput before
larger data/longer training. Keep the prepared100->500 data plan, but do not
submit it merely because the data audit finished. Unchanged science remains
the default: architecture, effective daily batch32, T7/T30, losses, precision,
full-network AdamW and physical/gradient gates.

## Completed measurements

15329444 micro2 COMPLETED0:0 in01:13:44 onibc13b03n03.
15329445 micro4 COMPLETED0:0 in01:11:51 onibc13b03n04.
Elapsed-time additive cost estimates are3.39 and3.31yuan, total6.70yuan,
not invoices. Source6302071; source100 checkpoint and artifact742 fixed.
Both pass daily scalar, all75578497 gradient coordinates, hard counts,
full parameter/AdamW and joint-loss comparisons at original tolerances.
No retained training updates or new scientific-accuracy claims.

| Measurement | micro2 | micro4 |
| --- | ---: | ---: |
| Matched scalar32 full update, seconds |28.671|28.701|
| Candidate full update median, seconds |23.924|21.457|
| Wall-time reduction |16.56%|25.24%|
| Throughput multiplier |1.198|1.338|
| Compiled live estimate,GiB |11.391|21.015|
| Cumulative actual allocator peak,GiB |13.673|23.292|
| Reserved pool,GiB |25.391|25.391|
| Candidate lowering+compilation,minutes |45.533|44.366|

Each timing is a median of3 warmed full joint updates, after1 warmup.
micro2 replays24.506/23.902/23.924s; micro4 21.975/21.457/21.428s.
Memory peaks include the full probe process, not an isolated candidate-only
increment. The reserved pool is not actual live usage. micro4 is the faster
admitted candidate so far, not a final choice while8/16 are still running.

At collection15329519/15329520 micro8/16 remain RUNNING at~34min, both in
candidate compilation. Their fit/speed results are unknown. micro4's23.3GiB
actual peak means larger sizes cannot be assumed to fit or be faster; let
the already authorized probes produce their bounded results.

## Where time now goes

Representative micro4 measured replay: daily32 7.772s, T7 2.610s,
T30 11.011s, optimizer0.022s, complete update21.457s. The corresponding
scalar daily part was about15.0s. Daily batching nearly halves that component,
but T7+T30 still cost13.62s (~63% of the full update). Even hypothetical zero
daily cost would leave that serial cost. Microbatch growth alone cannot deliver
arbitrary whole-update speedup.

All3 reference graphs loaded from persistent cache in each paid job, with0
backend compiles. Cache hits still incur tracing/lowering and executable load.
For micro4, reference lowering totals789.83s and loading324.74s, before warm
measurement; candidate lowering+compilation adds2661.95s. Do not describe all
startup time as a cache miss or GPU arithmetic. The XLA slow-compile warning
was followed by completion, not a numerical exception or OOM.

## Next speed work, before larger training

1. Collect8/16 and choose an admitted execution point from measured full-step
   time and memory. Do not infer optimum from batch size alone. Preserve
   compiled candidates and existing reference results.
2. Make stable-graph reuse and startup amortization explicit in the training
   path. Stop repeating expensive reference work as part of unrelated probes.
   Persistent caching is now proven for reference graphs, but avoidance of
   tracing/lowering across processes is a separate optimization question.
   Inspect deployable executable reuse only where supported by actual JAX;
   do not promise zero startup or invalidate caches with gratuitous variants.
3. Revisit two-device execution with the new component timings, using shared
   losses/update functions. The previous default daily vs(T7+T30) split is
   imbalanced under micro4. An alternative(daily+T7) vsT30 has idealized loads
   about10.38 vs11.01s, compared with7.77 vs13.62s for the older split. These
   are arithmetic lower bounds excluding synchronization, transfer, placement
   changes and memory. They justify a bounded load-balance comparison, not a
   measured1.95x speedup. Changing-data full updates and cost per update must
   be measured before making dual-device training the default.
4. Integrate the selected path into a small changing-data training segment
   and report actual end-to-end updates/hour, daily draws/hour, windows/hour,
   compilation/IO/monitor/checkpoint time andyuan per fixed amount of work.
   A fixed-batch kernel result is insufficient for a whole-epoch budget.

Do not reduce recurrent loss frequency, shorten horizons or change science
weights merely to report a faster step. Those are training-method experiments
requiring separate efficacy evidence. Do not restart FP64 investigations
without a newly measured remaining precision bottleneck; reuse completed work.
New paid allocations still require concrete approval. Existing8/16 jobs remain
authorized; no new paid job was submitted during this result collection.

Raw evidence: `outputs/shared-staged-15329444.json` and
`outputs/shared-staged-15329445.json`; remote matching logs and JSON under
`runtime/outputs/diagnostics/`. CLI readout uses `variants.micro2/micro4` and
`reference_graphs`. Follow-up learning coverage/preparation remains documented
in `shared_learning_extension_plan_20260922.md`.
