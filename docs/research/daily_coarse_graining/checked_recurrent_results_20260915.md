# Checked recurrent execution: bounded admission passed

Free gln01 GPU0 attempt v3 completed at immutable427ccce, wrapper exit0,
runner2932.49s (48.87min). Bash15683/Python15724 exited. No paid job, retained
parameter updates, new checkpoint or Teacher generation. Evidence is
`runtime/outputs/diagnostics/mixed-learning-checked-profile-v3.{json,log}`;
the JSON is copied locally to `outputs/`.

Both T7 and T30 pass full109464375-parameter gradient comparisons against
preserved original references, using unchanged coordinate and per-leaf L2
tolerances. Zero failed coordinates, equal named physical counters, all hard
counts zero. Loss equality passes at the declared tolerance, not bitwise.

| Horizon | Original loss | Checked loss | Gradient error L2 | Reference L2 |
| --- | ---: | ---: | ---: | ---: |
| T7 | 121.8003196598 | 121.8003208346 | .000289264 | 2634.853903 |
| T30 | 143.0433661250 | 143.0433665221 | .000159642 | 1496.980048 |

Complete fixed-update200 replays keep daily64/micro8, the same first T7/T30
pair, all weights/AdamW state, losses, precision and physical checks unchanged.
Proposals are computed and discarded. The ordinary warm three-repeat median
falls from14.18122s to12.07914s:14.82296% less time,1.17403x throughput. This
passes the declared5% admission target. Complete mixed losses, component and
combined norms and clipping agree within unchanged tolerances across all six
ordinary/instrumented replays; maximum relative scalar/norm difference2.68e-8.

This uses a preserved baseline from another process, explicitly marked
`same_process_timing_comparison=false`. It is a bounded fixed-data/warm-cache
comparison, not a contemporaneous paired benchmark or a guarantee for every
rotating window. Earlier independent original timing was14.0924s, consistent
with a similar practical saving. Peak live memory17126535936bytes (~15.95GiB),
reserved26.98GiB; compiled live estimates stay below18GiB.

T7 reused its compiled cache: `.compile()`86.17s; T30's first checked
compilation2054.24s; daily cached compilation104.09s. The new physical reducer
compiled in.55s/1.76s and retained243418/1043220 observation bytes on device.
These startup times are separate from warm update timing. Both recurrent
compiled artifacts now exist for future reuse in the same runtime.

Decision: the opt-in split execution is admitted on this bounded panel for
subsequent training (`--checked-recurrent --separate-checked-validation`).
The default CLI has not been changed. This improves execution efficiency,
not scientific accuracy; no trained model acceptance is implied. Do not repeat
these reference gradients or turn historical custom-AD discrepancies into
new prerequisites for unchanged next training. Next scientific planning should
address substantial data exposure/learning duration and RA/canopy drift, with
an explicit reproducible method and separately approved paid resource budget.
