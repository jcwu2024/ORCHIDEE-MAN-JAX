# Shared-v3 persistent-output forward result (2026-09-21)

Free gln01 GPU1 wrapper46671 completed exit0 in673.68s main-test time.
Immutable158192f, protocol shared_transformer_admission_v3, seed20260921,
fresh75,578,497 parameters, zero updates. Environment check separately
passed67.73s. No paid job or new Teacher data.

Matched comparison uses the same v2 rows, seed, loss/objective and frozen
science scales. Eight persistent readouts now use day-start plus full-TRAIN
delta coordinates; bare cimean follows current annualCa. Loss weights,
canonical absolute targets and neural capacity are unchanged.

| Quantity | Row819 v2 | Row819 v3 | Row982 v2 | Row982 v3 |
| --- | ---: | ---: | ---: | ---: |
| Total loss | 676.137143 | 0.456601 | 434.227872 | 0.479491 |
| State-change anchor | 4054.495809 | 0.435095 | 2602.689260 | 0.219685 |
| Process-label loss | 1.327280 | 1.304747 | 0.637545 | 0.616835 |
| Carbon flux group | 0.045846 | 0.045846 | 0.161221 | 0.161221 |
| Carbon stock group | 0.454038 | 0.454038 | 0.858993 | 0.858993 |

Rows: artifact819, landpoint307.0-095.0/year1961/start86; artifact982,
landpoint137.0-087.0/year1963/start171. All hard counters zero on both:
finite state/output/science, water/carbon budget, stock nonnegativity,
non-dynamic masks, discrete states and valid science terms. Carbon science
terms are exactly unchanged, as are canopy/carbon/snow/soil/thermal/water
state groups. The corrected `other` group falls28379.44->1.01370 and
18217.81->0.52073. Former fuel/dead-leaf/cimean extremes no longer dominate.

This is initialization/output-coordinate repair, not learned or long-term
accuracy. Row982 PFT14 NPP remains -0.45614 against Teacher1.39334 in native
units; growth respiration is0.04447 against0.54185. Remaining state errors
include vegetation cover, root/water distribution, thermal derived fields
and annual moisture tracking. Do not start another broad output audit or
change loss weights merely because untrained errors remain.

Forward allocator peak557,975,808bytes (~0.520GiB), distinct from reserved
pool27,263,710,208bytes (~25.391GiB). This is not revised gradient/update
memory; prior v1 gradient memory is not a v3 measurement.

## Next gate launched

Free GPU1 was idle before launching wrapper100848 from the same immutable
158192f checkout. Main bound120min plus launcher's separate5min runtime check.
This allows full reverse, directional JVP comparison and two AdamW updates
without repeating the earlier60min truncation; original-transform admission
plus recovery took about95min combined. Compiled-gradient ceiling stays24GiB.
Both tokenizer and shared-backbone gradients must be finite and nonzero.
Completed gradients are saved before JVP compilation. This is free bounded
admission, not production training.

Report/log prefix:
`runtime/outputs/diagnostics/shared-persistent-158192f-admission`.
After a pass, implement the shared T7/T30 route prepared in the v3 plan,
using native checkpointed AD and exact three-group weights. No paid training
or final-model claim follows from these two-row checks.

Local evidence: `outputs/shared-persistent-158192f-forward.json`,
`outputs/shared-persistent-158192f-comparison.json`, and
`outputs/compare_shared_persistent_forward.py`.
