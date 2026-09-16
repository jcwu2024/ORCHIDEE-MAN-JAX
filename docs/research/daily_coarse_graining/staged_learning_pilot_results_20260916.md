# First matched-compute learning segment: results

Both paid arms completed normally. Learning is substantial, but neither has
full scientific acceptance. Joint is more accurate on development T30 GPP,
NPP and biomass RMSE; staged is better on canopy/nonliving carbon and daily
monitor accuracy, with smaller mean T30 production/biomass biases. This is a
tradeoff in an early learning segment, not a settled ranking of final methods.

## Execution and evidence

- Frozen method: `staged_learning_pilot_frozen_20260916.md`; immutable0d18d3c.
- Array15225197_0 joint600: COMPLETED0:0,2:46:36, ended14:50:17 server time.
- Array15225197_1 daily400+mixed400: COMPLETED0:0,2:49:00, ended14:52:41.
- Both began2026-09-16 12:03:41, each1V100/8allocated CPUs. Additive charge
  estimate `(9996+10140)/3600*(2.20+8*.07)` =15.44yuan, not a billing receipt;
  approved maximum22.08yuan. No paid extension or new submission.
- Both independently initialized109,464,375 parameters from the declared seed;
  all trained, no inherited exploratory checkpoint. Recovery replay passed.
- Every recorded update and monitor/window physical hard check passed, including
  budgets, nonfinite values, negative stocks, masks and discrete state.
  This does not imply scientific accuracy: both `scientific_acceptance=false`.
- Remote reports: `runtime/outputs/training/mixed-learning-stages-pilot-v1/`
  `{joint,staged}/report.json`, with full checkpoints retained beside them.
- Local copies: `outputs/staged-pilot-joint-600-report.json` and
  `outputs/staged-pilot-daily400-mixed400-report.json`.
- Read-only extraction: `outputs/summarize_staged_pilot.py`; derived
  `outputs/staged-pilot-comparison.json` and `outputs/staged-pilot-learning-curves.png`.
  No new model inference was needed. Luna/high independently audited hard
  checks, data counts, compute, clipping and gate status against the raw reports.

Actual update compute was7641.73s joint and7689.41s staged, only0.62% apart.
Staged used2617.08s daily and5072.34s mixed. Mean complete update costs were
12.736s joint,6.543s staged daily,12.681s staged mixed. Python elapsed totals
9920.92/10065.59s include startup, monitors and IO. This supports the intended
matched-compute comparison more directly than the original fixed-data estimate.

Executed data: joint38400 daily draws and600 windows per horizon; staged51200
daily draws and400 windows per horizon. No padding. These overlap and are not
additive unique coverage counts. Neither is an epoch. Do not mistake the
5415-row `data_plan` metadata for executed consumption; use `data_consumption`.

## Fixed-development physical accuracy

All checkpoints used the same16 development windows,8 atT7 and8 atT30.
The initial physical predictions match exactly across arms. This is a reused
development panel, not final independent acceptance. RH Teacher values are
zero here, so zero RH error cannot prove learned nonzero respiration.

The table reports RMSE in gC/m2. Flux rows compare30-day accumulated flux;
stock rows compare day30 inventory (equivalently change error from identical
initial states). These are errors, not percentages and not mean signed biases.

| Quantity | Fresh initialization | Joint600 | Daily400+mixed400 |
| --- | ---: | ---: | ---: |
| GPP |63.083|8.699|13.287|
| NPP |54.640|7.470|11.047|
| Plant respiration RA |10.003|3.475|3.246|
| Aboveground biomass carbon |37.238|5.015|7.573|
| Belowground biomass carbon |17.207|2.525|3.446|
| Canopy organic carbon |81478.757|2418.468|362.115|
| Nonliving ecosystem carbon |88076.997|8969.425|6006.950|

For the64 excluded-TRAIN daily monitor samples, GPP RMSE changed3.0648 to
0.9422 joint and0.6700 staged, in gC/m2/day. Their common daily objective fell
to0.9634 and0.1591, respectively; fixed-training daily losses were0.39984 and
0.07505. These common monitor losses are comparable. Daily-only and mixed
update losses themselves are different objectives and must not be ranked.

## Full learning curves and remaining drift

Matched-compute checkpoints are joint0/100/200/400/600 versus staged0/200/400/600/800.
The plot uses measured cumulative update seconds, including actual rotating
data costs; all checkpoints are shown. Axes are logarithmic for readability.

| Arm/update | Update hours | T30 GPP RMSE | T30 NPP RMSE | T30 RA RMSE | Canopy RMSE | Nonliving RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint0 |0|63.08|54.64|10.00|81478.8|88077.0|
| Joint100 |0.359|58.55|47.84|12.08|81474.2|88071.6|
| Joint200 |0.711|49.28|37.97|12.26|81327.1|87922.7|
| Joint400 |1.420|16.53|8.93|7.67|29907.2|36504.3|
| Joint600 |2.123|8.70|7.47|3.48|2418.5|8969.4|
| Staged0 |0|63.08|54.64|10.00|81478.8|88077.0|
| Staged200 |0.366|54.79|46.14|10.24|81367.6|87964.5|
| Staged400 |0.727|42.48|37.60|6.70|33820.7|40166.0|
| Staged600 |1.432|16.75|13.87|3.88|2419.6|8442.6|
| Staged800 |2.136|13.29|11.05|3.25|362.1|6007.0|

Thus both routes improve late in this segment on theseT30 quantities. The
joint respiration error initially worsened before improving; staged canopy
learning accelerated during daily-only updates. Neither endpoint establishes
a plateau or guarantees further training will converge. Staged T7 NPP is a
counterexample to blanket monotonic claims: theT7 science panel passed at600,
then failed NPP at800. Do not hide this by selecting the favorable checkpoint.

Final existing key gates: jointT7 passes, jointT30 fails GPP/NPP/NEE/NEP and
ecosystem/nonliving carbon. StagedT7 andT30 each fail NPP. Gate counts do not
rank total accuracy: jointT30 NPP has lower RMSE but more mean bias, whereas
staged has lower bias and higher RMSE. Canopy diagnostics must be considered
separately from the existing key-gate inventory. Normalized gate errors use
fixed acceptance scales, not the local Teacher mean.

Mean quantities over the8 T30 windows (gC/m2):

| Quantity | Teacher | Joint600 | Staged800 |
| --- | ---: | ---: | ---: |
| Accumulated GPP |52.3565|57.6380|54.4097|
| Accumulated NPP |17.2429|22.6376|18.9624|
| Accumulated RA |35.1136|35.0004|35.4473|
| Aboveground biomass change |1.1731|4.5454|2.3506|
| Belowground biomass change |0.1528|2.1331|0.6853|
| Nonliving carbon change |125.2697|-3995.7604|-2525.3781|

Stocks begin from Teacher initial states, so their large initial inventories
can make endpoint relative error look small despite wrong growth rates. Both
arms predict the wrong AGB change sign in3/8 windows. Both still lose too much
nonliving carbon. Mean bias also permits cancellation across windows; retain
per-window errors and RMSE. These are material obstacles to long-term use,
despite substantial improvement from random initialization.

All updates clipped at norm threshold1. Median multiplier: joint4.52e-4;
staged daily3.35e-4 and staged mixed1.24e-2. This describes optimizer behavior,
not a standalone proof of undertraining or an instruction to change clip_norm.

## Next decision

Keep the architecture and admitted loss/AD route. There is demonstrated
learning; these data do not support a new architecture or elimination of
either schedule. Preserve both complete states. Before another paid segment,
freeze its scope, additional data coverage, compute budget, checkpoints and
interpretation rule. A longer mixed-training segment should examine whether
the still-improving curves continue, whether growth/nonliving-carbon drift
shrinks, and whether the current tradeoff persists. Do not declare a winner
or a final method from this one seed/small development panel. A larger
development panel is a useful independent check; no365-day/product acceptance
or additional paid submission has been performed here.

Free dual v4 independently FAILED exit1 after1328.9s while this analysis was
being collected. The newly scopedGPU1 T7 validator compiled and executed;
the subsequent unscoped gradient-norm helper loadedGPU0's `jit__lambda_`
cache entry and rejectedGPU1 buffers. Python110215 has exited. Dual speed/parity
remain unavailable. This does not invalidate either single-GPU learning run.
Next free debugging should cover all cross-device helper compilations in a
complete tiny update, rather than another validator-only test/retry. No retry
was launched in this reporting turn. Evidence `outputs/mixed-learning-dual-profile-v4.json`.
