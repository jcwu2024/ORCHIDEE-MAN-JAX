# Four-arm litter learning: completed500-update results

## Execution and conclusion

Array15290789_0..3 completed normally, all exit0:0, immutable2918966.
All four arms completed500 NEW updates (absolute4100->4600), with all three
100/250/500 assessments and checkpoints. Technical/physical monitoring passed;
no science guard stopped a run. Scientific acceptance remains false.

| Arm | Scheduler elapsed | Change from A |
| --- | --- | --- |
| A | 02:10:23 | Original settings |
| B | 02:10:43 | Litter weight divided by4 |
| C | 02:10:43 | T30 outer weight doubled |
| D | 02:11:27 | LR1e-5->3e-6 |

At8CPUs*.07+1GPU*2.20=2.76yuan/hour per task, recorded task durations imply
24.07yuan total, below33.12 approved ceiling. This is a tariff estimate, not
a billing receipt. No task remains active and no additional paid job submitted.

Main result: T7 and carbon-stock-change errors improve, but all four final
T30 GPP/NPP/RA/AGB/BGB errors exceed their common selected100 starting values.
The favorable100-update production results did not persist uniformly. There
is no accepted model or evidence that simply extending the current inherited
chain will meet the scientific goal. This does not prove the architecture or
fresh training method cannot succeed.

## Evidence and physical-unit comparisons

Local source files: `outputs/litter-parallel-v1-{A,B,C,D}.json`.
Remote reports: `runtime/outputs/training/litter-parallel-v1/{A,B,C,D}/report.json`.
Scheduler logs: same parent directory, `slurm-15290789_0.log` through`_3.log`.
Frozen method: `litter_parallel_training_protocol_20260920.md`.
All arms inherit SAME selected100 parameters/AdamW;32000 daily draws and500
T7/500T30 windows each, same ordering. No new Teacher data or architecture.

RMSE units below are gC/m2: cumulative fluxes and stock change from the common
Teacher initial state. Eight development windows per horizon; the T7 andT30
panels are different windows, so the difference between horizon aggregates
alone does not isolate trajectory-length effects. These are repeatedly used
development panels, not an independent generalization test.

| T30, new update | GPP | NPP | RA | AGB change | BGB change | Canopy change | Nonliving change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Common start | 5.7692 | 4.5888 | 1.6329 | 3.1382 | 1.4652 | 12.0304 | 903.0992 |
| A100 | 3.0355 | 2.9196 | 1.8189 | 1.9189 | 1.0178 | 18.5332 | 818.1218 |
| A250 | 14.1505 | 10.1415 | 4.1505 | 6.8054 | 3.3073 | 12.9581 | 631.0891 |
| A500 | 7.6574 | 5.6763 | 2.4510 | 3.8241 | 1.8446 | 8.5154 | 629.7216 |
| B100 | 4.70 | 4.17 | 1.34 | 2.76 | 1.44 | 6.43 | 867.98 |
| B250 | 16.84 | 12.47 | 4.49 | 8.51 | 3.92 | 22.34 | 753.54 |
| B500 | 8.64 | 6.65 | 2.41 | 4.42 | 2.21 | 11.79 | 788.48 |
| C100 | 3.75 | 3.27 | 1.27 | 2.06 | 1.22 | 11.05 | 825.60 |
| C250 | 11.34 | 8.79 | 3.20 | 6.22 | 2.53 | 8.48 | 766.17 |
| C500 | 8.18 | 6.31 | 2.39 | 4.30 | 2.00 | 8.52 | 787.31 |
| D100 | 7.17 | 5.90 | 1.67 | 4.05 | 1.85 | 9.51 | 855.79 |
| D250 | 12.01 | 8.96 | 3.20 | 5.91 | 3.03 | 8.51 | 758.67 |
| D500 | 8.82 | 6.34 | 2.64 | 4.17 | 2.16 | 6.77 | 697.33 |

| T7, new update | GPP | NPP | RA | AGB change | BGB change | Canopy change | Nonliving change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Common start | 2.8137 | 2.1744 | 0.7462 | 1.6787 | 0.5037 | 2.5143 | 58.1776 |
| A500 | 0.91 | 0.73 | 0.43 | 0.52 | 0.23 | 2.77 | 26.50 |
| B500 | 1.82 | 1.77 | 0.41 | 1.30 | 0.47 | 2.07 | 39.73 |
| C500 | 1.25 | 1.16 | 0.49 | 0.84 | 0.33 | 2.77 | 33.12 |
| D500 | 1.65 | 1.40 | 0.36 | 1.03 | 0.38 | 3.00 | 33.98 |

A has the lowest final GPP/NPP/biomass/nonliving errors among these four, but
does not dominate every metric and does not meet joint scientific acceptance.
C has slightly lower finalT30 RA; D lower canopy. Reducing litter weight,
doubling T30 weight, or lowering LR did not establish an overall solution in
this inherited-state experiment. Do not call a difference statistically robust
from one seed and eight windows.

## Localized errors from saved trajectories

Direct per-window recalculation of finalT30 GPP and nonliving RMSE agrees with
the reported aggregates for all four arms. No new forward inference was used.

For A,97.90% of T30 cumulative-GPP squared error comes from TWO windows at
landpoint189.0-093.0 (out of eight windows):

| Year/start row | Teacher30-day GPP | A500 GPP | Signed error |
| --- | ---: | ---: | ---: |
| 1983/24 | 115.7827 | 132.4526 | +16.6699 |
| 1979/188 | 89.2960 | 102.7634 | +13.4674 |

Both are overpredictions, not a current universal GPP-collapse signal.
The same site's two windows contribute70.68%/89.17%/94.58% of squared GPP
error for B/C/D. Compared with the starting checkpoint, GPP absolute error
worsens in4/8 A windows,5/8 B,6/8 C and5/8 D; other windows can improve.

Within the SAME A1983T30 window, signed daily GPP error is+0.1852 on day1,
averages+0.3404 over days1–7 and+0.8221 over days24–30. The1979 window starts
at+0.2261 and averages+0.4169/+0.4916 over those intervals. This provides
concrete time-resolved error evidence, but cannot by itself separate changing
forcing/direct head error from recurrent-state feedback.

For every arm, one different window,141.0-075.0/year2000/start192, contributes
about97.65–97.72% of finalT30 nonliving squared error. Teacher stock change is
+404.5798; A predicts-1356.0707, B-1800.0269, C-1795.9423, D-1544.8284gC/m2.
Thus the residual stock drift is still severe despite lower aggregate RMSE.
Across all eightT30 windows, Teacher mean change+125.2697; predicted means
A-163.2766,B-237.1002,C-237.7670,D-197.6139. Wrong-sign counts A4/B6/C5/D5,
versus5/8 at the common start. Do not hide wrong direction behind a lower RMSE.

## Decision and remaining limitations

Collective numerical execution succeeded; simultaneous scientific accuracy
did not. Current small daily TRAIN panels also cannot establish broad fit or
generalization. Nonzero RH remains untested on the current zero-target monitor;
budget bookkeeping is not independent ecosystem carbon-flux closure.

Follow `fresh_method_verification_plan_20260920.md`: prepare a complete
fresh-initialization/full-network recipe, including retained scientific losses
from the declared start, instead of another default continuation sweep.
The unchanged A settings are the least-modified reference candidate, not an
accepted optimum or a reason to exclude other fresh-start strategies.
Freeze at most two justified recipes before new training; use shared data
ordering and an independent initialization repeat as planned. Actual fresh
coefficient admission, independent panel, runtime and paid budget remain open.

Use the localized difficult windows for transparent diagnostics and coverage
checks, not special-case fitting or deletion. Confirm on a broader preregistered
development/validation selection under the existing split policy; leave the
sealed test untouched for routine selection. No automatic long run, new
architecture, Teacher regeneration, or paid extension follows this report.
