# Continuation v3: seven retained updates, stopped at update 42

GPU job 14991510 completed with Slurm exit 0:0 after 02:13:44. The application
finished its stop/report lifecycle, not the planned 40-update experiment.
Updates 35-41 were retained; update 42 was rejected for `gradient_score` and
updates 43-74 were not attempted. Last accepted recovery is update 41.
Duration-based additive charge estimate: (2 + 13/60 + 44/3600) *
(2.20 + 8*0.07) = CNY 6.15, not the requested CNY 33.12 ceiling.

## Stop evidence and limits

All seven retained updates have 96 passing streamed/production score comparisons
each (672 total). Attempt 42 passed its train-only proposal criterion but failed
one of 96 score comparisons, at horizon 30, zero-based callback ordinal 239:

- Observation-path linearized objective: 0.02459177191733883.
- Production-path linearized objective: 0.024385581647048654.
- Difference: 0.00020619027029017412, about 0.8455% of production.
- Existing tolerance: atol=1e-6, rtol=1e-5.

This compares forward objective values used by the derivative calculation; it
is not itself a finite-difference test proving parameter gradients wrong. The
cause remains unresolved. Do not call this ordinary loss oscillation, OOM, a
proven architecture failure, or harmless rounding. Do not relax the tolerance
without explaining the discrepancy. The stopped attempt never replaced the
accepted update-41 recovery. Existing continuation logic returns immediately
from a stopped state, so unchanged resubmission would not resume useful work.

Next diagnostic: reproduce only this update-41/window combination on a free
GPU, align observation and production daily forward outputs/masks/statistics,
and identify the first divergent day/component before testing the implicated
derivative. Resolve its artifact/window from the saved update-42 schedule and
callback ordering. Do not rerun full training or a broad architecture search.

## Learning and physical-unit results

Fixed train-split monitor direct loss improves 5.5984%, science loss 4.0853%.
The seven updates consume 280 distinct package IDs spanning 1961-2004; this
does not represent a full v6 epoch or the proposed larger data pool.
The reused 24-window development panel improves its normalized science macro
1.3702%. Relative transfer passes; absolute scientific acceptance still fails.
Physical hard gates pass. No sealed test or new model inference was performed
for this review: the report already contains both validation trajectories.

Daily free-run GPP RMSE across 444 day positions improves 2.33039 -> 2.30039
gC m-2 day-1. Teacher-state GPP RMSE improves 1.74793 -> 1.70676. These are
paired errors, not an additive causal decomposition of feedback error.

The following values use twelve 30-day windows, equal weight per window.
Flux cumulative RMSE and biomass net-change RMSE are in gC m-2 land.
The common day-zero state makes endpoint stock error equal net-change error.

| Metric | Update 34 RMSE | Update 41 RMSE |
| --- | ---: | ---: |
| Cumulative GPP | 65.9055 | 64.8228 |
| Cumulative NPP | 45.5019 | 44.7650 |
| Cumulative maintenance respiration | 5.1488 | 5.0896 |
| Cumulative growth respiration | 17.2434 | 16.9588 |
| Cumulative autotrophic respiration | 20.7640 | 20.4310 |
| AGB net change | 29.4153 | 29.0914 |
| BGB net change | 18.6226 | 18.2379 |

Thirty-day mean Teacher/model NPP sums are 23.7668/51.6668 (model previously
52.8802); GPP sums are 91.2187/132.2601; maintenance respiration sums are
57.1171/59.9094; growth respiration sums are 10.3349/20.6839. Progress is small
and productivity/growth errors remain substantial. Existing stock-level gates
do not establish long-term biomass-growth accuracy.

All numerical physical-progress review criteria pass for the retained endpoint,
but this is a shortened seven-update result. It does not satisfy completion of
the planned stage or clear its unresolved computational mismatch. No 365-day,
D2, product acceptance, full-method acceptance, or new paid training follows.

## Artifacts and parallel coverage preparation

Remote root:
`runtime/outputs/training/conditioned-snow-free-continuation-v3-14991510/`.
`recovery.pkl` holds accepted update 41; `report.json` includes the rejected
attempt and paired curves. Local report:
`outputs/conditioned-continuation-v3-14991510.json`.
Offline physical summary and figure:
`outputs/conditioned-gpp-feedback-14991510/analysis.json` and `gpp_feedback.png`.
Reproduce with:

```text
conda run -n ORCJAX python scripts/dev/analyze_conditioned_gpp_feedback.py outputs/conditioned-continuation-v3-14991510.json - outputs/conditioned-gpp-feedback-14991510
```

CPU data-preparation job 14992138 remains separate and can continue. Keep its
ce7cd79 source checkout unchanged while it runs. Expanded-data training needs
the score discrepancy resolved and a matched update budget. This seven-update
checkpoint cannot be labeled the planned 40-update fixed-pool comparator.
