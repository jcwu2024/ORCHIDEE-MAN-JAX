# Attention Mixed-Point Annealed Continuation Result

Date: 2026-08-16

Status: **completed; candidate not admitted**.

## Execution

Explore1000 job `14569673` completed with exit code `0:0` at commit
`f702696d345833ff2034e08d1a622b52dc7f92d6`.

- Partition and resources: one `gnall` V100, four Slurm CPUs.
- Wall time: `00:35:11` under the approved two-hour ceiling.
- Runtime-prorated charge estimate: CPU CNY `0.16` plus GPU CNY `1.29`,
  CNY `1.45` total. Actual accounting may apply scheduler rounding.
- Training: all 250 updates and 1,000 unique point-years completed.
- Data per update: 32 one-day anchors and four 30-day windows from four
  distinct point-years.
- Learning rate: cosine decay from `2.9815142e-4` to `3e-5`.
- Peak reported device bytes in use: `13,012,724,480` (about 12.1 GiB).
- Teacher regeneration, historical v5 use, sealed-test access, 48-record
  forcing reconstruction, and a 48-step candidate state scan: all false.

The final report is
`runtime/outputs/training/successor-attention-mixed-annealed-f702696/report.json`,
SHA256 `e67ca672...8ed36`. The 200 MiB final Attention state is retained beside
it with SHA256 `59081704...c2d9`; it is diagnostic and not promotion eligible.

## Training Behavior

All updates were finite and applied. Every per-update state, mask, discrete,
stock, and budget hard count was zero.

| Updates | Learning-rate range | Median loss | Median one-day anchor | Median raw gradient norm | Clipped updates |
|---|---:|---:|---:|---:|---:|
| 1-50 | `2.982e-4` to `2.733e-4` | 0.5585 | 0.3782 | 1.0037 | 25/50 |
| 51-100 | `2.723e-4` to `2.065e-4` | 0.5458 | 0.4068 | 0.9356 | 24/50 |
| 101-150 | `2.049e-4` to `1.233e-4` | 0.5091 | 0.3829 | 1.4898 | 31/50 |
| 151-200 | `1.217e-4` to `5.581e-5` | 0.5397 | 0.3935 | 1.3985 | 31/50 |
| 201-250 | `5.482e-5` to `3.000e-5` | 0.5324 | 0.3745 | 0.9057 | 22/50 |

The first-to-last 50-update median loss decreased by 4.7% and the one-day
anchor decreased by 1.0%. Cumulative-flux MSE and bias-squared medians
decreased by 17.7% and 35.6%. In contrast, terminal-stock MSE and
bias-squared medians increased by 47.2% and 34.8%. Gradient clipping occurred
in 133/250 updates and did not decrease monotonically.

This is not a no-learning or numerical-collapse result. The optimizer changed
the learned behavior, but the aggregate objective accepted a tradeoff that
improved cumulative-flux and maintenance behavior while worsening terminal
stock behavior.

## Scientific Acceptance

The unchanged non-test 7/30-day panel produced zero hard failures, but the
candidate failed all three research-primary improvement checks.

| Check | Required | Observed | Result |
|---|---:|---:|---|
| Primary NPP/AGB/BGB/total-live composite | at least 0.5% better | 6.98% worse | fail |
| Each primary key aggregate | no worse | all four worse | fail |
| Improved temporal/spatial/joint 7/30-day cells | at least 4/6 | 0/6 | fail |
| Maintenance aggregate | at most 20% worse | 22.32% better | pass |
| Maintenance individual cells | at most 30% worse | all six better | pass |
| Hard failures | zero | zero | pass |

Primary aggregate normalized RMSE changed as follows:

| Quantity | Update-1000 baseline | Final candidate | Relative change |
|---|---:|---:|---:|
| NPP | 1.78049 | 1.90474 | +6.98% |
| Aboveground biomass carbon | 0.032683 | 0.033514 | +2.54% |
| Belowground biomass carbon | 0.011577 | 0.016049 | +38.63% |
| Total live biomass carbon | 0.033054 | 0.036213 | +9.56% |

The six primary composite changes were temporal 7-day `+4.14%`, temporal
30-day `+8.11%`, spatial 7-day `+11.08%`, spatial 30-day `+33.27%`, joint
7-day `+1.77%`, and joint 30-day `+5.57%`. Positive values are regressions.
Maintenance aggregate RMSE decreased from `0.54737` to `0.42518`.

## Decision

The frozen recipe is rejected. More updates under the same objective and
schedule are not authorized: the completed broad-data pass and real annealing
did not merely fail to improve the primary outputs; they moved every primary
7/30-day cell in the wrong direction while improving maintenance.

This result does not by itself reject process-axis Attention or prove that
backbone capacity is inadequate. It shows that the present combination of
Attention parameters, staged-v3 scalar objective, and continuation policy is
misaligned with the stated research target. The update-1000 checkpoint remains
the incumbent; the final continuation checkpoint is diagnostic only.

The 365-day screen, long-held points, Gate D2, sealed test, and product
acceptance remain closed. Before another network or paid run, discuss with the
user whether to prioritize a research-output-aligned loss repair, a different
backbone/capacity allocation, or a matched combination. Any architecture
change requires explicit discussion first.
