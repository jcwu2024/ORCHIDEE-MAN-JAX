# Successor 7/30-day Free-Smoke Acceptance

Status: **accepted execution and integrity gate; not promotion eligible; the
formal 96-window bounded screen is now active**.

Date: 2026-08-11.

## Decision

The frozen three-window `free_smoke` panel completed successfully on the free
`gln01` V100 at evaluation commit
`753db81bd39edf55f17c7cbec069859f2f2b1cda`. It admits the accepted 500-update
dual checkpoint and proves that both frozen arms can execute 30-day free
rollouts through the canonical boundary on one temporal, one spatial, and one
joint non-test window.

This is an execution and numerical-integrity acceptance only. The frozen
contract declares `free_smoke.promotion_eligible=false`; accordingly the
report has `classification=null`. These three observations neither promote
nor reject either arm and do not authorize changing the model, loss, gates,
or deterministic window-selection rule after seeing the values.

## Bound Evidence

- report SHA256:
  `966313db540bf147c339a5c11021536bfe0db0f75362294feb7df0a9ca458e22`;
- prepared-manifest SHA256:
  `7c4a391f16cd7fe45018e6e2dca9855e8152308ea4c5b6831a039bda2597b093`;
- window-plan file SHA256:
  `41fd20af5e4d4c384c5088f434811fa1f42fda14303fe884e199c9160eb4705e`;
- internal plan SHA256:
  `e0b17cf05a0b94478de9439d583aceb8b1e6d5e2b9e1a8628c76c22bbee1e566`;
- fit report SHA256:
  `ea4b66cd2d09d20f5ac4754de461382d560e4feedca5d52e211bebe77a4a259a`;
- checkpoint manifest SHA256:
  `f8eed186a90c2f5ecad39c6f9114b343fb29ab468266fed39dfe70e353275e40`.

The report is archived locally at
`outputs/diagnostics/successor-7-30-free-smoke-753db81/rollout_screen_report.json`
and remains on Explore1000 under
`runtime/outputs/training/successor-7-30-free-smoke-753db81/`.

The run took `225.4896` seconds. Each arm compiled exactly one executable on
the first artifact and added none on the second or third. The first-artifact
times were `54.166` seconds for attention and `49.808` seconds for control;
all later per-artifact evaluations were `0.108-0.176` seconds.

## Integrity Gates

All five hard counts are zero for both arms on all three slices:

- nonfinite defined prediction;
- negative declared stock;
- defined-mask mismatch;
- discrete-state mismatch;
- water/carbon budget-residual failure.

Both arms completed all three windows. The run used five native forcing
records per day and 30 daily-operator steps per window. It did not reconstruct
48 half-hour forcing records, execute a 48-step state scan, use cross-day
neural hidden state, regenerate Teacher data, read historical v5 data, or open
the sealed test split. Optimizer state was not loaded for evaluation.

## Observed Scientific Metrics

Primary-composite normalized RMSE is shown below. Negative relative change is
better for attention; positive change is worse.

| Slice | Day | Attention | Control | Attention relative to control |
| --- | ---: | ---: | ---: | ---: |
| temporal | 7 | 8.767397 | 0.819250 | +970.17% |
| temporal | 30 | 3.438392 | 0.810322 | +324.32% |
| spatial | 7 | 0.235987 | 0.319454 | -26.13% |
| spatial | 30 | 0.208736 | 0.205038 | +1.80% |
| joint | 7 | 1.606567 | 1.726989 | -6.97% |
| joint | 30 | 1.664671 | 1.823226 | -8.70% |

The temporal attention error is driven chiefly by normalized NPP RMSE
`21.2547/8.31527` and maintenance-respiration RMSE `27.5371/9.59307` at Day
7/30, versus control values `0.723799/0.545696` and `2.07759/2.33362`.
Temporal total-live-biomass RMSE is also larger for attention:
`0.129884/0.213743` versus `0.004428/0.014238`.

The spatial window is much closer: attention improves the Day-7 composite and
is 1.8% worse at Day 30. Attention improves both joint composites, although
both arms have large joint DOC, litter, and ecosystem-carbon normalized errors.
Every per-key normalized RMSE, signed bias, mean absolute error, and evaluated
count is retained in the hash-bound JSON report.

Because each cell contains only one window, its RMSE equals its absolute
signed bias. The temporal signal is serious enough to preserve and test, but
three windows cannot estimate population-level model-selection performance.
Only the frozen 32-window-per-slice formal panel may classify the arms.

## Next Gate

Materialize the deterministic 96-window formal panel on a free CPU test node,
then request explicit approval for the already frozen paid `gnall` screen.
The formal run, not this smoke, applies every preregistered improvement,
regression, signed-bias, and hard constraint. The 365-day rollout, long-held
points, Gate D2, sealed test, and product acceptance remain closed.
