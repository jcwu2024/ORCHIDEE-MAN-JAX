# Successor Training Sufficiency, Loss, and Gradient Audit

Status: **active diagnosis controlling the next successor gate**.

This audit asks whether the rejected 500-update attention checkpoint was a
clean architecture test. It was not. The checkpoint remains rejected for
promotion, but the evidence does not isolate attention as the cause because
the data schedule, objective calibration, and shared output head were all
confounded.

## Scope

The audit uses only accepted v6 `train/train` assets and the existing matched
pair:

- `source_routed_process_attention_v2`;
- `source_routed_graph_mlp_control_v2`;
- 8,722,689 parameters per arm;
- the canonical `bind_daily_operator(...)` boundary;
- native forcing records and the canonical retained tail.

It does not regenerate Teacher data, read v5, use sealed test data, train a
network, write a checkpoint, reconstruct 48 forcing records, or execute a
48-step candidate state scan.

## Data Sufficiency Finding

The accepted v6 `train/train` product contains 8,591,565 daily transitions in
23,540 point-years. The completed fit used only 500 unique point-years, or
2.12% of the available point-years.

Within those 500 updates it used:

- 8,000 one-day anchors, 0.093% of available train/train days;
- 1,000 thirty-day windows, 0.349% of nominal train/train day positions;
- one year from each selected landpoint;
- no fixed held-out validation curve during optimization.

The training history was still improving at update 500. Attention one-day
median loss fell from 9.4646 over updates 301-400 to 1.9942 over updates
401-500; control fell from 6.0194 to 2.8947. Every update in both arms had a
pre-clip global gradient norm above the clip threshold of 1.0. The run was
therefore neither data-sufficient nor demonstrably converged.

## Objective Bug

The original rollout objective evaluated bias squared inside each rollout
window and then averaged windows. Each per-window reduced field was scalar,
so mean squared error and bias squared became identical throughout the 500
updates:

- cumulative-flux MSE equaled cumulative-flux bias squared;
- terminal-stock MSE equaled terminal-stock bias squared.

Commit `70c2f62` corrected the reduction order by making the rollout batch
time-major before the long-horizon objective. Bias is now measured jointly
across windows. Focused unit tests and a real-v6 probe distinguish MSE from
bias squared when residuals differ.

## Coefficient Audit

The earlier fit calibrated four coefficients from the first attention batch
and reused them for control. The successor sufficiency v2 probe instead
prepared 16 distinct `train/train` point-years and measured both arms, giving
32 gradient records without a training update.

A shared median exists numerically, but it is not a stable training
calibration:

| Component | Shared median coefficient | Achieved gradient-ratio range |
| --- | ---: | ---: |
| cumulative flux MSE | 0.0045588 | approximately 1.5e-29 to 9.15 |
| cumulative flux bias squared | 0.0072146 | approximately 2.4e-29 to 15.43 |
| terminal stock MSE | 0.0108455 | approximately 1.7e-27 to 1.14 |
| terminal stock bias squared | 0.0162821 | approximately 2.5e-27 to 1.30 |

The extreme candidates came from updates 12 and 15, where both arms had
long-horizon parameter-gradient norms near `1e-20` or below. A global median
would hide rather than repair that failure, so these coefficients are not
frozen for a new fit.

## Activity Audit

A free-`test01` audit reopened all 16 prepared artifacts and aligned their
rollout windows, PFT fractions, Teacher process targets, and gradient records.
The two near-zero-gradient point-years were not empty or dormant:

- vegetation fraction was 1.0 on all 120 rollout days;
- Teacher GPP RMS was 2.052 for update 12 and 3.292 for update 15;
- many process targets and expected state changes were nonzero.

This rules out inactive vegetation, an undefined objective, and empty Teacher
supervision as explanations.

## Gradient-Path Finding

A zero-update, one-day free-`gln01` probe differentiated the objective first
with respect to the 181 named physical outputs and then with respect to network
parameters. On updates 12 and 15:

- cumulative-flux gradient norms at the named-output boundary were generally
  order 1 on active windows;
- the same component's parameter-gradient norm was approximately
  `1e-24` to `1e-20`;
- predicted active-PFT GPP was approximately `1e-31` to `1e-23` for update 12
  and `1e-31` to `1e-27` for update 15, while active Teacher values were order
  1 to 5;
- maintenance-respiration predictions showed the same collapse;
- attention and control behaved almost identically.

The loss and retained-tail derivative are therefore present. The gradient is
lost inside the shared network-to-physical-output mapping. The shared
nonnegative decoder is driven far into its softplus negative tail for some
landpoints at initialization.

The all-16 extension shows a broader initialization problem, not two isolated
outliers. Considering windows with Teacher GPP above 0.1:

- 53 windows were active per arm;
- 9 predicted less than `1e-8` of Teacher in each arm;
- 22 attention and 23 control windows exceeded 10 times Teacher;
- median prediction/Teacher ratios were 5.42 and 5.67.

For maintenance respiration with Teacher sum above 0.01:

- 60 windows were active per arm;
- 12 were below `1e-8` of Teacher in each arm;
- 41 exceeded 10 times Teacher in each arm;
- median ratios were 24.38 and 26.03.

This combines dead negative-tail samples with extremely large positive
samples. It explains both the near-zero long-horizon gradients on some batches
and persistent global gradient clipping on others. Loss coefficients cannot
repair this conditioning problem.

## Decision

The completed 500-update attention checkpoint remains rejected by its frozen
7/30-day screen. The control remains unpromoted. However, the screen only
rejects those checkpoints under the old sampling, loss, coefficient, and
initialization regime. It does not establish that attention is the cause or
that changing the process-interaction mechanism would solve the problem.

The user approved the first matched repair. Both arms retain their process
bodies, inputs, outputs, parameter tree and count, parameter routing, retained
tail, and inference boundary. Only the common `decoder_second.kernel`
initialization is multiplied by `1e-3`; its bias remains exactly zero. This is
nonzero so the first backward pass can still reach the preceding decoder and
body. The implementation and 59 related local tests pass.

The active gate is now:

1. repeat the existing 16-batch forward and gradient-path audit on free
   `gln01`, with zero training and no checkpoint;
2. require finite execution, the unchanged per-arm parameter count, no
   negative-tail collapse on active Teacher GPP/maintenance windows, and
   usable parameter gradients wherever named-output gradients are active;
3. verify that the former widespread extreme-positive initialization is
   materially reduced, without treating initial Teacher agreement as a model
   accuracy test;
4. only then recalibrate the long-horizon objective across multiple batches;
5. freeze a validation panel disjoint from the old formal 96-window panel;
6. implement milestone validation/checkpointing and the expanded schedule of
   at most 10,000 unique point-years.

No paid preparation or training is authorized. The 365-day rollout,
long-held-point evaluation, Gate D2, sealed test, and product acceptance remain
closed.

## Evidence

- 16-batch coefficient report:
  `runtime/outputs/training/successor-multibatch-coefficients-c7c6af6/probe_report.json`
- prepared activity report:
  `runtime/outputs/diagnostics/successor-multibatch-activity-c7c6af6/activity_report.json`
- two-batch gradient-path report:
  `runtime/outputs/diagnostics/successor-low-gradient-path-c7c6af6/probe_report.json`
- all-16 gradient-path report:
  `runtime/outputs/diagnostics/successor-low-gradient-path-all16-c7c6af6/probe_report.json`
- training-sufficiency probe contract:
  `manifests/coarse_graining/successor_training_sufficiency_v2.json`
