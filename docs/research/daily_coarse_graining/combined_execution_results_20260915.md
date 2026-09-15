# Combined execution admitted; return to training

Free gln01 GPU0 v2 completed at immutable `364cdef`, wrapper exit0,923.85s
(15.40min). Bash58101/Python58147 have exited. No active performance task,
paid job, retained training update or Teacher regeneration remains from this
study. The preceding v1 timeout and successful recurrent comparisons are
preserved in `combined_execution_admission_20260915.md`.

## Numerical and update evidence

The explicit `mixed_learning_execution_fp32_checked_v1.json` policy combines
Attention FP32 with native checkpointed AD and separate validation of retained
recurrent observations. Architecture,109,464,375 parameters, daily64/micro8,
fixed TRAIN T7/T30 windows, losses and optimizer are unchanged. Physical
arithmetic remains FP64; neural pooling remains at legacy precision.

T7/T30 full-gradient comparisons pass with zero failed coordinates and relative
L2 differences1.45425e-7/1.62676e-7 against the original preserved references.
Scalar losses and every named physical counter pass; all hard counts are zero.
Six ordinary/instrumented complete-update replays have finite combined
gradients and optimizer proposals. Mixed/component losses, gradient norms and
clipping agree within existing tolerances, maximum relative difference4.781e-8.
Daily FP32 full-gradient/auxiliary evidence is reused from its completed
admission. No new AD algorithm or tolerance relaxation was introduced.

This is fixed-checkpoint execution admission, not improved scientific accuracy,
coordinatewise optimizer-proposal equality or final-model acceptance. Each
proposal is discarded; this benchmark does not add training steps.

## Throughput and memory

| Warm complete-update measurement | Seconds |
| --- | --- |
| Original preserved baseline | 14.18122 |
| Previously admitted checked split, legacy Attention | 12.07914 |
| Combined Attention FP32 + checked split | 11.76113 |

Combined ordinary samples:11.76444,11.76113,11.73794s. Total time reduction is
17.065%, throughput1.20577x; incremental time reduction versus checked-only is
2.633%. Both reference timings come from earlier processes, not interleaved
paired measurements. This fixed-data, warm V100 result does not establish
all-window training throughput or RTX5090 performance.

Peak live device allocation16,812,204,544bytes (15.658GiB); allocator pool
28,967,692,544bytes (26.978GiB). These are distinct quantities. Cached T7/T30
compile calls now take80.63/80.51s and daily99.04s. The v1 startup work is
successfully reused; initial compilation remains separate from warm speed.

At the measured fixed-update rate,5,415 updates would take17.69 GPU wall-clock
hours versus21.33h at the original rate, a3.64h saving. This is arithmetic for
budget planning, not a full job estimate: add first compile, variable-data IO,
monitoring/checkpoint costs and operational margin before requesting resources.

## Adoption and next task

Admit the explicit combined policy for subsequent declared mixed training.
The current justified performance work is complete. Do not launch additional
pooling/norm microbenchmarks, broad cache audits or historical custom-adjoint
checks as prerequisites for training. Legacy behavior remains available for
reference reproduction; future training declares the combined policy and
records it in report/checkpoint method identity.

Next prepare a substantial data-coverage training protocol and its executable
schedule. The existing ready pool is346,539 daily rows, about5,415 batch64
updates for one pass. The prior200-update run used12,800 unique daily draws;
its failure to meet scientific gates does not establish adequate training.
Preserve the Attention architecture and daily+T7/T30 objective while increasing
coverage and update count. Specify fresh initialization/full-network learning,
sampling, recurrent windows, optimizer schedule, milestones, recovery and
checkpoint selection before execution. Retain strict monitor/development
exclusions, and measure actual covered sites/years/days rather than merely
requesting a larger update count. Reuse existing Teacher containers.

Validate schedule/runner changes in proportion to their scope, reusing accepted
gradient evidence. Then present concrete paid resources, wall limit and cost;
no new paid submission is approved by this result. Monitor physical-unit
GPP/NPP/RA/RH, AGB/BGB, canopy and nonliving carbon. Learning sufficiency and
generalization, not further minor speed gains, are now the research priority.

Evidence: local `outputs/mixed-learning-fp32-checked-v2.json`; remote
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/mixed-learning-fp32-checked-v2.{json,log}`
and sibling `.T7.checked-gradient.pkl` / `.T30.checked-gradient.pkl`.
