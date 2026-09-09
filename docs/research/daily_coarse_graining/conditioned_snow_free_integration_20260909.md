# Corrected-head full train-only step

Status: frozen before implementation and execution, 2026-09-09; implementation
and local qualification pass, free-GPU execution next.

## Why this is next

The corrected-head derivative check `e6a00b2` passes all three windows and
eight reverse/JVP/finite-difference pairs, with actual descent at both
magnitudes. This admits checking the complete training update. It does not
establish accurate prediction, and the preceding correction worsens two
individual windows. Judge progress from freshly computed corrected-head
starting losses; preserve the existing absolute science targets.

## Fixed mechanism and reference

Reuse the existing streamed integration runner and `_train_only_update` for
logical update 4 from accepted update-3 recovery `45ab263`. Enable the warm
snow-free boundary before constructing every forward/direct/state/gradient
helper. Use the compact canonical trace admitted in the derivative check for
the full streamed horizon pullbacks. Leave the existing direct and state
gradient methods, Attention weights/architecture, loss definitions, 43-source
direction solver, calibration and descending scales 1 through 1/64 unchanged.
All parameter proposals are discarded after numerical reporting.

The original five groups of eight train/train artifacts supply 40 direct-GPP
constraints; scheduled group 541,717,596,867,322,528,278,442 supplies 64 seven-day
and 32 thirty-day gradients. All initial losses, global-bias statistics and
finite-step science scores are computed through the corrected model. No
historical score totals or uncorrected callbacks may enter this run.
The reference is the same corrected starting model, with the completed
uncorrected integration `1e6c521` retained only as historical context. There
is no new architecture arm, Teacher data, validation or monitor exposure.

## Gate and resource bound

Require all 96 scheduled gradients, their production/observation score
agreement at atol=1e-6 and rtol=1e-5, and the original train-only step-selection
and detailed science-progress gates. Record absolute science gates separately;
a successful one-step reduction is not achievement of final accuracy.
Keep physical budgets, discrete/defined behavior and source regime semantics
unchanged. Do not loosen a gate or try new scales in response to failure.

One free gln01 V100, maximum 90 minutes (previous full integration: 35.53
minutes), no paid submission. Reuse the existing environment, artifacts and
compiled caches. The daily reverse keeps its 24 GiB estimate limit; do not
compile whole-30-day reverse. Before launch, test default-off compatibility,
corrected bootstrap and compact-trace routing, both actual horizon callers,
full runner serialization/discarding, and launcher arguments.

If this passes, the next gate is paired evaluation of the corrected baseline
and candidate on the already declared fixed monitor before any retained
continuation. A failure first requires interpreting its numerical/science
record; do not restart broad architecture search. No 365-day run, Gate D2 or
product acceptance is opened by this experiment.

## Local qualification

Seventeen targeted integration, corrected-gradient and persistent-training
tests pass in 64.37 seconds. Both default and corrected runner paths execute
the full mocked 96-window lifecycle; actual horizon callers match full
differentiation with changing data/targets and float32 parameters. Corrected
bootstrap and compact-trace routing are explicit, and no proposed parameter
tree enters JSON or checkpoint output. The existing default path is unchanged.
Ruff, parser/launcher argument coverage and whitespace checks pass.
