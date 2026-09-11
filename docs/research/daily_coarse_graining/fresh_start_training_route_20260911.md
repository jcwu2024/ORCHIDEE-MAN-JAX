# Fresh-start scientific training route

User decisions clarified and recorded 2026-09-11. This document specifies the
next deliverable; it is not an accepted trained model or a paid-job approval.

## Why the work order changes

The user prioritizes a paper method that can be independently specified and
executed. The inherited 109,464,375-parameter checkpoint is the product of
exploration, with only 15,960,211 GPP-related parameters recently trainable.
Direct DOC process heads, learned maintenance and shared parent modules were
frozen. Previous pretraining supplied some useful predictions but did not
establish complete scientific accuracy or justify permanent freezing.

The default formal route is fresh initialization of the consolidated neural
architecture. Retain the Attention design as the current implementation
starting point, the canonical assembler/updater/retained tail, immutable v6
Teacher assets and useful engineering fixes. Existing checkpoints serve
diagnosis and resource measurement, not an obligatory weight source for the
paper. Do not immediately reopen architecture search or run a warm-start versus
random-start ranking as the next main task.

## What the input experiments established

All-state and DOC-only T=16 log-tail screens both completed. They remove the
observed DOC rounding amplification and pass the three tested forward-score,
mask and physical-hard gates. Both fail at least one old fixed-weight science
regression cap in every case. DOC-only improves NPP and final AGB/BGB absolute
errors on these three train cases, while maintenance worsens and GPP worsens in
one. Error compensation remains possible. Neither transform has real-model
derivative or trained-model scientific acceptance yet.

These are evidence for input-conditioning design, not a requirement that a new
model preserve every old prediction. Fresh initialization needs its own
numerical/physical admission and learning test. Do not carry over the old
"within 5% of unchanged update-41 MSE at zero updates" criterion. Do not loosen
budget or gradient-correctness requirements. The input transform is part of
the model and must be identical during training, rollout and final inference.

## First concrete deliverable

Build a small, checkpoint-independent training prototype for the current
Attention architecture, before a large training submission:

1. One initialization entry point produces every neural parameter from a
   declared seed and architecture configuration. Respect sensible per-layer
   and output-scale initialization. No hidden load of exploratory weights or
   mandatory chain of historical training-report/checkpoint arguments.
2. The optimizer can update all neural parameters, including DOC, maintenance
   and shared features. Physical formulas/constants are not neural weights.
   Verify actual trainable ownership; finite gradients alone do not prove
   that every intended module can learn. Structural masked/inactive parameters
   need not have nonzero gradients on every individual sample.
3. Before execution, declare one input-conditioning candidate, train-only
   normalization provenance, losses/scales and their weights, optimizer,
   clipping/learning-rate schedule, sample identities, batch, seed, update
   budget and stop criteria. The tested all-state and DOC-only transforms are
   available candidates, not silently accepted final policy. Do not tune them
   against the sealed test or require another old-checkpoint ablation first.
4. Local forward/loss/gradient and canonical one-day state/budget checks first.
   Then use a small real-v6 batch on a free node to measure all-parameter
   gradients, useful module connectivity, actual memory and cold/hot time.
   Verify derivatives with a short direct-AD/JVP/finite-difference reference
   before relying on a custom streamed reverse for longer windows.
5. Once admitted, run a bounded learning probe on a declared small multi-point
   train-only set. Show Teacher and predicted GPP, maintenance/growth/total
   autotrophic respiration, NPP and biomass changes in physical units. Record
   whether repeated training improves fit and whether hard gates remain valid.
   Its purpose is to establish learnability of the complete new route; it
   cannot establish generalization or long-term acceptance.

The next implementation should make these steps executable. Exact numerical
training settings and an execution budget must be written before starting the
probe; this roadmap does not invent unmeasured batch sizes or authorize paid
work. Free-node tests remain authorized by the session. Do not regenerate v6,
train on v5, reconstruct 48 forcing steps, or execute a 48-step inference scan.

## Following stages

After single-day learning works, introduce short recurrence, then 7/30-day
joint objectives with derivative admission at the actual parameter scope.
Define stage transitions in advance. A full-network objective is the default;
any temporary freezing must state why and when it ends. Use measured resources
to select batch/window/accumulation rather than assuming the old frozen scope
is necessary. The available 2,000-package expansion can support later wider
coverage; it has not been consumed by new training.

After choosing the effective method, consolidate architecture, data/sampling,
seeds, initialization, input processing, losses, update rules, stages, stopping
and selection into a single executable protocol and run it end to end. Only
then claim a reproducible final method, with independent science validation.
365-day/long-held-out, Gate D2 and product acceptance remain later gates.
Neither improvement on a few train cases nor the existence of this document
establishes success. Do not retrospectively call exploratory choices a
preregistered method.
