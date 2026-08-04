# Daily Coarse-Graining Research

This directory contains the research path for replacing the Teacher's 48
half-hour process block with a learned daily transition. It is separate from
the stable PFT14 Teacher and does not yet provide a user-facing surrogate run
mode.

## Read First

- `../../NEXT_STEPS.md`: current operation, dependency order, and acceptance
  gates.
- `../../CODE_MAP.md`: active and historical code classification.
- `../../DOCUMENT_STATUS.md`: active, conditional, and historical documents.
- `conservative_daily_process_operator_v1.md`: active true-daily architecture,
  PFT extensibility, parameter ownership, and the pre-training admission gate.
- `physical_parameter_candidate_registry_v1.md`: accepted source-driven
  sensitivity/inversion candidate classification and PFT14 shortlist.
- `gate_e1_daily_operator_skeleton.md`: active implementation packet for the
  next true-daily neural boundary, local acceptance tests, and forbidden paths.
- `failed_architecture_lessons.md`: active negative-design authority; read it
  before selecting an architecture or objective so rejected experiments are
  not repeated.
- `../../current-status.md`: current project and research status authority.
- `HANDOFF.md`: historical chronological log; consult only for a named past
  experiment or server job.
- `development_standard.md`: architecture, evidence, data, split, rollout,
  and performance rules.
- `daily_markov_contract_v5.md`: current state, fast-day target, native forcing
  and shard contract.
- `daily_markov_contract_v4.md`: superseded contract retained as migration
  provenance.
- `daily_markov_contract_v3.md`: historical contract used by the first bounded
  dataset and rejected neural baseline.
- `daily_boundary_audit.md`: superseded v0 boundary and replay evidence.
- `teacher_dataset_generation.md`: restartable landpoint-year shard contract,
  669-point production planner, asset staging, Slurm array, and aggregation gate.
- `teacher_pilot_v2.md`: frozen 12-landpoint pilot and promotion gates.
- `teacher_capture_gpu_benchmark_20260721.md`: accepted V100 Teacher capture
  measurement and its limited conclusion.

All other dated files are experiment reports. Preserve their original
conditions and conclusions; do not edit an old report to describe a newer
gate.

## Current Boundary

The implemented v5 boundary and its 669-point dataset remain accepted Teacher
and historical neural assets. They are not the active final surrogate design.
The final inference target consumes native six-hour records, predicts daily
process fluxes/rates, and performs one constrained state update without
reconstructing 48 forcing steps or running a 48-step state scan.

Implemented:

- Teacher capture and daily target extraction;
- Daily Fast-Day Teacher Contract v5 and native-forcing reconstruction;
- complete-day input ownership audit and named condition slices;
- fixed split and provenance-aware v4 shard generation;
- hash-verifying training dataset reader, sentinel-aware streaming train-only
  statistics, persistence-centered targets, and bounded-prefetch collation;
- deterministic multi-landpoint production split, cold-start asset staging,
  restartable Slurm worker-array and after-success aggregation plumbing;
- CPU and V100 compatibility gates;
- parameter-conditioned neural training plumbing.
- the frozen 50-label water/carbon/energy inventory, three default-off
  daily-reduced Teacher capture families, and accepted non-neural constrained
  replay through the retained exact daily tail.

Not implemented or not accepted:

- a scientifically validated daily neural surrogate;
- free-running 7-, 30-, 365-day, or 50-year surrogate closure;
- a user-facing `daily-surrogate` CLI mode;
- a decision that inference must run on GPU.

## Current Work

Gate C's parameter ownership, daily water/carbon/energy labels, supplemental
capture, and non-neural constrained replay are complete. Gate D1 Teacher
gradients and the physical-parameter candidate registry are also accepted. Do
not reopen them or regenerate 669-point data unless their frozen contracts
drift. The active next task is Gate E1 true-daily operator implementation; it
must use the accepted Gate C updater rather than a direct next-stock predictor.

The network interface must be PFT-extensible even though the current data and
scientific claim remain PFT14-only. Continue from
[`../../NEXT_STEPS.md`](../../NEXT_STEPS.md), which owns the next milestone.
