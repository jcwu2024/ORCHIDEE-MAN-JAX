# Daily Coarse-Graining Research

This directory contains the research path for replacing the Teacher's 48
half-hour process block with a learned daily transition. It is separate from
the stable PFT14 Teacher and does not yet provide a user-facing surrogate run
mode.

## Read First

- `../../current-status.md`: current project and research status authority.
- `development_standard.md`: architecture, evidence, data, split, rollout,
  and performance rules.
- `daily_boundary_audit.md`: Teacher/coarse boundary and state ownership.
- `teacher_dataset_generation.md`: restartable landpoint-year shard contract.
- `teacher_capture_gpu_benchmark_20260721.md`: accepted V100 Teacher capture
  measurement and its limited conclusion.

All other dated files are experiment reports. Preserve their original
conditions and conclusions; do not edit an old report to describe a newer
gate.

## Current Boundary

Implemented:

- Teacher capture and daily target extraction;
- fixed split and provenance-aware shard generation;
- CPU and V100 compatibility gates;
- parameter-conditioned neural training plumbing.

Not implemented or not accepted:

- a scientifically validated daily neural surrogate;
- free-running 7-, 30-, 365-day, or 50-year surrogate closure;
- a user-facing `daily-surrogate` CLI mode;
- a decision that inference must run on GPU.
