# Distribution-Aware Teacher-Guard Calibration V1

Date: 2026-08-28

Status: frozen before zero-update calibration

Machine contract:
`manifests/coarse_graining/target_oriented_distribution_aware_guard_calibration_v1.json`

Contract SHA256: `c101940963465df1e25878f596645037e7d2de22dcefcf21f622fcd1de7179d7`

## Purpose

The accepted distribution-aware maintenance candidate must see both its
ordinary recursively predicted state distribution and the independent
Teacher-day-start distribution that exposed the predecessor's coupling
failure. This protocol fixes the relative coefficient before any optimizer is
created. It is not a training run.

## Frozen Panel

Use prepared train/train artifacts `352, 353, 354, 355, 356, 357, 359, 360`.
Each has 32 one-day samples, eight 7-day windows, and four 30-day windows. The
release remains Teacher v6; validation values, sealed test, and historical v5
are prohibited.

For every artifact, compute two gradients over exactly the same
`9,750,529`-parameter adapter subtree while the complete parent remains fixed:

- the existing one-day state/process plus 7/30-day component-resolved free
  objective;
- a Teacher-day-start guard containing six equally weighted cells:
  7/30 days crossed with maintenance respiration, growth respiration, and
  NPP.

Each guard cell is
`0.5 * normalized MSE + 0.5 * normalized signed bias squared`; each receives
weight `1/6`. Teacher state is reset independently at every day, so this guard
does not duplicate free-rollout recurrence.

## Coefficient

Let `m_free` be the median of the eight free adapter-gradient norms and
`m_guard` the median of the eight Teacher-guard adapter-gradient norms. Freeze

```text
teacher_guard_coefficient = clip(m_free / m_guard, 0.1, 10.0)
```

The final numeric value and calibration-report identity must be written into a
separate bounded training contract. It cannot be changed using validation or
rollout outcomes.

## Gate

All eight batches must have finite, strictly positive gradient norms for both
objectives, full `9,750,529`-coordinate coverage, and zero physical hard
failures. The run creates no optimizer, update, checkpoint, or recovery. A
free `gln01` GPU may execute it; paid training remains closed.
