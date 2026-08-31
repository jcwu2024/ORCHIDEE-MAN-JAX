# Structured-GPP State-Feedback Attribution V1

Date: 2026-08-31

Status: frozen before zero-training execution

Contract SHA256: `858b59077d4a9dbdc7ecaeb06345725bd94c2a570f1b78d258d8dee480bc69ed`

## Question

The accepted fresh attribution proves that the broad step-1000 candidate has
both a day-local GPP generalization error and a material cross-day state
feedback error. This bounded train-only audit asks which source-owned state
component carries the latter error into later GPP predictions.

It does not train, rank architectures, or use the fresh 96-window targets.

## Matched Intervention

The fixed panel uses prepared `train/train` updates `301`, `601`, and `901`.
Each update contains eight 7-day and four 30-day windows. The candidate,
forcing, parameters, constrained updater, retained exact tail, and initial
Teacher state are identical in every mode.

The modes are ordinary free rollout, complete Teacher day-start state, and six
single-component Teacher-carry interventions:

- `state_canopy`: `diffuco_previous_step_state`;
- `state_energy`: `enerbil_previous_step_state`;
- `state_water`: `hydrol_previous_step_state`;
- `state_thermal`: `thermosoil_previous_step_state`;
- `state_carbon`: `slowproc_stomate_previous_step_state`;
- `state_driver`: `driver_previous_step_state`.

For a component intervention, every continuous leaf owned by that source is
replaced by its matched Teacher day-start value before days 2 onward. Other
continuous state and all discrete state remain candidate-predicted. A runtime
state mask is used so all six interventions share the same compiled 7-day and
30-day graphs.

## Measurement And Gate

For cumulative area-weighted GPP, the accepted score is

```text
0.5 * (normalized RMSE^2 + normalized signed bias^2)
```

For every update and horizon, component capture is

```text
(free score - component score) / (free score - full-Teacher score)
```

Only positive free-to-Teacher gaps above `1e-12` are eligible. A component is
material at a horizon when at least two of three groups are eligible, at least
two have positive capture, and median capture is at least `10%`. A component
enters the first coupled GPP state/process-head scope when it is material at
either 7 or 30 days.

This is a scope gate, not a promotion gate. If no individual component is
material, no component is discarded; interactions must be audited before an
architecture edit.

## Execution Boundary

The audit creates no optimizer, applies no parameter step, writes no
checkpoint, regenerates no Teacher data, and reads neither v5 nor fresh
validation targets. It reconstructs no 48 half-hour forcing records and runs
no 48-step state scan. It may run on free `gln01`; paid execution, 365-day
rollout, Gate D2, sealed test, and product acceptance remain closed.
