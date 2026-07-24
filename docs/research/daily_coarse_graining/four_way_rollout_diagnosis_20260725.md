# Four-Way Seven-Day Rollout Diagnosis

Date: 2026-07-25

## Question

Determine whether the failed seven-day neural rollout is primarily caused by
recursive optimization on already-seen conditions, temporal generalization,
or spatial generalization. This diagnosis uses the frozen v5 checkpoint and
existing Teacher data only. It does not train a new model, generate Teacher
data, or access the sealed test split.

## Controlled Matrix

All cases use days 100-106, free neural state feedback, the checkpoint with
SHA256 `79728593fa78f45f0c77dbe219f983114f76fe39bc0b63443c4af11e612b5bc2`,
and the accepted nine-point v5 dataset. Point `281.0-095.0` is in the spatial
training split and point `215.0-119.0` is in the spatial validation split.
Years 1961-2004 are training years; 2005-2007 are validation years.

| Spatial split | Temporal split | Point/year | Day-1 RMSE | Day-7 RMSE |
|---|---|---|---:|---:|
| train | train | `281.0-095.0`, 2004 | 0.028362 | 0.145471 |
| train | validation | `281.0-095.0`, 2005 | 0.042092 | 0.134233 |
| validation | train | `215.0-119.0`, 2004 | 0.257418 | 0.519886 |
| validation | validation | `215.0-119.0`, 2005 | 0.072900 | 0.429058 |

Every case has zero defined/undefined-status mismatch and zero discrete-state
mismatch.

## Findings

1. Moving from a training year to a validation year at the same training
   landpoint does not degrade the final error (`0.145471` versus `0.134233`).
   Lack of temporal samples is therefore not the leading explanation in this
   controlled pair.
2. Moving to the spatial validation landpoint degrades final error to
   `0.429058-0.519886`, even when the year remains in the training split. The
   current six training landpoints do not provide adequate spatial-condition
   coverage for this validation point.
3. Recursive optimization is still imperfect on seen conditions. At the
   train/train case, `litterpart` normalized RMSE grows from `0.541` on Day 1
   to `3.652` on Day 7. Growth respiration and NPP also remain among the
   leading field errors.
4. A global RMSE gate alone is unsafe. The train/train final global RMSE is
   below the provisional `0.29` gate while a low-dimensional carbon state is
   already severely wrong.
5. The validation-point/train-year case has large first-day `cgrnd/dgrnd`
   errors in ENERBIL and THERMOSOIL, followed by biomass, litter, temperature,
   and energy-flux drift. This supports a spatial-conditioning deficiency in
   addition to recursive carbon-state drift.

## Decision

Do not add more years of the existing landpoints: the controlled evidence does
not identify temporal coverage as the current bottleneck. Do not immediately
generate all 669 landpoints either.

Use a bounded sequence:

1. On the existing dataset, perform one controlled objective A/B that adds
   process-balanced next-state supervision and state-increment supervision.
   It must improve both global error and named carbon-state errors on the
   train/train case under an equal update budget.
2. In parallel, select a small spatially diverse Teacher-data pilot from the
   669-point inventory using parameters, static soil/landpoint attributes, and
   forcing climatology. Selection must be source-driven, not based on neural
   failures alone.
3. Generate the pilot only after the objective passes the seen-condition gate.
   Preserve independent spatial validation and sealed test points.
4. Re-run this same four-way matrix. Additional spatial data is useful only if
   spatial validation improves without degrading seen-condition stability.

The provisional global `<=0.29` gate must be supplemented with process-family
and named slow-state gates before any 30-day promotion.

## Evidence Assets

Server directory:

```text
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/experiments/
canonical-v5-curriculum-gln01-23a1851/validation/
```

Assets:

- `split-train-train-281.0-095.0-2004-day100-7d-825935b.json`
- `split-train-validation-281.0-095.0-2005-day100-7d-825935b.json`
- `split-validation-train-215.0-119.0-2004-day100-7d-825935b.json`
- `joint-215.0-119.0-2005-day100-7d-drift-18471aa.json`

