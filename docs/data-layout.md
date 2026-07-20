# External Data Layout

The Git repository contains no large scientific inputs. The canonical layout
is recorded in `manifests/paper_pft14_data.yaml`.

```text
$ORCHIDEE_DATA_ROOT/
  forcing/
  forcing_VN/
  INPUTDIR_ZZ/
  MICT_BIOE/Input/

$ORCHIDEE_REFERENCE_ROOT/
  OUT/orc_calibrate_250919_sen/arg2_1.0/<landpoint>/...

$ORCHIDEE_OUTPUT_ROOT/
  checkpoints/
  acceptance/
  logs/
  xla_cache/
```

Run `orchidee-jax inventory` before submitting jobs. Full Fortran references
are needed for acceptance, but not for a model run that starts from a supplied
JAX-compatible initial state and does not request reference comparison.
