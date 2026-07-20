# ORCHIDEE-MAN Reference Truth

This directory contains the local numerical truth for the paper PFT14 runs.
The downloaded server tree is already the canonical organized layout. Do not
copy the 669 landpoint packages into another normalized tree.

## Canonical Assets

```text
reference/
  OUT/orc_calibrate_250919_sen/arg2_1.0/<landpoint>/<iteration>/<parameter-set>/
  script/orc_cali_250919/modelout_sen/arg2_1.0/modelout_<landpoint>.csv
  script/orc_cali_250919/sen/arg2_1.0/Job/<landpoint>/<iteration>/<parameter-set>/
```

Each of the 669 output packages contains annual `stomate_history_1961.nc`
through `stomate_history_2010.nc`, start/restart files, the archived `run.def`,
and `z1/used_run.def`. The matching script tree contains paper modelout targets
and the generated job/run-definition assets.

Resolve assets by landpoint ID with
`jax_orchidee.driver.reference_layout.resolve_paper_landpoint_reference`.
Validation runners must bind both the selected landpoint's `used_run.def` and
its output directory. Mixing configuration or restart state between
landpoints invalidates the comparison.

## Compatibility Assets

- `case_001_071/` is the original single-point fixture.
- `paper_250919/landpoints/` is an older four-point normalized copy.

Both are read-only compatibility fallbacks. New validation must resolve to
layout `raw_server_copy`. They must not be expanded to 669 copied packages.

`paper_250919/manifest.csv` is a derived inventory and can be regenerated. It
is not numerical truth and is not required to locate assets.
