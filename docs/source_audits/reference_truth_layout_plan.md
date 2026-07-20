# Reference Truth Layout Audit

## Canonical Layout

The complete downloaded server hierarchy is canonical:

```text
reference/OUT/orc_calibrate_250919_sen/arg2_1.0/
  <landpoint>/<iteration>/<parameter-set>/
    stomate_history_1961.nc ... stomate_history_2010.nc
    driver_start.nc, sechiba_start.nc, stomate_start.nc
    driver_restart.nc, sechiba_restart.nc, stomate_restart.nc
    run.def
    z1/used_run.def

reference/script/orc_cali_250919/
  modelout_sen/arg2_1.0/modelout_<landpoint>.csv
  sen/arg2_1.0/Job/<landpoint>/<iteration>/<parameter-set>/
```

As audited on 2026-07-11, this contains 669 independent landpoint packages.
Every package has one selected iteration/parameter directory, 50 continuous
annual histories covering 1961-2010, all three component start/restart pairs,
`run.def`, and `z1/used_run.def`.

No extra normalization or NetCDF copying is needed. The model itself has no
fixed spatial resolution here: each package is one independent landpoint
selected from the forcing/domain protocol, and the paper mosaic is assembled
from the 669 runs.

## Resolver Contract

`jax_orchidee.driver.reference_layout.resolve_paper_landpoint_reference`
resolves in this order:

1. complete raw `reference/OUT` and extracted `reference/script` assets;
2. old `reference/paper_250919/landpoints` normalized copies;
3. legacy `reference/case_001_071` fixture.

New runners must pass both:

- a runtime run definition based on that landpoint's own `z1/used_run.def`,
  with its archived `run.def` static overrides;
- the same landpoint output directory as `reference_run_dir`.

This prevents configuration, start state, restart state, and history truth
from silently coming from different landpoints. Year-dynamic forcing, CO2,
and restart keys are selected by the runner rather than copied as static case
configuration.

## Compatibility Directories

`reference/case_001_071` and `reference/paper_250919/landpoints` remain
read-only compatibility fixtures. They are not canonical and must not be
expanded. New code must use the resolver rather than hard-coded paths.

The CSV manifest is a derived, regenerable inventory. It helps audit package
completeness but neither replaces nor reorganizes the canonical raw tree.

## Validation Protocol

Broad validation uses the existing annual histories and modelout CSVs. Heavy
half-hour traces are not required for each point and should only be acquired
after a failed annual comparison has been localized to a process boundary.

The staged numerical gate is:

1. representative 12-point cold-start 1961 comparison;
2. 1961-1965 comparison on the highest-risk dry, snow, and baseline points;
3. broader annual cross-section over the 669 local packages;
4. targeted source-driven micro-cases for PFT14-triggerable branches not
   discriminated by those paper forcings.

The annual gate compares the 12 underlying `stomate_history` modelout fields,
not only the four derived AGB/BGB/GPP/NPP values. Passing a paper landpoint is
evidence for its forcing/configuration path, not proof of every PFT14 branch.

## Current Checkpoint

- All 669 packages are locally resolvable as `raw_server_copy`.
- A deterministic 12-point sample spans coordinates, four calibrated PFT14
  parameters, and archived AGB/BGB/GPP/NPP values.
- Nine sample points closed for cold-start 1961 at approximately `1e-7` annual
  error before the latest per-landpoint run-definition correction.
- The active explicit-snow execution gap at `319.0-057.0` is fixed; the full
  year now executes.
- A restart binding bug that loaded point `001.0-071.0` state for other points
  is fixed.
- Validation now prefers each selected point's own `z1/used_run.def`; the
  remaining high-risk cold-start points must be rerun under this corrected
  binding before broader multi-year validation.

