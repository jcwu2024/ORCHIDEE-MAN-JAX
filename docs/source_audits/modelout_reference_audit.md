# Modelout Reference Audit

Source truth:

- `fortran_run_scripts/paper_250919/Job0_bio`
- `fortran_run_scripts/paper_250919/peat-leak_xml_zz/file_def_orchidee_pods_year_240405.xml`
- `fortran_run_scripts/paper_250919/c2.4_Model_run_functions_sensitivity.py`
- `reference/case_001_071`

## Job Output Flow

File: `fortran_run_scripts/paper_250919/Job0_bio`

- XML source directory: line 10
- copy XML files: line 46
- restart and output filenames:
  - `RESTART_FILEOUT driver_restart.nc`: line 103
  - `SECHIBA_rest_out sechiba_restart.nc`: line 104
  - `STOMATE_RESTART_FILEOUT stomate_restart.nc`: line 105
  - `OUTPUT_FILE sechiba_history.nc`: line 106
  - `STOMATE_OUTPUT_FILE stomate_history.nc`: line 107
  - `STOMATE_IPCC_OUTPUT_FILE stomate_ipcc_history.nc`: line 108
- cold-start restart inputs set to `NONE`: lines 288-290
- yearly XML alias: `file_def_orchidee_pods_year_240405.xml` copied to
  `file_def_orchidee.xml`: line 309
- yearly restart rollover: lines 315-322
- check `stomate_history.nc` exists: lines 349-351
- rename `stomate_history.nc` to `stomate_history_${YEAR}.nc`: line 354
- copy yearly restart files into `out${YEAR}`: lines 368-370

`sechiba_history` is not retained in the reference run. The relevant move is
commented in `Job0_bio`, and the XML disables SECHIBA history files.

## Enabled XML Output

File: `file_def_orchidee_pods_year_240405.xml`

The only enabled history file for the paper run is:

- file id: `stomate1`
- name: `stomate_history`
- line: 747
- enabled: `.True.`
- output frequency: `1y`
- output level: `1`

Other declared files are disabled:

- `sechiba_history`
- `sechiba_zhaoz`
- `sechiba_history2`
- `sechiba_history_alma`
- `sechiba_out_2_alma`
- `stomate_ipcc_history`

## Reference Outputs

Directory:

`reference/case_001_071/OUT/orc_calibrate_250919_sen/arg2_1.0/001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658`

Retained files:

- `stomate_history_1961.nc` through `stomate_history_2010.nc`
- `driver_start.nc`
- `sechiba_start.nc`
- `stomate_start.nc`
- `driver_restart.nc`
- `sechiba_restart.nc`
- `stomate_restart.nc`

Not retained:

- `sechiba_history.nc`
- `sechiba_history2.nc`
- `stomate_ipcc_history.nc`

## Modelout CSV

File:

`reference/case_001_071/script/orc_cali_250919/modelout_sen/arg2_1.0/modelout_001.0-071.0.csv`

Shape:

- rows: 1
- columns: 16

Columns:

```text
Unnamed: 0, lon, lat, lonr, latr, age, AGB, BGB, GPP, NPP,
gf_flag, Igrid, AGB_model, BGB_model, GPP_model, NPP_model
```

Reference row:

- `Igrid = 001.0-071.0`
- `age = 50`
- target year: `1961 + age - 1 = 2010`
- `AGB_model = 96.398818359375`
- `BGB_model = 35.8754345703125`
- `GPP_model = 4.085646152496338`
- `NPP_model = 1.6014299392700195`

## Modelout Derivation

Source script:

`fortran_run_scripts/paper_250919/c2.4_Model_run_functions_sensitivity.py`

Relevant logic:

- reads `stomate_history_${1961+age-1}.nc`: line 50
- extracts `time=0`, `veget=13`, `lat=0`, `lon=0`: lines 60-73
- PFT14 is zero-based vegetation index 13
- aboveground raw sum: line 75
- belowground raw sum: line 76
- `AGB_model` and `BGB_model` multiply raw sums by `0.02`: lines 115-116
- `GPP_model` and `NPP_model` are copied directly: lines 117-118

JAX first-stage modelout fields:

- `LEAF_M`
- `SAP_M_AB`
- `HEART_M_AB`
- `AGR_SAP_ST_M`
- `AGR_HRT_ST_M`
- `AGR_SAP_PN_M`
- `AGR_HRT_PN_M`
- `SAP_M_BE`
- `HEART_M_BE`
- `ROOT_M`
- `GPP`
- `NPP`

Derived fields:

```text
AGB_model = 0.02 * (
    LEAF_M + SAP_M_AB + HEART_M_AB
    + AGR_SAP_ST_M + AGR_HRT_ST_M
    + AGR_SAP_PN_M + AGR_HRT_PN_M
)

BGB_model = 0.02 * (SAP_M_BE + HEART_M_BE + ROOT_M)
GPP_model = GPP
NPP_model = NPP
```

Second-stage useful diagnostics:

- `LAI`
- `VEGET_MAX`
- `control_salinity`
- `control_inudate`
- `AGE`
- `HEIGHT`
- `NPP_ABOVE`
- `NPP_BELOW`

## Annual History Aggregation

The paper modelout script does not read local daily runner values directly. It
reads yearly `stomate_history_${YEAR}.nc` records.

Source/output provenance:

- `fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj` sends
  `NPP_STOMATE`, `GPP`, and the biomass pool fields through
  `xios_orchidee_send_field` at lines 1677-1699.
- The legacy `histwrite_p` path writes the same fields at lines 2200-2246.
- `fortran_run_scripts/paper_250919/peat-leak_xml_zz/file_def_orchidee_pods_year_240405.xml`
  enables `stomate_history` with `output_freq="1y"` at line 747 and includes
  the modelout fields at lines 762-782.
- Local NetCDF metadata for `stomate_history_1961.nc` records these fields as
  `online_operation = average`, `interval_operation = 1800 s`, and
  `interval_write = 1 yr`.

JAX output-layer implementation:

- `jax_orchidee/stomate/modelout.py::annual_history_mean_fields_from_daily_modelout`
  aggregates completed local daily STOMATE output sends into one
  history-shaped yearly record.
- For the current single-point paper case it returns `(time, vegetation, lat,
  lon)` arrays with shape `(1, nvm, 1, npts)`, so the existing
  `select_history_point_fields` and `compute_modelout_from_fields` functions
  can consume the result using the same paper-script selectors.
- This is only an output-layer aggregation. It does not change HYDROL,
  DIFFUCO, ENERBIL, THERMOSOIL, or STOMATE process state.

Retained diagnostics:

- `scripts/dev/compare_multiday_modelout_to_reference.py` now records
  `annual_history_mean_diagnostic`, which compares the local annualized
  diagnostic with the selected reference history record.
- `scripts/dev/run_multiyear_modelout_lite.py` now records
  `annual_history_modelout` for each completed in-memory model year.
- The same runner now reads the archived paper modelout CSV through
  `jax_orchidee.stomate.reference.read_paper_modelout_csv` and records
  `paper_modelout_csv_delta` when a completed annual JAX year matches a CSV
  target year. This is an output-validation gate only; it uses the paper
  script age-to-year and PFT14 selector and does not alter process state.
- Smoke output:
  `outputs/multiyear_modelout_lite_1961_1962_1day_with_annual_modelout.json`.
- Current five-year gate:
  `outputs/reference_mode/multiyear_1961_1965_current_paper_csv_gate.json`
  completes 1961-1965 from cold start with zero handoff gaps. The full-year
  annual modelout absolute errors are below `7e-7` for AGB/BGB and below
  `6e-8` for GPP/NPP across all five years. The archived CSV target year is
  2010, so `paper_modelout_csv_delta` is empty for 1961-1965 and will become
  active when the runner reaches 2010.
- For long paper-target validation, `run_multiyear_modelout_lite.py` supports
  `--year-checkpoint-dir`. When enabled, each completed year writes a
  resumable `paper_driver_${YEAR}_year_end_state.pkl` plus a JSON partial
  summary. Current checkpoints are in
  `outputs/cache/paper_current_checkpoints/` through
  `paper_driver_1965_year_end_state.pkl`. The full current checkpointed run is
  `outputs/reference_mode/multiyear_1961_1965_current_checkpointed.json`, and
  `outputs/reference_mode/smoke_current_checkpoint_resume_1966_1d.json`
  verifies that 1966 can resume from the 1965 cache with zero handoff gaps.

Important validation boundary:

- `outputs/server_1961_trace_full_20260623/run/used_run.def` is the current
  trace-backed cold-start semantic target. Its active PFT14 sensitivity values
  include `VCMAX25__00014=50`, `MAINT_RESP_SLOPE_C__00014=0.05`,
  `RESIDENCE_TIME__00014=80`, and `ALLOC_MIN__00014=0.3`.
- The archived paper sensitivity reference uses
  `VCMAX25__00014=63.2061836`, `MAINT_RESP_SLOPE_C__00014=0.0876862`,
  `RESIDENCE_TIME__00014=50.6583452`, and `ALLOC_MIN__00014=0.2019211`.
- Therefore, reference `stomate_history_YYYY.nc` comparisons are strong
  numerical parity checks only when the local run uses the same parameters,
  forcing year, CO2, and restart mode as that reference file. Otherwise they
  are alignment diagnostics, not semantic failures.
