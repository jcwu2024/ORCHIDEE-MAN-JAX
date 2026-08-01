# Manifests

- `paper_pft14_data.yaml` defines the external forcing, static, and reference
  directory contract.
- `landpoints_669.json` records the paper mosaic point IDs, selected parameter
  set, and compact annual Fortran targets used by acceptance aggregation.
- `coarse_graining/ok_leak_auxiliary_capture_96day_v1.json` is the frozen,
  train-only 96-day plan for bounded exact-OK_LEAK driver capture.

Large NetCDF, restart, forcing, and output files are never embedded in these
manifests. Data distributions should publish a separate checksum inventory
when their final archival location is chosen.
