# Stage 3 Fortran Oracle - Next Session Plan

Date prepared: 2026-07-13

## Starting checkpoint

- Target ledger: 48 active/conditional PFT14 process entries.
- Formally accepted pilot baseline: 7 entries.
- New path-level compiled evidence: 32 additional entries across eight fragments.
- Path evidence must not be promoted until its reachable Fortran arms are mapped and covered.
- Known failed/partial assets:
  - `driver_lifecycle_pending`
  - `hydrol_soil_owner_cwrr_water_root_peat_alt`
  - `stomate_soilcarbon_owner`
- Known Fortran undefined contract:
  - `thermosoil_main` has an uninitialized local `LOGICAL, SAVE :: ok_zimov` selector.
- Model fixes already made during Oracle work:
  - HYDROL reinfiltration writes `ru_ns` after subtracting `water2infilt`.
  - HYDROL bare evaporation aggregation uses `soiltile`, not `mask_soiltile`.
  - STOMATE prescribe cold-start sapling initialization has an explicit landpoint axis.

## Batch 1 - Repair the certification foundation

1. Normalize every `oracle_families/*.yaml` fragment to the v2 manifest schema.
2. Keep only complete, passed families under `families`; move partial work under `pending_families`.
3. Make `run_fortran_micro_oracles.py --family all --verify` discover and run all fragments.
4. Remove retained `debug.exe`, generated harnesses, and `.mod` files from evidence directories.
5. Generate a source-driven mapping from each ledger entry to scientific PFT14-reachable control-flow arms using:
   - `fortran_control_flow_inventory.json`
   - `pft14_arm_disposition.json`
   - the ledger Fortran file/procedure/line spans.
6. Split statuses explicitly:
   - `path_verified`
   - `branch_complete_fortran_oracle`
   - `partial`
   - `fortran_undefined_contract`

Acceptance: the unified audit runs without manual exclusions and cannot count an entry when required arms are absent or uncovered.

## Batch 2 - Re-certify existing Oracle assets

1. Review every existing family against its mapped arm set.
2. Preserve passing numeric assets; do not rebuild a harness unless its source boundary or inputs are invalid.
3. Convert hard-coded case descriptions into complete deterministic input assets or record the generated input hash and full arrays.
4. Downgrade over-broad claims, especially:
   - SECHIBA module-order static audit
   - restart schema/encoding-only evidence
   - large STOMATE processes tested with one path
5. For small procedures, add the minimal missing threshold/mask cases and promote them immediately.

Acceptance: every promoted entry has source hashes, actual compiler metadata, full comparison assets, and a case-to-arm coverage record.

## Batch 3 - Close current semantic blockers

Run three independent workstreams:

1. HYDROL owner
   - Replace temporary litter thresholds with the real `hydrol_var_init` state/formula.
   - Re-run the existing five-case owner harness.
   - Close `evap_bare_lim` and add regression tests for the two JAX fixes.

2. Soil carbon owner
   - Continue from the first divergence at active-pool `carbon_32l` layer 12.
   - Determine the exact Fortran vertical update range and fix JAX if required.
   - Close soilcarbon, TF-DOC, and active-layer first/later state writeback.

3. THERMOSOIL recurrence
   - Preserve the verified true/false Zimov arms and coefficient recurrence.
   - Do not guess the undefined selector.
   - Record the entry as defined-domain-complete plus source undefined contract unless paper compiler evidence establishes the selector value.

Acceptance: no failed comparison is promoted and each model fix has a focused regression.

## Batch 4 - Fill only genuine branch gaps

- Parallelize by disjoint family files after Batch 1 produces the authoritative missing-arm report.
- Reuse existing compile units and add cases, rather than creating one program per arm.
- Prioritize PFT14 branches reachable through forcing, landpoint, and parameter changes.
- Explicitly classify PFT14-impossible branches; do not execute irrelevant crop/DGVM/other-PFT paths.

Acceptance: all defined PFT14-reachable arms are covered or linked to an explicit source-level inactive/undefined disposition.

## Final Stage 3 gate

Run:

```powershell
conda run -n ORCJAX python scripts/dev/run_fortran_micro_oracles.py --family all --verify
conda run -n ORCJAX python scripts/dev/audit_fortran_oracle_coverage.py --require-complete
conda run -n ORCJAX python scripts/dev/audit_pft14_equivalence_certificate.py
conda run -n ORCJAX pytest -q <oracle and affected model tests>
conda run -n ORCJAX ruff check <changed files>
```

Stage 3 ends only when all defined PFT14 process behavior is branch-complete. The `ok_zimov` source defect must remain visible in the certificate and cannot be converted into a guessed numerical pass.

## Budget checkpoints

- Batch 1: CNY 8-15
- Batch 2: CNY 12-20
- Batch 3: CNY 15-30
- Batch 4 and final regression: CNY 10-20
- Expected total: CNY 45-70

Stop and report at cumulative CNY 15 and CNY 40 equivalents. Do not exceed approximately CNY 70 without a new estimate and explicit user discussion. The estimate assumes existing path-level assets are reused and no additional large JAX semantic divergence is found.
