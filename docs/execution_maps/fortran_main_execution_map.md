# Fortran Main Execution Map

Source truth:

- `fortran_source/ORCHIDEE`
- `fortran_run_scripts/paper_250919`
- `configs/orchidee_man_250919.yaml`

This map follows the offline driver path requested for the clean JAX rebuild.

## Driver To SECHIBA

1. `src_driver/dim2_driver.f90`
   - `PROGRAM driver`
   - calls `intersurf_initialize_2d`: lines 1111-1125
   - calls `intersurf_main_2d`: lines 1293-1309

2. `src_sechiba/intersurf.f90`
   - `intersurf_initialize_2d`: lines 113-443
   - calls `sechiba_initialize`: lines 377-387
   - `intersurf_main_2d`: lines 458-761
   - calls `sechiba_main`: lines 640-648

3. `src_sechiba/sechiba.f90`
   - `sechiba_initialize`: lines 480-791
   - `sechiba_main`: lines 829-1786
   - `sechiba_finalize`: lines 1800-1923
   - `sechiba_main` calls `sechiba_finalize` on restart write: lines 1780-1784

## SECHIBA Initialize Order

File: `src_sechiba/sechiba.f90`

1. `sechiba_init`: line 602
2. `slowproc_initialize`: lines 605-617
3. `diffuco_initialize`: lines 635-637
4. `enerbil_initialize`: lines 640-644
5. `hydrol_initialize` when `hydrol_cwrr`: lines 664-688
6. `condveg_initialize`: lines 691-698
7. `thermosoil_initialize` when `hydrol_cwrr`: lines 701-730
8. `routing_initialize` when `river_routing .AND. nbp_glo .GT. 1`: lines 749-755

Inactive in the reference branch:

- `hydrolc_initialize`: lines 648-657
- `thermosoilc_initialize`: lines 731-738

## SECHIBA Main Order

File: `src_sechiba/sechiba.f90`

1. `sechiba_var_init`: line 982
2. `diffuco_main`: lines 997-1005
3. `enerbil_main`: lines 1013-1019
4. `hydrol_main` when `hydrol_cwrr`: lines 1046-1072
5. `enerbil_fusion` only when `.NOT. ok_explicitsnow`: lines 1077-1081
6. `condveg_main`: lines 1085-1091
7. `thermosoil_main` when `hydrol_cwrr`: lines 1093-1118
8. `slowproc_main`: lines 1184-1216
9. `routing_main` when `river_routing .AND. nbp_glo .GT. 1`: lines 1227-1234
10. `sechiba_finalize` at restart write: lines 1780-1784

Inactive in the reference branch:

- `hydrolc_main`: lines 1027-1040
- `thermosoilc_main`: lines 1119-1128

## SECHIBA Finalize Order

File: `src_sechiba/sechiba.f90`

1. `diffuco_finalize`: lines 1838-1839
2. `enerbil_finalize`: lines 1841-1844
3. `hydrol_finalize` when `hydrol_cwrr`: lines 1857-1866
4. `condveg_finalize`: lines 1870-1871
5. `thermosoil_finalize` when `hydrol_cwrr`: lines 1874-1876
6. `routing_finalize` when `river_routing .AND. nbp_glo .GT. 1`: lines 1882-1885
7. `slowproc_finalize`: lines 1906-1913

## STOMATE Entry Through SLOWPROC

File: `src_sechiba/slowproc.f90`

- `slowproc_initialize` calls `stomate_initialize` when `ok_stomate`: lines 278-290
- `slowproc_main` calls `stomate_main` when `ok_stomate`: lines 936-973
- `slowproc_finalize` calls `stomate_finalize` when `ok_stomate`: lines 1273-1281

## Active Reference Branches

- `STOMATE_OK_STOMATE = y`: STOMATE path active.
- `HYDROL_CWRR = y`: use `hydrol` and `thermosoil`, not Choisnel `hydrolc`.
- `PEAT_HYDRO = y`: peat-aware hydrology code enabled.
- `OK_PEAT = n`: two-layer peat carbon branch inactive.
- `OK_PC = n`: deep peat carbon branch inactive.
- `OK_LEAK = y`: STOMATE leak carbon path active.
- `RIVER_ROUTING = y`: routing flag active, but calls still require `nbp_glo > 1`.
- `tides = y`: tide logic active.
- `STOMATE_OK_DGVM = n`: DGVM-specific dynamics inactive.
