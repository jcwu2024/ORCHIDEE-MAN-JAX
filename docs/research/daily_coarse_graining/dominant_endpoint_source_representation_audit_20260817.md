# Dominant Endpoint Source And Representation Audit

Date: 2026-08-17.

Status: completed corrected audit and non-test mechanism panels; revised
architecture requires user review before implementation.

## Scope

This audit follows the failed strong-baseline one-point loss-floor diagnostic.
It reads the accepted v6 base shard for `001.0-071.0`, 1961 and the existing
train/train fast-owner statistics. It does not regenerate Teacher data, read
v5, train a network, reconstruct 48 forcing records, or scan 48 subdaily
states. The numerical audit ran on free Explore1000 node `test03`.

The executable is
`research/daily_coarse_graining/fast_owner_delta_representation_audit.py`.
The superseded server report is
`runtime/outputs/diagnostics/fast-owner-delta-representation-04b0c2e/report-source-v3.json`.
The corrected report is `report-source-v4-corrected.json` in the same
directory, with SHA256
`7e1255624a923ec1fd21c452b83a8d7784d600896d949ce94e32ac0dabf8d003`.

## What The Earlier Good-Looking Metrics Mean

The earlier GPP, maintenance respiration, and AGB one-day NRMSE values below
`0.05` show that several aggregate outputs are near the provisional one-day
threshold. They do not establish a successful daily operator:

- the complete process loss reduction is `95.4744%`, below the `99%`
  one-point memorization gate;
- BGB tendency (`0.05064`) and NPP (`0.05694`) still fail even on the training
  point-year;
- five process labels hold `91.06%` of the final process loss;
- no 7/30-day or longer rollout gate is open;
- zero state and budget hard failures only show that the constrained wrapper
  remains physically admissible, not that the learned process is accurate.

The model is therefore stable but not yet scientifically qualified.

## Corrected Endpoint Reachability Result

The first version of this audit made an incorrect inference. It compared the
65 signed `day_end - day_start` targets directly with each field's absolute
physical range and reported every negative delta of a nonnegative field as
impossible. The implemented path in `output_transform.py` instead adds the
day-start owner before applying `softplus`, `sigmoid`, or `tanh`; the loss then
subtracts day start again. A nonnegative state can therefore fall from 10 to 4,
for example, while its reconstructed endpoint remains valid.

The corrected audit reconstructs each absolute endpoint and analytically
inverts the actual decoder to obtain the required raw output. On the same 364
days it finds:

For the audited 364 days:

| Quantity | Result |
| --- | ---: |
| Delta fields | 65 |
| Defined scalar targets | 260,988 |
| Fields with hard endpoint-range violations | 0 |
| Strictly unrepresentable endpoints | 0 |
| Strictly unrepresentable fraction | 0% |
| Exact 0/1 boundary endpoints requiring infinite raw for exact equality | 24,532 |
| Interior endpoints requiring absolute raw above 20 | 34 |

The boundary count is concentrated in valid zero-valued hydrology and snow
outputs. It is an optimization and output-status question, not evidence that
negative deltas are excluded. Of the 34 high-raw interior cases, 28 are in
`thermosoil_previous_step_state.shum_ngrnd_permalong`. The current absolute
endpoint constraints are retained. A global signed-delta decoder is rejected
because it would remove useful physical protection without fixing the observed
loss floor.

## Empirically Verified Daily-Boundary Reconstruction

True v6 day-end primitive states were supplied to existing source-backed JAX
kernels. Each call used either no forcing or one selected native forcing
boundary; no subdaily trajectory was created.

| Reconstructed field | Maximum absolute error | Decision |
| --- | ---: | --- |
| `z0m` | `1.33e-15` | Exact daily-boundary reconstruction verified |
| `roughheight` | `0` | Exact reconstruction verified |
| active `roughheight_pft` | `0` | Exact reconstruction verified |
| `drysoil_frac` | `3.33e-16` | Exact reconstruction verified |
| `gtemp` | `0` | Exact reconstruction verified |
| `stempdiag` | `0` | Exact reconstruction verified |
| `tdeep` from final `ptn` | `0` | Exact reconstruction verified |
| `hsdeep` from final long humidity | `0` | Exact reconstruction verified |

`z0m` uses `condveg_main_minimal` with final vegetation/snow primitives and
the final native forcing record. `drysoil_frac` uses
`hydrol_litter_top_diagnostics` and `hydrol_litter_grid_diagnostics` with
day-end soil moisture and static soil layers. The thermal diagnostic fields
use `thermosoil_final_state` and `thermosoil_diaglev_from_ptn`.

A follow-up non-test panel evaluated eight spatial-train points and two
spatial-validation points in both 1961 and temporal-validation year 2006. All
20 point-years pass for `z0m`, `roughheight`, active `roughheight_pft`,
`drysoil_frac`, `gtemp`, `stempdiag`, `tdeep`, and `hsdeep`. Maximum errors are
`2.22e-15` for `z0m`, `4.44e-16` for `drysoil_frac`, and exactly zero for the
other admitted fields. The report SHA256 is
`0b906b0cac4c7644525bdf7e45780526d7c875c06399f575c892c8c4c7925f74`.
No spatial-test or temporal-test shard was read.

A second 20-point-year panel supplied the complete true v6 day-end primitive
boundary to the source-backed thermal sequence. All ten former coefficient
heads pass every point-year within `1e-10 + 1e-12*abs(target)`:

`cgrnd`, `cgrnd_snow`, `dgrnd`, `dgrnd_snow`, `lambda_snow`, `pcapa_en`,
`soilcap`, `soilcap_pft`, `soilflx`, and `soilflx_pft`.

The report is
`runtime/outputs/diagnostics/thermal-coefficient-exact-reconstruction-04b0c2e/report-v1.json`
with SHA256
`adbdb831e0bce9793d8a38e0d02d16b4c6df35fe824490cd4f30928ec6e579e8`.
It materialized no 48-step forcing and executed no subdaily state scan. These
ten fields are now verified exact daily-boundary derivations, not conditional
candidates.

`z0h` is not admitted as exact. The original final-record reconstruction has
maximum error `0.0017066`, and none of the five individual native records
reproduces the endpoint. It then fails all 20 panel point-years with maximum
error `0.0137343`. `z0h` remains learned; the mismatch is consistent with its
dependence on the state entering the final half-hour CONDVEG call rather than
the post-HYDROL day-end snow state available at the daily boundary.

The global train/train target statistics also show that `roughheight` and the
active `roughheight_pft` target deltas have exactly zero variance. Their
normalization falls back to scale `1`, so asking a neural head to reproduce
them creates avoidable optimization damage. In the failed run their losses
grew from about `1e-8`/`8e-8` to `0.0775`/`0.0448` despite the true delta being
zero.

`z0m` has a different conditioning issue: the audited point-year's median
delta is `0.000567`, its 99th percentile is `0.001992`, and its maximum is
`2.533` on the lifecycle boundary. Exact reconstruction removes both this
outlier burden and a redundant neural target.

## Dominant-Label Classification

| Loss owner | Classification | Required handling |
| --- | --- | --- |
| `fast_owner.diffuco` roughness and dry-soil fields | Verified exact or carry-derived, except `z0h` | Remove verified fields from neural outputs; retain `z0h` until resolved |
| `fast_owner.diffuco` evaporation, `qsurf`, `temp_sol_pft` | Coupled energy primitives/endpoints | Use one coupled energy decoder rather than independent scalar heads |
| `fast_owner.driver` `fluxsens` and `vevapp` | Coupled energy endpoints | Share the same energy primitive decoder and consistency equations |
| `fast_owner.thermosoil` final diagnostics | Verified exact from learned primitive states | Remove `gtemp`, `stempdiag`, `tdeep`, and `hsdeep` from neural outputs |
| `fast_owner.thermosoil` coefficients | Verified exact from the complete daily boundary | Remove all ten independent neural heads and execute the source sequence after learned primitives |
| `carbon.doc_vertical_diffusion` | Physical boundary baseline plus learned residual | Use the endpoint-trapezoid source magnitude as the baseline; do not claim an exact daily kernel |
| `water.canopy_inventory_endpoints` | Necessary learned endpoint | Replace signed storage/debt transition heads with capacity-normalized occupancy plus explicit rare debt/status semantics |

The thermal closure uses final learned `ptn`, hydrology/snow primitives,
static grids, and the final native forcing record. Its panel proves that the
complete inference boundary contains the required inputs and that the source
call sequence reproduces the Teacher endpoints. The exact derived fields
remain downstream diagnostics in the loss, but they no longer need trainable
heads.

The DOC claim required a different correction. The v6 label is the sum of 48
source-ordered nonnegative interface magnitudes. Diffusion direction is owned
by the adjacent concentration ordering at each call; v6 does not store those
48 orderings. A single day-start, day-end, or trapezoid boundary call is
therefore not exact. On the same 20-point-year non-test panel, the endpoint
trapezoid nevertheless reaches correlation `0.9833`, NRMSE `0.2119`, and
residual variance fraction `0.0450` overall and passes its frozen gate on all
four train/validation slices. It is accepted as a physical baseline for a
learned residual. The report and complete split metrics are recorded in
[`doc_diffusion_boundary_baseline_result_20260817.md`](doc_diffusion_boundary_baseline_result_20260817.md).

`doc_advective_interface` has the same magnitude/direction distinction: its
stored value is nonnegative and the sign of `wat_flux` owns direction. The old
`daily signed interface sum` wording is rejected for both labels. This changes
their interpretation, not the immutable v6 values.

## Architecture Consequence For Review

The evidence does not reject Attention, Transformer blocks, RNNs, or the
49.75-million-parameter axis-aligned backbone. It rejects the present output
boundary. The next candidate should be a mechanistic-output revision of the
strong process-expert baseline:

1. retain the axis-preserving encoders, process-local Attention, independent
   expert capacity, and canonical `bind_daily_operator(...)` boundary;
2. predict primitive state changes and conservative transfers while retaining
   endpoint-aware absolute physical constraints;
3. reconstruct verified roughness, hydrology, and thermal diagnostics exactly;
4. introduce a coupled energy primitive head for the jointly constrained
   `fluxsens`/evaporation/`qsurf`/surface-temperature family;
5. compose the DOC boundary baseline and learned residual without calling it
   exact, and use a capacity-aware canopy endpoint representation;
6. add explicit activity/status treatment only for genuinely boundary-heavy
   zero-valued families, rather than changing every endpoint to an unbounded
   signed decoder.

The complete proposed `149 learned primary + 16 exact derived + 1 physical
baseline/residual + 15 auxiliary/status` output inventory, energy coupling,
retained Attention blocks, and qualification gates are now specified in
[`mechanistic_output_architecture_review_v1.md`](mechanistic_output_architecture_review_v1.md).
That review is not implementation approval and must be discussed with the
user before any network or training-contract change.
