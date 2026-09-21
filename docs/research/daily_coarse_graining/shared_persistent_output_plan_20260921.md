# Shared output coordinate policy v3 (2026-09-21)

This isolated admission changes output coordinates, not the shared Transformer,
capacity, targets, loss weights, data split or seed. It follows the completed
remaining-state and cimean source audits. No paid training is authorized.

Eight `endpoint.ok_leak_aux` readouts (two lignin fractions, litterpart,
dead_leaves and four fuel pools) use their corresponding
`slowproc_stomate_previous_step_state` day-start inventory plus a learned
daily residual. The canonical typed endpoint remains absolute. Its existing
targets and loss normalization remain unchanged. Reuse the full TRAIN
state-delta count/mean/variance: source ownership and a 512-row, 16-artifact
identity check show that the paper retained tail leaves all eight fast
endpoints unchanged (maximum difference zero). This is not all-branch proof.

Pool moments over the declared compact PFT axis, including between-population
mean differences, then broadcast them to preserve PFT permutation equivariance.
Residual scale is `max(sqrt(pooled variance), 1e-12)` in native units. Unlike
the generic statistics unit-scale fallback, a zero variance does not create
order-one daily changes. The positive floor retains a trainable residual;
zero variance in the bounded sample never establishes a constant field.
Use the existing nonnegative/unit-interval transform around start plus TRAIN
mean, retaining its documented numerical interior floors and full range.

Bare-soil cimean is materialized as current annual atmospheric CO2 from the
legal canonical input, identified by `role_bare_soil`, not compact position.
The physical endpoint is Ca even across a year boundary; its delta from the
actual day-start state need not be zero. Vegetation values/gradients and all
active masks are preserved. No half-hour forcing reconstruction is involved.

Admission: test pooled statistics, ownership, physical bounds, raw/state/Ca
gradients, masks, PFT permutation and year boundaries; then actual-data CPU
shape/calibration and matched free-GPU forward on the same two v2 TRAIN rows.
Report every hard counter, dominant state errors and carbon science groups.
Only after valid forward values run the changed full-model native AD and two
optimizer updates, followed by T7/T30. Old v1/v2 remain reproducible; their
saved gradients cannot certify v3. A smaller fresh loss is initialization
admission, not learned accuracy or long-term scientific success.

Implementation evidence: local full161-output shared-head integration plus
residual/recovery checks passed20tests in261.67s; bare-Ca gradient/mask/year
boundary checks passed4tests in4.32s. Ruff and diff whitespace checks pass.
All eight actual owner slices have8,591,565 observed TRAIN values per
coordinate, matching compact PFT order(0,13), with finite means/variances.
No full-TRAIN zero-count columns occur in these eight owner slices. Existing
bounded identity report: `outputs/shared-endpoint-sample-audit-20260921.json`.

Deployed immutable158192f to `runtime/checkouts/shared-persistent-158192f`.
Actual test02 shape admission wrapper32673 completed0 in371.73s, with full
forward/gradient shapes and unchanged75,578,497 parameters/161fields/39,006
coordinates. Report `outputs/shared-persistent-158192f-cpu.json`.
GPU1 was idle immediately before launch. Free gln01 forward wrapper46671 is
running with30min main-test bound (plus separate5min runtime check), prefix
`runtime/outputs/diagnostics/shared-persistent-158192f-forward`. Target-shell
CPU/GPU dry-runs passed before execution. Inspect completed report first;
v3 native numerical AD/two updates still follow a valid paired forward.
