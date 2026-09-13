# Canopy Inventory Supervision Audit, 2026-09-13

## Scope

This is a CPU-only reading of the completed `trained-forward-attribution-v1`
T30 archives. It does not rerun the checkpoint, regenerate Teacher data, alter
the architecture, or change a loss. The fixed checkpoint is the completed
fresh-one-day-pool job `15048786`; the archives are the eight selected T30
windows, 240 daily records total.

The executable reader is
`scripts/dev/audit_trained_canopy_inventory.py`. Its output is
`outputs/diagnostics/trained-canopy-inventory-audit-v1.json` and it uses only
the downloaded archives plus frozen v6 base/process statistics.

## Source Semantics

`interception_storage` is not hydrological canopy interception water. It is
the STOMATE TF-DOC carbon pool, declared as storage of DOC and soluble organic
carbon in the canopy in
`fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90`, `soilcarbon_leak`,
lines 731-737. The active v6 state has one element, carbon; there is no
nitrogen coordinate to sum or to confuse with carbon.

The same Fortran routine adds `interception_storage(:,m,icarbon)` directly to
the initial and final balance pools (lines 1245-1251 and 2326-2332). In
contrast, its DOC and `carbon_32l` terms are multiplied by `veget_max`. The
source therefore treats this stored canopy carbon as already on the land-area
basis used by that balance. The general diagnostic reducer's PFT-fraction
weighting is not source-consistent for this one pool. The audit preserves both
reductions for comparison; future scientific accounting must use the direct
sum for this field.

The source transition is also explicit. In the TF-DOC block (lines 1497-1525),
the carbon coordinate changes by:

```text
dry_dep_canopy + DOC_precip2canopy - DOC_canopy2ground
```

where the canopy-to-ground amount is capped by the stored carbon. This shows
why the stock is a transfer-capable organic-carbon inventory. It does not by
itself make it a GPP/Ra/Rh ecosystem-budget owner.

The same source loops from Fortran `j=2` through `nvm` for that TF-DOC update
(lines 1497-1525), excluding the one-indexed bare-soil slot. The selected
Teacher T30 records have exactly zero bare-slot start and endpoint storage.
The saved model predictions do not: each reset-path day has a strictly positive
bare-slot incoming amount and endpoint. This is a source-applicability mismatch
in the archived output, not a license to add an untested clamp. It must be
included in the typed-label probe as a separate exact-zero/applicability count.

## Archived Numerical Evidence

The selected T30 records all have PFT14 fraction exactly `1.0`, so source-direct
and historical fraction-weighted totals happen to be almost identical on this
panel. That coincidence is not a justification for retaining the incorrect
weighting rule on another panel.

| Quantity, source-consistent direct sum | Mean gC m-2 day-1 or stock |
| --- | ---: |
| Teacher canopy start stock | 42144.566 |
| Teacher endpoint change | +4.543 |
| Free-run model start stock | 35674.663 |
| Free-run predicted endpoint change | -368.855 |
| Free-run endpoint error | -6843.301 |
| Teacher-reset predicted endpoint change | -275.061 |
| Teacher-reset endpoint error | -279.603 |
| Teacher-reset predicted incoming amount | +0.159 |
| Teacher-reset predicted outgoing amount | 275.219 |

The reset trajectory starts from each Teacher day-start state. Its mean
`-275.061 gC m-2 day-1` predicted fast change therefore cannot be attributed
solely to prior model-state drift. It is a conditional one-day error for this
checkpoint and input panel. The archived endpoint identity residual remains
within the existing numerical tolerance.

For positive starts, the predicted outgoing amount divided by its same-day
start gives a PFT14 outgoing fraction with mean `0.00725125` per day. The
bare-soil slot is reported only on days with a strictly positive start; zero
starts are intentionally not divided. These are predicted fractions, not
Teacher-label errors.

## Existing Supervision and Conditioning

The v6 process schema assigns both
`inventory.carbon.interception_storage.outgoing_fraction` and
`inventory.carbon.interception_storage.incoming_amount` to the
`carbon.ok_leak_inventory_endpoints` label. The outgoing field uses the
unit-interval sigmoid direct decoder and unit scale. Incoming uses the
mean/scale-centred softplus decoder and its frozen train/train standard
deviation. The process objective pools valid coordinates within each label,
then averages active labels with the target contract's category weights (equal
within a category). The one-day prototype also multiplies process loss by0.01.

The v6 state scale for the stored carbon slots is `[1.0, 47559.687579]`
gC m-2 for bare soil and PFT14. On the first reset archive, the physical
canopy endpoint error corresponds to a canopy-only normalized MSE of
`0.001967` and a canopy-only normalized state loss of `0.001423`. Thus the
one-day state anchor substantially downscales this physically dominant stock
error. This is a scale-conditioning observation, not an argument to retune a
scale after inspecting validation outputs.

The current endpoint representation is still independent external incoming
and outgoing branches. It guarantees the per-coordinate endpoint arithmetic
but has no coupling that proves ecosystem conservation against GPP,
autotrophic respiration, or heterotrophic respiration.

## Required Follow-Up Probe

### Completed raw-v6 readback

The test02 execution at a136fda completed without a model run. Report:
`runtime/outputs/diagnostics/trained-canopy-inventory-label-audit-v1.json`
(local copy under outputs/ with the same name). All240 selected days match
the raw v6 start/terminal state exactly; canopy fast and terminal coordinates
also agree exactly. The production encoder, rather than an inferred gross
transfer decomposition, supplies the targets.

- Teacher PFT14 incoming mean4.54273245gC/m2/day; all240 outgoing targets zero.
  Bare-slot incoming and outgoing targets are zero on every selected day.
- Reset predicted PFT14 outgoing fraction mean0.0081325875, RMS0.00866934;
  positive-start outgoing MSE7.51574489e-5. The earlier0.00725125 was a
  window-specific value, not the full240-day mean.
- Incoming field normalized MSE1.07545896 across both slots; PFT14 normalized
  residual RMS1.46653529. Incoming fit is also poor, but its missing few grams
  does not explain the hundreds of grams lost through outgoing.
- Zero-start bare outgoing fractions remain unavailable from saved outgoing
  amounts (240 coordinates). No exact full-label macro or gradient attribution
  is claimed from this report.

This confirms a mismatch between physical inventory-change importance and
dimensionless outgoing-fraction error. Existing global train/train statistics
provide canopy state_delta scale5.40392607gC/m2 (mean5.54059836), compared with
next-state scale47559.68758. These are pre-existing training statistics, not
scales fitted to this diagnostic panel. A separately weighted stock-change
objective is now evidence-supported for the next loss proposal. Do not simply
replace all state scales: measure the revised objective/gradient balance on
training data first, and preserve the current science panel for assessment.

The following describes the original readback plan and remaining full-label
loss limitation; raw endpoint/label alignment is now complete.

The Teacher branch split is deterministic, not a gross-flux decomposition. The
v6 target builder calls `encode_inventory_endpoints` in
`constrained_daily_replay.py`, lines 109-158: after the default `1e-12` shift,
a decrease maps to an outgoing fraction and a non-decrease maps to an incoming
amount. `training_data.py`, lines 364-369, preserves those values and their
defined masks. Therefore a Teacher fast start/end pair is sufficient to
reconstruct the typed labels exactly.

The remaining archival limits are specific. The attribution NPZ contains the
Teacher final state, whereas the label builder consumes the fast endpoint. The
report says the five measured retained-tail *pool* changes are zero, but that
aggregate fact must not be promoted to coordinatewise fast/final equality.
One selected prepared record must establish that equality for this field before
the archived terminal state is used as its label endpoint. Also, at a zero
start the saved external outgoing *amount* is zero for every fraction; the
archive cannot reconstruct the raw predicted outgoing fraction and therefore
cannot reconstruct that raw output's loss.

The next bounded CPU probe must read only the selected existing v6 rows and:

1. Read one existing selected prepared row and verify, coordinatewise, that its
   typed fast `interception_storage` endpoint equals the saved Teacher terminal
   state. This is a data-boundary check, not a new Teacher run.
2. Reconstruct labels from the validated fast start/end rule above, including
   their defined masks. Compare the saved reset/free incoming values directly;
   recover outgoing fractions only where the matching start is strictly
   positive.
3. For the full exact outgoing-field loss, retain or re-export the direct
   predicted outgoing fractions at the zero-start bare coordinate. Its
   external amount alone is intentionally non-identifying. Apply the frozen
   process scales and the exact masked, equal-label macro calculation, and
   report the `carbon.ok_leak_inventory_endpoints` contribution before and
   after the target field is removed from that label only as an offline
   attribution, never as a changed objective.
4. Count the source-excluded bare-slot labels and predictions separately;
   preserve the current result as evidence rather than imposing a new output
   rule inside this diagnostic.
5. Retain source-direct canopy stock accounting and the historical
   fraction-weighted comparator in the same report.

No new Teacher capture, model execution, training, loss edit, decoder edit, or
paid job is required for that probe. A result showing that the exact label loss
is small despite the endpoint drain would isolate the problem to representation
or loss geometry. A large outgoing-branch loss with low total contribution
would instead demonstrate underweighting/scale dilution. Neither conclusion is
available from an endpoint pair alone.

The optional `--prepared-manifest` mode of the audit reader searches all 240
selected T30 day keys in the persisted `batch.one_day` rows of the same eight
artifacts. It returns `complete` only when every selected day has a typed
target; it returns `partial` with the missing semantic day keys otherwise.
It is intended for a small `test02` CPU run and does not import, compile, or
evaluate a model.

The preferred `--dataset-manifest` mode now reads the eight selected train/train
base shards directly. These canopy labels are constructed from stored state
and `ok_leak` fast endpoints in the base shard, not from the supplemental
gross-process labels. It uses the actual production `encode_inventory_endpoints`
(default1e-12 shift), checks fast-versus-terminal equality coordinatewise,
and measures incoming and recoverable positive-start outgoing loss on all240
days. It does not rebuild prepared data or execute the Teacher. The outgoing
loss on the zero-start bare slot remains explicitly unavailable in the archives.
