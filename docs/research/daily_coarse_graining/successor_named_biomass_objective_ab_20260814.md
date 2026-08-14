# Named-Biomass Objective A/B Decision, 2026-08-14

## Decision

Reject and close the fixed-coefficient named-AGB/BGB tendency objective. Do
not add training updates to this objective and do not use this result to open
365-day, long-held-point, Gate D2, sealed-test, or product validation.

This is a loss-mechanism rejection, not an Attention-architecture rejection.
The run used the same `source_routed_process_attention_v2` architecture,
update-1000 parameters, optimizer state, point-year groups, anchor samples,
rollout windows, and learning rates for both objectives.

## Frozen Protocol

- contract SHA256:
  `c161a7de7414b3f95784347a77337cb98162b81b36ca798879528fe3cbcc784a`;
- control: accepted staged-v3 objective;
- candidate: selected long-horizon fluxes plus named AGB/BGB tendencies;
- updates: 100 per objective, optimizer updates 1001-1100;
- sampling: four distinct point-years, 32 one-day anchors, and four 30-day
  windows per update;
- validation: the existing 48-window temporal/spatial/joint non-test panel at
  7 and 30 days;
- hard gates: state, mask, discrete-state, and budget failures must be zero;
- no Teacher regeneration, historical v5 data, checkpoint, recovery, paid
  node, or sealed-test use.

## Result

The corrected free-`gln01` run at commit `79953c6` completed all 100 updates
for both objectives and the complete validation panel. Report:

- runtime path:
  `outputs/training/successor-named-biomass-objective-ab-79953c6/report.json`;
- file SHA256:
  `fb6f6f695db59a165d6c6c664507d37d3d696f411a7bec521445edc68f03b89d`;
- elapsed Python time: `2114.89 s`;
- peak JAX device bytes: `13,226,032,128`;
- hard failures: zero for control and candidate.

The research-primary composite is the RMS over NPP, AGB, BGB, and total live
biomass across all six slice/horizon cells.

| Metric | Control | Candidate | Candidate change |
|---|---:|---:|---:|
| Four-key composite | 0.872112 | 0.967749 | +10.97% |
| NPP | 1.743572 | 1.934779 | +10.97% |
| AGB | 0.032711 | 0.033931 | +3.73% |
| BGB | 0.011236 | 0.016515 | +46.98% |
| Total live biomass | 0.032826 | 0.036810 | +12.14% |
| Maintenance respiration | 0.586628 | 0.404486 | -31.05% |

The candidate improves `0/6` research-primary slice/horizon cells. Every cell
regresses, from `+6.47%` for temporal 7-day to `+37.89%` for spatial 30-day.
Maintenance improves in all six cells and passes both maintenance guards, but
that cannot compensate for systematic regression of all four research-primary
outputs. Consequently every research-primary admission check fails while all
hard and maintenance checks pass.

Gradient clipping is report-only under the frozen contract. It occurred on
`47/100` control updates and `93/100` candidate updates. At update 100, the
reported raw gradient norm is `0.559` for control and `13.625` for candidate.
The one-day anchor at that update is lower for candidate (`0.3955`) than
control (`0.4409`), yet the candidate is worse on every long-horizon primary
cell. This is evidence that the calibrated fixed tendency weight produces an
optimization direction that is poorly aligned with the validation objective;
it is not evidence that more updates will repair the result.

Post-run, report-only attribution strengthens that interpretation. Across all
100 paired updates, candidate raw gradient norm has median `6.587` and 90th
percentile `42.530`, versus control median `0.927` and 90th percentile `4.859`.
Candidate clipping is `21/25`, `22/25`, `25/25`, and `25/25` across the four
successive quarters; its quarter-median raw norm rises to `13.954` in updates
76-100. The named tendency terms account for a median `73.9%` of the
candidate's weighted long-horizon contribution, and the complete long-horizon
part accounts for a median `47.4%` of total candidate loss. On the identical
paired batches, candidate one-day anchor is worse than control on `73/100`
updates with median difference `+0.0130`. These are descriptive statistics
over changing scheduled batches, not a replacement for fixed-panel
validation, but they show persistent tendency dominance rather than an
underweighted biomass signal.

## Operational Record

The first run at commit `7a7f195` completed all updates and validation
computations but failed before report serialization because the runner read a
nonexistent `scale_asset.asset_sha256` attribute. No checkpoint was authorized
or written, so that in-memory result could not be recovered. Commit `79953c6`
uses the admitted asset payload and file-hash fields and adds a direct unit
test. The complete run was then repeated from the same update-1000 recovery.
The failed run used only free `gln01`; its log remains under
`outputs/training/successor-named-biomass-objective-ab-7a7f195/`.

## Next Gate

Do not tune the tendency coefficient or continue this candidate. The next
gate is design review of a materially different multi-objective training
mechanism that prevents one long-horizon component from dominating the update
and treats cross-output gradient conflict explicitly. Freeze its mechanism,
matched control, and stop gates before implementation or another GPU run.
Architecture remains an open research dimension, but this experiment alone
does not justify excluding or replacing Attention.
