# Science-first continuation and canopy-carbon metric correction

## User decision

After bounded trained T7/T30 native-checkpointed AD agreement, gradient tooling
is no longer the main research task. Prefer native JAX AD for the next actual
multi-day training objective. When that objective is specified, verify its
gradient and a few complete parameter updates, including science improvement,
memory and speed. Historical custom streamed-reverse discrepancies are retained
as unresolved evidence, not prerequisites to this science work. Do not rerun
expensive unchanged tests or seek bitwise equality as an independent objective.

Main work: identify the GPP recurrent-state mediator and audit canopy organic
carbon labels/normalization/supervision. Keep the current Attention architecture
and final update1998 diagnostic checkpoint. Future formal training remains a
reproducible declared fresh-initialization/full-network method; exploratory
diagnoses are not retroactively a formal recipe.

## Source-confirmed metric correction

Fortran `stomate_soilcarbon.f90::soilcarbon_leak` treats litter, SOC and DOC as
per-PFT-area stocks, but interception_storage already represents land-area
carbon associated with each PFT. Evidence:

- line1225 multiplies dry deposition by veget_max before adding to canopy stock;
- lines1245-1251 multiply SOC by veget_max but add interception_storage directly;
- lines1497-1525 implement the canopy deposition/washoff stock update;
- lines2326-2332 multiply DOC by veget_max but add interception_storage directly.

JAX source port preserves the deposition weighting in
`jax_orchidee/stomate/soilcarbon_kernels.py:1559-1562`. The daily science extractor
incorrectly applied another PFT fraction to interception_storage. Correct the
research extractor to sum the carbon element over active PFT slots without
another cover factor; maintain the existing factors for litter/SOC/DOC. The
independent host accounting helper receives the same source correction.

This changes derived ecosystem/nonliving metrics, not Teacher data, model
states, neural outputs, GPP/NPP/AGB/BGB or the earlier diagnostic AD scalar.
Preserve old reports explicitly as the former weighted convention. Corrected
comparisons can be derived from saved states without new model trajectories.
The selected T30 Teacher PFT14 fraction is1, so this semantic error does not
explain the large observed raw canopy drift. Bare-PFT model stock must still
be included according to its validity, not hidden by a zero fractional area.

One source regression checks no second cover dependence, carbon-element-only
selection, active PFT semantics and the absence of a spurious canopy derivative
with respect to cover. It passes. The other11 accounting/long-horizon tests
pass unchanged. No repeated large gradient compilation is warranted for this
correction to an unused output of the admitted diagnostic scalar.

## Parallel task ownership

Sol(high) implements frozen, bidirectional same-day state-group interventions
using saved free/Teacher starts and the existing canonical daily boundary.
Terra(high) audits canopy endpoint labels, scales and actual supervision,
primarily from saved arrays. Root verifies evidence and integrates the changes.
No paid job, architecture replacement or new training follows automatically.
