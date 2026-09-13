# Fixed-panel forward error attribution

User approved attribution of the failed trained7/30-day screen, alongside
native T30 AD. Keep final update1998 parameters, all16 selected windows,
forcing and retained processes unchanged. No training, new Teacher data,
architecture revision or paid task.

Use the existing public canonical rollout to reproduce the free trajectory.
Replay each daily transition from its saved free-running carry to expose the
conservative inventory start, fast endpoint, predicted incoming/outgoing and
post-tail endpoint. Separately evaluate the same daily operator from each
prepared Teacher day-start state. This reset path is diagnostic teacher forcing,
not a candidate inference path, acceptance trajectory or new Teacher execution.

Report daily GPP/NPP/maintenance/growth/Rh on all three paths: Teacher label,
free-running model and daily-reset model. Reset minus Teacher isolates the
one-day conditional prediction error; free minus reset measures the effect of
the accumulated recurrent-state difference, not a unique causal state owner.
Preserve daily curves rather than inferring an attribution from aggregate loss.

Independently reduce litter_above, litter_below, carbon_32l, DOC and canopy
interception storage to area-weighted carbon units. Sum scientific axes exactly
once; for non-SOC pools select only carbon element0. Check against the existing
shared ecosystem-minus-live extractor. Fortran soilcarbon.f90:1446 sums the
carbon_32l depth stocks directly. Diagnostic accounting changes no physical unit.
The current endpoint representation assigns decreases to external outgoing
and increases to external incoming (neural_process_head.py:124-143); its zero
internal-transfer budget is not a GPP/Ra/Rh-linked ecosystem conservation proof.

Split fast change = fast endpoint - day start, retained change = final - fast
endpoint. Retain incoming/outgoing and the endpoint identity residual. PFT weights
for each within-day decomposition are held to that day's canonical input context.
Stock trajectory weights retain the original shared-extractor convention.
Check diagnostic replay against original canonical states and GPP, and original
GPP against the saved completed screen. Numerical mismatches must be assessed
before treating small differences as mechanisms. Save full state arrays so later
field attribution does not rerun model trajectories.

Local accounting tests and actual-data CPU shape/pool extraction check precede
one free GPU1 run,60-minute cap plus tiny runtime preflight. Actual shell
argument expansion is checked before launch. The AD main checkout stays688e14c;
use a separate runtime/checkouts/trained-forward-attribution-v1 checkout.
This is a diagnosis, not a newly loosened model acceptance gate.
