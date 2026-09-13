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

## Source audit before numerical attribution

Independent review finds no obvious stock-unit or double-counting error in
the current extractor: Fortran stomate_soilcarbon.f90:1446 sums carbon_32l
over depth;1843-1913 explicitly converts DOC stocks to water concentration
then back before state writeback. The saved DOC therefore needs no additional
layer-thickness multiplication. PFT weighting still needs numerical comparison.

Current direct carbon outgoing heads use sigmoid(raw), without mean centering
(output_transform.py:44-46); incoming heads use mean/scale-centered softplus.
This makes a persistent overestimated outgoing fraction a source-supported
hypothesis, not yet a measured cause. The one-day loss (fresh_start_training.py)
is daily flux MSE +0.1 state anchor +0.01 essential-process supervision, with
no dedicated nonliving-stock-change or ecosystem-budget-coupling term. Thus
slow carbon is supervised, but improvement of total loss does not establish
accurate slow-carbon increments. Do not change decoder/loss on source evidence
alone; attribute actual predicted changes first.

Keep compact state-layout metadata and full start/fast/incoming/outgoing
inventory arrays in the diagnostic archives. This permits later pool/layer,
effective-outgoing-fraction and fixed-PFT-weight analysis without new forward runs.

## Execution record

Three local accounting tests and Ruff pass. Independent review found no blocking
layout, carry-boundary or accounting issue; explicit science-validity counts and
reset-defined masks are retained. Actual-data CPU shapes and independent pool
extraction pass199.3s on test02 at fcbb035. Reporting-only follow-up a16ea78
adds inventory archives and validity counts, leaving that numerical graph unchanged.
Actual target-shell argument expansion passes. Free GPU1 launcher PID105072
uses isolated `runtime/checkouts/trained-forward-attribution-v1` at a16ea78.
Log/report are `runtime/outputs/diagnostics/trained-forward-attribution-v1-pipeline.log`
and `trained-forward-attribution-v1.json`. Keep the checkout fixed until completion.

## Completed numerical attribution

All16 windows finish in468.207s, exit0. All shared flux/stock validity counts and
canonical hard counts are zero. Canonical GPP exactly reproduces the prior
screen. Independent pool sums match the shared extractor within2.91e-11gC/m2.
Separate daily replay does not pass whole-state allclose: maximum raw state
error0.05011; pool error0.004708gC/m2; raw PFT GPP error3.47e-6. Retain these
limits; they cannot explain the thousands of gC/m2 inventory drift but preclude
claiming exact all-state replay equality.

| Horizon/output | Daily-reset RMSE | Free-run RMSE | Reset aggregate bias | Free aggregate bias |
| --- | --- | --- | --- | --- |
| T7 GPP | 0.40614 | 1.18555 | -2.257% | -30.535% |
| T7 NPP | 0.30269 | 0.85186 | -4.743% | -65.719% |
| T30 GPP | 0.10116 | 0.87144 | -0.179% | -31.581% |
| T30 NPP | 0.06116 | 0.61604 | +1.110% | -68.727% |

RMSE units are gC/m2/day; bias is sum residual/sum Teacher, not mean window
percentages. T7/T30 are different selected panels, so this is not a horizon
scaling comparison. Reset success shows conditional fitting is substantially
better on this panel, not that all one-day training is complete. Recurrence
worsens accuracy, but no specific state owner has yet been causally isolated.

Mean T30 pool inventories in gC/m2:

| Pool | Start | Teacher final | Free final | Predicted fast change |
| --- | --- | --- | --- | --- |
| litter_above | 478.96 | 480.16 | 312.28 | -166.68 |
| litter_below | 1725.32 | 1730.31 | 1275.24 | -450.08 |
| carbon_32l | 0.00145 | 0.00146 | 13.94 | +13.94 |
| DOC | 868.65 | 851.45 | 613.52 | -255.12 |
| interception_storage | 42078.70 | 42214.98 | 31011.25 | -11067.45 |

All five measured retained-tail inventory changes are exactly zero; endpoint
identity residual max2.36e-11. Canopy interception organic carbon contributes
11203.73/12050.67=92.972% of the final nonliving carbon error. Even on daily
Teacher reset its mean predicted fast change is-275.13gC/m2/day, so its drain
is not solely created by feeding back erroneous model states. These values
describe the existing Teacher/model configuration; they are not observations
of real-world canopy carbon or evidence that canopy drift causes GPP decline.

Next: identify the GPP feedback owner through targeted same-forcing state-group
replacements; inspect canopy endpoint incoming/outgoing supervision against
existing labels before deciding a decoder/loss correction. Use archived arrays
for field/layer and weight decomposition, with no repeated AD or Teacher runs.
