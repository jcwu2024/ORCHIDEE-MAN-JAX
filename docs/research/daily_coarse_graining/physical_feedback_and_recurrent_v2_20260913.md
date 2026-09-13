# Physical feedback balance and recurrent training v2

Status: user authorizes implementation and bounded free checks. No new paid
training is approved. Architecture, v6 source, input conditioning and physical
daily boundary remain unchanged. This document specifies the next diagnostic
candidate before optimizer updates; it is not a final paper-training protocol.

## Evidence and mechanism

Expansion recovery15061586 completed and the346539-day TRAIN catalogue passes
merge/readback. Current final1998 passes T7 and11/13 T30 keys. The remaining
nonliving carbon bias is-9986.906 gC/m2; summing the saved daily canopy residuals
accounts for-9277.601 (92.90%). This is accounting, not an intervention claim.
Other litter/SOC/DOC contributions are-709.305 combined. The dominant window
contributes64% of signed panel drift, so report individual windows as well as
the mean. The source-direct canopy land-area reduction remains unchanged.

Litterpart input scales are1 for bare coordinates and0.007890912519 for PFT14
coordinates. Using them as squared-loss denominators weights the latter16058.5
times more. The near-common-output optimum is0.9999377; the observed near0.9964
prediction is compatible with this geometry. Coordinates are distinguishable
by the architecture, so replacing Attention or hardcoding bare values is not
the next mechanism. Input normalization itself need not change.

## Declared next loss candidate

For a Teacher-start one-day sample keep the existing base objective, and use
`base + 0.1 * canopy_pseudo_huber + 0.1 * litterpart_physical_mse`.
The two0.1 coefficients are a declared bounded trial, not claimed optimal:
the auxiliary terms should correct physically scaled state errors alongside
the unit-weight base. Do not select them retrospectively from monitor results.

Litterpart uses scale1 for every valid coordinate, retaining masks and both
bare/vegetated source slots. Input channels, output decoder and labels are
unchanged. All physical coordinate errors are treated equally before averaging.

Canopy uses the existing frozen TRAIN daily-change scale5.40392606553 gC/m2
and the pseudo-Huber penalty `sqrt(1+r*r)-1` for normalized change residualr.
Its slope is bounded and nonzero for large errors: unlike the previous MSE
with coefficient3e-9 selected near a huge fresh error, it can retain a useful
correction after training while limiting extreme initial output gradients.
It is quadratic near zero (one-half r squared) and asymptotically linear.
Do not change source units, mask semantics, labels or physical gates.

Before an actual update run, require local loss/mask/gradient tests and the
deployed forward/loss/gradient plus two complete optimizer updates on declared
TRAIN samples. Report each physical coordinate and canopy change, not merely
the combined objective. Fail numerical/physical invalidity; report scientific
non-improvement honestly instead of changing tolerance or auto-extending.

## Mixed recurrent objective and reproducibility

Implement recurrent science supervision through the public rollout loss and
native checkpointed JAX AD, using the same shared science extractor as reporting.
The model carries its own state through each complete window. Teacher supplies
initial state and targets, never daily resets inside a recurrent window.
Use daily and cumulative GPP/NPP/autotrophic/heterotrophic respiration errors,
AGB/BGB stock-change errors and source-direct canopy-stock-change errors.
Normalize stock change by declared daily-change scales and elapsed days, not
large standing stocks. Keep parameter/state/fraction gradient paths intact.

First specify and test this common scalar locally; actual full-network
directional/update admission follows on TRAIN windows disjoint from the fixed
monitor/development panel. Loss weights, window sampler and stage transition
must be frozen before training; do not use development windows for updates.
Begin with T7, then T30 after resource/update evidence. Historical custom
streamed reverse discrepancies do not block this native route.

The next bounded recurrent-scalar specification is
manifests/coarse_graining/recurrent_science_probe_v2.json: each of four fluxes
gets0.125 for daily MSE and0.125 for horizon-mean cumulative MSE. AGB/BGB
each get0.025 for elapsed-normalized trajectory and endpoint increment error;
canopy gets0.05 for each, with the same pseudo-Huber penalty. Frozen TRAIN
state-delta statistics supply biomass change scales, and source PFT14 supplies
canopy change scale. These are declared diagnostic weights, not a final mixed
training budget. A subsequent combined-update stage still needs admission.

One TRAIN T7 case is selected before evaluation using seed20260914, excluding
all64 monitor days and all296 fixed developmental window days. Prepared old
multiday source manifests retain their existing identities; no new Teacher
data. The first selected T7 is used without choosing by results. Final1998
weights are explicitly a diagnostic asset for native-gradient/resource probing,
not a formal training initialization. Reverse cap65min, then JVP cap35min on
one free GPU, preserving completed native.pkl before verification. It measures
two directions, finite/physical outcomes and resources, and executes zero
optimizer updates. New actual recurrent scalar evidence is required; the old
diagnostic scalar's pass is not claimed to certify it.

Current checkpoints can be diagnostic assets. The final paper route remains
fresh initialization/all-network optimization under a consolidated executable
recipe, with any one-day then mixed stages and transitions specified up front.
The v2 bounded diagnostic does not automatically become that final recipe.

## Expanded fixed batches

New optional packing consumes all346539 source rows per epoch and pads only
the final11-row batch with21 deterministic replays. Total10830 full32-row
updates/346560 draws, micro4; report unique coverage separately from padding.
The old sampler and historical protocol remain unchanged. New runs share the
same disk pool and sampler cursor; no per-update/window file duplication.
Full TRAIN cache planning retains the user's10-million
filesystem-entry ceiling; actual eligible TRAIN scope is8568025 days.
