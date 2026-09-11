# Fresh-start one-day full-network prototype v1

Protocol is frozen before real-data execution in
`manifests/coarse_graining/fresh_start_one_day_probe_v1.json`. This is an
engineering/learnability probe, not final scientific training.

Keep the existing 109,464,375-parameter Attention topology, exact daily binding
and accepted warm snow-free boundary. Split seed 20260911 into four constructor
keys for the target-oriented base, maintenance, structured GPP and carbon
feedback; initialize the 16 attention temperatures analytically as before.
Constructors assemble the final topology but perform no pretraining and load no
weights. All final neural parameters enter the optimizer, including shared
encoders, DOC state/process heads and learned maintenance. Existing v6 data and
statistics are read-only. The CLI has no inherited-checkpoint argument.

The provisional input protocol applies the tested C1 log tail with T=16 to the
named DOC field on every neural input route. Other features retain legacy
arithmetic. Use it consistently from initialization through the saved result.
This remains a candidate until actual new-model numerical/learning admission.

## Data and loss

Use sample zero of the one-day batches from existing train/train artifacts
322, 442 and 883; report the exact point/year/day metadata before computing.
One update accumulates gradients from these three independent samples, with
parameters held fixed until the mean gradient is applied. This small fixed
set tests basic learning, not data coverage or validation. The already prepared
2,000-package extension remains available for subsequent wider training.

The loss is daily independent carbon-flux normalized MSE (coefficient 1), plus
the existing complete eligible one-day state anchor (0.1), plus the existing
category-weighted necessary-process supervision (0.01). Fluxes are GPP,
maintenance, growth and heterotrophic respiration, using frozen training
scales. NPP, total autotrophic respiration and AGB/BGB changes are also reported
in physical units. There is no redundant independent NPP loss. No 7/30-day
objective or custom reverse recurrence is used in this first prototype.

AdamW: LR 1e-5 constant, beta1=.9, beta2=.999, epsilon=1e-8, weight decay=0,
global clipping norm=1.0. This is a declared short-probe setting, not evidence
that 1.0 is optimal; record raw norms and clipping at every update. Twelve
updates maximum, fixed sample order, monitors at 0/1/4/8/12. No post-hoc
learning-rate search or checkpoint selection. Save final full parameters and
optimizer with protocol/source identity, explicitly as a prototype.

## Admission and stopping

Local full parameter initialization/forward/process-loss/gradient and bound
one-day budgets precede real-v6 free-node execution. On real data first check
all three initial forwards; then compile one-day complete autodiff and report
its buffer estimate before executing it. The 24 GiB compiled-live envelope is
not total process VRAM. Report actual allocator memory separately when available.
Check full-gradient finiteness and module participation; inactive coordinates
need not have nonzero derivatives on every sample. Compare reverse projections
with JVP and finite differences at epsilons 1e-3 and 3e-4, using the actual float32
rounded parameter directions, atol=1e-6 and rtol=1%. Physical/mask checks also
apply to perturbed evaluations. Proceed to the 12 updates only on admission.

Stop on nonfinite values, undeclared mask/discrete discrepancies, physical
budget violations or failed derivative agreement; record the actual reason.
The final learning screen requires total and independent-flux losses to both
fall with hard gates valid. It does not require matching the old checkpoint,
per-field 95/99% reductions, or a generalization claim. Failure must be located
at the new route; do not silently fall back to frozen 15.96M training.

One idle free gln01 GPU, at most 60 minutes; no paid submission. The wrapper
uses the existing canonical environment/container and records code/device.
The data/schema loader reuses existing v6-loading code with its unused legacy
probe initialization explicitly disabled. No candidate weight is inherited.
The numerical kernels
are shared model code, not an unexplained checkpoint chain.

## Local implementation admission

Two local tests pass in 336.45s: all-family optimizer updates, and actual
109,464,375-parameter fresh initialization with DOC conditioning, finite
complete process loss/full-network gradient, participating state/maintenance/
GPP families and one canonical bound day with valid water/carbon budgets.
The local retained tail is the established plumbing adapter; real retained
physics and the combined science/state/process gradient are checked on v6
before the first optimizer update. No full-real-model gradient claim follows
from the local process-loss test alone.
