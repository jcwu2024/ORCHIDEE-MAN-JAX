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

## Free-node execution

Launched on 2026-09-11 around 12:36 China time, free gln01/ibc13b02n04 GPU 1,
source **cdf7eb3**, Python PID **102459**, 60-minute cap. GPU 1 was idle before
launch; GPU 0 held another process and was left untouched. Working directory:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.

Command: `GPU_DEVICE=1 bash scripts/hpc/run_fresh_start_one_day.sh`.
Report: `runtime/outputs/diagnostics/fresh-start-one-day-v1.json`.
Log: `runtime/logs/fresh-start-one-day-v1.log`.
Optional final prototype checkpoint: report path with `.checkpoint.pkl` suffix.
This first execution failed; details and correction are below. No paid GPU
task was submitted.

## First real-data attempt and interface correction

The cdf7eb3 execution ended after 1064.4s, with zero updates. Bootstrap took
1045.7s and produced all 109,464,375 freshly initialized trainable parameters.
The first complete objective trace then raised
`ValueError: trajectory PFT context must have shape (1, 1, 2)`.
State and GPP had received their one-day axes, while PFT fractions/active masks
had not. This is a new caller wiring error, not a failed learned model or a
measured gradient/memory failure. The previous local test covered complete
process loss and a bound day, but missed the combined science-loss adapter.

Commit cb3c454 adds the missing day axes through one shared extraction helper.
A regression test runs the actual physical extractor under JIT, verifies area
weighting (0.2*2 + 0.8*10 = 8.4), its input gradient [0.2,0.8], stock shape and
inactive-PFT masking. It passes in 3.92s. Ruff and whitespace checks pass.

A `--shape-only` CPU mode now traces the complete objective on the three actual
prepared samples before another GPU run. Free test02 execution at cb3c454
passed in 218.69s, within its 15-minute cap, with report
`runtime/outputs/diagnostics/fresh-start-one-day-shapes-v2.json`. It performs
fresh initialization and abstract loss tracing, no forward model execution,
gradient updates or training. All three complete losses have scalar output;
all 109,464,375 parameters are in scope. Do not infer numerical acceptance from shapes.

The selected Teacher rows are point 217.0-109.0/year 1963/day 6,
331.0-083.0/year 1982/day 100, and 299.0-071.0/year 1986/day 115. They are three
existing train samples, not the prior three rollout windows themselves.

## Corrected free-GPU rerun

The CPU preflight passed. The unchanged bounded protocol is now running on
free gln01 GPU 1 at **cb3c454**, with a new output path preserving v1 evidence:
`runtime/outputs/diagnostics/fresh-start-one-day-v2.json` and
`runtime/logs/fresh-start-one-day-v2.log`. GPU 1 was idle (3 MiB, 0%) before
launch; GPU 0 was occupied and left untouched. Same 60-minute cap, seed,
three samples, full-neural-parameter scope, losses, and at most 12 updates
only after numerical/physical/derivative admission. No paid job submitted.

Command from the project root:
`GPU_DEVICE=1 OUTPUT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/fresh-start-one-day-v2.json bash scripts/hpc/run_fresh_start_one_day.sh`.
Next read the v2 report; do not duplicate this running process or update its
server checkout. CPU report also copied locally to
`outputs/fresh-start-one-day-shapes-v2.json`.
