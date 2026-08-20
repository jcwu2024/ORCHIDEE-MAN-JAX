# DOC Status And Branch-Magnitude Attention v5 Local Acceptance

Date: 2026-08-21.

Status: accepted local foundation. The no-checkpoint 10-epoch early screen is
prepared but has not been submitted or executed.

## Accepted Mechanism

`coordinate_coupled_doc_status_branch_magnitude_attention_v5` preserves the
complete v4 coordinate-aligned Attention backbone and changes only its final
DOC endpoint decoder. Each DOC inventory coordinate now emits three status
logits plus separate incoming and outgoing magnitude coordinates. Softmax
status probabilities gate a centered-softplus incoming magnitude and a
centered-sigmoid outgoing magnitude. The unchanged conservative updater and
retained exact tail consume the resulting bounded physical endpoints.

The official loss directly supervises status and both active-branch latent
coordinates, then adds a status-macro physical endpoint term. These four terms
are equally weighted. The ordinary pooled DOC endpoint contribution is removed
before the revised DOC composite is inserted, so it is not duplicated.

The initialized model has exactly `59,973,309` parameters, below the frozen
62-million engineering memory ceiling. The manifest SHA256 is
`46f442b2f003af80b4037ba8128e1543f1c031305ed844c9d6f30f7d4ea4142e`.

## Local Evidence

The focused V5 suite passed `6` tests in `313.60` seconds. It covers exact
parameter/head shapes, status probability simplexes, bounded physical
endpoints, eager/JIT forward, the canonical `bind_daily_operator` path, one
complete day of state and water/carbon budgets, inactive-PFT isolation, strict
PFT permutation equivariance, and nonzero gradients through the status head,
both magnitude heads, all five aligned source projections, and inventory
Attention.

The unchanged v4 regression suite passed `7` tests in `319.06` seconds. The
V5 core results were obtained before the training-entry integration; that
integration changed only the runner, launcher, and their tests. The resulting
runner suite passes `24` tests. Ruff, Python byte compilation, Bash syntax, and
`git diff --check` also pass.

## Executable Early Gate

The shared one-point runner now selects V5 only with
`--doc-status-branch-magnitude`, requires the mechanistic litter/POC/DOC path,
uses the V5 loss, enforces the exact parameter count, and reports the V5 schema
separately from v3/v4. The launcher uses
`DOC_STATUS_BRANCH_MAGNITUDE=1` and rejects simultaneous v3 or v4 selection.

At epoch 5 the runner evaluates the complete point-year process breakdown and
one-day state/budget metrics. It stops without a checkpoint if any hard gate
fails or either DOC endpoint loss exceeds twice its initialization. If that
review passes, epoch 10 requires at least `90%` litter/POC/DOC aggregate loss
reduction, at least `50%` reduction for each DOC endpoint, no regression of
free-to-adsorbed or advective transfer, identical loss counts, finite
gradients, and zero hard failures. These are absolute gates; no new V4 control
is trained.

The prepared Slurm envelope is one `gnall` V100, four CPUs, and two hours. Its
worst-case additive estimate is CPU CNY `0.56` plus GPU CNY `4.40`, total CNY
`4.96`. The script is
`scripts/hpc/slurm_doc_status_branch_magnitude_v5_early_screen.sh`. It writes no
checkpoint and has not been submitted.

## Boundary And Data Policy

The candidate enters only through the canonical daily binding, uses native
daily inputs and the accepted v6 Teacher release, regenerates no Teacher data,
uses no historical v5 rows, reconstructs no 48 half-hour forcing records, and
executes no 48-step candidate state scan.

The active gate is explicit approval and execution of the prepared 10-epoch
screen. A 100-epoch run opens only if every early gate passes. Multi-point
training, rollout, sealed-test access, Gate D2, and product acceptance remain
closed.
