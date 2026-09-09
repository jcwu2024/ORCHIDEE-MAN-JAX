# Corrected snow-free head: bounded network-gradient gate

Status: completed and passed at `e6a00b2`; admits the corrected-head train-only
discarded-step integration, not retained training or model promotion.

The preceding forward screen passes snow-state and score consistency, but
science regresses in two windows. This gate establishes whether the corrected
network can supply reliable training derivatives; it does not promote accuracy.

## Model, objective and directions

Use accepted conditioned-Attention recovery update 3, native v6 inputs and the
default-off warm snow-free boundary enabled explicitly. Rebuild the head
before creating every production, direct, state and horizon gradient helper.
The forward-screen helper that only replaced forward functions is unsuitable.
Keep the Attention architecture, parameters, data and loss definitions fixed.

Use the same train windows: artifact 322/windows 5 and 1 (7 days), artifact
442/window 0 (30 days). Recompute the global bias totals from all eight 7-day
windows or all four 30-day windows of that artifact with the corrected head.
Never reuse the historical uncorrected probe's loss totals.

For each window, obtain the full streamed one-day-VJP parameter gradient and
test its negative unit direction. Also test the saved historical 442/0 probe
direction on the 30-day window. These are diagnostic perturbations, not
optimizer updates or retained weights. Use central differences at 1e-4 and
1e-5, accounting for the actual float32-rounded parameter displacements.
Evaluate JVP and reverse projections on each pair's same effective direction.

## Computation and gates

Record the canonical compact daily trajectory. Differentiate the existing
observation loss in state, GPP and PFT-fraction coordinates. Propagate the
complete future-state adjoint from the last day to the first; no truncation.
Carry parameter tangents forward through the identical one-day function at
the recorded carries. One cached daily reverse and one daily JVP executable
serve all equal-shaped days/windows. Do not compile whole-30-day reverse.

Every case must retain zero physical hard failures; declared dynamic state
mask changes follow the existing policy. Parameter perturbations must keep
defined/discrete masks and the selected warm snow-free regime unchanged.
Observation and production linearized scores must agree at atol=1e-6,
rtol=1e-5. JVP/reverse/central-secant comparisons use relative error <=1%
or absolute difference <=1e-6 for near-zero derivatives. Record all three
pairwise comparisons and the actual effective direction, not only signs.
All full reverse gradients and active tangent/score values must be finite.
Negative-gradient directions must yield a real production-loss decrease at
both declared magnitudes. Historical probe direction has no descent demand.
Report scaled JVP-primal discrepancies as localization evidence, not as a
replacement for the derivative/science gates.

## Resource and continuation boundary

One free gln01 V100, at most 60 minutes; inspect device availability first.
Reuse existing runtime, checkpoints, artifacts and compiled caches. The
one-day reverse uses the existing 24 GiB compiled-memory envelope. No new
Teacher data, paid job, architecture search, validation/monitor exposure or
checkpoint retention. Persist case-level JSON incrementally. The existing
source-rule option stays default off outside this diagnostic.

First qualify bootstrap head propagation, mixed-precision 7/30-day full
gradient/JVP/FD equivalence on a small recurrent fixture, actual HardCounts
handling, complete runner lifecycle and launcher arguments. If the real gate
passes, proceed to the bounded train-only discarded-step integration with
corrected starting losses. A failure is localized from the recorded case and
direction before another training attempt; do not lower the accuracy target.

## Local qualification

The targeted suite passes 26 tests, including the complete three-window runner
with real streamed reverse, daily JVP, dtype-rounded finite differences and
production `HardCounts`. The recurrent fixture uses float32 parameters and
float64 states, both horizons, all eight direction/epsilon pairs, recomputed
panel totals and JSON serialization. Bootstrap tests prove all five helper
bindings receive the identical corrected head, while the default stays off.
Launcher required arguments, Ruff and `git diff --check` pass. An initial toy
fixture incorrectly put a horizon axis on constant retained inputs; it was
corrected to the actual prepared-batch layout before GPU execution.

## Execution handoff

Launched on free gln01 GPU 1 (host `ibc13b02n04`, V100 32 GB), after both
devices reported 0 MiB and 0% utilization. Local and remote execution commit:
`e6a00b20a24088f137bb02fc604f71964f68cac8`. Remote shell syntax passed.
The detached wrapper reached Python `stage=bootstrap`; no sbatch was used.

Output stem under remote `runtime/outputs/diagnostics/`:
`conditioned-snow-free-gradient-e6a00b2` (`.log` and incremental `.json`).
Command: `GPU_DEVICE=1 EXPECTED_GIT_HEAD=e6a00b20a24088f137bb02fc604f71964f68cac8
OUTPUT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/conditioned-snow-free-gradient-e6a00b2.json
bash scripts/hpc/audit_conditioned_snow_free_gradient.sh`.
Working directory is `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`; timeout 60m.
Inspect this run before any restart. A completed process alone is insufficient:
read `passed`, each case, all pair comparisons and actual loss changes. If a
timeout interrupts incremental JSON, its partial cases are not full admission.

## Completed result (read 2026-09-09)

The full JSON was retrieved and parsed locally at
`outputs/conditioned-snow-free-gradient-e6a00b2.json`. It reports
`status=completed`, `passed=true`, elapsed 1402.1835 seconds (23.37 minutes).
All three cases, eight direction/epsilon pairs, sixteen perturbed forwards
and six negative-gradient descent checks pass. Zero physical hard failures,
unchanged masks, all selected warm snow-free days, finite active tangents and
matching production/observation scores throughout. Reverse compiled once;
its buffer estimate is 1,609,272,056 bytes (1.50 GiB), not measured device peak.

| Artifact/window | Days | Reverse projection at 1e-4 | Actual central secant | Actual positive-step change |
| --- | ---: | ---: | ---: | ---: |
| 322/5 | 7 | -1.2884642434 | -1.2884357637 | -0.00012856913 |
| 322/1 | 7 | -1.0619267350 | -1.0618592298 | -0.00010612342 |
| 442/0 | 30 | -2.8305290598 | -2.8304228952 | -0.00028280328 |

At 1e-5, the actual changes are -1.27887e-5, -1.05734e-5 and
-2.81814e-5 respectively. Maximum pairwise derivative relative discrepancy
over all pairs is approximately 0.0650%, below the frozen 1% threshold.
The historical 442 probe predicts and produces an increase (+2.54746 versus
+2.54754 derivative at 1e-4); its purpose is independent sign/agreement,
not descent. The maximum scaled JVP-primal gap is 0.00026952.

These scores are frozen-global-bias linearized window contributions, not
the full recomputed multi-window objective or prediction-error percentages.
The next integration must recompute the true loss at each proposed update.
This result does not reverse the preceding two accuracy regressions, prove
all climates/branches, or alter accepted recovery update 3. No weights were
retained; no paid job was submitted. The next contract is
[`conditioned_snow_free_integration_20260909.md`](conditioned_snow_free_integration_20260909.md).
