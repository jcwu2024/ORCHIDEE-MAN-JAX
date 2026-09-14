# Thirty-day mixed update admission

Declared before execution. The completed T7 mixed admission proves finite
full-network updates and small same-draw loss reductions. It does not prove
T30 gradients, memory, or useful trained accuracy. This bounded extension
changes the recurrent horizon to30 days while preserving the Attention model,
fresh seed20260911, all109,464,375 trainable parameters, AdamW, physical-v2 daily
loss, recurrent weights and frozen TRAIN scales.

Machine protocol: `manifests/coarse_graining/mixed_physical_recurrent_t30_admission_v1.json`.
Reuse the first128 expanded TRAIN daily rows, micro8/batch64, in two updates.
Select the first four T30 entries in the existing accepted recurrent TRAIN
selection (filter by horizon before taking four), pairing two per update.
Both objectives use their own means and outer weights1+1; clip and update once.

The four selected T30 windows are fixed from the accepted metadata:

| Artifact | Site | Year | Start row | Window index |
| --- | --- | ---: | ---: | ---: |
| 1199 | 261.0-105.0 | 1970 | 187 | 1 |
| 1538 | 295.0-113.0 | 1971 | 37 | 2 |
| 2477 | 297.0-069.0 | 2004 | 170 | 1 |
| 1071 | 295.0-085.0 | 1987 | 249 | 1 |

Daily state is predicted recursively through30 days; no Teacher reset, new
Teacher data, new architecture, or custom observation cotangents.

Before reverse, check all four fresh T30 trajectories and the existing fixed
daily TRAIN panels. Reuse native checkpointed AD. Preserve the first actual
T30 native gradient before compiling a JVP check. Compare two Rademacher
directional projections of that identical scalar using the existing1% relative
and1e-6 absolute tolerances and a resolved derivative. Then complete two actual
mixed AdamW updates and evaluate the same panels after both proposals. A changed
horizon does not certify the revised T30 scalar merely from historical tests
of another diagnostic scalar; no historical streamed-reverse repair is needed.

Acceptance: finite meaningful gradients, directional agreement, two complete
finite optimizer steps and zero physical/validity failures. Report same-draw
loss changes, flux RMSE in gC/m²/day, cumulative flux and biomass/canopy changes
in gC/m², gradient norms/alignment and actual live versus reserved GPU memory.
Retain the distinction between technical admission and useful learning. Zero
heterotrophic-respiration Teacher values do not prove a learned nonzero response.

Use free gln01 GPU0 or1 when idle, one GPU at a time, a90-minute actual-model
limit, pool fraction0.85 and24GiB per-executable memory envelope. Reuse the
existing compilation cache and unchanged T7 results. No paid submission and
no reusable model checkpoint in this admission. The two diagnostic gradient
artifacts are container files, not per-day outputs. Keep T7 output files intact.

After T30 passes, the next work is a declared learning run on the expanded
346,539-day TRAIN pool with rotating TRAIN windows, not another two-update
benchmark. Freeze its update budget, sampling, monitor/checkpoint schedule and
science interpretation before running; use actual T30 timing for the resource
estimate. Existing exploratory checkpoints must not become the default start.

## Execution record

Completed: exit0, two updates, both native/JVP directions and all physical
gates pass. Final evidence and limitations are in
`mixed_physical_recurrent_t30_results_20260914.md`. The launch record below is
historical; Bash18684/Python19718 have exited.

Commit70dd976 passes41 focused mixed/recurrent/rollout tests. Actual test02
shapes, including the optional T30 JVP and full optimizer, pass398.4s;
target-shell argument expansion passes. Free gln01 GPU0 Bash18684/Python19718 launched
from immutable `runtime/checkouts/mixed-physical-recurrent-t30-v1`.
Log/report: `runtime/outputs/diagnostics/mixed-physical-recurrent-t30-v1-gpu.{log,json}`.
Deployed GPU runtime check passes48.8s (JAX0.4.38/NumPy2.4.6), and actual-model
bootstrap is running. Numerical/update results are pending. No paid submission.
