# T7 B1/B4 State JVP Diagnostic Proposal

## Purpose

Test whether the three repeatable sample-3 state-gradient separations at coordinates 434, 2382, and 2621 remain in native JVPs, using coordinate 200 as a predeclared control. This diagnostic localizes an AD-route discrepancy; it does not establish its cause, scientific impact, or route adoption.

## Fixed Method

- Use the exact saved candidate latent32/private-reader64 parameters and the four existing `TRAIN` windows with prepared artifact IDs `742`, `275`, `482`, and `299`.
- Reuse the unchanged public checkpointed recurrent objective, guarded turnover implementation, FP32 neural computation, FP64 physics, and declared hard physical/mask checks.
- B1 evaluates `objective / 4` once inside its JVP; the resulting B1 tangent is already on the B4 mean-loss scale. B4 differentiates the existing `mean(vmap(public_objective))`. Each unit direction is zero outside sample 3.
- Evaluate four directions (434, 2382, 2621, 200), in four alternating cycles per route: 32 scalar JVP calls total. Compile at most one new JVP executable for each route. Do not compile or recompile either saved VJP executable.
- Load the exact saved B1 candidate and B4 mean-objective executables. Run one full-state VJP baseline before and after the JVP cycles for each route: four saved VJP calls total. Validate complete finite gradients, auxiliary values, and physical/mask gates, and preserve complete beginning/end state baselines.
- Compare the already-scaled B1 JVPs directly with B4 mean-objective JVPs; divide only the saved B1 VJP projection by four before comparison. Compare each route's JVPs against its saved VJP projection at both endpoints, compare the route projections with each other, and place the three discrepancy-coordinate JVPs against their corresponding same-session paid B1/B4 observed repeat ranges. The control has no saved paid-pair range. Paid-range membership is descriptive, not an acceptance gate.
- Keep strict parity at `rtol=1e-5`, `atol=1e-6`. Preserve directions, identities, executable files, full VJP results, all JVP values, timing, memory statistics, and the final comparison report under `outputs/diagnostics/shared-latent-reader-20261004/`.

## Resource Proposal

Proposed free-node target: one V100 on `gln01`; at most 3600 GPU-seconds total occupation, including deployed-runtime preflight, setup and asset loads, both JVP graph compilations, all calls, preservation, and release. No paid fallback is authorized. This proposal does not reserve the node or guarantee a GPU is idle.

Caps: 180 seconds for lowering and 900 seconds for backend compilation per route; 1500 seconds per route including its compilation and calls; 300 seconds per saved executable load; 240 seconds for the actual deployed JAX/CUDA runtime preflight; 3060 seconds maximum probe wall; and a protected 300-second exit reserve, for 3600 GPU-seconds total. There are at most two new JVP graph compilations, four saved VJP calls, 32 JVP calls, and zero optimizer updates. The original B1 state JVP recorded 81.110 seconds lowering and 279.696 seconds backend compilation; this is historical context only and does not guarantee the candidate B1 or B4 compile duration.

## Approval Boundary

`T7-state-jvp-20261009.approval.json` is the deliberately false template used for CPU preparation. The separate `T7-state-jvp-20261009.approved.json` records the user's approved free-node scope and is imported only by the execute path. CPU readiness on `test01` and shell dry-run on `gln01` do not consume GPU time or imply runtime or scientific acceptance. Keep the paid and free occupation scopes separate.

## Commands

- CPU metadata-only readiness: `bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-state-jvp-20261009/run_shared_latent_T7_state_jvp.sh cpu-readiness`
- Target shell expansion check: `bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-state-jvp-20261009/run_shared_latent_T7_state_jvp.sh execute --dry-run`
- Approved GPU execution, after preparation acceptance: `T7_STATE_JVP_GPU_INDEX=1 bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-state-jvp-20261009/run_shared_latent_T7_state_jvp.sh execute`

## Preparation Artifacts

- `scripts/hpc/probe_shared_latent_T7_state_jvp.py`
- `scripts/hpc/shared_latent_T7_state_jvp.py`
- `scripts/hpc/run_shared_latent_T7_state_jvp.py`
- `scripts/hpc/run_shared_latent_T7_state_jvp.sh`
- `tests/unit/test_shared_latent_T7_state_jvp.py`
- `outputs/diagnostics/shared-latent-reader-20261004/T7-state-jvp-20261009.approval.json` (false template)
- `outputs/diagnostics/shared-latent-reader-20261004/T7-state-jvp-20261009.approved.json` (approved execute record)
