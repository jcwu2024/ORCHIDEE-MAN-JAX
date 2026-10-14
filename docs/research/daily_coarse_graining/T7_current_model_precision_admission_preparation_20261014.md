# Current-model precision admission preparation

2026-10-14: **not a real-model executable package**. The requested fixed B1
artifact299/window0/T7 FP64 plus independent physical-direction FD diagnostic
cannot yet be specified for all four directions: coordinate2621 lacks a verified
physical domain for an independent unit perturbation. This means evidence is
missing, not that every such perturbation has been proved illegal. No real
model graph, executable, GPU proposal or run follows. Local disconnected
precision/FD guards and counterexamples are delivered instead.

## Exact object and hypotheses

The intended object is the existing66,123,713-parameter/355-leaf
latent32/private-reader64 field/chunk64 model, with unchanged saved weights,
normalization/calibration, physical formulas/masks and seven-day public
`make_shared_recurrent_objective` → `RolloutLoss.bind_checkpointed_loss_with_aux`
loss `(flux+stock)/9`. Use native unscaled B1 loss/full3854-state VJP and dynamic
unit directions434/2382/2621/200; mean-scale comparisons remain separate.
Existing FP32 results supply the comparison; no new parameter-gradient graph
is selected. The old64,884,481/artifact742 FP6432-call pass had no independent
FD and is neither this model's admission nor derivative truth.

[Saved identity bindings](../../../outputs/T7-current-model-precision-admission-preparation-20261014/preparation-status.json)
retain c599236bcf57b745e9bafed678298010f4c3b58f plus the executed transformer/
latent/guarded-turnover overlays, weight SHA `4f95d18d…9806c14`, context
`ff3fdabe…be9026`, prepared299 `ee7fa6e0…1f18aa`, full case `8c0ed07c…820448`
and initial state `54d01d0f…624f5d`. Parameter initialization provenance names742;
the intended evaluated input is299. These are reused saved metadata identities,
not a new full asset/model admission. The selected rebuild/assembler/materializer
sources have no difference from c599236 in this checkout.

The diagnostic hypothesis would be that exact-widened FP64 native AD agrees
with independently converged directional FD while saved FP32 differs. JVP=VJP
alone cannot establish that hypothesis or identify all root causes. A precision
or production-interface change would still require scientific discussion.

## Bounded source and route findings

| Direction | Source/domain evidence | Permitted conclusion in this preparation |
|---|---|---|
|434, frozen fraction0|`hydrol.f90::hydrol_soil_froz:8317–8454`: `profil_froz_hydro_ns=1-x`, with frozen fraction bounds and upper limiting. | Negative central endpoint is outside the nonnegative domain. Positive-side FD would require a certified initial-state domain and can describe only a right derivative at the boundary. |
|2382, carbon_32l3.224057494e-8|Material pool: `stomate_soilcarbon.f90:2916–2918,2941–2945,2992–2996` stops on negative stocks. A numeric central radius strictly below the positive stock can be proposed. | This is a numeric nonnegative candidate, not independent physical-domain or internal-branch certification. A predeclared grid must be checked independently of AD agreement. |
|2621, deepC_peat6.329964314e-7|`stomate_soilcarbon.f90:1385–1405` reconstructs the quantity from three same-layer carbon_32l pools. `fast_owner_reconstruction.py:61,368–370` marks source-exact derivation and likewise rebuilds that sum. | Independent unit-direction physical legality is unverified. Coordinating pool changes would change the requested direction and is not selected. Default FD classification is unsupported-domain/inconclusive. |
|200, canopy2ground0|`hydrol.f90:5017,5073–5094` uses `MAX(precisol-precip2ground,0)`. | Negative central endpoint is inadmissible; right derivative does not automatically equal an AD branch convention. Initial-state physical certification remains necessary. |

The selected public input path `successor_training.py:161–184` accepts the
continuous vector after width/PFT-view checks. Array acceptance by that consumer
does not certify independent physical perturbations. The contract/source review
does not establish an allowed2621 domain, and therefore cannot close the requested
four-direction physical FD specification. This is a bounded review of selected
writers/consumers, not an all-physics branch audit.
None of the four real-model physical FD domains is admitted by this preparation.
The parent graph is ready with known-excluded daily scopes; exact source fallback
supports this bounded review without reindexing main.

The old diagnostic precision helper lists eight modules but omits
`shared_latent_reader`. Its current line277 FP32 scatter narrows decoded values;
lines261/290/329–330/386 also create FP32 buffers. Exact parameter/context
widening alone cannot produce the intended full-neural FP64 graph. Any future
integration must include the real reader and check residual low-precision tensors
in the single admitted lowering. Exact R4 `2**-23` widening preserves that source
threshold; replacing it with FP64 machine epsilon changes branch semantics.

The old combined native pattern (parallel FP64 probe:242–247) can, in principle,
return public loss/aux, full-state VJP and dynamic JVP from one compiled graph.
Perturbed-state calls could supply scalar losses to FD without a second compile;
every such call would still perform AD. FD estimates must use only scalar forward
loss differences, with steps/convergence selected before consulting AD. No current
full-graph precision, memory, timing or public-loss binding admission was performed.

## Delivered method guards and interpretation

`scripts/hpc/shared_current_T7_precision_guard.py` provides exact FP32 array
widening, unchanged mask/integer storage, R4 constant checks, current-object
metadata rejection and reader AST coverage. Its temporary patch accepts only
synthetic fixture modules, restores bindings and stays disabled by default.
It deliberately cannot patch or execute the real package.

`scripts/hpc/shared_current_T7_fd_guard.py` consumes supplied scalar samples,
an explicit domain/branch certificate and a predeclared descending step grid.
It rejects unsupported domains, illegal central endpoints, unknown/changed
branches, nonfinite/cancellation-limited quotients and missing convergence.
Output masks alone do not certify all internal branches; unknown real branch
coverage is inconclusive. Stable consecutive steps are required before comparing
AD, with fixed `rtol=1e-5, atol=1e-6`; no best-step/AD-guided selection or /4 rescue.
State20's source-undefined NaN remains a separate strict failure, never filled
or exempted. These methods are disconnected from the current public objective.

| Future evidence, if separately specified | Allowed interpretation |
|---|---|
|Known legal smooth domain, independently converged FD, FP64 JVP/VJP both agree, saved native FP32 differs|Bounded precision sensitivity for that model/loss/direction, not universal truth or production FP64 acceptance.|
|Only FP64 JVP=VJP|Internal consistency only; independent correctness unestablished.|
|Unsupported domain, unknown/changed branch, no credible convergence or boundary-only right derivative|Inconclusive for the requested two-sided AD/precision judgment; preserve all four direction records.|
|State20 remains nonfinite, even if a loss derivative is finite|Separate strict state admission fails; derivative diagnostics cannot waive it.|

CPU fixtures passed24 precision and19 FD tests in ORCJAX, with only the existing
pytest `cache_dir` warning when its cache plugin is disabled. They validate
exact widening/narrowing counterexample, R4 branch
semantics, wrong model/case/normalization rejection, default restoration,
polynomial FD, boundary/domain/branch/nonconvergence rejection and native-scale
handling. They establish method behavior only, not real-model AD correctness,
public binding or deployed JAX0.4.38/CUDA admission. Logs and bounded static
reviews are in `outputs/T7-current-model-precision-admission-preparation-20261014/`.
The original FD10-test log and later v2 intermediate/final results are retained;
the final guard rejects tolerance overrides and lacks any unknown-branch pass.

## Stop and unresolved decision

The supervisor needs to decide whether a source-supported independent2621
initial-state domain exists, or whether a constrained multi-coordinate manifold
direction is scientifically appropriate. The latter changes the original target
and cannot be introduced automatically. This preparation stops at that explicit
boundary; no next-layer diagnosis or new science/resource scope starts.

The requested future design ceiling was one nonexclusive V100, four actual Slurm
CPUs,90 minutes,24GiB and at most one new real FP64 compile, including preflight,
checks/FD/save/release. Current deployment rates (CNY0.07/core-hour +2.20/GPU-hour)
give `(4×0.07+2.20)×1.5 = CNY3.72`; actual billing is unknown. The old total
diagnostic occupation was46.4 minutes, including backend2328.48s (about38.8
minutes); its CPU affinity covered0–111 rather than the requested four actual
Slurm CPUs. That timing does not guarantee feasibility. This is an unexecuted
design boundary, not a ready proposal: no executable command/SHA or current-graph
admission is supplied, and no approval or submission is requested.
