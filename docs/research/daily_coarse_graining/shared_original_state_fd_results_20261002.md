# Original T7 state finite-difference diagnosis

Completed 2026-10-02: **inconclusive for all three selected failed coordinates
and the passing control.** This bounded check cannot independently adjudicate
the original-versus-V4 strict derivative differences. It does not validate or
invalidate either AD result, waive a strict gate, reopen a closed candidate, or
change the objective/precision/masks/training method.

## Executed scope

Only the existing original **default-policy** T7 dual-AD executable ran:
`shared-recurrent-checkpoint-policy-20260930-v2-T7.reference.executable.pkl`.
The report's historical `candidate_policy` metadata describes its separate
selective candidate; its `mode=reference` and original source/identity establish
the saved binary used here. No selective/V4/R4/G2 candidate executable ran.
Saved V4 gradient vectors were read for comparison only.

Inputs remain update0, artifact742/window0, all64,884,481 parameters/300 leaves,
the public T7 scalar and complete parameter/state AD, FP32 neural and FP64
physical precision. Only one named FP64 initial-state element changes per call;
parameters, days, targets, discrete states and other initial coordinates stay
fixed. Source/profile/method/checkpoint/admission identity and serialized input
trees/shapes/dtypes were checked through the existing `_assets` and
`validate_inputs` utilities. Saved preflight from the completed CSE screen was
reused on the unchanged canonical JAX0.4.38 runtime.

The completed attempt made59 original-binary calls: two baseline repeats,
seven plus/minus step pairs at each of four coordinates, then a final baseline.
Every call synchronizes the whole saved dual-gradient result; there are no new
forward/gradient/model compilations. A compile guard covers saved executable
loading and all model evaluations. It recorded0 forbidden attempts. Unchanged
import-time physical-table eager bootstrap precedes that guard.

The first baseline matches all20 saved loss/auxiliary coordinates bitwise.
All64,884,481 parameter derivatives pass the existing strict comparison
(`rtol=1e-5, atol=1e-6`; maximum absolute difference2.980232e-8). The full
initial-state saved replay has one known self-drift failure; it is retained
independently. All initial-state derivatives are finite on every call. Baseline
parameter derivatives were also checked finite. Three baseline losses are
identically0.08665354178956992; this does not bound perturbed neural FP32 noise.

## Domain and mask finding

**All four selected values are exactly zero and sit on a nonnegative domain
boundary.** The original central plan therefore supplied infeasible negative
values. Those raw results remain preserved, but central slopes are discarded
as physical derivative evidence. The lead's direction to stop further central
negative calls arrived after the already running59-call batch finished. No
additional GPU calls were made; right-sided first-order and nonuniform
three-point second-order slopes use saved positive losses and baselines only.

| Coordinate | Semantic field/index | Source-backed domain |
| --- | --- | --- |
| 33 | `diffuco_previous_step_state.snowgrain[0,0]` | Snow grain size in metres; `explicitsnow.f90::explicitsnow_initialize`, lines56–86, declares grain size and initializes zero |
| 199 | `hydrol_previous_step_state.canopy2ground[0,0]` | `hydrol.f90::hydrol_canop`, lines4992–5094, explicitly sets `MAX(precisol-precip2ground,zero)` |
| 428 | `hydrol_previous_step_state.profil_froz_hydro_ns[0,4,3]` | `hydrol.f90::hydrol_soil_froz`, lines8332–8426, computes the frozen fraction `1-x` with `x` in `[0,1]` |
| 200, control | `hydrol_previous_step_state.canopy2ground[0,1]` | Same nonnegative canopy-to-ground flux as199; saved V4/original seven pairs pass at this coordinate |

Positive initial perturbations respect these scalar domains. This does not
establish unchanged internal branch decisions, or that an independently
perturbed coordinate lies on the complete source-generated state manifold.
The right derivative at a boundary is not the default unconstrained two-sided
AD derivative. Neither can be used here to declare an AD implementation correct.

All14 exposed physical hard counts are0 for all59 calls, including the
infeasible negative inputs. `declared_mask_change=14` stays fixed. These aggregate
checks therefore do not certify the negative domain or hidden branch stability.
Initial defined masks use the exact canonical assembler rule from
`input_assembly.py::_defined`, lines51–54: finite and absolute value below
`0.5e20`; `assemble`, lines165–185, assigns `other=continuous`. Selected
coordinates are defined. Original masks, PFT masks/fractions, target/science
masks and declared-dynamic state masks are saved, rather than recomputed by a
new model graph. Full internal rollout masks/branch traces are not returned by
the existing executable and were not observed.

## Right-sided numerical result

Step sizes decrease across seven values in the ratio sequence
`0.03,0.01,0.003,0.001,0.0003,0.0001,0.00003` of the selected local magnitude
`max(abs(initial),abs(input_scale),1e-8)`. Actual rounded plus/minus increments
and asymmetry, all losses/auxiliary/hard/mask-count outputs and selected native
gradients are preserved. Offline analysis also retains every positive loss
difference, first-order slope and six nonuniform three-point second-order
slopes. No convenient epsilon is used to announce a pass.

| Coordinate | Current original AD median | Saved V4 AD median | Seven positive h, largest → smallest | Right first-order slopes, largest h → smallest h |
| --- | ---: | ---: | --- | --- |
| 33 | 0.368628711 | 0.368810892 | 4.653816e-8 → 4.653816e-11 | 73.1866,75.8322,74.3456,63.3953,63.6376,27.0678,36.2806 |
| 199 | -0.0491823504 | -0.0491852588 | 0.03 → 3e-5 | 7.04947e-6,2.07536e-6,-4.62319e-7,-3.88836e-5,-1.26534e-4,-3.04960e-4,-7.03408e-4 |
| 428 | 1.226487403 | 1.226458796 | 1.302913e-7 → 1.302913e-10 | 4.76699,1.66128,-0.40390,-1.25718,-6.51770,-17.36964,-92.77913 |
| 200, control | 972.166724966 | 972.167794412 | 1.052980e-5 → 1.052980e-8 | 2.29950,6.84682,17.41006,34.19527,74.39986,168.65752,362.21362 |

Each result is inconclusive for a concrete reason:

- **33:** no stable right-sided slope platform; later values fluctuate strongly.
  The smallest adjacent slope change is0.24224, far above the original/V4
  derivative gap. Current native self range is5.29885e-5, also above the
  strict allowance4.68629e-6. Second-order slopes range8.78298–77.15496.
- **199:** slopes change sign and then grow in magnitude as h decreases; no
  convergence platform supports a local derivative band. Second-order slopes
  continue changing from approximately-4.12e-7 to-8.74e-4. The observed sequence
  cannot resolve the current original/V4 gap2.90843e-6 under allowance1.49182e-6.
- **428:** slopes change sign and grow from4.76699 to-92.77913; second-order
  values also diverge in the explored sequence. The minimum adjacent change
  is0.85328, far above gap2.86075e-5 and allowance1.32649e-5.
- **200 control:** first- and second-order slopes keep growing with decreasing
  h; neither reaches a stable platform. Even a coordinate passing the historic
  strict AD comparison is not independently resolved by this bounded FD
  sequence. Its historic gap0.00106945 is below allowance0.00972267.

No reliable uncertainty interval is assigned: `stable_resolution_band=null`
for every coordinate. FP64 scalar ULP and zero baseline loss variation are
only output-representation observations; they do not cover neural FP32
intermediate quantization. Quantization, local nonlinearity and hidden branch
effects were not independently separated. Large FD/AD differences do not by
themselves establish an incorrect AD derivative, particularly at these domains.
The old strict gate remains unchanged and the closed candidates remain closed.

## Artifacts and runtime

Compact local directory: `outputs/diagnostics/shared-original-state-fd-20261002/`.
Key artifacts are `results-v3.json`, `right-side-analysis.json`,
`results-v3.losses-and-state-gradients.pkl`, `results-v3.fixed-masks.pkl`, source
snapshots, logs and actual-exit records. The raw central data remains inside
these results. Historical original/candidate vectors were read from
`shared-command-buffer-loops-whole-vmap-T7-default-20261001-v4.*.repeat{0..6}.initial_state_vjp.pkl`;
no candidate binary was loaded. Repeated300-leaf parameter outputs were not
downloaded or materialized as additional files.

Remote overlay/results use matching `shared-original-state-fd-20261002`
directories under `runtime/transfer/` and `runtime/outputs/diagnostics/`, using
immutable `runtime/checkouts/shared-split-width-v1` at `c599236bcf57b745e9bafed678298010f4c3b58f`.
Existing prepared bundle/profile/context and update0 checkpoint were reused.

Free gln01 actual host `ibc13b02n04`, one V100 GPU1, driver525.85.12,
JAX0.4.38, OMP4. Before each attempt both GPUs were0MiB/0% and no compute
processes were visible. Final-attempt host load0.18/0.63/0.44, affinity0–111.
Target Bash syntax and actual dry-run expansion passed. Each process has a
420s external timeout/10s kill grace and410s internal bound after initialization.

Attempt1 exited1 after3.266256s at0 binary calls: the guard was placed before
unchanged import-time table primitives. Attempt2 exited1 after100.823017s at0
calls: canonical input does not contain a redundant defined mask. Both errors
and exits remain preserved. Ordinary initialization/source-rule fixes led to
attempt3 exit0: Python197.701736s, launcher201.152042s,59 calls totaling93.730974s.
All three launchers total305.241315s / **0.084789254 one-GPU card-hours**.
Model compilations0, candidate executions0, paid jobs0, training updates0.
Estimated live memory3,488,289,277bytes; observed process peak3,507,481,856bytes,
below24GiB. Right-sided saved-array analysis ran locally with no GPU calls.
