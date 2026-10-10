# Original T7 native state JVP diagnosis

Completed 2026-10-02: **the original public T7 native JVP and saved original
native VJP disagree under the unchanged strict gate.** Coordinates33 and428
fail against both fresh original VJP evaluations. Coordinate199 passes against
the first VJP but narrowly fails against the end VJP. Control200 passes both.
All eight JVP values at each coordinate are bitwise identical. This establishes
a discrepancy between these native AD executions; it does not identify which
derivative is scientifically correct, certify the historical V4 candidate,
waive a gate, or admit a changed training method.

## Executed scope and checks

One executable takes a **dynamic FP64 direction of shape[3854]** and applies
`jax.jvp(..., has_aux=True)` directly to the original public
`make_shared_recurrent_objective`. The objective is lowered with the admitted
original expert128/backbone staging adapters and unchanged default native
checkpoint policy. There is no custom AD, stop-gradient, precision change,
objective change, physical perturbation, or candidate execution. Neural
arithmetic remains FP32 and physical arithmetic remains FP64; a FP64 direction
and scalar do not make the neural computations FP64.

Inputs remain update0, artifact742/window0, T7, all64,884,481 parameters/300
leaves, original states/discrete fields/days/targets/loss context. The saved
original dual executable runs twice, before and after the JVP repetitions,
returning complete parameter and initial-state gradients. Only historical
original/V4 state vectors are read for context. Four initial basis evaluations
and seven alternating-order cycles give32 JVP calls through one compiled graph.
No additional model compilation occurs during evaluation.

Identity, shapes/dtypes and source/profile/method/checkpoint/admission bridges
pass. The first original dual replay matches all20 saved loss/auxiliary
coordinates bitwise; all64,884,481 parameter derivatives pass strict comparison
(maximum absolute difference2.980232e-8). The saved full initial-state replay
retains one known strict self-drift failure, independently of the JVP result.
The full end original result is checked finite; its parameter gradients are
not compared a second time.

Every JVP returns finite FP64 loss/tangent, all14 physical hard counts0 and
fixed `declared_mask_change=14`. All20 public primal loss/auxiliary coordinates
pass the unchanged strict bridge. JVP loss0.08665354392203385 versus original
0.08665354178956992 is not bitwise identical; maximum auxiliary bridge error is
5.821355e-8. Thus primal agreement is strict numerical agreement, not exact
identity or an internal branch trace.

The new canonical actual-GPU runtime preflight passes, as does a small native
checkpointed-scan JVP/VJP smoke:1.3451513419060286 versus1.3451513419060288,
absolute error2.220446e-16. This smoke establishes runtime support, not
real-model derivative agreement.

## Coordinate results

Strict scalar allowance is `1e-6 + 1e-5*abs(reference)` throughout. Each row has
eight identical JVP results; the table gives absolute error/allowance against
the first and end fresh VJP. Full exact metrics are in `model.json`.

| Coordinate | Native JVP | First original VJP | End original VJP | First error / allowance | End error / allowance | Strict outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 33 | 0.368439987820042 | 0.368620663881302 | 0.368647783994675 | 1.806761e-4 / 4.686207e-6 | 2.077962e-4 / 4.686478e-6 | Fails both |
| 199 | -0.0491836531091353 | -0.0491823271079559 | -0.0491821578029036 | 1.326001e-6 / 1.491823e-6 | 1.495306e-6 / 1.491822e-6 | Passes first; narrowly fails end |
| 428 | 1.22653347465716 | 1.22648835821019 | 1.22648909555573 | 4.511645e-5 / 1.326488e-5 | 4.437910e-5 / 1.326489e-5 | Fails both |
| 200, control | 972.169729038440 | 972.166774150496 | 972.166841987365 | 2.954888e-3 / 9.722668e-3 | 2.887051e-3 / 9.722668e-3 | Passes both |

Against the saved original vector,33 and428 also fail;199 and200 pass. There
are no JVP self differences across call order or repetitions. The original
VJP's first/end loss/auxiliary results are bitwise identical, but its full state
gradient has one strict self failure: coordinate33 drifts2.712011e-5. The global
maximum state-gradient difference27.346292 occurs at coordinate1785 with
magnitude approximately1.3996e7 and passes its relative allowance; it is not
the strict failed coordinate. Finite original gradients therefore need not be
strictly stable at every coordinate.

JVP stability does not remove the JVP/VJP discrepancy. Coordinate33's larger
discrepancy and428's repeatable difference persist against both fresh VJP
observations. Coordinate199's strict classification depends on the fresh VJP
observation; its end excess is approximately3.484654e-9, and the tolerance is
not changed. Control200 is a passing bounded comparison, not global AD proof.

## Domains and interpretation

These unchanged initial coordinates are all exactly zero. Source-backed
semantics from the accepted finite-difference diagnosis remain:

| Coordinate | Field/index | Fortran provenance |
| --- | --- | --- |
| 33 | `diffuco_previous_step_state.snowgrain[0,0]` | `explicitsnow.f90::explicitsnow_initialize`, lines56–86; nonnegative grain size and zero initialization |
| 199 | `hydrol_previous_step_state.canopy2ground[0,0]` | `hydrol.f90::hydrol_canop`, lines4992–5094; `MAX(precisol-precip2ground,zero)` |
| 428 | `hydrol_previous_step_state.profil_froz_hydro_ns[0,4,3]` | `hydrol.f90::hydrol_soil_froz`, lines8332–8426; frozen fraction1−x |
| 200, control | `hydrol_previous_step_state.canopy2ground[0,1]` | Same canopy-to-ground provenance as199 |

Initial defined masks follow canonical `input_assembly.py::_defined`,
lines52–54: finite and `abs<0.5e20`; `assemble` supplies `other=continuous`.
Every selected coordinate is defined. Native JVP evaluates at the original
state using JAX's derivative conventions and does not cross the nonnegative
domains by perturbation. It still does not establish a physical right
derivative, source-generated state-manifold derivative, unchanged hidden branch
trace, or correctness of either AD direction. The earlier saved-data
right-sided FD diagnosis remains inconclusive and is not repeated here.

The result rules out announcing that this bounded native forward-mode check
strictly confirms the original reverse reference. It supplies no independent
ground truth for the historical V4 differences. The candidate remains closed,
all tolerances remain unchanged, and no further JVP/FP64/gradient expansion is
performed as part of this task.

## Runtime and artifacts

Free gln01 actual host `ibc13b02n04`, one V100-SXM2-32GB GPU1,
driver525.85.12, canonical JAX0.4.38/Singularity runtime, OMP4 and disabled LLVM
module compilation parallelism. At model start both GPUs were0MiB/0%, with no
compute processes, load0.01/0.09/0.24 and affinity0–111. Target Bash syntax and
both actual `--dry-run` argument expansions passed before execution.

| Stage | Python seconds | Launcher seconds | One-GPU card-hours | Actual exit |
| --- | ---: | ---: | ---: | ---: |
| New canonical runtime preflight | 24.259321 | 28.685064 | 0.007968073222 | 0 |
| Original T7 dynamic-direction JVP | 480.862057 | 487.952833 | 0.135542453594 | 0 |
| Total launcher scope | — | 516.637897 | **0.143510526816** | — |

Lowering81.110097s and backend compilation279.695955s remain below their
240s/900s limits. The XLA alarm's2m20.390459s is an internal operation, not the
whole backend compile. Internal process bound1485s, external timeout1500s with
10s kill grace. The32 synchronized JVP calls total15.432966s; repeat-only
coordinate medians are0.478675–0.479031s. This specialized state JVP timing is
not complete-update or training-throughput evidence.

Compiled JVP estimated live memory661,933,593bytes; observed process peak
active memory3,507,744,000bytes, below24GiB. The process peak includes retained
original dual runtime work; it is not attributed solely to the JVP executable.
GPU count1, model compilations1, original dual calls2, JVP calls32, paid jobs0,
candidate executions0 and training updates0.

Compact local artifacts under
`outputs/diagnostics/shared-original-state-jvp-20261002/` include final
`model.json`, `model.directional-results.pkl`, first/end original full state
gradient pickles, source/launcher snapshots, logs and actual exits, plus all
preflight artifacts. `artifact-manifest.json` records paths/sizes and collection
scope. The23,731,823byte lowered StableHLO and60,182,425byte serialized executable
remain preserved remotely, without duplicate local transfer.

Remote results directory:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-original-state-jvp-20261002`.
Overlay launcher/probe have the matching directory under `runtime/transfer/`.
Existing immutable checkout `runtime/checkouts/shared-split-width-v1` remains at
`c599236bcf57b745e9bafed678298010f4c3b58f`; original prepared/profile/context and
update0 checkpoint are reused. The only new diagnostic sources are
`scripts/hpc/probe_shared_original_state_jvp.py` and
`scripts/hpc/run_shared_original_state_jvp.sh`.
