# Source-family pooling FP32: bounded primitive result

The explicit neural-family pooling FP32 option passes the synthetic source-shaped
V100 comparison. Forward+input-gradient time falls from0.2320 to0.1944ms (16.22%),
while forward alone falls from0.09545 to0.09404ms (1.48%, a very small difference).
The absolute saving is small and is not a whole-model speedup.

Free gln01 GPU1, Tesla V100-SXM2-32GB, JAX0.4.38, immutable `c11efa6`;
normal completion in21.94s under a10min ceiling. GPU0 concurrently prepared the
separate daily reference, so CPU/shared-host effects remain possible. This is a
warmed alternating paired diagnostic, not an allocated-node performance claim.

`scripts/hpc/probe_source_pooling_fp32.py` exercises the actual `_source_family_bank`
with seven expert banks,20 families,393 FP32 input tokens of width384. Values are
synthetic, seed20260915; family19 is empty, several keys invalid, and seven
expert/family connections disabled. The source mask and cotangent weights are
runtime arguments to prevent constant-folding a fixed linear backward. Attention
precision stays legacy. Timing uses0.5s warmup and51 synchronized alternating pairs,
with compilation outside timing.

Output relative L2 error6.78e-8, max absolute1.19e-7. Full input-gradient comparison
passes with zero failed coordinates and relative L2 error4.55e-8. Masks agree
exactly; empty/disallowed families are invalid, invalid outputs are zero, and
masked-token gradients are exactly zero. Finite output/gradient and scalar gates
pass at existing primitive tolerances. The local dynamic-mask/cotangent CPU
functional probe also passes.

Evidence: local `outputs/source-pooling-fp32-v1.json`, remote
`runtime/outputs/diagnostics/source-pooling-fp32-v1.json`. No Teacher read,
new gradient algorithm, parameter updates, paid task or full-model admission.
The option is implemented independently as
`NeuralPrecisionPolicy(source_family_pooling_fp32=True)`. Leave it off for the
first actual-model Attention comparison to preserve attribution.
