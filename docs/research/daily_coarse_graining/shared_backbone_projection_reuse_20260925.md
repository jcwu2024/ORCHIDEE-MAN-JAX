# Shared-source backbone projection diagnostic

Status: actual-shape scalar V100 screen completed with a numerical gate failure;
micro4 was not executed and production integration is not admitted.
This preserves the declared network function and capacity and is not wired
into the production process head. No formal training or paid submission.

Free96055 exited1 after scalar parity checks. Reference/candidate native
forward+backward helper medians27.572/14.406ms (1.914x component speedup),
lower0.769/0.679s, compile8.905/6.598s. Compiler temporary storage315.13/
257.18MB. This is not a full-update speedup. Both executed, with no runtime
exception. The loss comparison passes;7/5376 output coordinates,61954/
56,702,976 parameter-gradient coordinates and7/1,059,840 input-gradient
coordinates fail the unchanged rtol1e-5/atol1e-6 check. Maximum absolute
differences are2.265e-6/8.106e-6/2.205e-6 respectively. Do not conclude
gradient implementation failure, benign roundoff or scientific equivalence
from these counts alone. Preserve the scalar reference; next localize the
projection/reduction differences independently before any integration or
tolerance decision. Original JSON/log and reference pickle remain available.

## Mechanism and evidence boundary

The production context broadcasts393 identical source tokens across seven
expert banks before each backbone block projects keys and values. Distinct
140-token expert-axis banks are concatenated to those sources. The diagnostic
projects source K/V once per block, projects expert-axis K/V separately, then
restores the same concatenated attention bank. Expert queries, source/axis
masks, a single masked softmax per query, bounded cosine attention,
normalization, residuals and FFN are preserved. All derivatives are native
JAX derivatives; nothing is cached across parameter updates or detached.

The tenfold-budget note estimates22.25billion forward MACs avoided per
sample-day in the written backbone. That is not measured GPU work saved or
a predicted whole-training speedup. Decoder, retained physics and recurrence
remain outside this helper.

## Checks and executable

Three focused tests pass with JAX_ENABLE_X64=true. Scalar/micro4 masked
fixtures, including an entirely masked expert, compare outputs and every
parameter/input gradient at unchanged rtol1e-5/atol1e-6. Tiny compiled CPU
micro1/micro2 checks and saved-reference resume also pass. Parameters and
synthetic payloads are explicitly FP32, including when global x64 is enabled.
Ruff and formatting checks pass. Parent reviewed the implementation.

`shared_backbone_projection_reuse.py` owns the isolated helper;
`probe_shared_backbone_projection_reuse.py` compares native forward/backward
at the current768-wide/eight-block/seven-expert backbone shape, for scalar
and micro4. It records stages before lowering/compilation, synchronized warm
helper timings, compiler memory and process-wide allocator statistics. Those
allocator peaks are not isolated live-memory measurements for each variant.
Completed references are retained atomically in one sibling pickle before
candidate compilation; `--reuse-reference` reuses matching trusted artifacts.

Free-only launcher `run_shared_backbone_projection_reuse.sh` bounds the probe
to20minutes and supports GPU_INDEX, OUTPUT, OVERLAY and CLI passthrough.
Target-shell syntax and actual `--dry-run --microbatches 1 4` expansion pass.
The code is staged under
`runtime/transfer/shared-backbone-reuse-20260925`, against unchanged c599236
in `runtime/checkouts/shared-split-width-v1`. After both reference probes
completed, free launcher96055 began17:03:39CST onGPU0,20min bound; both GPUs
were empty at dispatch. Report/log stem is
runtime/outputs/diagnostics/shared-backbone-reuse-20260925. GPU1 runs the
separate expert128 screen, so host CPU is shared by the two diagnostics.

Next: diagnose this numerical discrepancy independently and require output/
full-gradient evidence plus material measured benefit before integrating it.
Any full-model integration must compare actual complete updates and retain
state/parameter gradients and physical checks. A helper pass alone does not
establish10x throughput, scientific accuracy, or final training admission.
