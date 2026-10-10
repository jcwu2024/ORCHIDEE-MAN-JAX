# Bounded execution-route screen

Date: 2026-10-02. No new optimization candidate passed this source-only screen.
No GPU model compilation, evaluation, implementation or default change ran.
This is a bounded negative result, not a claim that all possible kernels or
dataflow changes have been exhausted.

The unchanged original mix is daily32 in eight distinct micro4 evaluations,
one T7 and one T30, complete parameter/initial-state AD and one AdamW update.
FP32 neural/FP64 physical work, architecture, losses, masks and sampling remain
fixed. The measured whole-tree changing-update median is11.573968s. Separate
forward/parameter-AD measurements give2.104821s forward and9.228212s additional
AD/recompute cost (81.43%); this does not identify pure backward, neural or
retained-physics wall shares. The incomplete perturbed GPU trace cannot fill
that gap. Removing the separately measured0.331s outside evaluator calls
would imply only1.029x under fixed other costs.

The actual expert adapter was read from
`runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py`.
It already restores static eligible coordinates, reuses decoder context and
field-local projections, and keeps the original GPP route. The saved four
samples have zero wholly inactive chunks across32 route/sample records;
expert5 has7542 active rows in each sample and0/59 wholly inactive chunks.
Neither source cleanup nor an inactive-block skip provides a measured new
steady-state opportunity on this workload. Prior normalized-bank, gather,
map batching, rematerialization/capture and precision failures retain their
original dispositions; changing their names provides no new causal evidence.

One new interface feasibility check was completed: deployed JAX's cuDNN
attention `check_layout` in `jax/_src/cudnn/fused_attention_stablehlo.py:282`
accepts only BF16/F16/FP8 Q/K/V and rejects FP32. The generic
`jax.nn.dot_product_attention` API has no logits-clipping argument. The actual
`shared_transformer._attention_from_projected_segments:451-534` normalizes
heads, clips each segment's dot products to[-1,1], applies the temperature,
then shared masked softmax and separately accumulated value segments. A direct
standard-attention replacement therefore lacks a demonstrated same-function,
FP32 route. A bespoke kernel has neither native parameter/state AD admission
nor a measured wall share or speed estimate here; no custom gradient is
authorized by this screen. FLOP estimates do not establish absolute savings.

The available attention APIs provide no direct FP32 replacement that preserves
the clipped segmented function. If an implementable new kernel appears, first
measure absolute savings and native parameter/state AD on representative
original inputs, then choose public T1/T7/T30 and complete-update validation
according to the observed benefit. Missing wall-share attribution limits
savings forecasts; it does not make comprehensive profiling a prerequisite
for reasonable implementation trials. Resource changes or changes to
architecture, precision, losses, sampling or scientific derivative criteria
remain explicit user decisions. The independent original-reference JVP is
owned elsewhere and was not duplicated.

The chosen next action is instead to close the existing whole-tree slicing
acceptance gap. Its measured1.097337x three-update result did not separately
return initial-state derivatives. The saved original F4S/T7/T30 dual binaries
can be shared by both leaf/tree routes without another model compilation.
Three discarded changing updates will retain diagnostic state gradients while
only parameter gradients enter AdamW, under unchanged strict/hard/memory gates.
Small slice/tree/AdamW helpers may compile; any model compilation must fail
closed. This is an acceptance completion for an already effective route, not
a new speed forecast or a renamed failed candidate.

Graph Verify generation2026-10-02T04:30:36Z was confirmed ready. Structural
lookups were fully paginated for the bounded symbols. Coverage reports no
recorded gaps but metadata changes in transformer/rollout source and a missing
local expert adapter. Exact local source and actual remote adapter/runtime
library reads provide the material evidence; graph call edges containing
`__file__` are not treated as real call relationships.

Evidence: current NEXT_STEPS/DOCUMENT_STATUS/CODE_MAP, the execution variant
registry, original-mix bottleneck and forward/AD reports, the precision report,
and `outputs/diagnostics/shared-gather-hoist-20261001/inactive-chunks.json`.
