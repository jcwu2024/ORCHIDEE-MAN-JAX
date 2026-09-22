# Shared decoder static cost audit

2026-09-23. This is a metadata-only forward-operation audit of the admitted
shared Transformer. It changes no model, loss, state boundary, checkpoint, or
training method. It reads no Teacher data and performs no remote/GPU work.

The reproducible calculator is
`scripts/hpc/audit_shared_decoder_cost.py`; its local result is
`outputs/shared-decoder-cost-audit-20260923.json`. The source-derived layout
matches the accepted free-CPU admission report
`outputs/shared-decoder-094dfd2-cpu.json` exactly: 161 fields, 39,006 complete
coordinates, 8,550 static-PFT-eligible coordinates, and active full-PFT indices
0 and 13. The accepted report, not fixture inference alone, anchors those
counts.

## Result

The current shared network is dominated by repeated coordinate-decoder dense
work in the source-visible tensor shapes. One multiply-add is counted as one
MAC. These are forward MAC estimates, not FLOPs, compiler instructions,
reverse-mode work, or predicted seconds.

| Forward component | Static MACs | Share of counted current total |
| --- | ---: | ---: |
| Shared input projection plus eight-block backbone | 35.992G | 14.3% |
| Two-block coordinate decoder | 215.802G | 85.7% |
| Counted total | 251.794G | 100% |

The estimate excludes the canonical tokenizer before the width-768 input
projection, biases, normalization, activations, masks, gather/scatter, physical
output transforms, retained processes, loss, reverse AD and optimizer work.
Consequently it is useful for identifying repeated dense work, not for
estimating an end-to-end update budget.

The coordinate decoder estimate separates as follows:

| Decoder operation | Static MACs | Decoder share |
| --- | ---: | ---: |
| Repeated dynamic axis/local K/V projections | 100.589G | 46.6% |
| Width768 -> FFN3072 -> width768 | 85.369G | 39.6% |
| Query and attention-output projections | 21.342G | 9.9% |
| Already reused family/process/carbon K/V | 5.976G | 2.8% |
| Readout | 1.781G | 0.8% |
| Attention score and value dot products | 0.729G | 0.3% |
| Local/carbon scalar input projections | 0.017G | <0.1% |

Attention's dot products are not the main arithmetic issue. The cost is the
dense projections and FFNs surrounding attention.

## Executed shape

Static gather already removes 30,456 permanently ineligible full-PFT
coordinates. The retained 8,550 coordinates become 9,046 executed query slots
because each field is padded independently to its field-local chunk shape.
There are 496 padding slots (5.48%) and 265 `lax.map` chunks across 161 Python
field calls. Therefore eliminating padding alone cannot explain or deliver a
large speedup.

The work is concentrated by ownership. `litter_poc_doc` has 7,542 eligible
queries and 138 chunks. `hydrology_snow` has 722 eligible queries but 1,218
padded slots and 81 chunks; all 496 padding slots occur there. The remaining
five experts together have 286 eligible queries and 46 chunks. This is a shape
fact, not evidence that litter/DOC outputs are scientifically expendable.

The implementation already projects family, process and 2,386 raw-carbon bank
tokens once per decoder block rather than once per output coordinate. It does
not similarly reuse dynamic axis/local projections: each coordinate block calls
K/V projection on its axis tokens plus local token. Across the admitted layout,
those inputs reduce to only 339 distinct `(expert, slot, axis, position)` axis
variants and 1,798 canonical local-state/unmapped variants.

## Function-preserving execution candidate

The strongest next candidate keeps the full 75,578,497-parameter tree, two
width-768 decoder blocks, every eligible coordinate, every mask, and every
gradient path:

1. For each decoder block, project each distinct expert/slot/axis/position
   token once, then gather its K/V by query. Project local tokens once per
   canonical state index plus the unmapped sentinel, then gather them. Apply
   the existing runtime active mask and preserve dynamic-token order.
2. Pack independent queries by dynamic-token count and carbon-bank presence,
   gather the expert-specific family/process projections and masks, run the
   unchanged decoder blocks/readout in bounded packed chunks, and scatter results
   to the unchanged complete output shapes. This can replace 161 field boundaries
   and their 265 chunk iterations with fewer shared loops; the count depends on
   bucket/chunk sizes. It does not require materializing all 8,550 queries at
   once. Its padding-only arithmetic saving remains just 5.48%.
3. In each backbone block, project the 393 source tokens once before gathering
   them into the seven expert streams. Axis K/V, query state, masks and the
   single softmax remain expert-specific, so this introduces no cross-expert
   information path.

At source-visible shapes, packed queries plus dynamic K/V reuse reduce the
decoder estimate from 215.802G to 114.260G MACs (47.1%). Reusing the broadcast
source K/V reduces the backbone estimate from 35.992G to 13.739G. Combined,
the optimistic ceiling is 127.999G instead of 251.794G (49.2% fewer counted
MACs). XLA may already eliminate some broadcast source projections, so the
backbone saving can overlap compiler work and must be checked in lowered HLO or
runtime evidence. The decoder cross-field reuse is also unimplemented and
untimed; neither percentage is a promised wall-time reduction.

Local-token reuse must be keyed by canonical state index and current
value/defined state, not merely by output field or ownership label. Axis-token
reuse must include expert, axis slot, axis role and position. Inactive queries
must retain their existing zero-token and false-mask behavior. These details
are necessary for exact output and parameter/state-gradient preservation.

After this ceiling, decoder FFNs account for 70.6% of the remaining estimated
decoder MACs. Batching can improve execution efficiency but cannot remove those
MACs. A narrower, shallower or low-rank decoder would therefore be a real model
architecture/capacity change and needs a separate scientific decision; this
audit does not recommend one. Decoder rematerialization is likewise a memory
tool that can increase work, not a speed result.

## Timing context and limitations

The current micro4 measurement is daily32 7.772s, T7 2.610s and T30 11.011s.
The old model's raw recurrent profile was T7 1.022s and T30 4.295s. Those runs
differ in model, loss, checkpoint, execution wrapper and device layout. Static
MAC counts support profiling the shared decoder and projection boundaries; they
do not provide controlled attribution of the timing difference.

The following local checks passed:

- `conda run -n ORCJAX python scripts/hpc/audit_shared_decoder_cost.py --self-test`
- `conda run -n ORCJAX python scripts/hpc/audit_shared_decoder_cost.py`
- `conda run -n ORCJAX ruff check scripts/hpc/audit_shared_decoder_cost.py`
- `conda run -n ORCJAX ruff format --check scripts/hpc/audit_shared_decoder_cost.py`

No GPU profile, HLO inspection, reverse-memory measurement or implementation
comparison is claimed here. The next bounded gate is a same-checkpoint
function/gradient comparison and measured forward/reverse timing for these
execution-only transformations before changing neural capacity or resuming
long training.
