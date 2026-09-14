# Mixed physical and recurrent update admission

Declared before execution on 2026-09-14. This is the next bounded engineering
gate, not a claim of a useful trained model. Keep the admitted Attention
architecture and fresh seed 20260911; optimize all 109,464,375 parameters.

Each of two updates uses 64 one-day TRAIN examples (microbatch 8) and two
independent seven-day TRAIN windows (one trajectory at a time). Pair the first
four epoch-0 fixed32 batches of the expanded 346,539-day pool. Pair the first
four T7 entries in the accepted recurrent TRAIN selection in stored order.
No window is selected by its model outcome. Fixed monitor and development dates
remain excluded. Overlap between ordinary one-day TRAIN and recurrent TRAIN is
allowed. No Teacher data is regenerated.

The scalar is the mean physical-v2 one-day loss plus the mean recurrent-v2
window loss, with outer coefficients 1 and 1. Daily and window means have
separate denominators (64 and 2); their gradients are summed before one global
clip and one AdamW update. One-day loss remains base + 0.1 canopy pseudo-Huber
+ 0.1 physical litter-fraction MSE. Recurrent loss uses the previously admitted
four daily/cumulative fluxes and AGB/BGB/canopy endpoint/trajectory terms with
unchanged frozen TRAIN scales. The model carries its own state across seven
days; native checkpointed AD differentiates that feedback. There are no daily
Teacher resets or handwritten observation cotangents.

The machine specification is
`manifests/coarse_graining/mixed_physical_recurrent_admission_v1.json`.
Its initial equal outer weights are a declared trial, not a claim of optimal
loss balancing. Record both component gradient norms and their cosine so that
later balancing decisions have evidence.

Before reverse compilation, evaluate all four fresh T7 windows and the 64-day
excluded TRAIN monitor. Then check two actual full updates and repeat those
panels after each update. Use the shared physical/state/budget checks and report
GPP, NPP, autotrophic/heterotrophic respiration in gC/m²/day and AGB/BGB/canopy
changes in gC/m². Training-window changes are in-sample; the one-day monitor is
excluded TRAIN, not independent geographic/temporal validation. Fresh weights
need valid physics and meaningful gradients, not trained-model accuracy.

Technical acceptance requires two finite updates, finite nonzero gradients
from both objectives and their sum, and zero physical/validity failures.
Loss direction and physical accuracy are reported separately: two updates are
too few for a learning or generalization claim. Preserve the first mixed
gradient before subsequent work; write no reusable model checkpoint.

The local tests compare accumulated gradients against native differentiation
of the complete scalar and compare one AdamW step. Reuse unchanged component
AD evidence; no new historical streamed-reverse work. On the deployed runtime,
check actual shapes, fresh forward, optimizer execution, measured peak memory,
and cold versus warm timings. Use a free gln01 GPU, 90-minute hard limit,
0.85 allocator fraction and 24 GiB per-executable compiled-memory envelope.
Microbatch 8 is chosen because 16 saved only about 0.55 s per one-day update
while adding about 10.32 GiB peak memory. Paid submission is not authorized by
this protocol. After this gate, admit actual T30 mixed updates and specify a
larger expanded-pool learning run.

## Execution admission

Implementation commit9e0b7bd:28 local mixed/recurrent/public-rollout tests pass;
independent Sol review found no remaining material math or panel issue.
The real-data CPU check on test02 passes in338.4s, including one-day and T7
native gradients, their sum, full AdamW state/update and science reporting
shapes. Target-shell actual argument expansion passes for both launcher modes.

Free gln01 GPU0 launcher Bash97713/Python99324 is running from the immutable checkout
`runtime/checkouts/mixed-physical-recurrent-v1`. It executes the small deployed
runtime check, then a90-minute actual forward/two-update gate. Log and report
are `runtime/outputs/diagnostics/mixed-physical-recurrent-v1-gpu.{log,json}`.
Deployed synthetic GPU runtime check passes in56.7s (JAX0.4.38, NumPy2.4.6);
the actual-model process has entered bootstrap. Actual update/science results
are pending; no paid submission. Inspect same-draw mixed descent, both gradient
contributions, excluded TRAIN monitor and physical T7 panels before T30/scale-up.
