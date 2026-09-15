# Combined Attention FP32 and checked recurrent execution

User approves combining the two demonstrated optimizations and returning to
training afterwards. This is the final currently justified performance gate;
small pooling/norm experiments and exhaustive cache investigations do not
block broader learning once this gate passes.

## Declared implementation

`manifests/coarse_graining/mixed_learning_execution_fp32_checked_v1.json`
is an explicit executable policy. `run_mixed_learning.py --execution-policy`
loads it before bootstrap: Attention scores/softmax/value aggregation FP32;
native checkpointed recurrent AD with observations retained for a separate
small physical-validation program; source pooling and physical arithmetic
unchanged. The existing model, loss, data, optimizer and masks are unchanged.
Legacy defaults remain available. No custom reverse pass is introduced.

The complete policy is stored in the report and checkpoint method, not merely
in a command log. Resume must match the policy; old checkpoints cannot silently
become new-policy continuations. Fixed-checkpoint diagnostics intentionally
load the original source method and record their distinct effective policy.
The protocol file is ready for training use but is not itself a pass certificate.

## Bounded comparison before adoption

Use the same original update-200 checkpoint, daily64/micro8 and first fixed
TRAIN T7/T30 pair. Reuse original full gradients and baseline timings from
`mixed-learning-checked-profile-v3.json`. Compare both combined recurrent
gradients, scalar losses and every named physical count to these preserved
references using existing tolerances. Preserve newly computed candidate
gradients before the next compilation, even if a numerical gate fails.
Daily FP32 is unchanged from its completed full-gradient/auxiliary admission.

Then run three warm ordinary and three instrumented complete-update replays,
including AdamW proposals, which are discarded. Require finite updates,
zero physical failures and mixed scalar/norm parity. Report complete time
against original14.181 s and separately against checked-only12.079 s; both
references are cross-process. Do not add standalone savings or require a
second arbitrary incremental speed improvement. If the combination materially
regresses the admitted checked-only route, use the faster admitted policy and
report the result before larger training. No scientific accuracy claim follows.

Free gln01 GPU0, one GPU, 90-minute ceiling to accommodate first compilation of
both combined recurrent graphs and daily cache loading/recompilation. This
is a bounded integration benchmark, not a paid training run. Reuse the existing
checkout after confirming it is idle and clean, then keep it immutable.
Run the small deployed runtime check and target-shell `--dry-run` before launch.
Output: `runtime/outputs/diagnostics/mixed-learning-fp32-checked-v1.{json,log}`.

## Return to training

After numerical admission and useful throughput, close this performance phase.
Use the consolidated execution policy for the next declared training method.
Prioritize broader daily coverage and sufficient full-network daily+T7/T30
updates, monitoring GPP/NPP/RA/RH, AGB/BGB, canopy and nonliving carbon.
The ready pool has346,539 daily rows; a single batch64 pass needs about5,415
updates. Extend rotating recurrent metadata to that budget without regenerating
Teacher data or making one file per update. Use combined measured time for
resource estimates and explicitly specify initialization, sampling, recovery,
milestones and checkpoint selection. The paper route remains reproducible
fresh initialization/full-network optimization. An existing checkpoint can
serve diagnostics, not an unexplained formal pretraining chain.

Paid submission remains a separate concrete resource/cost approval. No paid
work or new architecture is authorized or needed by this integration step.

## Implementation and live admission

Implementation commits85147e3/364cdef; local15 runner/profile tests pass,
then12 runner tests pass after independent Sol review. Ruff/diff checks pass.
Review follow-up requires the saved reference for combined admission and passes
the resolved policy into bootstrap. Diagnostic source-checkpoint method is
recorded separately from effective execution policy. No new reference gradient
or file-hashing preflight is introduced.

Deployed JAX0.4.38 runtime check passes48.50s on GPU0; target-shell dry-run
confirms `checked-profile`, legacy reference report, execution-policy and90min
timeout arguments. Server checkout is fast-forwarded to364cdef and immutable.
Combined run launched on free gln01 GPU0: Bash91078 / Python91133. Report/log
use the output stem above. No paid submission. Combined numerical/speed result
is pending; the small synthetic runtime check is not real-model acceptance.
