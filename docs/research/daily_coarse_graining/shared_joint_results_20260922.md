# Shared joint numerical admission results

Free gln01 GPU1 wrapper7359, immutablec2498e4, COMPLETED0. Main13939.51s
(232.33min); separate runtime check36.10s. Fresh seed20260921, all75,578,497
parameters, two full daily/T7/T30 AdamW updates. No paid computation.

Native reverse and forward-mode projections agree on the declared first
T7/T30 windows: relative differences8.2462e-8 and5.4644e-9. All component
gradients are finite; the combined gradient reaches both neural groups.
All initial, updated and final replay physical/science hard gates pass.
Optimizer checkpoint roundtrip passes. This admits the bounded native joint
update, not other losses/architectures or long-term predictive accuracy.

Memory: daily reverse6.672GiB, T7 reverse3.283GiB, T30 reverse3.285GiB
compiled live estimates. Measured total allocator peak8.108GiB; pool25.391GiB.
Native daily checkpointing permits the recurrent graphs to use less live
memory than the standalone daily objective. No decoder capacity reduction.

| Component | Initial loss | After two updates | Change |
| --- | ---: | ---: | ---: |
| Artifact819 daily | 0.345516 | 0.350584 | +1.47% |
| Artifact819 T7 | 0.124708 | 0.056925 | -54.35% |
| Artifact819 T30 | 0.286004 | 0.158400 | -44.62% |
| Artifact982 daily | 0.252777 | 0.395687 | +56.54% |
| Artifact982 T7 | 0.040508 | 0.071496 | +76.50% |
| Artifact982 T30 | 0.304949 | 0.108557 | -64.40% |

These are weighted loss contributions, not percentages of physical GPP or
biomass error. Both T30 flux/stock groups improve. Artifact982 T7 flux/stock
groups regress. Daily flux/stock groups improve in both rows, but state-change
anchors rise0.4351->1.3985 and0.2197->1.6881, making daily loss worse.
On update2's same input, combined loss increases0.541796->0.575740 (+6.27%);
this is not solely a comparison between different batches. AdamW history and
nonlinear finite steps need learning-curve evidence; two updates neither
prove stable learning nor justify changing the architecture or loss weights.

Cold/first compilation dominates: forward setup finishes1391s; daily reverse
takes2426s, T7 reverse4330s, T30 reverse4109s; first JVP stages take roughly
13min each. Update2 reuses all three reverse executables and completes the
whole instrumented segment in about42s (including repeated forwards, gradient
statistics and multi-gigabyte diagnostic serialization). This is not the
throughput of the planned32-day batch. Do not multiply232min by training steps.

Evidence: `outputs/shared-joint-c2498e4-gpu.json` and `.log`; checkpoint and
component gradients remain on server under the same diagnostics prefix.

Next: fixed fresh100-update learning protocol with32 daily rows and one
T7/oneT30 window per update, existing TRAIN batches, fixed fit monitor and
all48 existing held-out development windows. Keep the architecture, native
AD, scales, optimizer and loss fixed. Do not rerun this numerical admission
or initialize the learning run from its incidental two-update checkpoint.
See `shared_learning_protocol_20260922.md`.
