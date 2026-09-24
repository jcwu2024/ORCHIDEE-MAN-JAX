# Split decoder fused-attention helper screen

Status: local and free V100 fragment completed, 2026-09-25. The production
shared Transformer, capacity, output masks, losses and training runner are
unchanged. No full-model compilation, optimizer update or Teacher data was used.

## GPU result and decision

Free GPU1 wrapper52409 exited0 on deployed JAX0.4.38. Strict FP32 output and
all697,728 parameter/payload-gradient coordinates pass at rtol1e-5/atol1e-6.
With20 warm timed replays, production/fused medians are0.70006/0.64156ms:
8.36% less time for this helper, while compiler temporary memory rises from
3,623,768 to6,282,592bytes (73.4% more). Reference/fused compilation observations
19.30/1.62s were sequential in one process and are not a controlled cold-compile
speed comparison. Maximum output/gradient differences5.22e-7/6.98e-10.

This small component gain with substantially higher temporary memory does not
justify a whole-model compile. Keep the production segmented helper. Next
screen exact-shape scalar-field grouping, which targets repeated tiny decoder
calls and graph duplication while preserving each field's tokens and masks.
Neither fragment establishes full-model speed, memory or scientific fit.

Artifacts: remote runtime/outputs/diagnostics/
`shared-split-fused-attention-20260925.{json,log}`, local JSON under outputs/.
Actual launcher was runtime/transfer/launch_shared_fused_attention.sh: canonical
GPU Python inside foundationpose with driver bindings, GPU1, five-minute cap.
Both candidate Python files were transferred under runtime/transfer and loaded
against unchanged c599236. Target-shell syntax and actual --dry-run passed.
No paid job. The original local screen follows.

The split profile records 62,993 GPU events in one daily micro4 reverse, but
does not uniquely attribute most device time. This screen tests one narrower
hypothesis: replace the decoder's three non-carbon segmented score contractions
and three value contractions with one each after concatenating normalized
family, dynamic and process K/V. The diagnostic helper uses production
projection, head normalization, softmax-mask and output-linear operations;
shared banks are broadcast only within a coordinate chunk. GPP's optional
carbon bank is checked numerically but excluded from the cost result.

`shared_decoder_attention_fused.py` is not wired into the model. Four local
cases compare the unchanged production helper with the candidate, including
preprojected and direct dynamic K/V, carbon present/absent, inactive rows,
complete attention-parameter and input-payload native gradients. All four
pass the existing execution parity tolerances (rtol 1e-5, atol 1e-6);
inactive rows remain zero.

The width384, eight-head, 64-row, 20-family/3-dynamic/1-process-token CPU
screen (`outputs/shared-decoder-attention-fused-local.json`) includes the
whole helper's projections and native reverse:

| Variant | Median warm ms | Compiler temporary MiB | Primal JAXPR equations |
| --- | ---: | ---: | ---: |
| Production segmented | 6.929 | 1.40 | 152 |
| Fused contraction | 8.991 | 6.83 | 140 |

All 24,577 loss/output and 697,728 parameter/payload-gradient coordinates
pass comparison at rtol 1e-5/atol 1e-6; maximum absolute differences are
5.96e-7 and 9.31e-10. The fused variant is 29.8% slower in this three-replay
CPU sample and uses 4.88 times the compiler temporary bytes. The earlier
same-shape run was inadvertently FP64 because local JAX has x64 enabled;
this corrected FP32 run is the performance reference. CPU timing noise does
not support a precise speed penalty. The smaller primal graph is insufficient
grounds for a whole-model compile.

These measurements use local CPU JAX 0.10.1, not deployed V100/JAX 0.4.38;
the component has no tokenizer, 161-field layout, recurrent state, GPP's
2,386-token carbon bank or physical updater. It cannot establish GPU speed
or full-model memory. The measured memory regression, together with the
project's co-equal memory target, provides no justification to advance it to
that expensive stage without a bounded actual-GPU fragment. The accepted
split architecture and existing profile remain the reference.

The probe exits nonzero on any output or complete
parameter/payload-gradient failure at rtol 1e-5/atol 1e-6. It records
compiler temporary and warm component timing separately for the unchanged
reference and fused helper. Future reproduction must use the accepted container
launcher rather than invoking the container-dependent GPU Python directly.
