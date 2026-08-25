# Target-Oriented Update-1000 Continuation V1

Status: frozen implementation and admission contract; paid execution requires
separate resource approval.

## Decision

Continue `target_oriented_hierarchical_coordinate_attention_v1` from update
250 through update 1000 without changing its architecture, output boundary,
loss coefficients, optimizer, clipping threshold, or validation panel.

This is justified by the formal update-250 result: the 96-window macro improves
`28.4812%`, clipping is only `3/250`, every physical hard gate passes, and
long-horizon gradients remain connected. Update 250 is also the end of linear
warmup and the first update at the peak learning rate. The result is therefore
positive pacing evidence, not a converged loss floor.

There is no weak matched control in this stage. The current model is not a
promoted baseline, and control-relative ranking cannot establish scientific
fitness. The matched reference is the exact update-250 checkpoint under the
same frozen validation panel; promotion remains absolute.

## Immutable Parent

The only admitted parent is:

- checkpoint:
  `runtime/outputs/training/target-oriented-training-formal-e363482/stage-update-0250.pkl`;
- SHA256:
  `099d48c76277cafb04daf868c492200d0271fd6a4a2c6fa29ca6e4bdbe7989a5`;
- completed updates: 250;
- executable parent commit:
  `e363482a75cbec3d6eaaa46555cf7191f3f55350`;
- validation macro: `0.6894834688713738`;
- physical hard failures: zero.

The continuation must restore both parameters and AdamW first/second moments,
and the optimizer step must equal 250. It must not initialize a new optimizer,
load `best.pkl` in place of the stage checkpoint, or reinterpret the parent as
a recovery file.

## Data And Schedule

Build the frozen deterministic schedule through absolute update 1000. Prepare
only records 251 through 1000 from the accepted v6 train/train release. The
first 250 schedule entries are the already executed prefix and are not copied
or trained again.

The continuation adds 750 effective updates. Each update retains the frozen
32 one-day samples, eight 7-day windows, and four 30-day windows from 28
distinct point-years/landpoints in one runtime year, totaling 208 Teacher-day
usages per update. This stage therefore adds 156,000 Teacher-day usages.

Preparation reads existing v6 Teacher shards and creates executable batch
artifacts. It does not regenerate Teacher data, use historical v5, or read the
sealed test. A partial prepared manifest must declare the full 1000-update
schedule identity while covering exactly the requested absolute range.

## Optimizer And Runtime

Use the frozen AdamW state and absolute learning-rate schedule. Update 251 is
immediately after warmup; update 1000 has learning rate approximately
`8.44687e-5`. The global clip threshold remains `5323655.8449` and is not
recalibrated from validation or continuation batches.

Write alternating recovery generations every 50 absolute updates. A recovery
contains the original 1-250 records plus all completed continuation records,
so `len(update_records) == completed_updates`. At completion write
`stage-update-1000.pkl` and evaluate the unchanged 96-window non-test panel.

## Ordered Admission

1. Local synthetic tests must validate partial-range manifests, exact parent
   stage loading, optimizer step 250, absolute stage bounds, and recovery/stage
   roundtrip.
2. Prepare only update 251 on a free CPU test node and admit its manifest.
3. Run one real update-251 continuation admission on free `gln01`. It must
   reload the exact parent, use the update-251 learning rate, complete the
   optimizer update, recovery write/reload, and bounded 7/30-day cells with
   zero physical hard failures.
4. Only then prepare 251-1000 on free CPU nodes, merge and admit the complete
   partial-range manifest, and submit a separately approved paid GPU job.

The one-update admission is engineering evidence only. It neither promotes nor
rejects the checkpoint.

## Update-1000 Gate

The complete frozen 96-window panel controls the decision:

- absolute 7/30-day pass opens separately authorized 365-day work;
- otherwise, zero physical hard failures and at least `5%` relative macro
  improvement from update 250 admits review of updates 1001-3000;
- otherwise stop training and attribute the remaining failure.

The `5%` pacing threshold requires the update-1000 macro to be at most
`0.6550092954278051`. Relative improvement is not promotion. Sealed-test read,
365-day rollout, long held-out points, Gate D2, and product acceptance remain
closed until the absolute short-horizon gate passes.
