# Mixed-learning performance profile

User authorized a bounded performance diagnosis before expanding learning.
This is a free gln01 V100 probe, capped at 40 minutes including startup and
compilation. It is not another scientific training run or a paid submission.

Use the completed mixed-learning-v1 update200 checkpoint and its AdamW state,
the original first daily64 TRAIN batch (micro8), and first T7/T30 TRAIN pair.
Keep parameters and optimizer state fixed between all replays; compute and
discard proposals. No checkpoints, new Teacher data, architecture, precision,
loss, or differentiation changes. Reuse the admitted native checkpointed AD
and shared loss/updater implementation. The original training checkout remains
immutable; diagnostic code uses `runtime/checkouts/mixed-learning-profile-v1`.

Warm up, then measure three production-like complete update replays. Separately
warm up and repeat three instrumented replays with synchronization at component
boundaries. Report both total times and their ratio, scalar/norm agreement,
physical and finite checks, and peak memory. Instrumented components cover
daily read/pack, transfer, backward, checks and accumulation; recurrent case
preparation, extra forward, validation and native backward; gradient merging,
norms, finite checks and AdamW. Recurrent preparation already includes some
device operations, so it must not be called pure disk IO. Repeated fixed data
measure warm-cache throughput; compare with the completed rotating-data run.

An optional single JAX trace follows timing. Unsupported trace collection does
not discard component timings. Parameter dtype and lowered matrix-result dtype
counts supplement the source audit; static counts do not establish time or FLOP
shares. A time share attributable specifically to FP64 requires stronger kernel
evidence or a separately admitted precision comparison.

Entry: `scripts/hpc/run_mixed_learning.sh profile`, using the existing GPU
container entry. JSON: `runtime/outputs/diagnostics/mixed-learning-profile-v1.json`.
Review measured bottlenecks and potential speedup before proposing a larger
learning budget. Do not treat a profile as scientific model acceptance.
