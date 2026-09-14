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

Execution: local34 relevant tests pass; deployed target-shell profile dry-run
passes. Started free gln01 GPU0 with Bash71391, immutable checkout6f26537.
Log is the sibling `mixed-learning-profile-v1.log`. Both GPUs were idle at
launch; no paid job submitted. COMPLETED, wrapper exit0; runner1048.85s,
including bootstrap, compilation, warmups, timing and optional trace export.
Source report is copied to `outputs/mixed-learning-profile-v1.json` locally.

## Measured result

All109464375 parameter elements are FP32 (437857500bytes,1363 leaves).
Peak live15.963GiB, reserved26.978GiB, host21.536GiB. There were zero retained
parameter updates and no saved checkpoint. All replay physical/finite checks
passed; maximum scalar/norm relative difference across modes/repeats was
3.08e-10. This tests timing equivalence, not model scientific accuracy.

Production-like replay:14.1026s mean,14.0924s median,14.0846..14.1309s range.
Synchronized component replay:14.5984s mean,14.5730s median; instrumentation
adds3.41% to median. This is reasonably close to the prior rotating-data
200-update warm mean15.2607s, but one fixed pair and a warm cache do not measure
all data-dependent paths or cold filesystem behavior.

| Component | Median seconds/update | Fraction of instrumented total |
| --- | ---: | ---: |
| Daily64 loss/gradient, eight micro8 calls | 5.240 | 36.0% |
| T7 checkpointed loss/gradient | 1.022 | 7.0% |
| T30 checkpointed loss/gradient | 4.295 | 29.5% |
| Extra T7/T30 validation forward | 1.405 | 9.6% |
| Recurrent physical/feedback/science checks | .936 | 6.4% |
| Daily and recurrent preparation/transfers | .646 | 4.4% |
| Gradient accumulation/norm/merge, AdamW, tree-finite checks | .542 | 3.7% |
| Daily hard checks | .035 | .2% |
| Remaining Python/allocation/report-independent overhead | .452 | 3.1% |

Component medians need not sum exactly to the median total. Daily reader had
only3 artifact loads in this fixed-data probe. In the actual rotating-data
200-update run it recorded677 loads/337MB/28.03s; that old counter excludes
recurrent preparation. Both observations argue against prioritizing daily IO.
AdamW alone is .069s/update, less than .5%; optimizer arithmetic is not the
principal time sink. Component backward timings include their forward and,
for recurrent AD, checkpoint recomputation, device execution and runtime
dispatch. They do NOT measure pure GPU arithmetic or the neural head alone.

T7 and T30 gradient lowerings each contain8924 FP32 and936 FP64 matrix-result
operations; daily has7816/725. These are static counts, including physical/loss
operations and derivative paths, not evidence of their sizes or time shares.

The optional CUPTI trace exported successfully but issued a buffer warning.
Its GPU events span only3.825s while the recorded replay spans16.883s: GPU
coverage is incomplete. Captured kernel names include SGEMM and DGEMM, but
their sums cannot establish whole-update GPU utilization, FP64 fraction, or
whether the complete run is limited by arithmetic, launch latency or bandwidth.
Do not use the2.55s captured GPU duration as a full-run utilization estimate.
Trace artifacts are retained server-side (one11.6MB gzip JSON and626.7MB
xplane file); only the gzip was copied locally for inspection.

## Decision and next work

The priority is execution efficiency, with loss/gradient calls accounting for
about72.4% of update wall time and redundant recurrent validation another16.1%.
There is no evidence that a globally FP64 neural network or daily data IO is
the main explanation. The contribution of FP64 physical kernels is not yet
quantified. Do not downgrade physical precision merely from this profile.

First optimization candidate: compose the existing differentiable recurrent
loss with compact physical/finite/coverage auxiliary counters from the same
primal rollout, removing the separate validation forward and duplicate terms.
The explicitly duplicated forward costs1.405s/update (~9.6%). All extra forward
and validation work totals2.341s (~16.1%); even deleting that entire category
would yield only~1.19x speedup. Real checks must remain, so this is an optimistic
ceiling for that category, not a promised result or several-fold acceleration.
Verify identical scalar/gradient/counter behavior plus actual-runtime memory
and throughput for this consequential graph-boundary change before adopting it.

Do not spend the next workstream rewriting readers or AdamW. Larger microbatch
remains bounded by the prior micro8/16 throughput and memory evidence; do not
assume it gives proportional speedup. A separate neural mixed-precision or
broader execution-fusion study is justified only after this scoped optimization
and must retain physical/AD/science checks. No such precision change or new
paid training is implemented/approved by this report.

Longer recovery-capable learning segments amortize fixed startup/compile costs;
they do not remove the measured per-update cost. The previous200 run spent
3151.6s in update records out of4562.5s runner time. Repeated tiny jobs are
therefore inefficient, but a full-pool schedule still requires a meaningful
wall-time budget and adequate scientific monitoring.

## Source audit (not time attribution)

The active neural matrices and parameters are FP32: shared `_linear` casts
inputs before matmul, and AdamW moments/updates are FP32. FP64 remains in
physical reconstruction, state transitions, losses, and one masked token
pooling operation before an FP32 cast. Global x64 enablement does not mean the
109M-parameter network is FP64. Lowered matrix results are recorded by the probe.

Each recurrent component currently runs a separate validation rollout, then
the checkpointed differentiable loss runs its own rollout. Validation also
evaluates recurrent science terms separately and copies state/budget arrays to
NumPy in `probe_public_rollout.physical_checks`; feedback checks and per-term
scalar conversions add host interactions. A possible follow-up is a shared
checkpointed scalar with compact validation auxiliary outputs, using
`value_and_grad(..., has_aux=True)`. It must preserve every existing physical,
mask, discrete and metric-population check, return only compact diagnostics,
and keep gradients through the same differentiable rollout. Existing ordinary
forward observations cannot simply replace that differentiable rollout.

Rank this candidate using measurements. Device-side counter reduction and
prebinding feedback plans may help; merging full-tree gradient reductions is
another candidate if its measured cost matters. Removing synchronization can
increase live memory, so correctness alone is not sufficient for accepting it.
