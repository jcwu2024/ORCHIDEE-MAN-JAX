# Actual-model Attention FP32: one-day numerical comparison

Completed on free Explore1000 gln01 GPU0, V100 32 GB, deployed JAX 0.4.38.
Code `c11efa65d553e95877c2dc80592d1328c180ce47`; wrapper exit 0;
elapsed 1128.047 s (18.80 min). This admits the bounded one-day comparison,
not full training, long-horizon accuracy or a final scientific model.

## Fixed comparison

Both sides use the same update-200 checkpoint, all 109,464,375 parameters,
64 real daily samples and microbatch 8. Only ordinary Attention scaling,
softmax and value aggregation switch to FP32. Pooling, physical computation,
loss and optimizer precision are unchanged. No parameter updates are retained.

| Quantity | Result |
| --- | --- |
| Legacy / FP32 daily loss | 120.4785230215995 / 120.47852156035489 |
| Relative loss difference | 1.213e-8 |
| Full-gradient relative L2 difference | 7.551e-8 |
| Failed gradient coordinates | 0 |
| Daily loss terms and masked physical/feedback auxiliary | Pass; 0 failed coordinates |
| GPP / NPP maximum absolute difference, native output units | 1.680e-7 / 1.538e-7 |
| Budget, finite state/output, negative stock, mask/discrete and feedback failures | All 0; counters agree |

The auxiliary comparison covers named fluxes/stocks, canopy-stock change and
litter feedback under their applicability masks. It is not an exhaustive
coordinate-by-coordinate equivalence claim for the complete next-day state.
Existing tolerances were used without relaxation. Neither equal losses nor
equal gradients imply that the checkpoint itself fits Teacher targets well.

## Precision, memory and timing

Static daily gradient lowering changes matrix result-type counts from
FP32 7,816 / FP64 725 to FP32 8,539 / FP64 2. These are operation counts,
not FLOP shares or percentages of runtime. Physical FP64 operations remain.

Compiled estimated live memory decreases from 15,798,461,424 to
15,503,905,904 bytes (14.71 to 14.44 GiB; about 281 MiB saved).
This is a compiler estimate, not the allocator reservation shown by nvidia-smi.

The new daily graph compiles in 988.27 s. The legacy 103.77 s compilation used
an existing cache and cannot establish a precision-related compile regression.
First daily64 executions take 9.866 s versus 9.631 s, including data/transfer
and auxiliary work. This single first-execution pair is not a qualified warm
complete-update speed measurement; no whole-training gain is claimed yet.

## Evidence and next gate

Local report: `outputs/mixed-learning-attention-fp32-daily-v1.json`.
Remote evidence root:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/`.

- Candidate report/log: `mixed-learning-attention-fp32-daily-v1.{json,log}`.
- Candidate gradient: `mixed-learning-attention-fp32-daily-v1.daily.candidate-gradient.pkl`.
- Preserved daily reference: `mixed-learning-attention-legacy-daily-v2.daily.precision-reference.pkl`.

T7 and T30 comparisons have now started on free gln01 GPU0 and GPU1,
respectively, with separate 50-minute ceilings. Immutable checkout:
`runtime/checkouts/mixed-learning-checked-reuse-v1` at `c11efa6`.
T7 Bash/Python: 49620/49665; T30: 49702/49800. Both reached candidate
compilation after about 143 s bootstrap. Diagnostics use
`mixed-learning-attention-fp32-T7-v1` and `mixed-learning-attention-fp32-T30-v1`.
They reuse preserved gradients resolved through `mixed-learning-checked-profile-v3.json`;
no Teacher data is regenerated and no paid job is submitted.

After both pass, measure warm complete updates with the same recurrent
implementation on each side. The precision harness uses original recurrent
execution; do not compare it against the separate checked-split 12.08 s result.
Joint adoption should explicitly record the precision policy and validate the
intended efficient checked route before broader training. Ordinary defaults
remain unchanged until that evidence is available.
