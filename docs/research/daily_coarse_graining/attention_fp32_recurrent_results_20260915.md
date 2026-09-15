# Attention FP32: fixed T7/T30 numerical admission

Both free V100 runs completed with exit 0 at code `c11efa6`. The unchanged
update-200 checkpoint has 109,464,375 parameters. Only Attention precision
changes; physical computation, losses, optimizer and neural pooling retain
their previous precision. No training updates are retained.

| Comparison | T7 | T30 |
| --- | --- | --- |
| Total runtime | 2241.25 s (37.35 min) | 2344.59 s (39.08 min) |
| Candidate compilation | 1821.67 s | 1901.20 s |
| Legacy loss | 121.80031965981357 | 143.0433661249959 |
| FP32 loss | 121.80031672575879 | 143.04336577836006 |
| Relative loss difference | 2.409e-8 | 2.423e-9 |
| Full-gradient relative L2 difference | 1.499e-7 | 1.328e-7 |
| Failed gradient coordinates | 0 | 0 |
| Physical counters | Equal; all zero | Equal; all zero |

Each case uses its fixed TRAIN window and preserved original gradient.
Existing coordinate tolerances pass without relaxation. The scalar, complete
gradient and named physical-counter evidence supports this bounded precision
change through 30 days. It does not certify every state coordinate, all
development windows, long-term accuracy, or a learned science improvement.
One-day64 physical/auxiliary evidence is in
`attention_fp32_daily_results_20260915.md`.

Both recurrent lowerings contain 9,860 FP32 matrix-result operations and no
FP64 matrix-result operations. This static count does not measure runtime
shares and does not mean all physical scalar/reduction arithmetic is FP32.
Compiled live-memory estimates are 2,390,263,494 and 2,393,138,672 bytes for
the respective recurrent gradient executables, not complete-training memory.

## Evidence

Local reports: `outputs/mixed-learning-attention-fp32-T7-v1.json` and
`outputs/mixed-learning-attention-fp32-T30-v1.json`.
Remote diagnostic root:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/`.
Corresponding logs and `.T7.candidate-gradient.pkl` / `.T30.candidate-gradient.pkl`
files are preserved there. Original references are
`mixed-learning-checked-profile-v1.T7.reference.pkl` and
`mixed-learning-checked-profile-v2.T30.reference.pkl`.
Bash/Python 49620/49665 and 49702/49800 have exited.

## Next execution

Full-update timing is now running on free gln01 GPU0, Bash 65605 / Python 65655,
40-minute ceiling, immutable checkout `runtime/checkouts/mixed-learning-checked-reuse-v1`
at `c11efa6`. Report/log: `mixed-learning-attention-fp32-full-v1.{json,log}`.
Target-shell `precision-profile --dry-run` confirmed the full-stage, FP32,
saved daily reference and `mixed-learning-checked-profile-v3.json` arguments.
The runner reuses compiled artifacts and original reference gradients, compares
the three components, then measures complete fixed updates with three warm
repetitions. Proposals are discarded; this is not another training experiment.

Both timing sides use original recurrent execution. The original timing was
recorded in another process, so report that limitation; do not substitute the
separate checked-split 12.08 s baseline or claim an RTX5090 speedup.
After timing, jointly validate the intended efficient checked route and record
the precision policy in the reproducible training method before adoption.
Pooling remains a separate candidate. No paid submission or Teacher regeneration.
