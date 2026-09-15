# Attention FP32: complete-update measurement

Free gln01 GPU0 completed with exit 0 at immutable `c11efa6` in 1867.28 s
(31.12 min). Bash/Python 65605/65655 have exited; no test remains running.
No paid job, retained parameter update or Teacher regeneration occurred.

## Result

The fixed update-200 replay keeps all 109,464,375 parameters, daily64/micro8,
one T7 and one T30 TRAIN window, original recurrent execution, physical
precision, losses and optimizer unchanged. Attention FP32 is the only change.

| Measurement | Value |
| --- | --- |
| Preserved original warm complete-update median | 14.18122 s |
| FP32 warm complete-update median | 13.73007 s |
| FP32 three repetitions | 13.73007 / 13.68918 / 13.75838 s |
| Time reduction / throughput multiplier | 3.181% / 1.03286x |
| Saved time per fixed update | 0.45115 s |
| Observed peak live device allocation | 15.676 GiB |
| Allocator pool reservation | 26.978 GiB |

The baseline comes from another process, not an interleaved contemporaneous
pair. This is a modest measured reduction on this V100 workload, not a
hardware-independent speed guarantee. The separate checked-split result
(12.079 s) changes execution structure and must not serve as this precision
comparison's baseline. Do not add or multiply the two gains before measuring
the combined implementation. No RTX5090 measurement exists.

All daily/T7/T30 full-gradient coordinate comparisons pass again with zero
failed coordinates. Loss and physical-count comparisons pass, with all hard
counts zero. The six ordinary/instrumented replays have finite combined
gradients and finite optimizer proposals. Mixed/component losses, gradient
norms and clipping agree with the preserved reference within existing
tolerances; maximum relative scalar/norm difference is 2.409e-8. This does
not assert coordinatewise equality of optimizer proposals or a trained-model
accuracy improvement. Proposals are discarded after each replay.

Instrumented execution costs 14.1525 s (3.08% above ordinary execution).
Major measured components: daily backward 5.1643 s, T30 backward 4.1376 s,
T7 backward 0.9847 s, extra T7/T30 forward 1.3607 s, and AdamW 0.0653 s.
Static FP64 matrix counts disappearing does not imply a similar reduction
in elapsed time; substantial computation and data movement remain.

## Startup cost and evidence

T7/T30 compile calls took 205.81/210.13 s. The daily compile call took
972.58 s and emitted a slow-compilation warning despite a previously compiled
daily artifact. Thus the intended cache reuse did not eliminate this startup
cost; the cause is not established by these logs. Do not label the entire
run cached or include startup in the warm-update throughput measurement.

Local report: `outputs/mixed-learning-attention-fp32-full-v1.json`.
Remote report/log:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/mixed-learning-attention-fp32-full-v1.{json,log}`.
The report preserves references, component comparisons, compile estimates,
all timing samples and effective precision policy. Gradient assets are retained.

## Decision and next work

The bounded Attention-only numerical and speed study is complete. FP32 is an
admitted option for this tested route; ordinary training defaults have not been
changed. The result supports removing this unnecessary FP64, with a modest
V100 benefit. It does not complete a global FP64 audit or justify lowering
sensitive physical-state/budget arithmetic.

Next integrate the explicit Attention policy with the already admitted
`--checked-recurrent --separate-checked-validation` route, using one bounded
combined comparison and complete-update measurement against preserved
references. Record the effective policy in the reproducible training protocol
and checkpoint identity before broader training. Reuse completed gradient
evidence; do not reopen architecture or historical custom-AD work. Pooling is
still a separate primitive-only candidate with small absolute saving; avoid
making more such microbenchmarks the main workstream. Return to meaningful
data exposure/learning duration and RA/canopy monitoring after integration.
No new computation has been launched by this result review.
