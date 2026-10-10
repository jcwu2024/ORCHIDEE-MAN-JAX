# T30 temporal scan partial-unroll screen results

**Disposition:** close the route before a long actual-T30 GPU compile unless
new evidence materially changes the expected payoff.

The CPU-only prescreen compared temporal scan `unroll=1` and `unroll=2`. On the 30-step
deterministic `RolloutLoss` fixture, both dynamic input cases matched bitwise
across 374 output/loss/auxiliary/gradient coordinates, with zero strict-tolerance
failures (`rtol=1e-5`, `atol=1e-6`). Altering the equal-shaped dynamic input
changed 363/367 compared coordinates in both arms. This is adapter/interface
evidence, not scientific model validation.

The reused original expert5 sample0 component lowered a two-step recurrence
with unchanged native AD and `prevent_cse=True`. It used a 7542x1x384 query, 59
chunks of 128 and 3,548,928 decoder parameters. Compact IR changed as follows:

| Measure | Unroll1 | Unroll2 | Change |
|---|---:|---:|---:|
| Operations | 5,763 | 5,739 | -24 |
| `stablehlo.dot_general` | 144 | 144 | 0 |
| Loops | 6 | 4 | -2 |
| Tensor result-shape count | 6,565 | 6,431 | -134 |

The removed loops contained 28 and 25 nested operations; the four larger
decoder-map loops remained. Sequential lowering observations were 2.10s and
1.61s. They are not speed measurements. Because the two-step screen equals its
whole temporal horizon, it does not predict T30's 15 groups, graph growth,
compile cost, memory, GPU behavior or complete-update throughput. No decoder
backend compilation or model outputs were executed.

**Guard deviation:** attempts 2 and 3 stopped on import/source-overlay failures;
attempt 4 completed in 7.43s after imports. Each historical invocation received
a fresh 120s timeout, contrary to the intended single persistent window. The
process intervals sum to approximately 22s, but an exact accumulated duration
cannot be reconstructed. The historical run is recorded as guard-noncompliant.
The launcher has since been repaired to persist one origin/deadline across
retries and pass only the remaining budget. Attempt4's completed evidence
blocks accidental execution.

No GPU, paid job, training, production default change or actual-T30 numerical
comparison occurred.
