# T30 temporal scan partial-unroll prescreen

Screened 2026-10-04. The diagnostic adapter changes only the static temporal
`lax.scan` unroll value from1 to2 for this probe. Production defaults, the
`jax.checkpoint(transition)`, its policy/CSE settings, complete carry and
observation history remain unchanged. The source boundary is
`rollout_loss.py::RolloutLoss._scan`, lines79-95; the public checkpointed
objective is bound by `shared_recurrent_training.py::make_shared_recurrent_objective`,
lines152-246, with `(flux + stock) / 9`. The diagnostic does not establish
physical or full-model T30 parity.

## Result

The 30-step deterministic fixture used the public `RolloutLoss` checkpointed
loss interface, full outputs and auxiliary values, scalar loss, and native
parameter/initial-state gradients. With unchanged `rtol=1e-5, atol=1e-6`,
`unroll=1` and `unroll=2` matched bitwise for both equal-shaped dynamic input cases: 374
coordinates per case, zero failures, maximum absolute error0. The changed
dynamic input changed 363 of367 compared coordinates in both arms. This
validates the adapter and fixture interface only.

The actual decoder component reused the saved expert5 sample0 recurrence from
the checkpoint-CSE screen: query shape 7542x1x384, 59 chunks of 128, and
3,548,928 decoder parameters. It lowered a two-step recurrence with native AD
and `prevent_cse=True` on both arms, leaving the checkpoint policy unchanged.
Unroll2 changed compact IR operation count from 5,763 to 5,739 (-24),
`stablehlo.dot_general` remained 144, and loop count changed from 6 to 4. The two
removed loops had small bodies (28 and 25 nested operations); the four larger
decoder-map loops remained. Tensor result-shape count changed from 6,565 to 6,431.
Both lowerings completed; their observed times 2.10s and 1.61s are sequential CPU
observations, not speed evidence.

The expert5 check covers only two temporal steps. Unroll2 therefore spans its
entire screen horizon and does not establish graph growth, behavior, compile
cost, memory, or performance for the 15 unroll groups in T30. No full decoder
backend compilation or model output execution occurred. The small structural
change and lack of runtime evidence do not support proceeding to a long T30
GPU compile; keep this route closed unless new evidence changes the expected
payoff.

## Guard and scope

The first three attempts failed at source-overlay/import boundaries; attempt 4
completed in 7.43s after imports. The earlier wrapper incorrectly restarted a
fresh 120s timeout for each invocation. The attempt launcher records therefore
do not implement the required single persistent 120s window. Process intervals
sum to approximately 22s, but exact accumulated timing is unavailable; do not
claim that the historical run complied with a single-window guard. The
launcher now records one persistent origin/deadline and passes only the
remaining budget on retries. Existing completed evidence blocks further
execution.

No GPU, paid job, training update, complete T30 objective, or throughput
measurement was part of this screen. No production code or rollout default
was changed. Full observations, derivatives, physical checks and the original
strict gates remain required for any later actual-T30 comparison.
