# Native T30 parameter differentiation extension

User authorized this extension alongside forward-error attribution. Reuse
job15048786's final update1998 weights and the frozen forward panel. No new
training, architecture change, Teacher regeneration or paid allocation.

## Case and unchanged derivative contract

Use the first existing T30 selection: artifact1887/window2,
point101.0-115.0/year1995/start_row9. This is a metadata-order selection,
not a choice based on forward accuracy or gradient results. Only the initial
state is Teacher supplied; canonical recurrent state and retained tail remain.

Keep the accepted T7 diagnostic scalar (four normalized cumulative flux MSEs
and terminal AGB/BGB MSEs), fixed initial state, full109,464,375-parameter tree,
native `jax.value_and_grad(checkpointed_loss)` and the identical-loss JVP.
Keep PCG64 seeds20260913/20260914 with per-coordinate exact +/-1 directions,
float64 host dot products, derivative atol1e-6/rtol.01, primal
atol1e-6/rtol1e-5, resolved projections above1e-5, and finite nonzero gradient.
Preserve existing forward hard checks and18GiB compiled-buffer ceiling.

## Execution and preserved evidence

The existing probe accepts `--horizon 30`; T7 remains the default and retains
its old durable identity. T30 writes a different probe identity and stage
directory, so it cannot silently reuse the completed T7 gradient.

Use one idle free gln01 GPU. Reverse has a90-minute limit; JVP has a60-minute
limit and starts only after the reverse report passes. Each launcher runs the
tiny deployed-runtime check first. Reuse the accepted real T30 forward/layout
evidence; do not duplicate the previously incomplete expensive CPU AD trace.
Run focused local selection checks and actual Bash4.2 command-expansion dry run
before launching. Actual GPU lowering supplies the T30 graph and memory check.
These time bounds are caps, not a prediction of how long the test will take.

Launcher: `scripts/hpc/run_trained_t30_ad_chain.sh`, with absolute `CODE_ROOT`
and `GPU_DEVICE=0`. Reports under `runtime/outputs/diagnostics/`:

- `trained-forward-ad-v1-ad-t30-reverse.json`
- `trained-forward-ad-v1-ad-t30-verify.json`
- durable gradient: `trained-forward-ad-v1-ad-t30-stages/native.pkl`
- chain log: `trained-forward-ad-v1-ad-t30-pipeline.log`

Preserve a completed reverse if verification fails or times out. Do not rerun
accepted T7 or the unchanged completed T30 reverse. Diagnose failures without
relaxing tolerances to obtain a pass.

A pass covers this checkpoint, T30 window, scalar and two parameter directions.
It does not prove all coordinates, initial-state gradients, finite differences,
all30-day cases, a future changed training loss, long-horizon accuracy or D2.
It is evidence for designing the next reproducible multi-day training stage,
not automatic authorization to launch that training.

## Execution record

Launched on free GPU0 at main checkout688e14c, PID100003. Actual Bash4.2
argument expansion passes both modes. Deployed-runtime preflight passes in
69.50s under JAX0.4.38/NumPy2.4.6. Real T30 forward completes in232.6s;
native reverse compilation has started. Keep this checkout unchanged until
the chain ends. No paid queue or parameter update.
