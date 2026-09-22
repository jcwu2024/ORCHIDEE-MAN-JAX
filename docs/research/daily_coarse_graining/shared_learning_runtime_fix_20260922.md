# Shared learning pilot: deployed JAX monitor compatibility

Free gln01 GPU1 wrapper83621 atbd50f51 exited1 after268.15seconds, with zero
optimizer updates. It saved the declared fresh initialization, then failed
before the first T7 monitor forward compilation. No learned-science comparison
or new training gradient ran. This was not an OOM or numerical rejection.

The new monitor's executable-cache key used `jax.typeof`, available locally
but absent in the deployed GPU JAX. Metadata-only CPU preparation did not
exercise that helper, and local tests used the newer API. The integration
checks therefore missed a deployed-version incompatibility.

The fix abstracts an identity function with public `jax.eval_shape`, preserving
the PyTree/shape/dtype/weak-type cache key without running the model or copying
device arrays. No architecture, parameter, loss, AD boundary, sampling order,
physical threshold or optimizer change. The monitor test now deletes local
`jax.typeof` before exercising the complete tiny forward monitor. Both focused
tests and Ruff pass locally. Run these tiny monitor tests using the actual GPU
Python environment before resuming; do not rerun the expensive AD admission.

Recovery source (same fresh run, update0):
`runtime/outputs/diagnostics/shared-learning-bd50f51-gpu.update-000000.checkpoint.pkl`.
Keep the failed report/log at their original prefix. Resume the unchanged
100-update protocol into a new output prefix only after the deployed test.
Free-node work remains authorized; no paid job.

## Deployed verification and recovery

At4408daf, the actual `.venvs/orcjax_gpu` Python inside the accepted container
passes both tiny monitor tests in16.85seconds with the CPU backend. They
exercise the monitor through forward execution, metric extraction and cache
reuse across parameter dtypes while `jax.typeof` is absent. This tests the
deployed-library compatibility; it does not repeat full-model GPU/AD admission.
Disabling pytest's cache produced one harmless cache_dir configuration warning.
The actual shell resume command expansion also passes.

Both GPUs were idle before same-method recovery on free gln01 GPU1. Wrapper
87042 runs immutable checkout`runtime/checkouts/shared-learning-4408daf`,
240min ceiling, using the original saved update0 parameters and AdamW.
Report/log prefix:`runtime/outputs/diagnostics/shared-learning-4408daf-gpu`.
No paid job or change to the declared100-update method. Initial startup is
not evidence of completed updates; collect its fixed physical panels next.
