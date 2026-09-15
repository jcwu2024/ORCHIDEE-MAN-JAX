# Reporting gradient norm FP32 comparison

The reporting-only norm can reproduce both preserved real gradient norms in
FP32 to relative error below5e-8. Its isolated V100 saving is tiny:0.06--0.13ms
per norm. It is not the main performance target and does not explain the cost
of a whole parameter update.

Free gln01 GPU0, V100-SXM2-32GB, JAX0.4.38, code `d683b09`; normal completion
in19.65s under a10min ceiling. Both input trees contain109,464,375 FP32 elements.
These are the saved actual T7/T30 update200 reference gradients, not random
arrays. No gradients were recomputed, no Teacher data read or updates retained.
The deployed synthetic runtime check also passed (22.12s) before this work.

| Input | FP64 norm | FP32 norm | Relative error | Old/new median time |
| --- | ---: | ---: | ---: | ---: |
| T7 | 2634.8539030 | 2634.8540039 | 3.83e-8 | 4.590 / 4.525ms |
| T30 | 1496.9800480 | 1496.9799805 | 4.51e-8 | 4.578 / 4.447ms |

Measurement uses the same expression as `run_mixed_learning.py`, identical
trees, separate compilation,0.5s warmup and51 synchronized alternating pairs.
Acceptance rtol1e-5/atol1e-6 passes. Raw result: local
`outputs/reporting-norm-fp32-v1.json`; remote
`runtime/outputs/diagnostics/reporting-norm-fp32-v1.json`.
Entry: `scripts.hpc.probe_reporting_norm_fp32 --references <T7-reference>
<T30-reference>`, through the existing free GPU wrapper.

This does not modify or benchmark the optimizer's independent FP32 clipping
norm. The runner also uses diagnostic component norms for finite/nonzero
checks; an optional conversion must retain those checks. Extreme overflow or
underflow equivalence is not established by these two trained trees.
Retain Attention as the next actual-model precision intervention; avoid mixing
this insignificant timing change into its attribution experiment.
