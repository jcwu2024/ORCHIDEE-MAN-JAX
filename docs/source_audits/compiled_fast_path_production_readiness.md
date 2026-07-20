# Compiled Fast-Path Production Readiness

## Accepted candidate

The accepted production candidate enables `compiled_sechiba_day` and keeps
the strict transition available as the diagnostic fallback. It does not
enable the experimental NumPy daily accumulator.

Evidence:

- `outputs/performance/compiled_fast_path_acceptance/final_3point_annual_compiled.json`
- `outputs/performance/compiled_fast_path_acceptance/reuse_3points_current.json`

The fixed acceptance matrix uses paper landpoints `001.0-071.0`,
`069.0-119.0`, and `319.0-057.0`. For every point it compares strict and
compiled execution over:

- a three-day cold-start/cross-day window;
- all 365 days of 1961;
- all 12 annual history fields and four derived paper modelout values;
- the complete year-end day-state packet;
- direct 1962 Day1-Day2 continuation;
- independent strict and compiled three-file NetCDF restart roundtrips; and
- split-restart 1962 Day1-Day2 continuation.

The numerical fast-path gate is `atol=1e-8`, `rtol=1e-10`. Integer, logical,
mask, schema, and other discrete leaves are exact. The report records maximum
absolute, relative, and ULP differences. Large relative or ULP counts occur
only for near-zero state values and pass the fixed combined tolerance; annual
history/modelout maximum absolute differences are `1.35e-12`, `2.11e-11`,
and `2.60e-11` for the three points.

## Performance

Measured 365-day A/B timings in the same acceptance process:

| Landpoint | Strict | Compiled | Speedup | Compiled seconds/day |
| --- | ---: | ---: | ---: | ---: |
| `001.0-071.0` | 237.84 s | 79.33 s | 3.00x | 0.217 |
| `069.0-119.0` | 249.40 s | 94.16 s | 2.65x | 0.258 |
| `319.0-057.0` | 268.63 s | 91.37 s | 2.94x | 0.250 |

One structural scan callable is used. The first point creates the fixed
lifecycle executable variants; later points add no executable, and repeated
runs add none. Landpoint-specific soil overlap arrays are Driver aggregation
metadata and are not inputs to the compiled SECHIBA transition.

## Rejected companion optimization

The NumPy daily accumulator reduced compiled annual time to about 66-80
seconds in the same matrix. It remains disabled because `319.0-057.0` reached
an annual DOC state difference of `1.345e-8`, above the fixed `1e-8` fast-path
absolute gate. Annual modelout and restart roundtrip still passed, but the
production threshold was not relaxed.

Evidence:

- `outputs/performance/compiled_fast_path_acceptance/final_3point_annual_numpy.json`

## Production default

The production default is now enabled with the accepted policy:

1. The paper production/669-point runner selects
   `compiled_sechiba_day=on` by default.
2. Keep an explicit `off`/strict-debug option and keep low-level orchestration
   API defaults strict so source-level diagnostics remain reproducible.
3. Keep `use_numpy_accumulator=False`.
4. Add the three-point acceptance asset and its audit test to the production
   release gate. Any future fast-path change must regenerate this evidence.

The default switch must not alter scientific owners, tolerance policy,
restart serialization, or modelout aggregation.
