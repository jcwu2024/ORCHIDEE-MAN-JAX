# Exact OK_LEAK Driver-Capture Probe

Date: 2026-07-31

## Decision

The real-day capture and exact-replay plumbing passes its first scientific
gate. The successor architecture may proceed to a bounded, train-only
auxiliary capture design. This result does not authorize paid training or a
669-point recapture.

## Frozen Input

- source dataset: `daily-teacher-v4-contract-smoke-d54811b`
- Teacher commit recorded by that dataset:
  `d54811b8f6998633a681ff1ac0e156e701287714`
- landpoint/year/day: `001.0-071.0`, 1961, Day 2
- source shard SHA256:
  `2baabbf9867f0e1ff13e8ef178737d77afc15a4f0175393dbc6efa50a833d9bd`
- sealed test used: false

The probe reconstructs the real day-start Markov state, runs the current
Teacher once, intercepts the already materialized inputs to
`_paper_compiled_ok_leak_fold_jit`, persists the 13 driver series, and replays
the same compiled scan from the persisted arrays. It does not reproduce any
scientific formula in the writer.

## Result

- all 13 captured arrays equal the original scan inputs bit-exactly;
- the persisted-driver scan replay is schema-equal and bit-exact, with maximum
  error `0.0`;
- all 14 compact `ok_leak.*` endpoints, including `carbon_32l`, `DOC`, and
  `deepC_peat`, are bit-exact;
- next continuous Markov state maximum error is
  `2.842170943040401e-14`, below the `1e-12` gate;
- every next discrete state is exact;
- the uncompressed driver payload is 486,912 bytes per selected day;
- compressed capture SHA256:
  `8d0a6f3a07921988252b0d661871455af3780708dd1fa301384eb04e71045d8b`.

The historical v4 whole-`B_fast` comparison has one unrelated mismatch:
`daily_interface.t2m_min_daily` differs by `299.47472890218097`. Every other
reported target leaf is exact. This transient daily-interface field is not an
`OK_LEAK` endpoint, and the next Markov state still closes at float64 noise.
The probe therefore reports this old-asset drift but does not use it to reject
the exact carbon-driver interface.

The known nonfatal XLA algebraic-simplifier 50-run warning appeared, after
which compilation and execution completed normally.

## Reproduction

```powershell
$env:PYTHONPATH = (Get-Location).Path
conda run -n ORCJAX python -m scripts.dev.probe_ok_leak_driver_capture `
  --dataset-manifest outputs/training/daily-teacher-v4-contract-smoke-d54811b/dataset_manifest.json `
  --plan outputs/tmp/daily_teacher_v4_contract_smoke_plan.json `
  --landpoint-id 001.0-071.0 --year 1961 --day-index 2 `
  --output outputs/diagnostics/ok-leak-driver-capture-smoke-ea7128b
```

The machine report and NPZ remain under `outputs/diagnostics/` and are not Git
assets. The report's `passed` status requires exact captured inputs, exact scan
replay, every present `ok_leak.*` endpoint within `1e-12`, next continuous
state within `1e-12`, and exact discrete state.

## Next Gate

Define a deterministic, bounded set of train-only days spanning wetness,
temperature, productivity, peat, and low-carbon conditions. Capture only
those days. Before any paid fit, complete the remaining cheap gates from the
ownership audit: source carbon-budget closure, feasible perturbed drivers,
finite forward/reverse scan, and a tiny parent-no-regression fit.
