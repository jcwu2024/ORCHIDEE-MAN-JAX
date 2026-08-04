# Gate C2 Constrained Daily Replay v1

Status: accepted.

Date: 2026-08-04.

## Boundary

The three frozen missing-label families are captured as daily reductions
inside the compiled Teacher owners:

- `water_transfer_daily_v1`;
- `ok_leak_transfer_daily_v1`;
- `energy_flux_daily_v1`.

Capture defaults off, does not alter ordinary state or modelout, and retains
no 48-step axis. Endpoint labels remain inventory/tendency targets; they are
not relabelled as process fluxes.

The non-neural updater uses bounded outgoing fractions and nonnegative incoming
amounts for water and independent carbon inventories. ORCHIDEE's source-defined
signed `qsintveg` carry is represented losslessly as nonnegative canopy storage
minus nonnegative evaporation debt. Thermal endpoints use declared bounded
daily tendencies. Non-inventory fast-boundary fields remain explicit endpoint
reconstruction targets; they are not falsely assigned to a conservation
budget. No post-hoc stock clipping is used.

## Lifecycle Matrix

The executable matrix covers:

1. 1961 Day 2 cold-start continuation;
2. 1961 Day 3 ordinary later-day execution;
3. 1962 Day 1 from a rebased restart-year boundary.

All cases feed the reconstructed fast boundary through the existing retained
season/STOMATE tail. The accepted three-file restart evidence is composed with
the restart-year replay case.

## Results

| Check | Result |
| --- | ---: |
| Frozen missing labels | 29/29 |
| Captured daily arrays | 72 |
| Day-end state leaves | 349/349 |
| Modelout fields | 26/26 |
| Largest state error | `2.7755575615628914e-17` |
| Largest water residual | `4.698463840213662e-13 kg m-2 day-1` |
| Largest carbon residual | `8.049482858041301e-9 gC m-2 day-1` |
| Largest flux-side energy residual | `6.984919309616089e-9 J m-2` |
| Post-hoc clipping | none |
| Candidate 48-step forcing/state scan | none |

The carbon verdict uses the source `min_stomate=1e-8` absolute scale. Thermal
flux identity closes independently; converting endpoint temperatures to an
energy tendency remains intentionally undeclared because the candidate
boundary does not invent an effective heat capacity.

## Reproduction

```bash
python scripts/dev/verify_daily_flux_capture.py
python scripts/dev/verify_constrained_daily_replay.py
```

Generated comparison:
`outputs/research/daily_coarse_graining/gate_c2_constrained_replay/comparison.json`.

Local comparison SHA256:
`a5e2ccd15ca02272d0fd18e2716b8def1962491dcd5fb46c3fb78d4c98b370eb`.

Gate C is complete. The next dependency gate is D1 canonical Teacher
physical-parameter gradient validation; this acceptance does not itself
authorize paid neural training or 669-point regeneration.
