# Successor Fast-Owner GPU Probe Acceptance (2026-08-09)

Status: **accepted; bounded-small-fit preparation is now active**.

## Scope

The free-`gln01` probe used the accepted v6 manifest, 101-field process
statistics, 79-value fast-owner statistics, successor contract, and 160-leaf
owner contract. Both matched arms used the same real train/train Day-2 row,
seed `20260815`, 181-field schema, 48 loss labels, 180 value-loss scales, and
parameter-tree structure. Each arm contained exactly 8,722,689 parameters.

The probe ran through
`bind_daily_operator(canonical_assembler, candidate_or_control,
canonical_retained_tail_adapter)` with the accepted fast-owner materializer.
It used native forcing records, did not reconstruct 48 half-hour records or
execute a 48-step state scan, and performed no training update, rollout, or
checkpoint write.

## Threshold Diagnostic

The first complete run at commit `50ca9f1` produced a failed diagnostic report
only because the probe had introduced a pure absolute eager/JIT threshold of
`1e-9`. Both arms otherwise passed finite forward, all-parameter gradients,
inactive-PFT isolation, canonical transition, and water/carbon budgets. Their
maximum eager/JIT absolute differences were `2.98e-8` and `3.73e-8`.

Commit `7c555d1` replaced that ad hoc rule with the existing canonical
field-aware policy `atol=1e-8, rtol=1e-10` and records the maximum normalized
tolerance ratio, responsible field, and reference magnitude. It did not
change the model graph, parameters, data, loss, or physical gate. The failed
diagnostic report is retained at SHA256
`a3cbc698cd7fa356057bd5c00b236c99239bcfeebf22088ef621349f4ecd487b`.

## Accepted Result

The accepted run used commit
`7c555d1e8fa4f3835ff4d3ae3f9d213abdd96956` on one V100 32 GB and reported
GPU backend `cuda:0`.

| Check | Attention candidate | Graph control |
| --- | ---: | ---: |
| Initial fitted loss | 611.8750885749594 | 606.8323009486777 |
| JIT replay seconds | 0.01649 | 0.00904 |
| Maximum tolerance ratio | 0.00021935 | 0.00007432 |
| Gradient L2 norm | 3106.962468383593 | 3060.6957555122444 |
| Finite/nonzero gradient leaves | 155/155 | 155/155 |
| Water budget residual | `2.22e-16` | `0.0` |
| Carbon budget residual | `-1.02e-14` | `6.48e-15` |

For each arm, all 181 fields and 39,371 decoded scalars were finite and obeyed
their declared constraints. All 8,722,689 gradient elements were finite. All
78 PFT-axis fields remained exactly zero in the disabled slot. The canonical
transition returned finite state, 181 named outputs, and budget residuals
within the fixed `2e-12` gate.

The accepted report is
`runtime/outputs/training/successor-fast-owner-gpu-probe-7c555d1/probe_report.json`
at SHA256
`5bb1f885367b3fc5183216c476a6bbe8d37122cb698d990bd5ca346006c0aaa7`.

## Gate Decision

The real-v6 GPU probe is accepted. This is execution and differentiability
evidence, not accuracy or architecture-ranking evidence. The next gate is the
frozen bounded small fit: one matched seed, at most 500 updates per arm,
no checkpoint, and the predeclared one-day plus long-horizon loss-reduction
gates. Its runner and resource envelope must be validated before any paid GPU
submission. The 7/30-day screen, 365-day validation, long-held points, Gate
D2, and product acceptance remain closed.
