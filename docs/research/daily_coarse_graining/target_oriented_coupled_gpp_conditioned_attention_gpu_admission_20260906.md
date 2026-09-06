# Coupled GPP Conditioned Attention GPU Admission

Date: 2026-09-06

Status: accepted free real-v6 zero-update admission; discarded one-update
preflight design is next

Commit: `812e851258da55a05cb2faf55aa8c6da16724325`

Architecture contract SHA256:
`b9bdd9dde51a8f6f1de2c21cd6678f5a51232011397501459d614b549ca560d4`

Result:
`outputs/diagnostics/target-oriented-coupled-gpp-conditioned-attention-812e851.json`

Result SHA256:
`080f6b9dbe3ba7f74af1311dec4ee72ed1fba11dbf634ddec97bb08019233263`

## Execution

The direct free-`gln01` run completed on host `ibc13b02n04`, backend `gpu`,
device `cuda:0`, in `1753.274` seconds. It used the existing Teacher-v6
train/train artifact, accepted update-2 recovery, rejected update-3 proposal,
and accepted head/token attribution report. It made zero optimizer updates and
wrote no checkpoint.

The runner reconstructed the rejected update-3 direction in memory. All 24
artifact-local GPP gradients and eight scheduled gradients were recomputed;
the combined norm was `0.0253903907913`, the nominal displacement was
`0.00012123395861`, and the realized displacement was
`0.000121277465951`. This binds the continuity test to the same parameter path
that produced the accepted parent pathology.

## Pathological Continuity Result

The parent endpoints reproduce exactly enough for the frozen gate:

```text
lower parent GPP = 2.767322160379006
upper parent GPP = 4.869901243389088
parent jump      = 2.102579083010082
```

The conditioned candidate and fixed-temperature reference both produce:

```text
lower conditioned GPP = 5.3164683409802045
upper conditioned GPP = 5.3164683409802045
conditioned jump      = 0.0
jump fraction         = 0.0
maximum |logit|       = 4.24096155166626
```

The frozen gates require jump fraction at most `0.25` and maximum absolute
logit at most `12.000001`; both pass. The candidate trace agrees with the
production factorization, all GPP values are finite and nonnegative, and the
learnable-temperature candidate agrees with the fixed-temperature reference
at initialization.

This is strong evidence that per-head Q/K normalization removes the known
unbounded-score discontinuity. It is not evidence that `5.316468` is the
scientifically correct GPP value. The conditioning changes the initial
attention geometry rather than preserving either parent endpoint, so absolute
7/30-day science remains a required gate.

## Real-v6 One-day Result

The complete real-v6 one-day transition passes:

- complete mechanistic loss: `0.11606020402811287`;
- all `15,960,211` trainable gradient coordinates finite;
- total gradient L2 norm: `0.030141818356488694`;
- both temperature blocks have nonzero gradients;
- both potential blocks have finite nonzero Q/K/V/output gradients;
- all 181 declared output fields are finite where defined;
- next-state defined count: `5,736`, all finite;
- water budget residual: `0.0`;
- carbon budget residual: `0.0`;
- peak live device bytes reported by JAX: `11,016,873,728`.

The candidate and fixed reference match, and all non-GPP learned outputs are
bit-identical to the coupled parent.

## Runtime Note

XLA emitted one nonfatal HLO integer-parse message while using the existing
shared persistent compilation cache. The process continued, recompiled the
affected graph, passed every integrity gate, wrote the complete result, and
returned exit code zero. This is an infrastructure/cache note, not a model
failure. It does not justify a paid rerun.

## Policy And Next Gate

The run regenerated no Teacher data, used no historical v5 data, read no
fresh validation or sealed-test values, reconstructed no half-hour forcing,
and performed no subdaily state scan. It authorizes no retained update,
checkpoint, paid execution, 365-day rollout, Gate D2, or product acceptance.

The next task is to freeze and implement one discarded-update preflight at the
conditioned candidate's actual initialization. It must use existing
train/train artifacts, report the zero-update 7/30-day science state before
selecting any step, recompute the multi-objective direction with the expanded
trainable tree, select a bounded step using train-only information, and then
evaluate the fixed non-test monitor at most once. A proposal is discarded even
if it passes. Only that evidence may justify a bounded persistent-training
contract.
