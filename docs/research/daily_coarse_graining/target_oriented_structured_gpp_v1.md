# Target-Oriented Structured GPP V1

Date: 2026-08-29

Status: architecture approved by the user; frozen for local implementation
and zero-update real-v6 admission only

Contract SHA256:
`ce211fa7c6f5a09bd900e02e31a864b61fb32f6b06ee96d4262402c513d620ea`

## Decision

The accepted Teacher-GPP bridge makes all nine required 7/30-day aggregate
keys pass while preserving predicted cross-day state, the accepted
distribution-aware maintenance adapter, and every other learned output. The
next candidate therefore replaces only `daily.fast_gpp`. It preserves the
coordinate-aligned Attention parent as an auxiliary representation and adds a
fresh GPP specialist with direct access to all 45 native forcing tokens.

The specialist is Transformer-style attention inside one day. It has no
cross-day neural hidden state, reconstructs no 48 half-hour forcing sequence,
and executes no subdaily state scan.

## Mechanism

The five native records contain one predecessor plus four six-hour records for
`Tair`, `PSurf`, `Qair`, `Wind_E`, `Wind_N`, `Rainf`, `Snowf`, `SWdown`, and
`LWdown`. Two fresh day-local attention blocks retain all 45 field/time tokens.
Every compact PFT then owns three coordinate queries:

1. a potential branch for light, temperature, CO2, LAI, structure, and dynamic
   photosynthetic capacity;
2. a limitation branch for VPD, wind, precipitation, canopy/snow/water, and
   thermal context;
3. a bounded correction branch over the admitted union plus the parent GPP
   hidden coordinate.

The output is

```text
active vegetation
  * train-only GPP scale
  * softplus(potential)
  * sigmoid(limitation)
  * exp(log(4) * tanh(correction))
  * max(vcmax25, 0) / 100
```

The correction is restricted to `[0.25, 4]`, while the potential branch remains
unbounded above, so the factorization does not impose an artificial cap on the
Teacher mapping. `pft_fraction` is not multiplied inside the field output; the
daily Teacher boundary has already removed the instantaneous vegetation-cover
factor and aggregation applies the declared PFT fraction separately.

Static `vcmax25` remains an explicit final factor. Dynamic day-start
`assim_param` remains a distinct input because retained `vmax` writes it for
the next day's DIFFUCO. Maintenance coefficients are excluded from the GPP
head, and `alloc_min` plus `residence_time` remain retained-exact-only.

Fortran provenance is
`fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_trans_co2:2018-2982`,
including the source inputs at lines 2030-2070 and the GPP construction at
lines 2896-2897. The dynamic-capacity writeback provenance is
`fortran_source/ORCHIDEE/src_stomate/stomate.f90:4567-4569`.

## Capacity And Training Boundary

The old 769-parameter GPP projection is removed. The specialist contains
12,879,363 parameters, giving 106,383,527 total parameters over the accepted
93,504,933-parameter distribution-aware parent. Only the specialist is
trainable in the first stage; parent weights remain frozen but still execute
on the candidate's evolving state.

Training must use existing v6 `train/train` rows sampled across the global
training population. The objective contains a direct one-day GPP anchor,
7/30-day cumulative GPP and retained respiration/NPP, and terminal live plus
nonliving carbon stocks. Direct GPP supervision prevents the specialist from
compensating for another frozen error by predicting physically wrong GPP.

The matched control shares the input registry, factorization, capacity, loss,
data, optimizer, seed, and absolute gates, but replaces attention with
token-local MLP transforms and deterministic duration-weighted pooling. It is
run only after the candidate independently passes the absolute 7/30-day gate;
it is an attribution control, not a weak promotion baseline.

## Admission Order

Local admission requires exact parameter ownership, finite eager/JIT forward,
complete loss and gradients, nonzero access to the three branches, nonzero
`vcmax25` sensitivity, zero GPP sensitivity to maintenance/allocation/mortality
parameters, inactive-PFT isolation, PFT permutation, and one complete
state/budget day. A free `gln01` real-v6 zero-update admission follows.

A separate contract is required before any optimizer update or paid job. The
candidate must then pass absolute 7/30-day GPP, respiration, NPP, biomass, and
ecosystem-carbon gates before validation, 365 days, long-held points, Gate D2,
sealed test, or product acceptance can open.
