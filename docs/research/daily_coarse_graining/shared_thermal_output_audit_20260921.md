# Shared thermal output audit

Status: read-only source and TRAIN-metadata audit at `55b05e9`. This report
does not adopt a model change, alter a loss, or authorize paid work.

## Finding

The `+80` to `+87 K` fresh predictions are caused by a mismatch between the
shared decoder's order-one initialization and the existing direct
`signed_unit` thermal transform. They are not produced by a wrong thermal
bound, a faulty conservative update, or a `tdeep`/`cgrnd` reconstruction bug.

The complete path is:

1. Thermal labels are exact bounded tendencies,
   `q = (day_end - day_start) / B`, with `B=100 K` for `temp_sol`, `ptn`,
   `stempdiag`, and `snowtemp` (`constrained_daily_replay.py:43-49,180-229`).
2. The schema assigns thermal fields `constraint="signed_unit"`; the base
   `OutputFieldSpec` default leaves them at `value_representation="direct"`
   (`neural_architecture.py:56-68,850-917`).
3. The shared head delegates these fields to `normalized_output_to_physical`
   (`shared_process_head.py:42-59`). That function loads TRAIN `mean` and
   `scale`, but its direct signed-unit arm returns `tanh(raw)` and uses neither
   value (`output_transform.py:21-95`).
4. The named-output adapter preserves that value as `normalized_tendency` and
   attaches the declared bound (`neural_process_head.py:177-197`). The updater
   then evaluates `T_next = T_start + B * normalized_tendency`
   (`conservative_update.py:86-109`).

Consequently the current active-coordinate algebra is exactly

```text
q_old(r) = tanh(r)
T_next(r) = T_start + B tanh(r)
T_next(0) = T_start
dT_next/dr at 0 = B
```

The shared final readout uses an ordinary fan-in normal kernel with standard
deviation `1/sqrt(256)` and zero bias (`shared_transformer.py:252-257,1008-1012`;
`strong_baseline_architecture.py:729-738`). With normalized hidden activations,
order-one logits are therefore unsurprising. A logit of about `1.10` produces
an `80 K` change and a logit of about `1.35` produces an `87.4 K` change. The
saved row-819 prediction, `389.1448 K` versus `301.7850 K`, is the expected
failure mode of this algebra.

This is best classified as a fresh-initialization/output-coordinate defect.
It creates a real, unacceptable physical prediction, but it is not evidence
that a trained shared model has learned wrong soil physics. All current hard
gates can still pass because `tanh` enforces the very broad source-safe
`+/-100 K/day` bound.

## Actual TRAIN statistics

The authoritative arrays were read without modification on `test01` from:

```text
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/
first-neural-process-head-v1-statistics/process_training_statistics.npz
```

All values below are for coordinates with `count > 0`.

| field | count | TRAIN mean range | TRAIN scale range |
| --- | ---: | ---: | ---: |
| `temp_sol` | 1 | `-2.599627369e-6` | `0.01273255558` |
| `ptn` | 64 | `[-2.537411326e-6, 1.048164937e-6]` | `[1.186178330e-6, 0.01136814890]` |
| `stempdiag` | 11 | `[-2.537411326e-6, 2.908202547e-7]` | `[0.0002650684593, 0.01136814890]` |
| `snowtemp` | 3 | `[-7.966080118e-10, -4.656963918e-10]` | `[3.707670655e-5, 4.580012931e-5]` |
| `snowheat` | 3 | `[-3.383522405e-12, -3.330032739e-12]` | `[9.202867250e-7, 1.792037416e-6]` |

All 82 observed thermal centers and scales are finite; every center is strictly
inside `(-1,1)` and every scale is positive. No observed thermal coordinate
has scale `1.0` in this artifact.

The `ptn` array has shape `(1,32,14)`: 64 coordinates have observations and
384 have count zero. Those 384 columns have the generic loader fallback
`scale=1.0`, but they are the 12 non-selected PFT columns. The frozen product
selects PFT indices `[0,13]`. Because `ptn` is a full-PFT-axis output, the
shared static gather retains only those indices (`shared_transformer.py:623-648`),
and the runtime applicability mask independently marks the others inactive
(`shared_transformer.py:592-615,651-665`). Inactive outputs are restored as
zero and masked again by the physical transform. Thus the current fallback
coordinates cannot cause this admission failure.

`scale=1.0` should not be banned as a numeric value in general: a genuinely
observed field could have that empirical scale. The correct gate is based on
observation count/applicability: every eligible thermal coordinate must have
a finite interior center and a finite positive scale; count-zero fallback
coordinates must be ineligible.

The deep-state scale reported as `0.000118617833 K` is also internally
consistent rather than corrupt. It equals the smallest observed normalized
`ptn` scale above multiplied by the `100 K` physical bound. Dividing an
`80 K` initialization error by this legitimate slow-coordinate scale explains
the roughly `6.8e5` normalized error.

## Derived thermal ownership

The downstream thermal materialization is source-aligned:

- `tdeep` is not an independent prediction. `thermosoil_final_state` sets
  `deeptemp_prof=ptn` from Fortran `thermosoil_main` lines 1011-1032
  (`jax_orchidee/sechiba/thermosoil.py:3020-3047`), and the driver stores that
  array as `tdeep` (`jax_orchidee/driver/orchestration.py:2905-2928`).
- `cgrnd` is recomputed after the fast update from the updated `ptn`, moisture,
  vegetation, snow, and source-backed thermal functions
  (`mechanistic_output_reconstruction.py:375-533,859-968`). The driver writes
  the same downstream `cgrnd` array into both thermosoil and enerbil state.

The large `tdeep` and `cgrnd` errors are therefore causal consequences of the
bad `ptn` initialization, not independent decoder failures.

There is, however, inherited duplicate loss ownership. The objective directly
supervises the learned `ptn` tendency, supervises resulting `ptn` again in the
named thermal state group, and retains the exact `tdeep=ptn` mirror in the
residual `other` state group. Both identical `cgrnd` state records also remain
in `other`. The state-anchor exclusion list removes `ptn` but not `tdeep` or
the `cgrnd` mirrors (`target_oriented_state_anchor_amendment_v1.json:22-39`;
`target_oriented_training.py:486-530`). In the two saved rows:

- `ptn + tdeep` account for the reported `98.95%` dominance;
- enerbil and thermosoil `cgrnd` have identical loss contributions in each row
  (`53,094,642.39` each for row 819 and `48,704,580.32` each for row 982).

This duplication distorts objective weighting but does not generate the
physical error. It should remain unchanged during the isolated transform
comparison, then receive a separate canonical-owner audit before the final
training objective is frozen.

## Minimal correction and algebra

The most localized scientifically justified correction is a shared-head-only,
TRAIN-centered bounded coordinate for direct signed-unit thermal fields:

```text
c = clip(TRAIN_mean, -1 + eps, 1 - eps)
l = max(1 - c*c, eps_slope)
q_new(r) = tanh(atanh(c) + r * TRAIN_scale / l)
T_next(r) = T_start + B * q_new(r)
```

For the audited interior centers:

```text
q_new(0) = c
dq_new/dr at 0 = TRAIN_scale
T_next(0) = T_start + B*c
dT_next/dr at 0 = B*TRAIN_scale
```

For `ptn`, zero raw therefore gives only `[-0.000253741,+0.000104816] K`
of daily change, while the local physical response spans the observed
`[0.000118618,1.136814890] K` per unit raw. The outer `tanh` preserves the
full source-safe `(-B,B)` range; this is a reparameterization, not a narrow
physical clamp. It also preserves nonzero gradients into the complete shared
network, unlike an exactly zero final readout kernel.

Do not apply this change silently to historical heads, reuse the old saved
gradient under it, lower thermal/state loss weights, remove coordinates, or
change the `100 K` source bound. Those actions would either mix protocols or
hide the initialization fault.

## Bounded admission gate

Use the same fresh seed and the same two TRAIN rows for a paired zero-training
comparison, changing only the shared thermal transform mode.

1. Assert from the loaded arrays and decoder plan that every eligible thermal
   coordinate has `count>0`, finite `abs(mean)<1`, and finite positive scale;
   assert every count-zero coordinate is ineligible.
2. Unit-check `raw=0`, the analytic local derivative, inactive-coordinate
   zeros, finite gradients, and the full asymptotic signed-unit range.
3. Report per-field raw logits, normalized tendencies, physical endpoint
   errors, and state-anchor attribution. Require removal of the tens-of-kelvin
   initial `ptn` error without worsening any hard counter. Do not accept on
   aggregate loss alone.
4. Re-run the revised one-day native reverse/JVP and two complete updates.
   Confirm finite, nonzero shared-backbone and readout gradients and unchanged
   memory/resource bounds before any T7/T30 qualification.

The knowledge graph generation was `2026-09-21T10:50:47Z` at branch head
`55b05e9`. It recorded no parse gaps for the cited code, but marked filesystem
metadata changed; all material line ranges above were therefore read directly
from source. The generated admission JSON and remote NPZ are outside graph
coverage and were inspected directly.
