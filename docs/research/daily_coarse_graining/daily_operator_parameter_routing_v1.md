# Daily Operator Parameter Routing v1

Status: implemented local boundary; successor neural architecture not selected.

Date: 2026-08-08.

Code authority:
`research/daily_coarse_graining/daily_operator/parameter_routing.py`.

Interface authority:
`manifests/coarse_graining/daily_neural_pft_interface_v1.json`.

## Boundary Meaning

`PFTAxisInput.pft_parameters` is the complete operator-level parameter
carrier. It is shared by the composition root so that the process head and
the retained tail can receive their own named values. It is not a generic
feature matrix that may be concatenated wholesale into a neural encoder.

The current ownership rule is:

```text
parameter affects a replaced fast process
  -> named explicit factor around the learned environmental response

parameter affects a retained source-backed process
  -> direct retained-process input, bypassing the neural encoder

parameter affects both
  -> route to both named consumers; still do not use an opaque embedding
```

The machine-derived ownership views are complete and disjoint. A parameter
can nevertheless have more than one scientific consumer.

## Current Routes

| Parameter | Neural encoder | Fast-process route | Retained route |
| --- | --- | --- | --- |
| `vcmax25` | excluded | named GPP/photosynthesis factor | named input to daily `vmax_step` |
| `maint_resp_slope_a/b/c` | excluded | named maintenance-temperature polynomial | none after the supplied fast maintenance boundary |
| `alloc_min` | excluded | none | named input to `allocation_step` |
| `residence_time` | excluded | none | named input to `gap_mortality_step` |

The maintenance polynomial retained from the source parameterization is

```text
slope(T_longterm) = c + b*T_longterm + a*T_longterm^2
```

The successor architecture must still freeze how its learned environmental
modifier combines with this slope and with `vcmax25`. This boundary does not
choose that architecture or reinterpret the closed first-candidate outputs.

The successor must also route the polynomial's temperature from the named
day-start `slowproc_stomate_previous_step_state.t2m_longterm` state. The source
maintenance formula evaluates the polynomial at long-term air temperature and
then uses it inside a compartment-temperature multiplier. A freely learned
temperature proxy, or multiplication by `slope` alone, is not sufficient for
a later parameter-gradient claim. The exact source-shaped multiplier and its
units must be frozen with the successor architecture before implementation;
see `successor_process_head_architecture_options_v1.md`.

## Runtime Repair

The first formal Transformer A/B packed all six columns in the v6 adapter but
its neural local-feature assembler omitted the complete parameter matrix. Its
GPP and maintenance outputs also lacked named external factors. Therefore the
closed first pair cannot support a parameter-conditioned claim and is not a
Gate-D2 candidate.

The retained adapter previously carried `pft_parameters` in
`RetainedTailInput` but used only construction-time STOMATE parameter arrays.
It now installs active runtime values into the full PFT vectors on every day:

- `alloc_min` reaches retained allocation;
- `residence_time` reaches retained gap mortality;
- `vcmax25` reaches the retained leaf-age-weighted `vmax_step` update;
- `maint_resp_slope` remains outside the retained tail because maintenance
  respiration is supplied at the replaced fast-process boundary.

Inactive compact PFT slots preserve their construction-time retained values,
so arbitrary inactive-slot perturbations remain isolated.

## Local Acceptance

The bounded tests require:

- complete and disjoint ownership column sets;
- fast-factor invariance to `alloc_min` and `residence_time` perturbations;
- retained-exact ownership-view invariance to `vcmax25` and maintenance-slope
  perturbations, while the separate STOMATE route intentionally forwards
  `vcmax25` to `vmax_step`;
- zero cross-route gradients;
- active compact-to-full PFT replacement and inactive-slot isolation;
- runtime `vcmax25` replacement for retained `vmax_step` while the unrelated
  retained maintenance-slope array remains unchanged;
- existing allocation and constant-mortality source-kernel tests;
- existing daily-operator forward, loss, gradient, and multistep regressions.

This is a local interface acceptance only. It authorizes neither a successor
network implementation nor GPU training, 365-day validation, Gate D2, or a
parameter-inversion claim.
