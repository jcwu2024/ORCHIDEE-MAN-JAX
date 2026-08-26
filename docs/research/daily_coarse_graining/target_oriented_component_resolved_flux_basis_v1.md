# Target-Oriented Component-Resolved Flux Basis V1

Status: frozen before the zero-update geometry audit; local implementation and
qualification accepted. No retained optimizer update or paid training is
authorized by this contract.

## Motivation

The update-1000 checkpoint passes biomass and slow-carbon gates but still
fails GPP, NPP, maintenance respiration, growth respiration, and total
autotrophic respiration. The completed 32-window gradient audit shows that
larger batches help GPP, but do not make maintenance or total autotrophic
gradients converge. More importantly, the historical optimization basis
contains total autotrophic respiration while the absolute gate separately
requires its maintenance and growth components.

That basis permits compensation: an equal maintenance overestimate and growth
underestimate can give exact total autotrophic respiration and zero historical
flux loss. This is a loss-identifiability defect, not an Attention-backbone or
Teacher-data defect.

## Frozen Amendment

The historical basis remains unchanged and replayable:

```text
GPP
total autotrophic respiration
heterotrophic respiration
```

New work must explicitly select the component-resolved basis:

```text
GPP
maintenance respiration
growth respiration
heterotrophic respiration
```

Total autotrophic respiration, NPP, NEP, and NEE remain exact derived
identities and required absolute report keys. They are not duplicated in the
optimization macro. The three independent stock coordinates remain AGB, BGB,
and nonliving ecosystem carbon.

Within the flux macro, the four active independent fields have equal weight.
The normalized MSE and normalized signed-bias-squared terms retain equal
weight. The total flux macro and total stock macro retain their previous equal
weight. The one-day state, 7-day science, 30-day science, and necessary
process coefficients remain `1.0/1.0/1.0/0.1`.

This amendment does not change:

- the 83,755,173-parameter hierarchical coordinate Attention architecture;
- the 161 learned fields or 39,006 learned scalar coordinates;
- the canonical daily-operator binding or retained exact tail;
- the update-1000 parameters;
- Teacher v6 or prepared training artifacts.

The machine contract is
`manifests/coarse_graining/target_oriented_flux_basis_amendment_v1.json`,
SHA256
`6c81e8f56c2c81fc2ac024a71da98f3e875c4ebc429f9e3d1ff4e8e3ef6bfb45`.
It binds the parent target/training contracts, update-1000 checkpoint, and
completed variance-audit report.

## Implementation Boundary

`target_oriented_flux_basis.py` owns the amendment identity and loader.
`target_oriented_contract.py` and `target_oriented_training.py` retain the
historical aggregate basis as their default but accept the frozen explicit
component basis. Scale derivation, direct science-window evaluation,
sufficient statistics, exact microbatch accumulation, and gradient-exact
linearization all use the selected basis.

The science linearization additionally exposes flux and stock contributions
separately. Their sum is exactly the complete science-window gradient. This is
diagnostic decomposition, not a change to the total loss.

## Zero-Update Geometry Gate

The first execution uses the exact update-1000 state and the eight previously
fixed train/train updates `251, 358, 465, 572, 679, 786, 893, 1000`. Each
effective group uses its existing 32 one-day samples, eight 7-day windows, and
four 30-day windows. It evaluates five gradient sources:

1. one-day state plus necessary process supervision;
2. 7-day component-resolved flux;
3. 7-day independent stocks;
4. 30-day component-resolved flux;
5. 30-day independent stocks.

A valid audit requires all five sources to be finite and nonzero in every
group, exact coverage of all `83,755,173` parameters, finite Gram statistics,
and zero physical hard failures. It reports source-to-source cosines by model
family, cross-batch source variance, and the combined amended-flux cosine to
the three stock/state protection sources.

This gate deliberately does not choose a step size from gradient cosines. Its
only successful action is
`geometry_measured_bounded_virtual_step_required`. After reading the result,
a separate virtual-step contract must freeze the direction, step-size grid,
and allowable regression caps before exact forward evaluation. No parameter
from this audit is retained.

## Data And Execution Boundary

- Teacher regeneration: forbidden;
- historical v5 use: forbidden;
- sealed-test read: forbidden;
- formal 96-window tuning: forbidden;
- optimizer invocation: forbidden;
- checkpoint write: forbidden;
- 365-day rollout, Gate D2, and product acceptance: closed.

The runner is
`scripts/hpc/audit_target_oriented_component_resolved_geometry.py`; the
free-node launcher is
`scripts/hpc/run_target_oriented_component_resolved_geometry_audit.sh`. The
launcher accepts only `gln01` identities and contains no Slurm submission.

## Local Qualification

The focused amendment, objective, geometry-helper, historical-contract,
training, prepared-batch, and prior-audit group passes `30/30`. The complete
target-oriented unit group passes `53/54`; the sole failure is the documented
unchanged historical representative-probe wall-time assertion and is outside
this path. Ruff, Python compilation, Bash syntax, and `git diff --check` pass.
