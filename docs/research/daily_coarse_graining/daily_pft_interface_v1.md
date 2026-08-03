# Daily Neural PFT Interface v1

Status: frozen preimplementation contract.

Date: 2026-08-03.

Machine authority:
`manifests/coarse_graining/daily_neural_pft_interface_v1.json`.

Parser and adapter authority:
`research/daily_coarse_graining/daily_pft_interface.py`.

## Boundary

The future true-daily operator consumes one explicit leading PFT axis:

```text
pft_state[n_pft, n_pft_state_features]
pft_parameters[n_pft, n_parameter_channels]
pft_traits[n_pft, n_trait_channels]
pft_fraction[n_pft]
active_pft_mask[n_pft]
```

Stable PFT IDs remain adapter metadata for restart remapping and provenance.
They are not network features. Landpoint ID, Fortran PFT number, MTC embedding,
and a PFT14 embedding are also forbidden. Model weights are shared across the
PFT axis. A different legal `n_pft` can recompile a static JAX shape but cannot
require an architecture rewrite.

The state-feature registry is intentionally owned by Gate C. Freezing a state
list before the daily water/carbon/energy label inventory would turn an
unverified historical packet into the final scientific boundary.

## Parameter Ownership

Every parameter exposed to the interface has one declared owner:

1. `retained_exact`: consumed by a retained source-backed daily formula;
2. `explicit_parameterized_fast_process_factor`: remains explicit around a
   learned environmental response or flux head;
3. `learned_conditional`: learnable only when controlled Teacher perturbations
   cover the response.

The v1 registry is exhaustive for the four paper-calibrated parameters and
includes all three maintenance-temperature polynomial coefficients needed to
avoid baking a PFT-specific constant into the model:

| Parameter | Ownership |
| --- | --- |
| `vcmax25` | explicit parameterized fast-process factor |
| `maint_resp_slope_a/b/c` | explicit parameterized fast-process factor |
| `alloc_min` | retained exact daily allocation |
| `residence_time` | retained exact daily mortality |

A parameter that is constant in available Teacher samples cannot be claimed
as learnably invertible. It must retain an explicit formula or receive a
controlled perturbation dataset before promotion to `learned_conditional`.
Gate C must register any additional parameter before the constrained updater
or neural operator consumes it.

## Structural Acceptance

The interface parser fails closed on incomplete or duplicate channels,
non-float64 numerical arrays, non-boolean masks, non-finite values, identity
embeddings, and nonzero fractions on inactive slots. Named maps are packed in
manifest order so callers never own positional channel semantics.

Accepted structural tests cover:

- PFT permutation equivariance;
- inactive-PFT output and aggregate isolation;
- variable legal `n_pft` values;
- exact boolean mask semantics;
- rejection of learned-condition claims without controlled perturbations.

These tests establish extensibility of the interface only. They do not confer
scientific support on PFT2-PFT13, which remain `structural_only`.
