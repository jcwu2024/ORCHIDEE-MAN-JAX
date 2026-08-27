# Target-Oriented Source-Shaped Maintenance v1

Date: 2026-08-27

Status: frozen before local implementation

Machine contract:
`manifests/coarse_graining/target_oriented_source_shaped_maintenance_v1.json`

Contract SHA256: `3c4d0c085c767f14b001837ccefb76aca291a56b8fd9d3a5e33279602af57fc4`

## Decision

Preserve the update-1000 coordinate-aligned Attention body and every
non-maintenance output head. Remove only the existing 769-parameter direct
maintenance final projection. Replace it with two small source-shaped
correction branches for aboveground and belowground plant parts.

This is a new output materialization mechanism, not an Attention-versus-MLP
comparison and not an assertion that the parent is a satisfactory baseline.
The unchanged parent checkpoint is retained only as the matched diagnostic
reference required to attribute a later change. Promotion still requires a
substantial maintenance improvement and the absolute target-oriented science
gates.

## Current Defect

The current learned schema has two compact active PFT slots: bare soil and
PFT14. Its primary science category contains two GPP values and 24
part-resolved maintenance values. The current maintenance head applies one
shared `768 -> 1` linear projection to all 24 coordinate-specific hidden
states, then a generic nonnegative transform. Only afterward does the existing
parameter route apply the named maintenance-temperature polynomial.

The accepted head-conditioning audit proves that those 769 parameters have a
common maintenance descent direction. It also proves that the tested finite
direction cannot satisfy every 7/30-day protection. This motivates a
source-shaped output without rejecting the shared Attention body.

## Source Formula

Fortran `stomate_resp.f90::maint_respiration:234-375` defines part-resolved
maintenance as carbon biomass times a PFT/part coefficient and a nonnegative
temperature multiplier. Leaves receive the additional exact LAI correction.
The accepted JAX source port is
`jax_orchidee/stomate/carbon_kernels.py::_maintenance_respiration_core:2470-2529`.

The candidate materializes

```text
base[pft, part]
  = biomass_carbon[pft, part]
  * coeff_maint_zero[pft, part]
  * exact_leaf_factor[pft, part]

temperature_baseline[pft, part]
  = max(1 + slope(t2m_longterm) * (T_air_or_root - 273.15), 0)

temperature_effective[pft, part]
  = max(temperature_baseline + delta_air_or_root[pft], 0)

maintenance[pft, part]
  = base[pft, part] * temperature_effective[pft, part]
```

`delta_air` and `delta_root` are bounded to `[-1, 1]`. They are the only new
learned maintenance quantities. The reserve compartment uses the exact source
rule: air temperature for trees and root temperature otherwise. Bare soil,
inactive PFTs, biomass below `1e-8`, and leaf LAI below `1e-8` are exact zero.

The daily air temperature comes from the accepted native daily deterministic
reduction. Root temperature is reconstructed from the learned daily soil
temperature and source-owned `z_soil/rprof`. The three `maint_resp_slope`
parameters remain explicit and are not concatenated into the neural encoder.
Fixed `coeff_maint_zero` is supplied by
`CanonicalRetainedTailAdapter.static_owner.context.stomate_static.parameters`,
then compacted to the active PFT axis when the new head is constructed. It is
not read as a Teacher target, copied into `PFTAxisInput`, or exposed as an
opaque network feature.

This formula does not reproduce 48 half-hour forcing records or scan 48
states. The correction learns only the missing true-daily aggregation effect.

## New Head

Reuse the parent maintenance coordinate hidden states. Reshape them to compact
PFT by plant part, pool the source-owned aboveground and belowground part
groups separately, and apply one two-layer MLP per group:

```text
768 -> GELU(256) -> 1 -> tanh
```

The two branches contain `394,242` parameters. Removing the old 769-parameter
head gives an expected total of `84,148,646` parameters. Each final layer is
initialized to exact zero, so the new candidate begins at the source-shaped
daily baseline rather than an arbitrary random maintenance flux.

The parent Attention and non-maintenance parameters may initialize the new
candidate. The old maintenance head and parent optimizer state may not. A new
optimizer must be initialized if training is later authorized.

## Gates Before Training

Local acceptance requires exact parameter-tree ownership and count, eager and
JIT forward, finite complete loss, finite nonzero gradients in the new head
and reused Attention path, inactive/bare isolation, PFT permutation
equivariance, and one bound daily transition with passing state and budgets.
Zero correction must equal the source-shaped daily formula exactly. The code
must contain no final-inference half-hour reconstruction or scan.

No optimizer update is authorized by this architecture contract.

## Bounded Science Gate

A later separately authorized train/train screen must reduce median
maintenance score by at least `20%` at both 7 and 30 days relative to the
unchanged update-1000 diagnostic reference. A small relative win is not
sufficient. GPP, growth and total autotrophic respiration, NPP, AGB, BGB,
total live biomass, and ecosystem carbon must have nonpositive median score
change, no group may regress by more than `1%`, and all hard/nonfinite counts
must remain zero. The absolute target-oriented acceptance contract also
remains binding.

Teacher v6 is reused unchanged. Historical v5, validation values before the
train/train gate, sealed test, 365-day rollout, Gate D2, paid training, and
product work remain closed.

## Local Implementation Result

The implementation is complete at `84,148,646` parameters. It preserves the
parent coordinate-aligned Attention body and all non-maintenance heads,
removes only the old 769-parameter maintenance projection, and initializes
both new final layers to exact zero. The retained static owner is
`CanonicalRetainedTailAdapter.static_owner.context.stomate_static.parameters`;
`coeff_maint_zero` is compacted when the process head is constructed and is
never exposed to the neural encoder.

The executable source formula applies the two frozen nonnegative stages:
`baseline=max(1+slope*(T-273.15),0)` followed by
`effective=max(baseline+delta,0)`. Reserve hidden state and materialization use
the aboveground branch for trees and belowground branch otherwise. The PFT
state extractor derives its biomass-element width from the canonical input
and selects `ICARBON`, covering the current one-element PFT14 boundary without
assuming a wider historical layout.

All six focused local tests pass. They cover contract/hash and exact parameter
ownership, eager/JIT finite forward, finite complete loss, finite nonzero
gradients through both new branches and the reused Attention body, exact
bare/inactive zero, nonzero-correction PFT permutation equivariance, canonical
single-day binding, exact water/carbon budgets, and the absence of a 48-step
inference scan. Ruff, Python compilation, and `git diff --check` pass.

The next gate is a zero-update real-v6 GPU admission initialized from the exact
update-1000 parent checkpoint. No optimizer or training update is authorized
until that admission passes.
