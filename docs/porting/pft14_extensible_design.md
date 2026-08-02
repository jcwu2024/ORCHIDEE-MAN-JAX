# PFT14-First, PFT-Extensible Design

Status: active target contract; the complete Teacher does not yet satisfy it.

This project targets the ORCHIDEE-MAN PFT14 mangrove reference case first, but
process code should remain easy to extend to additional PFTs.

## Current Reality

The current implementation is partly shape-generic, not PFT-pluggable.

- Many SECHIBA and STOMATE kernels preserve a full `nvm` axis and dispatch on
  trait arrays such as `is_tree`, `natural`, `is_c4`, and `is_peat`.
- The production paper config fixes `NVM = 14` and imposes only PFT14 cover.
- Driver initialization materializes paper-specific PFT/MTC mappings and
  vegetation cover.
- The STOMATE parameter loader, orchestration boundary, coupled helpers,
  multiyear modelout summary, traces, and acceptance tooling explicitly name
  or select PFT14.
- Some Fortran processes are themselves tied to canonical numbered PFTs, for
  example the agricultural peat path over PFT12-16. Those semantics cannot be
  made generic by renaming an array index.

Therefore changing `NVM`, removing PFT14, or adding a new PFT is not currently
a supported end-to-end operation. Dynamic array shapes alone do not establish
that capability.

## Current Target

- Reference PFT: PFT14, mangrove.
- Current validation case: `reference/case_001_071`.
- Current hydrology trace: `traces/hydrol_1961_v9`.
- Active run configuration:
  - `NVM = 14`
  - all imposed vegetation cover is zero except PFT14
  - `SECHIBA_VEGMAX__00014 = 1.0`
  - `PREF_SOIL_VEG__00014 = 4`
  - `PFT_TO_MTC__00014 = 2`

## Design Rules

- Keep PFT as an explicit axis in state, parameter, and output arrays.
- Avoid hard-coding PFT14 inside process kernels.
- Put case selection, such as "validate PFT14 only", in driver/config/test code.
- Keep process kernels compatible with shapes such as:
  - grid axis: `npts`
  - PFT axis: `nvm`
  - soil tile axis: `nstm`
  - hydrology layer axis: `nslm`
- Use named indices or small selectors at boundaries, not magic constants in
  process equations.
- Every process implementation must cite Fortran provenance: file, subroutine,
  and line span, or an audited trace reference.

## Target Teacher Architecture

Use two explicit layers.

### PFT catalog and run layout

A source-backed catalog entry owns:

```text
pft_id                 stable semantic identity
fortran_pft_id         canonical source identity when one exists
mtc_id                 source MTC mapping
traits                 tree/grass/crop, C3/C4, natural/managed, peat, ...
parameters             complete source-backed parameter record
capabilities           process families required by this PFT
```

A run layout selects catalog entries, fractions, and active masks. The layout
creates dense arrays for JAX execution, but restart and output metadata retain
stable `pft_id` values. The bare-soil slot remains explicit and cannot be
silently confused with a vegetated PFT.

### Process capability registry

Parameter-only PFT differences use shared process kernels. New code is needed
only when the Fortran source selects a genuinely different process family,
such as crop/STICS, mangrove/peat controls, or a phenology mode. Each
capability has an explicit source-backed implementation and support status.

This is intentionally not one Python plugin per PFT. That design would copy
shared equations into multiple implementations and make source equivalence
harder to maintain.

## Boundary Pattern

Process functions should receive full arrays where Fortran does, even when the
current test extracts only PFT14:

```text
params[pft, ...]
state[grid, pft, ...]
fluxes[grid, pft, ...]
outputs[grid, pft, ...]
```

Reference tests may select PFT14 with a named selector:

```text
pft14_index = 13  # zero-based PFT14
```

The selector belongs in tests, diagnostics, or modelout extraction, not inside
generic process code.

## First Implementation Implication

The first HYDROL target is tile4 bottom-layer mineral CWRR parity. Even there,
the implementation should separate:

- generic CWRR table construction
- generic `hydrol_soil_coef` mineral kernel
- reference-case extraction for tile4/bottom-layer trace comparison

That keeps the first milestone narrow without making the model a one-off PFT14
script.

## Admission Tests

Before calling the Teacher PFT-selectable:

1. construct all parameter, trait, state, restart, and output PFT axes from a
   declared catalog/layout rather than the paper constant `NVM=14`;
2. run the unchanged paper PFT14 layout with no numerical regression;
3. remove one inactive supported PFT and prove active PFT state is unchanged;
4. run two supported active PFTs and prove their state and flux ownership stay
   separate;
5. permute execution slots while retaining stable IDs and compare remapped
   results;
6. reject a PFT whose required process capability is not implemented;
7. keep canonical-number Fortran branches explicit and guarded;
8. require source branch closure and numerical evidence for every newly
   claimed PFT.

Passing these structural tests makes addition and removal technically safe.
It does not grant scientific support to PFTs that lack Fortran-equivalent
process coverage and validation data.
