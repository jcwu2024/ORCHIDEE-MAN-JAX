# PFT14-First, PFT-Extensible Design

This project targets the ORCHIDEE-MAN PFT14 mangrove reference case first, but
process code should remain easy to extend to additional PFTs.

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
