# Physical-Parameter Candidate Registry v1

Status: accepted candidate inventory; no new public tunable channel is
promoted by this document.

Date: 2026-08-04.

## Decision

The canonical PFT14 Teacher currently exposes many numeric configuration
values, but they are not interchangeable inversion variables. The
source-driven registry groups related coefficients by process ownership and
classifies them before any neural-interface expansion.

The accepted inventory contains:

| Class | Parameter families | Meaning |
| --- | ---: | --- |
| Inversion candidates | 9 | Plausibly observable continuous PFT14 responses |
| Sensitivity-only | 17 | Meaningful but currently confounded, weakly observed, or thresholded |
| Discrete controls | 2 | Configuration or shape choices, never ordinary gradient variables |
| Not identifiable now | 6 | Inactive for PFT14 or underconstrained by current observations |
| **Total** | **34** | **91 scalar components after expanding grouped/vector families** |

The difference between 34 families and 91 components is intentional. For
example, the 20 biochemical temperature coefficients and the 12-class
`CWRR_KS` table are joint parameter families. Counting every coefficient as
an independent inversion target would overstate both information content and
identifiability.

## Red-Mangrove Shortlist

The first wave remains the four paper-calibrated families:

- `VCMAX25`;
- `MAINT_RESP_SLOPE_C`;
- `ALLOC_MIN`;
- `RESIDENCE_TIME`.

These already have accepted canonical-Teacher gradient evidence. The second
wave is deliberately bounded to five families:

- `ARJV`/`BRJV` Jmax-to-Vcmax acclimation;
- `SLA_MIN`/`SLA_MAX`;
- `FRAC_GROWTHRESP`;
- `CONTROL_SALINITY_MIN`;
- `CONTROL_INUDATE_MIN`.

Second-wave membership is not promotion. Each family still needs a defensible
PFT14 prior, controlled Teacher perturbations, Gate-D gradient cases, and an
observable-identifiability argument. Mangrove aerial-root allocation and
ventilation parameters remain high-value sensitivity families, but sparse
root observations currently make independent inversion premature.

## PFT Expansion Rule

The registry separates PFT-axis parameters, soil-class parameters, global
parameters, and mangrove capability parameters. A future PFT does not inherit
the PFT14 shortlist automatically:

- shared process parameters may remain candidates when that process family is
  source-validated for the new PFT;
- phenology, senescence, crop, peat, or other capability parameters become
  eligible only when the corresponding PFT activates those branches;
- a parameter inactive for PFT14 can be valid for a future PFT without being
  learnable from PFT14 data;
- supporting another PFT requires its own priors, Teacher perturbations,
  gradient evidence, and held-out observations.

This is why the neural architecture keeps named parameter and trait channels
with a PFT axis instead of a fixed anonymous vector or a PFT identity
embedding.

## Source And Validation Boundary

The machine contract is
`manifests/coarse_graining/physical_parameter_candidate_registry_v1.json`.
Its loader verifies source-file SHA256 values, source occurrence of every
run-definition key, unique registry ownership, classification and shortlist
consistency, and declared counts.

`SECHIBA_QSINT` was considered but excluded from this source-bound registry:
the key is present in runtime materialization and JAX, while the archived
Fortran source tree does not contain its configuration owner. It must not be
promoted until that provenance discrepancy is resolved.

Verify locally with:

```powershell
conda run -n ORCJAX python scripts/dev/verify_physical_parameter_registry.py
conda run -n ORCJAX python -m pytest tests/unit/test_physical_parameter_registry.py -q
```

The next implementation gate is Gate E. It should start with the accepted
first-wave parameter interface and preserve extension points for the second
wave; it must not add all registry members as neural inputs.
