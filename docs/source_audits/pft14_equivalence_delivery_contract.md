# PFT14 Equivalence Delivery Contract

This contract defines the evidence required before the JAX implementation may
be described as equivalent to the Fortran PFT14 model. It supersedes any
earlier use of `verified` that meant only that a source citation or JAX test
existed.

## Target

The target is every process path reachable for PFT14 when changing an
independent landpoint, forcing values or source, continuous model parameters,
and supported paper-workflow state. A path that requires a non-PFT14 role or a
different model configuration must be classified with a source-backed
unreachability reason. The 669 paper cases are validation cases, not 669
different implementations.

## Required Evidence Chain

1. **Fortran control-flow denominator**
   - Start at the paper driver and follow the Fortran call graph through
     SECHIBA and STOMATE to modelout.
   - Give every `IF`, `WHERE`, `SELECT CASE`, configuration gate, PFT mask,
     time gate, and state mask a stable source ID.
   - Classify every source ID as PFT14 active, PFT14 conditional, PFT14
     unreachable, or outside the declared workflow. Unreachable and outside
     classifications require source provenance and a reason.
2. **State-transition denominator**
   - Record every process input, same-step output, carried field, module
     `SAVE` variable, restart field, and daily/annual memory.
   - Record producer, consumer, update phase, writeback phase, shape, and
     Fortran source span.
   - Machine checks must reject missing producers, dangling writebacks,
     ambiguous Day-N/Day-N+1 ownership, and production paths that omit a
     carried field.
3. **Independent numerical oracle**
   - Every active or conditional branch needs a minimal numerical case whose
     expected result is produced by original Fortran code or an immutable
     audited Fortran output boundary.
   - A JAX unit test derived from the JAX implementation is regression
     evidence, not an independent oracle.
   - Prefer process-level Fortran micro-drivers. Use narrow full-model traces
     only when the original routine cannot practically be isolated.
4. **Production-path conformance**
   - The optimized compiled path, not only a diagnostic path, must pass the
     same process and state-transition cases.
   - Disabling an optimization is permitted for A/B diagnosis only and never
     establishes Fortran truth.
5. **Integrated scientific acceptance**
   - Cold start, day/year handoff, restart, and paper modelout must pass the
     tolerance policy fixed before execution.
   - The complete 669-case paper mosaic is the final integrated acceptance
     gate after source and state closure. It does not replace them.

## Evidence Levels

Each reachable branch advances independently through these levels:

| Level | Meaning |
| --- | --- |
| `inventoried` | Stable Fortran control-flow ID exists. |
| `source_mapped` | Reachability, source span, JAX kernel, and state contract exist. |
| `jax_regression` | Focused JAX regression/micro-case exists. |
| `fortran_oracle` | Independent Fortran numerical oracle passes. |
| `production_verified` | Production compiled path passes the oracle and state-writeback checks. |

The word `verified` without one of these explicit levels is not sufficient for
an equivalence claim.

## Release Gates

The PFT14 equivalence claim is allowed only when all of the following are true:

- 100% of the reachable Fortran control-flow denominator is classified.
- 100% of PFT14-active and PFT14-conditional entries are
  `production_verified`.
- The state-transition audit has zero missing, dangling, ambiguous, or
  production-omitted edges.
- Cold-start, restart, and year-handoff integration gates pass.
- All 669 paper cases pass the predeclared scientific acceptance policy on the
  production compiled path.

Floating-point equality is governed by
`docs/source_audits/numeric_tolerance_ledger.md`. Bit identity may be retained
as a diagnostic aid, but it is not substituted for source and state closure.

## Execution Order

1. Generate and classify the Fortran control-flow denominator.
2. Generate and close the state-transition denominator.
3. Add Fortran process oracles and fix all reported semantic gaps by module
   order: driver/forcing, SECHIBA, HYDROL, DIFFUCO, ENERBIL, THERMOSOIL,
   STOMATE, restart, modelout.
4. Certify the production compiled path against the same cases.
5. Run the 669-case integrated scientific acceptance gate.

Point-specific trace chasing and additional performance experiments are not
the critical path until these gates require them.
