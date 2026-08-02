# PFT Catalog V1

Status: Gate B1 foundation implemented; additional PFTs remain unsupported.

## Runtime Form

PFTs are data-backed entries, not copies of `jax_orchidee` process code. The
versioned catalog at
`configs/pft_catalogs/orchidee_man_paper_250919.json` declares:

- a stable `pft_id` independent of array position;
- canonical Fortran PFT and MTC identities;
- source-derived traits and indexed parameter ownership;
- required process capabilities and their Fortran/JAX owners;
- named dense layouts used for JAX execution.

Shared SECHIBA and STOMATE formulas continue to receive full PFT-axis arrays.
Only genuinely different source process families use capability dispatch.

## Current Scientific Scope

The exact paper layout remains `paper_250919_legacy14`. Its active supported
entries are `bare_soil` and `mangrove_pft14`; canonical PFT2-PFT13 entries are
recorded as `structural_only` and cannot be activated in production. A new PFT
must receive source branch closure and numerical evidence before its status
can become `source_validated`.

## Stable-ID Boundaries

`RunScalars` and the STOMATE parameter bundle retain a `PFTRunLayout`.
Indexed run.def parameters are selected through each entry's
`fortran_pft_id`, so compacting or permuting execution slots cannot silently
attach another PFT's parameters. Modelout can select a column by `pft_id` and
records its catalog, layout, source PFT and MTC identities. New JAX STOMATE
restart files and multiyear state checkpoints record the same layout metadata.

The helper `remap_pft_axis` maps state by stable identity for removal,
addition, and reorder operations. Missing target identities are explicitly
filled; slot coincidence is never treated as identity.

## Remaining Gate B1 Work

- bind every restart import path to stable-ID verification/remapping;
- remove remaining paper-only PFT selectors from generic orchestration
  boundaries while retaining explicit paper diagnostics;
- prove a complete compact-layout cold start and restart cycle;
- add a second PFT only after its Fortran reachable branches and numerical
  evidence are accepted.
