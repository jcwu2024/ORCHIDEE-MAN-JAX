# Release Checklist

## Required Before the First Public Release

- Select and add the JAX repository license.
- Confirm whether the Fortran source may be redistributed; keep it excluded
  until that decision is documented.
- Add project authors, citation metadata, and the paper DOI when finalized.
- Run `uv sync --frozen --extra dev` from a clean source checkout.
- Run the release CI suite and `orchidee-jax inventory` against the external
  data package.
- Review cluster-specific Slurm account, partition, wall time, memory, and
  scratch paths before the 669-landpoint submission.
- Complete and archive the 669-landpoint acceptance summary.

## Repository Boundary

The supported deployment is a Git source checkout. Large scientific inputs,
Fortran references, raw traces, checkpoints, model outputs, and XLA caches are
external. Python wheels contain the model package and CLI but still require
`ORCHIDEE_REPO_ROOT` to point to a checkout containing `configs/` and
`manifests/`.

Historical paper Fortran run scripts are tracked unchanged as provenance.
They are not production launchers and their absolute cluster paths are not
portable.
