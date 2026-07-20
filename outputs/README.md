# Runtime Outputs

Generated model output, checkpoints, validation reports, and XLA compilation
caches belong here during local development and are not tracked by Git.

`_local_artifacts/` may hold quarantined compiler/build products removed from
the repository root. It is not scientific output and may be deleted after the
corresponding source and evidence have been committed.

For production runs, set `ORCHIDEE_OUTPUT_ROOT` to a scratch or project output
directory outside the cloned repository.
