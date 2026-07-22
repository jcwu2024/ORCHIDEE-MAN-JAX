# Runtime workspace

This directory is the canonical server-local home for unversioned ORCHIDEE-MAN-JAX
runtime assets. Everything except this file is ignored by Git.

Expected layout:

```text
runtime/
  assets/    # reference restarts, run.def snapshots, and checkpoints
  cache/     # XLA, XDG, and other regenerable caches
  data/      # forcing and static scientific inputs
  logs/      # scheduler and application logs
  outputs/   # generated benchmarks, shards, checkpoints, and results
  transfer/  # Git bundles used to update the server checkout
```

Project-specific uv environments live in the repository-level `.venvs/`
directory. Large runtime assets and environments must not be committed.
