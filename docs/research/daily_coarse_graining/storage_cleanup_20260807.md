# Explore1000 Runtime Storage Cleanup

Status: **complete**.

## Scope

The cleanup reduced file-count pressure under
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime` without changing the formal
Teacher v6 training product. Historical v5 worker directories were archived,
not discarded.

The successful archive job was Explore1000 job `14502703`. It completed in
`01:26:54` with exit code zero on one `cnmix` CPU. Two earlier attempts stopped
fail-closed before deleting any v5 worker: job `14502662` preserved a unique
historical Git commit, and job `14502665` rejected ownership-only differences
from GNU tar's metadata comparison. The accepted job used temporary extraction
and recursive byte-content comparison, so UID/GID differences were excluded
without weakening data validation.

## Historical v5 Archive

The historical dataset manifest remains at:

```text
runtime/outputs/training/pft14-daily-teacher-669-1961-2010-v5-7397d1e-w100/dataset_manifest.json
```

Its SHA256 remains:

```text
89ea2f24bad8f4b39129937f4106285dd53936c35d8dc610810f865b02d4e508
```

The 100 loose worker directories were replaced by 100 uncompressed tar files
under:

```text
runtime/archives/pft14-daily-teacher-v5-workers/
```

Each tar was listed, temporarily extracted, recursively compared with its
source directory, hashed, and recorded before the source directory was
removed. There are no remaining source worker directories or `.incomplete`
artifacts. The 100-row archive manifest is:

```text
runtime/archives/pft14-daily-teacher-v5-workers/archive_manifest.tsv
```

Its SHA256 is:

```text
7007644e9c025ed0b069ee2f220ab8c920407f10cb9fc1b04c65a4a12a767786
```

Restore only the worker needed by a historical experiment:

```bash
tar -C runtime/outputs/training/pft14-daily-teacher-669-1961-2010-v5-7397d1e-w100/workers \
  -xf runtime/archives/pft14-daily-teacher-v5-workers/worker-NNN-of-100.tar
```

Historical v5 must not be joined into formal v6 training.

## Other Cleanup

Regenerable XLA/JAX/uv caches, historical experiment worktrees, and superseded
Git transfer bundles were removed. The current complete research bundle was
retained at:

```text
runtime/transfers/ORCHIDEE-MAN-JAX-research-49384ce.bundle
```

Its SHA256 is
`e0089f296369c49ce610c6a3b0894d2c04c83db02f8d85ead851ee4cd0c2fb35`.
Unique historical Git history and tracked worktree differences were preserved
under `runtime/archives/worktree-patches/` before worktree removal.

Across the complete server operation, runtime file count changed from 297,053
to 143,250, a reduction of 153,803 files. Runtime size changed from
363,721,217,730 to 340,663,055,143 bytes, releasing 23,058,162,587 bytes
(21.475 GiB).

## Protected v6 Evidence

The accepted v6 dataset and evidence were not moved or rewritten. Post-cleanup
hashes are unchanged:

- dataset manifest: `5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99`
- acceptance report: `8784ed42dc64ef77c9642f613eb2e5940635edc6e744a1af4ea8634293baa8e3`
- statistics JSON: `ab7f040a930071bd5604c17355a6493b23f3c76cba1e3084dcd5d5653101a4bf`
- statistics NPZ: `e88d7cec504c6051aab6c9feacb619dca94961730ffa18f840563a48fc815a3a`

