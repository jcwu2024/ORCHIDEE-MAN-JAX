# Teacher point storage

The accepted v6 release may store one landpoint's complete 1961–2010 history
in `points/<landpoint_id>.sqlite`, with the four original objects per year as
unchanged BLOBs. `teacher-storage.json` at the dataset root binds each complete
point package to the original `dataset_manifest.json` SHA256. The original
root/worker manifests and progress files remain unchanged. Storage discovery
requires no environment variable. A mapped member takes priority over a loose
copy. Serialized classes, scientific records, splits and masks retain their
original identities.

`teacher_assets.py` supplies read-only, immutable SQLite access, verifies each
opened object's actual SHA256, and keeps at most eight connections per thread.
`daily_markov_contract`, `markov_dataset`, `typed_sidecar` and `teacher_shards`
use it for production loading, completion checks and annual checkpoints.
Generation and original-manifest aggregation into a mapped release are refused.
An explicit alternate aggregation report remains available.

Historical checkouts and scripts that directly call `np.load`/`Path.open` on
original members need exact loose restoration or this integrated code version.
In particular, the historical cold-start promotion shell and v4→v5 migration
are not transparent consumers. Prepared `.pkl.gz` training data and shared initial
input caches are separate assets; this migration does not package them.

## Maintenance interface

Run from an exact committed daily checkout using the canonical CPU Python:

```bash
PYTHON=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python
DATA=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/pft14-daily-teacher-v6
WORK=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/maintenance/teacher-point-storage-20261021
"$PYTHON" -B -m research.daily_coarse_graining.teacher_point_storage --dataset-root "$DATA" --work-dir "$WORK" plan
"$PYTHON" -B -m research.daily_coarse_graining.teacher_point_storage --dataset-root "$DATA" --work-dir "$WORK" pack --point 087.0-109.0
"$PYTHON" -B -m research.daily_coarse_graining.teacher_point_storage --dataset-root "$DATA" --work-dir "$WORK" prove --point 087.0-109.0 --proof "$WORK/point-proof.json" --scan-full-index
```

The SQLite ledger `WORK/migration.sqlite` holds the original 200 paths, file
identities, hashes, package SHA, restoration and reader evidence, events and
per-file retirement intent. No per-year receipt files are created. Proof compares
all 50 years through real base/typed/window/checkpoint/completion readers with a
full exact restoration. A separate shadow release then physically lacks all 200
loose members and exercises actual index/sample/window/training adapter readers.
It records the deployed Git commit and seven source module hashes. This proof
does not execute Teacher transitions or neural training.

Retirement additionally requires an internal review token and a current dependency
audit JSON with `dataset_manifest_sha256`, `active_queued_audit_complete: true`,
future `expires_unix`, and exact release-relative `preserve_paths`. Active/queued
legacy readers must pin their required original paths. Full package verification
and map/package identities are checked before and immediately around each exact
leaf unlink. Durable intent precedes unlink; an interrupted removal resumes only
when that intent exists. Recreated retired files and conflicting restores fail.

Restore into a new inspection directory with `restore --point ID --destination
NEW_DIRECTORY`. To put original loose files back, use `--destination "$DATA"
--original-paths`; existing files must already have identical bytes/mode/nsmtime.
The package and map remain authoritative. Restore never overwrites a conflict.

## Failure and recovery boundaries

The dataset-root `teacher-storage.lock` and maintenance `migration.lock` use
exclusive creation. No automatic stale-lock takeover exists. Before removing a
stale lock, inspect its recorded host/PID/root/work/time, confirm the exact process
is absent on that host and no relevant queued/running job owns this work, record
the lock's SHA and stat identity, then recheck the identity and remove only that
named lock. Preserve a copy of its bytes in the maintenance recovery record. A
PID alone is insufficient because it may have been reused. Do not remove locks
belonging to another live work directory.

A completed stage is fully verified and reused after interruption. An incomplete
`<point>.building.sqlite` fails closed. After recording its current SHA, explicit
`discard-incomplete-stage --point ID --stage-sha256 SHA` removes only that stage
and its exact SQLite sidecars, and only after all 200 originals still match the
ledger. A valid complete stage is never discarded by this command. Atomic JSON
publication leaves a `.publishing` file on interruption and refuses to overwrite
it; compare its bytes to the durable ledger/map before an exact reviewed repair.

A failed proof preserves its owned `<point>.proof-view` for inspection. It is not
silently reused or recursively removed. Compare its exact namespace/bytes with
the ledger, preserve the failure evidence, and move that owned view to a fresh
maintenance recovery name before retrying. The next proof creates a fresh view.
These repairs require explicit identities and ownership checks, not a blanket
`rm -rf`, force-unlock or automatic retry.

## Production limits and accounting

The first real point is a bounded free CPU-node validation. The full approximately
167 GB workload is production work and must use an explicitly approved finite
`cnmix` allocation. The serial batch entry and Slurm launcher require a maximum
point count, elapsed budget and exact commit; they do not submit themselves.
Paid submission needs the user's reviewed resources, wall time, command, working
directory and worst-case cost. Use the daily deployment document's conservative
CPU estimate CNY0.07 per allocated CPU-hour (the generic skill still says0.05).

669 points replace 133800 original objects with 669 packages. With 103 retained
manifest/progress files and one central map the theoretical release has 773
files before pins. Report measured removed/added files and directories, deducting
all actual maintenance receipts, ledger, code checkouts, retained originals and
other artifacts. Account quota snapshots alone do not prove a project reduction.
No full migration or source retirement is implied by a successful reader proof.
