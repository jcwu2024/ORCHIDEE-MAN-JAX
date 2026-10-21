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
The frozen release's self-hash/contracts and original plan hash are verified;
each annual completion check verifies actual input hashes. The historical
producer source tree is not revalidated against today's different checkout.
The config, PFT catalog and typed contract file byte hashes remain strict gates.
`load_plan` exposes this explicit inspection option; generation defaults retain
strict source verification. The real proof reports this provenance limitation.

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

## Accepted bounded result

Point `087.0-109.0` has completed production migration at deployed commit
`98fcc1a6fda8640fd27e46d50e9726c198eb1efe`. All 200 objects were restored with
exact bytes/mode/nsmtime, all 50 annual production readers matched, and the
shadow missing-file proof passed through real sample/window/training adapters.
After technical review, exactly those 200 originals were retired. Actual-root
reads of all 200 logical objects, all 50 checkpoints, all three temporal splits
and the full 33450-reference existence scan passed. All 103 original control
files retained their hashes and file attributes.

The package SHA256 is
`22eaa71afc03f98987e77bf7676732e96be8ad737fab46821e94a72090e66ea8`;
the accepted reader proof SHA256 is
`c8a4e536fc24bf88c841063b9e42e2472a0d7732d668408bdff3c6877292a25e`.
Dataset files changed from133903 to133705, a net reduction of198. Two obsolete
code checkouts (4980 files/148 directories) were verified into one archive and
removed. The retained active checkout has2490 files and maintenance has25;
including six observed deployment Git pack/index files gives a scoped net
increase of2323 files for this bounded stage. No pre-deployment Git-object
inventory exists, so that attribution is limited to the observed deployment
window; it is not a whole-account or whole-project before/after claim.

The first new package took6.364s. Proof took114.632s, including27.103s for the
first-only full index scan. Excluding that scan, package+proof projects17.42h
for668 remaining points before retirement/storage variation. This diagnostic
used test03, nice15, two-CPU affinity and one BLAS thread; it does not establish
stable one-CPU throughput. Peak RSS was about1.46GiB.

The concrete pending production proposal uses cnmix, one CPU,36h hard wall
time and35h30m soft admission budget, max669 points with completed points
skipped. At the project rate0.07, the ceiling is CNY2.52. The fixed submission
wrapper overrides the generic launcher's12h directive with36h; target Bash
syntax/argument expansion and parser dry-runs passed. No paid job was submitted.
Fresh dependency/pin admission and explicit user resource approval remain
required before the fixed command runs.

Local reviewable evidence is in
`E:/ORCHIDEE-MAN-JAX/outputs/maintenance/20261016/server-file-count/TEACHER-POINT-20261021/RESULT.md`.
The remote maintenance namespace of the same suffix contains `RESULT.md`,
`INDEX.json`, `file-count.json`, the durable ledger, exact reader/retirement
proofs, restoration instructions, verified source archives and the complete
Git bundle. Its fixed command is:

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/maintenance/teacher-point-storage-20261021/submit-full-migration.sh --submit
```
