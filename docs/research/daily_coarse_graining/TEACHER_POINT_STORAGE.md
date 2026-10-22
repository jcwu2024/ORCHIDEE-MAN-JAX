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

## Accepted terminal migration and startup limitation

Sole approved job16007307 completed0:0 at2026-10-22 12:31:24+08:00 on
ibc11b11n11, after21h46m46s (78406 allocated one-CPU seconds). The batch
processed668 points, skipped the previously accepted point, and reported
`stop_reason=complete` after78385.347746s. Batch MaxRSS was1438780K.
The project-rate estimate is CNY1.524561 (0.07/core-hour), below the approved
CNY2.52 ceiling; the actual bill is unknown. No retry/second allocation occurred.

Terminal integrity evidence confirms669 retired points,669 actual packages
matching the map/ledger,133800 members and removed retirement rows,200 per
point, no pins/remaining originals/locks/incomplete stages/proof views.
All103 original manifest/progress files retain their SHA256 and stat identities.
Every package retains the exact dev/inode/size/nsmtime/nsctime fingerprint from
its full byte verification at retirement. Map/ledger package and snapshot
hashes/member sets agree. Full per-package metadata/BLOB verification is reused,
not rerun. Embedded proofs cover133800 exact byte/mode/nsmtime restorations,
33450 annual base/typed/window/checkpoint/completion comparisons, and three
absent-loose temporal-split training-adapter probes per point.

Current availability evidence combines the unchanged original control hashes,
all133800 map/ledger members,669 actual packages and exact retirement
fingerprints, the complete namespace inventory, and the full embedded annual
proofs. No new full production index constructor completed. The default
`verify_files=True` and `verify_files=False` constructors both remained
incomplete in bounded probes and are not marked PASS. A final harness avoided
full index path resolution but retained the production full-plan loader; it
returned no annual results before its identity-guarded stop at300.95s. The
active phase was not instrumented. The planned9-point/9-annual reader
sample (spatial×temporal3×3, T1/T7/T30, typed/checkpoint/completion/adapters)
was not completed and is not claimed as new evidence. Exact observed scope,
times and owned-process exit confirmations are in `terminal-final.json`.
The migration's full per-point proofs remain valid through the unchanged
package/control identities. These checks execute no Teacher transitions or
neural training and establish no new scientific/Fortran-equivalence result.

Measured dataset accounting:133903→773 files (net−133130),2300→2301
directories excluding the root,167547836510→167872273962 logical bytes
(+324437452). The773 files are669 packages,103 unchanged controls and one
central map. The BLOBs retain their original bytes; packaging does not promise
compression. After deducting all29 maintenance files (including the terminal
report),2490 active-checkout files, one job log and six previously observed
deployment Git files, the bounded net reduction is130604 files. The earlier
4980 archived obsolete-checkout files are not credited again. Git attribution
is limited to the recorded deployment window, since no before-object inventory
exists. No whole-account/project quota reduction is claimed. Final namespace
bytes/cutoff are in the terminal report; `file-count.json` stays historical.

Use the normal integrated readers at the unchanged dataset path/manifest,
from immutable98fcc1a6fda8640fd27e46d50e9726c198eb1efe or a compatible
descendant. No environment switch is needed. Legacy direct annual `Path.open`
or `np.load` consumers require integrated code or explicit exact restoration
into a new directory using the command above/INDEX. Do not restore the full
loose dataset merely for routine reading or regenerate Teacher data. Preserve
the immutable checkout/shared Git objects, old Teacher pilot, archives and all
recovery evidence.

The next maintenance issue is full-index startup cost. The default probe was
still incomplete after1271.47s and stopped by exact PID/start identity. This
was shared test03 with audit and reader IO overlapping at adjacent points;
it is not an independent stable benchmark. Elapsed/IO evidence is retained.
The subsequent `verify_files=False` constructor remained incomplete at620.80s
and was also stopped with exact identity and exit confirmation, without opening
a container at the final observed snapshot.
The real manifest has669 consecutive point runs of50 years, not year-wise
point thrashing. Static simulation of8-entry LRU gives669 misses per complete
traversal and2007 for the original three constructor traversals; this is a
sequence-derived count, not an instrumented counter. Parsed map lookup is
cached, but hits still stat paths/manifest identities, and new container opens
read/check200 metadata rows. Low CPU/page-read waits support a metadata/IO
bottleneck hypothesis without quantitatively isolating the cause. Independently,
even `verify_files=False` still resolves each shard path in the full base
constructor; missing loose-path resolution can cost filesystem metadata work
before a container opens. No production
optimization or repeated full benchmark is claimed in this maintenance stage.
The production plan loader also resolves every entry's run-definition/reference
directory even with `require_inputs=False`; this is another plausible startup
cost in the final harness, not a measured phase attribution. Its incomplete sampling is an explicit remaining
check to perform after startup costs are fixed or a bounded selected-plan path
is reviewed. No fourth reader probe was launched in this closeout.

Evidence root:
`E:/ORCHIDEE-MAN-JAX/outputs/maintenance/20261016/server-file-count/TEACHER-POINT-20261021/`.
`RESULT.md`, `server-evidence/terminal-final.json`, original terminal scheduler/
log receipts, terminal ledger audit and reader receipt own the final facts.
The matching remote maintenance namespace has updated RESULT/INDEX, one
compact terminal report, durable ledger/embedded proofs, historical submission/
count records, verified recovery archives and the complete Git bundle.
Storage migration is complete; startup performance is an explicit open issue.
No automatic paid job, retry, cancellation, training, regeneration or merge to
main follows. Scientific work and closed scopes remain in `NEXT_STEPS.md`.
