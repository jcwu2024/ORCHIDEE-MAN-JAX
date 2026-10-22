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

## Accepted migration and production startup

Sole approved job16007307 completed0:0 at2026-10-22 12:31:24+08:00 on
ibc11b11n11, using78406 allocated one-CPU seconds. It processed668 points and
reused the first accepted point. Estimated CNY1.524561 is below the approved
CNY2.52 ceiling; actual billing is unknown. All669 points are retired, with
133800 original objects absent and103 original controls unchanged.

Measured dataset accounting remains133903→773 files,2300→2301 directories
excluding the root,167547836510→167872273962 logical bytes (+324437452).
The773 files are669 packages,103 original controls and the central map.
Packaging preserves original bytes and makes no compression claim. Completed
full per-point byte/restore/annual-reader proofs are reused through unchanged
exact retirement package fingerprints; no167GB payload rehash was repeated.

Code7d14adbded43b6590cf9167d0770e1eedd2d016f adds operation-local index
admission, shared by coherent typed→base construction. It checks each distinct
directory/package once and rechecks map/manifest/package/directory identities
at exit. Admission verifies bound canonical source-row metadata and exact
covering-index member inventory; actual selected reads still verify row
metadata, BLOB size/SHA and package identity. Maintenance retains full metadata
validation by default. There is no persistent admission trust. Loose-path,
symlink/escape and hash gates remain; retired empty directories are optional.
The ordinary plan loader resolves each distinct exact input once per call,
retaining original leaf-symlink semantics and no cross-call path cache.

Free test03 validation passed actual public33450-record base and coherent
constructors with `verify_files=True, verify_hashes=False`, including the coherent
parent. First base122.087s; subsequent warm coherent11.369s. The complete original
33450-entry production plan passed in2.090s with `require_inputs=False,
verify_release_sources=False`, preserving original plan/release/noncode gates
in historical reading mode. The nine spatial×temporal3×3 annual cases passed
81 direct and27 public T1/T7/T30 windows, base/typed day/mask/class checks,
nine real training adapters, checkpoints and completion checks. Worker199.165s;
supervisor205.715s, rc0 and exact owned-child exit confirmed.

The first two bounded attempts stopped during imports and are preserved. The
second localized import-time JAX lowering in an unchanged physical module.
At load≈50, a one-second CPU sample selected idle cores25/26, nice15 and
single-thread BLAS/OpenMP settings; module imports then completed. Sequential
cache warming and CPU selection prevent a controlled speedup-ratio claim.
Representative1/3-point package checks took2.353/1.263s; extra metadata-only
constructors took3.7–3.95ms and used SQLite's covering path index. Process
read_bytes still showed filesystem readahead; this is not zero physical IO.

Post-reader audit reconfirms103 control SHA/stat identities,669 exact retirement
package fingerprints,133800 map/ledger bindings/absent originals, and the unchanged
773-file namespace in10.358s. New local tests plus existing reader/plan tests:
67 passed/1 Windows symlink-permission skip; existing storage maintenance:13 passed.
No Teacher transitions, neural training or new scientific equivalence is claimed.

Both immutable checkouts remain: old98fcc1a6fda8 and new
`runtime/checkouts/teacher-point-startup-7d14adbded43`. Retain their shared root
Git objects and recovery evidence. Routine reading uses the unchanged original
manifest/path, without an environment switch or loose restoration. Historical
direct `Path.open`/`np.load` consumers still need integrated code or exact restore.

This startup deployment adds2519 checkout files/33188494 bytes, two attributable
Git files with305822 net Git bytes (including FETCH_HEAD+49), and seven maintenance
files/197620 bytes. Against the previous bounded scope this adds2528 files and
33691936 bytes:5827 retained files/168419089611 bytes, net128076 fewer files and
871253101 additional bytes against the original scope. Whole-account quota is
not inferred; unchanged Git objects and previously archived checkouts get no
new credit. Detailed file inventories/cutoffs are retained in the startup report.

Evidence root:
`E:/ORCHIDEE-MAN-JAX/outputs/maintenance/20261016/server-file-count/TEACHER-POINT-20261021/startup-20261022/`.
`startup-summary.json`, raw production-readers-v3.jsonl, terminal-readers.json and
post-read-integrity.json own this acceptance. Earlier incomplete terminal reports
remain historical; their fresh-reader startup limitation is superseded here.
Initial GitHub proxy/direct pushes failed; the final bounded push succeeded.
Reviewed code and documentation are on origin. A verified134949-byte incremental
code bundle remains locally/remotely, requiring old98fcc1a6fda8 objects.
No new paid job, retry of migration, restoration/deletion, data write, Teacher
generation, training or merge to main follows. Scientific work remains in NEXT_STEPS.
