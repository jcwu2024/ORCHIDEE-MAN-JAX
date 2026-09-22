# Runtime working-file archival, 2026-09-22

The user requested archival of unused runtime assets because filesystem entry
quota is harder to expand than storage capacity. A read-only inventory counted
351386 entries,341543 regular files,169078 files smaller than64KiB and506.781GiB
logical file sizes under the server runtime. These are runtime-local counts,
not account-wide quota headroom. Active jobs may create additional entries.

The first bounded operation targets55 completed checkout/worktree working
trees, declared in `manifests/runtime_archive_20260922.json`:131998 regular
files,135183 removable entries, about1.90GiB uncompressed. Preserve all `.git`
metadata in place because shared clones depend on `.git/objects` chains.
Keep the entire active shared-execution0507e1b and shared-dual8cce49e trees,
shared-joint-c2498e4-full deployment donor, and shared-learning4408daf source.
Existing Git-only incomplete clones have no useful working-file reduction
and are not included. Training data, weights, results and caches are unchanged.

Activity evidence includes scheduler workdirs/commands and process checkout
references ontest01-test04/gln01. Privilege-separated sshd and Singularity
supervisors hide cwd/environ; their readable child workers and scheduler
records are inspected. Other unresolved process inspection blocks the tool.
Paid node SSH is unavailable; the only project allocation15328832 references
the explicitly preserved dual checkout. Other current shared-account jobs
belong to a different project and were not changed.

Tool:`scripts/hpc/archive_runtime_checkouts.py`. It defaults to inventory.
The selected archive excludes only root Git stores (nested repos are rejected),
preserves untracked working files, uses a null-delimited target list and never
follows symlinks. It compares the complete archive with originals before
removing working files; before/after entry snapshots also reject additions or
changes during packaging. Each archived directory keeps a restoration marker.
The archive is recoverable code history, not a retained training checkpoint.

Sol/high reviewed destructive scope and dependency handling; identified path
normalization, multiple process-reference occurrences and file-set completeness
issues were addressed before execution. Five tiny tests pass on free test02
using its native GNU tar1.26, including byte-preserving restore of tracked and
untracked fixtures, resume after a completed pack, and refusal to remove newly created files. Initial container
tests exposed group-name remapping (fixed with numeric owners); native execution
exposed an unsupported redundant tar flag, removed before any removal occurred.

Final archive destinations (both under the canonical runtime `archives/`):

- `inactive-checkouts-20260922.tar.gz`:45 checkout working trees.
- `inactive-worktrees-20260922.tar.gz`:10 historical worktree working trees.

Their adjacent `.tar.json` files are compact manifests and operation reports.
The exact split target lists are `manifests/runtime_archive_checkouts_20260922.json`
and `manifests/runtime_archive_worktrees_20260922.json`. No paid job or GPU work
is added; packing/verifying runs on free test02/test03.

The first55-tree pack reached its20min bound while writing the worktree tail;
no working files were removed. `repack_runtime_checkout_prefix.py` recovered
the complete45-checkout prefix (118960 archive members), stopping at the first
worktree header before the truncated portion. This avoided rereading/repacking
the completed checkout source files. The recovered archive must still pass
the exact current member-set check plus GNU tar byte comparison before any
cleanup; this is the tool's tested `--resume-packed` path. The remaining10
worktrees are packed separately. Revised free-operation limits are60min for
checkout verification/removal and30min for the smaller worktree operation.

Restore one working tree before using its archived source path:

```bash
tar -xzf /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/archives/inactive-checkouts-20260922.tar.gz -C /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime checkouts/NAME_FROM_MANIFEST
```

Existing `.git` metadata remains in place; restore the working files into the
same path, then remove only that checkout's `ARCHIVED_WORKING_FILES.md` marker.
For a historical worktree, use the worktrees archive and `worktrees/NAME_FROM_MANIFEST`.
Do not treat absent archived working files as intentional source
deletions to commit. Historical experiment reports keep their original source
paths; this manifest documents how to restore them.

Execution result and after-counts are recorded below once complete.
