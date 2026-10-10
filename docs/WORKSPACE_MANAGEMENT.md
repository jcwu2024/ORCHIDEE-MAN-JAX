# Fixed workspaces and versioned research

Updated: 2026-10-10. This policy manages code and handoff; it grants no scientific
experiment, training, paid allocation, or automatic research merge.

## Local entry points

| Purpose | Windows working directory | Branch |
| --- | --- | --- |
| Accepted Teacher / shared project maintenance | `E:/ORCHIDEE-MAN-JAX` | `main` |
| Daily neural operator research | `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining` | `research/daily-coarse-graining` |
| Global physical-model throughput research | `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput` | `research/global-throughput` |

Assign each Codex chat and terminal one fixed working directory. Confirm it
with `git rev-parse --show-toplevel` and `git branch --show-current` before edits.
Opening a file in another workspace does not change a chat's working directory.
Do not switch branches in another chat's checkout. Parallel tasks sharing a
branch need explicit file ownership; use a separate worktree for a separate line.

Daily onboarding is `docs/START_HERE.md`, `docs/NEXT_STEPS.md`, `docs/CODE_MAP.md`
in the daily checkout, followed by `DOCUMENT_STATUS.md` and the named contract.
Global onboarding is `docs/research/global_throughput/README.md` in the global
checkout. Their scientific status and closed grants remain independent.
Research results are not automatically accepted onto `main`: review the exact
change, scientific gates and deployment impact before any later merge.

## Commit and synchronize

Commit at a coherent stage: an implemented and checked change, a completed
diagnostic/report, or a handoff. Review `git status`, `git diff` and explicit
untracked paths; stage only the files belonging to that line. Keep source,
tests, compact reports and portable configuration in Git. Do not add local
credentials, environments, Teacher shards, checkpoints, HLO/binaries or bulk
generated outputs. Check pending file sizes and obvious credentials before push.

Push the branch after each reviewed checkpoint and before a handoff; record the
full commit SHA and remaining dirty paths. An upstream is configured for each
research branch. Use ordinary `git push`, never automated force push/reset/clean.
Fetch and inspect divergence before fast-forward updates. A failed push leaves
the local commit valid; preserve a verified Git bundle and disclose the failure.

Run the read-only inventory from any checkout:

```powershell
conda run -n ORCJAX python scripts/dev/workspace_status.py --workspace-root E:/ORCHIDEE-MAN-JAX
git -C E:/ORCHIDEE-MAN-JAX worktree list
```

The inventory performs no fetch, push, branch switch, cleanup or submission.
Its ahead/behind counts use locally known refs; verify remote heads separately
with `git ls-remote --heads origin` when synchronization matters.

## Assets and execution paths

Large local assets stay at their existing paths under the root workspace:
`data`, `reference`, `traces`, `fortran_source`, `outputs` and `runtime`.
Daily worktree asset junctions reuse existing ignored child directories without
copying data. Tracked README/manifests remain checkout-owned. `runtime` is not
junctioned into its descendant worktree (that would create a cycle); use the
declared absolute root runtime path. Global's existing output tree stays local
to its checkout. New reports must name the actual input/output paths.
Use `conda run -n ORCJAX` with the intended cwd; confirm imports resolve into that
checkout, rather than trusting an old editable environment install.

Do not remove a junction or an asset target as ordinary Git housekeeping.
Git worktrees share the main repository's `.git` objects: never delete the root
repository, its worktree registration, or a server Git store while dependants
remain. The old `main-teacher-sync` directory is retained as a detached historical
snapshot, including its ignored local assets/environment.

## Explore1000 deployment

Follow the line's accepted deployment document and the Explore1000 skill.
Deploy code by exact commit into a declared checkout under
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime`. If GitHub is inaccessible,
create and verify a Git bundle, transfer it through the working login endpoint,
fetch it into the server repository and update only an idle compatible checkout
with `git merge --ff-only`. A bundle does not carry ignored data or dirty edits.

Record full code SHA, clean/dirty state, any explicitly approved source overlay
and its identities, environment, command, data identities and output directory.
Do not silently replace historical overlays with today's branch HEAD.
Running or queued jobs use immutable code: deploy a new version elsewhere and
retain old code/shared objects until its tasks finish. Login nodes are transfer
and scheduler surfaces, not compute nodes. This policy grants no paid jobs.
Data and scientific outputs remain outside code snapshots and are preserved;
runtime archival requires a verified archive and dependency/active-task audit.

## Recovery

The local migration evidence is under
`E:/ORCHIDEE-MAN-JAX/outputs/maintenance/20261010/local`:
`initial-inventory.json`, verified `pending-and-local-controls.tar.gz`,
`repository-before-migration.bundle` and the final migration/sync record.
The controls archive can contain private local access settings; keep it local.
It does not duplicate Teacher data or ignored output collections, which remain
in place. Restore into a new inspection directory before replacing any live file.

For Git recovery, verify the bundle and clone/fetch it into a new repository;
inspect its saved refs before selecting a commit. To recover pending content,
extract the tar into a new directory and compare its hashes with the inventory.
Retained history tags identify earlier research context; see the daily branch's
`docs/RESEARCH_WORKFLOW.md` and `HISTORY_INDEX.md`. No history was rewritten.
