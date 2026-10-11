# Server runtime deployment entry

This directory registers the server-local runtime and independent Git branch mirror.
`registry.json` binds exact checkout SHAs, source overlay recovery, existing archives,
scientific inputs/results, environments, process evidence and shared Git dependencies.
The workstation repository and its reviewed branches are the development authority.

## Branches and immutable execution

- `main`: local project root and shared maintenance policy.
- `research/daily-coarse-graining`: local daily worktree; existing server execution
  uses `runtime/checkouts/shared-split-width-v1` at
  `c599236bcf57b745e9bafed678298010f4c3b58f` with registered transfer overlays.
- `research/global-throughput`: local global worktree; retained B1024 execution
  checkout `runtime/worktrees/global-throughput-gpu-fixed1024-v1` is at
  `bd9e4c04af68d77e79b984bfc8aa064977f893d3`.

`repository.git` is a self-contained bare branch mirror, with no new working tree
or alternate-object dependency. `registry.json` records expected/observed imported
branch SHAs after bundle verification. The existing root at `688e14c` and all
historical execution trees are preserved; a mirrored ref does not update them.
Use a declared commit and immutable checkout for future approved execution.
Never overwrite an in-use or dirty checkout to make it match a branch tip.

For future bundle updates (from this directory, Git 1.8.3.1 compatible):

```bash
git --git-dir=repository.git bundle verify /absolute/runtime/transfer/BUNDLE
git --git-dir=repository.git fetch /absolute/runtime/transfer/BUNDLE refs/heads/BRANCH:refs/heads/BRANCH
git --git-dir=repository.git rev-parse refs/heads/BRANCH
```

First declare expected SHAs, compare the advertised bundle refs, import without
force or implicit merges, compare observed refs and update the registry. Preserve
the transferred bundle and its checksum receipt. Reuse unchanged executable trees;
do not create one working tree per incoming bundle. Git object maintenance is not
part of ordinary deployment. No jobs or automations are created by mirroring.

## Assets and evidence

All paths below are relative to the canonical project root:

| Asset | Canonical location |
| --- | --- |
| Forcing and static scientific inputs | `runtime/data/` |
| 669-point reference assets | `runtime/assets/teacher_669_a403cd3/landpoints/` |
| Global prepared input | `runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl` |
| Daily prepared data and Teacher | `runtime/outputs/training/target-oriented-prepared-1000-a243029/`, `runtime/outputs/training/pft14-daily-teacher-v6/` |
| Daily arrays, full gradients, HLO and compiled binaries | `runtime/outputs/diagnostics/shared-latent-reader-20261004/` |
| Global completed evidence | `runtime/outputs/research/global_throughput/` |
| Accepted CPU/GPU environments | `.venvs/orcjax_cpu/`, `.venvs/orcjax_gpu/` |
| Transfer overlays | `runtime/transfer/shared-latent-reader-20261004/`, `runtime/transfer/T7-raw-neural-jvp-20261010/`, `runtime/transfer/shared-combined-recurrent-20260926/` |
| CUDA tools/wrappers | `runtime/toolchains/`, registered transfer `nvlink-bin/` |

Scientific failures are retained evidence. B1024 is the best valid global screen;
B2048 OOM and daily state-gradient/raw-neural failures do not establish production
acceptance. Closed budgets and historical commands do not authorize further runs.
Do not run scientific computation on login nodes. Paid submissions require the
project's explicit resource/time/command/directory/cost approval.

## Recovery

The existing 2026-09-22 archive reports remain the completion authority for
45 checkout and 10 historical worktree working-file archives. Restore only a
named member from its recorded archive while preserving its original `.git`:

```bash
RT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime
tar -tzf "$RT/archives/inactive-checkouts-20260922.tar.gz" checkouts/NAME
tar -xzf "$RT/archives/inactive-checkouts-20260922.tar.gz" -C "$RT" checkouts/NAME
```

For historical worktrees use `inactive-worktrees-20260922.tar.gz` and
`worktrees/NAME`. Inspect target usage/contents first; after verified restoration,
remove only that tree's `ARCHIVED_WORKING_FILES.md` marker. Never delete `.git` or
its alternate ancestors. This registry lists every direct/transitive retained dependency.

`maintenance-20261010/daily-source-overlays.tar.gz` is a verified 132-member
snapshot of current transfer source overlays, excluding Python caches. Its JSON
receipt binds every file SHA-256, file mode and symlink target. Originals remain
in place. For non-overwriting recovery, verify the archive SHA-256 from the receipt
and extract into a fresh staging directory before reviewing/replacing missing source:

```bash
RT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime
sha256sum "$RT/deployment/maintenance-20261010/daily-source-overlays.tar.gz"
mkdir "$RT/deployment/recovery-YYYYMMDD"
tar -xzf "$RT/deployment/maintenance-20261010/daily-source-overlays.tar.gz" -C "$RT/deployment/recovery-YYYYMMDD"
```

Preserve scientific data, unique results, prepared-data module/class identities,
environments, compiled-cache identities and ongoing process paths. Runtime file
quota is scarce: prefer compact receipts, consolidated assets and verified archives,
without creating duplicate working trees or deleting evidence to reach a cleanup target.
The quota snapshot is for the shared liwei account, not this project's usage.

## Verified mirror import, 2026-10-10

| Branch | Expected = observed mirror SHA | Local checkout |
| --- | --- | --- |
| `main` | `7cae3e02adad23674767a590fdf4d27f239ce065` | `E:/ORCHIDEE-MAN-JAX` |
| `research/daily-coarse-graining` | `7461227ad38bf8a4fb399c47cc588ce0e1b771c6` | `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/daily-coarse-graining` |
| `research/global-throughput` | `c537f59f33568b6e5960254fe55fb307f82ca503` | `E:/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput` |

The bare mirror is synchronized. Existing scientific execution checkout SHAs and
overlay identities remain those recorded above. This is not a full server source
update or acceptance of a new science/model version. Local multi-line development
rules are in `docs/WORKSPACE_MANAGEMENT.md`; `scripts/dev/workspace_status.py`
is a read-only status entry.

The final v2 bundle only adds the root management entries
`docs/deployment-explore1000.md`, `docs/RESEARCH_WORKFLOW.md` and `runtime/README.md`.
The mirror main ref was updated by non-forced fast-forward fetch; daily/global refs
are unchanged. Both transferred bundles and versioned import receipts are retained.
Original execution and scientific-asset checks are reused for this documentation update.
