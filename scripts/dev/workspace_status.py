"""Read-only Git inventory for the explicitly named workspace root."""

import argparse
import json
from pathlib import Path
import subprocess


def run(path, *args):
    result = subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    if run(root, "rev-parse", "--show-toplevel") is None:
        parser.error("workspace root is not a Git checkout")
    worktrees = []
    listing = run(root, "worktree", "list", "--porcelain")
    for block in listing.split("\n\n"):
        fields = dict(line.split(" ", 1) for line in block.splitlines() if " " in line)
        path = Path(fields["worktree"])
        status = run(path, "status", "--porcelain", "--untracked-files=all")
        worktrees.append({
            "path": str(path),
            "head": fields.get("HEAD"),
            "branch": fields.get("branch", "detached"),
            "upstream": run(path, "rev-parse", "--abbrev-ref", "@{upstream}"),
            "ahead_behind_local_refs": run(path, "rev-list", "--left-right", "--count", "HEAD...@{upstream}"),
            "status": status.splitlines() if status else [],
            "status_available": status is not None,
        })
    print(json.dumps({"workspace_root": str(root), "network_used": False, "worktrees": worktrees}, indent=2))


if __name__ == "__main__":
    main()
