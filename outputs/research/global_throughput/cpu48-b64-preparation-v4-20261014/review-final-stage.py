"""Review compact staged publication artifacts without model execution."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PREPARATION = OUT.relative_to(ROOT).as_posix() + "/"
RECEIPT = PREPARATION + "final-staged-review.json"
DOCS = {
    "docs/research/global_throughput/README.md",
    "docs/research/global_throughput/SERVER_BENCHMARK.md",
    "docs/research/global_throughput/SUPERVISOR_HANDOFF.md",
    "docs/research/global_throughput/NEW_PROJECT_PROMPT.md",
}


def git(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, check=True
    ).stdout


def main() -> None:
    git("diff", "--cached", "--check")
    staged = git("diff", "--cached", "--name-only").decode().splitlines()
    paths = [path for path in staged if path != RECEIPT]
    assert paths
    pattern = re.compile(
        rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"
        rb"|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
        rb"|sk-[A-Za-z0-9_-]{30,}"
    )
    records = []
    for path in paths:
        assert path in DOCS or path.startswith(PREPARATION), path
        assert Path(path).suffix not in {".bundle", ".pkl", ".pickle", ".npz", ".npy"}, path
        assert not any(part in path for part in ("-tests-tmp/", "/tmp/", "/repository/")), path
        blob = git("show", ":" + path)
        assert len(blob) < 400_000, path
        assert not pattern.search(blob), "obvious credential pattern in " + path
        raw = (ROOT / path).read_bytes()
        exact = blob == raw
        if not exact:
            assert path.endswith(".md") and blob == raw.replace(b"\r\n", b"\n"), path
        records.append({"path": path, "bytes": len(blob),
                        "git_blob_sha256": hashlib.sha256(blob).hexdigest(),
                        "disk_sha256": hashlib.sha256(raw).hexdigest(),
                        "byte_exact": exact})
    receipt = {
        "schema": "cpu48-v4-compact-stage-review-v1",
        "status": "passed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "conda run -n ORCJAX python -B " + Path(__file__).relative_to(ROOT).as_posix(),
        "scope": "explicit staged compact v4 artifacts and four current entry documents",
        "receipt_itself_excluded_from_scope": True,
        "staged_diff_check": "passed",
        "obvious_credentials_found": False,
        "bundles_pickles_or_fixture_trees_staged": False,
        "files_reviewed": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "byte_exact_artifacts": sum(record["byte_exact"] for record in records),
        "records": records,
        "model_or_jax_execution": False,
    }
    (ROOT / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
