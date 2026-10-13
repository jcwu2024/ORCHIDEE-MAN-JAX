"""Stdlib candidate/source identity before any paid child or JAX import."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from scripts.dev.global_source_compatibility import source_identity


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_identity(args, root=None) -> dict:
    code = Path(root or args.entry.resolve().parents[2]).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    source = source_identity(code)
    reference = Path(args.gpu_reference_dir).resolve() / "stage_B4_group_references.pkl"
    return {"job_id": os.environ.get("SLURM_JOB_ID"), "execution_commit": commit,
            "execution_code_dir": str(code), "prepared_input_path": str(args.prepared_input.resolve()),
            "prepared_input_sha256": sha256_file(args.prepared_input),
            "gpu_reference_path": str(reference), "gpu_reference_sha256": sha256_file(reference),
            "certified_science_sha256": source["science_sha256"],
            "actual_science_source_sha256": source["actual_science_source_sha256"],
            "physical_science_tree_oid": subprocess.check_output(
                ["git", "rev-parse", f"{commit}:jax_orchidee"], cwd=code, text=True).strip(),
            "baseline_ref": "133e35aec7b83b9d79888c46092f64d518e97299"}
