"""Finite serial Teacher storage maintenance, resumed from migration.sqlite."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from .teacher_assets import close_asset_readers, point_key, safe_path
from .teacher_point_storage import Migration, atomic_json

CHECKOUT = Path(__file__).resolve().parents[2]
REMOTE_HOME = Path("/WORK/liwei_work/jcwu")


def validate_checkout(expected_commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("expected commit must be the full 40-character lowercase Git SHA")
    if Path.cwd().resolve() != CHECKOUT:
        raise ValueError("run from the exact declared code checkout")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != expected_commit:
        raise ValueError("code checkout does not match expected commit")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], text=True).strip():
        raise ValueError("code checkout must be clean before maintenance")
    return actual


def validate_execution(root: Path, work: Path) -> None:
    host = socket.gethostname().split(".")[0].lower()
    if (host.startswith(("cln", "test", "gln")) or
            os.environ.get("SLURM_JOB_PARTITION") != "cnmix" or
            not re.fullmatch(r"\d+", os.environ.get("SLURM_JOB_ID", ""))):
        raise ValueError("production maintenance requires a paid cnmix Slurm CPU allocation")
    if (os.environ.get("SLURM_CPUS_PER_TASK") != "1" or
            os.environ.get("SLURM_NTASKS", "1") != "1"):
        raise ValueError("serial maintenance requires exactly one task and one CPU")
    for path in (root, work, CHECKOUT):
        if not path.is_relative_to(REMOTE_HOME):
            raise ValueError("production paths must remain under /WORK/liwei_work/jcwu/")


def check_dependencies(migration, path: Path) -> None:
    dependency = json.loads(safe_path(path).read_bytes())
    expiry = dependency.get("expires_unix", 0)
    if (dependency.get("dataset_manifest_sha256") != migration.manifest_sha256 or
            dependency.get("active_queued_audit_complete") is not True or
            not isinstance(expiry, (int, float)) or not math.isfinite(expiry) or
            expiry <= time.time()):
        raise ValueError("current manifest-bound dependency audit required for retirement")


def run_batch(migration, *, max_points: int, time_budget_seconds: float,
              points: tuple[str, ...] = (), retire=False, review_token=None,
              dependencies: Path | None = None) -> dict:
    """Finish each admitted point atomically by stage; never start after deadline.

    A point already in progress may finish after the soft budget. Slurm's finite
    wall time bounds the allocation; the durable ledger supports interruption.
    """
    from .teacher_point_verification import prove_point

    started = time.monotonic()
    available = {r["landpoint_id"] for r in migration.manifest["shards"]}
    selected = sorted(set(points) if points else available)
    if set(selected) - available:
        raise ValueError("unknown requested points: " + ", ".join(sorted(set(selected) - available)))
    for point in selected:
        point_key(point)
    result = {"processed": [], "skipped_complete": 0, "stop_reason": "complete"}
    for point in selected:
        record = migration.record(point)
        stage = None if record is None else record["stage"]
        if stage in {"retired", "pinned"} or (stage == "reader_verified" and not retire):
            result["skipped_complete"] += 1
            continue
        if len(result["processed"]) >= max_points:
            result["stop_reason"] = "max_points"
            break
        if time.monotonic() - started >= time_budget_seconds:
            result["stop_reason"] = "time_budget"
            break
        if retire:
            check_dependencies(migration, dependencies)
        if stage not in {"reader_verified", "retiring"}:
            migration.pack(point)
            proof = prove_point(migration, point, scan_full_index=False)
            # One rolling proof file is immediately embedded into the durable
            # point record; later points do not depend on this file's contents.
            proof_path = migration.work / "batch-reader-proof.json"
            atomic_json(proof_path, proof)
            migration.accept_reader_proof(point, proof_path)
        elif stage == "retiring" and not retire:
            raise ValueError("interrupted retirement requires explicit retirement arguments")
        if retire:
            # The audit may expire during the full reader/restore proof.
            check_dependencies(migration, dependencies)
            migration.retire(point, review_token=review_token, dependencies=dependencies)
        result["processed"].append(point)
        print(json.dumps({"point": point, "stage": migration.record(point)["stage"],
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--max-points", type=int, required=True)
    parser.add_argument("--time-budget-seconds", type=float, required=True)
    parser.add_argument("--point", action="append", default=[])
    parser.add_argument("--retire", action="store_true")
    parser.add_argument("--review-token")
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_points < 1 or not math.isfinite(args.time_budget_seconds) or args.time_budget_seconds <= 0:
        parser.error("max-points and time-budget-seconds must be finite and positive")
    if args.retire != bool(args.review_token and args.dependencies):
        parser.error("retirement requires --retire --review-token --dependencies together")
    if not args.retire and (args.review_token or args.dependencies):
        parser.error("retirement arguments require --retire")
    commit = validate_checkout(args.expected_commit)
    root, work = safe_path(args.dataset_root), safe_path(args.work_root)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "checkout": str(CHECKOUT), "commit": commit,
                          "dataset_root": str(root), "work_root": str(work),
                          "max_points": args.max_points, "time_budget_seconds": args.time_budget_seconds,
                          "points": args.point, "retire": args.retire}, sort_keys=True))
        return 0
    validate_execution(root, work)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        with Migration(root, work) as migration:
            result = run_batch(migration, max_points=args.max_points,
                               time_budget_seconds=args.time_budget_seconds,
                               points=tuple(args.point), retire=args.retire,
                               review_token=args.review_token, dependencies=args.dependencies)
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        close_asset_readers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
