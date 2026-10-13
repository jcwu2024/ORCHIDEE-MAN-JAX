"""Build an immutable CPU48 execution ref without checking out old science.

Only explicit harness/test paths come from a reviewed research commit. Every
jax_orchidee blob remains exactly the accepted historical base's blob. This
stdlib/Git preparation never imports or runs the physical model or deploys code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASE = "03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8"
SCIENCE = "11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5"
BASELINE = "133e35aec7b83b9d79888c46092f64d518e97299"
REFERENCE = "a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6"
TARGET = "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v4"
EXECUTION_REF = "codex/global-cpu48-b64-v4"
PIN_SCHEMA = "cpu48-b64-immutable-execution-v4"
PHYSICAL_TREE = "75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba"
DEFAULT_BUDGET_SECONDS = 3600
PROJECT = "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX"
PREPARED_INPUT = PROJECT + "/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl"
PREPARED_INPUT_SHA = "be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7"
GPU_REFERENCE = PROJECT + "/runtime/outputs/research/global_throughput/gpu-round4/model/stage_B4_group_references.pkl"
SCALAR_DIRECTORY = PROJECT + "/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar"
SCALAR_CHECKPOINT_SHA = "1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa"
ADMISSION_REMOTE = PROJECT + "/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/scalar-audit/scalar-admission.json"


def git(repository, *arguments, env=None, data=None):
    result = subprocess.run(["git", *arguments], cwd=repository, env=env, input=data,
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def revision(repository, value):
    return git(repository, "rev-parse", "--verify", f"{value}^{{commit}}").decode().strip()


def paths_at(repository, value, prefix):
    return git(repository, "ls-tree", "-r", "--name-only", value, "--", prefix).decode().splitlines()


def science_identity(repository, value):
    # This certificate is stdlib-only; no jax_orchidee or JAX import occurs.
    from scripts.dev.global_source_compatibility import DIFFUCO_PATH, audit_diffuco_bytes

    actual, certified = hashlib.sha256(), hashlib.sha256()
    paths = paths_at(repository, value, "jax_orchidee")
    if not paths:
        raise ValueError("execution base has no scientific source tree")
    for path in paths:
        content = git(repository, "show", f"{value}:{path}")
        original = audit_diffuco_bytes(content)[0] if path == DIFFUCO_PATH else content
        actual.update(path.encode() + b"\0" + content + b"\0")
        certified.update(path.encode() + b"\0" + original + b"\0")
    return {"science_sha256": certified.hexdigest(), "actual_science_source_sha256": actual.hexdigest(),
            "science_file_count": len(paths),
            "science_tree_oid": git(repository, "rev-parse", f"{value}:jax_orchidee").decode().strip()}


def validate_paths(paths):
    selected = sorted(set(paths))
    if not selected:
        raise ValueError("explicit harness paths are required")
    for path in selected:
        normalized = PurePosixPath(path)
        if (str(normalized) != path or normalized.is_absolute() or ".." in normalized.parts
                or "\\" in path or not path.startswith(("scripts/dev/", "scripts/hpc/", "tests/unit/"))
                or normalized.suffix not in {".py", ".sh"}):
            raise ValueError(f"outside execution harness/test allowlist: {path}")
    return selected


def build(repository, output, *, source, base, ref, harness_paths, expected_science=SCIENCE,
          budget_seconds=DEFAULT_BUDGET_SECONDS, scalar_audit_receipt=None,
          expected_scalar_audit_sha256=None):
    repository, output = Path(repository).resolve(), Path(output).resolve()
    if not output.is_relative_to(repository / "outputs"):
        raise ValueError("preparation output must stay under this checkout's outputs")
    if not ref.startswith("codex/"):
        raise ValueError("new execution ref must use codex/ prefix")
    full_ref = "refs/heads/" + ref
    git(repository, "check-ref-format", full_ref)
    selected = validate_paths(harness_paths)
    if not isinstance(budget_seconds, int) or isinstance(budget_seconds, bool) or not 1800 <= budget_seconds <= 5400:
        raise ValueError("CPU48 batch-only budget must be an integer from 1800 to 5400 seconds")
    source, base = revision(repository, source), revision(repository, base)
    if git(repository, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("research source checkout must be clean before version preparation")
    before = revision(repository, "HEAD")
    branch = git(repository, "symbolic-ref", "--short", "HEAD").decode().strip()
    identity = science_identity(repository, base)
    if identity["science_sha256"] != expected_science:
        raise ValueError("base scientific identity differs from the accepted source")
    if expected_science == SCIENCE and (
            base != revision(repository, BASE) or identity["science_tree_oid"] != PHYSICAL_TREE
            or identity["science_file_count"] != 104):
        raise ValueError("production package must use the reviewed v3 execution base and its exact physical tree")
    if expected_science == SCIENCE and scalar_audit_receipt is None:
        raise ValueError("production batch-only package requires an independently audited scalar receipt")
    if scalar_audit_receipt is not None:
        scalar_audit_receipt = Path(scalar_audit_receipt).resolve()
        if not scalar_audit_receipt.is_relative_to(output):
            raise ValueError("independent scalar audit receipt must stay inside the preparation output")
        audit_bytes = scalar_audit_receipt.read_bytes()
        if hashlib.sha256(audit_bytes).hexdigest() != expected_scalar_audit_sha256:
            raise ValueError("independent scalar audit receipt checksum mismatch")
    elif expected_scalar_audit_sha256 is not None:
        raise ValueError("independent scalar audit checksum requires a receipt")
    for path in selected:
        git(repository, "cat-file", "-e", f"{source}:{path}")
    patch = git(repository, "diff", "--binary", "--no-ext-diff", base, source, "--", *selected)
    if not patch:
        raise ValueError("no execution harness changes to package")
    # Temporary index isolates all old-tree operations from the working tree.
    with tempfile.TemporaryDirectory(prefix="orchjax-cpu48-index-") as temporary:
        index_env = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}
        git(repository, "read-tree", base, env=index_env)
        git(repository, "apply", "--cached", "--whitespace=error", env=index_env, data=patch)
        tree = git(repository, "write-tree", env=index_env).decode().strip()
    changed = git(repository, "diff", "--name-only", base, tree).decode().splitlines()
    if not set(changed).issubset(selected):
        raise ValueError("execution tree changed outside the declared harness allowlist")
    if git(repository, "diff", "--name-only", base, tree, "--", "jax_orchidee").strip():
        raise ValueError("execution preparation modified scientific source")
    execution_tree = git(repository, "rev-parse", f"{tree}:jax_orchidee").decode().strip()
    if execution_tree != identity["science_tree_oid"]:
        raise ValueError("scientific source subtree does not match the historical base")
    message = ("Prepare CPU48 FULL64 batch-only harness on accepted historical science\n\n"
               f"Research source: {source}\nAccepted execution base: {base}\n"
               f"Certified scientific identity: {identity['science_sha256']}\n"
               "All physical source blobs are unchanged from the accepted execution base.\n"
               "No deployment or paid execution is authorized by this commit.\n")
    exists = subprocess.run(["git", "show-ref", "--verify", "--quiet", full_ref], cwd=repository).returncode == 0
    if exists:
        execution = revision(repository, full_ref)
        if (git(repository, "rev-parse", f"{execution}^{{tree}}").decode().strip() != tree
                or git(repository, "rev-parse", f"{execution}^").decode().strip() != base):
            raise ValueError("existing execution ref differs; preserve it and select a new version")
    else:
        execution = git(repository, "commit-tree", tree, "-p", base, data=message.encode()).decode().strip()
    output.mkdir(parents=True, exist_ok=True)
    scalar_reference = None
    if scalar_audit_receipt is not None:
        from scripts.dev.global_server_reference import bind_cpu_reference_admission

        candidate = {"job_id": None, "execution_commit": execution, "execution_code_dir": TARGET,
            "prepared_input_path": PREPARED_INPUT, "prepared_input_sha256": PREPARED_INPUT_SHA,
            "gpu_reference_path": GPU_REFERENCE, "gpu_reference_sha256": REFERENCE,
            "certified_science_sha256": identity["science_sha256"],
            "actual_science_source_sha256": identity["actual_science_source_sha256"],
            "physical_science_tree_oid": execution_tree, "baseline_ref": BASELINE}
        admission_path = output / "scalar-audit" / "scalar-admission.json"
        if admission_path.exists():
            with tempfile.TemporaryDirectory(prefix="orchjax-cpu48-admission-") as temporary:
                checked_admission = Path(temporary) / "scalar-admission.json"
                bind_cpu_reference_admission(scalar_audit_receipt, checked_admission, candidate,
                    expected_receipt_sha256=expected_scalar_audit_sha256)
                if admission_path.read_bytes() != checked_admission.read_bytes():
                    raise ValueError("existing scalar admission differs; preserve it and select a new version")
        else:
            bind_cpu_reference_admission(scalar_audit_receipt, admission_path, candidate,
                expected_receipt_sha256=expected_scalar_audit_sha256)
        admission_bytes = admission_path.read_bytes()
        admission = json.loads(admission_bytes)
        if expected_science == SCIENCE and (
                str(admission["scalar_source"]["job_id"]) != "15820467"
                or admission["scalar_source"]["execution_commit"] != BASE
                or admission["scalar_source"]["stage_sha256"] != SCALAR_CHECKPOINT_SHA
                or admission.get("candidate_identity") != candidate):
            raise ValueError("scalar admission differs from the reviewed source or exact new candidate")
        scalar_reference = {"directory": SCALAR_DIRECTORY,
            "source_job_id": admission["scalar_source"]["job_id"],
            "source_execution_commit": admission["scalar_source"]["execution_commit"],
            "source_checkpoint_sha256": admission["scalar_source"]["stage_sha256"],
            "audit_receipt": {"name": scalar_audit_receipt.relative_to(output).as_posix(),
                "sha256": expected_scalar_audit_sha256, "bytes": len(audit_bytes)},
            "admission_record": {"name": admission_path.relative_to(output).as_posix(),
                "remote_path": ADMISSION_REMOTE,
                "sha256": hashlib.sha256(admission_bytes).hexdigest(), "bytes": len(admission_bytes)},
            "candidate_identity": candidate, "new_scalar_generation": False,
            "scientific_acceptance": False}
    if not exists:
        git(repository, "update-ref", full_ref, execution, "0" * 40)
    patch_path = output / "execution-harness.patch"
    patch_path.write_bytes(patch)
    bundle = output / f"cpu48-b64-{execution[:12]}.bundle"
    git(repository, "bundle", "create", str(bundle), full_ref, "^" + base)
    git(repository, "bundle", "verify", str(bundle))
    manifest = {
        "schema": PIN_SCHEMA, "status": "local_preparation_only",
        "source_research_commit": source, "base_execution_commit": base,
        "execution_code_commit": execution, "execution_ref": ref, "target_checkout": TARGET,
        "scientific_source": {**identity, "execution_science_tree_oid": execution_tree,
                              "every_scientific_blob_identical_to_base": True,
                              "research_science_tree_oid": git(repository, "rev-parse", f"{source}:jax_orchidee").decode().strip()},
        "harness_paths": selected, "execution_changed_paths": changed,
        "harness_blob_sha256": {path: hashlib.sha256(git(repository, "show", f"{execution}:{path}")).hexdigest()
                                for path in selected},
        "patch": {"name": patch_path.name, "sha256": hashlib.sha256(patch).hexdigest(), "bytes": len(patch)},
        "bundle": {"name": bundle.name, "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                   "bytes": bundle.stat().st_size, "prerequisite": base, "local_git_bundle_verify": "passed"},
        "fixed_contract": {"baseline_ref": BASELINE, "GPU_FULL64_reference_sha256": REFERENCE,
                           "prepared_input_sha256": PREPARED_INPUT_SHA,
                           "unique_points": 64, "point_days": 896, "blocks": [[2, 8], [9, 15]],
                           "half_hours_per_day": 48, "native_output_leaves": 760,
                           "atol": 1e-8, "rtol": 1e-10},
        "execution_route": "batch-only",
        "external_scalar_reference": scalar_reference,
        "allocation_plan": {"partition": "cnall", "nodes": 1, "tasks": 1, "allocated_CPUs": 48,
                            "GPUs": 0, "wall_seconds": budget_seconds,
                            "primary_worst_case_CNY": 48 * budget_seconds / 3600 * 0.05,
                            "conservative_worst_case_CNY": 48 * budget_seconds / 3600 * 0.07,
                            "whole_parent_allocation_billed": True,
                            "new_scalar_generation": False,
                            "billing_verified": False},
        "working_tree_preserved": {"HEAD": before, "branch": branch},
        "remote_deployment_or_submission": False, "model_or_GPU_execution": False,
    }
    if revision(repository, "HEAD") != before or git(repository, "symbolic-ref", "--short", "HEAD").decode().strip() != branch:
        raise ValueError("version preparation unexpectedly changed the research checkout")
    (output / "execution-code-pin.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--base-ref", default=BASE)
    parser.add_argument("--execution-ref", default=EXECUTION_REF)
    parser.add_argument("--harness-path", action="append", required=True)
    parser.add_argument("--budget-seconds", type=int, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--scalar-audit-receipt", type=Path, required=True)
    parser.add_argument("--expected-scalar-audit-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repository = ROOT
    if Path.cwd().resolve() != repository:
        raise ValueError("run preparation from the declared global-throughput checkout")
    print(json.dumps(build(repository, args.output_dir, source=args.source_ref, base=args.base_ref,
                           ref=args.execution_ref, harness_paths=args.harness_path,
                           budget_seconds=args.budget_seconds,
                           scalar_audit_receipt=args.scalar_audit_receipt,
                           expected_scalar_audit_sha256=args.expected_scalar_audit_sha256), indent=2))


if __name__ == "__main__":
    main()
