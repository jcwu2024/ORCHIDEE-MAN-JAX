"""Seal only the reviewed launcher/startup-test delta; never run the model or submit."""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "cpu-scalar-pool-preparation-20261014"
BASE = "50864f065363f6cd0de685c18d9c257723bf2838"
REF = "refs/heads/codex/global-cpu-scalar-pool-v2"
PROJECT = "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX"
CODE = PROJECT + "/runtime/worktrees/global-throughput-cpu-scalar-pool-v2"
REMOTE = PROJECT + "/runtime/outputs/research/global_throughput/" + HERE.name
PATHS = ["scripts/hpc/slurm_global_server_cpu_4core.sh",
         "tests/unit/test_global_server_cpu_startup.py"]


def git(*args, env=None, data=None):
    return subprocess.check_output(["git", *args], cwd=ROOT, env=env, input=data)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    (HERE / name).write_bytes(data)
    return {"name": name, "sha256": sha(data), "bytes": len(data)}


def main():
    global HERE, REF, CODE, REMOTE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=['v2', 'v3'], default='v2')
    args = parser.parse_args()
    if args.version == 'v3':
        HERE = HERE / 'v3'
        HERE.mkdir(exist_ok=True)
        REF = 'refs/heads/codex/global-cpu-scalar-pool-v3'
        CODE = PROJECT + '/runtime/worktrees/global-throughput-cpu-scalar-pool-v3'
        REMOTE += '/v3'
    source = git("rev-parse", "HEAD").decode().strip()
    assert git("branch", "--show-current").decode().strip() == "research/global-throughput"
    assert git("rev-parse", "HEAD:jax_orchidee").decode().strip() == "62e90547ee8eadab387713397bb111c399c81b6c"
    # The source stage must already contain these exact reviewed bytes.
    for path in PATHS:
        assert git("diff", "HEAD", "--", path) == b""
        assert git("show", source + ":" + path) == (ROOT / path).read_bytes()
    existing = subprocess.run(["git", "rev-parse", "--verify", REF], cwd=ROOT,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if existing.returncode == 0:
        pin = existing.stdout.decode().strip()
        assert git("rev-parse", pin + "^").decode().strip() == BASE
        for path in PATHS:
            assert git("rev-parse", pin + ":" + path) == git("rev-parse", source + ":" + path)
    else:
        fd, index_path = tempfile.mkstemp(prefix="telemetry-git-index-", dir=HERE)
        os.close(fd)
        os.unlink(index_path)
        env = dict(os.environ, GIT_INDEX_FILE=index_path)
        try:
            git("read-tree", BASE, env=env)
            for path in PATHS:
                fields = git("ls-tree", source, "--", path).decode().split()
                git("update-index", "--add", "--cacheinfo", fields[0], fields[2], path, env=env)
            tree = git("write-tree", env=env).decode().strip()
            message = "Keep optional CPU startup telemetry failures nonfatal\n\nOnly launcher telemetry and actual startup regression tests change from scalar-pool-v1.\n"
            pin = git("commit-tree", tree, "-p", BASE, data=message.encode()).decode().strip()
            git("update-ref", REF, pin, "0" * 40)
        finally:
            Path(index_path).unlink(missing_ok=True)
    changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", BASE, pin).decode().splitlines()
    assert sorted(changed) == sorted(PATHS), changed
    for scope in ("jax_orchidee", "configs", "scripts/dev"):
        assert git("rev-parse", BASE + ":" + scope) == git("rev-parse", pin + ":" + scope)
    assert git("rev-parse", pin + ":jax_orchidee").decode().strip() == "75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba"

    old_bytes = (OLD / "scalar-admission.json").read_bytes()
    assert sha(old_bytes) == "fafd685bb54c7bb366f0d10d38cba31957ba754b9a5bcd8801e75abfd7907883"
    old_admission = json.loads(old_bytes)
    admission = copy.deepcopy(old_admission)
    admission["candidate_identity"].update(execution_commit=pin, execution_code_dir=CODE)
    restored = copy.deepcopy(admission)
    for key in ("execution_commit", "execution_code_dir"):
        restored["candidate_identity"][key] = old_admission["candidate_identity"][key]
    assert restored == old_admission
    admission_file = save("scalar-admission.json", admission)
    proposal = (OLD / "PROPOSED_SUBMISSION.sh").read_text(encoding="utf-8").replace(BASE, pin)
    proposal = proposal.replace("global-throughput-cpu-scalar-pool-v1", CODE.rsplit('/', 1)[-1])
    proposal = proposal.replace(OLD.name, REMOTE.split('/global_throughput/', 1)[1])
    proposal = proposal.replace(sha(old_bytes), admission_file["sha256"])
    proposal_bytes = proposal.encode("utf-8")
    (HERE / "PROPOSED_SUBMISSION.sh").write_bytes(proposal_bytes)

    bundle = HERE / ("cpu-scalar-pool-" + pin[:12] + ".bundle")
    if not bundle.exists():
        git("bundle", "create", str(bundle), REF, "^" + BASE)
    verification = subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=ROOT,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    (HERE / "bundle-verify.txt").write_bytes(verification.stdout)
    old_pin = json.loads((OLD / "execution-code-pin.json").read_bytes())
    record = {
        "schema": "independent-scalar-pool-telemetry-fix-candidate-v1",
        "status": "prepared_no_paid_authorization",
        "source_research_commit": source,
        "base_execution_commit": BASE,
        "execution_code_commit": pin,
        "execution_ref": REF.removeprefix("refs/heads/"),
        "target_checkout": CODE,
        "scientific_source": old_pin["scientific_source"],
        "research_science_tree_oid": old_pin["research_science_tree_oid"],
        "execution_changed_paths": changed,
        "changed_blob_sha256": {path: sha(git("show", pin + ":" + path)) for path in PATHS},
        "unchanged_tree_oids": {scope: git("rev-parse", pin + ":" + scope).decode().strip()
                                for scope in ("jax_orchidee", "configs", "scripts/dev")},
        "scalar_admission": dict(admission_file, remote_path=REMOTE + "/scalar-admission.json",
            source_admission_sha256=sha(old_bytes),
            source_audit_receipt_sha256=old_pin["scalar_admission"]["source_audit_receipt_sha256"],
            candidate_identity=admission["candidate_identity"],
            changed_fields=["candidate_identity.execution_commit", "candidate_identity.execution_code_dir"],
            all_other_content_unchanged=True, new_scalar_generation=False),
        "bundle": {"name": bundle.name, "sha256": sha(bundle.read_bytes()), "bytes": bundle.stat().st_size,
                   "prerequisite": BASE, "verify": "passed"},
        "pilot": old_pin["pilot"],
        "fixed_contract": old_pin["fixed_contract"],
        "proposed_submission": {"name": "PROPOSED_SUBMISSION.sh", "sha256": sha(proposal_bytes),
            "cwd": CODE, "command": "bash " + REMOTE + "/PROPOSED_SUBMISSION.sh",
            "submissions_allowed_by_this_preparation": 0},
        "reused_evidence": {
            "unit_tests_passed": 81,
            "unit_tests_record": "docs/research/global_throughput/CPU_SCALAR_POOL_VALIDATION.md",
            "actual_reference_loading_job": "15866448",
            "actual_reference_loading_status": "COMPLETED 0:0; cntest/ibtest01; 1 CPU; 63 seconds; GPU760/scalar760",
            "first_failed_paid_job": "15866488; FAILED 1:0; 1 second; no physical dispatch",
            "rerun_science_or_array_audit": False},
        "physical_lower_compile_run": False,
        "paid_submitted": False,
        "run_root_pattern": old_pin["run_root_pattern"],
    }
    save("execution-code-pin.json", record)
    (HERE / "execution-harness.patch").write_bytes(git("diff", "--binary", BASE, pin))
    print(json.dumps({"execution_commit": pin, "admission": admission_file,
                      "changed_paths": changed, "paid_submitted": False}))


if __name__ == "__main__":
    main()
