"""Read-only B512 saved-result collector; approved job/allocation are mandatory inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

CODE_SHA = "ad6c0e1faa7e22b590627844e9cb0ad492e9deb3"
ACTUAL_SHA = "b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8"
SCIENCE_SHA = "11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5"
CONTRACT = "GPU-fixed512-repeated-FULL64-B512-only-continuation-v1"
STAGE = "capacity_B512_fixed512_v1"
PRIOR_SHA = "02168b06352f0909c9d9a9a666c6d56edbff0330befcacfc2e8a214ef8159d3f"


def load_prior():
    path = Path(__file__).resolve().parents[1] / "gpu-signature-repair-20261008/collect_continuation_result.py"
    data = path.read_bytes()
    if hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() != PRIOR_SHA:
        raise ValueError("sealed prior collector source changed")
    spec = importlib.util.spec_from_file_location("_b512_readonly_prior", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load_prior()
legacy = prior.legacy
demand, sha = prior.demand, prior.sha


def certificate():
    return {"active": True, "contract": "exact-DIFFUCO-host-mask-cache-repair-v1",
        "source_path": "jax_orchidee/sechiba/diffuco.py",
        "original_file_sha256": "04fc1193e85d2deca0f97cc06008989053ccaf956db3dc9552dd6b9cc2549d22",
        "actual_file_sha256": "42073dd14f427db8ed78803c90cd5a64b677bf822d1edf6b6fbed7aa97166a14",
        "baseline_full_source_sha256": SCIENCE_SHA, "baseline_checkout_bytes_sha256": SCIENCE_SHA,
        "baseline_checkout_representation": "canonical Linux LF source bytes",
        "actual_full_source_sha256": ACTUAL_SHA,
        "compatibility_scope": "exact helper-only immutable host Boolean mask; all other source bytes unchanged",
        "claim": "audited execution-only equivalence; source bytes differ"}


def execution_identity():
    return {"science_sha256": SCIENCE_SHA, "actual_science_source_sha256": ACTUAL_SHA,
            "execution_source_compatibility": certificate()}


def identity_checks(identity, admission):
    legacy.CONTRACT = CONTRACT
    legacy.identity_checks(identity, admission)
    for key, expected in {**execution_identity(), "candidate_shapes": [512], "budget_seconds": 2700,
        "continuation_B128_sha256": prior.B128_SHA,
        "continuation_B256_sha256": "ee7210f2c83ea11346e9aa752320e8760560b2a6c1e377aa2058aa53da16b08c",
        "warm_repeats": 5, "conditional_512": True, "optional_B64_baseline": False,
        "conditional_policy": "B512_only_30min_imported_B256compiler_memory_2GiB_GPU_8GiB_host_slack"}.items():
        demand(type(identity.get(key)) is type(expected) and identity.get(key) == expected,
               f"exact B512 execution identity mismatch: {key}")


def report_checks(report, identity):
    demand(report.get("execution_source_identity") == execution_identity(), "report source certificate/actual SHA mismatch")
    demand(report.get("scientific_acceptance") is False, "report cannot claim scientific acceptance")
    demand(report.get("status") == "completed_exploratory_capacity_screen"
        and report.get("completed_shapes") == [512]
        and report.get("comparison_shapes_including_imported_baseline") == [128, 256, 512],
        "only a completed new B512 stage can be promoted")
    decision = report.get("conditional_512_decision", {})
    demand(decision.get("permitted") is True, "B512 resource admission was not permitted")
    memory = decision.get("compiler_memory_origin", {})
    demand(memory.get("shape") == 256 and memory.get("sha256") == identity["continuation_B256_sha256"],
        "resource guard must preserve sealed B256 memory origin")


def validate_stage(directory, name, identity, record, reference, base, finite_failure, approval, **kwargs):
    demand(name == STAGE and identity.get("contract") == CONTRACT, "only new B512 stage is admissible")
    legacy.JOB, legacy.CODE_SHA, legacy.CONTRACT = approval.job_id, CODE_SHA, CONTRACT
    result = legacy.validate_stage(directory, name, identity, record, reference, base, finite_failure, **kwargs)
    demand(result["strict_numerical_passed"] is True and result["actual_saved_output_reference_passed"] is True,
        "tolerance-failed stage withheld; no valid B512 performance claim")
    runtime = record["execution_origin"]["environment"]
    resources = result["allocation"]
    slurm = resources["Slurm"]
    tres = dict(piece.split("=", 1) for piece in slurm.get("ORCHJAX_SLURM_ALLOC_TRES", "").split(",") if "=" in piece)
    demand(tres.get("node") == "1", "actual one-node allocation required")
    demand(runtime.get("hostname") == resources.get("hostname") == slurm.get("SLURM_JOB_NODELIST") == approval.node,
        "approved actual node mismatch")
    demand(runtime.get("physical_gpu", {}).get("uuid") == approval.gpu_uuid
        and runtime["physical_gpu"].get("driver_version") == approval.driver_version
        and str(slurm.get("SLURM_JOB_GPUS")) == approval.slurm_global_gpu_id,
        "approved actual allocated GPU identity mismatch")
    result["execution_origin"] = record["execution_origin"]
    result["numerical_validation_class"] = "strict_saved_numerical_pass"
    return result


def parent_accounting(text, approval):
    rows = [line.split("|") for line in text.splitlines() if line.strip()]
    parents = [row for row in rows if row[0] == approval.job_id]
    demand(len(parents) == 1, "exact single sacct parent row required")
    row = parents[0]
    demand(len(row) == 9, "exact raw9 sacct JobIDRaw|JobName|Partition|State|ElapsedRaw|AllocCPUS|AllocTRES|NodeList|ExitCode required")
    demand(row[1] == approval.job_name and row[2] == "gnall", "approved parent job name/partition mismatch")
    tres = dict(piece.split("=", 1) for piece in row[6].split(",") if "=" in piece)
    demand(tres.get("node") == "1", "parent actual one-node allocation required")
    normalized = "|".join((row[0], row[3], row[4], row[5], row[6], row[8], row[7]))
    legacy.JOB = approval.job_id
    result = legacy.parent_accounting(normalized)
    demand(result["terminal"], "parent job is unfinished; no result promotion")
    demand(result["node"] == approval.node and result["observed_wall_seconds"] <= 2700,
        "approved parent node/wall envelope mismatch")
    return {**result, "raw_parent_row": row,
        "raw_schema": "JobIDRaw|JobName|Partition|State|ElapsedRaw|AllocCPUS|AllocTRES|NodeList|ExitCode"}


def validate_baseline(origin, identity, directory, destination, size, capacity, jax, reference):
    from scripts.dev.global_server_reference import _evidence_mismatch

    expected = {**identity, "contract": capacity.CONTRACT if size == 128 else capacity.CONTINUATION_CONTRACT,
        "candidate_shapes": [128, 256] if size == 128 else [256, 512],
        "conditional_policy": "core_and_wrapper_5percent_30min_2GiBslack" if size == 128
            else "scale_curve_30min_B256compiler_memory_2GiB_GPU_8GiB_host_slack"}
    importer = capacity.import_b128_baseline if size == 128 else capacity.import_b256_baseline
    _, validated = importer(directory, expected, destination, jax, reference=reference)
    demand(isinstance(origin, dict), f"imported B{size} origin missing")
    for key in validated:
        mismatch = _evidence_mismatch(validated[key], origin.get(key), "record." + key)
        demand(mismatch is None, f"imported B{size} original evidence mismatch: {mismatch}")
    old_job = "15600405" if size == 128 else "15643868"
    demand(str(validated["provenance"]["server_resources"]["Slurm"].get("SLURM_JOB_ID")) == old_job,
        f"B{size} original job cannot be relabeled")
    return validated


def report_stage_checks(report, record):
    from scripts.dev.global_server_reference import _evidence_mismatch

    for actual, expected, path in (
        (record["execution_origin"]["environment"], report.get("environment"), "record.execution_origin.environment"),
        (record["provenance"]["server_resources"], report.get("server_resources"), "record.provenance.server_resources"),
        (record, report.get("stages", {}).get(STAGE), "record"),
        (record["gates"], report.get("gates"), "record.gates")):
        mismatch = _evidence_mismatch(actual, expected, path)
        demand(mismatch is None, f"report/stage recorded evidence mismatch: {mismatch}")
    for boundary in ("lower", "compile"):
        demand(report.get("timing", {}).get(f"capacity_B512_{boundary}_seconds") == record["startup"][f"{boundary}_seconds"],
            f"report/stage {boundary} timing mismatch")


def collect_stage(directory, identity, stages, report, reference, base, finite_failure, approval, **kwargs):
    accepted, withheld = [], []
    try:
        report_checks(report, identity)
        demand(STAGE in stages, "no complete B512 stage; denial/incomplete evidence only")
        report_stage_checks(report, stages[STAGE])
        accepted.append(validate_stage(directory, STAGE, identity, stages[STAGE], reference,
            base, finite_failure, approval, **kwargs))
    except (ValueError, KeyError, TypeError, AssertionError) as error:
        withheld.append({"stage": STAGE, "reason": str(error)})
    return accepted, withheld


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("code-dir", "stage-dir", "admission", "sacct-file", "b128-baseline-dir", "b256-baseline-dir", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    for name in ("job-id", "job-name", "node", "gpu-uuid", "slurm-global-gpu-id", "driver-version"):
        result.add_argument("--" + name, required=True)
    result.add_argument("--approved-cpus", type=int, required=True, choices=[4])
    result.add_argument("--approved-gpus", type=int, required=True, choices=[1])
    result.add_argument("--approved-wall-seconds", type=int, required=True, choices=[2700])
    result.add_argument("--approved-nodes", type=int, required=True, choices=[1])
    result.add_argument("--approved-tasks", type=int, required=True, choices=[1])
    return result


def admit_code_checkout(code_dir):
    demand(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_dir, text=True).strip() == CODE_SHA,
        "collector must import the declared immutable executed checkout")
    demand(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=code_dir, text=True).strip(), "collector code checkout must be clean before any execution imports")


def main():
    args = parser().parse_args()
    demand(args.job_id.isdecimal(), "explicit approved numeric parent JobID required")
    admit_code_checkout(args.code_dir)
    os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true", PYTHONDONTWRITEBYTECODE="1")
    sys.path.insert(0, str(args.code_dir.resolve()))
    from scripts.dev import benchmark_global_batch as base
    from scripts.dev import benchmark_global_gpu_capacity as capacity
    from scripts.dev.global_server_reference import finite_tolerance_failure, load_reference
    from scripts.dev.global_source_compatibility import source_identity

    demand(source_identity(args.code_dir) == execution_identity(), "executed checkout certificate/actual bytes changed")
    admission = json.loads(args.admission.read_text())
    manifest = json.loads((args.stage_dir / "stage_manifest.json").read_text())
    identity = manifest["identity"]
    identity_checks(identity, admission)
    demand(set(manifest["stages"]).issubset({STAGE}), "only new B512 manifest stage allowed; no historical relabel")
    report = json.loads((args.stage_dir / "report.json").read_text())
    reference_origin = admission["FULL64_reference_origin"]
    reference, actual_origin = load_reference(Path(reference_origin["path"]).parent, identity)
    from scripts.dev.global_server_reference import _evidence_mismatch

    admitted_proof = {key: value for key, value in reference_origin.items() if key != "backend_conditions"}
    actual_proof = {key: value for key, value in actual_origin.items() if key != "backend_conditions"}
    demand(_evidence_mismatch(actual_proof, admitted_proof, "record") is None,
        "full original admitted FULL64 reference origin changed")
    demand(actual_origin["backend_conditions"]["source"] == reference_origin["backend_conditions"]["source"],
        "admitted FULL64 source backend evidence changed")
    demand(_evidence_mismatch(actual_origin, report.get("reference_origin"), "record") is None,
        "report original FULL64 reference origin changed")
    with Path(reference_origin["path"]).open("rb") as stream:
        source_envelope = pickle.load(stream)
    sentinel_paths = {}
    for gate in source_envelope.get("record", {}).get("gates", []):
        for allowed in gate.get("allowed_sentinels", []):
            previous = sentinel_paths.setdefault(allowed["path"], allowed["contract"])
            demand(previous == allowed["contract"], "original FULL64 sentinel conflict")
    imported = {}
    for size, directory in ((128, args.b128_baseline_dir), (256, args.b256_baseline_dir)):
        imported[str(size)] = validate_baseline(report.get(f"imported_B{size}_baseline"), identity,
            directory, args.stage_dir, size, capacity, base.jax, reference)
    accounting = parent_accounting(args.sacct_file.read_text(), args)
    guard_memory = report.get("conditional_512_decision", {}).get("compiler_memory_origin", {}).get("actual_analysis")
    demand(_evidence_mismatch(guard_memory, imported["256"]["provenance"]["executable_memory_analysis"], "record.memory") is None,
        "resource guard actual B256 compiler memory evidence changed")
    accepted, withheld = collect_stage(args.stage_dir, identity, manifest["stages"], report,
        reference, base, finite_tolerance_failure, args, sentinel_paths=sentinel_paths)
    result = {"job_id": args.job_id, "code_commit": CODE_SHA, "contract": CONTRACT,
        "scientific_acceptance": False, "execution_source_identity": execution_identity(),
        "valid_exploratory_stages": accepted, "withheld_stages": withheld,
        "imported_historical_baselines": imported, "accounting": accounting,
        "admitted_FULL64_reference_origin": reference_origin, "actual_FULL64_reference_origin": actual_origin,
        "report_status": report.get("status"), "conditional_512_decision": report.get("conditional_512_decision"),
        "incomplete_stages": [{"stage": name, "status": record.get("status"), "reason": record.get("reason")}
            for name, record in manifest["stages"].items() if record.get("status") != "complete"],
        "parent_successfully_completed": accounting["state"] == "COMPLETED" and accounting["exit_code"] == "0:0",
        "comparison_scope": "same512 lanes; historical B128 lazy dispatch differs from later bound AOT; not isolated batch-size effect",
        "K_amortization": "not reported", "collector_execution": "CPU saved-array validation only; no model invocation or compilation",
        "prior_collector_normalized_source_sha256": PRIOR_SHA}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"job_id": args.job_id, "valid_stages": len(accepted), "withheld_stages": len(withheld)}))
    return 1 if withheld else 0


if __name__ == "__main__":
    raise SystemExit(main())
