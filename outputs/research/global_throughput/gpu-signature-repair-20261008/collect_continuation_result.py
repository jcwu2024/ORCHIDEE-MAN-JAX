"""Read-only saved-result validation for approved continuationjob15643868.

Reuse the SHA-pinned old collector's scientific gates without modifying it.
Run real saved-array collection on canonicaltest02; never execute model code.
"""

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

JOB = "15643868"
CODE_SHA = "12b495b9425de3ad1dc16efa3a5a0e69d7e89222"
CONTRACT = "GPU-fixed512-repeated-FULL64-B256-B512-continuation-v1"
OLD_JOB = "15600405"
OLD_CODE_SHA = "a51efcdabf7ff2e9f97a463a6f72e9f6a2cf478e"
OLD_CONTRACT = "GPU-fixed512-repeated-FULL64-capacity-v1"
B128_SHA = "e91db1391ccb5f525f9ec29f1cba24235c4ce9cd4153b9297fe2ed014d0c9cb5"
LEGACY_SHA = "56ac32805e39005a4bc9401036e2fb2d45f888acd9eb3a7d317af8eb9ccef2f9"
PROJECT = Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX")
BASELINE_DIR = PROJECT / "runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity"
BASELINE_FILE = BASELINE_DIR / "stage_capacity_B128_fixed512_v1.pkl"
NODE = "ibc13b03n02"
GPU_UUID = "GPU-949cb714-8c53-353c-1f6a-1832c629230e"
STAGE_NAMES = {"capacity_B256_fixed512_v1", "capacity_B512_fixed512_v1"}


def load_legacy():
    local = Path(__file__).resolve().parents[1] / "gpu-sweetspot-preparation-20261006/collect_capacity_result.py"
    default = (
        local
        if local.is_file()
        else PROJECT
        / "runtime/outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/collect_capacity_result.py"
    )
    path = Path(os.environ.get("ORCHJAX_CAPACITY_COLLECTOR_SOURCE", str(default))).resolve()
    data = path.read_bytes()
    if hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() != LEGACY_SHA:
        raise ValueError("legacy collector source SHA changed")
    spec = importlib.util.spec_from_file_location("_continuation_legacy_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.JOB, module.CODE_SHA, module.CONTRACT = JOB, CODE_SHA, CONTRACT
    return module, path


legacy, legacy_path = load_legacy()
demand, sha, samples = legacy.demand, legacy.sha, legacy.samples
GATE = legacy.GATE


def identity_checks(identity, admission):
    legacy.identity_checks(identity, admission)
    demand(
        identity.get("candidate_shapes") == [256, 512]
        and all(type(value) is int for value in identity["candidate_shapes"]),
        "exact B256/B512 candidate order required",
    )
    demand(identity.get("continuation_B128_sha256") == B128_SHA, "sealed imported B128 identity required")
    demand(
        identity.get("conditional_512") is True
        and identity.get("optional_B64_baseline") is False
        and identity.get("conditional_policy") == "scale_curve_30min_B256compiler_memory_2GiB_GPU_8GiB_host_slack",
        "continuation scale-curve resource policy required",
    )
    demand(identity.get("warm_repeats") == 5, "fixed five synchronized repeats required")


def validate_stage(directory, name, identity, manifest_record, reference, base, finite_failure, **kwargs):
    demand(name in STAGE_NAMES, "continuation allows only new B256/B512 stages; B128 is an imported baseline")
    demand(identity.get("contract") == CONTRACT, "new stage continuation contract required")
    result = legacy.validate_stage(
        directory, name, identity, manifest_record, reference, base, finite_failure, **kwargs
    )
    runtime = manifest_record["execution_origin"]["environment"]
    resources = result["allocation"]
    for container, key in ((runtime, "hostname"), (resources, "hostname"), (resources["Slurm"], "SLURM_JOB_NODELIST")):
        if container.get(key) is not None:
            demand(container[key] == NODE, f"approved continuation node mismatch: {key}")
    physical = runtime.get("physical_gpu", {})
    if physical.get("uuid") is not None:
        demand(
            physical["uuid"] == GPU_UUID and physical.get("driver_version") == "525.85.12",
            "approved allocated GPU UUID/driver mismatch",
        )
    if resources["Slurm"].get("SLURM_JOB_GPUS") is not None:
        demand(str(resources["Slurm"]["SLURM_JOB_GPUS"]) == "4", "approved Slurm global GPU identity mismatch")
    result["numerical_validation_class"] = (
        "strict_saved_numerical_pass"
        if result["strict_numerical_passed"] and result["actual_saved_output_reference_passed"]
        else "finite_only_exploratory_reference_failure"
    )
    return result


def parent_accounting(text):
    result = legacy.parent_accounting(text)
    if result["terminal"]:
        demand(result.get("node") == NODE, "approved parent terminal node mismatch")
    return result


def validate_imported_baseline(origin, identity, capacity, jax):
    """Revalidate sealed origin; retain old execution provenance, never relabel."""
    demand(
        isinstance(origin, dict)
        and origin.get("path") == str(BASELINE_FILE)
        and origin.get("manifest") == str(BASELINE_DIR / "stage_manifest.json"),
        "exact old B128 source paths required",
    )
    demand(
        origin.get("sha256") == B128_SHA
        and origin.get("bytes") == 220429366
        and origin.get("source_stage") == "capacity_B128_fixed512_v1"
        and origin.get("role")
        == "imported historical baseline; no new B128 measurement, executable, or destination stage",
        "exact sealed B128 imported role/SHA/bytes required",
    )
    expected_identity = {
        **identity,
        "contract": OLD_CONTRACT,
        "candidate_shapes": [128, 256],
        "conditional_policy": "core_and_wrapper_5percent_30min_2GiBslack",
    }
    expected_identity.pop("continuation_B128_sha256", None)
    # Existing source-backed importer verifies fullSHA, envelope/manifest,760x512
    # native schema,515 strict gates, rawsamples, and exact recorded provenance.
    _, validated = capacity.import_b128_baseline(
        BASELINE_DIR,
        expected_identity,
        PROJECT / "runtime/outputs/research/global_throughput/gpu-signature-repair-20261008",
        jax,
    )
    from scripts.dev.global_server_reference import _evidence_mismatch

    for key in (
        "path",
        "sha256",
        "bytes",
        "manifest",
        "manifest_sha256",
        "source_stage",
        "source_identity",
        "execution_origin",
        "timing",
        "provenance",
        "role",
    ):
        mismatch = _evidence_mismatch(validated.get(key), origin.get(key), "record." + key)
        demand(mismatch is None, f"imported B128 report/source evidence mismatch: {mismatch}")
    runtime = origin["execution_origin"]["environment"]
    slurm = origin["provenance"]["server_resources"]["Slurm"]
    demand(
        runtime.get("git_head") == OLD_CODE_SHA and str(slurm.get("SLURM_JOB_ID")) == OLD_JOB,
        "B128 must retain its old job/code execution origin",
    )
    return {
        **origin,
        "validation": "sealed historical SHA/identity/provenance/gates/760nativeleaves verified; no new B128 stage",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--sacct-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    demand(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.code_dir, text=True).strip() == CODE_SHA,
        "collector must import the actual immutable approved continuation checkout",
    )
    os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true")
    sys.path.insert(0, str(args.code_dir.resolve()))
    from scripts.dev import benchmark_global_batch as base
    from scripts.dev import benchmark_global_gpu_capacity as capacity
    from scripts.dev.global_server_reference import finite_tolerance_failure

    admission = json.loads(args.admission.read_text())
    manifest = json.loads((args.stage_dir / "stage_manifest.json").read_text())
    identity_checks(manifest["identity"], admission)
    demand(set(manifest["stages"]).issubset(STAGE_NAMES), "unexpected stage in continuation manifest")
    report_path = args.stage_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    imported = validate_imported_baseline(
        report.get("imported_B128_baseline"), manifest["identity"], capacity, base.jax
    )
    origin = admission["FULL64_reference_origin"]
    reference_path = Path(origin["path"])
    demand(sha(reference_path) == origin["sha256"], "original admitted FULL64 B4 referenceSHA changed")
    with reference_path.open("rb") as stream:
        reference_envelope = pickle.load(stream)
    demand(
        reference_envelope["identity"] == origin["source_identity"] and reference_envelope["status"] == "complete",
        "original complete B4 referenceidentity changed",
    )
    reference = reference_envelope["value"]
    reference = reference["outputs"] if isinstance(reference, dict) and "outputs" in reference else reference
    sentinel_paths = {}
    for gate in reference_envelope.get("record", {}).get("gates", []):
        for allowed in gate.get("allowed_sentinels", []):
            prior = sentinel_paths.setdefault(allowed["path"], allowed["contract"])
            demand(prior == allowed["contract"], "original B4 sentinel contract conflict")
    accepted, withheld, pending = [], [], []
    for name, record in manifest["stages"].items():
        if record.get("status") != "complete":
            pending.append({"stage": name, "status": record.get("status"), "reason": record.get("reason")})
            continue
        try:
            accepted.append(
                validate_stage(
                    args.stage_dir,
                    name,
                    manifest["identity"],
                    record,
                    reference,
                    base,
                    finite_tolerance_failure,
                    sentinel_paths=sentinel_paths,
                )
            )
        except (KeyError, TypeError, ValueError, AssertionError) as exc:
            withheld.append({"stage": name, "reason": str(exc)})
    result = {
        "job_id": JOB,
        "code_commit": CODE_SHA,
        "contract": CONTRACT,
        "scientific_acceptance": False,
        "valid_exploratory_stages": accepted,
        "withheld_stages": withheld,
        "incomplete_stages": pending,
        "imported_B128_baseline": imported,
        "conditional_512_decision": report.get("conditional_512_decision"),
        "report_status": report.get("status"),
        "accounting": parent_accounting(args.sacct_file.read_text()) if args.sacct_file else None,
        "dispatch_comparison_scope": "old B128 lazy C++ dispatch; new B256/B512 explicit Python AOT binding overhead included; not isolated batch-size effect",
        "K_amortization": "not reported; no separately observed setup model",
        "collector_execution": "CPU saved-array read/validation only; no model run or compilation",
        "legacy_collector": {"path": str(legacy_path), "normalized_source_sha256": LEGACY_SHA},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "job_id": JOB,
                "valid_stages": len(accepted),
                "withheld_stages": len(withheld),
                "incomplete_stages": len(pending),
                "imported_B128_is_new_stage": False,
                "output": str(args.output),
            }
        )
    )
    return 1 if withheld else 0


if __name__ == "__main__":
    raise SystemExit(main())
