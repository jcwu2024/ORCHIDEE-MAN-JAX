"""Stdlib cost scenarios for complete, checksummed same-workload server stages.

No model import/unpickling, execution, annual projection or scheduler mutation.
Prices are declared scenarios, not verified billing. Old local/free ledgers are
never consumed as new allocation budgets or new server CPU measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import after the path bootstrap so an explicit checkout takes precedence.
from scripts.dev.global_server_reference import finite_tolerance_failure  # noqa: E402, I001

PRECISION = "JAX x64 enabled; baseline native dtypes retained; geographical payload and solar subcalculation contain FP32"
FIXED_GATE = {"atol": 1e-8, "rtol": 1e-10, "discrete_schema_dtype": "exact",
              "sentinel": "declared_bare_soil_roughheight_only"}
EXPLORATORY_LABEL = "performance measured; cross-backend scientific consistency not accepted"


def metrics(value, point_days):
    samples = value.get("samples_seconds", [])
    if len(samples) < 5 or any(not math.isfinite(item) or item <= 0 for item in samples):
        raise ValueError("at least five positive synchronized samples required")
    median = statistics.median(samples)
    if value.get("valid_point_days") != point_days or not math.isclose(value.get("median_seconds", -1), median, rel_tol=1e-12):
        raise ValueError("exact common-workload denominator and preserved median required")
    return {"samples_seconds": samples, "median_seconds": median,
            "range_seconds": [min(samples), max(samples)],
            "point_days": point_days, "point_days_per_second": point_days / median}


def allocated_cpu_count(resources):
    request = resources.get("allocated_CPU_request_count", resources.get("allocated_CPU_cores"))
    if not isinstance(request, int) or request <= 0:
        raise ValueError("actual declared Slurm CPU allocation required")
    tres = resources.get("Slurm", {}).get("ORCHJAX_SLURM_ALLOC_TRES") or ""
    source = resources.get("Slurm", {}).get("ORCHJAX_SLURM_ALLOC_TRES_SOURCE") or "TRES source unrecorded"
    counts = [part.split("=", 1)[1] for part in tres.split(",") if part.strip().startswith("cpu=")]
    actual = int(counts[0]) if counts else request
    return actual, {"requested_CPU_count": request, "recorded_TRES": tres or None, "TRES_source": source if tres else None,
                    "cost_CPU_count": actual, "count_origin": f"{source}.cpu" if counts else "Slurm requested count; TRES absent",
                    "billing_verified": False, "physical_topology": resources.get("topology"),
                    "selected_logical_cpus": resources.get("selected_logical_cpus")}


def accept_stage(directory, name, manifest, report):
    record = manifest["stages"][name]
    if (manifest["identity"].get("validation_mode", "strict") == "exploratory"
            or record.get("evidence_class") == "exploratory_performance"):
        raise ValueError("exploratory performance cannot be promoted to accepted evidence")
    if record.get("strict_numerical_passed") is False:
        raise ValueError("failed strict numerical status cannot be accepted")
    if (manifest["identity"].get("cpu_reference_role") == "cpu-internal"
            or record.get("provenance", {}).get("CPU_scalar_reference_role") == "cpu_internal_comparison"):
        raise ValueError("CPU internal comparison cannot be promoted to accepted evidence")
    return _checked_stage(directory, name, manifest, report)


def exploratory_stage(directory, name, manifest, report):
    return _checked_stage(directory, name, manifest, report, exploratory=True)


def _checked_stage(directory, name, manifest, report, *, exploratory=False):
    record = manifest["stages"][name]
    if record.get("status") != "complete":
        raise ValueError("stage incomplete; throughput/cost withheld")
    identity = manifest["identity"]
    if identity.get("validation_mode", "strict") not in {"strict", "exploratory"}:
        raise ValueError("unknown validation mode")
    if exploratory and (identity.get("validation_mode") != "exploratory"
            or identity.get("mode") not in {"scalar", "pool", "batch"}
            or record.get("evidence_class") != "exploratory_performance"
            or record.get("measurement_status") != "complete"
            or identity.get("contract") != "server-CPU-same-workload-v1"):
        raise ValueError("complete explicitly exploratory CPU measurement required")
    if identity.get("gate") != FIXED_GATE:
        raise ValueError("unchanged fixed tolerance/schema/discrete/sentinel policy required")
    if identity.get("contract") not in {"server-CPU-same-workload-v1", "server-candidate-first-full-physics-v1"}:
        raise ValueError("only the new declared server benchmark contracts are cost candidates")
    path = directory / record.get("path", "")
    if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("required complete checkpoint missing/outside directory")
    expected = record.get("sha256", "")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if len(expected) != 64 or digest.hexdigest() != expected:
        raise ValueError("complete checkpoint SHA verification failed")
    proof = record["provenance"]
    size = proof.get("unique_real_points")
    if size not in {16, 64}:
        raise ValueError("only actual unique16/64 workload stages admitted")
    points = proof.get("point_ids")
    if len(set(points or [])) != size or points != [lane["landpoint_id"] for lane in identity["lanes"][:size]]:
        raise ValueError("exact selected point order required")
    if (proof.get("precision_policy") != PRECISION or identity.get("precision_policy") != PRECISION
            or proof.get("blocks") != [[2, 8], [9, 15]] or proof.get("half_hours_per_day") != 48
            or any(proof.get(key) is not True for key in ("full_outputs", "five_or_more_synchronized_repeats"))
            or (not exploratory and proof.get("full_two_block_parity") is not True)
            or (exploratory and proof.get("full_two_block_workload") is not True)):
        raise ValueError("unchanged native input/full state/full output/two7day/48step proof required")
    if not record.get("gates") or (not exploratory and any(gate.get("passed") is not True for gate in record["gates"])):
        raise ValueError("complete numerical gates required")
    names = {gate.get("name") for gate in record["gates"]}
    is_gpu = identity["contract"] == "server-candidate-first-full-physics-v1"
    if is_gpu:
        required = [f"B{size}_lane_{index}_{point}_both_blocks" for index, point in enumerate(points)]
        required += [f"B{size}_cross_group_permutation", f"B{size}_replacement_all_lanes_isolation"]
    else:
        mode = identity["mode"]
        required = [f"B{size}_{mode}_lane_{index}_{point}_GPU_reference" for index, point in enumerate(points)]
        required += [f"B{size}_{mode}_frozen_reset_repeat"]
        if mode != "scalar":
            required.append(f"B{size}_{mode}_cross_mode_server_scalar")
        if mode == "batch":
            required += [f"B{size}_server_batch_cross_group_permutation", f"B{size}_server_batch_replacement_isolation"]
    if len(names) != len(record["gates"]) or any(name not in names for name in required):
        raise ValueError("exact complete per-lane/self/cross-mode/isolation gate set required")
    failed = [gate for gate in record["gates"] if gate.get("passed") is not True]
    if exploratory:
        gpu_names = set(required[:size])
        if any(gate.get("passed") is True and gate.get("failures") for gate in record["gates"]):
            raise ValueError("gate passed status contradicts recorded failures")
        for gate in failed:
            if (gate.get("name") not in gpu_names or not finite_tolerance_failure(gate)
                    or gate.get("atol") != FIXED_GATE["atol"] or gate.get("rtol") != FIXED_GATE["rtol"]):
                raise ValueError("only finite GPU cross-backend tolerance failures admit exploratory timing")
        if (record.get("strict_numerical_passed") is not (not failed)
                or proof.get("full_two_block_parity") is not (not failed)):
            raise ValueError("actual strict numerical/parity status required")
    runtime = record.get("execution_origin", {}).get("environment", {})
    if runtime.get("jax") != "0.4.38" or runtime.get("jaxlib") != "0.4.38" or runtime.get("x64_enabled") is not True:
        raise ValueError("accepted server JAX/jaxlib0.4.38 x64 origin required")
    platform = "gpu" if is_gpu else "cpu"
    if {item.get("platform") for item in runtime.get("device_details", [])} != {platform}:
        raise ValueError("stage actual device origin mismatch")
    cpu_flags = ("frozen_reset_repeat",) if identity.get("mode") == "scalar" else ("frozen_reset_repeat", "cross_mode_scalar_parity")
    if not is_gpu and any(proof.get(flag) is not True for flag in cpu_flags):
        raise ValueError("CPU repeat/cross-mode scalar parity required")
    if exploratory and identity["mode"] != "scalar":
        role = identity.get("cpu_reference_role")
        reference = proof.get("CPU_scalar_reference") or {}
        source = reference.get("source_identity") or {}
        expected_role = "cpu_internal_comparison" if role == "cpu-internal" else "accepted"
        if (role not in {"accepted", "cpu-internal"}
                or proof.get("CPU_scalar_reference_role") != expected_role
                or (role == "cpu-internal" and reference.get("role") != expected_role)
                or (role == "accepted" and (reference.get("role") == "cpu_internal_comparison"
                    or source.get("validation_mode", "strict") != "strict"))
                or source.get("mode") != "scalar" or source.get("contract") != "server-CPU-same-workload-v1"
                or {item.get("platform") for item in source.get("runtime", {}).get("device_details", [])} != {"cpu"}
                or reference.get("native_input_identity_verified") is not True
                or reference.get("fixed_gate") != FIXED_GATE
                or reference.get("point_ids") != points
                or source.get("lanes", [])[:size] != identity["lanes"][:size]
                or any(source.get(key) != identity.get(key) for key in
                    ("science_sha256", "baseline_ref", "precision_policy", "numerical_reference_sha256"))):
            raise ValueError("explicit matching CPU scalar reference role and input/source proof required")
    if is_gpu and any(proof.get(flag) is not True for flag in ("cross_group_permutation", "replacement_isolation")):
        raise ValueError("GPU permutation/replacement isolation required")
    resources = proof.get("server_resources", report.get("server_resources", {}))
    slurm = resources.get("Slurm", {})
    if not slurm.get("SLURM_JOB_ID") or slurm.get("SLURM_JOB_PARTITION") not in ({"gnall"} if is_gpu else {"cnall", "cnmix"}):
        raise ValueError("actual matching paid Slurm allocation provenance required")
    count, allocation = allocated_cpu_count(resources)
    tres = slurm.get("ORCHJAX_SLURM_ALLOC_TRES") or ""
    allocated_gpus = [int(part.split("=", 1)[1]) for part in tres.split(",")
                      if part.strip().startswith("gres/gpu=")]
    if allocated_gpus and allocated_gpus != [1 if is_gpu else 0]:
        raise ValueError("actual AllocTRES GPU allocation differs from single-card/CPU contract")
    allocation["GPU_count_origin"] = f"{allocation['TRES_source']}.gres/gpu" if allocated_gpus else "declared contract; TRES.gres/gpu absent"
    values = record["timing"]
    point_days = size * 14
    core = metrics(values[f"B{size}_warm_frozen_state_core"], point_days)
    wrapper = metrics(values[f"B{size}_preprepared_wrapper_stack_transfer_materialize"], point_days)
    result = {"directory": str(directory), "stage": name, "backend": platform, "CPU_count": count,
            "GPU_count": 1 if is_gpu else 0, "allocation": allocation, "core": core, "wrapper": wrapper,
            "point_ids": points, "native_input_lanes": identity["lanes"][:size],
            "scientific_source_sha256": identity["science_sha256"], "baseline_ref": identity["baseline_ref"],
            "precision_policy": PRECISION, "checkpoint_sha256": expected, "execution_origin": record["execution_origin"],
            "NumPy_version": runtime.get("numpy"), "complete_gate_count": len(record["gates"]),
            "saved_output_checkpoint_IO": record.get("checkpoint_io"),
            "Fortran_baseline": "pending; no executable/source/build inventory accepted"}
    if exploratory:
        result.update(evidence_class="exploratory_performance", scientific_acceptance=False,
            performance_label=EXPLORATORY_LABEL, strict_numerical_passed=record["strict_numerical_passed"],
            failed_gates=failed, full_two_block_parity=proof["full_two_block_parity"],
            workload={"days": 14, "half_hours_per_day": 48, "full_outputs": True},
            worker_records=proof.get("worker_records"),
            CPU_mode=identity["mode"], controller_process_id=record["execution_origin"].get("pid"),
            CPU_scalar_reference_role=proof.get("CPU_scalar_reference_role"),
            thread_affinity_after_compile=report.get("thread_affinity_after_compile"),
            controller_thread_affinity_after_warm=proof.get("controller_thread_affinity_after_warm"),
            startup_and_compile_timing={key: value for key, value in report.get("timing", {}).items()
                if (key.startswith(f"B{size}_") or key.startswith("server_scalar_once_")
                    or key.startswith(f"server_batch_B{size}_")) and key not in values})
    return result


def cost_for_seconds(seconds, cpu_count, gpu_count, cpu_price, gpu_price):
    cpu_hours, gpu_hours = cpu_count * seconds / 3600, gpu_count * seconds / 3600
    return {"wall_seconds": seconds, "CPU_allocated_count_hours": cpu_hours,
            "GPU_card_hours": gpu_hours, "CNY": cpu_hours * cpu_price + gpu_hours * gpu_price}


def scenarios(stage, cpu_price, gpu_price, reuse_counts, job=None):
    result = {}
    for boundary in ("core", "wrapper"):
        metric = stage[boundary]
        value = cost_for_seconds(metric["median_seconds"], stage["CPU_count"], stage["GPU_count"], cpu_price, gpu_price)
        value.update(CNY_per_common_workload=value["CNY"], point_days_per_CNY=metric["point_days"] / value["CNY"],
                     pricing_kind="declared scenario; no bill verification", scope="bounded warm workload only; startup excluded")
        result[boundary] = value
    result["aggregate_benchmark_job_cost"] = None
    result["assumed_K_reuse"] = []
    if job is not None:
        directories = job.get("directories", [job.get("directory")])
        terminal = {"completed", "timeout", "timed_out", "failed", "cancelled", "canceled", "over_budget", "stopped"}
        if (str(job.get("status", "")).lower() not in terminal or not job.get("provenance") or stage["directory"] not in directories
                or job.get("allocated_CPU_count") != stage["CPU_count"] or job.get("GPU_count") != stage["GPU_count"]):
            raise ValueError("full-job external resource/directory/completion accounting mismatch")
        wall = job.get("wall_seconds")
        if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("external observed whole-job wall required")
        whole = cost_for_seconds(wall, stage["CPU_count"], stage["GPU_count"], cpu_price, gpu_price)
        whole.update(scope="aggregate observed benchmark job across methods/sizes/repeats; startup/compile/gates/IO included; never divided by one224/896 denominator",
                     accounting_origin=job.get("provenance"), benchmark_directories=directories,
                     observed_terminal_status=job["status"], reason=job.get("reason"),
                     not_additive_across_stages=True)
        result["aggregate_benchmark_job_cost"] = whole
        startup = job.get("stage_startup_and_validation_seconds", {}).get(stage["stage"])
        if startup is not None:
            if not isinstance(startup, (int, float)) or not math.isfinite(startup) or startup < 0:
                raise ValueError("explicit nonnegative observed startup/compile/validation/IO overhead required")
            io = stage.get("saved_output_checkpoint_IO") or {}
            if io.get("available") is not True:
                raise ValueError("measured saved-output checkpoint IO required for K-reuse model")
            io_seconds = io["serialization_seconds"] + io["atomic_write_seconds"]
            for count in reuse_counts:
                estimated = cost_for_seconds(startup + count * (stage["core"]["median_seconds"] + io_seconds),
                    stage["CPU_count"], stage["GPU_count"], cpu_price, gpu_price)
                estimated.update(assumed_K=count, assumed_K_is_observed=False,
                    CNY_per_common_workload=estimated["CNY"] / count,
                    point_days_per_CNY=count * stage["core"]["point_days"] / estimated["CNY"],
                    overhead_origin="explicit external stage_startup_and_validation_seconds",
                    measured_saved_IO_seconds_per_workload=io_seconds,
                    saved_IO_scope="complete diagnostic snapshot serialization+atomic write; not production checkpoint acceptance",
                    scope="declared K-reuse assumption for the same bounded workload; no annual measurement")
                result["assumed_K_reuse"].append(estimated)
    result["missing_job_accounting"] = job is None
    result["missing_amortization_overhead"] = not result["assumed_K_reuse"]
    return result


def summarize(directories, jobs, *, gpu_price=2.20, ratio=44.0, reported_cpu_price=0.07, reuse_counts=(1, 10, 100)):
    stages, exploratory, withheld = [], [], []
    for directory in directories:
        directory = Path(directory).resolve()
        manifest = json.loads((directory / "stage_manifest.json").read_text(encoding="utf-8-sig"))
        report = json.loads((directory / "report.json").read_text(encoding="utf-8-sig"))
        for name in manifest["stages"]:
            if "server_" not in name and "candidate_first_v1" not in name:
                continue
            try:
                if manifest["identity"].get("validation_mode", "strict") == "exploratory" or manifest["stages"][name].get("evidence_class") == "exploratory_performance":
                    exploratory.append(exploratory_stage(directory, name, manifest, report))
                else:
                    stages.append(accept_stage(directory, name, manifest, report))
            except (KeyError, TypeError, ValueError) as exc:
                withheld.append({"directory": str(directory), "stage": name, "reason": str(exc)})
    inferred = gpu_price / ratio
    pricing = {"primary": {"CPU_CNY_per_allocated_count_hour": inferred, "GPU_CNY_per_card_hour": gpu_price,
                            "CPU_price_origin": "inferred from user's44 ratio and declared GPU2.20; not independently bill-verified"},
               "conservative": {"CPU_CNY_per_allocated_count_hour": reported_cpu_price, "GPU_CNY_per_card_hour": gpu_price,
                    "CPU_price_origin": "deployment report0.07; conflicts with44 ratio; not a new verified bill"},
               "ratio_conflict": {"latest_human_ratio": ratio, "deployment_price_implied_ratio": gpu_price / reported_cpu_price,
                    "resolved": False}, "GPU_auxiliary_CPU_cost_included": True}
    for stage in stages + exploratory:
        job = next((item for item in jobs if stage["directory"] in item.get("directories", [item.get("directory")])), None)
        stage["cost_scenarios"] = {}
        for name, price in pricing.items():
            if name not in {"primary", "conservative"}:
                continue
            try:
                value = scenarios(stage, price["CPU_CNY_per_allocated_count_hour"], gpu_price, reuse_counts, job)
            except (KeyError, TypeError, ValueError) as exc:
                value = scenarios(stage, price["CPU_CNY_per_allocated_count_hour"], gpu_price, reuse_counts)
                value["job_accounting_withheld_reason"] = str(exc)
            stage["cost_scenarios"][name] = value
    comparisons = []
    for gpu in (stage for stage in stages if stage["GPU_count"]):
        for cpu in (stage for stage in stages if not stage["GPU_count"]):
            keys = ("point_ids", "native_input_lanes", "scientific_source_sha256", "baseline_ref", "precision_policy")
            if any(gpu[key] != cpu[key] for key in keys):
                continue
            qratio = gpu["core"]["point_days_per_second"] / cpu["core"]["point_days_per_second"]
            threshold = (ratio + gpu["CPU_count"]) / cpu["CPU_count"]
            comparisons.append({"GPU_stage": gpu["stage"], "CPU_stage": cpu["stage"], "CPU_directory": cpu["directory"],
                "GPU_directory": gpu["directory"], "same_workload_point_days": gpu["core"]["point_days"],
                "measured_throughput_ratio": qratio, "primary_cost_efficiency_threshold": threshold,
                "GPU_cheaper_per_warm_workload_primary": qratio > threshold,
                "formula": "Q_GPU/Q_CPU(N) > (44+M)/N; N actual allocated CPU count, M actual GPU auxiliary CPU count",
                "not_measured_44core_CPU": True, "runtime_conditions": {"CPU": cpu["execution_origin"], "GPU": gpu["execution_origin"]}})
    exploratory_comparisons = []
    for gpu in (stage for stage in stages if stage["GPU_count"]):
        for cpu in exploratory:
            keys = ("point_ids", "native_input_lanes", "scientific_source_sha256", "baseline_ref", "precision_policy")
            if any(gpu[key] != cpu[key] for key in keys) or gpu["core"]["point_days"] != cpu["core"]["point_days"]:
                continue
            exploratory_comparisons.append({"GPU_stage": gpu["stage"], "CPU_stage": cpu["stage"],
                "GPU_directory": gpu["directory"], "CPU_directory": cpu["directory"],
                "scientific_acceptance": False, "performance_label": EXPLORATORY_LABEL,
                "same_workload_point_days": gpu["core"]["point_days"],
                "measured_throughput_ratio": gpu["core"]["point_days_per_second"] / cpu["core"]["point_days_per_second"],
                "wrapper_throughput_ratio": gpu["wrapper"]["point_days_per_second"] / cpu["wrapper"]["point_days_per_second"],
                "primary_cost_efficiency_threshold": (ratio + gpu["CPU_count"]) / cpu["CPU_count"],
                "runtime_conditions": {"CPU": cpu["execution_origin"], "GPU": gpu["execution_origin"]}})
    status = "bounded_cost_scenarios" if stages else "exploratory_performance_measured" if exploratory else "no_accepted_server_measurements"
    return {"schema_version": 1, "status": status,
            "pricing": pricing, "accepted_stages": stages, "withheld_stages": withheld, "same_workload_comparisons": comparisons,
            "exploratory_stages": exploratory, "exploratory_same_workload_comparisons": exploratory_comparisons,
            "excluded_claims": ["annual/50year", "Fortran speedup", "10x achieved", "CPU1 times44 measured CPU44", "uncompleted B64 first-call throughput"],
            "full_job_accounting_schema": {"directories": ["all resolved benchmark outputs in the observed job"], "status": "completed", "wall_seconds": "observed whole job",
                "allocated_CPU_count": "actual CPU TRES/request count", "GPU_count": "actual cards", "provenance": "accounting artifact",
                "stage_startup_and_validation_seconds": {"stage_name": "explicit observed overhead; required for hypothetical K amortization"}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, action="append", default=[])
    parser.add_argument("--gpu-dir", type=Path, action="append", default=[])
    parser.add_argument("--job-accounting", type=Path, action="append", default=[])
    parser.add_argument("--reuse-counts", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--gpu-cny-per-hour", type=float, default=2.20)
    parser.add_argument("--reported-cpu-cny-per-core-hour", type=float, default=0.07)
    parser.add_argument("--gpu-to-cpu-price-ratio", type=float, default=44)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (not args.cpu_dir and not args.gpu_dir or any(count <= 0 for count in args.reuse_counts)
            or any(not math.isfinite(price) or price <= 0 for price in
                   (args.gpu_cny_per_hour, args.reported_cpu_cny_per_core_hour, args.gpu_to_cpu_price_ratio))):
        parser.error("declared positive prices/K and at least one measurement directory required")
    jobs = [json.loads(path.read_text(encoding="utf-8-sig")) for path in args.job_accounting]
    result = summarize(args.cpu_dir + args.gpu_dir, jobs, gpu_price=args.gpu_cny_per_hour,
        ratio=args.gpu_to_cpu_price_ratio, reported_cpu_price=args.reported_cpu_cny_per_core_hour, reuse_counts=args.reuse_counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "accepted": len(result["accepted_stages"]),
        "exploratory": len(result["exploratory_stages"]), "withheld": len(result["withheld_stages"])}))


if __name__ == "__main__":
    main()
