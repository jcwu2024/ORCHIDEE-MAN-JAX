"""Fixed-512-lane GPU capacity screen; repeated real inputs are not new points.

No process logic: execution uses the audited global_batch transition factory
and benchmark_global_batch.run_two unchanged (complete Days2-8 and9-15).
The unique FULL64 preparation/reference remains immutable numerical evidence.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import signal
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.dev.global_server_reference import _evidence_mismatch, finite_tolerance_failure

TOTAL_LANES = 512
UNIQUE_POINTS = 64
CONTRACT = "GPU-fixed512-repeated-FULL64-capacity-v1"
B128_SHA256 = "e91db1391ccb5f525f9ec29f1cba24235c4ce9cd4153b9297fe2ed014d0c9cb5"
B256_SHA256 = "ee7210f2c83ea11346e9aa752320e8760560b2a6c1e377aa2058aa53da16b08c"
CONTINUATION_CONTRACT = "GPU-fixed512-repeated-FULL64-B256-B512-continuation-v1"
B512_CONTRACT = "GPU-fixed512-repeated-FULL64-B512-only-continuation-v1"
FIXED1024_CONTRACT = "GPU-fixed1024-repeated-FULL64-B512-B1024-capacity-v1"
B512_SHA256 = "1c44d84ad73f3a04f6d21522a0b35011d157e3a1dbc5b155dc93833a44912a2a"
B512_BYTES = 220430830
B512_CODE_SHA = "ad6c0e1faa7e22b590627844e9cb0ad492e9deb3"
B512_ACTUAL_SOURCE_SHA256 = "b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8"
B512_PEAK_RSS_BYTES = 19496943616
LABEL = "capacity performance measured; global scientific consistency not accepted"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prepared-input", type=Path, required=True)
    result.add_argument("--full-reference-dir", type=Path, required=True)
    result.add_argument("--memory-baseline-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--config", type=Path, default=ROOT / "configs/orchidee_man_250919.yaml")
    result.add_argument("--asset-root", type=Path)
    result.add_argument("--baseline-ref", default="133e35aec7b83b9d79888c46092f64d518e97299")
    result.add_argument("--warm-repeats", type=int, default=5)
    result.add_argument("--budget-seconds", type=int, default=5400)
    result.add_argument("--conditional-512", action="store_true")
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--continue-256-512", action="store_true", help="Import sealed B128 timing origin; measure only B256 then resource-admitted B512, regardless of speedup.")
    modes.add_argument("--continue-512", action="store_true", help="Import sealed B128/B256 origins and B256 compiler memory; measure only resource-admitted B512.")
    modes.add_argument("--fixed-1024", action="store_true", help="Measure new same1024-lane B512 then resource-admitted B1024; historical B512 is numerical/memory evidence only.")
    result.add_argument("--b512-evidence-dir", type=Path)
    result.add_argument("--b128-baseline-dir", type=Path)
    result.add_argument("--b256-baseline-dir", type=Path)
    result.add_argument("--optional-b64-baseline", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--admit-inputs", action="store_true", help="Read/audit FULL64 and repeated identities only; no process factory or compilation.")
    return result


def capacity_layout(lanes, second, total_lanes=TOTAL_LANES):
    ids = [lane.landpoint_id for lane in lanes]
    if len(lanes) != UNIQUE_POINTS or len(second) != UNIQUE_POINTS or len(set(ids)) != UNIQUE_POINTS:
        raise ValueError("exact unique FULL64 source lanes and aligned second blocks required")
    if total_lanes not in (512, 1024):
        raise ValueError("capacity collection must contain512 or1024 lanes")
    origins = [{"lane_index": index, "source_index": index % UNIQUE_POINTS,
                "landpoint_id": ids[index % UNIQUE_POINTS], "repeat_index": index // UNIQUE_POINTS}
               for index in range(total_lanes)]
    return ([lanes[item["source_index"]] for item in origins],
            [second[item["source_index"]] for item in origins], origins)


def lane_metrics(samples, total_lanes=TOTAL_LANES):
    if len(samples) != 5 or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("five positive fully synchronized samples required")
    median = statistics.median(samples)
    return {"samples_seconds": samples, "median_seconds": median, "range_seconds": [min(samples), max(samples)],
        "lane_days": total_lanes * 14, "lane_days_per_second": total_lanes * 14 / median,
        "unique_real_points": UNIQUE_POINTS, "unique_point_days": UNIQUE_POINTS * 14,
        "unique_point_days_is_throughput_denominator": False}


def gate(recorder, name, actual, expected, compare, *, cross_backend=False):
    record = {"name": name, **compare(actual, expected, sentinel_paths=recorder.sentinel_paths),
              "gate_role": "GPU_reference" if cross_backend else "strict_GPU_internal"}
    recorder.result["gates"].append(record)
    if record["passed"] is not True:
        if cross_backend and finite_tolerance_failure(record):
            record["exploratory_continuation"] = True
        else:
            raise AssertionError(f"hard capacity gate failed: {name}: {record}")
    return record


def conditional_512_decision(records, remaining_seconds, estimated_bytes, free_bytes, *, continuation=False):
    reasons = []
    if set(records) != {128, 256}:
        reasons.append("both completed candidate records required")
    else:
        for boundary in (() if continuation else ("core", "wrapper")):
            if records[256][boundary]["median_seconds"] > .95 * records[128][boundary]["median_seconds"]:
                reasons.append(f"B256 {boundary} must improve same-512-workload median by at least5%")
    if remaining_seconds < 1800:
        reasons.append("at least30min remaining required")
    if estimated_bytes is None or free_bytes < estimated_bytes + 2 * 1024**3:
        reasons.append("estimated B512 buffers plus2GiB physical slack required")
    return {"permitted": not reasons, "reasons": reasons, "remaining_seconds": remaining_seconds,
            "estimated_device_bytes": estimated_bytes, "free_device_bytes": free_bytes,
            "scope": "preset capacity-screen condition, not OOM or completion guarantee",
            "policy": "resource_admitted_scale_curve" if continuation else "both_boundaries_5percent_improvement"}


def baseline_memory(directory):
    manifest = json.loads((directory / "stage_manifest.json").read_text())
    record = manifest["stages"].get("B64_candidate_first_v1", {})
    proof = record.get("provenance", {})
    analysis = proof.get("executable_memory_analysis", {})
    if record.get("status") != "complete" or analysis.get("available") is not True:
        raise ValueError("completed old B64 actual compiler memory analysis required; no new B64 compile")
    report = json.loads((directory / "report.json").read_text())
    peak = report.get("linux_ru_maxrss_peak_bytes") or report.get("peak_rss_bytes")
    if not isinstance(peak, int) or peak <= 0:
        raise ValueError("actual old benchmark host peak RSS required for bounded host admission estimate")
    analysis = {**analysis, "old_benchmark_peak_RSS_bytes": peak}
    return manifest, record, analysis


def admit_numerical_sources(args, lanes, second, scale, environment):
    """Read exact source identity and sealed numerical/memory prerequisites only."""
    from scripts.dev.global_server_reference import load_reference

    identity = scale.base.stage_identity(args, lanes, second, environment)
    identity.update(contract=CONTRACT, precision_policy=scale.PRECISION_POLICY)
    reference, reference_origin = load_reference(args.full_reference_dir, identity)
    old_manifest, old_record, analysis = baseline_memory(args.memory_baseline_dir)
    for key in ("lanes", "science_sha256", "baseline_ref", "precision_policy", "gate"):
        if old_manifest["identity"].get(key) != identity.get(key):
            raise ValueError(f"old B64 memory evidence input/science mismatch: {key}")
    return identity, reference, reference_origin, old_record, analysis


def import_b128_baseline(directory, expected_identity, destination, jax, *, reference=None):
    return _import_capacity_baseline(directory, expected_identity, destination, jax,
        size=128, sha256=B128_SHA256, byte_count=220429366, job_id="15600405",
        code_sha="a51efcdabf7ff2e9f97a463a6f72e9f6a2cf478e", reference=reference)


def import_b256_baseline(directory, expected_identity, destination, jax, *, reference=None):
    return _import_capacity_baseline(directory, expected_identity, destination, jax,
        size=256, sha256=B256_SHA256, byte_count=220429675, job_id="15643868",
        code_sha="12b495b9425de3ad1dc16efa3a5a0e69d7e89222", reference=reference)


def _import_capacity_baseline(directory, expected_identity, destination, jax, *,
                              size, sha256, byte_count, job_id, code_sha, reference):
    """Validate sealed historical evidence without modifying or relabeling its stage."""
    from scripts.dev.global_batch_stage_store import StageStore

    StageStore.validate_source_destination(directory, destination)
    directory = Path(directory)
    manifest_path = directory / "stage_manifest.json"
    manifest_data = manifest_path.read_bytes()
    manifest = json.loads(manifest_data)
    source_identity = manifest.get("identity", {})
    admission_keys = ("contract", "baseline_ref", "science_sha256", "lanes", "runtime", "compiler_environment",
        "warm_repeats", "gate", "precision_policy", "validation_mode", "workload_kind", "total_lanes",
        "lane_origins", "unique_real_points", "candidate_shapes", "conditional_512", "optional_B64_baseline",
        "conditional_policy", "numerical_reference_sha256")
    if size in (256, 512):
        admission_keys += ("continuation_B128_sha256",)
    if size == 512:
        admission_keys += ("continuation_B256_sha256", "budget_seconds", "actual_science_source_sha256", "execution_source_compatibility")
    for key in admission_keys:
        if key not in source_identity or source_identity[key] != expected_identity.get(key):
            raise ValueError(f"B{size} baseline identity mismatch: {key}")
    name = f"capacity_B{size}_fixed512_v1"
    record = manifest["stages"].get(name, {})
    if record.get("status") != "complete" or record.get("sha256") != sha256:
        raise ValueError(f"B{size} baseline must be the sealed job{job_id} stage")
    path = directory / ("stage_" + name + ".pkl")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256 or len(data) != byte_count:
        raise ValueError(f"B{size} baseline full checksum/bytes changed")
    envelope = pickle.loads(data)
    if (envelope.get("identity") != source_identity or envelope.get("name") != name
            or envelope.get("status") != "complete"):
        raise ValueError(f"B{size} baseline envelope identity/status mismatch")
    proof, value = envelope["record"], envelope["value"]
    if (proof.get("measurement_status") != "complete" or proof.get("strict_numerical_passed") is not True
            or len(proof.get("gates", [])) != 515 or any(g.get("passed") is not True for g in proof["gates"])
            or proof.get("timing") != value.get("measurements") or proof.get("timing") != record.get("timing")
            or proof.get("provenance") != record.get("provenance")
            or _evidence_mismatch(proof.get("execution_origin"), record.get("execution_origin"),
                                  "record.execution_origin") is not None):
        raise ValueError(f"B{size} baseline complete strict measurement/provenance mismatch")
    if proof["execution_origin"].get("environment", {}).get("git_head") != code_sha:
        raise ValueError(f"B{size} baseline original code origin mismatch")
    provenance = proof["provenance"]
    if any(provenance.get(key) != expected for key, expected in {
            "workload_kind": "capacity_screen_repeated_FULL64",
            "total_lanes": 512, "unique_real_points": 64, "chunk_size": size,
            "chunks_per_collection": TOTAL_LANES // size,
            "lane_days": TOTAL_LANES * 14, "unique_point_days": UNIQUE_POINTS * 14,
            "blocks": [[2, 8], [9, 15]], "half_hours_per_day": 48,
            "full_outputs": True, "full_two_block_workload": True}.items()):
        raise ValueError(f"B{size} baseline workload provenance mismatch")
    leaves = jax.tree_util.tree_leaves(value["outputs"])
    if len(leaves) != 760 or any(leaf.shape[0] != TOTAL_LANES for leaf in leaves):
        raise ValueError(f"B{size} baseline full output schema mismatch")
    if reference is not None:
        originals, original_tree = jax.tree_util.tree_flatten(reference)
        _, imported_tree = jax.tree_util.tree_flatten(value["outputs"])
        if imported_tree != original_tree or len(originals) != len(leaves) or any(
                leaf.shape != (TOTAL_LANES, *original.shape[1:]) or leaf.dtype != original.dtype
                for leaf, original in zip(leaves, originals, strict=True)):
            raise ValueError(f"B{size} baseline full native output schema/dtype mismatch")
    if set(value["measurements"]) != {"core", "wrapper"}:
        raise ValueError(f"B{size} baseline complete core/wrapper timings required")
    for metric in value["measurements"].values():
        if metric != {**metric, **lane_metrics(metric["samples_seconds"])}:
            raise ValueError(f"B{size} baseline synchronized timing summary mismatch")
    if size in (256, 512):
        analysis = provenance.get("executable_memory_analysis", {})
        if (analysis.get("available") is not True
                or type(analysis.get("temp_size_in_bytes")) is not int or analysis["temp_size_in_bytes"] < 0):
            raise ValueError("B256 baseline actual compiler memory analysis required")
    origin = {"path": str(path), "sha256": sha256, "bytes": len(data),
        "manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
        "source_stage": name, "source_identity": source_identity, "execution_origin": proof["execution_origin"],
        "timing": value["measurements"], "provenance": provenance,
        "role": f"imported historical baseline; no new B{size} measurement, executable, or destination stage"}
    return value["measurements"], origin


def memory_estimate(size, resident_input_bytes, full_output_bytes, memory_records):
    temp = max((value["temp_size_in_bytes"] * size // previous for previous, value in memory_records.items()
                if value.get("available") is True), default=None)
    return None if temp is None else 2 * resident_input_bytes + 3 * full_output_bytes + temp


def b512_evidence_identity(identity, origins, reference_sha256):
    """Exact historical job15657822 identity, independent of new timing origin."""
    from scripts.dev.global_source_compatibility import source_identity

    source = source_identity(ROOT)
    certificate = {**source["execution_source_compatibility"],
        "baseline_checkout_bytes_sha256": source["science_sha256"],
        "baseline_checkout_representation": "canonical Linux LF source bytes",
        "actual_full_source_sha256": B512_ACTUAL_SOURCE_SHA256}
    return {**identity, "contract": B512_CONTRACT, "validation_mode": "exploratory",
        "workload_kind": "capacity_screen_repeated_FULL64", "total_lanes": 512,
        "lane_origins": origins[:512], "unique_real_points": 64, "candidate_shapes": [512],
        "conditional_512": True, "optional_B64_baseline": False,
        "conditional_policy": "B512_only_30min_imported_B256compiler_memory_2GiB_GPU_8GiB_host_slack",
        "continuation_B128_sha256": B128_SHA256, "continuation_B256_sha256": B256_SHA256,
        "numerical_reference_sha256": reference_sha256, "warm_repeats": 5, "budget_seconds": 2700,
        "actual_science_source_sha256": B512_ACTUAL_SOURCE_SHA256,
        "execution_source_compatibility": certificate}


def import_b512_evidence(directory, expected_identity, destination, jax, *, reference=None):
    """Admit numerical/schema/compiler/RSS provenance; never return old timings."""
    _, origin = _import_capacity_baseline(directory, expected_identity, destination, jax,
        size=512, sha256=B512_SHA256, byte_count=B512_BYTES, job_id="15657822",
        code_sha=B512_CODE_SHA, reference=reference)
    report_data = (Path(directory) / "report.json").read_bytes()
    report = json.loads(report_data)
    name = origin["source_stage"]
    manifest = json.loads((Path(directory) / "stage_manifest.json").read_text())
    if (report.get("status") != "completed_exploratory_capacity_screen"
            or report.get("completed_shapes") != [512]
            or _evidence_mismatch(report.get("stages", {}).get(name), manifest["stages"][name], "report.stage") is not None
            or _evidence_mismatch(origin["execution_origin"]["environment"], report.get("environment"), "record.execution_origin.environment") is not None
            or _evidence_mismatch(report.get("gates"), manifest["stages"][name].get("gates"), "report.gates") is not None):
        raise ValueError("B512 historical report/manifest/origin mismatch")
    peak = origin["provenance"].get("host_peak_RSS_bytes")
    if peak != B512_PEAK_RSS_BYTES:
        raise ValueError("B512 historical actual process peak RSS mismatch")
    report_peak = report.get("linux_ru_maxrss_peak_bytes") or report.get("peak_rss_bytes")
    if type(report_peak) is not int or report_peak < peak:
        raise ValueError("B512 historical report peak RSS missing/inconsistent")
    for key in ("actual_science_source_sha256", "execution_source_compatibility"):
        if origin["source_identity"].get(key) != expected_identity[key]:
            raise ValueError(f"B512 source certificate mismatch: {key}")
    origin.pop("timing")
    origin.update(role="historical numerical/schema/compiler-temp/RSS evidence only; excluded from current timing curve",
        report_sha256=hashlib.sha256(report_data).hexdigest(), historical_process_peak_RSS_bytes=peak)
    return origin


def fixed1024_host_screen(current_b512_peak=None):
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:"))
    historical_projection = B512_PEAK_RSS_BYTES * 2
    projected = max(historical_projection, 2 * (current_b512_peak or 0))
    required = projected + 8 * 1024**3
    result = {"MemAvailable_bytes": available, "historical_B512_process_peak_RSS_bytes": B512_PEAK_RSS_BYTES,
        "historical_total1024_projection_bytes": historical_projection,
        "current_fixed1024_B512_process_peak_RSS_bytes": current_b512_peak,
        "conservative_required_bytes": required,
        "limitation": "max historical512-to1024 projection/current B512 process peak times2 plus8GiB; conservative planning, not measured peak or reservation"}
    if available < required:
        raise MemoryError(f"fixed1024 host memory admission failed: {result}")
    return result


def fixed1024_decision(size, records, stages, remaining_seconds, estimated_bytes, free_bytes):
    reasons = []
    if size == 1024:
        completed = stages.get("capacity_B512_fixed1024_v1", {})
        if 512 not in records or completed.get("status") != "complete":
            reasons.append("completed new fixed1024 B512 record and atomic checkpoint required")
        gates = completed.get("gates", [])
        if (completed.get("strict_numerical_passed") is not True or len(gates) != 1027
                or any(item.get("passed") is not True for item in gates)):
            reasons.append("new fixed1024 B512 strict numerical acceptance requires all1027 gates passed; finite-reference exploratory failure cannot admit B1024")
    if remaining_seconds < 2700:
        reasons.append("at least2700s remaining required; no completion guarantee")
    if estimated_bytes is None or free_bytes < estimated_bytes + 2 * 1024**3:
        reasons.append("scaled actual compiler buffers/full1024 collection plus2GiB physical slack required")
    return {"shape": size, "permitted": not reasons, "reasons": reasons,
        "remaining_seconds": remaining_seconds, "minimum_remaining_seconds": 2700,
        "estimated_device_bytes": estimated_bytes, "free_device_bytes": free_bytes,
        "requires_performance_improvement": False,
        "policy": "fixed1024_resource_only_2700s_2GiB_GPU_8GiB_host_slack",
        "scope": "conditional capacity admission; 2700s is a minimum, not completion/OOM guarantee"}


def device_screen(size, resident_bytes, output_bytes, memory_records, scale, total_lanes=TOTAL_LANES):
    physical = scale.visible_gpu_memory()
    estimated = memory_estimate(size, resident_bytes, output_bytes, memory_records)
    result = {"shape": size, f"resident_{total_lanes}_input_bytes": resident_bytes, f"full_{total_lanes}_output_bytes": output_bytes,
        "estimated_device_bytes": estimated, "physical_gpu": physical,
        "limitation": "scaled actual compiler temporary buffers plus full collection overlap; estimate is not a measured peak or OOM guarantee"}
    if total_lanes == 1024:
        result["input_byte_scope"] = "actual pure-helper host stacks for the complete1024 collection, including shared-static layout"
    if (not physical.get("applicable") or physical["total_bytes"] < 30 * 1024**3
            or estimated is None or physical["free_bytes"] < estimated + 2 * 1024**3):
        raise MemoryError(f"capacity shape memory preflight failed: {result}")
    return result


def host_screen(size, old_peak):
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    required = old_peak * size // 64 + 8 * 1024**3
    result = {"MemAvailable_bytes": available, "old_B64_job_peak_RSS_bytes": old_peak,
        "lane_scaled_required_bytes": required,
        "limitation": "old whole-benchmark RSS scaled by batch lanes plus8GiB; planning screen, not measured stage RSS or reservation"}
    if available < required:
        raise MemoryError(f"host memory admission failed: {result}")
    return result


def measure_shape(size, args, recorder, lanes, second, reference, scale, memory_records, total_lanes=TOTAL_LANES):
    """Compile once, measure the full fixed collection, return atomic-stage payload."""
    if len(lanes) != total_lanes or len(second) != total_lanes or total_lanes % size:
        raise ValueError("aligned complete fixed collection required")
    base, jax, np = scale.base, scale.jax, scale.np
    label = f"capacity_B{size}"
    gate_start, started = len(recorder.result["gates"]), time.perf_counter()
    rss_before = base.rss_bytes()
    chunks = [(lanes[start:start + size], second[start:start + size]) for start in range(0, total_lanes, size)]
    host = [(base.batch.stack_lanes(first), base.stack_second(first, other)) for first, other in chunks]
    host_stack_seconds = time.perf_counter() - started
    resident_bytes, output_bytes = scale.tree_bytes(host), scale.tree_bytes(reference) * (total_lanes // UNIQUE_POINTS)
    host_preflight = (fixed1024_host_screen(memory_records[512].get("current_process_peak_RSS_bytes"))
        if total_lanes == 1024 else host_screen(size, memory_records[64]["old_benchmark_peak_RSS_bytes"]))
    preflight = device_screen(size, resident_bytes, output_bytes, memory_records, scale, total_lanes)
    started = time.perf_counter()
    pairs = jax.device_put(host)
    base.sync(pairs)
    transfer_seconds = time.perf_counter() - started
    # Pure source-backed factory; no process logic copied or approximated here.
    base.driver._COMPILED_SECHIBA_SCAN_CACHE.clear()
    executable = base.lower_compile(base.batch.make_batch_transition(chunks[0][0]), pairs[0][0], label, recorder)
    memory = scale.executable_memory(executable)
    memory_records[size] = ({**memory, "temp_size_in_bytes": max(memory["temp_size_in_bytes"], memory_records[size]["temp_size_in_bytes"])}
        if total_lanes == 1024 and size in memory_records and memory.get("available") is True else memory)
    device_screen(size, resident_bytes, output_bytes, memory_records, scale, total_lanes)

    def run_collection(wrapper=False):
        outputs = []
        for index, (first, other) in enumerate(chunks):
            pair = jax.device_put((base.batch.stack_lanes(first), base.stack_second(first, other))) if wrapper else pairs[index]
            value = base.run_two(executable, *pair)
            outputs.append(jax.device_get(value) if wrapper else value)
        base.sync(outputs)
        return outputs

    def collect(outputs):
        return jax.tree_util.tree_map(lambda *parts: np.concatenate(parts, axis=0), *outputs)

    started = time.perf_counter()
    initial_chunks = run_collection()
    first_seconds = time.perf_counter() - started
    output = collect(jax.device_get(initial_chunks))
    # Release device copies before the ten warm collections; complete host outputs remain.
    del initial_chunks
    hard = []
    for index in range(total_lanes):
        try:
            gate(recorder, f"{label}_lane_{index}_source_{index % UNIQUE_POINTS}_GPU_reference",
                jax.tree_util.tree_map(lambda value: value[index], output),
                jax.tree_util.tree_map(lambda value: value[index % UNIQUE_POINTS], reference), base.compare, cross_backend=True)
        except AssertionError as exc:
            hard.append(str(exc))
    if hard:
        raise AssertionError("hard per-lane reference failures; no timings admitted: " + "; ".join(hard))
    core, wrapper = [], []
    for _ in range(args.warm_repeats):
        started = time.perf_counter()
        core_outputs = run_collection()
        core.append(time.perf_counter() - started)
        del core_outputs
        started = time.perf_counter()
        wrapped_chunks = run_collection(wrapper=True)
        wrapper.append(time.perf_counter() - started)
    final_output = collect(wrapped_chunks)
    gate(recorder, label + "_frozen_reset_repeat", final_output, output, base.compare)
    # Strict same-backend permutation and replacement preserve every potentially active lane.
    first, other = chunks[0]
    order = list(reversed(range(size)))
    permuted = base.run_two(executable, base.batch.stack_lanes([first[i] for i in order]),
        base.stack_second([first[i] for i in order], [other[i] for i in order]))
    gate(recorder, label + "_permutation", jax.device_get(permuted),
        jax.tree_util.tree_map(lambda value: value[np.asarray(order)], output), base.compare)
    changed = first[:-1] + [first[0]]
    replaced = base.run_two(executable, base.batch.stack_lanes(changed), base.stack_second(changed, other[:-1] + [other[0]]))
    gate(recorder, label + "_replacement_isolation", jax.device_get(replaced),
        jax.tree_util.tree_map(lambda value: value[np.asarray(list(range(size - 1)) + [0])], output), base.compare)
    measurements = {"core": lane_metrics(core, total_lanes), "wrapper": lane_metrics(wrapper, total_lanes)}
    for metric in measurements.values():
        metric["warm_cost_scenarios"] = {name: {
            "GPU_card_hours": metric["median_seconds"] / 3600,
            "auxiliary_CPU_count_hours": 4 * metric["median_seconds"] / 3600,
            "CNY": (2.2 + 4 * cpu_price) * metric["median_seconds"] / 3600,
            "scope": f"measured warm same{total_lanes}-lane collection; excludes startup/compile/checkpointIO; scenario pricing not billing"}
            for name, cpu_price in (("primary", .05), ("conservative", .07))}
    gates = recorder.result["gates"][gate_start:]
    return {"outputs": final_output, "measurements": measurements}, {
        "measurement_status": "complete", "evidence_class": "exploratory_capacity_performance",
        "scientific_acceptance": False, "performance_label": LABEL,
        "strict_numerical_passed": all(item["passed"] is True for item in gates), "gates": gates,
        "timing": measurements, "startup": {"host_stack_seconds": host_stack_seconds,
            "device_transfer_seconds": transfer_seconds, f"first_full{total_lanes}_call_seconds": first_seconds,
            "lower_seconds": recorder.result["timing"][label + "_lower_seconds"],
            "compile_seconds": recorder.result["timing"][label + "_compile_seconds"]},
        "provenance": {"workload_kind": "capacity_screen_repeated_FULL64", "total_lanes": total_lanes,
            "unique_real_points": UNIQUE_POINTS, "lane_days": total_lanes * 14, "unique_point_days": UNIQUE_POINTS * 14,
            "chunk_size": size, "chunks_per_collection": len(chunks), "blocks": [[2, 8], [9, 15]],
            "half_hours_per_day": 48, "full_outputs": True, "full_two_block_workload": True,
            "executable_memory_analysis": memory, "memory_preflight": preflight,
            "host_memory_preflight": host_preflight,
            "post_measurement_GPU_memory": scale.visible_gpu_memory(), "host_peak_RSS_bytes": base.linux_ru_maxrss_bytes(),
            "host_RSS_bytes_before": rss_before, "host_RSS_bytes_after": base.rss_bytes(),
            "RSS_peak_scope": "process cumulative peak through this stage; before/after are actual stage boundary snapshots",
            "VRAM_scope": "actual selected-card preflight/post-measurement snapshots; not a measured peak",
            "server_resources": recorder.result["server_resources"]},
        "execution_origin": {"available": True, "environment": recorder.result["environment"], "pid": os.getpid()}}


def benchmark(args, recorder, lanes, second, scale):
    if getattr(args, "fixed_1024", False):
        return benchmark_fixed1024(args, recorder, lanes, second, scale)
    from scripts.dev.global_batch_stage_store import StageStore

    base = scale.base
    start_epoch = float(os.environ.get("ORCHJAX_GPU_WINDOW_START_EPOCH", time.time()))
    screen = scale.screen_inputs(lanes, second)
    expanded, others, origins = capacity_layout(lanes, second)
    identity, reference, reference_origin, old_record, analysis = admit_numerical_sources(
        args, lanes, second, scale, recorder.result["environment"])
    continuation = getattr(args, "continue_256_512", False)
    only512 = getattr(args, "continue_512", False)
    identity.update(contract=CONTRACT, precision_policy=scale.PRECISION_POLICY, validation_mode="exploratory",
        workload_kind="capacity_screen_repeated_FULL64", total_lanes=TOTAL_LANES, lane_origins=origins,
        unique_real_points=UNIQUE_POINTS, candidate_shapes=[128, 256], conditional_512=args.conditional_512,
        optional_B64_baseline=args.optional_b64_baseline, conditional_policy="core_and_wrapper_5percent_30min_2GiBslack")
    for source in (args.full_reference_dir, args.memory_baseline_dir, args.prepared_input.parent):
        StageStore.validate_source_destination(source, args.output_dir)
    identity["numerical_reference_sha256"] = reference_origin["sha256"]
    baseline_timing, baseline_origin = None, None
    b256_timing, b256_origin = None, None
    if continuation or only512:
        baseline_identity = {**identity, "candidate_shapes": [128, 256], "conditional_512": True,
                             "optional_B64_baseline": False}
        baseline_timing, baseline_origin = import_b128_baseline(
            args.b128_baseline_dir, baseline_identity, args.output_dir, scale.jax, reference=reference)
        identity.update(contract=CONTINUATION_CONTRACT, candidate_shapes=[256, 512],
            continuation_B128_sha256=B128_SHA256, conditional_512=True, optional_B64_baseline=False,
            conditional_policy="scale_curve_30min_B256compiler_memory_2GiB_GPU_8GiB_host_slack")
        if only512:
            b256_timing, b256_origin = import_b256_baseline(
                args.b256_baseline_dir, identity, args.output_dir, scale.jax, reference=reference)
            identity.update(contract=B512_CONTRACT, candidate_shapes=[512],
                continuation_B256_sha256=B256_SHA256, budget_seconds=2700,
                conditional_policy="B512_only_30min_imported_B256compiler_memory_2GiB_GPU_8GiB_host_slack")
    # Historical admission uses its exact original schema above. The new
    # execution records both actual bytes and the narrowly certified identity.
    from scripts.dev.global_source_compatibility import source_identity

    source = source_identity(ROOT)
    if "science_sha256" in identity and identity["science_sha256"] != source["science_sha256"]:
        raise ValueError("capacity execution source changed during baseline admission")
    identity.update(source)
    store = StageStore(args.output_dir, identity)
    recorder.sentinel_paths = base.roughheight_sentinel_contract(lanes[0])
    recorder.result.update(stages=store.manifest["stages"], compatibility_audit=screen, lane_origins=origins,
        reference_origin=reference_origin, memory_baseline_origin={"directory": str(args.memory_baseline_dir),
            "checkpoint_sha256": old_record["sha256"], "analysis": analysis}, scientific_acceptance=False)
    recorder.result["execution_source_identity"] = {key: identity[key] for key in (
        "science_sha256", "actual_science_source_sha256", "execution_source_compatibility") if key in identity}
    base.legacy_factory(args.baseline_ref)
    memory_records, records = {64: analysis}, {}
    if continuation or only512:
        records[128] = baseline_timing
        recorder.result["imported_B128_baseline"] = baseline_origin
    if only512:
        records[256] = b256_timing
        recorder.result["imported_B256_baseline"] = b256_origin
        # Old B64 remains the historical host RSS origin only. The B512 device
        # guard uses sealed B256 actual compiler buffers, without a baseline compile.
        memory_records[64] = {**analysis, "available": False}
        memory_records[256] = b256_origin["provenance"]["executable_memory_analysis"]

    def run(size):
        (value, record), reused = store.run(f"capacity_B{size}_fixed512_v1",
            lambda: measure_shape(size, args, recorder, expanded, others, reference, scale, memory_records))
        records[size] = value["measurements"]
        memory_records[size] = record.get("provenance", {}).get("executable_memory_analysis", memory_records.get(size, {}))
        recorder.phase(f"capacity_B{size}_sealed", reused=reused)
        recorder.flush()
        # The sealed host snapshot survives cleanup; no executable is a prerequisite.
        base.driver._COMPILED_SECHIBA_SCAN_CACHE.clear()
        scale.jax.clear_caches()
        gc.collect()
        return record

    for size in (() if only512 else ((256,) if continuation else (128, 256))):
        run(size)
    if args.conditional_512 or continuation or only512:
        snapshot = scale.visible_gpu_memory()
        resident = scale.tree_bytes(([lane.arguments for lane in expanded], others))
        decision = conditional_512_decision(records, args.budget_seconds - (time.time() - start_epoch),
            memory_estimate(512, resident, scale.tree_bytes(reference) * 8, memory_records), snapshot["free_bytes"], continuation=continuation or only512)
        if only512:
            decision["compiler_memory_origin"] = {"shape": 256, "sha256": B256_SHA256,
                "source_stage": b256_origin["source_stage"], "actual_analysis": memory_records[256]}
        if memory_records.get(256, {}).get("available") is not True:
            decision["permitted"] = False
            decision["reasons"].append("actual B256 compiler memory analysis required")
        try:
            decision["host_memory"] = host_screen(512, analysis["old_benchmark_peak_RSS_bytes"])
        except MemoryError as exc:
            decision["permitted"] = False
            decision["reasons"].append(str(exc))
        recorder.result["conditional_512_decision"] = decision
        recorder.flush()
        if decision["permitted"]:
            run(512)
        elif only512:
            recorder.result["completed_shapes"] = []
            recorder.result["comparison_shapes_including_imported_baseline"] = [128, 256]
            raise MemoryError(f"B512-only resource admission failed: {decision}")
    if args.optional_b64_baseline:
        remaining = args.budget_seconds - (time.time() - start_epoch)
        if remaining >= 1200:
            run(64)
        else:
            recorder.result["optional_B64_skipped"] = "less than20min remaining; candidates retained"
    recorder.result["completed_shapes"] = [size for size in records if not (
        (continuation and size == 128) or (only512 and size in (128, 256)))]
    if continuation or only512:
        recorder.result["comparison_shapes_including_imported_baseline"] = list(records)


def benchmark_fixed1024(args, recorder, lanes, second, scale):
    """New 1024-lane timing curve; historical512 supplies no timing sample."""
    from scripts.dev.global_batch_stage_store import StageStore
    from scripts.dev.global_server_reference import load_reference
    from scripts.dev.global_source_compatibility import source_identity

    base = scale.base
    start_epoch = float(os.environ.get("ORCHJAX_GPU_WINDOW_START_EPOCH", time.time()))
    screen = scale.screen_inputs(lanes, second)
    expanded, others, origins = capacity_layout(lanes, second, 1024)
    identity = base.stage_identity(args, lanes, second, recorder.result["environment"])
    identity.update(contract=CONTRACT, precision_policy=scale.PRECISION_POLICY)
    reference, reference_origin = load_reference(args.full_reference_dir, identity)
    expected = b512_evidence_identity(identity, origins, reference_origin["sha256"])
    origin = import_b512_evidence(args.b512_evidence_dir, expected, args.output_dir, scale.jax, reference=reference)
    for directory in (args.full_reference_dir, args.prepared_input.parent):
        StageStore.validate_source_destination(directory, args.output_dir)
    source = source_identity(ROOT)
    if source["science_sha256"] != identity["science_sha256"]:
        raise ValueError("capacity execution source changed during numerical admission")
    identity.update(source)
    identity.update(contract=FIXED1024_CONTRACT, validation_mode="exploratory",
        workload_kind="capacity_screen_repeated_FULL64", total_lanes=1024, lane_origins=origins,
        unique_real_points=64, candidate_shapes=[512, 1024], warm_repeats=5, budget_seconds=5400,
        conditional_512=False, conditional_1024=True, optional_B64_baseline=False,
        historical_B512_evidence_sha256=B512_SHA256, numerical_reference_sha256=reference_origin["sha256"],
        conditional_policy="fixed1024_resource_only_2700s_2GiB_GPU_8GiB_host_slack")
    store = StageStore(args.output_dir, identity)
    recorder.sentinel_paths = base.roughheight_sentinel_contract(lanes[0])
    recorder.result.update(stages=store.manifest["stages"], compatibility_audit=screen,
        lane_origins=origins, reference_origin=reference_origin, imported_B512_evidence=origin,
        scientific_acceptance=False, completed_shapes=[], comparison_shapes=[])
    recorder.result["execution_source_identity"] = source
    base.legacy_factory(args.baseline_ref)
    historical_memory = origin["provenance"]["executable_memory_analysis"]
    memory_records, records = {512: historical_memory.copy()}, {}
    resident = scale.tree_bytes(([lane.arguments for lane in expanded], others))
    full_output = scale.tree_bytes(reference) * 16
    for size in (512, 1024):
        physical = scale.visible_gpu_memory()
        decision = fixed1024_decision(size, records, store.manifest["stages"],
            args.budget_seconds - (time.time() - start_epoch),
            memory_estimate(size, resident, full_output, memory_records), physical["free_bytes"])
        decision["compiler_memory_origin"] = {"historical_B512_sha256": B512_SHA256,
            "historical_actual_analysis": historical_memory, "current_memory_records": memory_records.copy()}
        decision.update(resident_1024_input_bytes_upper_bound=resident, full_1024_output_bytes=full_output,
            input_byte_scope="conservative pre-stack raw lane-list upper bound; counts repeated shared-static leaves; actual-stack bytes recorded in stage memory_preflight")
        if not physical.get("applicable") or physical.get("total_bytes", 0) < 30 * 1024**3:
            decision["permitted"] = False
            decision["reasons"].append("actual selected >=30GiB GPU memory snapshot required")
        try:
            decision["host_memory"] = fixed1024_host_screen(memory_records[512].get("current_process_peak_RSS_bytes"))
        except MemoryError as exc:
            decision["permitted"] = False
            decision["reasons"].append(str(exc))
        recorder.result[f"fixed1024_B{size}_decision"] = decision
        recorder.flush()
        if not decision["permitted"]:
            if size == 512:
                raise MemoryError(f"initial fixed1024 B512 resource admission failed: {decision}")
            recorder.result["conditional_B1024_skipped"] = decision["reasons"]
            break
        name = f"capacity_B{size}_fixed1024_v1"
        (value, record), reused = store.run(name,
            lambda size=size: measure_shape(size, args, recorder, expanded, others, reference, scale, memory_records, 1024))
        records[size] = value["measurements"]
        measured = record["provenance"]["executable_memory_analysis"]
        # Historical and current actual B512 compiler temporaries both bound the
        # next shape. RSS remains explicitly a cumulative process peak.
        memory_records[size] = {**measured,
            "temp_size_in_bytes": max(measured["temp_size_in_bytes"], memory_records.get(size, {}).get("temp_size_in_bytes", 0)),
            "current_process_peak_RSS_bytes": record["provenance"]["host_peak_RSS_bytes"]}
        recorder.result.update(completed_shapes=list(records), comparison_shapes=list(records))
        recorder.phase(f"capacity_B{size}_fixed1024_sealed", reused=reused)
        recorder.flush()
        base.driver._COMPILED_SECHIBA_SCAN_CACHE.clear()
        scale.jax.clear_caches()
        gc.collect()


def main():
    args = parser().parse_args()
    if args.fixed_1024:
        if args.b512_evidence_dir is None:
            raise ValueError("fixed1024 requires explicit sealed B512 evidence directory")
        if args.optional_b64_baseline or args.conditional_512 or args.b128_baseline_dir or args.b256_baseline_dir:
            raise ValueError("fixed1024 forbids legacy baseline/timing modes and optional B64")
        if args.admit_inputs:
            raise ValueError("fixed1024 reuses original FULL64 target-input admission; no repeated input re-audit")
    elif args.b512_evidence_dir is not None:
        raise ValueError("B512 evidence directory requires fixed1024 mode")
    if (args.continue_256_512 or args.continue_512) and (args.b128_baseline_dir is None or args.optional_b64_baseline):
        raise ValueError("B256/B512 continuation requires explicit sealed B128 baseline and forbids optional B64")
    if args.continue_512 and args.b256_baseline_dir is None:
        raise ValueError("B512-only continuation requires explicit sealed B256 baseline")
    if args.b256_baseline_dir is not None and not args.continue_512:
        raise ValueError("B256 baseline directory requires B512-only continuation mode")
    if args.b128_baseline_dir is not None and not (args.continue_256_512 or args.continue_512):
        raise ValueError("B128 baseline directory requires continuation mode")
    budget = 2700 if args.continue_512 else 5400
    if args.warm_repeats != 5 or args.budget_seconds != budget:
        raise ValueError(f"fixed five repeats and proposed {budget}s window contract required")
    if args.dry_run:
        from scripts.dev.global_source_compatibility import source_identity

        if args.fixed_1024:
            print(json.dumps({"status": "dry_run", "model_execution": False, "JAX_imported": "jax" in sys.modules,
                "execution_source_identity": source_identity(ROOT), "contract": FIXED1024_CONTRACT,
                "candidates": [512, 1024], "total_lanes": 1024, "unique_real_points": 64,
                "lane_days": 14336, "unique_point_days": 896, "warm_repeats": 5, "budget_seconds": 5400,
                "B512_evidence_dir": str(args.b512_evidence_dir), "historical_B512_evidence_sha256": B512_SHA256,
                "historical_B512_timing_is_curve_origin": False, "conditional_1024": True,
                "minimum_remaining_seconds_each_stage": 2700, "B1024_requires_speedup": False,
                "conditional_1024_scope": "2700s minimum remaining and sealed new B512 required; no completion guarantee",
                "historical_B512_host_RSS_bytes": B512_PEAK_RSS_BYTES,
                "initial_conservative_host_required_bytes": B512_PEAK_RSS_BYTES * 2 + 8 * 1024**3,
                "scientific_acceptance": False, "optional_B64": False}))
            return 0
        print(json.dumps({"status": "dry_run", "model_execution": False, "JAX_imported": "jax" in sys.modules,
            "execution_source_identity": source_identity(ROOT),
            "contract": B512_CONTRACT if args.continue_512 else (CONTINUATION_CONTRACT if args.continue_256_512 else CONTRACT),
            "candidates": [512] if args.continue_512 else ([256, 512] if args.continue_256_512 else [128, 256]),
            "conditional_512": args.conditional_512 or args.continue_256_512 or args.continue_512,
            "B512_requires_speedup": not (args.continue_256_512 or args.continue_512), "B128_baseline_dir": str(args.b128_baseline_dir),
            "B256_baseline_dir": str(args.b256_baseline_dir), "budget_seconds": budget,
            "total_lanes": TOTAL_LANES, "unique_real_points": UNIQUE_POINTS, "lane_days": 7168,
            "unique_point_days": 896, "scientific_acceptance": False, "optional_B64": args.optional_b64_baseline}))
        return 0
    if args.admit_inputs:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_ENABLE_X64"] = "true"
    from scripts.dev import benchmark_global_scale as scale
    scale.jax.config.update("jax_enable_x64", True)
    if args.admit_inputs:
        lanes, second = scale.read_prepared(args.prepared_input, args.config)
        screen = scale.screen_inputs(lanes, second)
        expanded, others, origins = capacity_layout(lanes, second)
        source_identity, reference, reference_origin, old_record, analysis = admit_numerical_sources(
            args, lanes, second, scale, scale.base.metadata(args))
        # Stack every repeated coordinate using the same pure schema helper, no transition.
        shape_checks = []
        for size in (128, 256, 512):
            first = scale.base.batch.stack_lanes(expanded[:size])
            other = scale.base.stack_second(expanded[:size], others[:size])
            count = 0
            for arguments, source_arguments in ((first, [lane.arguments for lane in expanded[:size]]), (other, others[:size])):
                for group_index, group in enumerate(arguments):
                    if group_index == 2:
                        continue  # Declared shared static group remains unbatched.
                    leaves = scale.jax.tree_util.tree_leaves(group)
                    sources = [scale.jax.tree_util.tree_leaves(source[group_index]) for source in source_arguments]
                    for leaf_index, leaf in enumerate(leaves):
                        if scale.np.asarray(leaf).shape[0] != size:
                            raise ValueError(f"capacity dynamic leaf leading shape mismatch: B{size}")
                        for index in range(size):
                            original = scale.np.asarray(sources[index][leaf_index])
                            stacked = scale.np.asarray(leaf)[index]
                            if (stacked.shape != original.shape or stacked.dtype != original.dtype
                                    or stacked.tobytes(order="C") != original.tobytes(order="C")):
                                raise ValueError(f"capacity native input byte identity changed: B{size}/lane{index}")
                        count += 1
            shape_checks.append({"batch_shape": size, "chunks_for_fixed512": 512 // size,
                "dynamic_leaves_checked": count, "every_stacked_lane_native_byte_identity_verified": True,
                "resident_512_input_bytes": scale.tree_bytes((first, other)) * (512 // size),
                "full_512_output_bytes": scale.tree_bytes(reference) * 8,
                "estimated_device_bytes": memory_estimate(size, scale.tree_bytes((first, other)) * (512 // size),
                    scale.tree_bytes(reference) * 8, {64: analysis}),
                "required_GPU_slack_bytes": 2 * 1024**3,
                "estimated_host_required_bytes": analysis["old_benchmark_peak_RSS_bytes"] * size // 64 + 8 * 1024**3,
                "memory_projection_scope": "scaled old actual compiler/RSS planning only; no selected GPU memory queried"})
        print(json.dumps({"status": "input_identity_shape_admitted", "model_execution": False, "compiled": False,
            "source_audit": screen, "total_lanes": len(expanded), "second_blocks": len(others), "lane_origins": origins,
            "capacity_shape_checks": shape_checks,
            "source_identity": source_identity, "FULL64_reference_origin": reference_origin,
            "B64_memory_origin": {"directory": str(args.memory_baseline_dir),
                "declared_checkpoint_sha256": old_record["sha256"], "actual_old_analysis": analysis},
            "unique_real_points": UNIQUE_POINTS, "lane_days": 7168, "unique_point_days": 896}))
        return 0
    resources = json.loads(os.environ["ORCHJAX_SERVER_RESOURCE_PLAN"])
    if resources["allocated_CPU_request_count"] != 4:
        raise ValueError("exact four auxiliary Slurm CPUs required")
    if (scale.jax.__version__ != "0.4.38" or len(scale.jax.devices()) != 1 or scale.jax.devices()[0].platform != "gpu"
            or "V100" not in scale.jax.devices()[0].device_kind):
        raise ValueError("accepted canonical JAX0.4.38 single-GPU runtime required")
    from scripts.dev.probe_global_batch_cache import configure_cache
    configure_cache("off", None)
    recorder = scale.base.Recorder(args.output_dir)
    recorder.result.update(environment=scale.base.metadata(args), server_resources=resources)
    previous = {}
    def stopped(number, frame):
        raise InterruptedError(f"signal{number}; completed capacity snapshots retained")
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, stopped)
    code = 0
    try:
        available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
        if available < 24 * 1024**3:
            raise MemoryError("at least24GiB host MemAvailable required")
        recorder.result["host_MemAvailable_bytes_at_start"] = available
        lanes, second = scale.read_prepared(args.prepared_input, args.config)
        benchmark(args, recorder, lanes, second, scale)
        recorder.result["status"] = "completed_exploratory_capacity_screen"
    except (InterruptedError, KeyboardInterrupt) as exc:
        code = 2
        recorder.result.update(status="stopped", message=str(exc))
    except Exception as exc:
        code = 1
        recorder.result.update(status="failed", message=str(exc), traceback=traceback.format_exc())
    finally:
        recorder.stop.set()
        start_epoch = float(os.environ.get("ORCHJAX_GPU_WINDOW_START_EPOCH", time.time()))
        recorder.result["observed_window_wall_seconds"] = time.time() - start_epoch
        recorder.result["whole_job_accounting"] = "external scheduler terminal accounting required; never divide whole job by one lane-day denominator"
        recorder.flush()
        for number, handler in previous.items():
            signal.signal(number, handler)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
