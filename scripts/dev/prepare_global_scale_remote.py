"""Rehydrate Linux B64 contexts/forcing from audited local Day1 states.

Never generates a cold Day1 state or compiles the physical transition. The four
remote anchors retain their original exact numerical inputs and closures.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dev import benchmark_global_scale as scale
from scripts.dev import prepare_global_frozen_arguments as frozen_io
from scripts.dev import prepare_global_scale_inputs as producer

base, batch, driver = scale.base, scale.base.batch, scale.base.driver


def values_digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_cold_source(point, state, source_values, record, config_bytes, identity_hash):
    expected = {**source_values, "VEGETATION_FILE": record["vegetation_file_path_only_mapping"]["local_value"]}
    mapping = record["vegetation_file_path_only_mapping"]
    if mapping["source_runtime_used_run_def_value"] != source_values["VEGETATION_FILE"]:
        raise ValueError(f"unauthorized source vegetation mapping: {point}")
    if values_digest(expected) != record["run_def_values_sha256"]:
        raise ValueError(f"cold source run.def identity changed: {point}")
    computed = producer._cold_identity_hash((expected, config_bytes, record["reference_dir"]))
    if (hashlib.sha256(config_bytes).hexdigest() != record["config_sha256"]
            or computed != record["cold_identity_sha256"] or computed != identity_hash):
        raise ValueError(f"cold source config/reference identity changed: {point}")
    if state.tstep != 47 or producer._state_digest(state) != record["cold_state_numeric_sha256"]:
        raise ValueError(f"cold Day1 numeric identity changed: {point}")
    return state


def rehydrate(args):
    started = time.perf_counter()
    manifest, run_defs, run_shas = producer._read_inventory(args.inventory_tar)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    ids = selection["selections"]["64"]
    if len(ids) != 64 or len(set(ids)) != 64 or tuple(ids[:4]) != base.POINTS:
        raise ValueError("remote rehydration requires the declared 64 unique selected points")
    source_report = json.loads(args.preparation_audit.read_text(encoding="utf-8"))
    if (source_report["status"] != "prepared" or source_report["selected_points"] != ids
            or source_report["prepared_point_count"] != 64
            or source_report["inventory_manifest_sha256"] != manifest["_sha256"]):
        raise ValueError("local preparation audit is not the completed exact FULL64 contract")
    config_bytes = args.config.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != source_report["config_sha256"]:
        raise ValueError("remote configuration differs from local audited science")
    assets = producer._verify_selected_assets(args.asset_root, ids, manifest, run_defs, run_shas)
    for point in ids:
        for key in ("runtime_run_def_sha256", "reference_file_sha256"):
            if assets[point][key] != source_report["asset_inputs"][point][key]:
                raise ValueError(f"remote source asset differs from local audit: {point}/{key}")
    cold_data = args.cold_states.read_bytes()
    if hashlib.sha256(cold_data).hexdigest() != source_report["cold_states_sha256"]:
        raise ValueError("consolidated cold state transport checksum changed")
    cold = scale.PreparedUnpickler(io.BytesIO(cold_data)).load()
    if producer._sha256_file(args.cold_identities) != source_report["cold_identities_sha256"]:
        raise ValueError("cold identity sidecar transport checksum changed")
    cold_identities = json.loads(args.cold_identities.read_text(encoding="utf-8"))
    if set(cold) != set(ids[4:]):
        raise ValueError("cold state consolidated point set differs from selected new60")
    if set(cold_identities) != set(cold):
        raise ValueError("cold identity sidecar point set differs from consolidated states")
    records = {item["landpoint_id"]: item for item in source_report["new_point_records"]}
    if set(records) != set(ids[4:]):
        raise ValueError("local audit does not cover every new60 point exactly")
    frozen_points, frozen_sha256 = frozen_io.load_frozen(
        args.frozen_arguments, args.frozen_arguments_sha256, source_report,
        producer._sha256_file(args.preparation_audit))
    vegetation = (args.data_root / "INPUTDIR_ZZ/PFT1860_mangr_025deg.nc").resolve()
    vegetation_sha = producer._sha256_file(vegetation)
    source_vegetation_proofs = {}
    for source_path in sorted({run_defs[point]["VEGETATION_FILE"] for point in ids[4:]}):
        original = Path(source_path)
        if not original.is_absolute():
            raise ValueError("remote archived vegetation source path is not absolute")
        if original.exists() and producer._sha256_file(original) != vegetation_sha:
            raise ValueError("available remote source and canonical vegetation assets differ")
        source_vegetation_proofs[source_path] = {"source_path_available": original.exists(),
            "remote_target": vegetation.as_posix(), "remote_target_sha256": vegetation_sha,
            "proof": "local actual source asset hash equals canonical remote target hash; archived path is provenance",
            "available_historical_source_sha256": vegetation_sha if original.exists() else None}
    for point in ids[4:]:
        if records[point]["vegetation_file_path_only_mapping"]["local_file_sha256"] != vegetation_sha:
            raise ValueError(f"remote vegetation file content differs: {point}")
        validate_cold_source(point, cold[point], run_defs[point], records[point], config_bytes, cold_identities[point])
    lanes, second = scale.read_prepared(args.anchor_prepared, args.config)
    if tuple(item.landpoint_id for item in lanes) != base.POINTS or len(second) != 4:
        raise ValueError("remote anchor source is not exactly original four")
    if {item.context.run_def_values["VEGETATION_FILE"] for item in lanes} != {vegetation.as_posix()}:
        raise ValueError("normalized remote vegetation target differs from canonical anchor run.def")
    anchor_ids = [base.semantic_reference_identity(args, item, other)
                  for item, other in zip(lanes, second, strict=True)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "remote-preparation-audit.json"
    report = {"status": "rehydrating", "selected_points": ids,
              "environment": {"python": sys.version, "jax": scale.jax.__version__,
                              "numpy": producer.np.__version__,
                              "jax_enable_x64": bool(scale.jax.config.jax_enable_x64),
                              "backends": sorted({device.platform for device in scale.jax.devices()}),
                              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
              "cold_Day1_generated": False, "physical_transition_compiled": False,
              "source_audit_sha256": producer._sha256_file(args.preparation_audit),
              "cold_states_sha256": hashlib.sha256(cold_data).hexdigest(),
              "cold_identities_sha256": producer._sha256_file(args.cold_identities),
              "frozen_arguments_sha256": frozen_sha256,
              "new60_numerical_input_origin": "exact original local prepared host NumPy arguments; Linux source contexts independently rebuilt",
              "anchor_prepared_sha256": producer._sha256_file(args.anchor_prepared),
              "anchor_canonical_inputs": anchor_ids, "asset_inputs": assets,
              "vegetation_mapping_byte_equivalence": source_vegetation_proofs,
              "config_sha256": hashlib.sha256(config_bytes).hexdigest(), "new_point_records": []}
    producer._atomic_json(report_path, report)
    for point in ids[4:]:
        if time.perf_counter() - started >= args.max_seconds:
            raise TimeoutError("bounded CPU rehydration duration exhausted; no GPU admission")
        point_started = time.perf_counter()
        state = validate_cold_source(point, cold[point], run_defs[point], records[point], config_bytes, cold_identities[point])
        values = {**run_defs[point], "VEGETATION_FILE": vegetation.as_posix()}
        run_def = args.output_dir / "run_defs" / point / "used_run.def"
        producer.write_materialized_run_def(values, run_def)
        if producer.read_run_def_values(run_def) != values:
            raise ValueError(f"remote normalized run.def round-trip changed: {point}")
        context = driver.prepare_paper_1961_driver_context(
            args.config, used_run_def_path=run_def, reference_run_dir=args.asset_root / "landpoints" / point / "reference")
        lane = batch.prepare_lane(point, context, state)
        shifted = driver.DriverPreviousStepStatePacket(
            tstep=8 * 48 - 1, fields_by_component=state.fields_by_component,
            provenance_by_component=state.provenance_by_component)
        other = batch.prepare_lane(point, context, shifted, day_numbers=range(9, 16)).arguments
        lane, other = frozen_io.bind_frozen(lane, other, frozen_points[point], records[point], vegetation.as_posix())
        for key, value in (("first_block_arguments_sha256", lane.arguments),
                           ("second_block_arguments_sha256", other)):
            if producer._tree_digest(value) != records[point][key]:
                raise ValueError(f"remote numerical prepared inputs differ from local audit: {point}/{key}")
        batch.audit_compatible_lanes([*lanes, lane])
        batch.audit_compatible_lanes([replace(item, arguments=value)
            for item, value in zip([*lanes, lane], [*second, other], strict=True)])
        lanes.append(lane)
        second.append(other)
        report["new_point_records"].append({"landpoint_id": point,
            "prepare_seconds": time.perf_counter() - point_started,
            "numeric_inputs_match_local": True, "cold_Day1_reused": True,
            "frozen_arguments_bound": True, "cold_argument_boundary_verified": True,
            "VEGETATION_FILE_mapping": {"source": run_defs[point]["VEGETATION_FILE"], "remote": vegetation.as_posix()},
            "static_signature_sha256": hashlib.sha256(repr(lane.static_signature).encode()).hexdigest()})
        report["checkpoint_count"] = len(lanes)
        report["elapsed_seconds"] = time.perf_counter() - started
        producer._atomic_pickle(args.output_dir / "prepared_inputs.progress.pkl", (lanes, second))
        producer._atomic_json(report_path, report)
    report["static_schema_input_screen"] = scale.screen_inputs(lanes, second)
    if [base.semantic_reference_identity(args, item, other)
            for item, other in zip(lanes[:4], second[:4], strict=True)] != anchor_ids:
        raise ValueError("remote original four anchor numerical identities changed")
    destination = args.output_dir / "prepared_inputs_remote_64.pkl"
    producer._atomic_pickle(destination, (lanes, second))
    # The production reader itself must admit the on-disk full64 before GPU use.
    actual, others = scale.read_prepared(destination, args.config)
    report["actual_full64_reader_screen"] = scale.screen_inputs(actual, others)
    report.update(status="prepared", prepared_point_count=64, prepared_pickle=str(destination),
                  prepared_pickle_sha256=producer._sha256_file(destination),
                  elapsed_seconds=time.perf_counter() - started,
                  precision_policy=scale.PRECISION_POLICY,
                  scope="Linux unpickle/path/static/schema/baseline native dtype admission; no forward or GPU science acceptance")
    producer._atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "prepared_pickle": str(destination),
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory-tar", "selection", "preparation-audit", "cold-states", "cold-identities", "anchor-prepared", "frozen-arguments",
                 "asset-root", "data-root", "config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frozen-arguments-sha256", required=True)
    parser.add_argument("--max-seconds", type=float, default=1500)
    parser.add_argument("--baseline-ref", default=base.BASELINE)
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or not 0 < args.max_seconds <= 1500:
        parser.error("remote rehydration requires Linux and a duration in (0, 1500] seconds")
    for path in (args.inventory_tar.parent, args.selection.parent, args.preparation_audit.parent,
                 args.cold_states.parent, args.cold_identities.parent, args.anchor_prepared.parent,
                 args.frozen_arguments.parent, args.asset_root, args.data_root):
        scale.StageStore.validate_source_destination(path, args.output_dir)
    os.environ.setdefault("ORCHIDEE_REPO_ROOT", str(ROOT))
    os.environ["ORCHIDEE_DATA_ROOT"] = str(args.data_root.resolve())
    os.environ["ORCHIDEE_REFERENCE_ROOT"] = str(args.asset_root)
    os.environ["ORCHIDEE_OUTPUT_ROOT"] = str(args.output_dir.resolve())
    def deadline(number, frame):
        raise TimeoutError("CPU rehydration internal wall limit exhausted; no GPU admission")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        rehydrate(args)
    except Exception as exc:
        path = args.output_dir / "remote-preparation-audit.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            report.update(status="failed", message=str(exc), exception_type=type(exc).__name__)
            producer._atomic_json(path, report)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
