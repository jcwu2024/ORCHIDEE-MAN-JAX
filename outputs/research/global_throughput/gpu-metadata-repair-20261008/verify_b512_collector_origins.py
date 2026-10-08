"""Actual historical saved-origin revalidation through the new read-only collector."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.dev import benchmark_global_batch as base
from scripts.dev import benchmark_global_gpu_capacity as capacity
from scripts.dev.global_server_reference import load_reference
from scripts.dev.global_source_compatibility import source_identity

spec = importlib.util.spec_from_file_location("b512_collector_origin_check", HERE / "collect_b512_result.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
directories = {128: HERE / "sealed-b128-origin-view",
    256: HERE.parent / "gpu-signature-repair-20261008/job-15643868-B256-evidence/capacity"}
identity = json.loads((directories[256] / "stage_manifest.json").read_text())["identity"]
reader = source_identity(ROOT)
assert base.scientific_source_digest() == reader["science_sha256"] == identity["science_sha256"]
identity.update(collector.execution_identity(), contract=collector.CONTRACT, candidate_shapes=[512], budget_seconds=2700)
reference, _ = load_reference(HERE.parent / "gpu-round4/model", identity)
results = {}
for size, directory in directories.items():
    expected = json.loads((directory / "stage_manifest.json").read_text())["identity"]
    importer = capacity.import_b128_baseline if size == 128 else capacity.import_b256_baseline
    _, origin = importer(directory, expected, HERE / "prospective-collector-output", base.jax, reference=reference)
    validated = collector.validate_baseline(origin, identity, directory, HERE / "prospective-collector-output",
        size, capacity, base.jax, reference)
    results[str(size)] = {key: validated[key] for key in ("sha256", "bytes", "source_stage", "role")}
    results[str(size)]["original_code"] = validated["execution_origin"]["environment"]["git_head"]
    results[str(size)]["original_job"] = validated["provenance"]["server_resources"]["Slurm"]["SLURM_JOB_ID"]
report = {"status": "actual_historical_origins_revalidated", "model_execution": False, "compiled": False,
    "current_reader_source_identity": reader, "declared_future_execution_identity_pin": collector.execution_identity(),
    "historical_origins": results, "new_B512_measurement": False,
    "limitation": "producer Linux SHA remains a declared exact pin; local Windows reader SHA is separately preserved. No B512 result exists."}
(HERE / "b512-collector-origin-verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "actual_historical_origins": len(results), "new_B512_measurement": False}))
