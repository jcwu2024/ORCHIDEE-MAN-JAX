"""Local read-only sealed baseline admission; no model execution or compilation."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import jax

from scripts.dev import benchmark_global_gpu_capacity as capacity
from scripts.dev.global_server_reference import load_reference
from scripts.dev.global_source_compatibility import source_identity


def main():
    artifact = Path(__file__).resolve().parent
    outputs = ROOT / "outputs/research/global_throughput"
    b256_dir = outputs / "gpu-signature-repair-20261008/job-15643868-B256-evidence/capacity"
    b256_identity = json.loads((b256_dir / "stage_manifest.json").read_text())["identity"]
    current_source = source_identity(ROOT)
    # Current production base API, never an old-identity replay.
    from scripts.dev.benchmark_global_batch import scientific_source_digest

    assert scientific_source_digest() == current_source["science_sha256"]
    assert current_source["science_sha256"] == b256_identity["science_sha256"]
    expected_identity = {**b256_identity, "science_sha256": current_source["science_sha256"]}
    reference, reference_origin = load_reference(outputs / "gpu-round4/model", expected_identity)
    original128 = outputs / "gpu-sweetspot-preparation-20261006/job-15600405-terminal-evidence"
    view128 = artifact / "sealed-b128-origin-view"
    view128.mkdir(exist_ok=True)
    for name in ("stage_manifest.json", "stage_capacity_B128_fixed512_v1.pkl"):
        target = view128 / name
        source = original128 / ("capacity__" + name)
        if not target.exists():
            os.link(source, target)
    b128_identity = {**expected_identity, "contract": capacity.CONTRACT,
        "candidate_shapes": [128, 256],
        "conditional_policy": "core_and_wrapper_5percent_30min_2GiBslack"}
    b128_identity.pop("continuation_B128_sha256", None)
    destination = artifact / "prospective-b512-stages"
    _, b128 = capacity.import_b128_baseline(view128, b128_identity, destination, jax, reference=reference)
    _, b256 = capacity.import_b256_baseline(b256_dir, expected_identity, destination, jax, reference=reference)
    memory = b256["provenance"]["executable_memory_analysis"]
    report = {"status": "sealed_baselines_admitted", "model_execution": False, "compiled": False,
        "FULL64_reference_origin": reference_origin,
        "B128": {key: b128[key] for key in ("sha256", "bytes", "source_stage", "role")},
        "B256": {key: b256[key] for key in ("sha256", "bytes", "source_stage", "role")},
        "B128_original_path": str(original128), "B128_view_policy": "local hard links to unchanged sealed flattened files; no data copy or source mutation",
        "B128_code": b128["execution_origin"]["environment"]["git_head"],
        "B256_code": b256["execution_origin"]["environment"]["git_head"],
        "strict_gates_per_origin": 515, "native_leaves_per_origin": 760,
        "full_output_schema_dtypes_against_FULL64": True,
        "imported_B256_actual_compiler_memory": memory,
        "current_execution_source_identity": current_source,
        "expected_science_identity_source": "current benchmark_global_batch.scientific_source_digest API",
        "current_certified_science_matches_sealed_origin": current_source["science_sha256"] == b256_identity["science_sha256"],
        "new_destination_stage_directory_created": destination.exists(),
        "limitation": "local origin admission only; current deployed GPU executable not validated"}
    (artifact / "b512-baseline-admission.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
