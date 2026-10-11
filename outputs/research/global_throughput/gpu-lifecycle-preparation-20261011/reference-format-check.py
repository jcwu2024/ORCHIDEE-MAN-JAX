"""Read the existing64-point reference only; no model/backend execution."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.dev import global_lifecycle_science as science
from scripts.dev import validate_global_gpu_lifecycle as validator

directory = ROOT / "outputs/research/global_throughput/gpu-round4/model"
manifest = json.loads((directory / "stage_manifest.json").read_text())
identity = {**manifest["identity"], "science_sha256": science.SCIENCE_SHA256,
            "comparison_reference": science.comparison_admission()}
reference, origin = science.load_comparison_reference(directory, identity)
schema = validator.native_schema(reference, reference, 64)
comparison = validator.compare_tree(reference, reference, sentinel_paths=origin["sentinel_paths"])
if not comparison["passed"]:
    raise ValueError(comparison)
report = {"status": "passed_real_frozen_reference_format_compatibility_only",
          "reference_path": origin["path"], "reference_sha256": origin["sha256"], "reference_bytes": origin["bytes"],
          "manifest_sha256": origin["manifest_sha256"], "source_science_sha256": science.OLD_SCIENCE_SHA256,
          "new_declared_science_sha256": science.SCIENCE_SHA256, "array_outer_lanes": 64, "native_leaves": len(schema),
          "dtype_leaf_counts": dict(Counter(item["dtype"] for item in schema)), "native_schema": schema,
          "sentinel_paths": origin["sentinel_paths"], "self_comparison": comparison,
          "JAX_imported": "jax" in sys.modules, "model_execution": False, "GPU_initialization": False,
          "claim": "restricted NumPy-only unpickler and comparator support the frozen real reference format/dtype/schema/declared NaN masks; no new-core/GPU numerical acceptance"}
destination = Path(__file__).with_name("reference-format-compatibility.json")
destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({key: report[key] for key in ("status", "reference_sha256", "reference_bytes", "array_outer_lanes", "native_leaves", "dtype_leaf_counts", "JAX_imported")}))
