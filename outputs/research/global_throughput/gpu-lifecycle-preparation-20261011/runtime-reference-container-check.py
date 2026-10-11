"""Existing frozen64 reference container bridge only; no model, JIT or GPU."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from jax_orchidee.stomate.modelout import ModeloutResult
from scripts.dev import benchmark_global_batch as base
from scripts.dev import benchmark_global_gpu_lifecycle as runner
from scripts.dev import global_lifecycle_science as science
from scripts.dev import validate_global_gpu_lifecycle as validator

directory = ROOT / "outputs/research/global_throughput/gpu-round4/model"
manifest = json.loads((directory / "stage_manifest.json").read_text())
identity = {**manifest["identity"], "science_sha256": science.SCIENCE_SHA256,
            "comparison_reference": science.comparison_admission()}
inert, origin = science.load_comparison_reference(directory, identity)
native = runner.restore_runtime_reference_containers(inert)
before, after = validator.flatten(inert), validator.flatten(native)
assert len(before) == len(after) == 760
assert all(path == other and array is restored for (path, array), (other, restored) in zip(before, after, strict=True))
assert validator.structure(inert) == validator.structure(native)
native_dtos = []

def visit(tree):
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):
        assert type(tree) is ModeloutResult
        native_dtos.append(tree)
    if isinstance(tree, dict):
        for value in tree.values():
            visit(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            visit(value)

visit(native)
assert native_dtos
# The exact comparator used in capacity.measure_shape's initial gates now sees
# the same native class on both sides. Four exact DTO fields remain unchanged.
comparison = base.compare(native, native, sentinel_paths=origin["sentinel_paths"])
assert comparison["passed"] and comparison["compared_leaves"] == 760
independent = validator.compare_tree(native, inert, sentinel_paths=origin["sentinel_paths"])
assert independent["passed"]
report = {
    "status": "passed_real_frozen_reference_runtime_container_bridge_only",
    "reference_sha256": origin["sha256"], "reference_bytes": origin["bytes"],
    "reference_lanes": 64, "native_leaves": 760, "native_ModeloutResult_containers": len(native_dtos),
    "native_container_class": "jax_orchidee.stomate.modelout.ModeloutResult",
    "exact_fields": list(ModeloutResult._fields), "array_objects_unchanged": True,
    "paths_shapes_dtypes_unchanged": True, "actual_base_compare": comparison,
    "independent_compare": independent, "JAX_imported": "jax" in sys.modules,
    "model_execution": False, "JIT_execution": False, "GPU_initialization": False,
    "claim": "runtime container identity compatibility only; no new-core or GPU numerical/scientific acceptance",
}
Path(__file__).with_name("runtime-reference-container-compatibility.json").write_text(
    json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({key: report[key] for key in ("status", "native_leaves", "native_ModeloutResult_containers", "array_objects_unchanged", "model_execution", "JIT_execution", "GPU_initialization")}))
