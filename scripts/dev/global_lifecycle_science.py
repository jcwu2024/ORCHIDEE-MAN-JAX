"""Independent scientific identity and explicit frozen-reference comparison admission.

The carbon fixes are declared scientific changes, never DIFFUCO-only execution
equivalence. This module does not import JAX or initialize any backend.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import pickle
import subprocess
from collections import namedtuple
from pathlib import Path

CONTRACT = "GPU-B1024-phased-memory-lifecycle-new-science-v1"
PROCESS_STRATEGY = "sequential_fresh_process_per_strategy"
STAGE_NAME = "capacity_B1024_lifecycle_v1"
STRATEGIES = ("phased_resident_control", "phased_release")
SCIENCE_PIN = "1fdacde59d86f696f384b56d69292c8f6fa8d30b"
SCIENCE_SHA256 = "8c8b1dfd0b207e96bbb81c2d3eb85ae6edccb9ed3c6b77fb470dabc58c5fa77d"
OLD_SCIENCE_SHA256 = "11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5"
REFERENCE_SHA256 = "a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6"
REFERENCE_STAGE = "B4_group_references"
RUN_TWO_SHA256 = "6dc004b320ec5ce0a613dd1a290e2fd67cf085b923b5d1ad3e7dc2f0a4e8d609"
PRECISION_POLICY = (
    "JAX x64 enabled; baseline native dtypes retained; geographical payload and solar subcalculation contain FP32"
)
GATE = {
    "atol": 1e-8,
    "rtol": 1e-10,
    "discrete_schema_dtype": "exact",
    "sentinel": "declared_bare_soil_roughheight_only",
}
BASELINE_REF = "133e35aec7b83b9d79888c46092f64d518e97299"

# Inert serialized DTO only: exact fields of modelout.py::ModeloutResult61-67.
# Importing that scientific module would initialize JAX; array validation does
# not need any of its process code. Both sides retain NamedTuple field paths.
ModeloutResult = namedtuple("ModeloutResult", "AGB_model BGB_model GPP_model NPP_model")


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def digest_file(path):
    digest, count = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            count += len(block)
    return digest.hexdigest(), count


def source_identity(root, *, expected_code_sha=None, require_clean=False):
    """Admit exact declared Git blobs or their explicit CRLF representation only."""
    root = Path(root).resolve()
    pin_paths = git(root, "ls-tree", "-r", "--name-only", SCIENCE_PIN, "--", "jax_orchidee").decode().splitlines()
    tracked = git(root, "ls-files", "jax_orchidee").decode().splitlines()
    if tracked != pin_paths:
        raise ValueError("tracked scientific source path set differs from declared science pin")
    executable_extensions = {".py", ".pyi", ".pyc", ".pyd", ".so", ".dll"}
    extras = [
        str(path.relative_to(root))
        for path in (root / "jax_orchidee").rglob("*")
        if path.is_file()
        and path.suffix.lower() in executable_extensions
        and path.relative_to(root).as_posix() not in set(pin_paths)
    ]
    if extras:
        raise ValueError(f"undeclared/untracked executable scientific source: {extras}")
    canonical, actual, representations = hashlib.sha256(), hashlib.sha256(), {}
    for path in pin_paths:
        expected = git(root, "show", f"{SCIENCE_PIN}:{path}")
        target = root / path
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError(f"scientific source symlink/path escape: {path}")
        data = target.read_bytes()
        if data == expected:
            representation = "exact Git blob"
        elif b"\r" not in expected and data == expected.replace(b"\n", b"\r\n"):
            representation = "exact declared LF-to-CRLF representation"
        else:
            raise ValueError(f"undeclared scientific source bytes: {path}")
        canonical.update(path.encode() + b"\0" + expected + b"\0")
        actual.update(path.encode() + b"\0" + data + b"\0")
        representations[path] = representation
    if canonical.hexdigest() != SCIENCE_SHA256:
        raise ValueError("declared science pin canonical content digest changed")
    code_sha = git(root, "rev-parse", "HEAD").decode().strip()
    if expected_code_sha is not None and code_sha != expected_code_sha:
        raise ValueError("execution code SHA differs from expected exact checkout")
    dirty = git(root, "status", "--porcelain", "--untracked-files=all").decode().strip()
    if require_clean and dirty:
        raise ValueError("execution requires a clean immutable code checkout")
    return {
        "science_sha256": SCIENCE_SHA256,
        "science_source_pin": SCIENCE_PIN,
        "actual_science_source_sha256": actual.hexdigest(),
        "science_source_contract": "exact-declared-new-carbon-core-source-v1",
        "science_source_representations": representations,
        "expected_code_sha": expected_code_sha or code_sha,
        "claim": "declared scientific changes; strict frozen-output comparison pending",
    }


def comparison_admission():
    return {
        "contract": "new-science-versus-frozen-old-FULL64-strict-comparison-v1",
        "old_science_sha256": OLD_SCIENCE_SHA256,
        "sha256": REFERENCE_SHA256,
        "stage": REFERENCE_STAGE,
        "role": "immutable comparison goldens; no source-equivalence or GPU-acceptance claim",
        "gate": GATE,
    }


def measurement_fields(strategy):
    if strategy not in STRATEGIES:
        raise ValueError("explicit phased lifecycle measurement strategy required")
    return {
        "measurement_strategy": strategy,
        "sampling_order": "five_core_then_five_wrapper",
        "wrapper_resident_input_policy": "release_core_inputs"
        if strategy == "phased_release"
        else "retain_core_inputs",
    }


def common_identity(identity):
    return {
        key: value
        for key, value in identity.items()
        if key not in {"measurement_strategy", "wrapper_resident_input_policy"}
    }


def validate_identity(identity, *, strategy, expected_code_sha):
    required = {
        "contract": CONTRACT,
        "process_strategy": PROCESS_STRATEGY,
        "stage_name": STAGE_NAME,
        "stage_shape": 1024,
        "total_lanes": 2048,
        "candidate_shapes": [1024],
        "unique_real_points": 64,
        "warm_repeats": 5,
        "budget_seconds": 7200,
        "science_sha256": SCIENCE_SHA256,
        "science_source_pin": SCIENCE_PIN,
        "expected_code_sha": expected_code_sha,
        "comparison_reference": comparison_admission(),
        "gate": GATE,
        "precision_policy": PRECISION_POLICY,
        **measurement_fields(strategy),
    }
    for key, value in required.items():
        if key not in identity or identity[key] != value:
            raise ValueError(f"lifecycle identity mismatch: {key}")
    runtime = identity.get("runtime", {})
    if set(runtime) != {
        "python",
        "jax",
        "jaxlib",
        "numpy",
        "device_details",
        "x64_enabled",
        "backend_platform_version",
    } or any(not runtime.get(key) for key in ("python", "numpy", "backend_platform_version")):
        raise ValueError("complete declared canonical runtime identity required")
    if (
        runtime.get("jax") != "0.4.38"
        or runtime.get("jaxlib") != "0.4.38"
        or runtime.get("x64_enabled") is not True
        or len(runtime.get("device_details", [])) != 1
        or runtime["device_details"][0].get("platform") != "gpu"
    ):
        raise ValueError("canonical single GPU JAX/jaxlib0.4.38/x64 runtime required")
    binding = identity.get("run_binding", {})
    physical = binding.get("physical_gpu", {})
    if (
        binding.get("allocator") != "default"
        or binding.get("preallocate") is not False
        or not binding.get("Slurm_job_id")
        or not binding.get("hostname")
        or not physical.get("uuid")
        or type(physical.get("total_bytes")) is not int
        or physical["total_bytes"] < 30 * 1024**3
    ):
        raise ValueError("actual allocation/node/UUID/default allocator identity required")
    for key in ("baseline_ref", "lanes", "compiler_environment", "lane_origins", "actual_science_source_sha256"):
        if not identity.get(key):
            raise ValueError(f"missing lifecycle common identity: {key}")
    if len(identity["lanes"]) != 64 or len(identity["lane_origins"]) != 2048:
        raise ValueError("exact FULL64/repeated lane identity required")
    epoch = binding.get("window_start_epoch")
    if (
        type(epoch) not in (int, float)
        or not 0 < epoch < float("inf")
        or binding.get("server_resources", {}).get("allocated_CPU_request_count") != 4
    ):
        raise ValueError("actual allocation epoch/four auxiliary CPUs required")
    expected_origins = [
        {
            "lane_index": i,
            "source_index": i % 64,
            "landpoint_id": identity["lanes"][i % 64]["landpoint_id"],
            "repeat_index": i // 64,
        }
        for i in range(2048)
    ]
    if identity["lane_origins"] != expected_origins:
        raise ValueError("repeated lane source/order identity mismatch")


class ArrayUnpickler(pickle.Unpickler):
    """Saved outputs contain only builtin trees/NumPy; never import model/JAX classes."""

    def find_class(self, module, name):
        if name == "ModeloutResult" and module in {"jax_orchidee.stomate.modelout", __name__}:
            return ModeloutResult
        allowed = {("numpy", "ndarray"), ("numpy", "dtype")}
        for prefix in ("numpy.core", "numpy._core"):
            allowed.update(
                {
                    (prefix + ".multiarray", "_reconstruct"),
                    (prefix + ".multiarray", "scalar"),
                    (prefix + ".numeric", "_frombuffer"),
                }
            )
        if (module, name) not in allowed:
            raise ValueError(f"saved-array pickle contains forbidden class: {module}.{name}")
        return super().find_class(module, name)


def load_array_pickle(data):
    return ArrayUnpickler(io.BytesIO(data)).load()


def load_comparison_reference(directory, identity):
    """Explicitly admit the old science as a comparison source, never current identity."""
    if identity.get("comparison_reference") != comparison_admission():
        raise ValueError("explicit independent old-reference comparison admission required")
    directory = Path(directory).resolve()
    manifest_data = (directory / "stage_manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    old = manifest["identity"]
    for key, expected in {
        "science_sha256": OLD_SCIENCE_SHA256,
        "baseline_ref": identity["baseline_ref"],
        "gate": GATE,
        "precision_policy": PRECISION_POLICY,
        "lanes": identity["lanes"],
    }.items():
        if old.get(key) != expected:
            raise ValueError(f"frozen FULL64 comparison source mismatch: {key}")
    runtime = old.get("runtime", {})
    if (
        runtime.get("jax") != "0.4.38"
        or runtime.get("jaxlib") != "0.4.38"
        or runtime.get("x64_enabled") is not True
        or {device.get("platform") for device in runtime.get("device_details", [])} != {"gpu"}
    ):
        raise ValueError("frozen FULL64 canonical GPU reference runtime required")
    stage = manifest.get("stages", {}).get(REFERENCE_STAGE, {})
    path = directory / stage.get("path", f"stage_{REFERENCE_STAGE}.pkl")
    if not path.resolve().is_relative_to(directory):
        raise ValueError("frozen reference path escapes directory")
    data = path.read_bytes()
    if (
        stage.get("status") != "complete"
        or stage.get("sha256") != REFERENCE_SHA256
        or hashlib.sha256(data).hexdigest() != REFERENCE_SHA256
    ):
        raise ValueError("frozen FULL64 exact checkpoint hash/status mismatch")
    envelope = load_array_pickle(data)
    record = envelope.get("record", {})
    if (
        envelope.get("identity") != old
        or envelope.get("name") != REFERENCE_STAGE
        or envelope.get("status") != "complete"
        or record.get("provenance") != stage.get("provenance")
        or record.get("gates") != stage.get("gates")
        or not record.get("gates")
        or any(gate.get("passed") is not True for gate in record["gates"])
        or old.get("validation_mode", "strict") != "strict"
        or stage.get("evidence_class", "accepted") != "accepted"
        or stage.get("provenance", {}).get("own_inputs") is not True
        or stage["provenance"].get("point_ids") != [lane["landpoint_id"] for lane in identity["lanes"]]
    ):
        raise ValueError("frozen FULL64 envelope/manifest/full-input proof mismatch")
    value = envelope["value"]
    output = value["outputs"] if isinstance(value, dict) and "outputs" in value else value
    sentinel_paths = {}
    for gate in record["gates"]:
        for sentinel in gate.get("allowed_sentinels", []):
            contract, sentinel_path = sentinel.get("contract", {}), sentinel.get("path")
            if (
                contract.get("field") != "roughheight_pft"
                or contract.get("component") not in {"diffuco_previous_step_state", "sechiba_finalize_state"}
                or contract.get("bare_soil_index") != 0
                or contract.get("inner_shape") != [1, 14]
                or not contract.get("provenance")
                or not isinstance(sentinel_path, str)
            ):
                raise ValueError("frozen reference sentinel source contract mismatch")
            if sentinel_path in sentinel_paths and sentinel_paths[sentinel_path] != contract:
                raise ValueError("frozen reference sentinel contract conflict")
            sentinel_paths[sentinel_path] = contract
    return output, {
        "path": str(path),
        "sha256": REFERENCE_SHA256,
        "bytes": len(data),
        "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
        "source_identity": old,
        "sentinel_paths": sentinel_paths,
        "role": comparison_admission()["role"],
    }


def run_two_binding(root):
    path = Path(root) / "scripts/dev/benchmark_global_batch.py"
    source = path.read_text(encoding="utf-8")
    fn = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "run_two")
    digest = hashlib.sha256(ast.get_source_segment(source, fn).encode()).hexdigest()
    if digest != RUN_TWO_SHA256:
        raise ValueError("continuous two-block source binding changed")
    return {
        "source_path": "scripts/dev/benchmark_global_batch.py",
        "function": "run_two",
        "function_sha256": digest,
        "science_pin": SCIENCE_PIN,
        "second_block_state_argument": "second._replace(values_by_component=a[0])",
        "array_evidence": "actual first-block output and second-block positional0 input from initial collection only",
    }
