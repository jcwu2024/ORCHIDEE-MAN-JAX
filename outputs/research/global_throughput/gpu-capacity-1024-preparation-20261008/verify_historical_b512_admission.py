"""Exercise new origin-only importer on saved B512; no input audit/model execution."""
import json
import pickle
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def main():
    started = time.perf_counter()
    import jax

    from scripts.dev import benchmark_global_gpu_capacity as capacity

    evidence = ROOT / "outputs/research/global_throughput/gpu-metadata-repair-20261008"
    directory = evidence / "job-15657822-saved-capacity"
    manifest = json.loads((directory / "stage_manifest.json").read_text())
    admission = json.loads((evidence / "job-15657822-original-input-admission.json").read_text())
    identity = manifest["identity"]
    for key in ("lanes", "science_sha256", "baseline_ref", "precision_policy", "gate"):
        if identity[key] != admission["source_identity"][key]:
            raise AssertionError(f"original independently admitted source identity drift: {key}")
    if identity["numerical_reference_sha256"] != admission["FULL64_reference_origin"]["sha256"]:
        raise AssertionError("original FULL64 reference hash drift")
    reference_path = ROOT / "outputs/research/global_throughput/gpu-round4/model/stage_B4_group_references.pkl"
    with reference_path.open("rb") as stream:
        reference_envelope = pickle.load(stream)
    reference = reference_envelope["value"]
    origin = capacity.import_b512_evidence(directory, identity,
        Path(__file__).resolve().parent / "prospective-fixed1024-capacity", jax, reference=reference)
    if "timing" in origin:
        raise AssertionError("historical timing must not enter fixed1024 curve")
    result = {"status": "passed", "scope": "new origin-only importer on original saved B512; no model execution, compilation or prepared-input audit",
        "historical_source_sha256": origin["sha256"], "bytes": origin["bytes"],
        "historical_timing_returned": False, "native_reference_schema_checked": True,
        "compiler_temporary_bytes": origin["provenance"]["executable_memory_analysis"]["temp_size_in_bytes"],
        "historical_process_peak_RSS_bytes": origin["historical_process_peak_RSS_bytes"],
        "code_origin": origin["execution_origin"]["environment"]["git_head"],
        "wall_seconds": time.perf_counter() - started, "fixed1024_runtime_accepted": False}
    output = Path(__file__).resolve().parent / "historical-b512-admission.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
