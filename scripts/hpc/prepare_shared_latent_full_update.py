"""CPU-only readiness check for corrected saved full-state latent-reader graphs."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def check_readiness(manifest_path):
    module_name = "scripts.hpc.probe_shared_latent_dual"
    path = Path(__file__).with_name("probe_shared_latent_dual.py")
    loaded = sys.modules.get(module_name)
    if loaded is None or Path(loaded.__file__).resolve() != path.resolve():
        spec = importlib.util.spec_from_file_location(module_name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = loaded
        spec.loader.exec_module(loaded)

    manifest = loaded.read_manifest(manifest_path)
    if manifest.get("admission_mode") != loaded.CORRECTED_PERFORMANCE_DIAGNOSTIC:
        raise ValueError("corrected performance diagnostic manifest opt-in required")
    result = loaded.validate_manifest_asset_readiness(manifest)
    result.update({"manifest": str(Path(manifest_path)),
                   "scope": "cpu_saved_graph_identity_and_asset_readiness",
                   "manifest_prepared_context_identity": manifest.get(
                       "performance_diagnostic", {}).get("prepared_context_identity"),
                   "candidate_adoption": False, "model_gradient_compilations": 0,
                   "gpu_backend_initialized": False, "optimizer_updates": 0,
                   "automatic_compile_fallback": False})
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--output", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    result = check_readiness(args.manifest)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if result["status"] == "ready_for_saved_graph_load" else 2


if __name__ == "__main__":
    raise SystemExit(main())
