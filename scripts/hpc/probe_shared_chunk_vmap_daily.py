"""Run the saved public daily-loss probe with whole-vmap metadata tagging."""
from __future__ import annotations

import importlib.util
import hashlib
import os
from pathlib import Path
import sys


probe_path = Path(os.environ["SHARED_DAILY_PROBE"])
adapter_path = Path(__file__).resolve().with_name("shared_chunk_vmap_execution.py")
wrapper_path = Path(__file__).resolve()
source_hashes = {
    "execution_adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
    "execution_wrapper_sha256": hashlib.sha256(wrapper_path.read_bytes()).hexdigest(),
    "source_probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
}
spec = importlib.util.spec_from_file_location("saved_shared_daily_probe", probe_path)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)

original_load = probe.load


def load(path, name):
    module = original_load(path, name)
    if Path(path).name == "probe_shared_expert_real_daily.py":
        original_atomic_json = module.atomic_json

        def atomic_json(path, payload):
            payload["variant"] = "field_expert128+backbone_source_reuse+expert5_whole_chunk_axis_vmap"
            payload["execution_adapter"] = str(adapter_path)
            payload["execution_wrapper"] = str(wrapper_path)
            payload["source_probe"] = str(probe_path)
            payload.update(source_hashes)
            payload["prior_synthetic_component_gate"] = (
                "whole_vmap_synthetic_component_strict_failure_preserved"
            )
            payload["execution_axis_semantics"] = (
                "original_59_by_128_chunk_axis_vmap_and_original_four_sample_axis_vmap"
            )
            identity = payload.get("identity")
            if isinstance(identity, dict) and "comparison_schema" in identity:
                identity["comparison_schema"] = "shared_chunk_vmap_original_micro4_v1"
            return original_atomic_json(path, payload)

        module.atomic_json = atomic_json
    return module


probe.load = load
probe.main()
