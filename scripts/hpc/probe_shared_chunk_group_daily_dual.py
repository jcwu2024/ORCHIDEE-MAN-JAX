"""G2 wrapper reusing the independently saved original-field public dual gate.

The public objective, native argnums=(0,1), dynamic three-argument ABI, complete
strict initial/seven-pair comparisons and raw preservation belong to the
unchanged shared daily dual probe. This wrapper identifies the new adapter and
does not treat the original-reference parameter bridge as G2 admission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys


VARIANT = "field_expert128+backbone_source_reuse+expert5_native_map_batch_size2"


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--public-gate-file", type=Path, required=True)
    wrapper, remaining = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location("shared_group2_public_dual_gate", wrapper.public_gate_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_load = module.load

    def annotate(report):
        if isinstance(report, dict) and report.get("scope") == "original_four_daily_micro4_public_loss_full_parameter_and_initial_state_AD":
            report["variant"] = VARIANT
            report["candidate_execution"] = VARIANT
            report["wrapper_execution_variant"] = "expert5_native_map_batch_size2"
            report["original_reference_recompiled"] = False
            report["parameter_report_role"] = "provenance_of_original_reference_parameter_only_bridge; supplies_no_G2_admission"
            identity = report.get("identity")
            if identity is not None:
                identity["comparison_schema"] = "shared_chunk_group2_daily_dual_micro4_v1"
                bridge = identity.pop("v4_parameter_report", None)
                if bridge is not None:
                    identity["original_reference_parameter_bridge_report"] = bridge
            artifacts = report.get("source_artifacts", {})
            artifacts["group2_wrapper"] = {"path": str(Path(__file__).resolve()),
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            if "v4_execution_adapter" in artifacts:
                artifacts["group2_execution_adapter"] = artifacts.pop("v4_execution_adapter")
        return report

    def load(path, name):
        loaded = original_load(path, name)
        if name == "scripts.hpc.probe_shared_expert_real_daily":
            atomic_json = loaded.atomic_json
            def annotated_json(output, report):
                return atomic_json(output, annotate(report))
            loaded.atomic_json = annotated_json
        return loaded

    module.load = load
    if "--stage" not in remaining or remaining[remaining.index("--stage") + 1] != "candidate":
        raise ValueError("G2 wrapper reuses existing original reference; candidate stage only")
    sys.argv = [str(wrapper.public_gate_file), *remaining]
    module.main()


if __name__ == "__main__":
    main()
