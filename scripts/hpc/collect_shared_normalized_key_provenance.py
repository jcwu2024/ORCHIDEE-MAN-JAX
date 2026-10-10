"""Read-only source audit/hashes for the staged field-only daily treatment."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overlay", type=Path, required=True)
    p.add_argument("--combined-overlay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    decoder = args.combined_overlay / "shared_decoder_expert_execution.py"
    source = decoder.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "shared_transformer_raw_outputs_expert128")
    names = [a.arg for a in node.args.args + node.args.kwonlyargs]
    calls = sorted({n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)})
    if "dynamic_projection_execution" in names or "_field_dynamic_projection_cache" not in calls or any("union" in x.lower() for x in calls):
        raise ValueError("this executed legacy adapter must be explicitly source-verified field-only")
    paths = [decoder, args.combined_overlay / "shared_backbone_execution.py", args.combined_overlay / "shared_backbone_projection_reuse.py",
             args.combined_overlay / "shared_decoder_dense.py", args.combined_overlay / "shared_decoder_macro_packed.py",
             args.overlay / "shared_normalized_key_execution.py", args.overlay / "probe_shared_normalized_keys_daily.py", args.overlay / "run_shared_normalized_keys_daily.sh"]
    report = {"field_execution_selection": "legacy_field_only_adapter_no_execution_policy_parameter",
              "entry": node.name, "arguments": names, "line_span": [node.lineno, node.end_lineno],
              "direct_calls": calls, "field_projection_cache_called": True, "union_calls": [],
              "sources": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for path in paths]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
