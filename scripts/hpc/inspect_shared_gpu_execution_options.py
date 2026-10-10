"""Read compiler defaults without initializing a device or compiling a graph."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import jax
    import jaxlib
    from jaxlib import xla_client

    compile_options = xla_client.CompileOptions()
    build_options = compile_options.executable_build_options
    options = build_options.debug_options
    terms = ("command_buffer", "latency_hiding", "triton", "graph", "compilation", "parallelism")
    fields = {}
    for name in dir(options):
        if any(term in name for term in terms):
            value = getattr(options, name)
            fields[name] = value if isinstance(value, (str, bool, int, float)) else repr(value)
    report = {
        "scope": "compiler_default_metadata_no_device_or_graph",
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "matching_fields": fields,
        "matching_proto_lines": [
            line for line in str(options).splitlines()
            if any(term in line for term in terms)
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
