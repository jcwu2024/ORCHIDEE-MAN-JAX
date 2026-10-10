"""Run the reviewed original T30 recurrent dual-AD reference mode only."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def load(path: Path):
    spec = importlib.util.spec_from_file_location(
        "shared_normalized_recurrent_reference_source", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reviewed recurrent probe from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-probe", type=Path, required=True)
    wrapper_args, probe_args = parser.parse_known_args()
    probe = load(wrapper_args.reference_probe)
    guard = argparse.ArgumentParser(add_help=False)
    guard.add_argument("--horizon", type=int, required=True)
    guard.add_argument("--stage", required=True)
    guard_args, _ = guard.parse_known_args(probe_args)
    if guard_args.horizon != 30 or guard_args.stage != "reference":
        raise ValueError("this entry point admits only the original T30 reference stage")
    if "--execution-module-file" in probe_args or "--llvm-parallelism" in probe_args:
        raise ValueError("the original reference must use the unmodified field route")
    sys.argv = [str(wrapper_args.reference_probe), *probe_args]
    probe.main()


if __name__ == "__main__":
    main()
