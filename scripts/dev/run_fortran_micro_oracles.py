from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from extract_fortran_micro_oracle import (
    DEFAULT_MANIFEST,
    OracleManifestError,
    load_manifest,
    validate_and_extract,
)
from fortran_oracle_common import DEFAULT_COMPILER, ROOT


def _family_records(document: dict[str, object]) -> dict[str, dict[str, object]]:
    records = {str(item["id"]): item for item in document.get("families", [])}
    records["qsat_moisture"] = {
        "id": "qsat_moisture",
        "runner": "run_qsat_micro_oracle:run_oracle",
    }
    return records


def _resolve_runner(specification: str):
    module_name, function_name = specification.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile and verify source-extracted Fortran micro-oracle families."
    )
    parser.add_argument("--family", required=True, help="Family id or 'all'.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Require every selected family to pass numerically.",
    )
    args = parser.parse_args(argv)
    try:
        document = load_manifest(args.manifest)
        validate_and_extract(document, root=ROOT)
        families = _family_records(document)
        selected = list(families) if args.family == "all" else [args.family]
        unknown = set(selected) - set(families)
        if unknown:
            raise OracleManifestError(f"unknown families: {sorted(unknown)}")
        results = []
        for family_id in selected:
            family = families[family_id]
            runner = _resolve_runner(str(family["runner"]))
            output_dir = (
                ROOT / "outputs" / "reference_mode" / "micro_oracles" / family_id
            )
            result = runner(output_dir=output_dir, compiler=args.compiler)
            results.append({"family": family_id, "status": result["status"]})
        passed = all(item["status"] == "passed" for item in results)
        print(
            json.dumps(
                {"status": "passed" if passed else "failed", "families": results},
                indent=2,
            )
        )
        return 0 if passed or not args.verify else 1
    except (
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        OracleManifestError,
        yaml.YAMLError,
    ) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
