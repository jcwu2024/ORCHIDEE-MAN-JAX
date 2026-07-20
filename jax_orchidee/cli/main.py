"""Stable CLI dispatch for production runs and asset validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(
    os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
DEFAULT_CONFIG = REPO_ROOT / "configs" / "orchidee_man_250919.yaml"


def _inventory(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="orchidee-jax inventory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    from jax_orchidee.driver.domain import load_case_config

    config = load_case_config(args.config)
    required = tuple(
        Path(value)
        for value in config.get("minimal_single_case", {}).get("required_files", ())
    )
    repo_root = args.config.resolve().parents[1]
    rows = [
        {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.is_file() else None,
        }
        for path in required
    ]
    payload = {
        "config": str(args.config.resolve()),
        "data_root": os.environ.get("ORCHIDEE_DATA_ROOT", str(repo_root / "data")),
        "reference_root": os.environ.get("ORCHIDEE_REFERENCE_ROOT", str(repo_root / "reference")),
        "output_root": os.environ.get("ORCHIDEE_OUTPUT_ROOT", str(repo_root / "outputs")),
        "required_count": len(rows),
        "missing_count": sum(not row["exists"] for row in rows),
        "files": rows,
    }
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"config: {payload['config']}")
        print(f"required inputs: {payload['required_count']}; missing: {payload['missing_count']}")
        for row in rows:
            status = "ok" if row["exists"] else "missing"
            print(f"[{status}] {row['path']}")
    return 0 if payload["missing_count"] == 0 else 2


def _run(argv: list[str]) -> int:
    from jax_orchidee.runners.multiyear import main as run_main

    defaults = ["--compiled-sechiba-day", "on", "--compiled-day-block-size", "7"]
    return run_main([*defaults, *argv])


def _validate_landpoints(argv: list[str]) -> int:
    from jax_orchidee.runners.acceptance import main as acceptance_main

    return acceptance_main(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--version"]:
        print("orchidee-man-jax 0.1.0")
        return 0
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(
            "usage: orchidee-jax {inventory,run,validate-landpoints} ...\n\n"
            "commands:\n"
            "  inventory            check external input assets\n"
            "  run                  run one PFT14 landpoint across one or more years\n"
            "  validate-landpoints  compare runs with annual Fortran truth"
        )
        return 0
    command, remainder = arguments[0], arguments[1:]
    if command == "inventory":
        return _inventory(remainder)
    if command == "run":
        return _run(remainder)
    if command == "validate-landpoints":
        return _validate_landpoints(remainder)
    print(f"orchidee-jax: unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
