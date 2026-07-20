from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dev.extract_stomate_restart_netcdf_schema import extract_schema  # noqa: E402


DEFAULT_ROOT = ROOT / "reference" / "OUT" / "orc_calibrate_250919_sen" / "arg2_1.0"
DEFAULT_OUTPUT = ROOT / "docs" / "source_audits" / "paper_restart_schema_mosaic.json"
COMPONENTS = {
    "driver": ("driver_start.nc", ROOT / "docs" / "source_audits" / "driver_restart_netcdf_schema.json"),
    "sechiba": ("sechiba_start.nc", ROOT / "docs" / "source_audits" / "sechiba_restart_netcdf_schema.json"),
    "stomate": ("stomate_start.nc", ROOT / "docs" / "source_audits" / "stomate_restart_netcdf_schema.json"),
}


def _structural(schema: dict) -> dict:
    scatter = schema["single_landpoint_scatter"]
    return {
        key: schema[key]
        for key in ("file_format", "dimensions", "physical_variables", "variables")
    } | {
        "single_landpoint_scatter": {
            "index_g": scatter["index_g"],
            "grid_shape": scatter["grid_shape"],
        }
    }


def scan(mosaic_root: Path = DEFAULT_ROOT) -> dict:
    components = {}
    for component, (filename, schema_path) in COMPONENTS.items():
        expected = _structural(json.loads(schema_path.read_text(encoding="utf-8")))
        expected_text = json.dumps(expected, sort_keys=True)
        paths = sorted(mosaic_root.rglob(filename))
        signatures = {
            json.dumps(_structural(extract_schema(path)), sort_keys=True)
            for path in paths
        }
        relative_paths = [path.relative_to(mosaic_root).as_posix() for path in paths]
        components[component] = {
            "filename": filename,
            "template_count": len(paths),
            "schema_variants": len(signatures),
            "all_match_expected": signatures == {expected_text},
            "path_list_sha256": hashlib.sha256("\n".join(relative_paths).encode()).hexdigest(),
            "expected_structure_sha256": hashlib.sha256(expected_text.encode()).hexdigest(),
        }
    return {
        "schema_version": 1,
        "scope": "669 independent paper landpoint driver/SECHIBA/STOMATE restart schemas",
        "mosaic_root": mosaic_root.relative_to(ROOT).as_posix(),
        "closed": all(
            item["template_count"] == 669
            and item["schema_variants"] == 1
            and item["all_match_expected"]
            for item in components.values()
        ),
        "components": components,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mosaic-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-closed", action="store_true")
    args = parser.parse_args()
    report = scan(args.mosaic_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.output.resolve()} closed={report['closed']}")
    return int(args.require_closed and not report["closed"])


if __name__ == "__main__":
    raise SystemExit(main())
