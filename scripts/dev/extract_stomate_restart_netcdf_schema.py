from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "docs" / "source_audits" / "stomate_restart_netcdf_schema.json"
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def extract_schema(
    template_path: str | Path,
    *,
    scope: str = "ORCHIDEE-MAN PFT14 independent single-landpoint STOMATE restart",
    fortran_file: str = "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90",
    subroutine: str = "writerestart",
    lines: str = "1751-2944",
) -> dict[str, Any]:
    template = Path(template_path).resolve()
    with Dataset(template) as dataset:
        variables: dict[str, Any] = {}
        for name, variable in dataset.variables.items():
            attributes = {
                attribute: _json_value(variable.getncattr(attribute))
                for attribute in variable.ncattrs()
                if attribute != "_FillValue"
            }
            declaration: dict[str, Any] = {
                "dtype": str(variable.dtype),
                "dimensions": list(variable.dimensions),
                "attributes": attributes,
                "chunking": variable.chunking(),
                "endian": variable.endian(),
            }
            if "_FillValue" in variable.ncattrs():
                declaration["fill_value"] = _json_value(
                    variable.getncattr("_FillValue")
                )
            variables[name] = declaration

        try:
            source_reference = template.relative_to(ROOT).as_posix()
        except ValueError:
            source_reference = str(template)
        return {
            "schema_version": 1,
            "scope": scope,
            "source_reference": source_reference,
            "fortran_provenance": {
                "file": fortran_file,
                "subroutine": subroutine,
                "lines": lines,
                "physical_file_owner": "IOIPSL restini/restput_p",
            },
            "file_format": dataset.file_format,
            "dimensions": {
                name: {"size": len(dimension), "unlimited": dimension.isunlimited()}
                for name, dimension in dataset.dimensions.items()
            },
            "global_attributes": {
                name: _json_value(dataset.getncattr(name))
                for name in dataset.ncattrs()
            },
            "physical_variables": [
                "nav_lon",
                "nav_lat",
                "nav_lev",
                "time",
                "time_steps",
            ],
            "single_landpoint_scatter": {
                "index_g": [1],
                "grid_shape": [1, 1],
                "fortran_provenance": {
                    "file": fortran_file,
                    "subroutine": subroutine,
                    "lines": f"{lines} restput_p scatter calls",
                },
            },
            "variables": variables,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scope",
        default="ORCHIDEE-MAN PFT14 independent single-landpoint STOMATE restart",
    )
    parser.add_argument(
        "--fortran-file",
        default="fortran_source/ORCHIDEE/src_stomate/stomate_io.f90",
    )
    parser.add_argument("--subroutine", default="writerestart")
    parser.add_argument("--lines", default="1751-2944")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    schema = extract_schema(
        args.template,
        scope=args.scope,
        fortran_file=args.fortran_file,
        subroutine=args.subroutine,
        lines=args.lines,
    )
    text = json.dumps(schema, indent=2, ensure_ascii=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != text:
            raise SystemExit(f"schema differs from {args.output}")
        print(f"OK {args.output.resolve()}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(
        f"WROTE {args.output.resolve()} dimensions={len(schema['dimensions'])} "
        f"variables={len(schema['variables'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
