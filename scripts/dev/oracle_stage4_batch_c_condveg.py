"""Executable owner evidence for Stage 4 Batch C CONDVEG owners.

The generated Fortran source contains byte-exact spans for the three owners.
Its small dependency module is deliberately limited to the restart and
interpolation boundaries; it supplies deterministic returned fields rather
than reimplementing any owner process.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from extract_fortran_micro_oracle import extract_procedure_bytes
from fortran_oracle_common import (
    DEFAULT_COMPILER, ROOT, compile_fortran, compiler_environment,
    float_comparison, write_point_comparisons, write_result,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAMILY = "condveg_batch_c"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/condveg.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_stage4_batch_c_condveg.f90.template"
PROCEDURES = ("condveg_background_soilalb", "condveg_soilalb", "condveg_initialize")


def _compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    path.write_bytes(TEMPLATE.read_bytes().replace(
        b"! <CONDVEG_BATCH_C_PROCEDURES>", b"\n\n".join(span.span_bytes for span in spans.values())
    ))
    return {name: span.span_sha256 for name, span in spans.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {key: np.asarray(value, dtype=np.float64) for key, value in fields.items()}


def _jax_values() -> dict[str, np.ndarray]:
    from jax_orchidee.sechiba.condveg import condveg_soilalb_from_interpolation

    fractions = np.array([[1.0, 0.0, 0.0], [0.2, 0.0, 0.8], [0.0, 0.0, 0.0]])
    result = condveg_soilalb_from_interpolation(
        soilcolrefrac=fractions, asoilcol=np.array([1.0, 1.0, 0.0]),
        vis_dry=(0.3, 0.2, 0.1), nir_dry=(0.6, 0.4, 0.2),
        vis_wet=(0.15, 0.1, 0.05), nir_wet=(0.3, 0.2, 0.1),
        albsoil_vis=(0.225, 0.15, 0.075), albsoil_nir=(0.45, 0.3, 0.15),
    )
    return {
        "soil_dry": np.asarray(result.soilalb_dry).ravel(order="F"),
        "soil_wet": np.asarray(result.soilalb_wet).ravel(order="F"),
        "soil_moy": np.asarray(result.soilalb_moy).ravel(order="F"),
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_condveg_batch_c_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        hashes = _compose(source)
        executable = build / "oracle.exe"
        metadata = compile_fortran(source, executable, compiler)
        subprocess.run([str(executable), str(csv_path)], cwd=build,
                       env=compiler_environment(compiler), check=True, capture_output=True, text=True)
    fortran, jax = _read(csv_path), _jax_values()
    compared = {key: value for key, value in fortran.items() if key in jax}
    comparisons = [float_comparison(key, value, jax[key], rtol=1e-12, atol=1e-14)
                   for key, value in compared.items()]
    (output_dir / "inputs.json").write_text(json.dumps({
        "schema_version": 1,
        "owners": list(PROCEDURES),
        "branch_matrix": [
            "background allocation continuation and allocated deallocation",
            "soil all/partial/zero class, availability fallback, subthreshold weight",
            "initialize restart-present/missing, MODIS/ordinary, IMPOSE_AZE, roughness dispatch",
        ],
        "dependencies": "minimal explicit restart/interpolation boundary stubs",
    }, indent=2) + "\n", encoding="ascii")
    write_point_comparisons(output_dir / "point_comparisons.csv", compared, jax, rtol=1e-12, atol=1e-14)
    return write_result(output_dir, FAMILY, comparisons, {"span_sha256": hashes, "build": metadata})


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "passed")
