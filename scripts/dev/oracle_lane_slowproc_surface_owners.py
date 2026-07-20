from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.slowproc import (  # noqa: E402
    get_soilcorr_usda_source_routed,
    slowproc_change_frac_source_routed,
    slowproc_veget_explicit,
)
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "slowproc_surface_owners"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_slowproc_surface_owners.f90.template"
PROCEDURES = {
    "get_soilcorr_usda",
    "slowproc_veget",
    "slowproc_checkveget",
    "slowproc_change_frac",
}


def compose(path: Path) -> dict[str, str]:
    payload = TEMPLATE.read_bytes()
    hashes = {}
    for procedure in sorted(PROCEDURES):
        span = extract_procedure_bytes(SOURCE, procedure)
        payload = payload.replace(f"! <{procedure.upper()}>".encode(), span.span_bytes)
        hashes[procedure] = span.span_sha256
    path.write_bytes(payload)
    return hashes


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.reader(handle):
            values.setdefault(row[0], []).append(float(row[1]))
    return {name: np.asarray(value) for name, value in values.items()}


def _veget_expected(ok_dgvm: bool) -> dict[str, np.ndarray]:
    lai = np.zeros((3, 14)); lai[:, 13] = [2.0, 1.0, 1.5]
    frac = np.array([[0.1, 0.2], [1.0, 0.0], [1e-8, 1e-8]])
    vmax = np.zeros((3, 14)); vmax[0, 0] = 0.2; vmax[0, 13] = 0.5; vmax[2, 0] = 0.3; vmax[2, 13] = 0.7; vmax[:, 6] = 5e-7
    result = slowproc_veget_explicit(
        lai=lai, frac_nobio=frac, veget_max=vmax,
        pref_soil_veg=np.r_[np.ones(7, int), np.full(7, 2, int)],
        ext_coeff_vegetfrac=np.full(14, 0.5), nstm=3, ok_dgvm=ok_dgvm,
    )
    prefix = "dgvm_" if ok_dgvm else "static_"
    return {
        prefix + "frac_nobio": np.asarray(result.frac_nobio).ravel(order="F"),
        prefix + "veget_max": np.asarray(result.veget_max).ravel(order="F"),
        prefix + "veget": np.asarray(result.veget).ravel(order="F"),
        prefix + "soiltile": np.asarray(result.soiltile).ravel(order="F"),
        prefix + "totfrac_nobio": np.asarray(result.totfrac_nobio),
        prefix + "biomass": np.arange(1.0, 43.0),
    }


def _change_expected(peat: bool) -> dict[str, np.ndarray]:
    lai = np.zeros((2, 14)); lai[:, 13] = [2.0, 1.0]
    base = np.zeros((2, 14)); base[:, 0] = [0.2, 0.3]; base[:, 13] = [0.7, 0.5]
    adjusted = np.zeros((2, 14)); adjusted[:, 0] = [0.3, 0.4]; adjusted[:, 13] = [0.6, 0.4]
    result = slowproc_change_frac_source_routed(
        agri_peat=peat, veget_max_new=base, veget_max_adjusted=adjusted,
        frac_nobio_new=np.array([[0.1, 0.0], [0.1, 0.0]]), lai=lai,
        pref_soil_veg=np.r_[np.ones(7, int), np.full(7, 2, int)],
        ext_coeff_vegetfrac=np.full(14, 0.5), nstm=3,
    )
    prefix = "change_peat_" if peat else "change_base_"
    vegetation = result.vegetation
    return {
        prefix + "frac_nobio": np.asarray(vegetation.frac_nobio).ravel(order="F"),
        prefix + "veget_max": np.asarray(vegetation.veget_max).ravel(order="F"),
        prefix + "veget": np.asarray(vegetation.veget).ravel(order="F"),
        prefix + "soiltile": np.asarray(vegetation.soiltile).ravel(order="F"),
        prefix + "totfrac_nobio": np.asarray(vegetation.totfrac_nobio),
        prefix + "tot_bare_soil": np.asarray(result.tot_bare_soil),
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_slowproc_surface_") as temp:
        build = Path(temp); source = build / "oracle.f90"; executable = build / "oracle.exe"
        hashes = compose(source); metadata = compile_fortran(source, executable, compiler)
        output = output_dir / "fortran_outputs.csv"
        subprocess.run([str(executable), str(output.resolve())], cwd=build,
                       env=compiler_environment(compiler), check=True, capture_output=True, text=True)
    actual = _read(output_dir / "fortran_outputs.csv")
    expected = {"usda_table": np.asarray(get_soilcorr_usda_source_routed()).ravel(order="F")}
    expected.update(_veget_expected(False)); expected.update(_veget_expected(True))
    expected.update(_change_expected(False)); expected.update(_change_expected(True))
    write_point_comparisons(output_dir / "point_comparisons.csv", actual, expected, rtol=1e-12, atol=1e-14)
    comparisons = [float_comparison(name, actual[name], expected[name], rtol=1e-12, atol=1e-14) for name in sorted(actual)]
    (output_dir / "inputs.json").write_text(json.dumps({"schema_version": 1, "cases": ["static", "dgvm", "change_base", "change_peat"]}, indent=2) + "\n", encoding="ascii")
    return write_result(output_dir, FAMILY, comparisons, {
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "procedure_span_sha256": hashes, "compiler": metadata,
    })


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
