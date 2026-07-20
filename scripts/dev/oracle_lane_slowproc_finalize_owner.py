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

from jax_orchidee.sechiba.slowproc import slowproc_finalize_restart_packet  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER, compile_fortran, compiler_environment, float_comparison,
    write_point_comparisons, write_result,
)

FAMILY = "slowproc_finalize_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_slowproc_finalize_owner.f90.template"
PROCEDURES = {"slowproc_finalize"}


def compose(path: Path) -> dict[str, str]:
    span = extract_procedure_bytes(SOURCE, "slowproc_finalize")
    path.write_bytes(TEMPLATE.read_bytes().replace(b"! <SLOWPROC_FINALIZE>", span.span_bytes))
    return {"slowproc_finalize": span.span_sha256}


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.reader(handle):
            values.setdefault(row[0], []).append(float(row[1]))
    return {name: np.asarray(value) for name, value in values.items()}


def _state() -> dict[str, np.ndarray | int]:
    lai = np.arange(1.0, 29.0).reshape((2, 14), order="F") / 10.0
    return {
        "veget": lai + 2, "veget_max": lai + 3, "lai": lai,
        "frac_nobio": np.array([[0.1, 0.3], [0.2, 0.4]]),
        "frac_age": np.full((2, 14, 2), 5.0), "njsc": np.array([4.0, 12.0]),
        "reinf_slope": np.array([0.2, 0.3]), "clayfraction": np.array([0.11, 0.22]),
        "sandfraction": np.array([0.33, 0.44]), "height": lai + 1,
        "laimap": np.full((2, 14, 12), 15.0), "veget_year": 1999,
        "peatPET_lastyear": np.array([1.0, 2.0]), "growth_day": np.array([3.0, 4.0]),
        "GSL": np.array([5.0, 6.0]), "peatPET_thisyear": np.array([7.0, 8.0]),
        "precipitation_lastsummer": np.array([9.0, 10.0]),
        "precipitation_thissummer": np.array([11.0, 12.0]),
        "summerpet_long": np.array([13.0, 14.0]), "summerp_long": np.array([15.0, 16.0]),
        "peatC": np.array([17.0, 18.0]), "peatC_ok": np.array([1.0, 0.0]),
        "soil_ph": np.array([6.5, 7.5]), "poor_soils": np.array([0.1, 0.2]),
        "bulk_density": np.array([1100.0, 1200.0]),
    }


def _expected() -> dict[str, np.ndarray]:
    state = _state()
    rename = {
        "reinf_slope": "reinf_slope", "clayfraction": "clay_frac", "sandfraction": "sand_frac",
        "peatPET_lastyear": "peatPET_last", "peatPET_thisyear": "peatPET_this",
        "precipitation_lastsummer": "precipitation_last",
        "precipitation_thissummer": "precipitation_this",
        "summerpet_long": "summerpet_longterm", "summerp_long": "summerp_longterm",
        "bulk_density": "bulk_dens",
    }
    expected: dict[str, np.ndarray] = {}
    for prefix, switches in (
        ("all_", dict(hydrol_cwrr=True, read_lai=True, map_pft_format=True, ok_stomate=True)),
        ("base_", dict(hydrol_cwrr=False, read_lai=False, map_pft_format=False, ok_stomate=False)),
    ):
        packet, call_stomate = slowproc_finalize_restart_packet(state=state, **switches)
        for name, value in packet.items():
            output_name = rename.get(name, name)
            expected[prefix + output_name] = np.asarray(value).ravel(order="F")
        if call_stomate:
            expected[prefix + "stomate_finalize_called"] = np.array([1.0])
    return expected


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_slowproc_finalize_") as temp:
        build = Path(temp); source = build / "oracle.f90"; executable = build / "oracle.exe"
        hashes = compose(source); metadata = compile_fortran(source, executable, compiler)
        output = output_dir / "fortran_outputs.csv"
        subprocess.run([str(executable), str(output.resolve())], cwd=build,
                       env=compiler_environment(compiler), check=True, capture_output=True, text=True)
    actual = _read(output_dir / "fortran_outputs.csv"); expected = _expected()
    write_point_comparisons(output_dir / "point_comparisons.csv", actual, expected, rtol=0.0, atol=0.0)
    comparisons = [float_comparison(name, actual[name], expected[name], rtol=0.0, atol=0.0) for name in sorted(actual)]
    (output_dir / "inputs.json").write_text(json.dumps({"schema_version": 1, "cases": ["all_switches", "base"]}, indent=2) + "\n", encoding="ascii")
    return write_result(output_dir, FAMILY, comparisons, {
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "procedure_span_sha256": hashes, "compiler": metadata,
    })


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2)); raise SystemExit(0 if result["status"] == "passed" else 1)
