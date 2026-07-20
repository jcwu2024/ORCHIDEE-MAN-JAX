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

from jax_orchidee.sechiba.thermosoil import thermosoil_finalize_restart_packet  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)


FAMILY = "thermosoil_finalize_state_io"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_thermosoil_finalize_state_io.f90.template"
CONTRACTS = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
NPTS, NGRND, NVM, NSNOW = 2, 3, 14, 3


def compose(path: Path) -> str:
    span = extract_procedure_bytes(SOURCE, "thermosoil_finalize")
    path.write_bytes(TEMPLATE.read_bytes().replace(b"! <THERMOSOIL_FINALIZE>", span.span_bytes))
    return span.span_sha256


def _arrays() -> dict[str, np.ndarray]:
    ii, jj, kk = (axis + 1 for axis in np.indices((NPTS, NGRND, NVM)))
    i2, j2 = (axis + 1 for axis in np.indices((NPTS, NGRND)))
    ic, jc, kc = (axis + 1 for axis in np.indices((NPTS, NGRND - 1, NVM)))
    ip, kp = (axis + 1 for axis in np.indices((NPTS, NVM)))
    isn, jsn = (axis + 1 for axis in np.indices((NPTS, NSNOW)))
    return {
        "ptn": 260.0 + ii + 0.1 * jj + 0.001 * kk,
        "refSOC": 100.0 * i2 + j2,
        "shum_ngrnd_prmlng": 0.1 * ii + 0.01 * jj + 0.001 * kk,
        "shum_ngrnd_perma": 0.2 * ii + 0.01 * jj + 0.001 * kk,
        "e_soil_lat": 30.0 * ip + 0.1 * kp,
        "cgrnd": 10.0 * ic + jc + 0.01 * kc,
        "dgrnd": 20.0 * ic + jc + 0.01 * kc,
        "gtemp": np.asarray([271.0, 272.0]),
        "soilcap": np.asarray([81.0, 82.0]),
        "soilcap_pft": 40.0 * ip + 0.1 * kp,
        "soilflx": np.asarray([91.0, 92.0]),
        "soilflx_pft": 50.0 * ip + 0.1 * kp,
        "cgrnd_snow": 60.0 * isn + jsn,
        "dgrnd_snow": 70.0 * isn + jsn,
        "lambda_snow": np.asarray([0.3, 0.7]),
    }


def _jax_outputs(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    packet = thermosoil_finalize_restart_packet(
        ptn=values["ptn"], refsoc=values["refSOC"],
        shum_ngrnd_perma=values["shum_ngrnd_perma"],
        shum_ngrnd_permalong=values["shum_ngrnd_prmlng"],
        cgrnd=values["cgrnd"], dgrnd=values["dgrnd"], gtemp=values["gtemp"],
        soilcap=values["soilcap"], soilcap_pft=values["soilcap_pft"],
        soilflx=values["soilflx"], soilflx_pft=values["soilflx_pft"],
        cgrnd_snow=values["cgrnd_snow"], dgrnd_snow=values["dgrnd_snow"],
        lambda_snow=values["lambda_snow"], ok_shum_ngrnd_permalong=True,
        ok_Ecorr=True, e_soil_lat=values["e_soil_lat"],
    )
    return {name: np.asarray(value).ravel(order="C") for name, value in packet.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _write_owner_evidence(output_dir: Path) -> None:
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    owner = next(
        record for record in contracts["owner_regions"]
        if record["fortran_procedure"] == "thermosoil_finalize"
        and record["fortran_file"] == SOURCE.relative_to(ROOT).as_posix()
    )
    proofs = json.loads(SOURCE_PROOFS.read_text(encoding="utf-8"))
    passed = {record["arm_id"] for record in proofs["records"] if record.get("passed") is True}
    required = set(owner["arm_ids"])
    covered = required & passed
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "contract": "Fortran restput_p field selection and complete array payload comparison",
        "required_arm_ids": sorted(required),
        "source_proof_arm_ids": sorted(covered),
        "complete": required == covered,
    }
    (output_dir / "state_io_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")
    evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": required == covered,
        "records": [{
            "owner_region_id": owner["owner_region_id"],
            "base_region_id": owner["base_region_id"],
            "fortran_procedure": owner["fortran_procedure"],
            "jax_owners": owner["jax_owners"],
            "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json",
            "state_io_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/state_io_coverage.json",
            "required_arm_ids": sorted(required),
            "covered_arm_ids": sorted(covered),
            "missing_arm_ids": sorted(required - covered),
            "passed": required == covered,
        }],
    }
    (output_dir / "owner_region_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_thermosoil_finalize_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        span_hash = compose(source)
        compiler_meta = compile_fortran(source, executable, compiler)
        output_path = output_dir / "fortran_outputs.csv"
        subprocess.run([str(executable), str(output_path.resolve())], cwd=build,
                       env=compiler_environment(compiler), check=True, capture_output=True, text=True)
    values = _arrays()
    fortran = _read(output_dir / "fortran_outputs.csv")
    jax = _jax_outputs(values)
    if set(fortran) != set(jax):
        raise RuntimeError(f"restart fields differ: Fortran={sorted(fortran)}, JAX={sorted(jax)}")
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14)
    comparisons = [float_comparison(name, fortran[name], jax[name], rtol=1e-12, atol=1e-14)
                   for name in sorted(fortran)]
    (output_dir / "inputs.json").write_text(json.dumps({"schema_version": 1, "kjit": 17,
        "branches": {"ok_shum_ngrnd_permalong": True, "ok_Ecorr": True},
        "fields": sorted(fortran)}, indent=2) + "\n", encoding="ascii")
    result = write_result(output_dir, FAMILY, comparisons, {
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "procedure_span_sha256": {"thermosoil_finalize": span_hash}, "compiler": compiler_meta,
        "verified_ledger_entries": [],
    })
    if result["status"] == "passed":
        _write_owner_evidence(output_dir)
    return result


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
