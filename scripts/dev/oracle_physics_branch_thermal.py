"""PB-03 original-source thermal increments and portable actual-component seam.

Build uses the unchanged source extraction from the accepted full thermal
oracle. Verify/tests need only the checked-in fixture, never ignored outputs.
Cases are source-consumable synthetic restored states, not natural histories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts/dev"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import oracle_thermosoil_full_component as accepted  # noqa: E402
from fortran_oracle_common import (  # noqa: E402
    COMPILE_FLAGS,
    DEFAULT_COMPILER,
    compiler_environment,
)

FIXTURE = ROOT / "tests/fixtures/physics_branch_thermal_reference.json"
RTOL = ATOL = 1.e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def encode(array):
    a = np.asarray(array)
    unique, inverse = np.unique(a, return_inverse=True)
    record = {"shape": list(a.shape), "dtype": str(a.dtype)}
    if unique.size == 1:
        record["fill"] = unique[0].item()
    elif a.ndim == 3 and np.array_equal(a, np.broadcast_to(a[:, :, :1], a.shape)):
        record["pft_broadcast"] = a[:, :, 0].tolist()
    elif unique.size * 3 < a.size:
        record.update(values=unique.tolist(), indices=inverse.tolist())
    else:
        record["data"] = a.tolist()
    return record


def decode(record):
    shape, dtype = tuple(record["shape"]), record["dtype"]
    if "fill" in record:
        return np.full(shape, record["fill"], dtype=dtype)
    if "pft_broadcast" in record:
        return np.broadcast_to(np.asarray(record["pft_broadcast"], dtype=dtype)[:, :, None], shape).copy()
    if "indices" in record:
        return np.asarray(record["values"], dtype=dtype)[record["indices"]].reshape(shape)
    return np.asarray(record["data"], dtype=dtype).reshape(shape)


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def decode_inputs(case):
    return {name: decode(value) for name, value in case["inputs"].items()}


def run_production(inputs, mode="eager"):
    """Return real owner closure; caller may replace declared moisture fields.

    inputs: original source array dictionary. K temperature; kg/m3 snowrho;
    m snowdz/grid; hPa pb; volumetric m3/m3 mc/mcl; integrated mm tmc;
    unitless saturation shumdiag_perma; gC/m3 refSOC. No hidden data lookup.
    """
    from jax_orchidee.sechiba import thermosoil
    if mode not in ("eager", "jit"):
        raise ValueError(mode)
    result = thermosoil.run_thermosoil_first_step_module(
        **accepted.owner_kwargs(inputs), use_jit=mode == "jit")
    if not result.ok:
        raise ValueError(result.missing_inputs)
    return result


def flatten_result(result):
    return accepted.flatten_result(result.module, True)


def build_hydrol_consumer(output):
    """Append one new same-input original thermal reference from real HYDROL."""
    from scripts.dev import oracle_physics_branch_hydrol as hy
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    fixture = load_fixture()
    if any(c["name"] == "hydrol_mineral_rain_canopy" for c in fixture["cases"]):
        raise ValueError("joint reference already exists")
    hydro_case = next(c for c in hy.make_cases() if c["name"] == "mineral_rain_canopy")
    producer = hy.run_production(hydro_case["inputs"], "eager")
    inputs = decode_inputs(fixture["cases"][0])
    for field in accepted.initial.MOISTURE_FIELDS:
        inputs[field] = np.asarray(getattr(producer.thermosoil_moisture, field))
    inputs["njsc"] = np.asarray(hydro_case["inputs"]["njsc"], dtype=np.int32)
    source, layout = accepted.compile_unit(inputs)
    sourcepath = output / "oracle.f90"
    sourcepath.write_text(source, encoding="utf-8", newline="\n")
    executable = output / "oracle.exe"
    command = [str(DEFAULT_COMPILER), *COMPILE_FLAGS, str(sourcepath.resolve()), "-o", str(executable.resolve())]
    process = subprocess.run(command, cwd=output, env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True, timeout=60)
    dump(output / "compile.json", {"command": command, "returncode": process.returncode, "stderr": process.stderr, "source_sha256": sha(sourcepath)})
    process.check_returncode()
    binary, out = output / "inputs.bin", output / "outputs.bin"
    with binary.open("wb") as handle:
        for a in inputs.values():
            handle.write(np.asarray(a, dtype="<i4" if a.dtype.kind in "iu" else "<f8").ravel(order="F").tobytes())
    process = subprocess.run([str(executable.resolve()), str(binary.resolve()), str(out.resolve())], cwd=output,
        env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True, timeout=20)
    dump(output / "execution.json", {"returncode": process.returncode, "stderr": process.stderr})
    process.check_returncode()
    reference, offset = {}, 0
    raw = np.fromfile(out, dtype="<f8")
    for field, shape in layout:
        size = int(np.prod(shape))
        reference[field] = raw[offset:offset+size].reshape(shape, order="F")
        offset += size
    assert offset == raw.size
    def portable(value):
        if isinstance(value, dict):
            return {k: portable(a) for k, a in value.items()}
        return value.tolist() if isinstance(value, np.ndarray) else value
    hydro_payload = portable(hydro_case["inputs"])
    identity = hashlib.sha256(json.dumps(hydro_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    case = {"name": "hydrol_mineral_rain_canopy", "inputs": {k: encode(a) for k, a in inputs.items()},
        "expected": {k: encode(a) for k, a in reference.items()}, "inputs_binary_sha256": sha(binary),
        "Fortran_output_binary_sha256": sha(out),
        "origin": {"domain": "source_consumable_actual_HYDROL_moisture_synthetic_thermal_restored_state_not_natural_history",
            "producer": "actual run_hydrol_first_step_module_from_precall -> hydrol_to_thermosoil_moisture_inputs",
            "producer_mode": "eager", "producer_case": hydro_case["name"], "producer_inputs": hydro_payload,
            "producer_inputs_canonical_sha256": identity, "producer_script_sha256": sha(hy.__file__),
            "producer_hydrol_sha256": sha(ROOT / "jax_orchidee/sechiba/hydrol.py"),
            "moisture_fields": list(accepted.initial.MOISTURE_FIELDS),
            "unit_phase": "post-HYDROL export: volumetric mc/mcl m3/m3; integrated tmc mm; saturation shumdiag_perma unitless",
            "temperature_K": 280., "thermal_refSOC_gC_m3": 0., "hydrol_soil_class": 3,
            "identity_limit": "same-input reference binds captured eager producer; later JIT producer may differ numerically and must be compared separately"}}
    fixture["cases"].append(case)
    dump(FIXTURE, fixture)
    checks = {}
    for mode in ("eager", "jit"):
        actual = flatten_result(run_production(inputs, mode))
        checks[mode] = accepted.compare(reference, actual)
    dump(output / "comparisons.json", checks)
    dump(output / "receipt.json", {"case": case["name"], "producer_inputs_canonical_sha256": identity,
        "same_input_thermal_fields": len(reference), "strict_failures": {mode: [c["name"] for c in rows if not c["passed"]] for mode, rows in checks.items()},
        "fixture_sha256": sha(FIXTURE), "source_spans": list(accepted.initial.SPANS), "new_Fortran_processes": 1,
        "actual_HYDROL_producer_calls": 1, "actual_thermal_calls": 2, "numerical_acceptance": False})
    return case


def cases():
    base, origin = accepted.saved_input("attempt03/complete_no_snow", True)
    configurations = (
        ("dry_warm_refsoc_zero_clear", 280., 0., 0., [0., 0., 0.], False),
        ("wet_frozen_refsoc_cap_deep", 268., .43, 130000., [.05, .5, .95], True),
        ("freeze_lower_equal_refsoc_half_thin", 272.15, .12, 65000., [.003, .004, .003], True),
        ("freeze_upper_equal_refsoc_above_thick", 274.15, .31, 195000., [.05, .2, .25], False),
        ("mixed_depth_freeze_refsoc_ramp", None, .2, None, [.02, .04, .02], True),
    )
    result = []
    for name, temperature, water, soc, snowdz, memory in configurations:
        v = {k: a.copy() for k, a in base.items()}
        profile = np.linspace(269., 277., 32) if temperature is None else np.full(32, temperature)
        v["ptn"][:] = profile[None, :, None]
        v["cgrnd"][:] = profile[None, 1:, None]
        v["dgrnd"][:] = 0.
        v["cgrnd_snow"][:] = profile[0]
        v["dgrnd_snow"][:] = 0.
        v["temp_sol_beg"][:] = profile[0] - .1
        v["temp_sol_new"][:] = profile[0] + .2
        v["temp_sol_new_pft"][:] = profile[0] + .2
        v["refSOC"][:] = np.linspace(0., 260000., 32)[None, :] if soc is None else soc
        v["snowdz"][:] = snowdz
        v["snowrho"][:] = [150., 300., 450.]
        v["snowtemp"][:] = [265., 269., 272.]
        v["snow"][:] = np.sum(v["snowrho"] * v["snowdz"])
        v["frac_snow_veg"][:] = .7 if sum(snowdz) > 0 else 0.
        v["frac_snow_nobio"][:] = 0.
        v["controls"][1] = int(memory)
        v["shum_ngrnd_permalong"][:] = .8
        # Integrated moisture obeys the source HYDROL layer geometry, not an
        # independently invented tmc tuple. Ice fraction is declared synthetic.
        from jax_orchidee.parameters.vertical_soil import vertical_soil_init
        thickness_mm = np.asarray(vertical_soil_init().dlh) * 1000.
        for prefix in ("mc", "mcl", "tmc"):
            values = np.full((1, 11), water)
            if prefix == "mcl":
                values *= .6
            if prefix == "tmc":
                values *= thickness_mm[None, :]
            v[f"{prefix}_layh"][:] = values
            v[f"{prefix}_layh_pft"][:] = values[:, :, None]
        v["shumdiag_perma"][:] = water / float(v["SMCMAX"][int(v["njsc"][0]) - 1])
        result.append((name, v, {**origin, "domain": "source_consumable_synthetic_restored_state_not_natural_history",
            "changed_fields": [k for k in v if not np.array_equal(v[k], base[k])],
            "temperature_target_K": profile.tolist(), "moisture_m3_m3": water}))
    return result


def build(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    chosen = cases()
    source, layout = accepted.compile_unit(chosen[0][1])
    sourcepath = output / "oracle.f90"
    sourcepath.write_text(source, encoding="utf-8", newline="\n")
    executable = output / "oracle.exe"
    command = [str(DEFAULT_COMPILER), *COMPILE_FLAGS, str(sourcepath.resolve()), "-o", str(executable.resolve())]
    process = subprocess.run(command, cwd=output, env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True, timeout=60)
    dump(output / "compile.json", {"command": command, "returncode": process.returncode,
        "stdout": process.stdout, "stderr": process.stderr, "source_sha256": sha(sourcepath), "compiler_sha256": sha(DEFAULT_COMPILER)})
    process.check_returncode()
    spans = list(accepted.initial.SPANS)
    fixture = {"schema": "physics-branch-thermal-v1", "rtol": RTOL, "atol": ATOL,
        "source_files": {str(accepted.initial.THERMAL.relative_to(ROOT)).replace("\\", "/"): sha(accepted.initial.THERMAL)},
        "source_spans": spans, "cases": []}
    summaries = []
    for name, inputs, origin in chosen:
        directory = output / name
        directory.mkdir()
        raw_input, raw_output = directory / "inputs.bin", directory / "outputs.bin"
        with raw_input.open("wb") as handle:
            for a in inputs.values():
                handle.write(np.asarray(a, dtype="<i4" if a.dtype.kind in "iu" else "<f8").ravel(order="F").tobytes())
        process = subprocess.run([str(executable.resolve()), str(raw_input.resolve()), str(raw_output.resolve())],
            cwd=directory, env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True, timeout=20)
        dump(directory / "execution.json", {"returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr})
        process.check_returncode()
        raw = np.fromfile(raw_output, dtype="<f8")
        reference, offset = {}, 0
        for field, shape in layout:
            size = int(np.prod(shape))
            reference[field] = raw[offset:offset + size].reshape(shape, order="F")
            offset += size
        assert offset == raw.size
        case = {"name": name, "origin": origin, "inputs": {k: encode(a) for k, a in inputs.items()},
            "expected": {k: encode(a) for k, a in reference.items()},
            "inputs_binary_sha256": sha(raw_input), "Fortran_output_binary_sha256": sha(raw_output)}
        fixture["cases"].append(case)
        runs = []
        for mode in ("eager", "jit"):
            actual = flatten_result(run_production(inputs, mode))
            checks = accepted.compare(reference, actual)
            record = {"case": name, "mode": mode, "passed": all(c["passed"] for c in checks), "fields": checks}
            dump(directory / f"{mode}.json", record)
            runs.append(record)
            print(json.dumps({k: record[k] for k in ("case", "mode", "passed")}), flush=True)
        summaries.append({"name": name, "passed": all(r["passed"] for r in runs)})
    dump(FIXTURE, fixture)
    dump(output / "source-spans.json", spans)
    receipt = {"cases": summaries, "all_passed": all(c["passed"] for c in summaries),
        "production_file": "jax_orchidee/sechiba/thermosoil.py", "production_sha256": sha(ROOT / "jax_orchidee/sechiba/thermosoil.py"),
        "runner_sha256": sha(__file__), "fixture_sha256": sha(FIXTURE), "original_Fortran_processes": len(chosen),
        "new_full_model_steps": 0, "numerical_acceptance": False}
    dump(output / "receipt.json", receipt)
    return receipt


def build_snow_predecessor(output, reuse_reference=None):
    """Two adjacent complete snow calls, with source state carried unchanged.

    The initial deep pack is synthetic. This establishes a legal immediate
    predecessor of the zero-layer transfer condition, not natural climate
    reachability. No previous maxmass/grain reference is reexecuted.
    """
    import csv
    from unittest.mock import patch

    import oracle_snow_maxmass_domain as snow_reference

    from jax_orchidee.sechiba import hydrol
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    case = dict(id="regular_positive_predecessor", mode="path",
        domain="synthetic_positive_regular_pack_with_source_adjacent_predecessor",
        rho=[600.] * 3, dz=[.05, .5, 9.45], temperature=[268.] * 3,
        liquid=[0.] * 3, soilcap=0., vevapsno=31., snowfall=0., profile_temperature=268.)
    with patch.object(snow_reference, "cases", lambda: [case]):
        spans = snow_reference._source(output)
    sourcepath = output / "oracle.f90"
    source = sourcepath.read_text(encoding="ascii")
    next_call = """
 capture_case=1;cap=800._r_std*chalfu0;evap=0
 call capture_phase('entry',n,rho,dz,st,liq,heat,grain,gflux)
 call explicitsnow_main(n,rain,fall,tair,pb,u,v,ts,cap,pg,frac,totnb,gtemp,lambda,cs,ds,evap,age,nbage,nb,rho,grain,dz,st,heat,snow,tadd,liq,subnb,gflux,melt,totm,resid,sink)
 call capture_phase('return',n,rho,dz,st,liq,heat,grain,gflux)
 call emit('return','snow',0,snow(1));call emit('return','snowmelt',0,melt(1))
 call emit('return','tot_melt',0,totm(1));call emit('return','vevapsno',0,evap(1))
 call emit('return','subsinksoil',0,sink(1));call emit('return','temp_sol_add',0,tadd(1))
"""
    assert source.count("close(10)") == 1
    sourcepath.write_text(source.replace("close(10)", next_call + "close(10)"), encoding="ascii")
    executable = output / "oracle.exe"
    command = [str(DEFAULT_COMPILER), *COMPILE_FLAGS, str(sourcepath.resolve()), "-o", str(executable.resolve())]
    csvpath = output / "outputs.csv"
    if reuse_reference is None:
        process = subprocess.run(command, cwd=output, env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True, timeout=60)
        dump(output / "compile.json", {"command": command, "returncode": process.returncode, "stderr": process.stderr, "source_sha256": sha(sourcepath)})
        process.check_returncode()
        process = subprocess.run([str(executable.resolve()), str(csvpath.resolve())], cwd=output,
            env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True, timeout=20)
        dump(output / "execution.json", {"returncode": process.returncode, "stderr": process.stderr})
        process.check_returncode()
    else:
        if sha(sourcepath) != sha(reuse_reference / "oracle.f90"):
            raise ValueError("source identity changed")
        csvpath.write_bytes((reuse_reference / "outputs.csv").read_bytes())
        dump(output / "execution.json", {"new_Fortran_calls": 0, "reused": str(reuse_reference), "reference_sha256": sha(csvpath)})
    reference = {}
    for row in csv.DictReader(csvpath.open(encoding="ascii")):
        reference.setdefault((int(row["case"]), row["phase"], row["field"]), []).append(float(row["value"]))
    inputs = snow_reference._path_inputs(case)
    inputs["snowheat"] = np.asarray(reference[(0, "entry", "snowheat")]).reshape(1, 3)
    fixture = load_fixture()
    portable = {"initial_inputs": {k: encode(a) for k, a in inputs.items()}, "calls": [],
        "source_file": str(snow_reference.SNOW_SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha(snow_reference.SNOW_SOURCE), "procedure_spans": spans,
        "domain": "synthetic_regular_positive_pack_to_supported_adjacent_zero_layer_predecessor_not_natural_climate_history"}
    from extract_fortran_micro_oracle import extract_procedure_bytes
    portable["procedure_source_spans"] = []
    for name, digest in spans.items():
        path = snow_reference.QSAT_SOURCE if name.startswith("snow3l") else snow_reference.SNOW_SOURCE
        procedure = extract_procedure_bytes(path, name)
        assert procedure.span_sha256 == digest
        portable["procedure_source_spans"].append(dict(name=name, path=path.relative_to(ROOT).as_posix(),
            first=procedure.start_line, last=procedure.end_line, sha256=digest))
        fixture["source_files"][path.relative_to(ROOT).as_posix()] = sha(path)
    actual_inputs = dict(inputs)
    checks = []
    carry = ("snowrho", "snowgrain", "snowdz", "snowtemp", "snowheat", "snowliq", "snow_age", "snow_nobio_age", "snow_nobio")
    for index in (0, 1):
        if index:
            actual_inputs["vevapsno"] = np.zeros(1)
            actual_inputs["soilcap"] = np.asarray([800. * snow_reference.LF])
        result = hydrol.hydrol_explicit_snow_step(**actual_inputs)
        expected = {k[2]: np.asarray(a) for k, a in reference.items() if k[:2] == (index, "return")}
        comparisons = []
        for name, values in expected.items():
            current = (np.sum(np.asarray(result.snow_state.snowrho) * np.asarray(result.snow_state.snowdz), axis=1)
                if name == "mass" else getattr(result.snow_state, name) if name in result.snow_state._fields else getattr(result, name))
            comparisons.append(snow_reference.float_comparison(name, values, np.asarray(current).ravel(), rtol=snow_reference.RTOL, atol=snow_reference.ATOL))
        checks.append({"call": index, "comparisons": comparisons})
        portable["calls"].append({"expected": {k: encode(a) for k, a in expected.items()},
            "actual_complete_inputs": {k: encode(a) for k, a in actual_inputs.items()},
            "actual_carry_after": {k: encode(getattr(result.snow_state, k) if k in result.snow_state._fields else getattr(result, k)) for k in carry},
            "source_unobserved_carry_fields": ["snow_age", "snow_nobio_age", "snow_nobio"],
            "entry_snowdz": reference[(index, "entry", "snowdz")],
            "transfer_snowdz": reference[(index, "transfer", "snowdz")],
            "pre_maxmass_snowdz": reference[(index, "sublimation", "snowdz")],
            "post_maxmass_snowdz": reference[(index, "maxmass", "snowdz")]})
        for name in carry:
            actual_inputs[name] = getattr(result.snow_state, name) if name in result.snow_state._fields else getattr(result, name)
        actual_inputs["snow"] = result.snow_state.snow
    fixture["snow_predecessor"] = portable
    dump(FIXTURE, fixture)
    dump(output / "comparisons.json", checks)
    dump(output / "receipt.json", {"original_Fortran_main_calls": 2, "actual_production_main_calls": 2,
        "source_sha256": portable["source_sha256"], "fixture_sha256": sha(FIXTURE),
        "strict_failures": [{"call": r["call"], "field": c["name"]} for r in checks for c in r["comparisons"] if not c["passed"]],
        "zero_layer_immediate_predecessor_demonstrated": reference[(1, "entry", "snowdz")][0] == 0.,
        "natural_climate_history_demonstrated": False, "numerical_acceptance": False})
    return portable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snow-predecessor", action="store_true")
    parser.add_argument("--reuse-snow-reference", type=Path)
    parser.add_argument("--hydrol-consumer", action="store_true")
    args = parser.parse_args()
    if args.hydrol_consumer:
        build_hydrol_consumer(args.output.resolve())
    elif args.snow_predecessor:
        build_snow_predecessor(args.output.resolve(), args.reuse_snow_reference)
    else:
        report = build(args.output.resolve())
        raise SystemExit(0 if report["all_passed"] else 1)
