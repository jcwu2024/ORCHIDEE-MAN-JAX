"""Actual native-like partial NetCDF -> initializer -> thermal consumer evidence.

Reuses C66's immutable same-input original initialization boundary. A build
executes source-extracted original initialize/thermal procedures; tests require
only the portable fixture. The file factory is synthetic, never an IOIPSL
producer identity. No complete model step, day or year executes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "scripts/dev"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from jax_orchidee.driver.orchestration import _thermosoil_restart_namespace_from_cold_start  # noqa: E402
from jax_orchidee.parameters.pft_catalog import (  # noqa: E402
    PFTRunLayout,
    build_selected_pft_run_layout,
    load_pft_catalog,
)
from jax_orchidee.sechiba import thermosoil as th  # noqa: E402
from scripts.dev import oracle_physics_branch_thermal as codec  # noqa: E402
from scripts.dev import oracle_thermosoil_full_component as full  # noqa: E402
from scripts.dev import oracle_thermosoil_initialization as accepted  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
)

FIXTURE = ROOT / "tests/fixtures/phase06_native_partial_reference.json"
C66 = ROOT / "tests/fixtures/thermosoil_partial_marker_reference.json"
FIELDS = tuple(accepted.COEFFICIENTS)
RTOL = ATOL = 1.e-12
VAL_EXP = 999999.
PFT_FIELDS = {"ptn", "shum_ngrnd_perma", "shum_ngrnd_prmlng", "e_soil_lat", "soilcap_pft", "soilflx_pft", "cgrnd", "dgrnd"}
CASES = [(f"{kind}_{field}", kind, field) for kind in ("absent", "all_val_exp") for field in FIELDS]
CASES += [("mixed_overwritten", "mixed_overwritten", "cgrnd"), ("complete_retained", "retained", None)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def values(record):
    return {name: codec.decode(value) for name, value in record["inputs"].items()}


def canonical_restart(record, kind, field):
    v = values(record)
    restart = {name: codec.decode(record["initial_expected"]["initial." + name]) for name in FIELDS}
    restart.update(ptn=v["ptn"], refSOC=v["refSOC"], shum_ngrnd_prmlng=v["shum_ngrnd_permalong"],
        gtemp=np.zeros(1), e_soil_lat=np.zeros((1, 14)))
    if kind == "absent":
        restart.pop(field)
    elif kind in ("all_val_exp", "mixed_overwritten"):
        restart[field] = np.full_like(restart[field], VAL_EXP)
    elif kind not in ("retained", "retained_mixed"):
        raise ValueError(kind)
    if kind in ("mixed_overwritten", "retained_mixed"):
        restart["soilcap_pft"][0, 13] = VAL_EXP
    return restart


def layouts():
    catalog = load_pft_catalog(ROOT / "configs/pft_catalogs/orchidee_man_paper_250919.json")
    entries = tuple(sorted(catalog.entries, key=lambda entry: entry.fortran_pft_id))
    target = build_selected_pft_run_layout(catalog, layout_id="phase06-normal",
        pft_ids=[entry.pft_id for entry in entries], fractions=np.ones(14))
    order = (0, *range(13, 0, -1))
    source = PFTRunLayout(target.catalog_id, "phase06-permuted", tuple(target.entries[i] for i in order),
        tuple(target.fractions[i] for i in order))
    return source, target, order


def write_native_like(path, restart, *, permuted=False):
    """Write explicit native-like axes and a conflicting second time record."""
    dataset = xr.Dataset(attrs={"fixture_kind": "synthetic_native_like_NOT_IOIPSL_producer"})
    order = layouts()[2] if permuted else tuple(range(14))
    for name, raw in restart.items():
        a = np.asarray(raw)
        if name in PFT_FIELDS:
            a = np.take(a, order, axis=a.ndim - 1)
        if a.ndim == 3:
            level = "z_g" if name in ("cgrnd", "dgrnd") else "z_f"
            dims = ("l_d", level, "y", "x")
            data = a[0].T[:, :, None, None]
        elif a.ndim == 2 and name in PFT_FIELDS:
            dims, data = ("l_d", "y", "x"), a[0, :, None, None]
        elif a.ndim == 2:
            dims, data = ("z_d" if name == "refSOC" else "z_snow", "y", "x"), a[0, :, None, None]
        else:
            dims, data = ("y", "x"), a.reshape(1, 1)
        dataset[name] = (("time", *dims), np.stack((data, data + 7.)))
    dataset.to_netcdf(path, format="NETCDF3_64BIT", engine="scipy")
    dataset.close()


def read_native_like(path, *, permuted=False):
    kwargs = {}
    if permuted:
        kwargs["source_pft_layout"], kwargs["target_pft_layout"], _ = layouts()
    return th.read_thermosoil_initialize_restart_fields(path, **kwargs)


def run_chain(record, path, *, permuted=False):
    """Keep C66 physical inputs, replacing its raw-mapping seam by actual IO."""
    v = values(record)
    restart = read_native_like(path, permuted=permuted)
    original_initialize = th.thermosoil_initialize
    original_closure = th.thermosoil_cold_start_coef_closure
    initialized, closures = [], []

    def observe_closure(**kwargs):
        result = original_closure(**kwargs)
        closures.append(result)
        return result

    def initialize_from_file(**kwargs):
        result = original_initialize(**{**kwargs, "restart_fields": restart})
        initialized.append(result)
        return result

    with patch.object(th, "thermosoil_initialize", initialize_from_file), patch.object(
        th, "thermosoil_cold_start_coef_closure", observe_closure
    ):
        arrays, flags = accepted.jax_results(v, {"cold_default_boundary": False})
    assert len(initialized) == 1
    result = initialized[0]
    # The actual maintained recurrence adapter consumes the real closure
    # computed inside initialize. No substitute physical arrays are generated.
    state = _thermosoil_restart_namespace_from_cold_start(ptn=result.ptn, coef=closures[-1],
        refsoc=result.refsoc, temp_sol_beg=result.temp_sol_beg,
        shum_ngrnd_permalong=result.shum_ngrnd_permalong)
    if result.calculate_coef:
        for name in ("cgrnd", "dgrnd", "cgrnd_snow", "dgrnd_snow", "lambda_snow"):
            np.testing.assert_array_equal(getattr(state, name), getattr(result, name))
    flags.update(actual_NetCDF_reader=True, raw_fields=sorted(restart),
        maintained_recurrence_adapter=True, authentic_IOIPSL_producer=False)
    return arrays, flags, result, state


def component_inputs(record, initial_arrays):
    v = values(record)
    result = {name: v[name] for name in (
        "QZ", "SMCMAX", "mcs", "so_capa_dry_ns", "zlt", "dlt", "dz1", "refSOC", "znt",
        "organic_layer_thick", "soilc_total", "snow", "snowrho", "snowtemp", "snowdz", "pb",
        "veget_max", "frac_snow_veg", "frac_snow_nobio", "totfrac_nobio", "njsc", "scalars", "ok_lai_i")}
    result["dz5"] = (v["zlt"][:-1] - v["znt"][:-1]) * v["dz1"]
    result.update({name: initial_arrays["moisture." + name] for name in accepted.MOISTURE_FIELDS})
    result.update({name: initial_arrays["initial." + name] for name in FIELDS})
    result.update({name: initial_arrays["initial." + name] for name in ("pcapa_en", "temp_sol_beg")})
    result["ptn"] = initial_arrays["initial.ptn_beg"]
    result["shum_ngrnd_permalong"] = v["shum_ngrnd_permalong"]
    result["temp_sol_new"] = v["post_temp_sol"]
    result["temp_sol_new_pft"] = v["temp_sol_new_pft"]
    result["controls"] = np.array([1, 1], dtype=np.int32)
    return result


def next_component(record, arrays, result, state, mode):
    """Execute the real owner operands, then its next profile/energy kernels.

    The unchanged C66 continuation is not admitted by the later getdiff guard;
    never publish a full-step result or relax that guard to extend this gate.
    """
    import jax

    from jax_orchidee.sechiba import coupling

    inputs = component_inputs(record, arrays)
    kwargs = full.owner_kwargs(inputs)
    kwargs["thermosoil_restart"] = state
    kwargs["enerbil_payload"]["soilcap"] = result.soilcap
    operands, boundaries = {}, []
    original_assembler = coupling.assemble_thermosoil_explicit_kwargs

    class ComponentBoundary(Exception):
        pass

    def capture(**actual):
        operands.update(actual)
        raise ComponentBoundary

    def assemble(**actual):
        boundary = original_assembler(**actual)
        boundaries.append(boundary)
        return boundary

    with patch.object(coupling, "assemble_thermosoil_explicit_kwargs", assemble), patch.object(
        th, "thermosoil_explicit_step", capture
    ):
        try:
            th.run_thermosoil_first_step_module(**kwargs, use_jit=False)
        except ComponentBoundary:
            pass
    assert operands and len(boundaries) == 1
    payload = boundaries[0].advertised_payload
    for field in ("cgrnd", "dgrnd", "cgrnd_snow", "dgrnd_snow", "lambda_snow"):
        np.testing.assert_array_equal(payload[field], getattr(result, field))
    np.testing.assert_array_equal(operands["pcapa_en_previous"], arrays["initial.pcapa_en"])
    np.testing.assert_array_equal(operands["soilcap_initial"], result.soilcap)
    np.testing.assert_array_equal(operands["temp_sol_beg"], result.temp_sol_beg)
    profile = th.thermosoil_profile_explicit_snow(**{name: operands[name] for name in (
        "ptn", "cgrnd", "dgrnd", "cgrnd_snow", "dgrnd_snow", "temp_sol_new", "snowtemp", "veget_max",
        "frac_snow_veg", "frac_snow_nobio", "totfrac_nobio", "veget_mask_2d")}, nslm=11)
    energy_kwargs = accepted.production_energy_kwargs(operands, profile)
    energy_call = jax.jit(th.thermosoil_energy_diagnostics) if mode == "jit" else th.thermosoil_energy_diagnostics
    energy = energy_call(**energy_kwargs)
    actual = {"consumer.ptn": np.asarray(profile.ptn), "consumer.stempdiag": np.asarray(profile.stempdiag)}
    actual.update({"consumer." + name: np.asarray(getattr(energy, name)) for name in (
        "surfheat_incr", "coldcont_incr", "ptn_beg", "temp_sol_beg")})
    return payload, actual


def full_component_rejection(record, arrays, result, state, mode):
    """Preserve the full-component source guard before downstream publication."""
    kwargs = full.owner_kwargs(component_inputs(record, arrays))
    kwargs["thermosoil_restart"] = state
    kwargs["enerbil_payload"]["soilcap"] = result.soilcap
    try:
        th.run_thermosoil_first_step_module(**kwargs, use_jit=mode == "jit")
    except Exception as exc:
        if "profil_froz differs across PFTs; Fortran aborts this path" not in str(exc):
            raise
        return {"mode": mode, "type": type(exc).__name__, "message": str(exc),
            "source_guard_preserved": True, "downstream_published": False}
    raise AssertionError("source-rejected full thermal continuation was unexpectedly admitted")


def _binary(path, inputs):
    with path.open("wb") as handle:
        for a in inputs.values():
            handle.write(np.asarray(a, dtype="<i4" if a.dtype.kind in "iu" else "<f8").ravel(order="F").tobytes())


def _reference(directory, executable, inputs, layout):
    directory.mkdir()
    binary, output = directory / "inputs.bin", directory / "outputs.bin"
    _binary(binary, inputs)
    command = [str(executable.resolve()), str(binary.resolve()), str(output.resolve())]
    process = subprocess.run(command, cwd=directory, env=compiler_environment(DEFAULT_COMPILER),
        capture_output=True, text=True, timeout=30)
    (directory / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (directory / "stderr.log").write_text(process.stderr, encoding="utf-8")
    write_json(directory / "execution.json", {"command": command, "returncode": process.returncode,
        "input_sha256": sha(binary), "executable_sha256": sha(executable)})
    process.check_returncode()
    raw, offset, result = np.fromfile(output, dtype="<f8"), 0, {}
    for name, shape in layout:
        count = int(np.prod(shape))
        result[name] = raw[offset:offset + count].reshape(shape, order="F")
        offset += count
    assert offset == raw.size
    np.savez(directory / "reference.npz", **result)
    return result


def build(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    (output / "runner.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "invocation.json", {"argv": sys.argv, "cwd": str(Path.cwd()),
        "python": sys.executable, "runner_sha256": sha(Path(__file__)),
        "production_identity": {str(path.relative_to(ROOT)): sha(path) for path in (
            ROOT / "jax_orchidee/sechiba/thermosoil.py", ROOT / "jax_orchidee/sechiba/coupling.py",
            ROOT / "jax_orchidee/driver/orchestration.py")}})
    c66 = json.loads(C66.read_text(encoding="utf-8"))
    record = {"schema": "phase06-native-partial-reference-v1", "rtol": RTOL, "atol": ATOL,
        "inputs": c66["inputs"], "initial_expected": c66["expected"], "cases": CASES,
        "origin": {"C66_fixture": str(C66.relative_to(ROOT)), "C66_fixture_sha256": sha(C66),
            "C66_original": c66["origin"]["original"],
            "scope": "synthetic native-like NetCDF normalization and creation seam; not an authentic IOIPSL producer or natural history"}}
    v = values(record)
    accepted.SPANS.clear()
    unit, layout = accepted.compile_unit(v)
    original_group = b"".join(accepted.THERMAL.read_bytes().splitlines(keepends=True)[664:714]).decode("utf-8")
    assert unit.count(original_group) == 1
    unit = unit.replace(original_group, original_group + "\n write(ou) merge(1._r_std,0._r_std,calculate_coef)\n")
    insertion = next(i for i, (name, _) in enumerate(layout) if name.startswith("transient."))
    layout.insert(insertion, ("source.calculate_coef", (1,)))
    source, executable = output / "initialize.f90", output / "initialize.exe"
    source.write_text(unit, encoding="utf-8", newline="\n")
    compile_record = compile_fortran(source, executable)
    write_json(output / "initialize-compile.json", compile_record)
    write_json(output / "initialize-spans.json", accepted.SPANS)
    write_json(output / "initialize-layout.json", layout)
    summaries = []
    record["case_expected_overrides"] = {}
    for name, kind, field in CASES:
        path = output / (name + ".nc")
        raw_restart = canonical_restart(record, kind, field)
        write_native_like(path, raw_restart)
        actual_read = read_native_like(path)
        for key, array in raw_restart.items():
            np.testing.assert_array_equal(actual_read[key], array)
        same_input = {key: array.copy() for key, array in v.items()}
        for coefficient in FIELDS:
            same_input["rest_" + coefficient] = actual_read.get(coefficient,
                np.full(accepted.COEFFICIENTS[coefficient], VAL_EXP))
        reference = _reference(output / (name + "-fortran"), executable, same_input, layout)
        expected_calculate = kind != "retained"
        assert bool(reference["source.calculate_coef"][0]) is expected_calculate
        actual, flags, _, _ = run_chain(record, path)
        np.savez(output / (name + "-jax.npz"), **actual)
        record["case_expected_overrides"][name] = {key: codec.encode(reference[key])
            for key in record["initial_expected"]
            if not np.array_equal(reference[key], codec.decode(record["initial_expected"][key]))}
        checks = [float_comparison(key, reference[key], actual[key], rtol=RTOL, atol=ATOL)
            for key in record["initial_expected"]]
        summary = {"case": name, "file_sha256": sha(path), "source_calculate_coef": expected_calculate,
            "actual_calculate_coef": flags["calculate_coef"], "checks": checks,
            "passed": all(check["passed"] for check in checks)}
        summaries.append(summary)
        print(json.dumps({"case": name, "passed": summary["passed"]}), flush=True)
    write_json(output / "creation-comparisons.json", summaries)
    original_initial = {key: codec.decode(value) for key, value in record["initial_expected"].items()}
    original_component_inputs = component_inputs(record, original_initial)
    source_text, component_layout = full.compile_unit(original_component_inputs)
    source, executable = output / "component.f90", output / "component.exe"
    source.write_text(source_text, encoding="utf-8", newline="\n")
    write_json(output / "component-compile.json", compile_fortran(source, executable))
    write_json(output / "component-spans.json", full.initial.SPANS)
    write_json(output / "component-layout.json", component_layout)
    try:
        _reference(output / "component-fortran", executable, original_component_inputs, component_layout)
    except subprocess.CalledProcessError as exc:
        stdout = (output / "component-fortran/stdout.log").read_text(encoding="utf-8")
        if "thermosoil_getdiffprofil_froz_mean wrong" not in stdout:
            raise
        source_rejection = {"returncode": exc.returncode, "message": stdout.strip(),
            "reason": "PFT-dependent frozen fraction after profile; source thermosoil_getdiff 2805-2807",
            "downstream_published": False}
    else:
        raise AssertionError("unchanged C66 full continuation unexpectedly escaped its original source guard")
    write_json(output / "full-component-source-rejection.json", source_rejection)
    component_reference = {key: array for key, array in original_initial.items() if key.startswith("consumer.")}
    record["component_inputs"] = {name: codec.encode(array) for name, array in original_component_inputs.items()}
    record["component_expected"] = {name: codec.encode(array) for name, array in component_reference.items()}
    record["full_component_source_rejection"] = source_rejection
    path = output / "absent_cgrnd.nc"
    arrays, flags, result, state = run_chain(record, path)
    modes = []
    for mode in ("eager", "jit"):
        payload, actual = next_component(record, arrays, result, state, mode)
        checks = [float_comparison(key, component_reference[key], actual[key], rtol=RTOL, atol=ATOL)
            for key in component_reference]
        np.savez(output / f"component-{mode}.npz", **actual)
        guard = full_component_rejection(record, arrays, result, state, mode)
        modes.append({"mode": mode, "checks": checks, "passed": all(check["passed"] for check in checks),
            "actual_owner_operand_assembler": True, "next_profile_energy_only": True,
            "energy_jit": mode == "jit", "boundary_fields": sorted(payload), "full_component_guard": guard})
    write_json(output / "component-comparisons.json", modes)
    record["source_identity"] = {str(path.relative_to(ROOT)): sha(path) for path in (
        accepted.THERMAL, accepted.HYDROL, accepted.SECHIBA, accepted.PARAMETERS)}
    record["original_execution"] = {"initial_compile_sha256": sha(output / "initialize.f90"),
        "component_compile_sha256": sha(output / "component.f90"), "original_creation_cases": len(CASES)}
    write_json(FIXTURE, record)
    write_json(output / "portable-fixture-identity.json", {"path": str(FIXTURE), "sha256": sha(FIXTURE)})
    import jax
    write_json(output / "runtime.json", {"python": sys.executable, "thermosoil_import": th.__file__,
        "jax_version": jax.__version__, "jax_x64": bool(jax.config.jax_enable_x64),
        "devices": [str(device) for device in jax.devices()], "elapsed_seconds": time.perf_counter() - started,
        "baseline": "e15005067403b36fbb34e3f3ad7b7c9b9cd204b2", "production_edits": [],
        "formal_counter": "72/38 unchanged", "numerical_acceptance": False})
    return all(case["passed"] for case in summaries) and all(mode["passed"] for mode in modes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    try:
        passed = build(args.build.resolve())
    except Exception as exc:
        if args.build.is_dir():
            write_json(args.build / "failure.json", {"type": type(exc).__name__,
                "message": str(exc), "traceback": traceback.format_exc()})
        raise
    raise SystemExit(0 if passed else 1)
