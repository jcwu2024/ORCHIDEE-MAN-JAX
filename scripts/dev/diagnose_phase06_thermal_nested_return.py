"""Preserve and diagnose the saved complete-return nested thermal strict gate.

Only a narrow source-backed component runs. No model step or state publication
is executed. Original failure assets and maintained numerical gates are immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import checkify  # noqa: E402

from jax_orchidee.sechiba import thermosoil  # noqa: E402
from scripts.dev import oracle_physics_branch_thermal as oracle  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
)

DEFAULT_OUTPUT = ROOT / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/thermal-nested-return"
OLD = ROOT / "outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/controller-repair/validation/local-initial"
OLD_NAMES = ("execution.json", "imports.json", "initial-interface-test.py", "launcher.py", "pytest.log", "pytest.xml")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def flatten(value, prefix=""):
    if hasattr(value, "_asdict"):
        values = value._asdict()
    elif isinstance(value, dict):
        values = value
    elif isinstance(value, SimpleNamespace):
        values = vars(value)
    else:
        return {prefix: np.asarray(value)} if hasattr(value, "dtype") or isinstance(value, (float, int, bool)) else {}
    result = {}
    for name, child in values.items():
        result.update(flatten(child, prefix + "." + name if prefix else name))
    return result


def identity(arrays):
    records = []
    total = hashlib.sha256()
    for name, value in arrays.items():
        array = np.asarray(value)
        raw = array.tobytes(order="C")
        header = json.dumps([name, str(array.dtype), list(array.shape)], separators=(",", ":")).encode()
        total.update(len(header).to_bytes(8, "little") + header + len(raw).to_bytes(8, "little") + raw)
        records.append({"name": name, "shape": list(array.shape), "dtype": str(array.dtype), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return {"layout_and_bytes_sha256": total.hexdigest(), "fields": records}


def compare(expected, actual, rtol=1e-12, atol=1e-12):
    assert expected.keys() == actual.keys()
    rows = []
    for name, e in expected.items():
        a = np.asarray(actual[name])
        e = np.asarray(e)
        assert a.shape == e.shape and a.dtype == e.dtype, name
        absolute = np.abs(a.astype(float) - e.astype(float))
        passing = np.isclose(a, e, rtol=rtol, atol=atol)
        if a.tobytes() != e.tobytes():
            maximum = int(np.argmax(absolute))
            rows.append({"field": name, "shape": list(e.shape), "max_abs": float(absolute.max(initial=0)),
                         "strict_failed_count": int(np.count_nonzero(~passing)), "count": a.size,
                         "max_flat_index": maximum, "expected_at_max": float(e.ravel()[maximum]), "actual_at_max": float(a.ravel()[maximum]),
                         "failed_flat_indices": np.flatnonzero(~passing).tolist()})
    return rows


def original_assets():
    return [{"path": str((OLD / name).relative_to(ROOT)).replace("\\", "/"), "bytes": (OLD / name).stat().st_size, "sha256": digest(OLD / name)} for name in OLD_NAMES]


def boundary_schema(boundary):
    bound = inspect.signature(thermosoil.thermosoil_explicit_step).bind(**boundary)
    bound.apply_defaults()
    return {name: {"kind": "None"} if value is None else {"kind": "numeric", "shape": list(np.shape(value)), "dtype": str(value.dtype if hasattr(value, "dtype") else np.asarray(value).dtype)}
            for name, value in bound.arguments.items()}


def coefficient_reference(output, inputs, baseline, nested):
    """Original exact coefficient span on one frozen current material tuple."""
    values = {
        "ptn": np.asarray(baseline.profile.ptn), "pcapa": np.asarray(baseline.getdiff.pcapa),
        "pkappa": np.asarray(baseline.getdiff.pkappa),
        "temp_sol_new": inputs["temp_sol_new"], "temp_sol_new_pft": inputs["temp_sol_new_pft"],
        "dlt": inputs["dlt"], "dz1": inputs["dz1"],
        "dt_input": np.asarray([float(inputs["scalars"][0])]),
        "lambda_input": np.asarray([float(inputs["znt"][0] * inputs["dz1"][0])]),
        "phigeoth_input": np.asarray([thermosoil.PHIGEOTH]),
        "lai_input": inputs["ok_lai_i"].astype(np.int32), "mask_input": np.ones((1, 14), dtype=np.int32),
    }
    sourcepath = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
    source_lines = sourcepath.read_bytes().splitlines(keepends=True)
    span = b"".join(source_lines[1517:1593])
    declarations = []
    for name, value in values.items():
        kind = "integer" if value.dtype.kind in "iu" else "real(r_std)"
        declarations.append(f" {kind}::{name}({','.join(map(str, value.shape))})")
    fields = {"cgrnd": (1, 31, 14), "dgrnd": (1, 31, 14), "zdz1": (1, 31, 14), "zdz2": (1, 32, 14),
              "soilcap_pft": (1, 14), "soilflx_pft": (1, 14), "soilcap_pft_nosnow": (1, 14), "soilflx_pft_nosnow": (1, 14)}
    header = """program coefficient_fragment
 use iso_fortran_env,only:real64
 implicit none
 integer,parameter::r_std=real64,nvm=14,ngrnd=32
 real(r_std),parameter::un=1._r_std
 real(r_std)::z1(1),cgrnd(1,31,14),dgrnd(1,31,14),zdz1(1,32,14),zdz2(1,32,14)
 real(r_std)::soilcap_pft(1,14),soilflx_pft(1,14),soilcap_pft_nosnow(1,14),soilflx_pft_nosnow(1,14)
 real(r_std)::dt_sechiba,lambda,phigeoth
 logical::veget_mask_2d(1,14),ok_laidev(14)
 integer::jv,jg
 character(1024)::input,output
"""
    header += "\n".join(declarations)
    header += "\n call get_command_argument(1,input);call get_command_argument(2,output)\n"
    header += " open(10,file=trim(input),access='stream',form='unformatted',status='old')\n"
    header += "\n".join(" read(10) " + name for name in values)
    header += "\n close(10);dt_sechiba=dt_input(1);lambda=lambda_input(1);phigeoth=phigeoth_input(1)\n"
    header += " veget_mask_2d=mask_input/=0;ok_laidev=lai_input/=0\n"
    footer = "\n open(20,file=trim(output),access='stream',form='unformatted',status='replace')\n"
    footer += "\n".join(" write(20) " + ("zdz1(:,:31,:)" if name == "zdz1" else name) for name in fields)
    footer += "\n close(20)\nend program\n"
    unit = output / "coefficient-original.f90"
    unit.write_bytes(header.encode() + b"\n" + span + footer.encode())
    executable = output / "coefficient-original.exe"
    metadata = compile_fortran(unit, executable)
    binary = output / "coefficient-input.bin"
    with binary.open("wb") as handle:
        for value in values.values():
            handle.write(np.asarray(value, dtype="<i4" if value.dtype.kind in "iu" else "<f8").ravel(order="F").tobytes())
    resultfile = output / "coefficient-output.bin"
    process = subprocess.run([str(executable), str(binary), str(resultfile)], cwd=output,
                             env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True, check=True, timeout=20)
    raw = np.fromfile(resultfile, dtype="<f8")
    reference = {}
    offset = 0
    for name, shape in fields.items():
        count = int(np.prod(shape))
        reference[name] = raw[offset:offset + count].reshape(shape, order="F")
        offset += count
    np.savez(output / "coefficient-original.npz", **reference)
    write(output / "coefficient-input.json", {k: oracle.encode(v) for k, v in values.items()})
    write(output / "coefficient-source.json", {"path": str(sourcepath.relative_to(ROOT)).replace("\\", "/"),
          "start": 1518, "end": 1593, "sha256": hashlib.sha256(span).hexdigest(), "source_sha256": digest(sourcepath),
          "compile": metadata, "stdout": process.stdout, "stderr": process.stderr,
          "input_binary_sha256": digest(binary), "output_binary_sha256": digest(resultfile),
          "origin": "current production material/profile values frozen identically; original coefficient arithmetic only, no initialization or whole-model step"})
    comparisons = {}
    for label, module in (("baseline", baseline), ("nested", nested)):
        actual = {name: np.asarray(getattr(module.coef.soil, name)) for name in fields}
        comparisons[label] = compare(reference, actual)
    write(output / "coefficient-original-comparisons.json", comparisons)
    # Existing complete-return c/d arrays are nonperturbing stage observations.
    # Explain their propagation offline without calling this replay an actual
    # source or production observation.
    propagation = {}
    for label, module in (("baseline", baseline), ("nested", nested)):
        soil = module.coef.soil
        z1, z2 = np.asarray(soil.zdz1), np.asarray(soil.zdz2)
        c, d = np.asarray(soil.cgrnd), np.asarray(soil.dgrnd)
        temp = np.asarray(module.profile.ptn)
        base = z1[:, 0] * (c[:, 0] + (d[:, 0] - 1) * temp[:, 0])
        lamb = values["lambda_input"][0]
        surface_z1 = lamb * (1 - d[:, 0]) + 1
        delta = temp[:, 0] * surface_z1 - lamb * c[:, 0] - values["temp_sol_new"][:, None]
        correction = np.asarray(soil.soilcap_pft) * delta / values["dt_input"][0]
        propagation[label] = {"surface_c": c[:, 0].tolist(), "surface_d": d[:, 0].tolist(),
                              "surface_zdz1": z1[:, 0].tolist(), "surface_zdz2": z2[:, 0].tolist(),
                              "offline_base": base.tolist(), "offline_temperature_residual": delta.tolist(),
                              "offline_correction": correction.tolist(), "offline_flux": (base + correction).tolist(),
                              "observed_actual_flux": np.asarray(soil.soilflx_pft).tolist()}
    write(output / "offline-propagation.json", propagation)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    saved_assets = original_assets()
    record = oracle.load_fixture()
    case = record["cases"][0]
    inputs = oracle.decode_inputs(case)
    kwargs = oracle.accepted.owner_kwargs(inputs)
    write(output / "original-failure-assets.json", saved_assets)
    write(output / "inputs.json", {k: oracle.encode(v) for k, v in inputs.items()})
    write(output / "input-layout.json", identity(inputs))
    write(output / "complete-owner-tuple.json", {k: oracle.encode(v) for k, v in flatten(kwargs).items()})
    write(output / "complete-owner-layout.json", identity(flatten(kwargs)))
    np.savez(output / "inputs.npz", **inputs)
    actual_inner = thermosoil._thermosoil_explicit_step_jit
    captured = []

    def observe_inner(**boundary):
        captured.append(boundary)
        return actual_inner(**boundary)

    with patch.object(thermosoil, "_thermosoil_explicit_step_jit", observe_inner):
        baseline_closure = thermosoil.run_thermosoil_first_step_module(**kwargs, use_jit=True)
    assert baseline_closure.ok and len(captured) == 1
    baseline = baseline_closure.module
    boundary_baseline = flatten(captured[0])
    baseline_schema = boundary_schema(captured[0])
    write(output / "effective-boundary-baseline-schema.json", baseline_schema)

    def owner(ptn):
        restart = SimpleNamespace(**{**vars(kwargs["thermosoil_restart"]), "ptn": ptn})
        closure = thermosoil.run_thermosoil_first_step_module(**{**kwargs, "thermosoil_restart": restart}, use_jit=True)
        return closure.module

    nested_function = jax.jit(checkify.checkify(owner))
    source_error, nested = nested_function(jnp.asarray(inputs["ptn"]))
    checkify.check_error(source_error)
    baseline_arrays, nested_arrays = flatten(baseline), flatten(nested)
    write(output / "output-tree.json", {"baseline": str(jax.tree_util.tree_structure(baseline)),
          "nested": str(jax.tree_util.tree_structure(nested)), "identical": jax.tree_util.tree_structure(baseline) == jax.tree_util.tree_structure(nested)})
    np.savez(output / "baseline-complete.npz", **baseline_arrays)
    np.savez(output / "nested-complete.npz", **nested_arrays)
    write(output / "baseline-layout.json", identity(baseline_arrays))
    write(output / "nested-layout.json", identity(nested_arrays))
    differences = compare(baseline_arrays, nested_arrays, record["rtol"], record["atol"])
    write(output / "current-before.json", {"rtol": record["rtol"], "atol": record["atol"], "complete_return_leaves": len(baseline_arrays),
          "differences": differences, "strict_failures": [d for d in differences if d["strict_failed_count"]],
          "material_getdiff_differences": compare(flatten(baseline.getdiff), flatten(nested.getdiff)),
          "complete_return_strict_pass": not any(d["strict_failed_count"] for d in differences),
          "source_error_checked": True, "origin_commit": "7cae3e02adad23674767a590fdf4d27f239ce065", "production_fix_recommendation": "pending bounded attribution; do not change sums, formulas, tolerances or expected values"})
    write(output / "effective-boundary-baseline.json", {k: oracle.encode(v) for k, v in boundary_baseline.items()})

    # Return the real delegate inputs as additional observations; validate
    # endpoints and physical equation lists before treating them as evidence.
    captured.clear()
    nested_schemas = []

    def observed_owner(ptn):
        captured.clear()
        result = owner(ptn)
        nested_schemas.append(boundary_schema(captured[-1]))
        return result, {k: v for k, v in captured[-1].items() if hasattr(v, "dtype") or isinstance(v, (float, int, bool))}

    with patch.object(thermosoil, "_thermosoil_explicit_step_jit", observe_inner):
        observed_error, (observed_module, observed_boundary) = jax.jit(checkify.checkify(observed_owner))(jnp.asarray(inputs["ptn"]))
        observed_jaxpr = jax.make_jaxpr(checkify.checkify(observed_owner))(jnp.asarray(inputs["ptn"]))
    checkify.check_error(observed_error)
    base_jaxpr = jax.make_jaxpr(checkify.checkify(owner))(jnp.asarray(inputs["ptn"]))
    observed_arrays = flatten(observed_module)
    boundary_nested = flatten(observed_boundary)
    boundary_comparison = compare(boundary_baseline, boundary_nested)
    write(output / "effective-boundary-nested.json", {k: oracle.encode(v) for k, v in boundary_nested.items()})
    write(output / "effective-boundary-nested-schema.json", nested_schemas[-1])
    write(output / "boundary-layout.json", {"baseline": identity(boundary_baseline), "nested": identity(boundary_nested)})
    write(output / "nonperturbation.json", {"nested_complete_endpoint_differences": compare(nested_arrays, observed_arrays),
          "complete_binary_identical": all(v.tobytes() == observed_arrays[k].tobytes() for k, v in nested_arrays.items()),
          "physical_equation_lists_equal": str(base_jaxpr.jaxpr.eqns) == str(observed_jaxpr.jaxpr.eqns),
          "equation_counts": [len(base_jaxpr.jaxpr.eqns), len(observed_jaxpr.jaxpr.eqns)],
          "effective_input_differences": boundary_comparison})
    (output / "unobserved.jaxpr").write_text(str(base_jaxpr), encoding="utf-8")
    (output / "observed.jaxpr").write_text(str(observed_jaxpr), encoding="utf-8")
    expected = {k: oracle.decode(v) for k, v in case["expected"].items()}
    baseline_original = oracle.accepted.flatten_result(baseline, True)
    nested_original = oracle.accepted.flatten_result(nested, True)
    write(output / "published-original-comparison.json", {"baseline": compare(expected, baseline_original), "nested": compare(expected, nested_original)})
    coefficient_reference(output, inputs, baseline, nested)
    write(output / "environment.json", {"python": sys.executable, "module": thermosoil.__file__, "jax": jax.__version__, "x64": jax.config.x64_enabled, "devices": str(jax.devices()),
          "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "fixture_sha256": digest(oracle.FIXTURE),
          "production_sha256": digest(ROOT / "jax_orchidee/sechiba/thermosoil.py"), "old_assets_unchanged": saved_assets == original_assets(), "whole_model_steps": 0})
    print(json.dumps({"output": str(output), "differences": differences}, indent=2))


def finalize(output):
    """Read saved actual observations; do not compile or execute a component."""
    def load(name):
        return json.loads((output / name).read_text(encoding="utf-8"))

    with np.load(output / "baseline-complete.npz") as saved:
        baseline = {k: saved[k].copy() for k in saved.files}
    with np.load(output / "nested-complete.npz") as saved:
        nested = {k: saved[k].copy() for k in saved.files}
    with np.load(output / "coefficient-original.npz") as saved:
        original = {k: saved[k].copy() for k in saved.files}
    current = load("current-before.json")
    inputs = {k: oracle.decode(v) for k, v in load("inputs.json").items()}
    nonperturbation = load("nonperturbation.json")
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    check("six original failure assets unchanged", load("original-failure-assets.json") == original_assets())
    check("effective complete inner entry fields identical", not nonperturbation["effective_input_differences"])
    baseline_schema, nested_schema = load("effective-boundary-baseline-schema.json"), load("effective-boundary-nested-schema.json")
    check("all41 effective fields including two None defaults identical", len(baseline_schema) == 41 and baseline_schema == nested_schema)
    check("complete return pytree structure and static provenance equal", load("output-tree.json")["identical"])
    check("read-only observer preserves complete module bytes", nonperturbation["complete_binary_identical"])
    check("observer preserves physical JAXPR equation list", nonperturbation["physical_equation_lists_equal"])
    check("54 numeric return leaves preserved", len(baseline) == len(nested) == 54)
    for group in ("humlev", "profile", "energy", "getdiff", "final"):
        check(group + " all numeric returns byte-identical", all(v.tobytes() == nested[k].tobytes() for k, v in baseline.items() if k.startswith(group + ".")))
    for field in ("zdz1", "zdz2"):
        check(field + " coefficients byte-identical", baseline["coef.soil." + field].tobytes() == nested["coef.soil." + field].tobytes())
    failures = {d["field"]: d for d in current["strict_failures"]}
    check("preserved strict complete-return failure and alias", set(failures) == {"coef.soil.soilflx_pft", "coef.soil.soilflx_pft_nosnow"})
    check("13 original coordinates retained", failures["coef.soil.soilflx_pft"]["failed_flat_indices"] == list(range(13)))
    check("original observed strict magnitude retained", failures["coef.soil.soilflx_pft"]["max_abs"] == 5.341060926866703e-12)
    first = {}
    for pft, label in ((13, "global_first"), (0, "first_failed_pft")):
        selected = None
        for row in range(30, -1, -1):
            for field in ("cgrnd", "dgrnd"):
                e, a = baseline["coef.soil." + field][0, row, pft], nested["coef.soil." + field][0, row, pft]
                if e != a:
                    selected = {"field": field, "index": [0, row, pft], "baseline": float(e), "nested": float(a),
                                "baseline_hex": float(e).hex(), "nested_hex": float(a).hex(), "ulp": float((a - e) / np.spacing(e)),
                                "same_input_original": float(original[field][0, row, pft])}
                    break
            if selected:
                break
        first[label] = selected
    check("global first difference one ULP bottom d", first["global_first"]["index"] == [0, 30, 13] and first["global_first"]["field"] == "dgrnd" and abs(first["global_first"]["ulp"]) == 1)
    check("first affected PFT one ULP recursive c", first["first_failed_pft"]["index"] == [0, 27, 0] and abs(first["first_failed_pft"]["ulp"]) == 1)
    # Freeze the actual operands for the first affected-PFT recurrence. These
    # are existing complete-return observations, not a replacement rollout.
    operands = {}
    for label, arrays in (("baseline", baseline), ("nested", nested)):
        operands[label] = {"ptn_jg": float(arrays["profile.ptn"][0, 28, 0]),
                           "zdz2_jg": float(arrays["coef.soil.zdz2"][0, 28, 0]),
                           "zdz1_prev": float(arrays["coef.soil.zdz1"][0, 27, 0]),
                           "zdz1_jg": float(arrays["coef.soil.zdz1"][0, 28, 0]),
                           "previous_c": float(arrays["coef.soil.cgrnd"][0, 28, 0]),
                           "previous_d": float(arrays["coef.soil.dgrnd"][0, 28, 0])}
    check("first affected recurrence complete operands equal", operands["baseline"] == operands["nested"])
    first["first_failed_pft"]["operands"] = operands
    source = load("coefficient-source.json")
    source_file = ROOT / source["path"]
    span = b"".join(source_file.read_bytes().splitlines(keepends=True)[source["start"] - 1:source["end"]])
    check("new original source fragment identity matches", hashlib.sha256(span).hexdigest() == source["sha256"])
    origin = subprocess.check_output(["git", "show", "--format=fuller", "--name-status", "7cae3e02adad23674767a590fdf4d27f239ce065"], cwd=ROOT, text=True)
    (output / "candidate-origin.txt").write_text(origin, encoding="utf-8")
    write(output / "first-difference.json", first)
    fixture_path = ROOT / "tests/fixtures/phase06_thermal_nested_return_reference.json"
    portable = {"schema_version": 1, "scope": "saved complete-module nested-JIT interface strict gate; attribution does not certify numerical acceptance",
                "base_case": oracle.load_fixture()["cases"][0]["name"], "rtol": current["rtol"], "atol": current["atol"],
                "inputs": load("inputs.json"), "input_identity": identity(inputs),
                "complete_return_layout": [{k: v for k, v in item.items() if k != "sha256"} for item in load("baseline-layout.json")["fields"]],
                "preserved_strict_fields": list(failures), "preserved_failed_flat_indices": list(range(13)),
                "original_failure_assets": load("original-failure-assets.json"),
                "external_source_identity": {k: source[k] for k in ("path", "start", "end", "sha256", "source_sha256")}}
    write(fixture_path, portable)
    remaining = {"status": "bounded attribution complete; strict numerical discrepancy retained",
                 "complete_return_strict_pass": False, "production_change_recommended": False,
                 "missing_for_attribution": [], "numerical_acceptance_missing": "The saved complete-return strict gate still fails 13/14 in soilflx_pft and its nosnow alias. Material/getdiff equality is not complete-return acceptance.",
                 "out_of_scope": ["unique optimized instruction/bitwise chase", "formula/sum changes to chase equality", "tolerance or expected-value changes", "closed normal/ALT/dry/joint/snow-groundflux investigations", "whole-model or remote validation"]}
    write(output / "remaining.json", remaining)
    report = {"schema_version": 1, "status": remaining["status"], "classification": "same-input compilation-context floating-point coefficient recurrence and surface cancellation",
              "scope": "P05-PB03-NESTED-RETURN-STRICT only; CPU/x64 narrow THERMOSOIL component",
              "same_input": {"effective_fields": len(baseline_schema), "numeric_fields": len(load("effective-boundary-baseline.json")), "None_defaults": [k for k, v in baseline_schema.items() if v["kind"] == "None"], "entry_differences": [], "leaf_count": 54},
              "getdiff_layout": {"source_oracle_material_fields": ["pcapa", "pcapa_en", "pkappa", "profil_froz", "pcappa_supp", "pcapa_snow", "pkappa_snow"], "additional_current_diagnostics": ["poros_net", "zx1", "zx2"], "all_current_getdiff_fields_byte_identical": True, "complete_return_acceptance": False},
              "before": current, "first_difference": first, "nonperturbation": nonperturbation,
              "source_micro_reference": source, "new_same_material_comparisons": load("coefficient-original-comparisons.json"),
              "offline_propagation": load("offline-propagation.json"), "checks": checks, "checks_passed": len(checks),
              "environment": load("environment.json"), "remaining": remaining,
              "discovery": {"graph": "parent index_status/search_graph/check_index_coverage Transport closed; no fresh graph completeness claim", "old_generation": "2026-11-19T20:44:49Z is historical only; Fortran excluded", "fallback": "exact current source and bounded original thermosoil_coef span1518-1593"},
              "candidate_origin": "git show 7cae3e0 contains documentation management restores only; no thermal implementation delta; no old checkout executed",
              "portable_fixture": str(fixture_path.relative_to(ROOT)).replace("\\", "/")}
    write(output / "report.json", report)
    summary = """# Phase06 thermal nested complete-return strict gate

The saved complete-module nested JIT failure is reproduced on the same complete input tuple. It is bounded to coefficient-recurrence floating-point differences and cancellation in the surface flux. No source-formula defect is established and no production change is recommended. The original strict gate remains failed, with unchanged inputs, expectations and rtol/atol=1e-12; getdiff/material equality does not certify complete return.

## Entry, output and observation qualification

The package preserves hashes/bytes of all six immutable Phase04 local-initial failure assets. Their saved test returns the complete module; the current maintained interface test returns getdiff. The new independent strict-xfail test preserves the complete return obligation. There are 41 effective inner boundary fields (39 numeric and the identical None defaults soilc_total/zx1), and 54 numeric complete-module leaves. Complete owner tuple, binary input arrays, shapes/dtypes, output PyTree/static provenance, layouts and bytes are saved. The effective baseline and nested boundary values are identical. Additional boundary observation returns add no physical arithmetic: all complete-module endpoint bytes and the 8/8 outer JAXPR equation lists remain identical. This does not assert identical optimized HLO.

All humlev, profile, energy, getdiff and final leaves are byte-identical between current standalone and nested executions. All zdz1/zdz2 values also match. The complete return failure is coef.soil.soilflx_pft at flat indices0–12 (13/14), baseline -2.0753236387593805, nested -2.0753236387647216, maxabs5.341060926866703e-12. The complete layout also exposes soilflx_pft_nosnow with the same failure; the old test stopped at the first leaf. No duplicate physical failure is inferred from that alias.

## Bounded first difference and propagation

Existing full-return arrays provide nonperturbing recurrence observations. Globally the first bottom-up difference is dgrnd[0,30,13], one ULP. For the 13 failed PFTs the first relevant difference is the recurrence at zero-based row27 (source Fortran c/d index28, input layer29): cgrnd279.96399076875394 versus279.9639907687539, one ULP. Its complete six arithmetic operands, including the previous c/d at row28, are identical. dgrnd at the same relevant recurrence also differs by one ULP. See first-difference.json for exact floating encodings and operands. Source thermosoil.f90:1533–1541; current thermosoil.py:2766–2781.

At the top of the 13 affected PFT columns, cgrnd is identical and dgrnd differs by one ULP:0.9406391780886577 versus0.9406391780886576. The source surface_base combines c+(d−1)*ptn, two approximately16.6K terms, then multiplies by zdz1≈171.6796. Offline NumPy evaluation of those captured operands gives surface_base9.148926348649372e-12 versus3.659570539459749e-12. Their difference≈5.489e-12 explains the measured5.341e-12 scale, while compiled evaluation of the surface correction has its own rounding. The offline replay fluxes differ from the actual fluxes by at most5.90e-13; they are explanatory replay and are not substituted for observed production endpoints. Source thermosoil.f90:1564–1593; current thermosoil.py:2782–2801. No unique-instruction claim or bitwise chase is made.

A new micro-component compiles unchanged original thermosoil_coef lines1518–1593 with the exact current common ptn/pcapa/pkappa/material tuple. Its first relevant c/d values match the standalone values, providing an independent source-backed recurrence anchor. Its flux at PFT0 is -2.075323638765311; the nested flux differs by5.893063814710331e-13. This reference freezes current material values (including existing conductivity rounding) and is a new diagnostic only: it neither replaces the old oracle nor upgrades complete-return scientific acceptance. The published original full-component comparisons are retained separately, including their existing strict residuals. Earlier normal/ALT, dry/joint thermal and snow-groundflux attributions are not reopened.

## Validation and limits

Existing ORCJAX/python.exe -B, JAX0.10.1 CPU/x64; no environment installation, model step/day/year, remote/GPU/training/paid work, research-tree changes, stage/commit/push or shared production edits. The source fragment, binary IO, compiler flags/identity and source hash/span identity are retained. Portable input/output layout tests and the preserved complete-return strict-xfail are separate from external_data source identity. The graph transport was closed, so this uses exact source fallback without a fresh exhaustive graph-coverage claim. git show7cae3e0 contains management documents only; no old source checkout was restored or run.

Reproduce with `python -B scripts/dev/diagnose_phase06_thermal_nested_return.py`, then `--finalize-only`; use the existing ORCJAX interpreter. Finalize only reads saved evidence and writes the independent portable fixture/report. remaining.json explicitly retains complete-return numerical acceptance as unresolved. receipt.json and manifest.json record completed checks; stop-writing.json is the writer handoff boundary.
"""
    (output / "README.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"checks_passed": len(checks), "fixture": str(fixture_path), "report": str(output / "report.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--finalize-only", action="store_true")
    arguments = parser.parse_args()
    (finalize if arguments.finalize_only else run)(arguments.output.resolve())
