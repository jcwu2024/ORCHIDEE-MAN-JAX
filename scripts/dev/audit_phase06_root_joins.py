"""Finite routing numerical join and current module-order evidence binding.

The routing reference is a read-only original Fortran output, not regenerated
from JAX. The full SECHIBA or model step is never invoked by this runner.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/root-joins"
FIXTURE = ROOT / "tests/fixtures/phase06_routing_zero_reference.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture():
    original = ROOT / "outputs/reference_mode/micro_oracles/driver_routing_zero/fortran_outputs.csv"
    fields = {}
    with original.open(encoding="ascii", newline="") as stream:
        for name, value in csv.reader(stream):
            fields.setdefault(name, []).append(float(value))
    source = ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"
    lines = source.read_bytes().splitlines(keepends=True)
    spans = [{"start": first, "end": last,
              "sha256": hashlib.sha256(b"".join(lines[first - 1:last])).hexdigest()}
             for first, last in ((1236, 1248), (1252, 1262))]
    record = {
        "original_output": str(original.relative_to(ROOT)).replace("\\", "/"),
        "original_output_sha256": digest(original),
        "original_input": "arrays initially nine; original no-routing ELSE then daily unit conversion",
        "source": str(source.relative_to(ROOT)).replace("\\", "/"),
        "source_spans": spans, "npts": 2, "nflow": 3, "dt_sechiba": 1800.0,
        "expected": fields,
        "scope": "independent point-wise zero block; two cells do not establish multi-point routing",
    }
    FIXTURE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def routing_fields(*, compiled=False):
    import jax
    import jax.numpy as jnp

    from jax_orchidee.sechiba.main import routing_to_slowproc_writeback
    from jax_orchidee.sechiba.routing import routing_zero_outputs

    state = routing_zero_outputs(2, 3, 3, 3)
    kwargs = {name: state[name] for name in ("reinfiltration", "irrigation", "returnflow")}
    kwargs.update(sed_deposition=state["sed_depositiontot"], poc_deposition=state["poc_depositiontot"], dt_sechiba=1800.0)

    def consumer(values):
        result = routing_to_slowproc_writeback(**values)
        return result.doc_to_topsoil, result.doc_to_subsoil, result.sed_deposition_d, result.poc_deposition_d

    outputs = jax.jit(consumer)(kwargs) if compiled else consumer(kwargs)
    fields = {name: state[name] for name in ("riverflow", "coastalflow", "returnflow", "reinfiltration", "irrigation", "stream_frac", "flood_res", "fastr")}
    fields.update(sed_deposition=state["sed_depositiontot"], poc_deposition=state["poc_depositiontot"])
    fields.update(zip(("DOC_to_topsoil", "DOC_to_subsoil", "sed_deposition_d", "poc_deposition_d"), outputs, strict=True))
    # flood_frac/streamfl_frac are reset in the SECHIBA source block, not
    # outputs of routing_zero_outputs. Do not invent returned helper fields.
    return {name: np.asarray(jnp.asarray(value)).ravel(order="F") for name, value in fields.items()}


def module_order_binding():
    record = next(item for item in yaml.safe_load(
        (ROOT / "docs/source_audits/production_families/lifecycle.yaml").read_text(encoding="utf-8")
    )["records"] if item["ledger_entry"] == "sechiba.active.module_order")
    owner = record["production_owner"]
    path = ROOT / owner["file"]
    source = path.read_text(encoding="utf-8")
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == owner["symbol"])
    span = "".join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    orchestration = ROOT / "jax_orchidee/driver/orchestration.py"
    tree = ast.parse(orchestration.read_text(encoding="utf-8"))
    targets = {"run_hydrol_first_step_module_from_precall", "run_condveg_first_step_module", "run_thermosoil_first_step_module"}
    maintained = []
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        calls = sorted((node.lineno, node.func.id) for node in ast.walk(function)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in targets)
        if {name for _, name in calls} == targets:
            maintained.append({"caller": function.name, "calls": [{"line": line, "callee": name} for line, name in calls]})
    return {
        "parent": "PROC:sechiba.active.module_order", "saved_comparison": record["comparison_asset"],
        "saved_comparison_sha256": digest(ROOT / record["comparison_asset"]),
        "unchanged_reference_owner_file": digest(path) == owner["file_sha256"],
        "unchanged_reference_owner_span": hashlib.sha256(span.encode()).hexdigest() == owner["span_sha256"],
        "owner": owner, "maintained_caller_order": maintained,
        "maintained_caller_level": "source_mapped",
        "scope": "Original component/lifecycle evidence reused for unchanged helper. Current whole-dispatch numerical execution not inferred from source order or old callsite hashes.",
        "remaining": "One complete source-admitted matched predecessor tuple across all maintained component boundaries, then separately authorized full-step original/current observation; no whole step executed here.",
    }


def run(output=PACKAGE):
    output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(FIXTURE.read_text(encoding="utf-8"))
    checks = []
    for compiled in (False, True):
        for name, actual in routing_fields(compiled=compiled).items():
            expected = np.asarray(reference["expected"][name])
            checks.append({"name": name, "compiled_consumer": compiled,
                           "passed": np.array_equal(actual, expected), "count": actual.size})
    binding = module_order_binding()
    result = {"routing_checks": checks, "module_order": binding, "numerical_acceptance": False}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"routing_checks": len(checks), "passed": all(row["passed"] for row in checks),
                      "unchanged_saved_module_owner": binding["unchanged_reference_owner_file"]}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-fixture", action="store_true")
    options = parser.parse_args()
    if options.build_fixture:
        build_fixture()
    run()
