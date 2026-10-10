"""CPU-only inspection of preserved complete corrected AD and independent JVP.

Keep repeat forward, parameter, state and DOC gates separate. This report is
technical evidence, not scientific adoption or trained-model accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    from scripts.hpc.probe_shared_biomass_state_amplification import adjacent, read_pickle
except ModuleNotFoundError:
    from probe_shared_biomass_state_amplification import adjacent, read_pickle

RTOL, ATOL = 1e-5, 1e-6


def compare_tree(jax, actual, reference, *, details=False):
    if jax.tree_util.tree_structure(actual) != jax.tree_util.tree_structure(reference):
        raise ValueError("complete comparison tree differs")
    coordinates, failed, nonfinite, leaves = 0, 0, 0, []
    for (path, actual), reference in zip(jax.tree_util.tree_flatten_with_path(actual)[0], jax.tree_util.tree_leaves(reference), strict=True):
        a, r = np.asarray(actual), np.asarray(reference)
        if a.shape != r.shape or a.dtype != r.dtype:
            raise ValueError("complete comparison shape/dtype differs: "+jax.tree_util.keystr(path))
        good = np.isclose(a, r, rtol=RTOL, atol=ATOL)
        count = int(np.count_nonzero(~good))
        coordinates += a.size
        failed += count
        nf = int(np.count_nonzero(~np.isfinite(a) | ~np.isfinite(r)))
        nonfinite += nf
        delta = np.abs(a.astype(np.float64)-r.astype(np.float64))
        record = {"path": jax.tree_util.keystr(path), "shape": list(a.shape), "coordinates": int(a.size),
            "failed": count, "nonfinite": nf,
            "maximum_absolute_difference": float(np.max(delta)) if a.size else 0.0}
        if details and count:
            record["failures"] = [{"index": [int(i) for i in index], "actual": float(a[tuple(index)]),
                "reference": float(r[tuple(index)]), "allowance": float(ATOL+RTOL*abs(r[tuple(index)]))}
                for index in np.argwhere(~good)]
        leaves.append(record)
    return {"passed": failed == 0 and nonfinite == 0, "coordinates": int(coordinates),
            "failed": failed, "nonfinite": nonfinite, "leaves": leaves}


def coordinate_binding(contract, coordinate):
    matches = [leaf for leaf in contract.state_leaves if leaf.start is not None and leaf.stop is not None and leaf.start <= coordinate < leaf.stop]
    return [{"field": leaf.key, "field_coordinate": int(coordinate-leaf.start),
             "field_index": list(np.unravel_index(coordinate-leaf.start, leaf.shape)),
             "source_ref": str(leaf.source_ref)} for leaf in matches]


def inspect_state(actual, reference, contract):
    a, r = np.asarray(actual), np.asarray(reference)
    if a.shape != (4, 3854) or r.shape != a.shape or a.dtype != r.dtype:
        raise ValueError("complete four-row state gradient ABI differs")
    bad = ~np.isclose(a, r, rtol=RTOL, atol=ATOL)
    failures = []
    for row, coordinate in np.argwhere(bad):
        row, coordinate = int(row), int(coordinate)
        failures.append({"micro4_row": row, "artifact_row": 16+row, "coordinate": coordinate,
            "coordinate2243": coordinate == 2243, "bindings": coordinate_binding(contract, coordinate),
            "actual": float(a[row, coordinate]), "reference": float(r[row, coordinate]),
            "absolute_difference": float(abs(a[row, coordinate]-r[row, coordinate])),
            "allowance": float(ATOL+RTOL*abs(r[row, coordinate]))})
    doc_leaves = [leaf for leaf in contract.state_leaves if "doc" in leaf.key.lower() and leaf.start is not None and leaf.stop is not None]
    doc = []
    for leaf in doc_leaves:
        aa, rr = a[:, leaf.start:leaf.stop], r[:, leaf.start:leaf.stop]
        diff = np.abs(aa-rr)
        doc.append({"field": leaf.key, "start": leaf.start, "stop": leaf.stop,
            "coordinates": int(aa.size), "failed": int(np.count_nonzero(~np.isclose(aa, rr, rtol=RTOL, atol=ATOL))),
            "finite": bool(np.all(np.isfinite(aa)) and np.all(np.isfinite(rr))),
            "maximum_absolute_difference": float(np.max(diff)) if aa.size else 0.0})
    return {"passed": not failures and bool(np.all(np.isfinite(a)) and np.all(np.isfinite(r))),
        "coordinates": int(a.size), "failed": len(failures), "failures": failures,
        "coordinate2243": {"actual": a[:, 2243].tolist(), "reference": r[:, 2243].tolist(),
            "passed": bool(np.all(np.isclose(a[:, 2243], r[:, 2243], rtol=RTOL, atol=ATOL)))},
        "DOC": {"available_fields": len(doc), "passed": bool(doc) and all(x["failed"] == 0 and x["finite"] for x in doc), "fields": doc}}


def same_inputs(actual, reference):
    for key in ("source_git_head", "manifest_sha256", "input_binding", "snapshots", "parameters", "parameter_leaves", "precision", "loss", "turnover_sha256", "sources"):
        if actual.get(key) != reference.get(key):
            raise ValueError("saved result identity differs: "+key)


def inspect_diagnostics(jax, actual, reference):
    """Gate exact masks and all defined values; disclose undefined observations."""
    records, undefined, mask_records = [], [], []
    failed, coordinates = 0, 0
    for values_key, masks_key in (("predicted", "predicted_defined"), ("expected", "expected_defined")):
        mask_result = compare_tree(jax, actual[masks_key], reference[masks_key], details=True)
        mask_records.append({"kind": masks_key, **mask_result})
        if jax.tree_util.tree_structure(actual[values_key]) != jax.tree_util.tree_structure(reference[values_key]):
            raise ValueError("diagnostic state group tree differs")
        rows = jax.tree_util.tree_flatten_with_path(actual[values_key])[0]
        values = jax.tree_util.tree_leaves(reference[values_key])
        actual_masks, reference_masks = jax.tree_util.tree_leaves(actual[masks_key]), jax.tree_util.tree_leaves(reference[masks_key])
        for (path, a), r, am, rm in zip(rows, values, actual_masks, reference_masks, strict=True):
            a, r, am, rm = np.asarray(a), np.asarray(r), np.asarray(am), np.asarray(rm)
            if a.shape != r.shape or a.dtype != r.dtype or am.shape != a.shape or rm.shape != a.shape or am.dtype != np.bool_ or rm.dtype != np.bool_:
                raise ValueError("diagnostic state value/mask ABI differs")
            defined = am | rm
            bad = defined & ~np.isclose(a, r, rtol=RTOL, atol=ATOL)
            path_name = values_key+jax.tree_util.keystr(path)
            failures = [{"index": [int(i) for i in index], "artifact_row": 16+int(index[0]),
                "diagnostic_field": path_name, "actual": float(a[tuple(index)]), "reference": float(r[tuple(index)]),
                "allowance": float(ATOL+RTOL*abs(r[tuple(index)]))} for index in np.argwhere(bad)]
            failed += len(failures)
            coordinates += int(np.count_nonzero(defined))
            records.append({"path": path_name, "shape": list(a.shape), "defined_coordinates": int(np.count_nonzero(defined)),
                "failed": len(failures), "failures": failures})
            inactive = ~defined
            source_changed = inactive & ~np.isclose(a, r, rtol=RTOL, atol=ATOL, equal_nan=True)
            undefined.append({"path": path_name, "undefined_coordinates": int(np.count_nonzero(inactive)),
                "actual_nonfinite": int(np.count_nonzero(inactive & ~np.isfinite(a))),
                "reference_nonfinite": int(np.count_nonzero(inactive & ~np.isfinite(r))),
                "equal_nan_observed_parity": not bool(np.any(source_changed)),
                "changed_source_values": [{"index": [int(i) for i in index], "artifact_row": 16+int(index[0]),
                    "actual": float(a[tuple(index)]), "reference": float(r[tuple(index)])} for index in np.argwhere(source_changed)]})
    for mask in mask_records:
        for leaf in mask["leaves"]:
            for failure in leaf.get("failures", []):
                failure["artifact_row"] = 16+failure["index"][0]
                failure["diagnostic_field"] = mask["kind"]+leaf["path"]
    return {"passed": failed == 0 and all(x["passed"] for x in mask_records), "defined_coordinates": coordinates,
        "defined_failed": failed, "defined_value_leaves": records, "exact_mask_comparisons": mask_records,
        "undefined_source_observations": undefined, "undefined_values_are_acceptance_gate": False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("corrected-output", "cpu-jvp", "prepared-context", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--original-control", type=Path)
    return p


def run(args):
    os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    import jax
    if jax.default_backend() != "cpu": raise ValueError("inspection requires CPU only")
    adjacent("training_artifact_io")
    # Preserve the staged serialized class/module identities used by the
    # immutable-checkout context and complete original AD result artifacts.
    adjacent("probe_shared_original_full_state_schedule").load_overlays()
    from scripts.hpc.training_artifact_io import _atomic_json
    if args.output.exists(): raise FileExistsError(args.output)
    context = read_pickle(args.prepared_context)
    contract = getattr(context["template"], "contract", None)
    contract_source = "template.contract"
    if contract is None:
        contract, contract_source = context["head"].contract, "head.contract"
    receipt = json.loads(args.corrected_output.read_text())
    report = {"schema_version": "shared_leaf_cuda_preserved_result_inspection_v1", "status": "inspected",
        "rtol": RTOL, "atol": ATOL, "GPU_executed": False, "scientific_acceptance": False,
        "source_corrected_report": str(args.corrected_output), "state_contract_source": contract_source,
        "raw_probe_same_input_strict_parity": receipt.get("same_input_strict_parity"),
        "repeat_comparisons": {}, "independent_JVP": [], "original_regression": {},
        "reported_resource_evidence": {"compiled_memory": receipt.get("compiled_memory"),
            "observed_peak_memory_passed": receipt.get("observed_peak_memory_passed"),
            "per_call": [{"schedule": row["schedule"], "repeat": row["repeat"],
                "observed_peak_bytes": row.get("observed_peak_bytes"), "JAX_device_memory_stats": row.get("JAX_device_memory_stats")} for row in receipt.get("results", [])],
            "NVML": "external parent evidence; not inferred from JAX allocator statistics"}}
    baseline_paths = {}
    for schedule in ("old", "new"):
        files = [args.corrected_output.with_suffix(f".{schedule}.repeat{repeat}.result.pkl") for repeat in (1, 2)]
        first, second = (read_pickle(path) for path in files)
        same_inputs(second["identity"], first["identity"])
        if first["schedule"] != schedule or second["schedule"] != schedule or first["repeat"] != 1 or second["repeat"] != 2:
            raise ValueError("saved corrected repeat/schedule identity differs")
        reference, actual = first["result"], second["result"]
        ordinary = lambda result: (result[0][0], {k: v for k, v in result[0][1].items() if k != "state_diagnostics"})
        forward = compare_tree(jax, ordinary(actual), ordinary(reference))
        parameter = compare_tree(jax, actual[1][0], reference[1][0])
        state = inspect_state(actual[1][1], reference[1][1], contract)
        diagnostics = inspect_diagnostics(jax, actual[0][1]["state_diagnostics"], reference[0][1]["state_diagnostics"])
        doc_aux = [leaf for leaf in forward["leaves"] if "doc" in leaf["path"].lower()]
        report["repeat_comparisons"][schedule] = {"artifacts": [str(path) for path in files],
            "forward_and_gates": forward, "all_parameter_gradients": parameter,
            "all_initial_state_gradients": state, "all_state_diagnostics": diagnostics,
            "DOC_auxiliary": {"available_named_leaves": len(doc_aux), "leaves": doc_aux,
                "passed": all(x["failed"] == 0 and x["nonfinite"] == 0 for x in doc_aux) if doc_aux else None}}
        baseline_paths[schedule] = files[0]
        del first, second, reference, actual
    jvp_report = json.loads(args.cpu_jvp.read_text())
    expected_domains = {"old:16", "old:17", "new:16", "new:17"}
    present = [row["domain"] for row in jvp_report.get("results", [])]
    if len(present) != len(set(present)): raise ValueError("duplicate independent JVP domain")
    report["missing_JVP_domains"] = sorted(expected_domains-set(present))
    for row in jvp_report.get("results", []):
        if row["domain"] not in expected_domains: raise ValueError("unexpected JVP domain")
        schedule, artifact_row = row["domain"].split(":")
        artifact_row = int(artifact_row)
        cpu = read_pickle(Path(row["artifact"]))
        corrected = read_pickle(baseline_paths[schedule])
        same_inputs(cpu["identity"], corrected["identity"])
        if cpu["domain"] != row["domain"]: raise ValueError("JVP artifact association differs")
        reference = float(np.asarray(cpu["scalar_derivative"])/4)
        actual = float(corrected["result"][1][1][artifact_row-16, 2243])
        report["independent_JVP"].append({"domain": row["domain"], "artifact_row": artifact_row,
            "coordinate": 2243, "bindings": coordinate_binding(contract, 2243), "CPU_artifact": row["artifact"],
            "corrected_artifact": str(baseline_paths[schedule]), "scalar_JVP": float(cpu["scalar_derivative"]),
            "micro4_mean_row_JVP": reference, "corrected_state_gradient": actual,
            "passed": bool(np.isfinite(actual) and np.isfinite(reference) and np.isclose(actual, reference, rtol=RTOL, atol=ATOL)),
            "absolute_difference": abs(actual-reference), "allowance": ATOL+RTOL*abs(reference),
            "CPU_finite_and_physical": bool(row["finite"] and row["physical_hard_counts_zero"])})
        del cpu, corrected
    if args.original_control:
        control = json.loads(args.original_control.read_text())
        for row in control.get("results", []):
            schedule = row["schedule"]
            original, corrected = read_pickle(Path(row["artifact"])), read_pickle(baseline_paths[schedule])
            same_inputs(original["identity"], corrected["identity"])
            original_result, corrected_result = original["result"], corrected["result"]
            ordinary = lambda result: (result[0][0], {k: v for k, v in result[0][1].items() if k != "state_diagnostics"})
            report["original_regression"][schedule] = {
                "original_artifact": row["artifact"], "corrected_artifact": str(baseline_paths[schedule]),
                "forward_and_gates": compare_tree(jax, ordinary(corrected_result), ordinary(original_result)),
                "all_parameter_gradients": compare_tree(jax, corrected_result[1][0], original_result[1][0]),
                "state_gradient_changes": inspect_state(corrected_result[1][1], original_result[1][1], contract)}
            del original, corrected, original_result, corrected_result
    comparisons = report["repeat_comparisons"].values()
    report["repeat_forward_parameter_passed"] = all(x["forward_and_gates"]["passed"] and x["all_parameter_gradients"]["passed"] for x in comparisons)
    report["repeat_all_state_passed"] = all(x["all_initial_state_gradients"]["passed"] and x["all_state_diagnostics"]["passed"] for x in comparisons)
    report["four_complete_loss_JVP_passed"] = not report["missing_JVP_domains"] and all(x["passed"] and x["CPU_finite_and_physical"] for x in report["independent_JVP"])
    report["original_forward_parameter_passed"] = (set(report["original_regression"]) == {"old", "new"} and all(
        x["forward_and_gates"]["passed"] and x["all_parameter_gradients"]["passed"] for x in report["original_regression"].values())) if args.original_control else None
    report["corrected_finite_physical_reported"] = receipt.get("finite_and_physical_passed")
    _atomic_json(args.output, report)
    return report


if __name__ == "__main__":
    run(parser().parse_args())
