"""Validate saved daily-slice diagnostic inputs and field replay on CPU."""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import jax
import numpy as np


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load diagnostic helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo-root",
        "micro4-probe",
        "real-daily-probe-file",
        "source-profile",
        "input-bundle",
        "prepared-context",
        "micro4-report",
        "field-replay",
        "field-executable",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root))
    if jax.default_backend() != "cpu":
        raise ValueError("artifact schema verification must run on CPU")
    micro4 = _module(args.micro4_probe, "slice_assets_micro4")
    helper = micro4.load_owner(args.real_daily_probe_file)
    args.source_profile = args.source_profile.resolve()
    args.prepared_context = args.prepared_context.resolve()
    args.input_bundle = args.input_bundle.resolve()
    profile, reference = helper.trusted_profile(args)
    identity = helper.context_identity(reference, helper.initial_checkpoint(reference), 742)
    with args.prepared_context.open("rb") as stream:
        context = pickle.load(stream)
    if context.get("identity") != identity:
        raise ValueError("saved context identity differs from artifact742 method")
    with args.input_bundle.open("rb") as stream:
        bundle = pickle.load(stream)
    if (
        bundle.get("schema_version") != "shared_serialized_dual_inputs_v1"
        or bundle.get("method") != reference["method"]
        or tuple(bundle.get("artifact_ids", ())) != (742, 275)
        or len(bundle.get("payloads", ())) != 2
    ):
        raise ValueError("saved bundle is not the original artifact742/275 pair")
    rows = micro4.validate_original_rows(jax, context["daily4"], bundle["payloads"][0]["daily"])
    daily32 = bundle["payloads"][0]["daily"]
    daily64 = jax.tree_util.tree_map(
        lambda a, b: np.concatenate((np.asarray(a), np.asarray(b)), axis=0),
        daily32,
        bundle["payloads"][1]["daily"],
    )
    leaves32 = jax.tree_util.tree_leaves(daily32)
    leaves64 = jax.tree_util.tree_leaves(daily64)
    if len(leaves32) != 590 or len(leaves64) != 590:
        raise ValueError("daily sample pytree does not have the expected 590 leaves")
    for left, right in zip(leaves32, leaves64, strict=True):
        if left.shape[0] != 32 or right.shape[0] != 64 or left.dtype != right.dtype:
            raise ValueError("daily32/64 concatenation changed leading shape or dtype")
    report = json.loads(args.micro4_report.read_text(encoding="utf-8"))
    if (
        report.get("status") != "passed"
        or report.get("union_admission", {}).get("gradient_coordinates") != 64_884_481
        or report.get("comparisons", {}).get("all_parameter_gradients", {}).get("passed") is not True
        or Path(report.get("baseline_binary", "")).resolve()
        != Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-combined-real-daily-20260925.candidate.executable.pkl")
    ):
        raise ValueError("completed micro4 report does not admit the expected field baseline")
    with args.field_replay.open("rb") as stream:
        saved = pickle.load(stream)
    saved_identity = saved.get("identity", {})
    if (
        saved.get("variant") != micro4.VARIANT
        or {key: value for key, value in saved_identity.items() if key in identity} != identity
        or saved_identity.get("original_rows") != rows
    ):
        raise ValueError("saved original-four field replay identity differs")
    from scripts.hpc.run_shared_learning import load_learning_checkpoint

    checkpoint = load_learning_checkpoint(
        Path(identity["initial_checkpoint"]), expected_method=reference["method"]
    )
    with args.field_executable.open("rb") as stream:
        serialized_executable = pickle.load(stream)
    if (
        not isinstance(serialized_executable, tuple)
        or len(serialized_executable) != 3
        or serialized_executable[1]
        != jax.tree_util.tree_structure(((checkpoint["parameters"], context["daily4"]), {}))
    ):
        raise ValueError("saved field executable input pytree ABI differs from original four-row inputs")
    admission = micro4.validate_result(jax, saved["result"], checkpoint["parameters"])
    print(
        {
            "backend": "cpu",
            "artifact_ids": [742, 275],
            "sample_leaves": len(leaves32),
            "daily_rows": [32, 64],
            "original_four_rows_verified": rows,
            "full_gradient_coordinates": admission["gradient_coordinates"],
            "field_replay_identity": "passed",
            "field_executable_input_treedef": "passed",
        }
    )


if __name__ == "__main__":
    main()
