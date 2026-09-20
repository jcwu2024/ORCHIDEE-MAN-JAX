"""Forward-only attribution of mixed-learning carbon inventory endpoints."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.hpc import probe_feedback_loss_calibration as calibration
from scripts.hpc import probe_feedback_loss_updates as admission
from scripts.hpc import run_fresh_start_one_day as prototype
from scripts.hpc.run_mixed_learning import configure_execution, validate_configuration

SCHEMA_VERSION = "mixed_carbon_endpoint_probe_v1"
CARBON_OWNERS = (
    "litter_above",
    "litter_below",
    "carbon_32l",
    "DOC",
    "interception_storage",
)
OWNER_LAYOUT = {
    "litter_above": {"pft_axis": 2, "element_axis": -1, "area_weighted": True},
    "litter_below": {"pft_axis": 2, "element_axis": -1, "area_weighted": True},
    "carbon_32l": {"pft_axis": 2, "element_axis": None, "area_weighted": True},
    "DOC": {"pft_axis": 1, "element_axis": -1, "area_weighted": True},
    # The source field is already on a land-area basis.
    "interception_storage": {"pft_axis": 1, "element_axis": -1, "area_weighted": False},
}


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


def _method_from_report(
    report: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any],
    model_protocol: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
) -> dict[str, Any]:
    method = {
        "implementation": "mixed_learning_v1",
        "configuration": dict(configuration),
        "model_protocol": dict(model_protocol),
        "scope": report["scope"],
        "integration_updates": report.get("integration_updates"),
        "source_git_head": report["source_git_head"],
        "execution_policy": dict(execution_policy),
        "recurrent_execution": execution_policy["recurrent_execution"],
    }
    if report.get("learning_schedule") is not None:
        method["learning_schedule"] = report["learning_schedule"]
    if report.get("device_execution") is not None:
        method["device_execution"] = report["device_execution"]
    return method


def _checkpoint_identity(
    report: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any],
    model_protocol: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    expected_update: int,
) -> dict[str, Any]:
    if report.get("configuration") != configuration or report.get("model_protocol") != model_protocol:
        raise ValueError("training report configuration or model protocol differs")
    if report.get("execution_policy") != execution_policy:
        raise ValueError("training report execution policy differs")
    if report.get("data_plan") != plan["summary"]:
        raise ValueError("training report data plan differs from validated inputs")
    completed = int(report.get("completed_updates", -1))
    if expected_update < 0 or expected_update > completed:
        raise ValueError("expected checkpoint update is outside reported progress")
    retained = [row for row in report.get("checkpoints", ()) if int(row.get("update", -1)) == expected_update]
    if len(retained) != 1:
        raise ValueError("training report does not identify exactly one retained checkpoint update")
    fresh = {
        "seed": model_protocol["seed"],
        "initialization": model_protocol["initialization"],
        "parameter_count": model_protocol["parameter_count"],
        "inherited_weights": False,
    }
    return {
        "expected_method": _method_from_report(
            report,
            configuration=configuration,
            model_protocol=model_protocol,
            execution_policy=execution_policy,
        ),
        "expected_source_identities": plan["identities"],
        "expected_schedule_identity": plan["identities"],
        "expected_fresh_initialization_identity": fresh,
        "maximum_updates": configuration["updates"],
    }


def endpoint_terms(
    start: Any,
    outgoing_fraction: Any,
    incoming_amount: Any,
    defined: Any,
) -> dict[str, np.ndarray]:
    start = np.asarray(start, dtype=np.float64)
    outgoing = np.asarray(outgoing_fraction, dtype=np.float64)
    incoming = np.asarray(incoming_amount, dtype=np.float64)
    mask = np.asarray(defined, dtype=np.bool_)
    if outgoing.shape != start.shape or incoming.shape != start.shape or mask.shape != start.shape:
        raise ValueError("carbon endpoint arrays differ in shape")
    safe_start = np.where(mask, start, 0.0)
    safe_outgoing = np.where(mask, outgoing, 0.0)
    safe_incoming = np.where(mask, incoming, 0.0)
    outgoing_amount = safe_start * safe_outgoing
    return {
        "defined": mask,
        "outgoing_fraction": safe_outgoing,
        "outgoing_amount": outgoing_amount,
        "incoming_amount": safe_incoming,
        "endpoint": np.where(mask, safe_start - outgoing_amount + safe_incoming, start),
    }


def compact_carbon_sum(
    value: Any,
    defined: Any,
    *,
    fraction: Any,
    active: Any,
    owner: str,
) -> dict[str, Any]:
    if owner not in OWNER_LAYOUT:
        raise KeyError(owner)
    layout = OWNER_LAYOUT[owner]
    array = np.asarray(value, dtype=np.float64)
    mask = np.asarray(defined, dtype=np.bool_)
    if array.shape != mask.shape:
        raise ValueError("carbon owner value/mask shapes differ")
    pft_axis = int(layout["pft_axis"]) % array.ndim
    element_axis = layout["element_axis"]
    if element_axis is not None:
        normalized_element = int(element_axis) % array.ndim
        if normalized_element == pft_axis or array.shape[normalized_element] < 1:
            raise ValueError("carbon owner element axis is invalid")
        array = np.take(array, 0, axis=normalized_element)
        mask = np.take(mask, 0, axis=normalized_element)
        if normalized_element < pft_axis:
            pft_axis -= 1
    weights = np.asarray(fraction, dtype=np.float64)
    active_mask = np.asarray(active, dtype=np.bool_)
    if weights.shape != active_mask.shape or weights.ndim < 1:
        raise ValueError("PFT fraction and active masks differ")
    if array.shape[pft_axis] != weights.shape[-1]:
        raise ValueError("carbon owner and PFT context widths differ")
    leading = weights.ndim - 1
    if array.shape[:leading] != weights.shape[:-1]:
        raise ValueError("carbon owner and PFT leading axes differ")
    broadcast_shape = [1] * array.ndim
    broadcast_shape[:leading] = weights.shape[:-1]
    broadcast_shape[pft_axis] = weights.shape[-1]
    relevant = np.broadcast_to(active_mask.reshape(broadcast_shape), array.shape)
    physical_weight = np.ones_like(weights) if not layout["area_weighted"] else weights
    physical_weight = np.broadcast_to(physical_weight.reshape(broadcast_shape), array.shape)
    valid = relevant & mask & np.isfinite(array) & np.isfinite(physical_weight)
    return {
        "value_gC_m2": float(np.sum(np.where(valid, array * physical_weight, 0.0))),
        "valid_coordinate_count": int(np.count_nonzero(valid)),
        "missing_active_coordinate_count": int(np.count_nonzero(relevant & ~valid)),
        "complete": bool(np.all((~relevant) | valid)),
    }


def _array_record(value: Any, defined: Any) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    mask = np.asarray(defined, dtype=np.bool_)
    return {
        "shape": list(array.shape),
        "value": array.reshape(-1).tolist(),
        "defined": mask.reshape(-1).tolist(),
    }


def _metadata_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [dict(row) for row in rows]
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"rows": normalized, "metadata_sha256": hashlib.sha256(encoded).hexdigest()}


def _panel_rows(plan: Mapping[str, Any], count: int) -> dict[str, list[dict[str, Any]]]:
    panels = {
        "fixed_training_daily": admission._rows(plan["batches"][0].rows[:count]),
        "excluded_train_daily": admission._rows(plan["monitor_rows"][:count]),
    }
    if any(len(rows) != count for rows in panels.values()):
        raise ValueError("requested one-day diagnostic panel is incomplete")
    return panels


def _sample_record(host: Mapping[str, Any], metadata: Mapping[str, Any], leaf_metadata: Mapping[str, Any]) -> dict[str, Any]:
    fraction = np.asarray(host["fraction"], dtype=np.float64)
    active = np.asarray(host["active"], dtype=np.bool_)
    owners = {}
    for owner in CARBON_OWNERS:
        row = host["owners"][owner]
        start = np.asarray(row["start"], dtype=np.float64)
        start_defined = np.asarray(row["start_defined"], dtype=np.bool_)
        predicted = endpoint_terms(start, row["predicted_outgoing"], row["predicted_incoming"], start_defined)
        teacher_defined = start_defined & np.asarray(row["teacher_defined"], dtype=np.bool_)
        teacher = endpoint_terms(start, row["teacher_outgoing"], row["teacher_incoming"], teacher_defined)
        post_predicted = np.asarray(row["post_predicted"], dtype=np.float64)
        post_predicted_defined = np.asarray(row["post_predicted_defined"], dtype=np.bool_)
        post_teacher = np.asarray(row["post_teacher"], dtype=np.float64)
        post_teacher_defined = np.asarray(row["post_teacher_defined"], dtype=np.bool_)

        def summarized(value: Any, mask: Any) -> dict[str, Any]:
            return compact_carbon_sum(value, mask, fraction=fraction, active=active, owner=owner)

        owners[owner] = {
            "axis_names": list(leaf_metadata[owner]["axis_names"]),
            "pft_axis": OWNER_LAYOUT[owner]["pft_axis"],
            "element_axis": OWNER_LAYOUT[owner]["element_axis"],
            "area_weighted": OWNER_LAYOUT[owner]["area_weighted"],
            "start": {**_array_record(start, start_defined), "compact": summarized(start, start_defined)},
            "predicted": {
                "outgoing_fraction": _array_record(predicted["outgoing_fraction"], predicted["defined"]),
                "outgoing_amount": {
                    **_array_record(predicted["outgoing_amount"], predicted["defined"]),
                    "compact": summarized(predicted["outgoing_amount"], predicted["defined"]),
                },
                "incoming_amount": {
                    **_array_record(predicted["incoming_amount"], predicted["defined"]),
                    "compact": summarized(predicted["incoming_amount"], predicted["defined"]),
                },
                "pre_tail_endpoint": {
                    **_array_record(predicted["endpoint"], predicted["defined"]),
                    "compact": summarized(predicted["endpoint"], predicted["defined"]),
                },
                "post_tail_endpoint": {
                    **_array_record(post_predicted, post_predicted_defined),
                    "compact": summarized(post_predicted, post_predicted_defined),
                },
            },
            "teacher": {
                "outgoing_fraction": _array_record(teacher["outgoing_fraction"], teacher["defined"]),
                "outgoing_amount": {
                    **_array_record(teacher["outgoing_amount"], teacher["defined"]),
                    "compact": summarized(teacher["outgoing_amount"], teacher["defined"]),
                },
                "incoming_amount": {
                    **_array_record(teacher["incoming_amount"], teacher["defined"]),
                    "compact": summarized(teacher["incoming_amount"], teacher["defined"]),
                },
                "pre_tail_endpoint": {
                    **_array_record(teacher["endpoint"], teacher["defined"]),
                    "compact": summarized(teacher["endpoint"], teacher["defined"]),
                },
                "post_tail_endpoint": {
                    **_array_record(post_teacher, post_teacher_defined),
                    "compact": summarized(post_teacher, post_teacher_defined),
                },
            },
        }
    stocks = {}
    for name, value in host["stocks"].items():
        predicted = np.asarray(value["predicted"], dtype=np.float64)
        teacher = np.asarray(value["teacher"], dtype=np.float64)
        defined = np.asarray(value["defined"], dtype=np.bool_)
        stocks[name] = {
            "predicted_gC_m2": float(np.sum(np.where(defined, predicted, 0.0))),
            "teacher_gC_m2": float(np.sum(np.where(defined, teacher, 0.0))),
            "defined_count": int(np.count_nonzero(defined)),
            "nonfinite_defined": int(np.count_nonzero(defined & (~np.isfinite(predicted) | ~np.isfinite(teacher)))),
        }
    return {
        "selection": dict(metadata),
        "fraction": fraction.tolist(),
        "active": active.tolist(),
        "owners": owners,
        "post_tail_stocks": stocks,
        "hard": {name: int(np.asarray(value)) for name, value in host["hard"].items()},
        "budgets": np.asarray(host["budgets"], dtype=np.float64).tolist(),
        "carbon_external_input": float(np.asarray(host["carbon_external_input"])),
        "carbon_external_output": float(np.asarray(host["carbon_external_output"])),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from research.daily_coarse_graining.daily_operator.fresh_start_training import (
        extract_one_day_trajectories,
        make_objective,
    )
    from research.daily_coarse_graining.daily_operator.mixed_training_state import load_mixed_training_checkpoint
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator_dynamic_retained_inputs
    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
    from scripts.hpc.check_daily_training_runtime import _atomic_json
    from scripts.hpc.mixed_learning_plan import load_plan
    from scripts.hpc.trained_screen_common import json_safe

    started = time.perf_counter()
    execution_policy = configure_execution(args)
    if execution_policy is None:
        raise ValueError("carbon endpoint probe requires the admitted execution policy")
    configuration = _read_json(args.mixed_protocol, name="mixed protocol")
    model_protocol = _read_json(args.protocol, name="model protocol")
    validate_configuration(configuration, model_protocol)
    if (args.mixed_protocol.parent / configuration["model_protocol"]).resolve() != args.protocol.resolve():
        raise ValueError("CLI and mixed model protocols differ")
    plan = load_plan(args, configuration)
    training_report = _read_json(args.training_report, name="training report")
    identity = _checkpoint_identity(
        training_report,
        configuration=configuration,
        model_protocol=model_protocol,
        execution_policy=execution_policy,
        plan=plan,
        expected_update=args.expected_update,
    )
    panels = _panel_rows(plan, args.samples_per_panel)
    panel_identity = {name: _metadata_identity(rows) for name, rows in panels.items()}
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "stage": "metadata",
        "scope": "forward_only_one_day_carbon_endpoint_attribution",
        "training_report": str(args.training_report.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "expected_update": args.expected_update,
        "panel_identity": panel_identity,
        "panels": {name: [] for name in panels},
        "new_optimizer_updates": 0,
        "gradient_evaluations": 0,
        "scientific_acceptance": False,
    }

    def emit(stage: str) -> None:
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started)
        _atomic_json(args.output, json_safe(report))
        print(f"mixed_carbon_endpoints stage={stage} samples={sum(map(len, report['panels'].values()))}", flush=True)

    emit("bootstrap")
    runtime, head, template, scales, fresh_parameters, _, _ = prototype.bootstrap(args, model_protocol)
    jax = runtime["jax"]

    devices = tuple(jax.local_devices())
    if not devices or devices[0].platform != "gpu":
        raise ValueError("carbon endpoint probe requires a visible GPU")
    device = devices[0]
    del fresh_parameters
    gc.collect()

    saved = load_mixed_training_checkpoint(args.checkpoint, **identity)
    if int(saved["completed_updates"]) != args.expected_update:
        raise ValueError("checkpoint progress differs from expected update")
    parameters = jax.device_put(saved["parameters"], device)
    checkpoint_record = {
        "completed_updates": int(saved["completed_updates"]),
        "daily_sampler_cursor": saved["daily_sampler_cursor"],
        "recurrent_schedule_cursor": int(saved["recurrent_schedule_cursor"]),
    }
    del saved
    gc.collect()
    report["checkpoint_identity"] = checkpoint_record

    graph = calibration._loss_protocol(
        model_protocol,
        canopy_source_pft_index=calibration._canopy_source_pft_index(runtime, head),
    )
    objective = make_objective(runtime, head, template, scales, graph)
    transition = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=head,
        retained_tail_template=template,
    )
    leaves = {
        owner: next(
            leaf for leaf in head.contract.state_leaves
            if leaf.key == f"slowproc_stomate_previous_step_state.{owner}"
        )
        for owner in CARBON_OWNERS
    }
    leaf_metadata = {
        owner: {"axis_names": tuple(leaf.axis_names), "shape": tuple(leaf.shape)}
        for owner, leaf in leaves.items()
    }

    def state_owner(state: Any, state_defined: Any, owner: str) -> tuple[Any, Any]:
        leaf = leaves[owner]
        value = state.other[leaf.start:leaf.stop].reshape(leaf.shape)
        defined = state_defined.other[leaf.start:leaf.stop].reshape(leaf.shape)
        return value, defined

    def forward(one_parameters: Any, sample: Any) -> Mapping[str, Any]:
        result = transition(one_parameters, sample.inputs, sample.day_inputs)
        assembled = runtime["adapter"].input_assembler.assemble(sample.inputs)
        _, auxiliary = objective(one_parameters, sample)
        owner_rows = {}
        for owner in CARBON_OWNERS:
            start, start_defined = state_owner(assembled.state, assembled.state_defined, owner)
            post_predicted, post_predicted_defined = state_owner(
                result.next_state, result.next_state_defined, owner
            )
            post_teacher, post_teacher_defined = state_owner(
                sample.expected_next_state, sample.expected_next_state_defined, owner
            )
            prefix = f"inventory.carbon.{owner}"
            teacher_outgoing = sample.process_targets[f"{prefix}.outgoing_fraction"]
            teacher_incoming = sample.process_targets[f"{prefix}.incoming_amount"]
            owner_rows[owner] = {
                "start": start,
                "start_defined": start_defined,
                "predicted_outgoing": result.prediction.named_outputs[f"{prefix}.outgoing_fraction"],
                "predicted_incoming": result.prediction.named_outputs[f"{prefix}.incoming_amount"],
                "teacher_outgoing": teacher_outgoing.value,
                "teacher_incoming": teacher_incoming.value,
                "teacher_defined": teacher_outgoing.defined & teacher_incoming.defined,
                "post_predicted": post_predicted,
                "post_predicted_defined": post_predicted_defined,
                "post_teacher": post_teacher,
                "post_teacher_defined": post_teacher_defined,
            }
        target_gpp = sample.process_targets["daily.fast_gpp"]
        predicted_science = extract_one_day_trajectories(
            result.next_state,
            result.next_state_defined,
            result.prediction.named_outputs["daily.fast_gpp"],
            target_gpp.defined,
            sample.inputs,
            head.contract,
        )
        teacher_science = extract_one_day_trajectories(
            sample.expected_next_state,
            sample.expected_next_state_defined,
            target_gpp.value,
            target_gpp.defined,
            sample.inputs,
            head.contract,
        )
        stocks = {
            name: {
                "predicted": predicted_science.stocks[name].value,
                "teacher": teacher_science.stocks[name].value,
                "defined": predicted_science.stocks[name].defined & teacher_science.stocks[name].defined,
            }
            for name in predicted_science.stocks
        }
        return {
            "owners": owner_rows,
            "fraction": assembled.pft.pft_fraction,
            "active": assembled.pft.active_pft_mask,
            "stocks": stocks,
            "hard": auxiliary["hard"],
            "budgets": auxiliary["budgets"],
            "carbon_external_input": result.diagnostics["carbon_external_input"],
            "carbon_external_output": result.diagnostics["carbon_external_output"],
        }

    compiled_forward = jax.jit(forward)
    pool = PreparedOneDayPool(plan["daily_source"], cache_size=5)
    if pool.identity != plan["catalogue"]["source_identity"]:
        raise ValueError("daily reader and validated source identity differ")

    emit("forward")
    for panel_name, rows in panels.items():
        for index, metadata in enumerate(rows):
            host_batch = pool.batch([(int(metadata["artifact_id"]), int(metadata["batch_row"]))])
            sample = jax.tree_util.tree_map(lambda value: value[0], host_batch)
            host = jax.device_get(compiled_forward(parameters, jax.device_put(sample, device)))
            report["panels"][panel_name].append(
                _sample_record(host, metadata, leaf_metadata)
            )
            emit(f"forward_{panel_name}_{index + 1}")

    report["reader_statistics"] = pool.statistics()
    report["status"] = "completed"
    emit("completed")
    return report


def parser() -> argparse.ArgumentParser:
    result = prototype.parser()
    for name in ("mixed-protocol", "catalogue-report", "recurrent-schedule", "selection",
                 "execution-policy", "training-report", "checkpoint"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--expected-update", type=int, default=4000)
    result.add_argument("--samples-per-panel", type=int, default=8, choices=range(1, 65))
    result.set_defaults(
        compare_checked_recurrent=False,
        checked_reference_report=None,
        dual_device_profile=False,
        dual_device_training=False,
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
