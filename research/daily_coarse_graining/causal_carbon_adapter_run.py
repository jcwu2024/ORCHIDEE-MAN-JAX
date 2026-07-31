"""Bounded train-only feasibility runner for Experiment C."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.canonical_daily_model import parameter_count
from research.daily_coarse_graining.canonical_training_run import (
    _adam_init,
    _adam_update,
    verify_training_acceptance,
)
from research.daily_coarse_graining.causal_carbon_adapter_daily_model import (
    assemble_causal_carbon_adapter_parameters,
    causal_carbon_adapter_trainable_parameters,
)
from research.daily_coarse_graining.causal_carbon_adapter_protocol import (
    ARM_IDS,
    CausalCarbonAdapterProtocol,
    load_causal_carbon_adapter_protocol,
)
from research.daily_coarse_graining.causal_carbon_objective import (
    CausalCarbonComponents,
    causal_carbon_candidate_loss,
    causal_carbon_interface_loss,
    causal_carbon_objective_layout_from_contract,
)
from research.daily_coarse_graining.daily_model_architecture import (
    CAUSAL_CARBON_ADAPTER_V1,
    build_daily_model_definition,
    initialize_causal_adapter_from_parent,
    verify_checkpoint_architecture,
)
from research.daily_coarse_graining.rollout_stability_run import (
    ARM_CHECKPOINT_SCHEMA_VERSION,
    _bind_dynamic_transition,
    _load_rollout_resources,
    prepare_rollout_update,
)

FEASIBILITY_SCHEMA_VERSION = "causal_carbon_adapter_feasibility_v1"
ADAPTER_CHECKPOINT_SCHEMA_VERSION = "causal_carbon_adapter_checkpoint_v1"
HARD_COMPONENTS = (
    "unexpected_defined_status_mismatches",
    "discrete_state_mismatches",
    "nonfinite_defined_values",
    "negative_source_nonnegative_carbon_stocks",
)
COUNT_COMPONENTS = (*HARD_COMPONENTS, "declared_dynamic_status_mismatches")


class AdapterUpdateResult(NamedTuple):
    parameters: Any
    optimizer: Any
    loss: Any
    components: Any
    gradient_norm: Any
    nonfinite_gradient_values: Any
    update_applied: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_git_head() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_pickle(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {path} is not a mapping")
    return value


def _tree_nonfinite_count(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0, dtype=jnp.int32)
    return sum(jnp.count_nonzero(~jnp.isfinite(leaf)) for leaf in leaves)


def _tree_l2_norm(tree):
    return jnp.sqrt(sum(jnp.sum(jnp.asarray(leaf) ** 2) for leaf in jax.tree_util.tree_leaves(tree)))


def _select_tree(predicate, accepted, rejected):
    return jax.tree_util.tree_map(
        lambda left, right: jnp.where(predicate, left, right),
        accepted,
        rejected,
    )


def _empty_components(loss) -> CausalCarbonComponents:
    zero = jnp.zeros_like(loss)
    count = jnp.asarray(0, dtype=jnp.int32)
    return CausalCarbonComponents(
        L_interface=loss,
        L_next=zero,
        L_rollout=zero,
        L_flux_bias=zero,
        L_stock_tendency_bias=zero,
        L_guard=zero,
        unexpected_defined_status_mismatches=count,
        declared_dynamic_status_mismatches=count,
        discrete_state_mismatches=count,
        nonfinite_defined_values=count,
        negative_source_nonnegative_carbon_stocks=count,
    )


def _hard_count(components: CausalCarbonComponents):
    return sum(getattr(components, name) for name in HARD_COMPONENTS)


def make_adapter_control_update(*, layout, model_apply):
    def update(parameters, optimizer, anchor_batch, learning_rate):
        def objective(value):
            return causal_carbon_interface_loss(
                value,
                anchor_batch,
                layout=layout,
                model_apply=model_apply,
            )

        loss, gradients = jax.value_and_grad(objective)(parameters)
        proposed_parameters, proposed_optimizer = _adam_update(
            parameters,
            gradients,
            optimizer,
            learning_rate=learning_rate,
        )
        nonfinite = _tree_nonfinite_count(gradients)
        applied = jnp.isfinite(loss) & (nonfinite == 0)
        return AdapterUpdateResult(
            parameters=_select_tree(applied, proposed_parameters, parameters),
            optimizer=_select_tree(applied, proposed_optimizer, optimizer),
            loss=loss,
            components=_empty_components(loss),
            gradient_norm=_tree_l2_norm(gradients),
            nonfinite_gradient_values=nonfinite,
            update_applied=applied,
        )

    return jax.jit(update)


def make_adapter_candidate_update(
    *,
    coefficients: Mapping[str, float],
    layout,
    model_apply,
    statistics,
    representation,
    retained_tail_transition,
    state_delta_scale,
    rematerialize: bool,
):
    def update(
        parameters,
        optimizer,
        anchor_batch,
        initial_states,
        initial_discrete_states,
        sequences,
        teacher_next_discrete_states,
        learning_rate,
    ):
        def objective(value):
            result = causal_carbon_candidate_loss(
                value,
                anchor_batch,
                initial_states,
                initial_discrete_states,
                sequences,
                coefficients=coefficients,
                layout=layout,
                statistics=statistics,
                representation=representation,
                retained_tail_transition=retained_tail_transition,
                state_delta_scale=state_delta_scale,
                teacher_next_discrete_states=teacher_next_discrete_states,
                rematerialize=rematerialize,
                model_apply=model_apply,
            )
            return result.loss, result.components

        (loss, components), gradients = jax.value_and_grad(
            objective,
            has_aux=True,
        )(parameters)
        proposed_parameters, proposed_optimizer = _adam_update(
            parameters,
            gradients,
            optimizer,
            learning_rate=learning_rate,
        )
        nonfinite = _tree_nonfinite_count(gradients)
        applied = jnp.isfinite(loss) & (nonfinite == 0) & (jnp.asarray(_hard_count(components)) == 0)
        return AdapterUpdateResult(
            parameters=_select_tree(applied, proposed_parameters, parameters),
            optimizer=_select_tree(applied, proposed_optimizer, optimizer),
            loss=loss,
            components=components,
            gradient_norm=_tree_l2_norm(gradients),
            nonfinite_gradient_values=nonfinite,
            update_applied=applied,
        )

    return jax.jit(update)


def _verify_artifacts(
    protocol: CausalCarbonAdapterProtocol,
    *,
    dataset_path: Path,
    statistics_path: Path,
    acceptance_path: Path,
    parent_checkpoint_path: Path,
    parent_report_path: Path,
) -> None:
    artifacts = protocol.raw["artifacts"]
    expected = (
        ("dataset manifest", dataset_path, artifacts["dataset_manifest_sha256"]),
        (
            "training statistics",
            statistics_path,
            artifacts["training_statistics_sha256"],
        ),
        ("acceptance report", acceptance_path, artifacts["acceptance_report_sha256"]),
        (
            "parent checkpoint",
            parent_checkpoint_path,
            protocol.raw["parent"]["checkpoint_sha256"],
        ),
        (
            "parent training report",
            parent_report_path,
            protocol.raw["parent"]["training_report_sha256"],
        ),
    )
    for name, path, expected_sha256 in expected:
        if _sha256_file(path) != expected_sha256:
            raise ValueError(f"{name} hash drift")
    accepted = verify_training_acceptance(
        acceptance_path,
        dataset_path,
        statistics_path,
        verify_dataset_hashes=False,
    )
    if accepted["identity"]["markov_contract_sha256"] is None:
        raise ValueError("accepted dataset is missing its contract identity")

    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    if parent_report.get("status") != "completed":
        raise ValueError("causal adapter parent training report is incomplete")
    if parent_report.get("arm_id") != protocol.raw["parent"]["arm_id"]:
        raise ValueError("causal adapter parent report arm drift")
    if parent_report.get("sealed_test_used") is not False:
        raise ValueError("causal adapter parent report used sealed test data")


def _host_component_record(components: CausalCarbonComponents) -> dict[str, Any]:
    host = jax.device_get(components)
    return {
        name: (int(getattr(host, name)) if name in COUNT_COMPONENTS else float(getattr(host, name)))
        for name in host._fields
    }


def _prediction_invariance(
    *,
    base_parameters,
    trainable_parameters,
    anchor_batch,
    base_definition,
    adapter_definition,
    protected_indices,
) -> dict[str, Any]:
    base = base_definition.apply(base_parameters, anchor_batch.model_input)
    adapter_parameters = assemble_causal_carbon_adapter_parameters(
        base_parameters,
        trainable_parameters,
    )
    adapted = adapter_definition.apply(
        adapter_parameters,
        anchor_batch.model_input,
    )
    base_target = np.asarray(jax.device_get(base.normalized_fast_day_target))
    adapted_target = np.asarray(jax.device_get(adapted.normalized_fast_day_target))
    protected_equal = np.array_equal(
        adapted_target[:, protected_indices],
        base_target[:, protected_indices],
    )
    undefined_equal = np.array_equal(
        np.asarray(jax.device_get(adapted.dynamic_undefined_flip_logits)),
        np.asarray(jax.device_get(base.dynamic_undefined_flip_logits)),
    )
    return {
        "protected_columns_bit_exact": bool(protected_equal),
        "dynamic_undefined_head_bit_exact": bool(undefined_equal),
        "maximum_protected_absolute_difference": float(
            np.max(np.abs(adapted_target[:, protected_indices] - base_target[:, protected_indices]))
        ),
    }


def _feasibility_horizons(protocol: CausalCarbonAdapterProtocol) -> tuple[int, ...]:
    count = int(protocol.raw["train_only_feasibility_gate"]["updates_per_arm"])
    required = tuple(int(value) for value in protocol.raw["train_only_feasibility_gate"]["required_horizon_coverage"])
    return tuple(required[index % len(required)] for index in range(count))


def run_train_only_feasibility(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    statistics_path: str | Path,
    acceptance_path: str | Path,
    parent_checkpoint_path: str | Path,
    parent_report_path: str | Path,
    output_root: str | Path,
    plan_path: str | Path | None = None,
) -> Path:
    """Run eight real-shard matched updates and fail before paid screening."""

    paths = {
        "dataset": Path(dataset_path).resolve(),
        "statistics": Path(statistics_path).resolve(),
        "acceptance": Path(acceptance_path).resolve(),
        "parent_checkpoint": Path(parent_checkpoint_path).resolve(),
        "parent_report": Path(parent_report_path).resolve(),
    }
    output_root = Path(output_root).resolve()
    report_path = output_root / "feasibility_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "passed":
            return report_path
        raise ValueError("existing causal adapter feasibility root did not pass")
    output_root.mkdir(parents=True, exist_ok=True)

    protocol = load_causal_carbon_adapter_protocol(protocol_path)
    _verify_artifacts(
        protocol,
        dataset_path=paths["dataset"],
        statistics_path=paths["statistics"],
        acceptance_path=paths["acceptance"],
        parent_checkpoint_path=paths["parent_checkpoint"],
        parent_report_path=paths["parent_report"],
    )
    resources = _load_rollout_resources(
        dataset_path=paths["dataset"],
        statistics_path=paths["statistics"],
        plan_path=None if plan_path is None else Path(plan_path).resolve(),
    )
    layout = causal_carbon_objective_layout_from_contract(resources.raw_manifest["markov_contract"])
    if layout.sha256 != protocol.raw["architecture"]["objective_layout_sha256"]:
        raise ValueError("causal carbon objective layout hash drift")

    adapter_definition = build_daily_model_definition(
        CAUSAL_CARBON_ADAPTER_V1,
        resources.model_definition.config,
        resources.raw_manifest["markov_contract"],
    )
    adapter_spec = adapter_definition.causal_carbon_adapter_spec
    if adapter_spec is None:
        raise ValueError("causal adapter definition has no static spec")
    if adapter_spec.interface_layout.sha256 != protocol.raw["architecture"]["adapter_layout_sha256"]:
        raise ValueError("causal carbon adapter layout hash drift")

    parent_checkpoint = _load_pickle(paths["parent_checkpoint"])
    if parent_checkpoint.get("schema_version") != ARM_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported causal adapter parent checkpoint schema")
    verify_checkpoint_architecture(
        parent_checkpoint["identity"],
        resources.model_definition,
    )
    base_parameters = jax.tree_util.tree_map(
        jnp.asarray,
        parent_checkpoint["parameters"],
    )
    full_initial = initialize_causal_adapter_from_parent(
        adapter_definition,
        parent_identity=parent_checkpoint["identity"],
        parent_parameters=base_parameters,
        seed=int(protocol.raw["optimization"]["seed"]),
    )
    initial_trainable = causal_carbon_adapter_trainable_parameters(full_initial)
    control_parameters = initial_trainable
    candidate_parameters = initial_trainable
    control_optimizer = _adam_init(control_parameters)
    candidate_optimizer = _adam_init(candidate_parameters)

    def model_apply(trainable, batch):
        return adapter_definition.apply(
            assemble_causal_carbon_adapter_parameters(
                base_parameters,
                trainable,
            ),
            batch,
        )

    control_step = make_adapter_control_update(
        layout=layout,
        model_apply=model_apply,
    )
    provisional_coefficients = {
        name: 1.0
        for name in (
            "L_next",
            "L_rollout",
            "L_flux_bias",
            "L_stock_tendency_bias",
        )
    }
    references = sorted(
        resources.index.select(spatial_split="train", temporal_split="train"),
        key=lambda item: (item.landpoint_id, item.year, str(item.path)),
    )
    if not references:
        raise ValueError("causal adapter feasibility has no train/train shard")
    reference = references[0]
    horizons = _feasibility_horizons(protocol)
    horizon_specs = {int(item["days"]): item for item in protocol.raw["mixed_horizon_sampling"]}
    anchor_batch_size = int(protocol.raw["optimization"]["anchor_batch_size"])
    learning_rate = float(protocol.raw["optimization"]["learning_rate"])
    seed = int(protocol.raw["optimization"]["seed"])
    runtime_cache = {}
    candidate_steps = {}
    history = []
    initial_invariance = None
    all_invariance = []

    for update, horizon in enumerate(horizons):
        spec = horizon_specs[horizon]
        prepared = prepare_rollout_update(
            reference=reference,
            update=update,
            horizon=horizon,
            anchor_batch_size=anchor_batch_size,
            rollout_batch_size=int(spec["batch_size"]),
            seed=seed,
            resources=resources,
            runtime_cache=runtime_cache,
        )
        if initial_invariance is None:
            initial_invariance = _prediction_invariance(
                base_parameters=base_parameters,
                trainable_parameters=initial_trainable,
                anchor_batch=prepared.anchor_batch,
                base_definition=resources.model_definition,
                adapter_definition=adapter_definition,
                protected_indices=np.flatnonzero(~np.any(layout.fast_field_masks, axis=0)),
            )
            if not all(
                initial_invariance[key]
                for key in (
                    "protected_columns_bit_exact",
                    "dynamic_undefined_head_bit_exact",
                )
            ):
                raise ValueError("zero adapter is not bit-exact with its parent")
        transition = _bind_dynamic_transition(resources, prepared.runtime)
        key = (horizon, prepared.trace_signature)
        if key not in candidate_steps:
            candidate_steps[key] = make_adapter_candidate_update(
                coefficients=provisional_coefficients,
                layout=layout,
                model_apply=model_apply,
                statistics=resources.statistics,
                representation=resources.representation,
                retained_tail_transition=transition,
                state_delta_scale=resources.state_delta_scale,
                rematerialize=bool(spec["rematerialize"]),
            )

        control = control_step(
            control_parameters,
            control_optimizer,
            prepared.anchor_batch,
            learning_rate,
        )
        candidate = candidate_steps[key](
            candidate_parameters,
            candidate_optimizer,
            prepared.anchor_batch,
            prepared.initial_states,
            prepared.initial_discrete_states,
            prepared.sequence,
            prepared.teacher_next_discrete_states,
            learning_rate,
        )
        control_parameters = control.parameters
        control_optimizer = control.optimizer
        candidate_parameters = candidate.parameters
        candidate_optimizer = candidate.optimizer
        control_host = jax.device_get(control)
        candidate_host = jax.device_get(candidate)
        invariance = _prediction_invariance(
            base_parameters=base_parameters,
            trainable_parameters=candidate_parameters,
            anchor_batch=prepared.anchor_batch,
            base_definition=resources.model_definition,
            adapter_definition=adapter_definition,
            protected_indices=np.flatnonzero(~np.any(layout.fast_field_masks, axis=0)),
        )
        all_invariance.append(invariance)
        record = {
            "update": update + 1,
            "horizon": horizon,
            "landpoint_id": reference.landpoint_id,
            "year": int(reference.year),
            "control": {
                "loss": float(control_host.loss),
                "gradient_norm": float(control_host.gradient_norm),
                "nonfinite_gradient_values": int(control_host.nonfinite_gradient_values),
                "update_applied": bool(control_host.update_applied),
            },
            "candidate": {
                "loss": float(candidate_host.loss),
                "gradient_norm": float(candidate_host.gradient_norm),
                "nonfinite_gradient_values": int(candidate_host.nonfinite_gradient_values),
                "update_applied": bool(candidate_host.update_applied),
                "components": _host_component_record(candidate.components),
            },
            "invariance": invariance,
        }
        history.append(record)
        print(
            "causal_adapter_feasibility "
            f"update={update + 1}/{len(horizons)} horizon={horizon} "
            f"control={record['control']['loss']:.8g} "
            f"candidate={record['candidate']['loss']:.8g}",
            flush=True,
        )

    all_updates_applied = all(record[arm]["update_applied"] for record in history for arm in ("control", "candidate"))
    protected_exact = all(
        item["protected_columns_bit_exact"] and item["dynamic_undefined_head_bit_exact"] for item in all_invariance
    )
    hard_counts = {name: sum(record["candidate"]["components"][name] for record in history) for name in HARD_COMPONENTS}
    passed = (
        all_updates_applied
        and protected_exact
        and all(value == 0 for value in hard_counts.values())
        and set(horizons) == set(protocol.raw["train_only_feasibility_gate"]["required_horizon_coverage"])
    )

    checkpoint_common = {
        "schema_version": ADAPTER_CHECKPOINT_SCHEMA_VERSION,
        "protocol_sha256": protocol.sha256,
        "training_git_head": _current_git_head(),
        "parent_checkpoint_sha256": protocol.raw["parent"]["checkpoint_sha256"],
        "model_architecture": adapter_definition.identity(),
        "objective_layout_sha256": layout.sha256,
        "updates": len(horizons),
        "coefficient_status": "provisional_unit_coefficients_for_feasibility_only",
    }
    checkpoint_paths = {}
    for arm_id, parameters, optimizer in (
        (ARM_IDS[0], control_parameters, control_optimizer),
        (ARM_IDS[1], candidate_parameters, candidate_optimizer),
    ):
        path = output_root / f"{arm_id}.pkl"
        _atomic_pickle(
            path,
            checkpoint_common
            | {
                "arm_id": arm_id,
                "adapter_parameters": jax.device_get(parameters),
                "optimizer": jax.device_get(optimizer),
            },
        )
        checkpoint_paths[arm_id] = {
            "path": str(path),
            "sha256": _sha256_file(path),
        }

    report = {
        "schema_version": FEASIBILITY_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "experiment_id": protocol.raw["experiment_id"],
        "training_git_head": _current_git_head(),
        "protocol": {
            "path": str(Path(protocol_path).resolve()),
            "sha256": protocol.sha256,
        },
        "inputs": {name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in paths.items()},
        "train_only": True,
        "sealed_test_used": False,
        "reference": {
            "landpoint_id": reference.landpoint_id,
            "year": int(reference.year),
            "spatial_split": reference.spatial_split,
            "temporal_split": reference.temporal_split,
            "sha256": reference.sha256,
        },
        "horizons": list(horizons),
        "all_updates_applied": all_updates_applied,
        "initial_parent_invariance": initial_invariance,
        "protected_columns_bit_exact_after_all_updates": protected_exact,
        "hard_constraint_totals": hard_counts,
        "base_parameter_count": parameter_count(base_parameters),
        "trainable_adapter_parameter_count": parameter_count(initial_trainable),
        "coefficients": {
            "status": "provisional_feasibility_only",
            "values": provisional_coefficients,
            "formal_calibration_required_before_paid_training": True,
        },
        "compiled_candidate_executables": len(candidate_steps),
        "history": history,
        "checkpoints": checkpoint_paths,
    }
    _atomic_json(report_path, report)
    if not passed:
        raise ValueError(f"causal adapter train-only feasibility failed: {report}")
    return report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--parent-report", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run_train_only_feasibility(
        protocol_path=args.protocol,
        dataset_path=args.dataset,
        statistics_path=args.statistics,
        acceptance_path=args.acceptance,
        parent_checkpoint_path=args.parent_checkpoint,
        parent_report_path=args.parent_report,
        output_root=args.output_root,
        plan_path=args.plan,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
