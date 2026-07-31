"""Bounded feasibility, calibration, and matched-training runner for Experiment C."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import time
from collections import defaultdict
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
    causal_carbon_rollout_components,
)
from research.daily_coarse_graining.daily_model_architecture import (
    CAUSAL_CARBON_ADAPTER_V1,
    build_daily_model_definition,
    initialize_causal_adapter_from_parent,
    verify_checkpoint_architecture,
)
from research.daily_coarse_graining.rollout_stability_run import (
    ARM_CHECKPOINT_SCHEMA_VERSION,
    _balanced_reference_indices,
    _bind_dynamic_transition,
    _exact_horizon_schedule,
    _load_rollout_resources,
    prepare_anchor_update,
    prepare_rollout_update,
)

FEASIBILITY_SCHEMA_VERSION = "causal_carbon_adapter_feasibility_v1"
CALIBRATION_SCHEMA_VERSION = "causal_carbon_adapter_calibration_v1"
ADAPTER_CHECKPOINT_SCHEMA_VERSION = "causal_carbon_adapter_checkpoint_v1"
SCHEDULE_SCHEMA_VERSION = "causal_carbon_adapter_schedule_v1"
EXECUTION_SCHEMA_VERSION = "causal_carbon_adapter_execution_v1"
TRAINING_CHECKPOINT_SCHEMA_VERSION = "causal_carbon_adapter_training_checkpoint_v1"
ACCEPTED_FEASIBILITY_REPORT_SHA256 = "ca8cacdccb42f20b94b5cd64796118cb170f7fbc7813c036cbc6b1edc8aa7fa8"
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


class PreparedAdapterExperiment(NamedTuple):
    protocol: CausalCarbonAdapterProtocol
    paths: Mapping[str, Path]
    resources: Any
    layout: Any
    adapter_definition: Any
    base_parameters: Any
    initial_trainable: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def make_adapter_gradient_calibration_step(
    *,
    layout,
    model_apply,
    statistics,
    representation,
    retained_tail_transition,
    state_delta_scale,
    rematerialize: bool,
):
    """Return component-resolved gradients from one shared finite forward."""

    def calibrate(
        parameters,
        anchor_batch,
        initial_states,
        initial_discrete_states,
        sequences,
        teacher_next_discrete_states,
    ):
        def component_vector(value):
            anchor = causal_carbon_interface_loss(
                value,
                anchor_batch,
                layout=layout,
                model_apply=model_apply,
            )
            components = causal_carbon_rollout_components(
                value,
                initial_states,
                initial_discrete_states,
                sequences,
                statistics=statistics,
                representation=representation,
                retained_tail_transition=retained_tail_transition,
                state_delta_scale=state_delta_scale,
                layout=layout,
                teacher_next_discrete_states=teacher_next_discrete_states,
                rematerialize=rematerialize,
                model_apply=model_apply,
            )._replace(L_interface=anchor)
            values = jnp.stack(
                (
                    anchor,
                    components.L_next,
                    components.L_rollout,
                    components.L_flux_bias,
                    components.L_stock_tendency_bias,
                )
            )
            return values, components

        values, pullback, components = jax.vjp(
            component_vector,
            parameters,
            has_aux=True,
        )
        gradients = jax.vmap(lambda cotangent: pullback(cotangent)[0])(jnp.eye(values.shape[0], dtype=values.dtype))
        squared_norms = sum(
            jnp.sum(
                jnp.asarray(leaf) ** 2,
                axis=tuple(range(1, jnp.asarray(leaf).ndim)),
            )
            for leaf in jax.tree_util.tree_leaves(gradients)
        )
        nonfinite_counts = sum(
            jnp.count_nonzero(
                ~jnp.isfinite(jnp.asarray(leaf)),
                axis=tuple(range(1, jnp.asarray(leaf).ndim)),
            )
            for leaf in jax.tree_util.tree_leaves(gradients)
        )
        return values, jnp.sqrt(squared_norms), nonfinite_counts, components

    return jax.jit(calibrate)


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


def _prepare_adapter_experiment(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    statistics_path: str | Path,
    acceptance_path: str | Path,
    parent_checkpoint_path: str | Path,
    parent_report_path: str | Path,
    plan_path: str | Path | None,
) -> PreparedAdapterExperiment:
    paths = {
        "dataset": Path(dataset_path).resolve(),
        "statistics": Path(statistics_path).resolve(),
        "acceptance": Path(acceptance_path).resolve(),
        "parent_checkpoint": Path(parent_checkpoint_path).resolve(),
        "parent_report": Path(parent_report_path).resolve(),
    }
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
    return PreparedAdapterExperiment(
        protocol=protocol,
        paths=paths,
        resources=resources,
        layout=layout,
        adapter_definition=adapter_definition,
        base_parameters=base_parameters,
        initial_trainable=causal_carbon_adapter_trainable_parameters(full_initial),
    )


def _verified_feasibility_report(
    path: str | Path,
    protocol: CausalCarbonAdapterProtocol,
) -> tuple[Path, Mapping[str, Any]]:
    path = Path(path).resolve()
    if _sha256_file(path) != ACCEPTED_FEASIBILITY_REPORT_SHA256:
        raise ValueError("causal adapter feasibility report hash drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != FEASIBILITY_SCHEMA_VERSION:
        raise ValueError("unsupported causal adapter feasibility schema")
    if report.get("status") != "passed":
        raise ValueError("causal adapter feasibility gate did not pass")
    if report.get("protocol", {}).get("sha256") != protocol.sha256:
        raise ValueError("causal adapter feasibility protocol drift")
    if report.get("train_only") is not True:
        raise ValueError("causal adapter feasibility was not train-only")
    if report.get("sealed_test_used") is not False:
        raise ValueError("causal adapter feasibility used sealed test data")
    if report.get("all_updates_applied") is not True:
        raise ValueError("causal adapter feasibility did not apply every update")
    if report.get("protected_columns_bit_exact_after_all_updates") is not True:
        raise ValueError("causal adapter feasibility changed protected columns")
    if any(int(value) != 0 for value in report["hard_constraint_totals"].values()):
        raise ValueError("causal adapter feasibility violated a hard constraint")
    return path, report


def _calibration_reference_schedule(
    references: Sequence[Any],
    *,
    records_per_horizon: int,
    seed: int,
) -> tuple[Any, ...]:
    by_landpoint: dict[str, list[Any]] = {}
    for reference in references:
        by_landpoint.setdefault(reference.landpoint_id, []).append(reference)
    if not by_landpoint:
        raise ValueError("causal adapter calibration has no train/train data")
    rng = np.random.default_rng(seed)
    landpoints = sorted(by_landpoint)
    selected_count = min(4, len(landpoints))
    selected = [
        landpoints[int(index)]
        for index in rng.choice(
            len(landpoints),
            size=selected_count,
            replace=False,
        )
    ]
    year_orders = {
        landpoint: sorted(
            by_landpoint[landpoint],
            key=lambda item: (item.year, str(item.path)),
        )
        for landpoint in selected
    }
    schedule = []
    for ordinal in range(records_per_horizon):
        landpoint = selected[ordinal % selected_count]
        candidates = year_orders[landpoint]
        schedule.append(candidates[(ordinal // selected_count) % len(candidates)])
    return tuple(schedule)


def _resolve_gradient_coefficients(
    records: Sequence[Mapping[str, Any]],
    protocol: CausalCarbonAdapterProtocol,
) -> Mapping[str, Any]:
    calibration = protocol.raw["objective"]["coefficient_calibration"]
    components = (
        "L_next",
        "L_rollout",
        "L_flux_bias",
        "L_stock_tendency_bias",
    )
    low, high = (float(value) for value in calibration["coefficient_clip"])
    resolved = {}
    audit = {}
    for component in components:
        ratios = []
        for record in records:
            anchor = float(record["gradient_norms"]["L_interface"])
            value = float(record["gradient_norms"][component])
            if not np.isfinite(anchor) or anchor <= 0.0:
                raise ValueError("calibration has an invalid interface gradient")
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"calibration has an invalid {component} gradient")
            if value > 0.0:
                ratios.append(value / anchor)
        if not ratios:
            raise ValueError(f"calibration has no positive {component} gradient")
        median_ratio = float(np.median(np.asarray(ratios, dtype=np.float64)))
        target = float(calibration["target_gradient_norm_ratios"][component])
        coefficient = float(np.clip(target / median_ratio, low, high))
        resolved[component] = coefficient
        audit[component] = {
            "positive_records": len(ratios),
            "median_unweighted_to_interface_gradient_norm_ratio": median_ratio,
            "target_weighted_to_interface_gradient_norm_ratio": target,
            "resolved_coefficient": coefficient,
        }
    return {"resolved_coefficients": resolved, "component_audit": audit}


def run_train_only_calibration(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    statistics_path: str | Path,
    acceptance_path: str | Path,
    parent_checkpoint_path: str | Path,
    parent_report_path: str | Path,
    feasibility_report_path: str | Path,
    output_root: str | Path,
    plan_path: str | Path | None = None,
) -> Path:
    """Calibrate candidate treatment gradients without validation access."""

    output_root = Path(output_root).resolve()
    calibration_path = output_root / "coefficient_calibration.json"
    output_root.mkdir(parents=True, exist_ok=True)

    experiment = _prepare_adapter_experiment(
        protocol_path=protocol_path,
        dataset_path=dataset_path,
        statistics_path=statistics_path,
        acceptance_path=acceptance_path,
        parent_checkpoint_path=parent_checkpoint_path,
        parent_report_path=parent_report_path,
        plan_path=plan_path,
    )
    feasibility_path, _ = _verified_feasibility_report(
        feasibility_report_path,
        experiment.protocol,
    )
    if calibration_path.exists():
        _, calibration = _load_verified_calibration(
            calibration_path,
            experiment.protocol,
        )
        if calibration.get("training_git_head") != _current_git_head():
            raise ValueError("existing causal adapter calibration Git head drift")
        return calibration_path
    resources = experiment.resources
    protocol = experiment.protocol

    def model_apply(trainable, batch):
        return experiment.adapter_definition.apply(
            assemble_causal_carbon_adapter_parameters(
                experiment.base_parameters,
                trainable,
            ),
            batch,
        )

    calibration_spec = protocol.raw["objective"]["coefficient_calibration"]
    records_per_horizon = int(calibration_spec["batches_per_horizon"])
    seed = int(protocol.raw["optimization"]["seed"])
    references = sorted(
        resources.index.select(spatial_split="train", temporal_split="train"),
        key=lambda item: (item.landpoint_id, item.year, str(item.path)),
    )
    reference_schedule = _calibration_reference_schedule(
        references,
        records_per_horizon=records_per_horizon,
        seed=seed,
    )
    horizon_specs = {int(item["days"]): item for item in protocol.raw["mixed_horizon_sampling"]}
    runtime_cache = {}
    compiled_steps = {}
    records = []
    component_names = (
        "L_interface",
        "L_next",
        "L_rollout",
        "L_flux_bias",
        "L_stock_tendency_bias",
    )
    update = 0
    for horizon, spec in horizon_specs.items():
        for reference in reference_schedule:
            prepared = prepare_rollout_update(
                reference=reference,
                update=update,
                horizon=horizon,
                anchor_batch_size=int(protocol.raw["optimization"]["anchor_batch_size"]),
                rollout_batch_size=int(spec["batch_size"]),
                seed=seed,
                resources=resources,
                runtime_cache=runtime_cache,
            )
            key = (horizon, prepared.trace_signature)
            if key not in compiled_steps:
                compiled_steps[key] = make_adapter_gradient_calibration_step(
                    layout=experiment.layout,
                    model_apply=model_apply,
                    statistics=resources.statistics,
                    representation=resources.representation,
                    retained_tail_transition=_bind_dynamic_transition(
                        resources,
                        prepared.runtime,
                    ),
                    state_delta_scale=resources.state_delta_scale,
                    rematerialize=bool(spec["rematerialize"]),
                )
            values, norms, nonfinite, components = compiled_steps[key](
                experiment.initial_trainable,
                prepared.anchor_batch,
                prepared.initial_states,
                prepared.initial_discrete_states,
                prepared.sequence,
                prepared.teacher_next_discrete_states,
            )
            values, norms, nonfinite, components = jax.device_get((values, norms, nonfinite, components))
            hard_counts = {name: int(getattr(components, name)) for name in HARD_COMPONENTS}
            if np.any(np.asarray(nonfinite) != 0):
                raise ValueError("causal adapter calibration has nonfinite gradients")
            if any(value != 0 for value in hard_counts.values()):
                raise ValueError("causal adapter calibration violated a hard constraint")
            record = {
                "ordinal": update,
                "horizon": horizon,
                "landpoint_id": reference.landpoint_id,
                "year": int(reference.year),
                "spatial_split": reference.spatial_split,
                "temporal_split": reference.temporal_split,
                "loss_components": {name: float(values[index]) for index, name in enumerate(component_names)},
                "gradient_norms": {name: float(norms[index]) for index, name in enumerate(component_names)},
                "gradient_nonfinite_values": {
                    name: int(nonfinite[index]) for index, name in enumerate(component_names)
                },
                "hard_constraint_counts": hard_counts,
            }
            records.append(record)
            update += 1
            print(
                "causal_adapter_calibration "
                f"record={update}/{records_per_horizon * len(horizon_specs)} "
                f"horizon={horizon} landpoint={reference.landpoint_id}",
                flush=True,
            )

    resolved = _resolve_gradient_coefficients(records, protocol)
    payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": "passed",
        "experiment_id": protocol.raw["experiment_id"],
        "training_git_head": _current_git_head(),
        "protocol_sha256": protocol.sha256,
        "feasibility_report": {
            "path": str(feasibility_path),
            "sha256": _sha256_file(feasibility_path),
        },
        "parent_checkpoint_sha256": protocol.raw["parent"]["checkpoint_sha256"],
        "train_only": True,
        "sealed_test_used": False,
        "method": calibration_spec["method"],
        "records_per_horizon": records_per_horizon,
        "horizons": list(horizon_specs),
        "record_count": len(records),
        "landpoint_ids": sorted({record["landpoint_id"] for record in records}),
        "records": records,
        "records_sha256": _canonical_sha256(records),
        "compiled_executables": len(compiled_steps),
        **resolved,
    }
    payload["canonical_sha256"] = _canonical_sha256(payload)
    return _atomic_json(calibration_path, payload)


def _load_verified_calibration(
    path: str | Path,
    protocol: CausalCarbonAdapterProtocol,
) -> tuple[Path, Mapping[str, Any]]:
    path = Path(path).resolve()
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if calibration.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported causal adapter calibration schema")
    if calibration.get("status") != "passed":
        raise ValueError("causal adapter calibration did not pass")
    if calibration.get("protocol_sha256") != protocol.sha256:
        raise ValueError("causal adapter calibration protocol drift")
    if calibration.get("feasibility_report", {}).get("sha256") != ACCEPTED_FEASIBILITY_REPORT_SHA256:
        raise ValueError("causal adapter calibration feasibility drift")
    if calibration.get("parent_checkpoint_sha256") != protocol.raw["parent"]["checkpoint_sha256"]:
        raise ValueError("causal adapter calibration parent drift")
    if calibration.get("train_only") is not True:
        raise ValueError("causal adapter calibration was not train-only")
    if calibration.get("sealed_test_used") is not False:
        raise ValueError("causal adapter calibration used sealed test data")
    expected_records = int(protocol.raw["objective"]["coefficient_calibration"]["batches_per_horizon"]) * len(
        protocol.raw["mixed_horizon_sampling"]
    )
    if int(calibration.get("record_count", -1)) != expected_records:
        raise ValueError("causal adapter calibration record-count drift")
    expected_components = {
        "L_next",
        "L_rollout",
        "L_flux_bias",
        "L_stock_tendency_bias",
    }
    if set(calibration.get("resolved_coefficients", {})) != expected_components:
        raise ValueError("causal adapter calibrated coefficient inventory drift")
    if any(
        not np.isfinite(float(value)) or float(value) <= 0.0 for value in calibration["resolved_coefficients"].values()
    ):
        raise ValueError("causal adapter calibration has invalid coefficients")
    canonical_sha256 = calibration.get("canonical_sha256")
    without_hash = dict(calibration)
    without_hash.pop("canonical_sha256", None)
    if canonical_sha256 != _canonical_sha256(without_hash):
        raise ValueError("causal adapter calibration canonical hash drift")
    return path, calibration


def build_training_schedule(
    experiment: PreparedAdapterExperiment,
) -> Mapping[str, Any]:
    """Freeze every matched update's shard and rollout horizon."""

    protocol = experiment.protocol
    references = sorted(
        experiment.resources.index.select(
            spatial_split="train",
            temporal_split="train",
        ),
        key=lambda item: (item.landpoint_id, item.year, str(item.path)),
    )
    optimization = protocol.raw["optimization"]
    updates = int(optimization["screening_updates"])
    seed = int(optimization["seed"])
    reference_indices = _balanced_reference_indices(
        references,
        updates=updates,
        seed=seed,
    )
    horizons = _exact_horizon_schedule(
        protocol.raw["mixed_horizon_sampling"],
        updates=updates,
        seed=seed,
    )
    inventory = [
        {
            "landpoint_id": reference.landpoint_id,
            "year": int(reference.year),
            "spatial_split": reference.spatial_split,
            "temporal_split": reference.temporal_split,
            "path": str(reference.path),
            "sha256": reference.sha256,
        }
        for reference in references
    ]
    entries = [
        {
            "update": update,
            "reference_index": int(reference_index),
            "horizon": int(horizons[update]),
        }
        for update, reference_index in enumerate(reference_indices)
    ]
    landpoint_counts: dict[str, int] = defaultdict(int)
    for reference_index in reference_indices:
        landpoint_counts[references[reference_index].landpoint_id] += 1
    payload = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "protocol_sha256": protocol.sha256,
        "dataset_id": experiment.resources.index.dataset_id,
        "dataset_contract_sha256": experiment.resources.index.contract_sha256,
        "seed": seed,
        "updates": updates,
        "selection_policy": "round_robin_landpoint_distinct_year_then_seeded_start_v1",
        "reference_inventory": inventory,
        "entries": entries,
        "horizon_counts": {
            str(spec["days"]): horizons.count(int(spec["days"])) for spec in protocol.raw["mixed_horizon_sampling"]
        },
        "landpoint_update_count_min": min(landpoint_counts.values()),
        "landpoint_update_count_max": max(landpoint_counts.values()),
        "sealed_test_used": False,
    }
    return payload | {"canonical_sha256": _canonical_sha256(payload)}


def create_training_execution(
    *,
    protocol_path: str | Path,
    dataset_path: str | Path,
    statistics_path: str | Path,
    acceptance_path: str | Path,
    parent_checkpoint_path: str | Path,
    parent_report_path: str | Path,
    feasibility_report_path: str | Path,
    calibration_path: str | Path,
    output_root: str | Path,
    plan_path: str | Path | None = None,
) -> Path:
    """Freeze all formal matched-training inputs before either arm runs."""

    output_root = Path(output_root).resolve()
    execution_path = output_root / "training_execution.json"
    experiment = _prepare_adapter_experiment(
        protocol_path=protocol_path,
        dataset_path=dataset_path,
        statistics_path=statistics_path,
        acceptance_path=acceptance_path,
        parent_checkpoint_path=parent_checkpoint_path,
        parent_report_path=parent_report_path,
        plan_path=plan_path,
    )
    feasibility_path, _ = _verified_feasibility_report(
        feasibility_report_path,
        experiment.protocol,
    )
    calibration_path, calibration = _load_verified_calibration(
        calibration_path,
        experiment.protocol,
    )
    current_head = _current_git_head()
    if calibration.get("training_git_head") != current_head:
        raise ValueError("calibration and matched training Git heads differ")

    schedule = build_training_schedule(experiment)
    schedule_path = output_root / "training_schedule.json"
    if schedule_path.exists():
        existing = json.loads(schedule_path.read_text(encoding="utf-8"))
        if existing != schedule:
            raise ValueError("existing causal adapter schedule identity drift")
    else:
        _atomic_json(schedule_path, schedule)

    payload = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "ready",
        "experiment_id": experiment.protocol.raw["experiment_id"],
        "training_git_head": current_head,
        "protocol": {
            "path": str(Path(protocol_path).resolve()),
            "sha256": experiment.protocol.sha256,
        },
        "dataset": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for name, path in experiment.paths.items()
        },
        "feasibility_report": {
            "path": str(feasibility_path),
            "sha256": _sha256_file(feasibility_path),
        },
        "coefficient_calibration": {
            "path": str(calibration_path),
            "sha256": _sha256_file(calibration_path),
            "canonical_sha256": calibration["canonical_sha256"],
            "resolved_coefficients": calibration["resolved_coefficients"],
        },
        "sampling_schedule": {
            "path": str(schedule_path),
            "sha256": _sha256_file(schedule_path),
            "canonical_sha256": schedule["canonical_sha256"],
        },
        "model_architecture": experiment.adapter_definition.identity(),
        "parent_checkpoint_sha256": experiment.protocol.raw["parent"]["checkpoint_sha256"],
        "adapter_initialization": "same_exact_zero_for_both_arms",
        "arms": list(ARM_IDS),
        "same_anchor_batches": True,
        "same_optimizer_updates": True,
        "train_only": True,
        "sealed_test_used": False,
    }
    payload["canonical_sha256"] = _canonical_sha256(payload)
    if execution_path.exists():
        existing = json.loads(execution_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing causal adapter execution identity drift")
        return execution_path
    return _atomic_json(execution_path, payload)


def _load_verified_execution(
    path: str | Path,
    experiment: PreparedAdapterExperiment,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    path = Path(path).resolve()
    execution = json.loads(path.read_text(encoding="utf-8"))
    if execution.get("schema_version") != EXECUTION_SCHEMA_VERSION:
        raise ValueError("unsupported causal adapter execution schema")
    if execution.get("status") != "ready":
        raise ValueError("causal adapter execution is not ready")
    if execution.get("training_git_head") != _current_git_head():
        raise ValueError("causal adapter execution training head drift")
    if execution.get("protocol", {}).get("sha256") != experiment.protocol.sha256:
        raise ValueError("causal adapter execution protocol drift")
    if execution.get("model_architecture") != experiment.adapter_definition.identity():
        raise ValueError("causal adapter execution architecture drift")
    if execution.get("arms") != list(ARM_IDS):
        raise ValueError("causal adapter execution arm inventory drift")
    if execution.get("same_anchor_batches") is not True:
        raise ValueError("causal adapter execution lost matched anchors")
    if execution.get("same_optimizer_updates") is not True:
        raise ValueError("causal adapter execution lost matched update counts")
    if execution.get("train_only") is not True:
        raise ValueError("causal adapter execution is not train-only")
    if execution.get("sealed_test_used") is not False:
        raise ValueError("causal adapter execution used sealed test data")
    canonical_sha256 = execution.get("canonical_sha256")
    without_hash = dict(execution)
    without_hash.pop("canonical_sha256", None)
    if canonical_sha256 != _canonical_sha256(without_hash):
        raise ValueError("causal adapter execution canonical hash drift")
    schedule_record = execution["sampling_schedule"]
    schedule_path = Path(schedule_record["path"]).resolve()
    if _sha256_file(schedule_path) != schedule_record["sha256"]:
        raise ValueError("causal adapter sampling schedule file drift")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if schedule.get("canonical_sha256") != schedule_record["canonical_sha256"]:
        raise ValueError("causal adapter sampling schedule identity drift")
    expected_schedule = build_training_schedule(experiment)
    if schedule != expected_schedule:
        raise ValueError("causal adapter sampling schedule content drift")
    calibration_record = execution["coefficient_calibration"]
    calibration_path = Path(calibration_record["path"]).resolve()
    if _sha256_file(calibration_path) != calibration_record["sha256"]:
        raise ValueError("causal adapter calibration file drift")
    _, calibration = _load_verified_calibration(
        calibration_path,
        experiment.protocol,
    )
    if calibration.get("canonical_sha256") != calibration_record["canonical_sha256"]:
        raise ValueError("causal adapter calibration identity drift")
    if calibration.get("resolved_coefficients") != calibration_record["resolved_coefficients"]:
        raise ValueError("causal adapter calibrated coefficient drift")
    return path, execution, schedule


def _training_identity(
    execution: Mapping[str, Any],
    experiment: PreparedAdapterExperiment,
    arm_id: str,
) -> Mapping[str, Any]:
    return {
        "execution_canonical_sha256": execution["canonical_sha256"],
        "training_git_head": execution["training_git_head"],
        "protocol_sha256": experiment.protocol.sha256,
        "arm_id": arm_id,
        "dataset_id": experiment.resources.index.dataset_id,
        "teacher_git_head": experiment.resources.index.teacher_git_head,
        "markov_contract_sha256": experiment.resources.index.contract_sha256,
        "model_architecture": experiment.adapter_definition.identity(),
        "parent_checkpoint_sha256": execution["parent_checkpoint_sha256"],
        "coefficient_calibration_sha256": execution["coefficient_calibration"]["sha256"],
        "schedule_canonical_sha256": execution["sampling_schedule"]["canonical_sha256"],
    }


def _completed_horizon_counts(
    schedule: Mapping[str, Any],
    next_update: int,
) -> Mapping[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for entry in schedule["entries"][:next_update]:
        counts[str(int(entry["horizon"]))] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _build_training_checkpoint(
    *,
    arm_id: str,
    identity: Mapping[str, Any],
    parameters,
    optimizer,
    next_update: int,
    schedule: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "arm_id": arm_id,
        "identity": dict(identity),
        "parameters": jax.device_get(parameters),
        "optimizer": jax.device_get(optimizer),
        "next_update": int(next_update),
        "schedule_canonical_sha256": schedule["canonical_sha256"],
        "completed_horizon_counts": _completed_horizon_counts(
            schedule,
            next_update,
        ),
        "history": list(history),
    }


def _verify_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    arm_id: str,
    identity: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> None:
    if checkpoint.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported causal adapter training checkpoint")
    if checkpoint.get("arm_id") != arm_id:
        raise ValueError("causal adapter checkpoint arm drift")
    if checkpoint.get("identity") != dict(identity):
        raise ValueError("causal adapter checkpoint identity drift")
    if checkpoint.get("schedule_canonical_sha256") != schedule["canonical_sha256"]:
        raise ValueError("causal adapter checkpoint schedule drift")
    next_update = int(checkpoint.get("next_update", -1))
    if next_update < 0 or next_update > int(schedule["updates"]):
        raise ValueError("causal adapter checkpoint update is out of range")
    if checkpoint.get("completed_horizon_counts") != _completed_horizon_counts(
        schedule,
        next_update,
    ):
        raise ValueError("causal adapter checkpoint horizon-count drift")
    if "parameters" not in checkpoint or "optimizer" not in checkpoint:
        raise ValueError("causal adapter checkpoint state is incomplete")


def _schedule_references(
    experiment: PreparedAdapterExperiment,
    schedule: Mapping[str, Any],
) -> tuple[Any, ...]:
    references = sorted(
        experiment.resources.index.select(
            spatial_split="train",
            temporal_split="train",
        ),
        key=lambda item: (item.landpoint_id, item.year, str(item.path)),
    )
    observed = [
        {
            "landpoint_id": reference.landpoint_id,
            "year": int(reference.year),
            "spatial_split": reference.spatial_split,
            "temporal_split": reference.temporal_split,
            "path": str(reference.path),
            "sha256": reference.sha256,
        }
        for reference in references
    ]
    if observed != schedule["reference_inventory"]:
        raise ValueError("causal adapter schedule reference inventory drift")
    return tuple(references)


def run_training_arm(
    *,
    arm_id: str,
    execution_path: str | Path,
    protocol_path: str | Path,
    dataset_path: str | Path,
    statistics_path: str | Path,
    acceptance_path: str | Path,
    parent_checkpoint_path: str | Path,
    parent_report_path: str | Path,
    output_root: str | Path,
    plan_path: str | Path | None = None,
    stop_after_updates: int | None = None,
) -> Path:
    """Run or exactly resume one formal matched Experiment C arm."""

    if arm_id not in ARM_IDS:
        raise ValueError(f"unknown causal adapter arm {arm_id!r}")
    experiment = _prepare_adapter_experiment(
        protocol_path=protocol_path,
        dataset_path=dataset_path,
        statistics_path=statistics_path,
        acceptance_path=acceptance_path,
        parent_checkpoint_path=parent_checkpoint_path,
        parent_report_path=parent_report_path,
        plan_path=plan_path,
    )
    execution_path, execution, schedule = _load_verified_execution(
        execution_path,
        experiment,
    )
    identity = _training_identity(execution, experiment, arm_id)
    output_root = Path(output_root).resolve()
    arm_root = output_root / arm_id
    arm_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arm_root / "checkpoint.pkl"
    if checkpoint_path.exists():
        checkpoint = _load_pickle(checkpoint_path)
        _verify_training_checkpoint(
            checkpoint,
            arm_id=arm_id,
            identity=identity,
            schedule=schedule,
        )
        parameters = jax.tree_util.tree_map(jnp.asarray, checkpoint["parameters"])
        optimizer = jax.tree_util.tree_map(jnp.asarray, checkpoint["optimizer"])
        next_update = int(checkpoint["next_update"])
        history = list(checkpoint["history"])
    else:
        parameters = experiment.initial_trainable
        optimizer = _adam_init(parameters)
        next_update = 0
        history = []

    budget = int(schedule["updates"])
    completed_report_path = arm_root / "training_report.json"
    if next_update == budget and completed_report_path.exists():
        completed_report = json.loads(completed_report_path.read_text(encoding="utf-8"))
        if completed_report.get("schema_version") != ("causal_carbon_adapter_training_report_v1"):
            raise ValueError("completed causal adapter report schema drift")
        if completed_report.get("status") != "completed":
            raise ValueError("completed causal adapter report status drift")
        if completed_report.get("arm_id") != arm_id:
            raise ValueError("completed causal adapter report arm drift")
        if completed_report.get("identity") != identity:
            raise ValueError("completed causal adapter report identity drift")
        if int(completed_report.get("updates", -1)) != budget:
            raise ValueError("completed causal adapter report update drift")
        if completed_report.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
            raise ValueError("completed causal adapter report checkpoint drift")
        if completed_report.get("train_only") is not True:
            raise ValueError("completed causal adapter report is not train-only")
        if completed_report.get("sealed_test_used") is not False:
            raise ValueError("completed causal adapter report used sealed test")
        return completed_report_path
    if stop_after_updates is not None and int(stop_after_updates) <= 0:
        raise ValueError("causal adapter stop-after-updates must be positive")
    target_update = budget if stop_after_updates is None else min(budget, next_update + int(stop_after_updates))
    if target_update < next_update:
        raise ValueError("causal adapter arm has no requested updates to run")
    references = _schedule_references(experiment, schedule)
    horizon_specs = {int(item["days"]): item for item in experiment.protocol.raw["mixed_horizon_sampling"]}
    optimization = experiment.protocol.raw["optimization"]
    coefficients = execution["coefficient_calibration"]["resolved_coefficients"]
    seed = int(optimization["seed"])
    learning_rate = float(optimization["learning_rate"])
    checkpoint_every = int(optimization["checkpoint_every_updates"])
    progress_every = int(optimization["progress_every_updates"])
    runtime_cache = {}
    candidate_steps = {}
    interval_losses = []
    interval_gradients = []
    interval_components: dict[str, list[float]] = defaultdict(list)
    interval_started = time.perf_counter()
    last_anchor = None
    if next_update == budget:
        last_entry = schedule["entries"][-1]
        last_reference = references[int(last_entry["reference_index"])]
        last_horizon = int(last_entry["horizon"])
        last_spec = horizon_specs[last_horizon]
        last_anchor = prepare_anchor_update(
            reference=last_reference,
            update=budget - 1,
            horizon=last_horizon,
            anchor_batch_size=int(optimization["anchor_batch_size"]),
            rollout_batch_size=int(last_spec["batch_size"]),
            seed=seed,
            resources=experiment.resources,
        )

    def model_apply(trainable, batch):
        return experiment.adapter_definition.apply(
            assemble_causal_carbon_adapter_parameters(
                experiment.base_parameters,
                trainable,
            ),
            batch,
        )

    control_step = make_adapter_control_update(
        layout=experiment.layout,
        model_apply=model_apply,
    )
    for update in range(next_update, target_update):
        entry = schedule["entries"][update]
        if int(entry["update"]) != update:
            raise ValueError("causal adapter schedule update ordinal drift")
        reference = references[int(entry["reference_index"])]
        horizon = int(entry["horizon"])
        spec = horizon_specs[horizon]
        if arm_id == ARM_IDS[0]:
            anchor = prepare_anchor_update(
                reference=reference,
                update=update,
                horizon=horizon,
                anchor_batch_size=int(optimization["anchor_batch_size"]),
                rollout_batch_size=int(spec["batch_size"]),
                seed=seed,
                resources=experiment.resources,
            )
            result = control_step(
                parameters,
                optimizer,
                anchor,
                learning_rate,
            )
        else:
            prepared = prepare_rollout_update(
                reference=reference,
                update=update,
                horizon=horizon,
                anchor_batch_size=int(optimization["anchor_batch_size"]),
                rollout_batch_size=int(spec["batch_size"]),
                seed=seed,
                resources=experiment.resources,
                runtime_cache=runtime_cache,
            )
            anchor = prepared.anchor_batch
            key = (horizon, prepared.trace_signature)
            if key not in candidate_steps:
                candidate_steps[key] = make_adapter_candidate_update(
                    coefficients=coefficients,
                    layout=experiment.layout,
                    model_apply=model_apply,
                    statistics=experiment.resources.statistics,
                    representation=experiment.resources.representation,
                    retained_tail_transition=_bind_dynamic_transition(
                        experiment.resources,
                        prepared.runtime,
                    ),
                    state_delta_scale=experiment.resources.state_delta_scale,
                    rematerialize=bool(spec["rematerialize"]),
                )
            result = candidate_steps[key](
                parameters,
                optimizer,
                anchor,
                prepared.initial_states,
                prepared.initial_discrete_states,
                prepared.sequence,
                prepared.teacher_next_discrete_states,
                learning_rate,
            )
        last_anchor = anchor
        host_loss, host_components, host_gradient_norm, host_nonfinite, host_applied = jax.device_get(
            (
                result.loss,
                result.components,
                result.gradient_norm,
                result.nonfinite_gradient_values,
                result.update_applied,
            )
        )
        if not bool(host_applied):
            failure = {
                "schema_version": "causal_carbon_adapter_training_failure_v1",
                "arm_id": arm_id,
                "update": update,
                "horizon": horizon,
                "landpoint_id": reference.landpoint_id,
                "year": int(reference.year),
                "loss": float(host_loss),
                "gradient_norm": float(host_gradient_norm),
                "nonfinite_gradient_values": int(host_nonfinite),
                "components": _host_component_record(host_components),
            }
            _atomic_json(arm_root / "training_failure.json", failure)
            raise ValueError(f"causal adapter training failed closed: {failure}")
        parameters = result.parameters
        optimizer = result.optimizer
        completed = update + 1
        interval_losses.append(float(host_loss))
        interval_gradients.append(float(host_gradient_norm))
        component_record = _host_component_record(host_components)
        for name, value in component_record.items():
            interval_components[name].append(float(value))

        if completed % progress_every == 0 or completed == target_update:
            elapsed = time.perf_counter() - interval_started
            record = {
                "completed_updates": completed,
                "last_horizon": horizon,
                "last_landpoint_id": reference.landpoint_id,
                "last_year": int(reference.year),
                "interval_updates": len(interval_losses),
                "interval_elapsed_seconds": elapsed,
                "mean_loss": float(np.mean(interval_losses)),
                "last_loss": interval_losses[-1],
                "mean_gradient_norm": float(np.mean(interval_gradients)),
                "last_gradient_norm": interval_gradients[-1],
                "mean_components": {name: float(np.mean(values)) for name, values in interval_components.items()},
            }
            history.append(record)
            print(
                "causal_adapter_training "
                f"arm={arm_id} update={completed}/{target_update} "
                f"horizon={horizon} loss={record['last_loss']:.8g}",
                flush=True,
            )
            interval_losses.clear()
            interval_gradients.clear()
            interval_components.clear()
            interval_started = time.perf_counter()

        if completed % checkpoint_every == 0 or completed == target_update:
            _atomic_pickle(
                checkpoint_path,
                _build_training_checkpoint(
                    arm_id=arm_id,
                    identity=identity,
                    parameters=parameters,
                    optimizer=optimizer,
                    next_update=completed,
                    schedule=schedule,
                    history=history,
                ),
            )

    final_checkpoint = _load_pickle(checkpoint_path)
    _verify_training_checkpoint(
        final_checkpoint,
        arm_id=arm_id,
        identity=identity,
        schedule=schedule,
    )
    if int(final_checkpoint["next_update"]) != target_update:
        raise ValueError("causal adapter arm stopped before its target update")
    if last_anchor is None:
        raise ValueError("causal adapter arm did not retain an invariance batch")
    invariance = _prediction_invariance(
        base_parameters=experiment.base_parameters,
        trainable_parameters=parameters,
        anchor_batch=last_anchor,
        base_definition=experiment.resources.model_definition,
        adapter_definition=experiment.adapter_definition,
        protected_indices=np.flatnonzero(~np.any(experiment.layout.fast_field_masks, axis=0)),
    )
    if not invariance["protected_columns_bit_exact"]:
        raise ValueError("causal adapter training changed protected columns")
    if not invariance["dynamic_undefined_head_bit_exact"]:
        raise ValueError("causal adapter training changed dynamic undefined output")

    report = {
        "schema_version": (
            "causal_carbon_adapter_training_prefix_v1"
            if target_update < budget
            else "causal_carbon_adapter_training_report_v1"
        ),
        "status": "passed" if target_update < budget else "completed",
        "arm_id": arm_id,
        "identity": identity,
        "execution": {
            "path": str(execution_path),
            "sha256": _sha256_file(execution_path),
        },
        "updates": target_update,
        "full_budget": budget,
        "completed_horizon_counts": final_checkpoint["completed_horizon_counts"],
        "trainable_parameter_count": parameter_count(final_checkpoint["parameters"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "compiled_candidate_executables": len(candidate_steps),
        "final_parent_invariance": invariance,
        "history": final_checkpoint["history"],
        "train_only": True,
        "sealed_test_used": False,
    }
    report_name = f"training_prefix_{target_update:05d}.json" if target_update < budget else "training_report.json"
    return _atomic_json(arm_root / report_name, report)


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

    output_root = Path(output_root).resolve()
    report_path = output_root / "feasibility_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") == "passed":
            return report_path
        raise ValueError("existing causal adapter feasibility root did not pass")
    output_root.mkdir(parents=True, exist_ok=True)

    prepared_experiment = _prepare_adapter_experiment(
        protocol_path=protocol_path,
        dataset_path=dataset_path,
        statistics_path=statistics_path,
        acceptance_path=acceptance_path,
        parent_checkpoint_path=parent_checkpoint_path,
        parent_report_path=parent_report_path,
        plan_path=plan_path,
    )
    protocol = prepared_experiment.protocol
    paths = prepared_experiment.paths
    resources = prepared_experiment.resources
    layout = prepared_experiment.layout
    adapter_definition = prepared_experiment.adapter_definition
    base_parameters = prepared_experiment.base_parameters
    initial_trainable = prepared_experiment.initial_trainable
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
    parser.add_argument(
        "--phase",
        choices=("feasibility", "calibrate", "manifest", "arm"),
        default="feasibility",
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--parent-report", required=True, type=Path)
    parser.add_argument("--feasibility-report", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--arm", choices=ARM_IDS)
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "protocol_path": args.protocol,
        "dataset_path": args.dataset,
        "statistics_path": args.statistics,
        "acceptance_path": args.acceptance,
        "parent_checkpoint_path": args.parent_checkpoint,
        "parent_report_path": args.parent_report,
        "output_root": args.output_root,
        "plan_path": args.plan,
    }
    if args.phase == "feasibility":
        output = run_train_only_feasibility(**common)
    elif args.phase == "calibrate":
        if args.feasibility_report is None:
            raise ValueError("calibration requires --feasibility-report")
        output = run_train_only_calibration(
            **common,
            feasibility_report_path=args.feasibility_report,
        )
    elif args.phase == "manifest":
        if args.feasibility_report is None or args.calibration is None:
            raise ValueError("manifest requires --feasibility-report and --calibration")
        output = create_training_execution(
            **common,
            feasibility_report_path=args.feasibility_report,
            calibration_path=args.calibration,
        )
    else:
        if args.execution is None or args.arm is None:
            raise ValueError("arm phase requires --execution and --arm")
        common.pop("feasibility_report_path", None)
        output = run_training_arm(
            **common,
            arm_id=args.arm,
            execution_path=args.execution,
            stop_after_updates=args.stop_after_updates,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
