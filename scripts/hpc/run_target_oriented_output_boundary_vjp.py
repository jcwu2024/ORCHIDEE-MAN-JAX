"""Localize real-v6 state-loss NaNs at the learned-output boundary."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    PROBE_SEED,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import (
    PARAMETER_COUNT,
    _head,
)
from scripts.hpc.run_target_oriented_full_step_preflight import _prepare_template
from scripts.hpc.target_oriented_training_prepared import (
    load_prepared_target_artifact,
    load_prepared_target_manifest,
)


@dataclass(frozen=True)
class _LearnedOutputBoundaryHead:
    parent: Any

    def assemble_context(self, parameters: Any, inputs: Any) -> None:
        del parameters, inputs
        return None

    def __call__(self, parameters: Mapping[str, Any], context: None, inputs: Any) -> Any:
        del context
        return self.parent.prediction_from_learned_outputs(parameters, inputs)

    def materialize_post_update(
        self,
        parameters: Mapping[str, Any],
        context: None,
        inputs: Any,
        prediction: Any,
        fast_update: Any,
    ) -> Any:
        del parameters, context
        return self.parent.materialize_post_update(
            {},
            None,
            inputs,
            prediction,
            fast_update,
        )


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _sample(value: Any, index: int) -> Any:
    import jax

    return jax.tree_util.tree_map(lambda leaf: leaf[index], value)


def _gradient_report(
    gradients: Mapping[str, Any],
    *,
    group_names: tuple[str, ...],
) -> Mapping[str, Any]:
    host = {name: np.asarray(value) for name, value in gradients.items()}
    result = {}
    for group_index, group_name in enumerate(group_names):
        affected = {}
        element_count = 0
        nonfinite_count = 0
        nonzero_count = 0
        squared_norm = 0.0
        maximum = 0.0
        for field_name, value in host.items():
            current = np.asarray(value[group_index])
            finite = np.isfinite(current)
            local_nonfinite = int(np.size(current) - np.count_nonzero(finite))
            local_nonzero = int(np.count_nonzero(finite & (current != 0.0)))
            finite_values = current[finite]
            local_maximum = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
            element_count += int(np.size(current))
            nonfinite_count += local_nonfinite
            nonzero_count += local_nonzero
            maximum = max(maximum, local_maximum)
            squared_norm += float(np.sum(np.square(finite_values), dtype=np.float64))
            if local_nonfinite:
                affected[field_name] = {
                    "element_count": int(np.size(current)),
                    "nonfinite_count": local_nonfinite,
                    "finite_nonzero_count": local_nonzero,
                    "maximum_finite_absolute_value": local_maximum,
                }
        result[group_name] = {
            "all_finite": nonfinite_count == 0,
            "element_count": element_count,
            "nonfinite_count": nonfinite_count,
            "finite_nonzero_count": nonzero_count,
            "finite_l2_norm": float(np.sqrt(squared_norm)),
            "maximum_finite_absolute_value": maximum,
            "affected_fields": affected,
        }
    return result


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.mechanistic_output_contract import (
        build_mechanistic_learned_schema,
        load_mechanistic_output_contract,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_process_head import (
        build_mechanistic_output_layout,
    )
    from research.daily_coarse_graining.daily_operator.model import (
        bind_daily_operator_dynamic_retained_inputs,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
        load_strong_baseline_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_attention import (
        initialize_target_oriented_attention_parameters,
        target_oriented_attention_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
        load_target_oriented_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        build_target_oriented_training_schedule,
        derive_target_oriented_scales,
        load_target_oriented_state_anchor_policy,
        load_target_oriented_training_contract,
        normalized_error_from_sufficient_statistics,
        one_day_state_sufficient_statistics,
    )
    from research.daily_coarse_graining.daily_operator.types import StateFeatureGroups

    architecture = load_strong_baseline_contract()
    mechanism = load_mechanistic_output_contract()
    target_contract = load_target_oriented_contract()
    training_contract = load_target_oriented_training_contract(args.training_contract)
    state_anchor_policy = load_target_oriented_state_anchor_policy()
    learned_schema = build_mechanistic_learned_schema(runtime["schema"], mechanism)
    layout = build_mechanistic_output_layout(
        adapter=runtime["adapter"],
        complete_schema=runtime["schema"],
        learned_schema=learned_schema,
        owner_contract=runtime["owner_contract"],
        architecture_contract=architecture,
        mechanism_contract=mechanism,
        inputs=runtime["probe_assembled"],
    )
    runtime = dict(runtime) | {
        "architecture_contract": architecture,
        "mechanism_contract": mechanism,
        "learned_schema": learned_schema,
        "target_layout": layout,
    }
    head = _head(runtime, target_contract)
    parameters = initialize_target_oriented_attention_parameters(
        jax.random.PRNGKey(PROBE_SEED),
        schema=learned_schema,
        layout=layout,
        inputs=runtime["probe_assembled"],
        architecture_contract=architecture,
        target_contract=target_contract,
    )
    if target_oriented_attention_parameter_count(parameters) != PARAMETER_COUNT:
        raise ValueError("target-oriented output-boundary parameter count drift")

    references = runtime["index"].select(spatial_split="train", temporal_split="train")
    schedule = build_target_oriented_training_schedule(references, training_contract, updates=8)
    prepared_identity = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))["identity"]
    prepared_manifest = load_prepared_target_manifest(
        args.prepared_manifest,
        training_contract_sha256=training_contract.sha256,
        schedule_sha256=schedule["schedule_sha256"],
        updates=8,
        source_git_head=str(prepared_identity["source_git_head"]),
    )
    artifact = load_prepared_target_artifact(
        args.prepared_manifest,
        prepared_manifest["entries"][0],
        training_contract_sha256=training_contract.sha256,
        schedule_entry=schedule["entries"][0],
    )
    template = _prepare_template(runtime, schedule["entries"][0])
    if template.static_owner.include_modelout_diagnostics:
        template = replace(
            template,
            static_owner=replace(template.static_owner, include_modelout_diagnostics=False),
        )
    scales = derive_target_oriented_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
        state_anchor_policy=state_anchor_policy,
    )

    sample_index = int(args.sample_index)
    if not 0 <= sample_index < 32:
        raise ValueError("sample index must lie in [0, 32)")
    inputs = _sample(artifact.batch.one_day.inputs, sample_index)
    expected_state = _sample(artifact.batch.one_day.expected_next_state, sample_index)
    expected_defined = _sample(artifact.batch.one_day.expected_next_state_defined, sample_index)
    day_inputs = _sample(artifact.batch.one_day.day_inputs, sample_index)
    assembled = runtime["adapter"].input_assembler.assemble(inputs)

    @jax.jit
    def learned_forward(values):
        context = head.assemble_context(values, assembled)
        return head.learned_outputs(values, context, assembled)

    started = time.perf_counter()
    learned = learned_forward(parameters)
    jax.block_until_ready(learned)
    learned_seconds = time.perf_counter() - started
    learned = jax.tree_util.tree_map(jax.lax.stop_gradient, learned)

    boundary_head = _LearnedOutputBoundaryHead(head)
    transition = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=boundary_head,
        retained_tail_template=template,
    )
    group_names = tuple(StateFeatureGroups._fields)

    def group_losses(outputs):
        result = transition(outputs, inputs, day_inputs)
        statistics = one_day_state_sufficient_statistics(
            result.next_state,
            expected_state,
            result.next_state_defined,
            expected_defined,
            scales.state,
            eligible=scales.state_loss_eligible,
        )
        losses = []
        for name in group_names:
            error = normalized_error_from_sufficient_statistics(statistics[name])
            losses.append(0.5 * (error.mse + error.bias_squared))
        return jnp.stack(losses)

    @jax.jit
    def weighted_gradient(outputs, weights):
        return jax.value_and_grad(lambda current: jnp.vdot(group_losses(current), weights))(outputs)

    gradient_started = time.perf_counter()
    gradients = []
    loss_values = None
    for group_index in range(len(group_names)):
        weights = jax.nn.one_hot(group_index, len(group_names), dtype=jnp.float64)
        value, gradient = weighted_gradient(learned, weights)
        jax.block_until_ready((value, gradient))
        gradients.append(gradient)
        if loss_values is None:
            loss_values = jax.jit(group_losses)(learned)
            jax.block_until_ready(loss_values)
    gradient_seconds = time.perf_counter() - gradient_started
    stacked = {
        name: jnp.stack(tuple(gradient[name] for gradient in gradients))
        for name in learned
    }
    gradient_report = _gradient_report(
        jax.device_get(stacked),
        group_names=group_names,
    )
    learned_report = {
        name: {
            "element_count": int(np.size(np.asarray(value))),
            "nonfinite_count": int(np.size(np.asarray(value)) - np.count_nonzero(np.isfinite(np.asarray(value)))),
        }
        for name, value in jax.device_get(learned).items()
    }
    report = {
        "schema_version": "target_oriented_output_boundary_vjp_v1",
        "status": "localized" if any(not value["all_finite"] for value in gradient_report.values()) else "finite",
        "identity": {
            "git_head": _git_head(),
            "training_contract_sha256": training_contract.sha256,
            "state_anchor_policy_sha256": state_anchor_policy.sha256,
            "schedule_sha256": schedule["schedule_sha256"],
        },
        "policy": {
            "sample_index": sample_index,
            "training_updates_retained": 0,
            "checkpoint_written": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "learned_forward_seconds": learned_seconds,
        "output_boundary_gradient_seconds": gradient_seconds,
        "group_loss": {
            name: float(np.asarray(loss_values[index])) for index, name in enumerate(group_names)
        },
        "learned_outputs": learned_report,
        "output_boundary_gradients": gradient_report,
    }
    _atomic_json(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"output-boundary report already exists: {args.output}")
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
