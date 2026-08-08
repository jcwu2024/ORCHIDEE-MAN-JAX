"""No-checkpoint real-v6 GPU smoke for the successor process-head pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend

EXPECTED = {
    "dataset_manifest": "5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99",
    "acceptance_report": "8784ed42dc64ef77c9642f613eb2e5940635edc6e744a1af4ea8634293baa8e3",
    "base_statistics_json": "ab7f040a930071bd5604c17355a6493b23f3c76cba1e3084dcd5d5653101a4bf",
    "base_statistics_npz": "e88d7cec504c6051aab6c9feacb619dca94961730ffa18f840563a48fc815a3a",
    "process_statistics_json": "9bc2cab43b3279e39a0fc2876ca298b2b371d6890986d44ef035859ca925eaab",
    "process_statistics_npz": "addf957afbd8338f4aa250155dcd299d568c1070b644e30279f9bf0ea37f9bf8",
    "successor_contract": "bb8bb34972e6fdff713f06f7ce48a2faa9eb4991af9f269c9aab26d96b592b5c",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str, name: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{name} hash drift: {observed} != {expected}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _stack_trees(jax: Any, values: list[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *items: np.stack(items), *values)


def _gradient_report(jax: Any, gradient: Any) -> dict[str, Any]:
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(gradient)]
    return {
        "leaves": len(leaves),
        "finite": all(np.all(np.isfinite(leaf)) for leaf in leaves),
        "norm": float(
            np.sqrt(
                sum(np.sum(leaf * leaf, dtype=np.float64) for leaf in leaves)
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size != 2:
        raise ValueError("successor real-v6 smoke batch is frozen at two rows")

    base_payload = json.loads(args.base_statistics.read_text(encoding="utf-8"))
    base_npz = args.base_statistics.parent / str(base_payload["statistics_npz"])
    process_payload = json.loads(
        args.process_statistics.read_text(encoding="utf-8")
    )
    process_npz = args.process_statistics.parent / str(
        process_payload["statistics_npz"]
    )
    for name, path in (
        ("dataset_manifest", args.dataset_manifest),
        ("acceptance_report", args.acceptance_report),
        ("base_statistics_json", args.base_statistics),
        ("base_statistics_npz", base_npz),
        ("process_statistics_json", args.process_statistics),
        ("process_statistics_npz", process_npz),
    ):
        _verify(path.resolve(), EXPECTED[name], name)

    jax, devices = initialize_gpu_backend()
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_markov_contract import (
        daily_markov_contract_from_metadata,
        load_markov_shard,
    )
    from research.daily_coarse_graining.daily_operator.conservative_update import (
        apply_conservative_update,
        validate_conservative_result,
    )
    from research.daily_coarse_graining.daily_operator.process_derivations import (
        NativeForcingFieldLayout,
    )
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
        SOURCE_ROUTED_GRAPH_CONTROL,
        assemble_successor_process_context,
        build_successor_token_layout,
        initialize_successor_parameters,
        successor_parameter_count,
        successor_process_outputs,
    )
    from research.daily_coarse_graining.daily_operator.successor_contract import (
        load_successor_candidate_contract,
    )
    from research.daily_coarse_graining.daily_operator.successor_process_head import (
        SuccessorProcessHead,
    )
    from research.daily_coarse_graining.daily_operator.training_contract import (
        load_first_candidate_contract,
    )
    from research.daily_coarse_graining.daily_operator.training_data import (
        DailyProcessTrainingAdapter,
    )
    from research.daily_coarse_graining.daily_operator.training_loss import (
        masked_equal_label_macro_mse,
    )
    from research.daily_coarse_graining.daily_operator.training_statistics import (
        load_process_training_statistics,
    )
    from research.daily_coarse_graining.markov_dataset import (
        load_training_statistics,
    )
    from research.daily_coarse_graining.typed_sidecar import (
        load_coherent_typed_sidecar_index,
        read_typed_sidecar_shard,
    )

    successor_contract = load_successor_candidate_contract()
    if successor_contract.sha256 != EXPECTED["successor_contract"]:
        raise ValueError("successor contract hash drift")
    successor_data = successor_contract.payload["data"]
    if (
        successor_data["training_statistics_json_sha256"]
        != EXPECTED["process_statistics_json"]
        or successor_data["training_statistics_npz_sha256"]
        != EXPECTED["process_statistics_npz"]
    ):
        raise ValueError("successor process-statistics identity drift")

    acceptance = json.loads(args.acceptance_report.read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed":
        raise ValueError("v6 acceptance report is not passed")
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    if contract.sha256 != successor_data["markov_contract_sha256"]:
        raise ValueError("successor Markov contract drift")
    adapter = DailyProcessTrainingAdapter.from_contract(contract)
    index = load_coherent_typed_sidecar_index(
        args.dataset_manifest,
        verify_hashes=False,
    )
    base_statistics = load_training_statistics(
        args.base_statistics,
        index=index.parent_index,
    )
    process_statistics = load_process_training_statistics(
        args.process_statistics,
        index=index,
        adapter=adapter,
        candidate_contract=load_first_candidate_contract(),
        base_statistics=base_statistics,
    )
    if process_statistics.sample_count != 8_591_565:
        raise ValueError("formal process-statistics sample count drift")

    reference = index.select(spatial_split="train", temporal_split="train")[0]
    parent = load_markov_shard(reference.parent.path, contract=contract)
    typed_values, typed_defined, day_index = read_typed_sidecar_shard(
        reference.path,
        contract=index.contract,
        expected_days=reference.day_count,
    )
    if not np.array_equal(day_index, parent.day_index):
        raise ValueError("selected coherent shard day index drift")
    examples = [
        adapter.assemble(
            parent.sample(row),
            supplemental_values={
                name: value[row] for name, value in typed_values.items()
            },
            supplemental_defined={
                name: value[row] for name, value in typed_defined.items()
            },
        )
        for row in range(args.batch_size)
    ]
    assembled = [
        adapter.input_assembler.assemble(example.inputs) for example in examples
    ]
    first = assembled[0]
    schema = process_statistics.schema
    layout = build_successor_token_layout(
        adapter.input_assembler,
        schema,
        successor_contract,
    )
    input_normalization = process_statistics.input_normalization()
    output_normalization = process_statistics.output_normalization()
    loss_scales = process_statistics.loss_scales
    field_labels = process_statistics.field_labels
    batch_inputs = _stack_trees(jax, assembled)
    batch_targets = _stack_trees(
        jax,
        [example.targets.supervision for example in examples],
    )
    parameters = initialize_successor_parameters(
        jax.random.PRNGKey(20260809),
        schema=schema,
        layout=layout,
        inputs=first,
        arm=SOURCE_ROUTED_ATTENTION,
    )
    parameter_count = successor_parameter_count(parameters)
    if parameter_count != 8_679_681:
        raise ValueError("successor parameter count drift")

    def one_loss(current_parameters, inputs, targets, arm):
        context = assemble_successor_process_context(
            current_parameters,
            inputs,
            normalization=input_normalization,
            layout=layout,
        )
        outputs = successor_process_outputs(
            current_parameters,
            context,
            inputs,
            schema=schema,
            layout=layout,
            normalization=output_normalization,
            forcing_layout=forcing_layout,
            arm=arm,
        )
        return masked_equal_label_macro_mse(
            outputs,
            targets,
            scales=loss_scales,
            field_labels=field_labels,
        ).total

    forcing_fields = contract.native_forcing.fields
    forcing_layout = NativeForcingFieldLayout(
        forcing_fields.index("Rainf"),
        forcing_fields.index("Rainf") + 1,
        forcing_fields.index("Snowf"),
        forcing_fields.index("Snowf") + 1,
        forcing_fields.index("Tair"),
        forcing_fields.index("Tair") + 1,
    )
    arms = {}
    for arm in (SOURCE_ROUTED_ATTENTION, SOURCE_ROUTED_GRAPH_CONTROL):

        def batch_loss(current_parameters):
            return jnp.mean(
                jax.vmap(
                    lambda inputs, targets: one_loss(
                        current_parameters,
                        inputs,
                        targets,
                        arm,
                    )
                )(batch_inputs, batch_targets)
            )

        reverse_started = time.perf_counter()
        loss, gradient = jax.jit(jax.value_and_grad(batch_loss))(parameters)
        jax.block_until_ready((loss, gradient))
        reverse_seconds = time.perf_counter() - reverse_started
        tangent = jax.tree_util.tree_map(
            lambda value: jnp.ones_like(value) / np.sqrt(float(parameter_count)),
            parameters,
        )
        forward_started = time.perf_counter()
        primal, directional = jax.jit(
            lambda current, direction: jax.jvp(
                batch_loss,
                (current,),
                (direction,),
            )
        )(parameters, tangent)
        jax.block_until_ready((primal, directional))
        forward_seconds = time.perf_counter() - forward_started
        gradient_report = _gradient_report(jax, gradient)
        finite = (
            np.isfinite(float(loss))
            and gradient_report["finite"]
            and gradient_report["norm"] > 0.0
            and np.isfinite(float(directional))
        )
        arms[arm] = {
            "status": "passed" if finite else "failed",
            "parameter_count": parameter_count,
            "loss": float(loss),
            "reverse_gradient": gradient_report,
            "forward_directional_derivative": float(directional),
            "jit_reverse_seconds": reverse_seconds,
            "jit_forward_seconds": forward_seconds,
        }

    head = SuccessorProcessHead(
        arm=SOURCE_ROUTED_ATTENTION,
        contract=contract,
        input_assembler=adapter.input_assembler,
        schema=schema,
        token_layout=layout,
        input_normalization=input_normalization,
        output_normalization=output_normalization,
        forcing_layout=forcing_layout,
        forcing_indices={
            name: index for index, name in enumerate(forcing_fields)
        },
    )
    context = head.assemble_context(parameters, first)
    prediction = head(parameters, context, first)
    update = apply_conservative_update(first, prediction)
    validate_conservative_result(update)
    water_residual = float(update.water.budget_residual)
    carbon_residual = float(update.carbon.budget_residual)
    conservative = (
        np.isfinite(water_residual)
        and np.isfinite(carbon_residual)
        and abs(water_residual) <= 2.0e-12
        and abs(carbon_residual) <= 2.0e-12
        and float(jnp.min(update.water.inventory)) >= -2.0e-12
        and float(jnp.min(update.carbon.inventory)) >= -2.0e-12
    )

    def fast_parameter_objective(values):
        current = first._replace(
            pft=first.pft._replace(pft_parameters=values)
        )
        outputs = head.named_outputs(parameters, context, current)
        return jnp.sum(outputs["daily.fast_gpp"]) + jnp.sum(
            outputs["daily.fast_maintenance_respiration"]
        )

    parameter_gradient = np.asarray(
        jax.jit(jax.grad(fast_parameter_objective))(
            first.pft.pft_parameters
        )
    )
    vegetation = (
        np.asarray(first.pft.active_pft_mask)
        & (np.asarray(first.pft.pft_fraction) > 0.0)
        & (np.asarray(first.pft.pft_traits)[:, 0] < 0.5)
    )
    parameter_routes = (
        np.all(np.isfinite(parameter_gradient))
        and np.any(vegetation)
        and np.all(parameter_gradient[vegetation, :4] != 0.0)
        and np.all(parameter_gradient[:, 4:] == 0.0)
    )

    direct_passed = (
        all(value["status"] == "passed" for value in arms.values())
        and conservative
        and parameter_routes
        and len(prediction.named_outputs) == 101
        and sum(
            np.asarray(value).size for value in prediction.named_outputs.values()
        )
        == 38_530
    )
    report = {
        "schema_version": "successor_process_head_real_v6_gpu_smoke_v1",
        "status": "passed" if direct_passed else "failed",
        "promotion_authorized": False,
        "checkpoint_written": False,
        "historical_v5_used": False,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "jax_version": jax.__version__,
        "batch_size": args.batch_size,
        "successor_contract_sha256": successor_contract.sha256,
        "data": {
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "acceptance_report_sha256": EXPECTED["acceptance_report"],
            "base_statistics_json_sha256": EXPECTED["base_statistics_json"],
            "base_statistics_npz_sha256": EXPECTED["base_statistics_npz"],
            "process_statistics_json_sha256": EXPECTED[
                "process_statistics_json"
            ],
            "process_statistics_npz_sha256": EXPECTED["process_statistics_npz"],
            "landpoint_id": reference.parent.landpoint_id,
            "year": reference.parent.year,
            "rows": list(range(args.batch_size)),
        },
        "architecture": {
            "parameter_count": parameter_count,
            "source_family_count": len(layout.family_names),
            "input_token_count": int(context.input_tokens.shape[0]),
            "named_output_count": len(prediction.named_outputs),
            "output_element_count": int(
                sum(
                    np.asarray(value).size
                    for value in prediction.named_outputs.values()
                )
            ),
        },
        "arms": arms,
        "one_day_conservative_construction": {
            "status": "passed" if conservative else "failed",
            "water_budget_residual": water_residual,
            "carbon_budget_residual": carbon_residual,
            "water_minimum_inventory": float(jnp.min(update.water.inventory)),
            "carbon_minimum_inventory": float(jnp.min(update.carbon.inventory)),
        },
        "parameter_routes": {
            "status": "passed" if parameter_routes else "failed",
            "gradient": parameter_gradient.tolist(),
            "vegetation_mask": vegetation.tolist(),
            "retained_exact_columns_zero": bool(
                np.all(parameter_gradient[:, 4:] == 0.0)
            ),
        },
        "pending_required_gates": [
            "canonical_retained_tail_cold_ordinary_restart_with_successor_outputs",
            "real_30_day_rollout_and_train_train_gradient_ratio_preflight",
        ],
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2))
    return 0 if direct_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
