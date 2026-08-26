"""Audit 30-day flux-gradient variance at the frozen update-1000 state."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_checkpoint import (
    load_target_oriented_stage_checkpoint,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_small_fit import _retained_tail_trace_signature
from scripts.hpc.run_target_oriented_attention_gpu_admission import PARAMETER_COUNT, _head
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _add_hard,
    _hard_failure_count,
    _make_steps,
    _prepare_template,
    _slice,
    _validate_prepared_objective_shapes,
    _zero_hard,
)
from scripts.hpc.target_oriented_training_prepared import (
    load_prepared_target_artifact,
    load_prepared_target_manifest_range,
)
from scripts.hpc.target_oriented_validation_prepared import (
    load_target_oriented_validation_prepared,
)

SCHEMA_VERSION = "target_oriented_flux_gradient_variance_audit_v1"
AUDIT_UPDATES = (251, 358, 465, 572, 679, 786, 893, 1000)
AUDIT_FLUX_KEYS = (
    "gpp",
    "maintenance_respiration",
    "growth_respiration",
    "autotrophic_respiration",
    "npp",
)
WINDOWS_PER_BASE_ESTIMATE = 4
AGGREGATE_WINDOW_COUNTS = (4, 8, 16, 32)
CHECKPOINT_UPDATE = 1000
FROZEN_CLIP_THRESHOLD = 5_323_655.8449


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _aggregation_matrix(base_count: int, base_windows: int, aggregate_windows: int) -> np.ndarray:
    if base_count <= 0 or base_windows <= 0 or aggregate_windows < base_windows:
        raise ValueError("gradient aggregation dimensions are invalid")
    if aggregate_windows % base_windows != 0:
        raise ValueError("aggregate window count is not a multiple of the base")
    width = aggregate_windows // base_windows
    if base_count % width != 0:
        raise ValueError("base gradients cannot form disjoint aggregate estimates")
    result = np.zeros((base_count // width, base_count), dtype=np.float64)
    for row in range(result.shape[0]):
        result[row, row * width : (row + 1) * width] = 1.0 / width
    return result


def _cosine_matrix(gram: np.ndarray) -> np.ndarray:
    values = np.asarray(gram, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("gradient Gram matrix must be square")
    norms = np.sqrt(np.maximum(np.diag(values), 0.0))
    denominator = np.outer(norms, norms)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(values, denominator, out=result, where=denominator > 0.0)
    finite = np.isfinite(result)
    result[finite] = np.clip(result[finite], -1.0, 1.0)
    return result


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _summarize_gram(
    gram: np.ndarray,
    *,
    base_windows: int = WINDOWS_PER_BASE_ESTIMATE,
) -> Mapping[str, Any]:
    values = np.asarray(gram, dtype=np.float64)
    if values.shape != (len(AUDIT_UPDATES), len(AUDIT_UPDATES)):
        raise ValueError("gradient Gram matrix does not cover the frozen audit groups")
    if not np.all(np.isfinite(values)) or not np.allclose(values, values.T, atol=1.0e-5, rtol=1.0e-5):
        raise ValueError("gradient Gram matrix is nonfinite or asymmetric")
    reference = np.full(len(AUDIT_UPDATES), 1.0 / len(AUDIT_UPDATES), dtype=np.float64)
    reference_norm_squared = float(reference @ values @ reference)
    result = {}
    for window_count in AGGREGATE_WINDOW_COUNTS:
        aggregation = _aggregation_matrix(len(AUDIT_UPDATES), base_windows, window_count)
        aggregate_gram = aggregation @ values @ aggregation.T
        cosine = _cosine_matrix(aggregate_gram)
        norms = np.sqrt(np.maximum(np.diag(aggregate_gram), 0.0))
        estimates = int(aggregation.shape[0])
        if estimates > 1:
            upper = cosine[np.triu_indices(estimates, 1)]
            pairwise = {
                "minimum": _finite_or_none(float(np.nanmin(upper))),
                "median": _finite_or_none(float(np.nanmedian(upper))),
                "mean": _finite_or_none(float(np.nanmean(upper))),
                "negative_fraction": _finite_or_none(float(np.mean(upper < 0.0))),
            }
        else:
            pairwise = {"minimum": None, "median": None, "mean": None, "negative_fraction": None}
        dot_reference = aggregation @ values @ reference
        denominator = norms * np.sqrt(max(reference_norm_squared, 0.0))
        cosine_reference = np.full(estimates, np.nan, dtype=np.float64)
        np.divide(dot_reference, denominator, out=cosine_reference, where=denominator > 0.0)
        finite_reference = np.isfinite(cosine_reference)
        cosine_reference[finite_reference] = np.clip(cosine_reference[finite_reference], -1.0, 1.0)
        mean_coefficients = np.mean(aggregation, axis=0)
        mean_norm_squared = float(mean_coefficients @ values @ mean_coefficients)
        mean_squared_norm = float(np.mean(np.diag(aggregate_gram)))
        noise = max(mean_squared_norm - mean_norm_squared, 0.0)
        result[str(window_count)] = {
            "estimate_count": estimates,
            "gradient_norm_minimum": float(np.min(norms)),
            "gradient_norm_median": float(np.median(norms)),
            "gradient_norm_maximum": float(np.max(norms)),
            "pairwise_cosine": pairwise,
            "cosine_to_32_window_reference": {
                "minimum": _finite_or_none(float(np.nanmin(cosine_reference))),
                "median": _finite_or_none(float(np.nanmedian(cosine_reference))),
                "mean": _finite_or_none(float(np.nanmean(cosine_reference))),
            },
            "relative_gradient_variance": (
                float(noise / mean_norm_squared) if mean_norm_squared > 0.0 else None
            ),
            "signal_to_noise_ratio": (
                float(mean_norm_squared / noise) if noise > 0.0 else None
            ),
        }
    return result


def _classify_batch_effect(key_reports: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    supported = []
    adequate = []
    details = {}
    for name in AUDIT_FLUX_KEYS:
        sizes = key_reports[name]["all_parameters"]
        cosine_4 = sizes["4"]["pairwise_cosine"]["median"]
        cosine_16 = sizes["16"]["pairwise_cosine"]["median"]
        supports_larger = (
            cosine_4 is not None
            and cosine_16 is not None
            and cosine_4 < 0.25
            and cosine_16 >= 0.50
            and cosine_16 - cosine_4 >= 0.25
        )
        current_adequate = cosine_4 is not None and cosine_4 >= 0.50
        if supports_larger:
            supported.append(name)
        if current_adequate:
            adequate.append(name)
        details[name] = {
            "four_window_pairwise_median_cosine": cosine_4,
            "sixteen_window_pairwise_cosine": cosine_16,
            "supports_larger_effective_batch": supports_larger,
            "current_four_window_batch_adequate": current_adequate,
        }
    if len(supported) >= 3:
        action = "larger_effective_batch_supported"
    elif len(adequate) >= 4:
        action = "current_four_window_batch_not_primary_blocker"
    else:
        action = "mixed_batch_and_objective_geometry"
    return {
        "action": action,
        "larger_batch_supported_keys": supported,
        "current_batch_adequate_keys": adequate,
        "details": details,
        "thresholds": {
            "unstable_four_window_median_cosine_below": 0.25,
            "stable_sixteen_window_pairwise_cosine_minimum": 0.50,
            "minimum_cosine_gain": 0.25,
            "key_count_for_action": 3,
        },
    }


def _audit_family_trees(gradient: Mapping[str, Any]) -> Mapping[str, Any]:
    target = gradient["target_oriented"]
    expected = {
        "domain_embedding",
        "domain_blocks",
        "science_query",
        "science_embedding",
        "science_cross_blocks",
        "science_mixer_blocks",
        "category_embedding",
        "primary_blocks",
        "state_blocks",
        "primary_decoder",
        "state_decoder",
        "primary_heads",
        "state_heads",
        "auxiliary_science_projection",
        "auxiliary_bottleneck",
        "auxiliary_label_heads",
    }
    if set(target) != expected:
        raise ValueError("target-oriented gradient family registry drift")
    return {
        "base_source_encoder": {name: value for name, value in gradient.items() if name != "target_oriented"},
        "shared_science_backbone": {
            name: target[name]
            for name in (
                "domain_embedding",
                "domain_blocks",
                "science_query",
                "science_embedding",
                "science_cross_blocks",
                "science_mixer_blocks",
                "category_embedding",
            )
        },
        "primary_path": {
            name: target[name]
            for name in ("primary_blocks", "primary_decoder", "primary_heads")
        },
        "state_path": {
            name: target[name]
            for name in ("state_blocks", "state_decoder", "state_heads")
        },
        "auxiliary_path": {
            name: target[name]
            for name in (
                "auxiliary_science_projection",
                "auxiliary_bottleneck",
                "auxiliary_label_heads",
            )
        },
    }


def _tree_size(jax: Any, tree: Any) -> int:
    return sum(int(np.prod(leaf.shape, dtype=np.int64)) for leaf in jax.tree_util.tree_leaves(tree))


def _zero_gradient(jax: Any, jnp: Any, parameters: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: jnp.zeros_like(value, dtype=jnp.float32), parameters)


def _add_gradient(jax: Any, jnp: Any, left: Any, right: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda first, second: first + jnp.asarray(second, dtype=jnp.float32),
        left,
        right,
    )


def _mean_gradient(jax: Any, jnp: Any, gradients: Sequence[Any]) -> Any:
    if not gradients:
        raise ValueError("cannot average an empty gradient sequence")
    total = _zero_gradient(jax, jnp, gradients[0])
    for gradient in gradients:
        total = _add_gradient(jax, jnp, total, gradient)
    return jax.tree_util.tree_map(lambda value: value / len(gradients), total)


def _family_gram(jax: Any, jnp: Any, gradients: Sequence[Any]) -> Mapping[str, np.ndarray]:
    from jax.flatten_util import ravel_pytree

    families = tuple(_audit_family_trees(gradients[0]))
    result = {}
    for name in families:
        vectors = [
            ravel_pytree(_audit_family_trees(gradient)[name])[0].astype(jnp.float32)
            for gradient in gradients
        ]
        matrix = jnp.stack(vectors, axis=0)
        gram = matrix @ matrix.T
        jax.block_until_ready(gram)
        result[name] = np.asarray(jax.device_get(gram), dtype=np.float64)
    result["all_parameters"] = sum(result.values(), np.zeros_like(next(iter(result.values()))))
    return result


def _hard_host(value: Any) -> Mapping[str, int]:
    return {name: int(np.asarray(item)) for name, item in value._asdict().items()}


def _load_context(args: argparse.Namespace) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.mechanistic_output_contract import (
        build_mechanistic_learned_schema,
        load_mechanistic_output_contract,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_process_head import (
        build_mechanistic_output_layout,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
        load_strong_baseline_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
        load_target_oriented_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        build_target_oriented_training_schedule,
        derive_target_oriented_acceptance_scales,
        derive_target_oriented_scales,
        load_target_oriented_state_anchor_policy,
        load_target_oriented_training_contract,
        verify_target_oriented_validation_panel,
    )

    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    import jax.numpy as jnp

    runtime = dict(runtime) | {"jnp": jnp}
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
    references = runtime["index"].select(spatial_split="train", temporal_split="train")
    schedule = build_target_oriented_training_schedule(
        references,
        training_contract,
        updates=int(args.prepared_updates),
    )
    raw_prepared = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))
    prepared = load_prepared_target_manifest_range(
        args.prepared_manifest,
        training_contract_sha256=training_contract.sha256,
        schedule_sha256=schedule["schedule_sha256"],
        updates=int(args.prepared_updates),
        source_git_head=str(raw_prepared.get("identity", {}).get("source_git_head", "")),
        start_update=int(raw_prepared["start_update"]),
        end_update=int(raw_prepared["end_update"]),
    )
    prepared_records = {int(record["update"]): record for record in prepared["entries"]}
    if any(update not in prepared_records for update in AUDIT_UPDATES):
        raise ValueError("prepared manifest does not cover the frozen audit updates")
    validation_prepared, panel = load_target_oriented_validation_prepared(
        args.validation_prepared_manifest,
        training_contract_sha256=training_contract.sha256,
        target_contract_sha256=target_contract.sha256,
    )
    verify_target_oriented_validation_panel(panel, runtime["index"], training_contract, target_contract)
    template = _prepare_template(runtime, schedule["entries"][AUDIT_UPDATES[0] - 1])
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
    acceptance_scales = derive_target_oriented_acceptance_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    first_artifact = load_prepared_target_artifact(
        args.prepared_manifest,
        prepared_records[AUDIT_UPDATES[0]],
        training_contract_sha256=training_contract.sha256,
        schedule_entry=schedule["entries"][AUDIT_UPDATES[0] - 1],
    )
    _validate_prepared_objective_shapes(first_artifact, complete_schema=runtime["schema"], scales=scales)
    if first_artifact.trace_signature != _retained_tail_trace_signature(template):
        raise ValueError("flux-gradient audit retained executable structure drift")
    if any(record["trace_signature"] != first_artifact.trace_signature for record in validation_prepared["entries"]):
        raise ValueError("flux-gradient audit validation retained structure drift")
    checkpoint = load_target_oriented_stage_checkpoint(
        args.stage_checkpoint,
        expected_sha256=args.stage_checkpoint_sha256,
        completed_updates=CHECKPOINT_UPDATE,
        parameter_count=PARAMETER_COUNT,
        clip_threshold=FROZEN_CLIP_THRESHOLD,
        training_contract_sha256=training_contract.sha256,
        target_contract_sha256=target_contract.sha256,
        state_anchor_policy_sha256=state_anchor_policy.sha256,
        validation_panel_sha256=panel["panel_sha256"],
    )
    parameters = jax.tree_util.tree_map(jnp.asarray, checkpoint["parameters"])
    steps = _make_steps(
        runtime,
        _head(runtime, target_contract),
        template,
        scales,
        target_contract,
        acceptance_scales=acceptance_scales,
        acceptance_flux_names=AUDIT_FLUX_KEYS,
    )
    return {
        "runtime": runtime,
        "parameters": parameters,
        "steps": steps,
        "schedule": schedule,
        "prepared_records": prepared_records,
        "training_contract": training_contract,
        "target_contract": target_contract,
        "state_anchor_policy": state_anchor_policy,
        "panel": panel,
        "trace_signature": first_artifact.trace_signature,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    context = _load_context(args)
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    parameters = context["parameters"]
    steps = context["steps"]
    schedule = context["schedule"]
    records = context["prepared_records"]
    training_contract = context["training_contract"]
    target_contract = context["target_contract"]
    state_anchor_policy = context["state_anchor_policy"]
    panel = context["panel"]

    parameter_families = _audit_family_trees(parameters)
    family_parameter_counts = {name: _tree_size(jax, tree) for name, tree in parameter_families.items()}
    if sum(family_parameter_counts.values()) != PARAMETER_COUNT:
        raise ValueError("flux-gradient audit parameter family coverage drift")
    memory_before = _memory_report(runtime["devices"][0])

    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        add_target_oriented_acceptance_statistics,
    )

    groups = []
    hard = _zero_hard(jnp)
    group_identities = []
    forward_started = time.perf_counter()
    for update in AUDIT_UPDATES:
        artifact = load_prepared_target_artifact(
            args.prepared_manifest,
            records[update],
            training_contract_sha256=training_contract.sha256,
            schedule_entry=schedule["entries"][update - 1],
        )
        if artifact.trace_signature != context["trace_signature"]:
            raise ValueError("flux-gradient audit prepared trace drift")
        microbatches = [_slice(artifact.batch.thirty_day, index, index + 1) for index in range(4)]
        values = [steps["thirty_acceptance_forward"](parameters, batch) for batch in microbatches]
        jax.block_until_ready(values)
        total = values[0][0]
        for value in values[1:]:
            total = add_target_oriented_acceptance_statistics(total, value[0])
        for value in values:
            hard = _add_hard(jax, hard, value[1])
        groups.append({"update": update, "microbatches": microbatches, "total": total})
        group_identities.append(
            {
                "update": update,
                "runtime_year": int(schedule["entries"][update - 1]["runtime_year"]),
                "landpoint_ids": [
                    str(record["landpoint_id"])
                    for record in schedule["entries"][update - 1]["components"]["thirty_day"]
                ],
            }
        )
        print(f"flux_gradient_audit forward update={update}", flush=True)
    jax.block_until_ready((hard, tuple(group["total"] for group in groups)))
    hard_host = _hard_host(jax.device_get(hard))
    if _hard_failure_count(hard_host) != 0:
        raise RuntimeError(f"flux-gradient audit failed a physical hard gate: {hard_host}")
    forward_seconds = time.perf_counter() - forward_started

    key_reports = {}
    aggregate_gradients = []
    gradient_started = time.perf_counter()
    for key_index, key in enumerate(AUDIT_FLUX_KEYS):
        weights = jnp.zeros((len(AUDIT_FLUX_KEYS),), dtype=jnp.float64).at[key_index].set(1.0)
        group_gradients = []
        group_losses = []
        for group in groups:
            gradient = _zero_gradient(jax, jnp, parameters)
            loss = jnp.asarray(0.0, dtype=jnp.float64)
            for microbatch in group["microbatches"]:
                value, current = steps["thirty_acceptance_gradient"](
                    parameters,
                    microbatch,
                    group["total"],
                    weights,
                )
                loss = loss + value
                gradient = _add_gradient(jax, jnp, gradient, current)
            jax.block_until_ready((loss, gradient))
            group_losses.append(float(np.asarray(loss)))
            group_gradients.append(gradient)
            print(
                f"flux_gradient_audit key={key} update={group['update']} loss={group_losses[-1]:.9g}",
                flush=True,
            )
        family_grams = _family_gram(jax, jnp, group_gradients)
        key_reports[key] = {
            name: _summarize_gram(gram)
            for name, gram in family_grams.items()
        }
        key_reports[key]["group_losses"] = group_losses
        aggregate_gradients.append(_mean_gradient(jax, jnp, group_gradients))
        del group_gradients

    cross_key_grams = _family_gram(jax, jnp, aggregate_gradients)
    cross_key = {}
    for family, gram in cross_key_grams.items():
        cross_key[family] = {
            "keys": list(AUDIT_FLUX_KEYS),
            "gradient_norms": [float(value) for value in np.sqrt(np.maximum(np.diag(gram), 0.0))],
            "cosine_matrix": _cosine_matrix(gram).tolist(),
        }
    decision = _classify_batch_effect(key_reports)
    all_cosine = np.asarray(cross_key["all_parameters"]["cosine_matrix"], dtype=np.float64)
    indexes = {name: index for index, name in enumerate(AUDIT_FLUX_KEYS)}
    maintenance_growth = float(all_cosine[indexes["maintenance_respiration"], indexes["growth_respiration"]])
    npp_gpp = float(all_cosine[indexes["npp"], indexes["gpp"]])
    npp_autotrophic = float(all_cosine[indexes["npp"], indexes["autotrophic_respiration"]])
    decision["objective_geometry"] = {
        "maintenance_growth_cosine": maintenance_growth,
        "npp_gpp_cosine": npp_gpp,
        "npp_autotrophic_respiration_cosine": npp_autotrophic,
        "respiration_decomposition_conflict": maintenance_growth < -0.25,
        "derived_npp_basis_conflict": min(npp_gpp, npp_autotrophic) < -0.25,
    }
    gradient_seconds = time.perf_counter() - gradient_started
    memory_after = _memory_report(runtime["devices"][0])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "identity": {
            "git_head": _git_head(),
            "stage_checkpoint": str(args.stage_checkpoint.resolve()),
            "stage_checkpoint_sha256": args.stage_checkpoint_sha256,
            "completed_updates": CHECKPOINT_UPDATE,
            "prepared_manifest": str(args.prepared_manifest.resolve()),
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
            "validation_panel_sha256": panel["panel_sha256"],
            "training_contract_sha256": training_contract.sha256,
            "target_contract_sha256": target_contract.sha256,
            "state_anchor_policy_sha256": state_anchor_policy.sha256,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "optimizer_updates_retained": 0,
            "checkpoint_written": False,
            "horizon_days": 30,
            "audit_updates": list(AUDIT_UPDATES),
            "base_window_count": WINDOWS_PER_BASE_ESTIMATE,
            "aggregate_window_counts": list(AGGREGATE_WINDOW_COUNTS),
            "flux_keys": list(AUDIT_FLUX_KEYS),
        },
        "groups": group_identities,
        "family_parameter_counts": family_parameter_counts,
        "hard_counts": hard_host,
        "hard_failure_count": _hard_failure_count(hard_host),
        "key_reports": key_reports,
        "cross_key_geometry": cross_key,
        "decision": decision,
        "timing_seconds": {
            "forward": forward_seconds,
            "gradient_and_gram": gradient_seconds,
            "total": time.perf_counter() - started,
        },
        "memory_before": memory_before,
        "memory_after": memory_after,
    }
    _atomic_json(output, report)
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
    parser.add_argument("--prepared-updates", type=int, default=CHECKPOINT_UPDATE)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
