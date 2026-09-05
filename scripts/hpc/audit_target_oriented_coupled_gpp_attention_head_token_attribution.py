"""Test head-slice and source-token causes of the accepted Day-5 GPP jump."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_attention_head_token_attribution import (
    ATTENTION_HEADS,
    DOC_BANK_INDEX,
    HEAD_WIDTH,
    LITTERPART_BANK_INDEX,
    MODEL_WIDTH,
    PARENT_CLASSIFICATION,
    PARENT_REPORT_SHA256,
    PARENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TARGET_BLOCK_INDEX,
    TARGET_HEAD_INDEX,
    TARGET_TOKEN_KEYS,
    attention_token_summary,
    classify_head_controls,
    projection_head_delta_coverage,
    replace_block_attention_head,
    state_summary_bank_indices,
    summarize_token_mask_controls,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    coupled_gpp_trainable_tree,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_potential_internal_attribution import (
    LOWER_FRACTION,
    UPPER_FRACTION,
    traced_potential_branch,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_snowheat_causal import (
    TARGET_ARTIFACT_UPDATE,
    TARGET_DAY_INDEX,
    TARGET_WINDOW_INDEX,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
    FORCING_FIELDS,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_day5_crossed_attribution import (
    _runtime_year_for_update,
    _tree_day_batch,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_localized_jump import (
    FORWARD_REPRODUCTION_ATOL,
    FORWARD_REPRODUCTION_RTOL,
    _make_trace_evaluator,
    reconstruct_localized_jump_parameters,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_potential_internal_attribution import (
    EXPECTED_RUNTIME_YEAR,
    EXPECTED_VEGETATION_PFT_INDEX,
    _evaluate_blocks,
    _make_potential_input_evaluator,
    _make_shadow_evaluators,
    _policy_record,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_potential_internal_attribution import (
    _parser as _internal_parser,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_snowheat_causal import (
    FACTORIZATION_ATOL,
    FACTORIZATION_RTOL,
    _make_factor_evaluator,
    _scenario_day_starts,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_structured_precision_attribution import (
    _direction_integrity,
    _interpolate_tree,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    _initialization_only,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import _git_head
from scripts.hpc.run_target_oriented_full_step_preflight import _slice


def _load_parent_report(path: Path) -> tuple[Mapping[str, Any], str]:
    digest = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    expected_target = {
        "artifact_update": TARGET_ARTIFACT_UPDATE,
        "runtime_year": EXPECTED_RUNTIME_YEAR,
        "window_index": TARGET_WINDOW_INDEX,
        "day_index": TARGET_DAY_INDEX,
        "vegetation_pft_index": EXPECTED_VEGETATION_PFT_INDEX,
        "lower_fraction": LOWER_FRACTION,
        "upper_fraction": UPPER_FRACTION,
    }
    if digest != PARENT_REPORT_SHA256:
        raise ValueError("head-token parent report SHA256 drift")
    if (
        report.get("schema_version") != PARENT_SCHEMA_VERSION
        or report.get("status") != "passed"
        or report.get("classification") != PARENT_CLASSIFICATION
        or report.get("integrity", {}).get("engineering_gate_passed") is not True
        or report.get("target") != expected_target
    ):
        raise ValueError("head-token parent report status or target drift")
    return report, digest


def _parent_endpoint(parent: Mapping[str, Any], fraction: float) -> Mapping[str, Any]:
    matches = tuple(
        record
        for record in parent["isolated_potential_path"]["production_float32"]
        if float(record["fraction"]) == float(fraction)
    )
    if len(matches) != 1:
        raise ValueError(f"head-token parent endpoint is absent: {fraction}")
    return matches[0]


def _make_masked_trace_evaluator(reconstruction: Any) -> Any:
    jax = reconstruction.jax
    jnp = reconstruction.jnp
    heads = int(reconstruction.bootstrap["head"].structured_config.attention_heads)

    @jax.jit
    def trace(
        blocks: Any,
        output: Any,
        branch_inputs: Mapping[str, Any],
        attention_head_mask: Any,
    ) -> Any:
        return traced_potential_branch(
            jax,
            jnp,
            blocks,
            output,
            branch_inputs["query"],
            branch_inputs["bank"],
            branch_inputs["bank_mask"],
            branch_inputs["active"],
            heads=heads,
            dtype=jnp.float32,
            attention_head_mask=attention_head_mask,
        )

    return trace


def _selected_shadow_record(value: Any, *, pft_index: int, gpp_multiplier: float) -> Mapping[str, float]:
    potential = float(np.asarray(value, dtype=np.float64).reshape(-1)[pft_index])
    return {"potential": potential, "gpp": potential * float(gpp_multiplier)}


def _selected_trace_record(
    trace: Mapping[str, Any],
    *,
    pft_index: int,
    gpp_multiplier: float,
) -> Mapping[str, float]:
    return _selected_shadow_record(trace["potential"], pft_index=pft_index, gpp_multiplier=gpp_multiplier)


def _head_controls(
    potential_evaluator: Any,
    lower_blocks: Any,
    upper_blocks: Any,
    output_parameters: Any,
    branch_inputs: Mapping[str, Any],
    *,
    pft_index: int,
    gpp_multiplier: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    upper_only = {}
    upper_without = {}
    records = {}
    for head in range(ATTENTION_HEADS):
        name = f"head_{head}"
        lower_mixed = replace_block_attention_head(
            lower_blocks,
            upper_blocks,
            block_index=TARGET_BLOCK_INDEX,
            head=head,
            heads=ATTENTION_HEADS,
        )
        upper_mixed = replace_block_attention_head(
            upper_blocks,
            lower_blocks,
            block_index=TARGET_BLOCK_INDEX,
            head=head,
            heads=ATTENTION_HEADS,
        )
        lower_value = potential_evaluator(lower_mixed, output_parameters, branch_inputs)
        upper_value = potential_evaluator(upper_mixed, output_parameters, branch_inputs)
        lower_record = _selected_shadow_record(
            lower_value,
            pft_index=pft_index,
            gpp_multiplier=gpp_multiplier,
        )
        upper_record = _selected_shadow_record(
            upper_value,
            pft_index=pft_index,
            gpp_multiplier=gpp_multiplier,
        )
        upper_only[name] = lower_record["gpp"]
        upper_without[name] = upper_record["gpp"]
        records[name] = {
            "upper_head_only": lower_record,
            "upper_without_head": upper_record,
        }
        print(f"coupled_gpp head_token head_control={head + 1}/{ATTENTION_HEADS}", flush=True)
    return {"upper_only_gpp": upper_only, "upper_without_gpp": upper_without}, records


def _crossed_head7_controls(
    potential_evaluator: Any,
    lower_blocks: Any,
    upper_blocks: Any,
    output_parameters: Any,
    branch_inputs: Mapping[str, Any],
    *,
    lower_gpp: float,
    upper_gpp: float,
    pft_index: int,
    gpp_multiplier: float,
) -> Mapping[str, Any]:
    effect = float(upper_gpp) - float(lower_gpp)
    records = {}
    for name, projections in (
        ("query_only", ("query",)),
        ("key_only", ("key",)),
        ("query_and_key", ("query", "key")),
    ):
        lower_mixed = replace_block_attention_head(
            lower_blocks,
            upper_blocks,
            block_index=TARGET_BLOCK_INDEX,
            head=TARGET_HEAD_INDEX,
            heads=ATTENTION_HEADS,
            projections=projections,
        )
        upper_mixed = replace_block_attention_head(
            upper_blocks,
            lower_blocks,
            block_index=TARGET_BLOCK_INDEX,
            head=TARGET_HEAD_INDEX,
            heads=ATTENTION_HEADS,
            projections=projections,
        )
        lower_record = _selected_shadow_record(
            potential_evaluator(lower_mixed, output_parameters, branch_inputs),
            pft_index=pft_index,
            gpp_multiplier=gpp_multiplier,
        )
        upper_record = _selected_shadow_record(
            potential_evaluator(upper_mixed, output_parameters, branch_inputs),
            pft_index=pft_index,
            gpp_multiplier=gpp_multiplier,
        )
        residual = upper_record["gpp"] - float(lower_gpp)
        records[name] = {
            "projections": list(projections),
            "upper_slices_in_lower": lower_record,
            "lower_slices_in_upper": upper_record,
            "signed_sufficiency_fraction": (lower_record["gpp"] - float(lower_gpp)) / effect,
            "necessity_capture_fraction": 1.0 - abs(residual) / abs(effect),
            "residual_effect_without_slices": residual,
        }
    return records


def _global_masked_inputs(
    jnp: Any,
    branch_inputs: Mapping[str, Any],
    *,
    pft_index: int,
    token_indices: tuple[int, ...],
) -> tuple[Mapping[str, Any], int]:
    before = np.asarray(branch_inputs["bank_mask"], dtype=np.bool_)
    mask = branch_inputs["bank_mask"].at[pft_index, jnp.asarray(token_indices, dtype=jnp.int32)].set(False)
    after = np.asarray(mask, dtype=np.bool_)
    result = dict(branch_inputs)
    result["bank_mask"] = mask
    return result, int(np.count_nonzero(before ^ after))


def _head_keep_mask(
    jnp: Any,
    lower_blocks: Any,
    branch_inputs: Mapping[str, Any],
    *,
    pft_index: int,
    token_indices: tuple[int, ...],
) -> tuple[Any, int]:
    shape = (
        len(lower_blocks),
        int(branch_inputs["bank"].shape[0]),
        ATTENTION_HEADS,
        int(branch_inputs["bank"].shape[-2]),
    )
    mask = jnp.ones(shape, dtype=jnp.bool_)
    indices = jnp.asarray(token_indices, dtype=jnp.int32)
    changed = mask.at[TARGET_BLOCK_INDEX, pft_index, TARGET_HEAD_INDEX, indices].set(False)
    return changed, int(np.count_nonzero(np.asarray(mask) ^ np.asarray(changed)))


def _token_mask_controls(
    reconstruction: Any,
    masked_trace_evaluator: Any,
    lower_blocks: Any,
    upper_blocks: Any,
    output_parameters: Any,
    branch_inputs: Mapping[str, Any],
    token_indices: Mapping[str, int],
    *,
    lower_gpp: float,
    upper_gpp: float,
    pft_index: int,
    gpp_multiplier: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any], bool]:
    jax = reconstruction.jax
    jnp = reconstruction.jnp
    all_keep, _ = _head_keep_mask(
        jnp,
        lower_blocks,
        branch_inputs,
        pft_index=pft_index,
        token_indices=(),
    )
    definitions = (
        ("doc", (int(token_indices["doc"]),)),
        ("litterpart", (int(token_indices["litterpart"]),)),
        ("doc_and_litterpart", (int(token_indices["doc"]), int(token_indices["litterpart"]))),
    )
    raw = {}
    summarized_input = {}
    integrity = True
    for scope in ("global", "head7"):
        for token_name, indices in definitions:
            name = f"{scope}_remove_{token_name}"
            if scope == "global":
                scenario_inputs, changed = _global_masked_inputs(
                    jnp,
                    branch_inputs,
                    pft_index=pft_index,
                    token_indices=indices,
                )
                head_mask = all_keep
            else:
                scenario_inputs = branch_inputs
                head_mask, changed = _head_keep_mask(
                    jnp,
                    lower_blocks,
                    branch_inputs,
                    pft_index=pft_index,
                    token_indices=indices,
                )
            integrity &= changed == len(indices)
            lower_trace = masked_trace_evaluator(lower_blocks, output_parameters, scenario_inputs, head_mask)
            upper_trace = masked_trace_evaluator(upper_blocks, output_parameters, scenario_inputs, head_mask)
            jax.block_until_ready((lower_trace, upper_trace))
            lower_host = jax.device_get(lower_trace)
            upper_host = jax.device_get(upper_trace)
            lower_record = _selected_trace_record(
                lower_host,
                pft_index=pft_index,
                gpp_multiplier=gpp_multiplier,
            )
            upper_record = _selected_trace_record(
                upper_host,
                pft_index=pft_index,
                gpp_multiplier=gpp_multiplier,
            )
            summarized_input[name] = {
                "lower_gpp": lower_record["gpp"],
                "upper_gpp": upper_record["gpp"],
            }
            raw[name] = {
                "scope": "all_potential_heads_and_blocks" if scope == "global" else "block_0_head_7_only",
                "removed_tokens": list(indices),
                "changed_mask_entry_count": changed,
                "lower": lower_record,
                "upper": upper_record,
                "lower_head7": attention_token_summary(
                    lower_host,
                    pft_index=pft_index,
                    block_index=TARGET_BLOCK_INDEX,
                    head_index=TARGET_HEAD_INDEX,
                    tokens=token_indices,
                ),
                "upper_head7": attention_token_summary(
                    upper_host,
                    pft_index=pft_index,
                    block_index=TARGET_BLOCK_INDEX,
                    head_index=TARGET_HEAD_INDEX,
                    tokens=token_indices,
                ),
            }
            print(f"coupled_gpp head_token token_control={name}", flush=True)
    return (
        summarize_token_mask_controls(
            lower_gpp=lower_gpp,
            upper_gpp=upper_gpp,
            scenarios=summarized_input,
        ),
        raw,
        integrity,
    )


def _all_records_valid(records: Any) -> bool:
    values = []

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child, key)
        elif key in {"potential", "gpp"} and isinstance(value, (int, float, np.number)):
            values.append(float(value))

    visit(records)
    return bool(values and np.all(np.isfinite(values)) and np.all(np.asarray(values) >= 0.0))


def run(args: Any) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    parent, parent_digest = _load_parent_report(args.potential_internal_report.resolve())
    bootstrap = _initialization_only(args)
    runtime = bootstrap["context"]["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("head-token attribution requires JAX float64 support")
    runtime_year = _runtime_year_for_update(bootstrap["context"]["schedule"], TARGET_ARTIFACT_UPDATE)
    if runtime_year != EXPECTED_RUNTIME_YEAR:
        raise ValueError("head-token runtime-year drift")

    reconstruction = reconstruct_localized_jump_parameters(args, bootstrap=bootstrap)
    direction = _direction_integrity(reconstruction, parent)
    control_structured = coupled_gpp_trainable_tree(reconstruction.control_parameters)["structured_gpp"]
    treatment_structured = coupled_gpp_trainable_tree(reconstruction.treatment_parameters)["structured_gpp"]
    lower_blocks = _interpolate_tree(
        jax,
        jnp,
        control_structured["potential_blocks"],
        treatment_structured["potential_blocks"],
        LOWER_FRACTION,
        dtype=jnp.float32,
    )
    upper_blocks = _interpolate_tree(
        jax,
        jnp,
        control_structured["potential_blocks"],
        treatment_structured["potential_blocks"],
        UPPER_FRACTION,
        dtype=jnp.float32,
    )
    coverage = projection_head_delta_coverage(
        jax.device_get(lower_blocks),
        jax.device_get(upper_blocks),
        block_index=TARGET_BLOCK_INDEX,
        heads=ATTENTION_HEADS,
    )

    artifact = _load_artifact(args, reconstruction.context, TARGET_ARTIFACT_UPDATE)
    microbatch = _slice(artifact.batch.seven_day, TARGET_WINDOW_INDEX, TARGET_WINDOW_INDEX + 1)
    trace_evaluator = _make_trace_evaluator(bootstrap)
    control_trace, control_trajectories = trace_evaluator(reconstruction.control_parameters, microbatch)
    jax.block_until_ready((control_trace, control_trajectories))
    control_gpp_trace = np.asarray(control_trajectories.fluxes["gpp"].value, dtype=np.float64).reshape(-1)
    control_trace_reproduced = bool(
        np.allclose(
            control_gpp_trace,
            np.asarray(parent["control_gpp_trace"], dtype=np.float64),
            rtol=FORWARD_REPRODUCTION_RTOL,
            atol=FORWARD_REPRODUCTION_ATOL,
        )
    )
    templates, control_starts, control_discrete = _scenario_day_starts(
        reconstruction,
        microbatch,
        control_trace,
    )
    template_batch = _tree_day_batch(jax, templates, TARGET_DAY_INDEX)
    discrete_batch = _tree_day_batch(jax, control_discrete, TARGET_DAY_INDEX)
    control_state = jnp.asarray(control_starts[TARGET_DAY_INDEX])
    pft_index = EXPECTED_VEGETATION_PFT_INDEX
    factor_evaluator = _make_factor_evaluator(reconstruction)
    evaluate_kwargs = {
        "control_structured": control_structured,
        "reconstruction": reconstruction,
        "factor_evaluator": factor_evaluator,
        "template_batch": template_batch,
        "control_state": control_state,
        "discrete_batch": discrete_batch,
        "pft_index": pft_index,
    }
    lower_factor = _evaluate_blocks(lower_blocks, **evaluate_kwargs)
    upper_factor = _evaluate_blocks(upper_blocks, **evaluate_kwargs)
    lower_parent = _parent_endpoint(parent, LOWER_FRACTION)
    upper_parent = _parent_endpoint(parent, UPPER_FRACTION)
    endpoint_reproduced = bool(
        np.isclose(lower_factor["gpp"], lower_parent["gpp"], rtol=FACTORIZATION_RTOL, atol=FACTORIZATION_ATOL)
        and np.isclose(upper_factor["gpp"], upper_parent["gpp"], rtol=FACTORIZATION_RTOL, atol=FACTORIZATION_ATOL)
        and np.isclose(
            np.asarray(lower_factor["potential"], dtype=np.float64).reshape(-1)[pft_index],
            np.asarray(lower_parent["potential"], dtype=np.float64).reshape(-1)[pft_index],
            rtol=FACTORIZATION_RTOL,
            atol=FACTORIZATION_ATOL,
        )
        and np.isclose(
            np.asarray(upper_factor["potential"], dtype=np.float64).reshape(-1)[pft_index],
            np.asarray(upper_parent["potential"], dtype=np.float64).reshape(-1)[pft_index],
            rtol=FACTORIZATION_RTOL,
            atol=FACTORIZATION_ATOL,
        )
    )
    lower_potential = float(np.asarray(lower_factor["potential"], dtype=np.float64).reshape(-1)[pft_index])
    upper_potential = float(np.asarray(upper_factor["potential"], dtype=np.float64).reshape(-1)[pft_index])
    lower_multiplier = float(lower_factor["gpp"]) / lower_potential
    upper_multiplier = float(upper_factor["gpp"]) / upper_potential
    multiplier_reproduced = bool(
        np.isclose(lower_multiplier, upper_multiplier, rtol=FACTORIZATION_RTOL, atol=FACTORIZATION_ATOL)
    )
    gpp_multiplier = 0.5 * (lower_multiplier + upper_multiplier)

    input_evaluator = _make_potential_input_evaluator(reconstruction)
    batched_inputs = input_evaluator(
        reconstruction.control_parameters,
        template_batch,
        control_state[None, :],
        discrete_batch,
    )
    jax.block_until_ready(batched_inputs)
    branch_inputs = jax.tree_util.tree_map(lambda value: value[0], batched_inputs)
    potential_evaluator, trace_function = _make_shadow_evaluators(reconstruction, jnp.float32)
    lower_shadow = potential_evaluator(lower_blocks, control_structured["potential_output"], branch_inputs)
    upper_shadow = potential_evaluator(upper_blocks, control_structured["potential_output"], branch_inputs)
    lower_trace = trace_function(lower_blocks, control_structured["potential_output"], branch_inputs)
    upper_trace = trace_function(upper_blocks, control_structured["potential_output"], branch_inputs)
    jax.block_until_ready((lower_shadow, upper_shadow, lower_trace, upper_trace))
    lower_shadow_record = _selected_shadow_record(
        lower_shadow,
        pft_index=pft_index,
        gpp_multiplier=gpp_multiplier,
    )
    upper_shadow_record = _selected_shadow_record(
        upper_shadow,
        pft_index=pft_index,
        gpp_multiplier=gpp_multiplier,
    )
    shadow_reproduced = bool(
        np.isclose(lower_shadow_record["potential"], lower_potential, rtol=FACTORIZATION_RTOL, atol=FACTORIZATION_ATOL)
        and np.isclose(
            upper_shadow_record["potential"], upper_potential, rtol=FACTORIZATION_RTOL, atol=FACTORIZATION_ATOL
        )
        and np.isclose(
            lower_shadow_record["gpp"], lower_factor["gpp"], rtol=FACTORIZATION_RTOL, atol=FACTORIZATION_ATOL
        )
        and np.isclose(
            upper_shadow_record["gpp"], upper_factor["gpp"], rtol=FACTORIZATION_RTOL, atol=FACTORIZATION_ATOL
        )
    )

    forcing_token_count = 5 * len(FORCING_FIELDS)
    resolved_by_key = state_summary_bank_indices(
        runtime["contract"].state_leaves,
        forcing_token_count=forcing_token_count,
        keys=tuple(TARGET_TOKEN_KEYS.values()),
    )
    token_indices = {name: resolved_by_key[key] for name, key in TARGET_TOKEN_KEYS.items()}
    token_mapping_reproduced = bool(token_indices == {"doc": DOC_BANK_INDEX, "litterpart": LITTERPART_BANK_INDEX})
    target_tokens_valid = bool(
        np.all(
            np.asarray(branch_inputs["bank_mask"], dtype=np.bool_)[
                pft_index,
                np.asarray(tuple(token_indices.values()), dtype=np.int32),
            ]
        )
    )
    selected_tokens = {name: int(index) for name, index in token_indices.items()}
    lower_attention = attention_token_summary(
        jax.device_get(lower_trace),
        pft_index=pft_index,
        block_index=TARGET_BLOCK_INDEX,
        head_index=TARGET_HEAD_INDEX,
        tokens=selected_tokens,
    )
    upper_attention = attention_token_summary(
        jax.device_get(upper_trace),
        pft_index=pft_index,
        block_index=TARGET_BLOCK_INDEX,
        head_index=TARGET_HEAD_INDEX,
        tokens=selected_tokens,
    )
    parent_head = next(
        record
        for record in parent["adjacent_activation_traces"]["production_equivalent_float32"]["attention_heads"]
        if int(record["block"]) == TARGET_BLOCK_INDEX and int(record["head"]) == TARGET_HEAD_INDEX
    )
    base_attention_reproduced = bool(
        lower_attention["winner_bank_index"] == int(parent_head["lower"]["maximum_weight_bank_index"])
        and upper_attention["winner_bank_index"] == int(parent_head["upper"]["maximum_weight_bank_index"])
        and np.isclose(
            lower_attention["winner_logit"],
            parent_head["lower"]["logit_maximum"],
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        and np.isclose(
            upper_attention["winner_logit"],
            parent_head["upper"]["logit_maximum"],
            rtol=1.0e-6,
            atol=1.0e-3,
        )
    )

    head_inputs, head_records = _head_controls(
        potential_evaluator,
        lower_blocks,
        upper_blocks,
        control_structured["potential_output"],
        branch_inputs,
        pft_index=pft_index,
        gpp_multiplier=gpp_multiplier,
    )
    head_attribution = classify_head_controls(
        lower_gpp=lower_factor["gpp"],
        upper_gpp=upper_factor["gpp"],
        upper_head_only_gpp=head_inputs["upper_only_gpp"],
        upper_without_head_gpp=head_inputs["upper_without_gpp"],
    )
    head_attribution["factorizations"] = head_records
    crossed = _crossed_head7_controls(
        potential_evaluator,
        lower_blocks,
        upper_blocks,
        control_structured["potential_output"],
        branch_inputs,
        lower_gpp=lower_factor["gpp"],
        upper_gpp=upper_factor["gpp"],
        pft_index=pft_index,
        gpp_multiplier=gpp_multiplier,
    )
    masked_trace_evaluator = _make_masked_trace_evaluator(reconstruction)
    token_attribution, token_records, mask_integrity = _token_mask_controls(
        reconstruction,
        masked_trace_evaluator,
        lower_blocks,
        upper_blocks,
        control_structured["potential_output"],
        branch_inputs,
        token_indices,
        lower_gpp=lower_factor["gpp"],
        upper_gpp=upper_factor["gpp"],
        pft_index=pft_index,
        gpp_multiplier=gpp_multiplier,
    )
    token_attribution["factorizations"] = token_records

    head7_doc = token_attribution["scenarios"]["head7_remove_doc"]
    if (
        head_attribution["target_head_sufficient"]
        and head_attribution["target_head_necessary"]
        and head7_doc["upper_token_descriptively_necessary"]
    ):
        scientific_classification = "head7_qk_pair_and_doc_token_mediate_sampled_jump"
        next_gate = "discuss_qk_conditioning_controls_with_user_before_architecture_change"
    elif head_attribution["target_head_sufficient"] or head_attribution["target_head_necessary"]:
        scientific_classification = "head7_qk_pair_material_without_head7_doc_necessity"
        next_gate = "inspect_head7_crossed_and_alternative_token_controls"
    elif head_attribution["selected_groups"]:
        scientific_classification = "sampled_jump_uses_other_or_distributed_attention_heads"
        next_gate = "inspect_selected_head_controls_before_conditioning_discussion"
    else:
        scientific_classification = "sampled_jump_not_localized_to_one_qk_head_pair"
        next_gate = "retain_distributed_block0_mechanism_and_reassess_localization"

    direction_reproduced = bool(
        direction["combined_gradient_norm_reproduced"] and direction["treatment_parameter_displacement_reproduced"]
    )
    geometry_reproduced = bool(
        coverage["width"] == MODEL_WIDTH
        and coverage["head_count"] == ATTENTION_HEADS
        and coverage["head_width"] == HEAD_WIDTH
        and coverage["exact_nonoverlapping_coverage"]
    )
    records_valid = _all_records_valid(
        {
            "endpoints": (lower_factor, upper_factor, lower_shadow_record, upper_shadow_record),
            "heads": head_records,
            "crossed": crossed,
            "tokens": token_records,
        }
    )
    engineering_passed = bool(
        direction_reproduced
        and control_trace_reproduced
        and endpoint_reproduced
        and multiplier_reproduced
        and shadow_reproduced
        and geometry_reproduced
        and token_mapping_reproduced
        and target_tokens_valid
        and base_attention_reproduced
        and mask_integrity
        and records_valid
    )
    if not engineering_passed:
        scientific_classification = "engineering_gate_failed"
        next_gate = "inspect_written_engineering_report_before_any_rerun"

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if engineering_passed else "engineering_gate_failed",
        "classification": scientific_classification,
        "identity": {
            "git_head": _git_head(),
            "potential_internal_report_sha256": parent_digest,
        },
        "target": dict(parent["target"]),
        "direction_reconstruction": direction,
        "control_gpp_trace": control_gpp_trace.tolist(),
        "endpoints": {
            "lower_factorization": lower_factor,
            "upper_factorization": upper_factor,
            "lower_shadow": lower_shadow_record,
            "upper_shadow": upper_shadow_record,
            "lower_gpp_multiplier": lower_multiplier,
            "upper_gpp_multiplier": upper_multiplier,
            "applied_gpp_multiplier": gpp_multiplier,
        },
        "head_projection_coverage": coverage,
        "head_attribution": head_attribution,
        "head7_crossed_attribution": crossed,
        "token_layout": {
            "forcing_token_count": forcing_token_count,
            "resolved_by_key": resolved_by_key,
            "named_indices": token_indices,
        },
        "baseline_head7_attention": {
            "lower": lower_attention,
            "upper": upper_attention,
        },
        "token_mask_attribution": token_attribution,
        "integrity": {
            "engineering_gate_passed": engineering_passed,
            "direction_reproduced": direction_reproduced,
            "control_trace_reproduced": control_trace_reproduced,
            "endpoint_factorizations_reproduced": endpoint_reproduced,
            "nonpotential_gpp_multiplier_reproduced": multiplier_reproduced,
            "float32_shadow_endpoints_reproduced": shadow_reproduced,
            "head_geometry_and_coverage_exact": geometry_reproduced,
            "target_token_mapping_reproduced": token_mapping_reproduced,
            "target_tokens_valid_in_original_mask": target_tokens_valid,
            "baseline_head7_attention_reproduced": base_attention_reproduced,
            "token_mask_changes_exact": mask_integrity,
            "all_intervention_records_finite_and_nonnegative": records_valid,
            "factorization_tolerance": {"rtol": FACTORIZATION_RTOL, "atol": FACTORIZATION_ATOL},
        },
        "policy": _policy_record(),
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "memory_after": _memory_report(bootstrap["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": next_gate,
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _internal_parser()
    parser.add_argument("--potential-internal-report", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
