"""Audit exact-shape field grouping for the shared coordinate decoder.

The production decoder currently emits one Python-unrolled ``_decode_field``
body per schema field.  This audit asks how many distinct bodies remain when
fields with identical execution shapes are stacked behind an outer
``jax.lax.map`` while retaining field-local dynamic projection banks.

The static audit initializes no neural parameters and reads no Teacher data.
The optional JAX benchmark uses a small synthetic differentiable decoder to
compare Python-unrolled and mapped application of the same body; it is not a
full-model or GPU performance claim.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "outputs/shared-decoder-packing-audit-20260923.json"


class FieldSignature(NamedTuple):
    eligible_coordinates: int
    chunk_width: int
    chunk_count: int
    axis_slots: int
    axis_bank: int
    local_bank: int
    uses_carbon_bank: bool


def _field_signature(
    row: Mapping[str, Any],
    *,
    axis_bank: int,
    local_bank: int,
    chunk_size: int,
) -> FieldSignature:
    size = int(row["eligible_coordinates"])
    if size <= 0:
        return FieldSignature(size, 0, 0, 0, axis_bank, local_bank, bool(row["uses_carbon_bank"]))
    chunk_width = min(chunk_size, size)
    chunk_count = math.ceil(size / chunk_width)
    return FieldSignature(
        size,
        chunk_width,
        chunk_count,
        int(row["dynamic_tokens_per_query"]) - 1,
        axis_bank,
        local_bank,
        bool(row["uses_carbon_bank"]),
    )


def _signature_payload(signature: FieldSignature) -> dict[str, Any]:
    return {
        "eligible_coordinates": signature.eligible_coordinates,
        "chunk_width": signature.chunk_width,
        "chunk_count": signature.chunk_count,
        "axis_slots": signature.axis_slots,
        "axis_bank": signature.axis_bank,
        "local_bank": signature.local_bank,
        "uses_carbon_bank": signature.uses_carbon_bank,
    }


def static_grouping_audit(*, width: int = 768, blocks: int = 2, chunk_size: int = 64) -> dict[str, Any]:
    from research.daily_coarse_graining.daily_operator.shared_transformer import (
        _decoder_dynamic_plan,
    )
    from scripts.hpc.audit_shared_decoder_cost import _load_static_layout

    current = _load_static_layout()
    schema = current["schema"]
    layout = current["layout"]
    rows = current["fields"]
    plan = _decoder_dynamic_plan(schema=schema, layout=layout)
    if len(rows) != len(plan.field_axis_bank_indices) or len(rows) != len(plan.field_local_bank_indices):
        raise ValueError("field rows and dynamic projection plan differ")

    grouped: dict[FieldSignature, list[dict[str, Any]]] = defaultdict(list)
    axis_total = 0
    local_total = 0
    for field_index, row in enumerate(rows):
        axis_bank = int(plan.field_axis_bank_indices[field_index].size)
        local_bank = int(plan.field_local_bank_indices[field_index].size)
        axis_total += axis_bank
        local_total += local_bank
        signature = _field_signature(
            row,
            axis_bank=axis_bank,
            local_bank=local_bank,
            chunk_size=chunk_size,
        )
        grouped[signature].append(
            {
                "field_index": field_index,
                "field": str(row["field"]),
                "expert": str(row["expert"]),
                "expert_index": int(row["expert_index"]),
            }
        )

    groups = []
    maximum_raw_group_bytes = 0
    maximum_projected_iteration_bytes = 0
    for signature, members in sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), tuple(item[0])),
    ):
        raw_group_bytes = len(members) * (signature.axis_bank + signature.local_bank) * width * 4
        projected_iteration_bytes = (signature.axis_bank + signature.local_bank) * width * 4 * 2 * blocks
        maximum_raw_group_bytes = max(maximum_raw_group_bytes, raw_group_bytes)
        maximum_projected_iteration_bytes = max(
            maximum_projected_iteration_bytes,
            projected_iteration_bytes,
        )
        groups.append(
            {
                "signature": _signature_payload(signature),
                "field_count": len(members),
                "experts": sorted({member["expert"] for member in members}),
                "stacked_raw_dynamic_token_bytes": raw_group_bytes,
                "field_local_projected_kv_bytes_per_mapped_iteration": projected_iteration_bytes,
                "members": members,
            }
        )

    field_count = len(rows)
    group_count = len(groups)
    mapped_groups = sum(int(group["field_count"]) > 1 for group in groups)
    singleton_groups = group_count - mapped_groups
    mapped_fields = sum(int(group["field_count"]) for group in groups if int(group["field_count"]) > 1)
    family_count = len(layout.source_family_names)
    expert_count = len(layout.expert_names)
    small_expert_cache_adjoint_bytes = expert_count * (family_count + 1) * width * 2 * blocks * 4
    field_local_projection_macs = (axis_total + local_total) * blocks * 2 * width * width
    ffn_width = 4 * width
    readout_width = 256
    dense_kernel_gradient_elements = (
        blocks * (4 * width * width + 2 * width * ffn_width) + width * readout_width + readout_width
    )
    dense_kernel_gradient_bytes = dense_kernel_gradient_elements * 4

    carbon_groups = [group for group in groups if group["signature"]["uses_carbon_bank"]]
    if len(carbon_groups) != 1 or int(carbon_groups[0]["field_count"]) != 1:
        raise ValueError("raw-carbon GPP routing is not isolated by the grouping signature")

    return {
        "schema_version": "shared_decoder_exact_shape_grouping_audit_v1",
        "scope": "static_layout_and_optional_synthetic_jax_evidence_no_teacher_or_full_model_compile",
        "candidate": {
            "mechanism": (
                "Stack fields with identical decoder execution shapes and apply one field-local "
                "decoder body through an outer jax.lax.map per signature. Select the small "
                "expert family/process projection cache dynamically, but construct and project "
                "axis/local banks inside each mapped field iteration."
            ),
            "grouping_signature": list(FieldSignature._fields),
            "raw_carbon_route": "separate signature and singleton; daily.fast_gpp is never mixed with non-carbon fields",
            "output_restore": "retain each field's static eligible-coordinate scatter after grouped decoding",
            "parameter_tree_change": False,
            "precision_change": False,
            "custom_ad": False,
            "coordinate_padding_added": 0,
            "dynamic_bank_padding_added": 0,
        },
        "shape_evidence": {
            "fields": field_count,
            "exact_shape_programs": group_count,
            "program_reduction": field_count - group_count,
            "program_reduction_fraction": 1.0 - group_count / field_count,
            "multi_field_groups": mapped_groups,
            "singleton_groups": singleton_groups,
            "fields_in_multi_field_groups": mapped_fields,
            "largest_group_fields": max(int(group["field_count"]) for group in groups),
            "eligible_coordinates": sum(int(row["eligible_coordinates"]) for row in rows),
            "existing_field_local_padded_coordinates": sum(int(row["padded_coordinates"]) for row in rows),
            "candidate_field_local_padded_coordinates": sum(int(row["padded_coordinates"]) for row in rows),
            "field_local_axis_bank_entries": axis_total,
            "field_local_state_bank_entries": local_total,
            "field_local_dynamic_projection_macs": field_local_projection_macs,
            "global_axis_variants_not_gathered_per_field": int(plan.axis_expert_index.size),
            "global_local_variants_not_gathered_per_field": int(plan.local_state_indices.size),
            "maximum_stacked_raw_dynamic_token_bytes": maximum_raw_group_bytes,
            "maximum_field_local_projected_kv_bytes_per_mapped_iteration": maximum_projected_iteration_bytes,
            "small_expert_cache_adjoint_bytes_if_cross_expert_groups_are_mapped": small_expert_cache_adjoint_bytes,
            "dense_decoder_kernel_gradient_accumulator_bytes": dense_kernel_gradient_bytes,
            "python_unrolled_dense_kernel_partial_cotangent_bytes_upper_bound": (
                field_count * dense_kernel_gradient_bytes
            ),
            "exact_shape_group_dense_kernel_partial_cotangent_bytes_upper_bound": (
                group_count * dense_kernel_gradient_bytes
            ),
        },
        "groups": groups,
        "interpretation": [
            "This candidate targets Python/XLA graph replication and repeated shared-parameter transpose structure, not decoder forward MACs.",
            "Exact signatures retain current coordinate padding and field-local dynamic projection work; no all-coordinate activation batch is introduced.",
            "The only cross-field dynamic gather is from the small expert family/process cache. Axis/local K/V banks remain field-local, avoiding the prior global-bank scatter-adjoint failure mode.",
            "Dense-kernel cotangent byte counts are source-structure upper bounds if every unrolled partial were simultaneously materialized. They are not executable live-memory estimates; XLA can fuse, schedule, or reassociate additions.",
            "JAX may choose different lowering strategies for lax.map; full-model compile, executable-memory, numerical-gradient, and complete-update admission remain required before adoption.",
        ],
    }


def _normal(key: Any, shape: tuple[int, ...], scale: float) -> Any:
    import jax

    return scale * jax.random.normal(key, shape, dtype=np.float32)


def _synthetic_parameters(key: Any, *, width: int, ffn_width: int, blocks: int) -> Mapping[str, Any]:
    import jax

    keys = iter(jax.random.split(key, blocks * 6 + 2))
    result_blocks = []
    for _ in range(blocks):
        result_blocks.append(
            {
                "query": _normal(next(keys), (width, width), 0.1),
                "key": _normal(next(keys), (width, width), 0.1),
                "value": _normal(next(keys), (width, width), 0.1),
                "output": _normal(next(keys), (width, width), 0.1),
                "ffn_in": _normal(next(keys), (width, ffn_width), 0.08),
                "ffn_out": _normal(next(keys), (ffn_width, width), 0.08),
            }
        )
    return {
        "blocks": tuple(result_blocks),
        "readout_first": _normal(next(keys), (width, width), 0.1),
        "readout_second": _normal(next(keys), (width, 1), 0.1),
    }


def _layer_norm(value: Any) -> Any:
    import jax.numpy as jnp

    mean = jnp.mean(value, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(value - mean), axis=-1, keepdims=True)
    return (value - mean) * jax_lax_rsqrt(variance + np.float32(1.0e-6))


def jax_lax_rsqrt(value: Any) -> Any:
    import jax.lax as lax

    return lax.rsqrt(value)


def _synthetic_decoder_body(parameters: Mapping[str, Any], payload: tuple[Any, Any, Any]) -> Any:
    import jax
    import jax.numpy as jnp

    query, bank, bank_mask = payload
    width = int(query.shape[-1])
    for block in parameters["blocks"]:
        projected_query = query @ block["query"]
        key = bank @ block["key"]
        value = bank @ block["value"]
        logits = jnp.einsum("qd,kd->qk", projected_query, key) / np.float32(math.sqrt(width))
        logits = jnp.where(bank_mask[None, :], logits, -jnp.inf)
        weights = jax.nn.softmax(logits, axis=-1)
        message = (weights @ value) @ block["output"]
        query = _layer_norm(query + message)
        query = _layer_norm(query + jax.nn.gelu(query @ block["ffn_in"]) @ block["ffn_out"])
    return (jax.nn.gelu(query @ parameters["readout_first"]) @ parameters["readout_second"])[:, 0]


def _tree_block_until_ready(value: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        leaf.block_until_ready()


def _jaxpr_statistics(function: Any, *args: Any) -> dict[str, Any]:
    import jax

    closed = jax.make_jaxpr(function)(*args)
    primitives: Counter[str] = Counter()
    visited: set[int] = set()

    def visit(value: Any) -> None:
        identity = id(value)
        if identity in visited:
            return
        if hasattr(value, "jaxpr") and not hasattr(value, "eqns"):
            visited.add(identity)
            visit(value.jaxpr)
            return
        if hasattr(value, "eqns"):
            visited.add(identity)
            for equation in value.eqns:
                primitives[str(equation.primitive)] += 1
                visit(equation.params)
            return
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(closed)
    return {
        "recursive_equations": sum(primitives.values()),
        "primitive_counts": dict(sorted(primitives.items())),
    }


def _compile_and_time(function: Any, args: tuple[Any, ...], *, repeats: int) -> tuple[Any, dict[str, Any]]:
    import jax

    jitted = jax.jit(function)
    lowered = jitted.lower(*args)
    hlo_text = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - started
    memory = compiled.memory_analysis()
    memory_analysis = {
        name: int(getattr(memory, name))
        for name in (
            "argument_size_in_bytes",
            "temp_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "generated_code_size_in_bytes",
        )
        if getattr(memory, name, None) is not None
    }
    result = compiled(*args)
    _tree_block_until_ready(result)
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = compiled(*args)
        _tree_block_until_ready(result)
        elapsed.append((time.perf_counter() - started) * 1000.0)
    return result, {
        "compile_seconds": compile_seconds,
        "lowered_hlo_text_bytes": len(hlo_text.encode("utf-8")),
        "compiled_memory_analysis": memory_analysis,
        "steady_execution_milliseconds": elapsed,
        "steady_execution_median_milliseconds": statistics.median(elapsed),
    }


def synthetic_jax_benchmark(
    *,
    fields: int = 16,
    queries: int = 4,
    bank_tokens: int = 6,
    width: int = 16,
    blocks: int = 2,
    map_batch_size: int = 4,
    repeats: int = 7,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    if min(fields, queries, bank_tokens, width, blocks, map_batch_size, repeats) <= 0:
        raise ValueError("synthetic benchmark dimensions must be positive")
    key_parameters, key_queries, key_banks = jax.random.split(jax.random.PRNGKey(20260923), 3)
    parameters = _synthetic_parameters(
        key_parameters,
        width=width,
        ffn_width=4 * width,
        blocks=blocks,
    )
    query = jax.random.normal(key_queries, (fields, queries, width), dtype=jnp.float32)
    bank = jax.random.normal(key_banks, (fields, bank_tokens, width), dtype=jnp.float32)
    bank_mask = (jnp.arange(bank_tokens)[None, :] + jnp.arange(fields)[:, None]) % 5 != 0

    def unrolled_forward(candidate: Mapping[str, Any], candidate_query: Any, candidate_bank: Any) -> Any:
        return jnp.stack(
            tuple(
                _synthetic_decoder_body(
                    candidate,
                    (candidate_query[index], candidate_bank[index], bank_mask[index]),
                )
                for index in range(fields)
            ),
            axis=0,
        )

    def mapped_forward(candidate: Mapping[str, Any], candidate_query: Any, candidate_bank: Any) -> Any:
        return jax.lax.map(
            lambda payload: _synthetic_decoder_body(candidate, payload),
            (candidate_query, candidate_bank, bank_mask),
        )

    def batched_mapped_forward(candidate: Mapping[str, Any], candidate_query: Any, candidate_bank: Any) -> Any:
        return jax.lax.map(
            lambda payload: _synthetic_decoder_body(candidate, payload),
            (candidate_query, candidate_bank, bank_mask),
            batch_size=map_batch_size,
        )

    def objective(forward: Any, candidate: Mapping[str, Any], candidate_query: Any, candidate_bank: Any) -> Any:
        output = forward(candidate, candidate_query, candidate_bank)
        return jnp.mean(jnp.square(output)), output

    unrolled = jax.value_and_grad(
        lambda candidate, candidate_query, candidate_bank: objective(
            unrolled_forward,
            candidate,
            candidate_query,
            candidate_bank,
        ),
        argnums=(0, 1, 2),
        has_aux=True,
    )
    mapped = jax.value_and_grad(
        lambda candidate, candidate_query, candidate_bank: objective(
            mapped_forward,
            candidate,
            candidate_query,
            candidate_bank,
        ),
        argnums=(0, 1, 2),
        has_aux=True,
    )
    batched_mapped = jax.value_and_grad(
        lambda candidate, candidate_query, candidate_bank: objective(
            batched_mapped_forward,
            candidate,
            candidate_query,
            candidate_bank,
        ),
        argnums=(0, 1, 2),
        has_aux=True,
    )

    # Initialize the selected backend before any measured compilation.
    jnp.add(jnp.float32(1.0), jnp.float32(2.0)).block_until_ready()
    benchmark_args = (parameters, query, bank)
    mapped_result, mapped_measurement = _compile_and_time(mapped, benchmark_args, repeats=repeats)
    batched_mapped_result, batched_mapped_measurement = _compile_and_time(
        batched_mapped,
        benchmark_args,
        repeats=repeats,
    )
    unrolled_result, unrolled_measurement = _compile_and_time(unrolled, benchmark_args, repeats=repeats)

    (
        (unrolled_loss, unrolled_output),
        (
            unrolled_parameter_gradient,
            unrolled_query_gradient,
            unrolled_bank_gradient,
        ),
    ) = unrolled_result
    gradient_references = [np.asarray(value) for value in jax.tree_util.tree_leaves(unrolled_parameter_gradient)]
    gradient_reference_max = max(float(np.max(np.abs(value))) for value in gradient_references)

    def comparison_metrics(
        candidates: Iterable[Any],
        references: Iterable[Any],
        *,
        rtol: float,
        atol: float,
    ) -> dict[str, Any]:
        failed = 0
        coordinates = 0
        maximum_absolute = 0.0
        difference_l2_squared = 0.0
        reference_l2_squared = 0.0
        for candidate_value, reference_value in zip(candidates, references, strict=True):
            candidate_array = np.asarray(candidate_value)
            reference_array = np.asarray(reference_value)
            difference = candidate_array - reference_array
            finite_close = np.isclose(candidate_array, reference_array, rtol=rtol, atol=atol)
            failed += int(np.size(finite_close) - np.count_nonzero(finite_close))
            coordinates += int(np.size(finite_close))
            if difference.size:
                maximum_absolute = max(maximum_absolute, float(np.max(np.abs(difference))))
                difference_l2_squared += float(np.vdot(difference.ravel(), difference.ravel()))
                reference_l2_squared += float(np.vdot(reference_array.ravel(), reference_array.ravel()))
        return {
            "status": "passed" if failed == 0 else "failed",
            "coordinates": coordinates,
            "failed_coordinates": failed,
            "max_abs": maximum_absolute,
            "relative_l2": math.sqrt(difference_l2_squared) / max(math.sqrt(reference_l2_squared), 1.0e-30),
            "rtol": rtol,
            "atol": atol,
        }

    def parity_against_unrolled(result: Any) -> dict[str, Any]:
        (
            (candidate_loss, candidate_output),
            (
                candidate_parameter_gradient,
                candidate_query_gradient,
                candidate_bank_gradient,
            ),
        ) = result
        forward = comparison_metrics(
            (candidate_loss, candidate_output),
            (unrolled_loss, unrolled_output),
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        parameter_gradient = comparison_metrics(
            jax.tree_util.tree_leaves(candidate_parameter_gradient),
            jax.tree_util.tree_leaves(unrolled_parameter_gradient),
            rtol=5.0e-4,
            atol=2.0e-5,
        )
        query_gradient = comparison_metrics(
            (candidate_query_gradient,),
            (unrolled_query_gradient,),
            rtol=5.0e-4,
            atol=2.0e-5,
        )
        bank_gradient = comparison_metrics(
            (candidate_bank_gradient,),
            (unrolled_bank_gradient,),
            rtol=5.0e-4,
            atol=2.0e-5,
        )
        parameter_differences = [
            np.asarray(left - right)
            for left, right in zip(
                jax.tree_util.tree_leaves(candidate_parameter_gradient),
                jax.tree_util.tree_leaves(unrolled_parameter_gradient),
                strict=True,
            )
        ]
        parameter_max_abs = max(float(np.max(np.abs(value))) for value in parameter_differences)
        passed = all(
            metrics["status"] == "passed" for metrics in (forward, parameter_gradient, query_gradient, bank_gradient)
        )
        return {
            "status": "passed" if passed else "failed",
            "forward_comparison": forward,
            "parameter_gradient_comparison": parameter_gradient,
            "query_payload_gradient_comparison": query_gradient,
            "bank_payload_gradient_comparison": bank_gradient,
            "loss": float(candidate_loss),
            "loss_abs_difference": abs(float(candidate_loss) - float(unrolled_loss)),
            "output_max_abs": float(np.max(np.abs(np.asarray(candidate_output - unrolled_output)))),
            "parameter_gradient_max_abs": parameter_max_abs,
            "parameter_gradient_relative_to_reference_max": parameter_max_abs / max(gradient_reference_max, 1.0e-30),
            "query_payload_gradient_max_abs": float(
                np.max(np.abs(np.asarray(candidate_query_gradient - unrolled_query_gradient)))
            ),
            "bank_payload_gradient_max_abs": float(
                np.max(np.abs(np.asarray(candidate_bank_gradient - unrolled_bank_gradient)))
            ),
        }

    scalar_parity = parity_against_unrolled(mapped_result)
    batched_parity = parity_against_unrolled(batched_mapped_result)
    parity_status = (
        "passed" if scalar_parity["status"] == "passed" and batched_parity["status"] == "passed" else "failed"
    )

    return {
        "scope": "small_synthetic_shared_weight_decoder_bounded_backend",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "configuration": {
            "fields": fields,
            "queries_per_field": queries,
            "bank_tokens_per_field": bank_tokens,
            "width": width,
            "ffn_width": 4 * width,
            "blocks": blocks,
            "map_batch_size": map_batch_size,
            "dtype": "float32",
            "timing_repeats": repeats,
        },
        "parity": {
            "status": parity_status,
            "forward_tolerance": {"rtol": 2.0e-5, "atol": 2.0e-6},
            "gradient_tolerance": {"rtol": 5.0e-4, "atol": 2.0e-5},
            "unrolled_loss": float(unrolled_loss),
            "parameter_gradient_reference_max_abs": gradient_reference_max,
            "scalar_map": scalar_parity,
            "batched_map": batched_parity,
        },
        "scalar_map": {
            "jaxpr": _jaxpr_statistics(mapped, *benchmark_args),
            **mapped_measurement,
        },
        "batched_map": {
            "jaxpr": _jaxpr_statistics(batched_mapped, *benchmark_args),
            **batched_mapped_measurement,
        },
        "python_unrolled": {
            "jaxpr": _jaxpr_statistics(unrolled, *benchmark_args),
            **unrolled_measurement,
        },
        "limits": [
            "This benchmark isolates repeated shared-weight decoder application; it omits the production inner chunk map and exact normalized segmented attention and is only a structural proxy.",
            "Small width CPU timings do not predict the full width-768 GPU model; use configurable production-width dimensions only for bounded device evidence.",
            "The production candidate needs one mapped program per exact shape rather than one map across all 161 fields.",
            "The benchmark does not include raw-carbon attention, field-coordinate construction, retained physics, optimizer state, or recurrent rollout.",
        ],
    }


def _run_self_test() -> None:
    row = {
        "eligible_coordinates": 66,
        "dynamic_tokens_per_query": 3,
        "uses_carbon_bank": False,
    }
    signature = _field_signature(row, axis_bank=17, local_bank=66, chunk_size=64)
    assert signature == FieldSignature(66, 64, 2, 2, 17, 66, False)
    assert _signature_payload(signature)["chunk_count"] == 2


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--jax-benchmark", action="store_true")
    result.add_argument("--synthetic-only", action="store_true")
    result.add_argument("--self-test", action="store_true")
    result.add_argument("--timing-repeats", type=int, default=7)
    result.add_argument("--synthetic-fields", type=int, default=16)
    result.add_argument("--synthetic-queries", type=int, default=4)
    result.add_argument("--synthetic-bank-tokens", type=int, default=6)
    result.add_argument("--synthetic-width", type=int, default=16)
    result.add_argument("--synthetic-blocks", type=int, default=2)
    result.add_argument("--map-batch-size", type=int, default=4)
    return result


def main() -> int:
    args = parser().parse_args()
    _run_self_test()
    if args.self_test:
        print(json.dumps({"status": "passed", "scope": "pure_packing_signature_self_test"}))
        return 0
    if args.synthetic_only:
        payload = {
            "schema_version": "shared_decoder_synthetic_grouping_probe_v1",
            "scope": "synthetic_only_no_layout_fixture_teacher_or_full_model_compile",
        }
    else:
        payload = static_grouping_audit()
    if args.jax_benchmark or args.synthetic_only:
        payload["synthetic_jax_benchmark"] = synthetic_jax_benchmark(
            fields=args.synthetic_fields,
            queries=args.synthetic_queries,
            bank_tokens=args.synthetic_bank_tokens,
            width=args.synthetic_width,
            blocks=args.synthetic_blocks,
            map_batch_size=args.map_batch_size,
            repeats=args.timing_repeats,
        )
    benchmark_rejected = (
        "synthetic_jax_benchmark" in payload and payload["synthetic_jax_benchmark"]["parity"]["status"] != "passed"
    )
    payload["status"] = "rejected" if benchmark_rejected else "passed"
    _write_json(args.output, payload)
    summary = {
        "status": payload["status"],
        "output": str(args.output),
    }
    if "shape_evidence" in payload:
        summary["shape_evidence"] = payload["shape_evidence"]
    if "synthetic_jax_benchmark" in payload:
        benchmark = payload["synthetic_jax_benchmark"]
        summary["synthetic_jax_benchmark"] = {
            "parity": benchmark["parity"],
            "scalar_map": benchmark["scalar_map"],
            "batched_map": benchmark["batched_map"],
            "python_unrolled": benchmark["python_unrolled"],
        }
    print(json.dumps(summary, indent=2))
    return 2 if benchmark_rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
