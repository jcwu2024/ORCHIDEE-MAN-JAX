"""Audit reverse-AD shapes for global versus field-local decoder K/V reuse.

The default audit initializes no neural weights, reads no Teacher data, and
compiles no executable.  It combines canonical layout metadata with a bounded
abstract JAXPR at the production decoder width.  ``--compile`` additionally
compiles the bounded synthetic functions and reports compiler memory analysis;
that result is not a full-model or GPU-peak measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "outputs/shared-projection-ad-audit-20260923.json"


def canonical_surface(*, width: int, blocks: int, microbatch: int) -> dict[str, Any]:
    """Return source-visible dense gather-transpose surfaces."""

    from research.daily_coarse_graining.daily_operator.shared_transformer import (
        _decoder_dynamic_plan,
    )
    from scripts.hpc.audit_shared_decoder_cost import _load_static_layout

    loaded = _load_static_layout()
    plan = _decoder_dynamic_plan(schema=loaded["schema"], layout=loaded["layout"])
    fields = len(loaded["schema"].fields)
    axis_global = int(plan.axis_expert_index.size)
    local_global = int(plan.local_state_indices.size)
    axis_field_local = sum(int(indices.size) for indices in plan.field_axis_bank_indices)
    local_field_local = sum(int(indices.size) for indices in plan.field_local_bank_indices)
    axis_occurrences = sum(int(indices.size) for indices in plan.field_axis_indices)
    local_occurrences = sum(int(indices.size) for indices in plan.field_local_indices)
    projection_streams = blocks * 2  # coordinate blocks times key/value
    itemsize = np.dtype(np.float32).itemsize
    return {
        "fields": fields,
        "width": width,
        "microbatch": microbatch,
        "coordinate_blocks": blocks,
        "global_variants": {
            "axis": axis_global,
            "local": local_global,
            "total": axis_global + local_global,
        },
        "summed_field_local_variants": {
            "axis": axis_field_local,
            "local": local_field_local,
            "total": axis_field_local + local_field_local,
        },
        "coordinate_token_occurrences": {
            "axis": axis_occurrences,
            "local": local_occurrences,
            "total": axis_occurrences + local_occurrences,
        },
        "summed_dense_scan_cotangent_surface_bytes": {
            "global_projected_banks": (
                fields * projection_streams * microbatch * (axis_global + local_global) * width * itemsize
            ),
            "field_local_projected_banks": (
                projection_streams * microbatch * (axis_field_local + local_field_local) * width * itemsize
            ),
        },
        "interpretation": (
            "Summed source-visible transpose outputs, not simultaneous live bytes; "
            "compiler scheduling, buffer reuse, and fusion determine actual peak allocation."
        ),
    }


def _bounded_functions(
    jax: Any,
    jnp: Any,
    *,
    width: int,
    heads: int,
    fields: int,
    chunks: int,
    microbatch: int,
    query_count: int,
    dynamic_tokens: int,
    global_bank: int,
    field_bank: int,
) -> tuple[Any, Any, dict[str, tuple[Any, ...]]]:
    """Build synthetic functions with the decoder's projection/map boundaries."""

    head_width = width // heads
    local_indices = (
        np.arange(chunks * query_count * dynamic_tokens, dtype=np.int32).reshape(chunks, query_count, dynamic_tokens)
        % field_bank
    )
    field_banks = tuple(
        (np.arange(field_bank, dtype=np.int32) + field_index * field_bank) % global_bank
        for field_index in range(fields)
    )

    def normalize(value):
        shaped = value.reshape(value.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
        maximum = jnp.max(jnp.abs(shaped), axis=-1, keepdims=True)
        scaled = shaped / jnp.where(maximum > 0.0, maximum, 1.0)
        norm = jnp.sqrt(jnp.sum(scaled * scaled, axis=-1, keepdims=True) + 1.0e-12)
        return jnp.where(maximum * norm > 1.0e-6, scaled / norm, 0.0)

    def attend(query, key, value):
        head_query = normalize(query[:, None, :])
        head_key = normalize(key)
        head_value = value.reshape(value.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
        weights = jax.nn.softmax(
            jnp.einsum("...hqd,...hkd->...hqk", head_query, head_key),
            axis=-1,
        )
        return jnp.einsum("...hqk,...hkd->...hqd", weights, head_value).sum()

    def mapped(projected_key, projected_value, queries, indices):
        def chunk(values):
            query, index = values
            return attend(query, projected_key[index], projected_value[index])

        return jax.lax.map(chunk, (queries, indices)).sum()

    def global_day(parameters, tokens, queries):
        projected_key = tokens @ parameters["key"]
        projected_value = tokens @ parameters["value"]
        result = 0.0
        for field_index, bank in enumerate(field_banks):
            indices = jnp.asarray(bank, dtype=jnp.int32)[jnp.asarray(local_indices)]
            result += mapped(projected_key, projected_value, queries[field_index], indices)
        return result

    def field_local_day(parameters, tokens, queries):
        result = 0.0
        for field_index in range(fields):
            field_tokens = tokens[field_index]
            projected_key = field_tokens @ parameters["key"]
            projected_value = field_tokens @ parameters["value"]
            result += mapped(
                projected_key,
                projected_value,
                queries[field_index],
                jnp.asarray(local_indices),
            )
        return result

    def global_objective(parameters, tokens, queries):
        return jnp.mean(jax.vmap(global_day, in_axes=(None, 0, 0))(parameters, tokens, queries))

    def field_local_objective(parameters, tokens, queries):
        return jnp.mean(jax.vmap(field_local_day, in_axes=(None, 0, 0))(parameters, tokens, queries))

    parameter_abstract = {
        "key": jax.ShapeDtypeStruct((width, width), jnp.float32),
        "value": jax.ShapeDtypeStruct((width, width), jnp.float32),
    }
    query_abstract = jax.ShapeDtypeStruct((microbatch, fields, chunks, query_count, width), jnp.float32)
    abstracts = {
        "global": (
            parameter_abstract,
            jax.ShapeDtypeStruct((microbatch, global_bank, width), jnp.float32),
            query_abstract,
        ),
        "field_local": (
            parameter_abstract,
            jax.ShapeDtypeStruct((microbatch, fields, field_bank, width), jnp.float32),
            query_abstract,
        ),
    }
    return global_objective, field_local_objective, abstracts


def _scan_rows(closed_jaxpr: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def walk(jaxpr):
        for equation in jaxpr.eqns:
            if equation.primitive.name == "scan":
                rows.append(
                    {
                        "reverse": bool(equation.params["reverse"]),
                        "num_consts": int(equation.params["num_consts"]),
                        "num_carry": int(equation.params["num_carry"]),
                        "output_shapes": [list(variable.aval.shape) for variable in equation.outvars],
                    }
                )
            for value in equation.params.values():
                nested = getattr(value, "jaxpr", None)
                if nested is not None:
                    walk(nested)

    walk(closed_jaxpr.jaxpr)
    return rows


def bounded_audit(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    global_objective, local_objective, abstracts = _bounded_functions(
        jax,
        jnp,
        width=args.width,
        heads=args.heads,
        fields=args.bounded_fields,
        chunks=args.chunks,
        microbatch=args.microbatch,
        query_count=args.query_count,
        dynamic_tokens=args.dynamic_tokens,
        global_bank=args.global_bank,
        field_bank=args.field_bank,
    )
    result: dict[str, Any] = {
        "jax_version": jax.__version__,
        "shape": {
            "width": args.width,
            "heads": args.heads,
            "coordinate_blocks_represented": 1,
            "fields": args.bounded_fields,
            "chunks_per_field": args.chunks,
            "microbatch": args.microbatch,
            "query_count_per_chunk": args.query_count,
            "dynamic_tokens_per_query": args.dynamic_tokens,
            "global_bank": args.global_bank,
            "field_bank": args.field_bank,
        },
        "boundaries": {
            "global": "project global bank, then call one chunk lax.map per field",
            "field_local": "construct/project compact bank per field, then call its chunk lax.map",
        },
    }
    differentiated = {}
    for name, objective in (("global", global_objective), ("field_local", local_objective)):
        abstract = abstracts[name]
        function = jax.value_and_grad(objective, argnums=(0, 1))
        closed = jax.make_jaxpr(function)(*abstract)
        rows = _scan_rows(closed)
        target_bank = args.global_bank if name == "global" else args.field_bank
        target_shape = [args.microbatch, target_bank, args.width]
        dense_outputs = sum(shape == target_shape for row in rows if row["reverse"] for shape in row["output_shapes"])
        reverse_rows = [row for row in rows if row["reverse"]]
        differentiated[name] = {
            "scan_count": len(rows),
            "forward_scan_count": len(rows) - len(reverse_rows),
            "reverse_scan_count": len(reverse_rows),
            "dense_projected_bank_cotangent_outputs": dense_outputs,
            "dense_projected_bank_cotangent_bytes": (
                dense_outputs * args.microbatch * target_bank * args.width * np.dtype(np.float32).itemsize
            ),
            "reverse_scans": reverse_rows,
        }
        if args.compile:
            started = time.perf_counter()
            compiled = jax.jit(function).lower(*abstract).compile()
            analysis = compiled.memory_analysis()
            differentiated[name]["compile_seconds"] = time.perf_counter() - started
            differentiated[name]["compiler_memory_analysis"] = {
                key: int(value)
                for key in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "alias_size_in_bytes",
                    "temp_size_in_bytes",
                    "host_argument_size_in_bytes",
                    "host_output_size_in_bytes",
                    "host_alias_size_in_bytes",
                    "host_temp_size_in_bytes",
                )
                if (value := getattr(analysis, key, None)) is not None
            }
    result["differentiated"] = differentiated
    result["compiler_memory_scope"] = (
        "bounded synthetic executable, not full-model memory or device peak" if args.compile else "not compiled"
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--width", type=int, default=768)
    result.add_argument("--heads", type=int, default=8)
    result.add_argument("--blocks", type=int, default=2)
    result.add_argument("--bounded-fields", type=int, default=4)
    result.add_argument("--chunks", type=int, default=2)
    result.add_argument("--microbatch", type=int, default=4)
    result.add_argument("--query-count", type=int, default=64)
    result.add_argument("--dynamic-tokens", type=int, default=3)
    result.add_argument("--global-bank", type=int, default=2_137)
    result.add_argument("--field-bank", type=int, default=32)
    result.add_argument("--compile", action="store_true")
    result.add_argument(
        "--bounded-only",
        action="store_true",
        help="Skip canonical metadata fixtures; use the already recorded local layout audit.",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if args.width <= 0 or args.width % args.heads:
        raise ValueError("width must be positive and divide heads")
    report = {
        "status": "passed",
        "canonical": None
        if args.bounded_only
        else canonical_surface(
            width=args.width,
            blocks=args.blocks,
            microbatch=args.microbatch,
        ),
        "bounded_reverse_jaxpr": bounded_audit(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
