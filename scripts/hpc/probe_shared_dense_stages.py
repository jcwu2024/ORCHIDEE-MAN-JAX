"""Benchmark actual coordinate-decoder helpers with block-major dense packing.

No Teacher/checkpoint reads, tokenizer, retained physics, or optimizer updates.
Synthetic inputs use production parameter initialization, normalized attention,
field-local differentiable K/V projections, FFNs and readout. This component
comparison cannot establish full-model scientific or performance acceptance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.daily_coarse_graining.daily_operator.shared_transformer import (  # noqa: E402
    SharedTransformerConfig,
    _DecoderBlockProjection,
    _field_dynamic_projection_cache,
    _project_key_value,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (  # noqa: E402
    _linear_parameters,
    _residual_block_parameters,
)


def load_decoder(path):
    name = "research.daily_coarse_graining.daily_operator.shared_decoder_dense"
    if path is None:
        return __import__(name, fromlist=["DenseDecoderField"])
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fixture(*, width, blocks, microbatch, field_count, queries):
    keys = iter(jax.random.split(jax.random.PRNGKey(20260923), blocks + 2))
    block_parameters = []
    for block_id in range(blocks):
        block = _residual_block_parameters(next(keys), width, 4 * width)
        # Nonzero biases expose inactive-query padding/projection semantics.
        for name in ("key", "value"):
            block["attention"][name]["bias"] = jnp.linspace(0.01, 0.02, width, dtype=jnp.float32) * (block_id + 1)
        block_parameters.append(block)
    shared = {
        "coordinate_blocks": tuple(block_parameters),
        "readout_first": _linear_parameters(next(keys), width, min(width, 256)),
        "readout_second": _linear_parameters(next(keys), min(width, 256), 1),
    }
    rng = np.random.default_rng(20260923)
    payload, metadata = [], []
    for field_id in range(field_count):
        count = max(2, queries - (field_id % 3) * 3)
        axes = 1 + field_id % 3
        axis_bank, local_bank = 9 + field_id % 3, 13 + field_id % 4
        has_carbon = field_id == 0
        specs = {
            "query": (count, 1, width),
            "family": (20, width),
            "process": (1, width),
            "axis": (axis_bank, width),
            "local": (local_bank, width),
        }
        if has_carbon:
            specs["carbon"] = (24, width)
        payload.append(
            {
                key: jnp.asarray(rng.normal(size=(microbatch,) + shape), dtype=jnp.float32)
                for key, shape in specs.items()
            }
        )
        active = np.ones(count, dtype=bool)
        active[-1] = False
        axis_indices = np.arange(count * axes).reshape(count, axes) % axis_bank
        local_indices = np.arange(count) % local_bank
        dynamic_mask = (np.arange(count * (axes + 1)).reshape(count, 1, axes + 1) + field_id) % 5 != 0
        family_mask = (np.arange(count * 20).reshape(count, 1, 20) + field_id) % 4 != 0
        dynamic_mask &= active[:, None, None]
        family_mask &= active[:, None, None]
        carbon_mask = None
        if has_carbon:
            owners = np.arange(24) % 3 - 1
            carbon_mask = (owners[None, None, :] == -1) | (
                owners[None, None, :] == (np.arange(count) % 2)[:, None, None]
            )
            carbon_mask &= active[:, None, None]
        metadata.append(
            {
                "count": count,
                "axis_indices": jnp.asarray(axis_indices, dtype=jnp.int32),
                "local_indices": jnp.asarray(local_indices, dtype=jnp.int32),
                "active": jnp.asarray(active),
                "dynamic_mask": jnp.asarray(dynamic_mask),
                "family_mask": jnp.asarray(family_mask),
                "carbon_mask": None if carbon_mask is None else jnp.asarray(carbon_mask),
            }
        )
    return shared, tuple(payload), tuple(metadata)


def prepare_fields(decoder, shared, payload, metadata):
    fields = []
    for data, meta in zip(payload, metadata, strict=True):
        projected = []
        for block in shared["coordinate_blocks"]:
            family_key, family_value = _project_key_value(block["attention"], data["family"])
            process_key, process_value = _project_key_value(block["attention"], data["process"])
            carbon_key = carbon_value = None
            if "carbon" in data:
                carbon_key, carbon_value = _project_key_value(block["attention"], data["carbon"])
            projected.append(
                _DecoderBlockProjection(
                    family_key,
                    family_value,
                    process_key,
                    process_value,
                    carbon_key,
                    carbon_value,
                    None,
                    None,
                    None,
                    None,
                )
            )
        projected = _field_dynamic_projection_cache(
            {"shared_transformer": shared},
            tuple(projected),
            (data["axis"], data["local"]),
        )
        dynamic = jnp.concatenate(
            (data["axis"][meta["axis_indices"]], data["local"][meta["local_indices"], None, :]), axis=1
        )
        dynamic = jnp.where(meta["active"][:, None, None], dynamic, 0.0)
        fields.append(
            decoder.DenseDecoderField(
                query=data["query"],
                dynamic_tokens=dynamic,
                dynamic_mask=meta["dynamic_mask"],
                family_mask=meta["family_mask"],
                process_mask=meta["active"][:, None, None],
                carbon_mask=meta["carbon_mask"],
                projections=projected,
                dynamic_projection_indices=(meta["axis_indices"], meta["local_indices"], meta["active"]),
                output_active=meta["active"],
            )
        )
    return tuple(fields)


def compare_tree(candidate, reference, *, rtol, atol):
    failed = coordinates = 0
    maximum = diff_square = ref_square = 0.0
    for new, old in zip(jax.tree_util.tree_leaves(candidate), jax.tree_util.tree_leaves(reference), strict=True):
        new, old = np.asarray(new), np.asarray(old)
        failed += int(np.count_nonzero(~np.isclose(new, old, rtol=rtol, atol=atol)))
        coordinates += old.size
        diff = new.astype(np.float64) - old.astype(np.float64)
        maximum = max(maximum, float(np.max(np.abs(diff), initial=0)))
        diff_square += float(np.sum(diff * diff))
        ref_square += float(np.sum(old.astype(np.float64) ** 2))
    return {
        "passed": failed == 0,
        "coordinates": coordinates,
        "failed_coordinates": failed,
        "max_abs": maximum,
        "relative_l2": float(np.sqrt(diff_square / max(ref_square, 1e-60))),
        "rtol": rtol,
        "atol": atol,
    }


def run(args):
    decoder = load_decoder(args.decoder_module_file)
    shared, payload, metadata = fixture(
        width=args.width, blocks=args.blocks, microbatch=args.microbatch, field_count=args.fields, queries=args.queries
    )
    config = SharedTransformerConfig(
        width=args.width, ffn_width=4 * args.width, coordinate_blocks=args.blocks, readout_width=min(args.width, 256)
    )
    report = {
        "status": "running",
        "scope": __doc__,
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "configuration": {k: getattr(args, k) for k in ("width", "blocks", "microbatch", "fields", "queries")},
        "actual_queries_per_field": [m["count"] for m in metadata],
        "variants": {},
        "retained_updates": 0,
        "scientific_acceptance": False,
    }

    def write(stage):
        report["stage"] = stage
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(stage, flush=True)

    reference = None
    for mode in ("reference",) + tuple(f"packed{n}" for n in args.dense_chunks):
        chunk_size = None if mode == "reference" else int(mode.removeprefix("packed"))

        def one(parameters, one_payload):
            fields = prepare_fields(decoder, parameters, one_payload, metadata)
            if mode == "reference":
                outputs = decoder.reference_decode_fields(parameters, fields, config)
            else:
                outputs = decoder.packed_decode_fields(parameters, fields, config, dense_chunk_size=chunk_size)
            output = jnp.concatenate(outputs)
            target = jnp.sin(jnp.arange(output.size, dtype=jnp.float32) * 0.11)
            return jnp.mean((output - target) ** 2), output

        def objective(parameters, all_payload):
            losses, outputs = jax.vmap(one, in_axes=(None, 0))(parameters, all_payload)
            return jnp.mean(losses), outputs

        function = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))
        write(f"{mode}_lower_compile")
        started = time.perf_counter()
        lowered = function.lower(shared, payload)
        lower_seconds = time.perf_counter() - started
        started = time.perf_counter()
        executable = lowered.compile()
        memory = executable.memory_analysis()
        row = {
            "lower_seconds": lower_seconds,
            "compile_or_cache_load_seconds": time.perf_counter() - started,
            "compiled_memory": {
                name: int(getattr(memory, name))
                for name in (
                    "argument_size_in_bytes",
                    "temp_size_in_bytes",
                    "output_size_in_bytes",
                    "alias_size_in_bytes",
                )
            },
        }
        report["variants"][mode] = row
        if memory.temp_size_in_bytes + memory.argument_size_in_bytes + memory.output_size_in_bytes > 12 * 2**30:
            row["status"] = "resource_limited"
            write(f"{mode}_resource_limited")
            continue
        result = jax.block_until_ready(executable(shared, payload))
        times = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = jax.block_until_ready(executable(shared, payload))
            times.append(time.perf_counter() - started)
        row.update(
            warm_seconds=times,
            warm_median_seconds=float(np.median(times)),
            process_memory=jax.devices()[0].memory_stats(),
        )
        result = jax.device_get(result)
        if mode == "reference":
            reference = result
        row["comparison"] = {
            "forward": compare_tree(result[0], reference[0], rtol=2e-5, atol=2e-6),
            "parameters": compare_tree(result[1][0], reference[1][0], rtol=5e-4, atol=2e-5),
            "payload": compare_tree(result[1][1], reference[1][1], rtol=5e-4, atol=2e-5),
        }
        row["status"] = "passed" if all(v["passed"] for v in row["comparison"].values()) else "rejected"
        write(f"{mode}_completed")
        del result, executable, lowered, function
    report["status"] = "passed" if all(v["status"] == "passed" for v in report["variants"].values()) else "rejected"
    write("completed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decoder-module-file", type=Path)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--fields", type=int, default=4)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--dense-chunks", type=int, nargs="+", default=[0, 128])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    report = run(args)
    raise SystemExit(0 if report["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
