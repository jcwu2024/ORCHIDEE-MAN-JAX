"""Compare 22 audited same-expert Q1 fields through actual split decoder helpers."""

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

from research.daily_coarse_graining.daily_operator.shared_decoder_dense import (  # noqa: E402
    DenseDecoderField,
    reference_decode_fields,
)
from research.daily_coarse_graining.daily_operator.shared_transformer import (  # noqa: E402
    SharedTransformerConfig,
    _DecoderBlockProjection,
    _project_key_value,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (  # noqa: E402
    _linear_parameters,
    _residual_block_parameters,
)

AUDITED_FIELD_INDICES = (8, 9, 14, 15, 40, 42, 44, 45, 46, 53, 56, 61, 118, 119, 124, 125, 129, 133, 135, 138, 141, 142)
AUDITED_SIGNATURE = {
    "eligible_coordinates": 1,
    "chunk_width": 1,
    "chunk_count": 1,
    "axis_slots": 0,
    "axis_bank": 0,
    "local_bank": 1,
    "uses_carbon_bank": False,
}
FAMILY_COUNT = 20


def load_candidate(path):
    name = "research.daily_coarse_graining.daily_operator.shared_decoder_q1_group"
    if path is None:
        module = __import__(name, fromlist=["grouped_decode_q1_fields"])
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load candidate module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.grouped_decode_q1_fields


def fixture(width, *, seed=20260925):
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 2 + len(AUDITED_FIELD_INDICES) * 2 + 4))
    blocks = []
    for block_index in range(2):
        block = _residual_block_parameters(next(keys), width, 4 * width)
        for part in ("key", "value"):
            block["attention"][part]["bias"] = jnp.linspace(0.01, 0.02, width, dtype=jnp.float32) * (block_index + 1)
        blocks.append(block)
    shared = {
        "coordinate_blocks": tuple(blocks),
        "readout_first": _linear_parameters(next(keys), width, min(width, 256)),
        "readout_second": _linear_parameters(next(keys), min(width, 256), 1),
    }
    payload = {
        "family": jax.random.normal(next(keys), (FAMILY_COUNT, width), dtype=jnp.float32),
        "process": jax.random.normal(next(keys), (1, width), dtype=jnp.float32),
        "fields": tuple(
            {
                "query": jax.random.normal(next(keys), (1, 1, width), dtype=jnp.float32),
                "local": jax.random.normal(next(keys), (1, width), dtype=jnp.float32),
            }
            for _ in AUDITED_FIELD_INDICES
        ),
    }
    masks = []
    for index in range(len(AUDITED_FIELD_INDICES)):
        active = index not in (0, 11)
        family = (np.arange(FAMILY_COUNT) + index) % 4 != 0
        if index == 7:
            family[:] = False
        masks.append(
            {
                "active": jnp.asarray([active]),
                "family": jnp.asarray(family[None, None, :] & active),
                "local": jnp.asarray([[[active and index % 5 != 0]]]),
                "process": jnp.asarray([[[active]]]),
            }
        )
    return shared, payload, tuple(masks)


def prepare_fields(shared, payload, masks):
    projections = []
    for block in shared["coordinate_blocks"]:
        family_key, family_value = _project_key_value(block["attention"], payload["family"])
        process_key, process_value = _project_key_value(block["attention"], payload["process"])
        projections.append((family_key, family_value, process_key, process_value, block))
    fields = []
    for data, mask in zip(payload["fields"], masks, strict=True):
        field_projection = []
        for family_key, family_value, process_key, process_value, block in projections:
            local_key, local_value = _project_key_value(block["attention"], data["local"])
            field_projection.append(
                _DecoderBlockProjection(
                    family_key=family_key,
                    family_value=family_value,
                    process_key=process_key,
                    process_value=process_value,
                    carbon_key=None,
                    carbon_value=None,
                    axis_key=jnp.zeros((0, data["local"].shape[-1]), dtype=jnp.float32),
                    axis_value=jnp.zeros((0, data["local"].shape[-1]), dtype=jnp.float32),
                    local_key=local_key,
                    local_value=local_value,
                )
            )
        fields.append(
            DenseDecoderField(
                query=data["query"],
                dynamic_tokens=data["local"][None, :, :],
                dynamic_mask=mask["local"],
                family_mask=mask["family"],
                process_mask=mask["process"],
                carbon_mask=None,
                projections=tuple(field_projection),
                dynamic_projection_indices=(
                    jnp.zeros((1, 0), dtype=jnp.int32),
                    jnp.zeros((1,), dtype=jnp.int32),
                    mask["active"],
                ),
                output_active=mask["active"],
            )
        )
    return tuple(fields)


def compare(actual, expected, *, rtol=1e-5, atol=1e-6):
    failed = count = 0
    max_abs = 0.0
    for new, old in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        new, old = np.asarray(new), np.asarray(old)
        failed += int(np.count_nonzero(~np.isclose(new, old, rtol=rtol, atol=atol)))
        count += old.size
        max_abs = max(max_abs, float(np.max(np.abs(new.astype(np.float64) - old.astype(np.float64)))))
    return {"passed": failed == 0, "coordinates": count, "failed": failed, "max_abs": max_abs, "rtol": rtol, "atol": atol}


def run(width, repeats, candidate, *, require_gpu, microbatch=1):
    if width % 8 or min(width, repeats) <= 0 or microbatch not in (1, 4):
        raise ValueError("width must be divisible by eight; repeats positive; microbatch 1 or 4")
    devices = jax.devices()
    if require_gpu and not all(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU required, found {devices}")
    shared, payload, masks = fixture(width)
    if microbatch == 4:
        samples = [payload]
        samples.extend(fixture(width, seed=20260925 + index)[1] for index in range(1, 4))
        payload = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *samples)
    config = SharedTransformerConfig(width=width, ffn_width=4 * width, coordinate_blocks=2, readout_width=min(width, 256))
    report = {
        "scope": "22 audited hydrology Q1 fields, decoder only; no Teacher, tokenizer, recurrence or full model",
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "width": width,
        "microbatch": microbatch,
        "blocks": 2,
        "ffn_width": 4 * width,
        "family_count": FAMILY_COUNT,
        "audited_field_indices": AUDITED_FIELD_INDICES,
        "audited_signature": AUDITED_SIGNATURE,
        "variants": {},
    }
    results = {}
    for name, decode in (("reference", reference_decode_fields), ("grouped", candidate)):
        def one(parameters, inputs):
            fields = prepare_fields(parameters, inputs, masks)
            outputs = jnp.concatenate(decode(parameters, fields, config))
            target = jnp.sin(jnp.arange(len(AUDITED_FIELD_INDICES), dtype=jnp.float32) * 0.13)
            return jnp.mean((outputs - target) ** 2), outputs

        if microbatch == 1:
            objective = one
        else:
            def objective(parameters, inputs):
                losses, outputs = jax.vmap(one, in_axes=(None, 0))(parameters, inputs)
                return jnp.mean(losses), outputs

        function = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))
        equations = len(jax.make_jaxpr(lambda p, x: objective(p, x)[0])(shared, payload).jaxpr.eqns)
        start = time.perf_counter()
        lowered = function.lower(shared, payload)
        lower_seconds = time.perf_counter() - start
        start = time.perf_counter()
        executable = lowered.compile()
        compile_seconds = time.perf_counter() - start
        results[name] = executable(shared, payload)
        jax.block_until_ready(results[name])
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = executable(shared, payload)
            jax.block_until_ready(result)
            timings.append((time.perf_counter() - start) * 1000.0)
        memory = executable.memory_analysis()
        report["variants"][name] = {
            "primal_jaxpr_equations": equations,
            "lower_seconds": lower_seconds,
            "compile_or_cache_load_seconds": compile_seconds,
            "median_warm_ms": float(np.median(timings)),
            "compiler_temporary_bytes": memory.temp_size_in_bytes,
            "compiler_argument_bytes": memory.argument_size_in_bytes,
        }
    ref, grouped = results["reference"], results["grouped"]
    report["comparison"] = {
        "loss_output": compare(grouped[0], ref[0]),
        "parameter_payload_gradients": compare(grouped[1], ref[1]),
    }
    report["numerical_parity_passed"] = all(row["passed"] for row in report["comparison"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--microbatch", type=int, choices=(1, 4), default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--candidate-module-file", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "shared-decoder-q1-group-local.json")
    args = parser.parse_args()
    candidate = load_candidate(args.candidate_module_file)
    report = run(args.width, args.repeats, candidate, require_gpu=args.require_gpu, microbatch=args.microbatch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["numerical_parity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
