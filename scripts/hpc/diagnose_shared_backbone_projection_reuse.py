"""Locate FP32 backbone source-K/V reuse differences on one real-sized block.

The fixture matches the first block and scalar payload of the existing
eight-block synthetic probe. This is not a full-model or training benchmark.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.daily_coarse_graining.daily_operator.shared_transformer import (  # noqa: E402
    SharedTransformerConfig,
    _normalized_attention,
    _run_blocks,
    _stable_head_l2_normalize,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (  # noqa: E402
    _linear,
    _residual_block_parameters,
)

_probe_path = Path(__file__).with_name("probe_shared_backbone_projection_reuse.py")
_probe_spec = importlib.util.spec_from_file_location("_staged_shared_backbone_probe", _probe_path)
if _probe_spec is None or _probe_spec.loader is None:
    raise RuntimeError(f"cannot load sibling probe from {_probe_path}")
_probe = importlib.util.module_from_spec(_probe_spec)
sys.modules[_probe_spec.name] = _probe
_probe_spec.loader.exec_module(_probe)
ATOL = _probe.ATOL
RTOL = _probe.RTOL
load_candidate = _probe.load_candidate
make_payload = _probe.make_payload
write_report = _probe.write_report


def comparison(actual: Any, expected: Any) -> dict[str, Any]:
    got = np.asarray(actual)
    want = np.asarray(expected)
    difference = np.abs(got.astype(np.float64) - want.astype(np.float64))
    failed = np.argwhere(~np.isclose(got, want, rtol=RTOL, atol=ATOL))
    return {
        "coordinates": int(want.size),
        "failed": int(len(failed)),
        "max_abs": float(np.max(difference)),
        "rms_abs": float(np.sqrt(np.mean(difference**2))),
        "first_failed_indices": failed[:8].tolist(),
    }


def against_oracle(reference: Any, candidate: Any, oracle: Any) -> dict[str, Any]:
    exact = np.asarray(oracle, dtype=np.float64)
    ref_error = np.abs(np.asarray(reference, dtype=np.float64) - exact)
    new_error = np.abs(np.asarray(candidate, dtype=np.float64) - exact)
    return {
        "reference_mean_abs": float(np.mean(ref_error)),
        "candidate_mean_abs": float(np.mean(new_error)),
        "reference_max_abs": float(np.max(ref_error)),
        "candidate_max_abs": float(np.max(new_error)),
        "candidate_closer_count": int(np.count_nonzero(new_error < ref_error)),
        "reference_closer_count": int(np.count_nonzero(ref_error < new_error)),
        "equal_distance_count": int(np.count_nonzero(ref_error == new_error)),
    }


def project_reference(attention: Any, source: Any, axis: Any) -> tuple[Any, Any]:
    broadcast_source = jnp.broadcast_to(source[None, :, :], axis.shape[:-2] + source.shape)
    bank = jnp.concatenate((broadcast_source, axis), axis=-2)
    return _linear(attention["key"], bank), _linear(attention["value"], bank)


def project_split(attention: Any, source: Any, axis: Any) -> tuple[Any, Any]:
    def one(projection: Any) -> Any:
        common = _linear(projection, source)
        local = _linear(projection, axis)
        return jnp.concatenate((jnp.broadcast_to(common[None, :, :], local.shape[:-2] + common.shape), local), axis=-2)

    return one(attention["key"]), one(attention["value"])


def project_oracle64(attention: Any, source: Any, axis: Any, *, split: bool) -> tuple[Any, Any]:
    if not jax.config.jax_enable_x64:
        raise RuntimeError("independent FP64 projection oracle requires JAX_ENABLE_X64=true")

    def linear64(projection: Any, value: Any) -> Any:
        return jnp.matmul(value.astype(jnp.float64), projection["kernel"].astype(jnp.float64)) + projection[
            "bias"
        ].astype(jnp.float64)

    if split:

        def one(projection: Any) -> Any:
            common = linear64(projection, source)
            local = linear64(projection, axis)
            return jnp.concatenate(
                (jnp.broadcast_to(common[None, :, :], local.shape[:-2] + common.shape), local), axis=-2
            )
    else:
        broadcast_source = jnp.broadcast_to(source[None, :, :], axis.shape[:-2] + source.shape)
        bank = jnp.concatenate((broadcast_source, axis), axis=-2)

        def one(projection: Any) -> Any:
            return linear64(projection, bank)

    return one(attention["key"]), one(attention["value"])


def attention_from_projected(
    attention: Any, query: Any, projected_key: Any, projected_value: Any, mask: Any, config: Any
) -> tuple[Any, Any, Any]:
    """Expose normalization/logits for attribution; use unchanged FP32 math."""

    width = int(query.shape[-1])
    heads = int(config.attention_heads)
    head_width = width // heads

    def per_head(value: Any) -> Any:
        return value.reshape(value.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)

    head_query = _stable_head_l2_normalize(per_head(_linear(attention["query"], query)), config.normalization_epsilon)
    head_key = _stable_head_l2_normalize(per_head(projected_key), config.normalization_epsilon)
    head_value = per_head(projected_value)
    logits = jnp.clip(jnp.einsum("...hqd,...hkd->...hqk", head_query, head_key), -1.0, 1.0)
    logits = logits * jnp.sqrt(jnp.asarray(head_width, dtype=jnp.float32))
    head_mask = jnp.expand_dims(mask, axis=-3)
    row_valid = jnp.expand_dims(jnp.any(mask, axis=-1, keepdims=True), axis=-3)
    masked_logits = jnp.where(head_mask, logits, -jnp.inf)
    weights = jax.nn.softmax(jnp.where(row_valid, masked_logits, 0.0), axis=-1)
    weights = jnp.where(head_mask, weights, 0.0)
    result = jnp.einsum("...hqk,...hkd->...hqd", weights, head_value)
    result = result.swapaxes(-3, -2).reshape(result.shape[:-3] + (result.shape[-2], width))
    return head_key, logits, _linear(attention["output"], result)


def save_arrays(path: Path, arrays: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **{key: np.asarray(value) for key, value in arrays.items()})
    os.replace(temporary, path)


def run(
    output: Path,
    candidate,
    *,
    width: int = 768,
    experts: int = 7,
    sources: int = 393,
    axes: int = 140,
    full_block_count: int = 8,
    require_gpu: bool = False,
) -> dict[str, Any]:
    if min(width, experts, sources, axes, full_block_count) <= 0 or width % 8:
        raise ValueError("positive dimensions and width divisible by eight required")
    if not jax.config.jax_enable_x64:
        raise RuntimeError("run with JAX_ENABLE_X64=true for the independent FP64 oracle")
    devices = jax.devices()
    if require_gpu and not all(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU required, found {devices}")
    arrays_path = output.with_suffix(".arrays.npz")
    report: dict[str, Any] = {
        "status": "initializing",
        "scope": "first backbone block, synthetic scalar payload matching eight-block probe; no full model or optimizer",
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "shape": {
            "width": width,
            "experts": experts,
            "source_tokens": sources,
            "axis_tokens_per_expert": axes,
            "full_probe_blocks": full_block_count,
        },
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "arrays_artifact": str(arrays_path),
        "stages": {},
    }
    write_report(output, report)
    config = SharedTransformerConfig(width=width, ffn_width=4 * width, backbone_blocks=full_block_count)
    block_key = jax.random.split(jax.random.PRNGKey(20260924), full_block_count)[0]
    block = _residual_block_parameters(block_key, width, 4 * width)
    payload, mask = make_payload(1, experts=experts, sources=sources, axes=axes, width=width)
    source, axis, query = payload["source"], payload["axis"], payload["query"]
    report["dtype"] = {
        "payload": sorted({str(leaf.dtype) for leaf in jax.tree_util.tree_leaves(payload)}),
        "parameters": sorted({str(leaf.dtype) for leaf in jax.tree_util.tree_leaves(block)}),
        "oracle": "float64 from original float32 values and weights",
    }
    if report["dtype"]["payload"] != ["float32"] or report["dtype"]["parameters"] != ["float32"]:
        raise ValueError("fixture is not strictly FP32")
    report["mask"] = {
        "valid": int(np.count_nonzero(np.asarray(mask))),
        "total": int(mask.size),
        "fully_masked_rows": int(np.count_nonzero(~np.any(np.asarray(mask), axis=-1))),
    }
    write_report(output, report)

    projections = {}
    for name, function in (
        ("reference_fp32", project_reference),
        ("split_fp32", project_split),
        ("reference_fp64", lambda attn, src, ax: project_oracle64(attn, src, ax, split=False)),
        ("split_fp64", lambda attn, src, ax: project_oracle64(attn, src, ax, split=True)),
    ):
        report["status"] = f"{name}:before_lower"
        write_report(output, report)
        started = time.perf_counter()
        lowered = jax.jit(function).lower(block["attention"], source, axis)
        lower_seconds = time.perf_counter() - started
        report["status"] = f"{name}:before_compile"
        write_report(output, report)
        started = time.perf_counter()
        executable = lowered.compile()
        compile_seconds = time.perf_counter() - started
        report["status"] = f"{name}:before_execute"
        write_report(output, report)
        projections[name] = jax.device_get(executable(block["attention"], source, axis))
        report["stages"][name] = {"lower_seconds": lower_seconds, "compile_seconds": compile_seconds}
        report["status"] = f"{name}:completed"
        write_report(output, report)

    key_ref, value_ref = projections["reference_fp32"]
    key_split, value_split = projections["split_fp32"]
    key64, value64 = projections["reference_fp64"]
    report["projection"] = {}
    for label, ref, new, exact in (("key", key_ref, key_split, key64), ("value", value_ref, value_split, value64)):
        report["projection"][label] = {
            "all": comparison(new, ref),
            "source": comparison(new[:, :sources], ref[:, :sources]),
            "axis": comparison(new[:, sources:], ref[:, sources:]),
            "oracle": against_oracle(ref, new, exact),
            "fp64_shape_check": comparison(projections["split_fp64"][0 if label == "key" else 1], exact),
        }
    report["status"] = "projection_analysis_complete"
    write_report(output, report)

    ref_stages = jax.device_get(attention_from_projected(block["attention"], query, key_ref, value_ref, mask, config))
    split_stages = jax.device_get(
        attention_from_projected(block["attention"], query, key_split, value_split, mask, config)
    )
    bank = jnp.concatenate((jnp.broadcast_to(source[None, :, :], axis.shape[:-2] + source.shape), axis), axis=-2)
    production_message = _normalized_attention(
        block["attention"],
        query,
        bank,
        heads=config.attention_heads,
        mask=mask,
        normalization_epsilon=config.normalization_epsilon,
    )
    reference_output = _run_blocks(query, bank, mask, (block,), config)
    candidate_output = candidate(query, source, axis, mask, (block,), config)
    report["attention"] = {
        "normalized_key": comparison(split_stages[0], ref_stages[0]),
        "unmasked_logits": comparison(split_stages[1], ref_stages[1]),
        "message": comparison(split_stages[2], ref_stages[2]),
        "reference_formula_check": comparison(ref_stages[2], production_message),
        "block_output": comparison(candidate_output, reference_output),
    }
    save_arrays(
        arrays_path,
        {
            "key_reference_fp32": key_ref,
            "key_split_fp32": key_split,
            "key_oracle_fp64": key64,
            "value_reference_fp32": value_ref,
            "value_split_fp32": value_split,
            "value_oracle_fp64": value64,
            "mask": mask,
            "block_reference": reference_output,
            "block_candidate": candidate_output,
        },
    )
    report["status"] = "completed_diagnostic_only"
    write_report(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--experts", type=int, default=7)
    parser.add_argument("--source-tokens", type=int, default=393)
    parser.add_argument("--axis-tokens", type=int, default=140)
    parser.add_argument("--full-block-count", type=int, default=8)
    parser.add_argument("--candidate-module-file", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "shared-backbone-projection-diagnosis.json")
    args = parser.parse_args()
    result = run(
        args.output,
        load_candidate(args.candidate_module_file),
        width=args.width,
        experts=args.experts,
        sources=args.source_tokens,
        axes=args.axis_tokens,
        full_block_count=args.full_block_count,
        require_gpu=args.require_gpu,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
