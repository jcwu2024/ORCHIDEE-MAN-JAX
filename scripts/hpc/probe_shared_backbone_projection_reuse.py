"""Bounded actual-shape common-source backbone K/V reuse diagnostic.

This uses synthetic inputs with the admitted split backbone geometry. It
does not load Teacher data, build the full model, or measure optimizer steps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
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
    _run_blocks,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (  # noqa: E402
    _residual_block_parameters,
)

RTOL = 1e-5
ATOL = 1e-6


def load_candidate(path: Path | None):
    name = "research.daily_coarse_graining.daily_operator.shared_backbone_projection_reuse"
    if path is None:
        module = __import__(name, fromlist=["run_backbone_with_shared_source_projection"])
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load candidate module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.run_backbone_with_shared_source_projection


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_reference(path: Path, archive: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(archive, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def allocator_snapshot(device: Any) -> dict[str, int] | None:
    try:
        stats = device.memory_stats()
    except (AttributeError, NotImplementedError):
        return None
    if stats is None:
        return None
    return {
        key: int(stats[key])
        for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_reserved", "peak_bytes_reserved")
        if key in stats
    }


def compare(actual: Any, expected: Any) -> dict[str, Any]:
    failed = coordinates = 0
    max_abs = 0.0
    for got, want in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        got_array, want_array = np.asarray(got), np.asarray(want)
        if got_array.shape != want_array.shape:
            raise ValueError(f"comparison shape mismatch: {got_array.shape} versus {want_array.shape}")
        failed += int(np.count_nonzero(~np.isclose(got_array, want_array, rtol=RTOL, atol=ATOL)))
        coordinates += want_array.size
        if want_array.size:
            max_abs = max(max_abs, float(np.max(np.abs(got_array.astype(np.float64) - want_array))))
    return {"passed": failed == 0, "coordinates": coordinates, "failed": failed, "max_abs": max_abs}


def make_payload(batch: int, *, experts: int, sources: int, axes: int, width: int) -> tuple[dict[str, Any], Any]:
    prefix = () if batch == 1 else (batch,)
    keys = jax.random.split(jax.random.PRNGKey(20260925 + batch), 3)
    payload = {
        "query": jax.random.normal(keys[0], prefix + (experts, 1, width), dtype=jnp.float32),
        "source": jax.random.normal(keys[1], prefix + (sources, width), dtype=jnp.float32),
        "axis": jax.random.normal(keys[2], prefix + (experts, axes, width), dtype=jnp.float32),
    }
    positions = jnp.arange(sources + axes)
    owners = jnp.arange(experts)
    allowed = (positions[None, :] + owners[:, None]) % 7 != 0
    allowed = allowed.at[-1, :].set(False)
    mask = jnp.broadcast_to(
        allowed[None, :, None, :] if batch != 1 else allowed[:, None, :], prefix + (experts, 1, sources + axes)
    )
    return payload, mask


def run(
    output: Path,
    candidate,
    *,
    width: int = 768,
    blocks_count: int = 8,
    experts: int = 7,
    sources: int = 393,
    axes: int = 140,
    microbatches: tuple[int, ...] = (1, 4),
    repeats: int = 3,
    require_gpu: bool = False,
    reuse_reference: bool = False,
) -> dict[str, Any]:
    if min(width, blocks_count, experts, sources, axes, repeats, *microbatches) <= 0 or width % 8:
        raise ValueError("all dimensions/repeats must be positive and width divisible by eight")
    devices = jax.devices()
    if require_gpu and not all(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU required, found {devices}")
    config = SharedTransformerConfig(width=width, ffn_width=4 * width, backbone_blocks=blocks_count)
    config.validate()
    reference_path = output.with_suffix(".reference.pkl")
    report: dict[str, Any] = {
        "status": "initializing",
        "scope": "synthetic backbone helper only; no full model, Teacher, recurrence or optimizer",
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "shape": {
            "width": width,
            "blocks": blocks_count,
            "heads": config.attention_heads,
            "experts": experts,
            "source_tokens": sources,
            "axis_tokens_per_expert": axes,
            "microbatches": microbatches,
        },
        "parity_tolerances": {"rtol": RTOL, "atol": ATOL},
        "reference_artifact": str(reference_path),
        "variants": {},
        "comparison": {},
    }
    write_report(output, report)
    keys = jax.random.split(jax.random.PRNGKey(20260924), blocks_count)
    parameters = tuple(_residual_block_parameters(key, width, 4 * width) for key in keys)
    parameter_dtypes = sorted({str(leaf.dtype) for leaf in jax.tree_util.tree_leaves(parameters)})
    if parameter_dtypes != ["float32"]:
        raise ValueError(f"backbone parameter dtype drift: {parameter_dtypes}")
    report["parameter_dtypes"] = parameter_dtypes
    metadata = {
        "jax_version": jax.__version__,
        "devices": report["devices"],
        "shape": report["shape"],
        "parameter_dtypes": parameter_dtypes,
        "initialization_seed": 20260924,
        "payload_seed_base": 20260925,
    }
    if reuse_reference:
        if not reference_path.is_file():
            raise FileNotFoundError(f"trusted reference artifact not found: {reference_path}")
        with reference_path.open("rb") as stream:
            archive = pickle.load(stream)  # noqa: S301 - opt-in, project-owned diagnostic artifact only
        if archive.get("metadata") != metadata:
            raise ValueError("reference artifact metadata differs from requested probe")
    else:
        archive = {"metadata": metadata, "results": {}}

    for batch in microbatches:
        payload, mask = make_payload(batch, experts=experts, sources=sources, axes=axes, width=width)
        payload_dtypes = {name: str(value.dtype) for name, value in payload.items()}
        if set(payload_dtypes.values()) != {"float32"}:
            raise ValueError(f"backbone payload dtype drift: {payload_dtypes}")
        report.setdefault("payload_dtypes", {})[f"micro{batch}"] = payload_dtypes
        weights = jnp.sin(jnp.arange(payload["query"].size, dtype=jnp.float32) * 0.13).reshape(payload["query"].shape)

        def reference(trainable, inputs):
            source = jnp.broadcast_to(
                inputs["source"][..., None, :, :],
                inputs["axis"].shape[:-2] + inputs["source"].shape[-2:],
            )
            bank = jnp.concatenate((source, inputs["axis"]), axis=-2)
            return _run_blocks(inputs["query"], bank, mask, trainable, config)

        def changed(trainable, inputs):
            return candidate(inputs["query"], inputs["source"], inputs["axis"], mask, trainable, config)

        results = {}
        for name, helper in (("reference", reference), ("shared_source", changed)):
            key = f"micro{batch}_{name}"
            if name == "reference" and f"micro{batch}" in archive["results"]:
                results[name] = archive["results"][f"micro{batch}"]
                report["variants"][key] = {"reused_trusted_reference": True}
                report["status"] = f"{key}:reused"
                write_report(output, report)
                continue
            report["status"] = f"{key}:before_lower"
            write_report(output, report)

            def objective(trainable, inputs):
                prediction = helper(trainable, inputs)
                return jnp.sum(prediction * weights), prediction

            function = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))
            started = time.perf_counter()
            try:
                allocator_before = allocator_snapshot(devices[0])
                lowered = function.lower(parameters, payload)
                lower_seconds = time.perf_counter() - started
                report["status"] = f"{key}:before_compile"
                report["variants"][key] = {"lower_seconds": lower_seconds}
                write_report(output, report)
                started = time.perf_counter()
                executable = lowered.compile()
                compile_seconds = time.perf_counter() - started
                report["status"] = f"{key}:before_execute"
                report["variants"][key]["compile_or_cache_load_seconds"] = compile_seconds
                write_report(output, report)
                result = executable(parameters, payload)
                jax.block_until_ready(result)
                timings = []
                for _ in range(repeats):
                    started = time.perf_counter()
                    timed = executable(parameters, payload)
                    jax.block_until_ready(timed)
                    timings.append((time.perf_counter() - started) * 1000.0)
                memory = executable.memory_analysis()
                report["variants"][key].update(
                    {
                        "median_warm_ms": float(np.median(timings)),
                        "compiler_temporary_bytes": None if memory is None else memory.temp_size_in_bytes,
                        "compiler_argument_bytes": None if memory is None else memory.argument_size_in_bytes,
                        "compiler_output_bytes": None if memory is None else memory.output_size_in_bytes,
                        "allocator_before": allocator_before,
                        "allocator_after_process_wide": allocator_snapshot(devices[0]),
                    }
                )
                results[name] = jax.device_get(result)
                if name == "reference":
                    archive["results"][f"micro{batch}"] = results[name]
                    report["status"] = f"{key}:before_reference_archive"
                    write_report(output, report)
                    write_reference(reference_path, archive)
                report["status"] = f"{key}:completed"
                write_report(output, report)
            except Exception as exc:
                report["status"] = f"{key}:failed"
                report["error"] = {"type": type(exc).__name__, "message": str(exc)}
                write_report(output, report)
                raise

        reference_result = results["reference"]
        candidate_result = results["shared_source"]
        comparison = {
            "loss": compare(candidate_result[0][0], reference_result[0][0]),
            "output": compare(candidate_result[0][1], reference_result[0][1]),
            "parameter_gradients": compare(candidate_result[1][0], reference_result[1][0]),
            "input_gradients": compare(candidate_result[1][1], reference_result[1][1]),
        }
        report["comparison"][f"micro{batch}"] = comparison
        report["status"] = f"micro{batch}:parity_complete"
        write_report(output, report)
        if not all(part["passed"] for part in comparison.values()):
            report["status"] = "parity_failed"
            write_report(output, report)
            return report

    report["status"] = "passed"
    write_report(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--experts", type=int, default=7)
    parser.add_argument("--source-tokens", type=int, default=393)
    parser.add_argument("--axis-tokens", type=int, default=140)
    parser.add_argument("--microbatches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--candidate-module-file", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--reuse-reference", action="store_true", help="Load only a trusted matching sibling reference.pkl"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "shared-backbone-projection-reuse.json")
    args = parser.parse_args()
    result = run(
        args.output,
        load_candidate(args.candidate_module_file),
        width=args.width,
        blocks_count=args.blocks,
        experts=args.experts,
        sources=args.source_tokens,
        axes=args.axis_tokens,
        microbatches=tuple(args.microbatches),
        repeats=args.repeats,
        require_gpu=args.require_gpu,
        reuse_reference=args.reuse_reference,
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
