"""Bounded CPU-only prescreen for temporal lax.scan(unroll=2)."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np

from shared_temporal_scan_unroll_execution import TemporalUnrollJax


RTOL = 1e-5
ATOL = 1e-6


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def compare_tree(jax, candidate, reference):
    left = jax.tree_util.tree_leaves(candidate)
    right = jax.tree_util.tree_leaves(reference)
    if len(left) != len(right):
        raise ValueError("tree leaf count changed")
    failures = count = 0
    max_abs = 0.0
    bitwise = True
    for a, b in zip(left, right, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("tree shape/dtype changed")
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        passed = np.isfinite(a) & np.isfinite(b) & (delta <= ATOL + RTOL * np.abs(b))
        failures += int(np.count_nonzero(~passed))
        count += a.size
        max_abs = max(max_abs, float(np.max(delta, initial=0.0)))
        bitwise &= bool(np.array_equal(a, b))
    return {"passed": failures == 0, "failed_coordinates": failures, "coordinates": count,
            "max_abs_error": max_abs, "bitwise_identical": bitwise}


def emit(report_path: Path, report: dict, started: float, stage: str, **fields) -> None:
    report.update(stage=stage, elapsed_seconds=time.monotonic() - started, **fields)
    atomic_json(report_path, report)
    print(stage, flush=True)


def compact_ir_summary(module) -> dict:
    op_counts: Counter[str] = Counter()
    result_shapes: Counter[str] = Counter()

    def descendants(operation):
        yield operation
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from descendants(child)

    operations = list(descendants(module.operation))
    for operation in operations:
        op_counts[str(operation.name)] += 1
        for result in operation.results:
            type_text = str(result.type)
            if type_text.startswith("tensor<"):
                result_shapes[type_text] += 1
    loops = [operation for operation in operations if str(operation.name) in {
        "stablehlo.while", "scf.for", "stablehlo.while_loop"}]
    loop_bodies = []
    for loop in loops:
        body_count = 0
        for region in loop.regions:
            for block in region.blocks:
                body_count += sum(1 for child in block.operations for _ in descendants(child))
        loop_bodies.append({"operation": str(loop.name), "regions": len(loop.regions),
                            "nested_operation_count": body_count})
    return {
        "operation_count": len(operations),
        "operation_counts": dict(sorted(op_counts.items())),
        "loop_count": len(loops),
        "loop_bodies": loop_bodies,
        "tensor_result_shape_count": sum(result_shapes.values()),
        "distinct_tensor_result_shapes": len(result_shapes),
        "largest_shape_frequencies": [
            {"type": shape, "count": count} for shape, count in result_shapes.most_common(30)
        ],
    }


def stage_fixture(jax, jnp, RolloutLoss, report_path, report, started):
    def step(parameters, state, discrete, day):
        modulation = jnp.float32(0.7) + parameters[0] * jnp.float32(0.01)
        forcing = day[0] + parameters[1] * day[1]
        next_state = jnp.tanh(state * modulation + forcing + parameters[2] * state * day[2] * 0.01)
        flux = next_state * parameters[1] + day[1]
        discrete_next = discrete + (day[0] > 0).astype(jnp.int32)
        return next_state, discrete_next, {"flux": flux, "state_copy": next_state + day[2]}

    def loss_with_aux(parameters, initial_state, observations, targets, context):
        residual = observations["flux"] - targets
        mse = jnp.mean(residual * residual)
        state_penalty = jnp.mean(observations["state"] ** 2)
        loss = mse + context * jnp.float32(0.01) * state_penalty
        return loss, {"mse": mse, "state_penalty": state_penalty,
                      "trajectory": observations, "initial_state": initial_state}

    parameters = jnp.asarray([0.8, -0.4, 0.3], dtype=jnp.float32)
    initial_state = jnp.asarray([0.1, -0.2, 0.3, 0.4], dtype=jnp.float32)
    initial_discrete = jnp.asarray(2, dtype=jnp.int32)
    index = np.arange(30, dtype=np.float32)
    days_a = jnp.asarray(np.stack((index * 0.01, np.sin(index * 0.13), np.cos(index * 0.07)), axis=1))
    days_b = days_a.at[:, 1].add(jnp.float32(0.015))
    targets = jnp.asarray(np.cos(index[:, None] * 0.05 + np.arange(4, dtype=np.float32)[None, :]) * 0.2)
    context = jnp.asarray(0.6, dtype=jnp.float32)

    arms = {}
    for unroll in (1, 2):
        facade = TemporalUnrollJax(jax, unroll)
        rollout = RolloutLoss(facade, jnp, step, lambda *_: jnp.float32(0.0))
        bound = rollout.bind_checkpointed_loss_with_aux(loss_with_aux)

        def objective(p, state, discrete, days, target, weight):
            return bound(p, state, discrete, days, target, weight)

        native = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))
        outputs = {}
        for name, days in (("a", days_a), ("b", days_b)):
            outputs[name] = jax.device_get(jax.block_until_ready(
                native(parameters, initial_state, initial_discrete, days, targets, context)))
        if (facade.lax.calls != [{"length": 30, "unroll": unroll}]
                or facade.checkpoint_calls != [{"name": "transition", "kwargs": []}]):
            raise AssertionError("the public objective did not trace the requested temporal scan")
        arms[unroll] = {"outputs": outputs, "scan_calls": facade.lax.calls,
                        "checkpoint_calls": facade.checkpoint_calls}
        emit(report_path, report, started, "fixture_arm_complete", unroll=unroll,
             scan_calls=facade.lax.calls, checkpoint_calls=facade.checkpoint_calls)

    comparisons = {name: compare_tree(jax, arms[2]["outputs"][name], arms[1]["outputs"][name])
                   for name in ("a", "b")}
    dynamic_changed = compare_tree(jax, arms[1]["outputs"]["a"][0], arms[1]["outputs"]["b"][0])
    dynamic_changed_candidate = compare_tree(jax, arms[2]["outputs"]["a"][0], arms[2]["outputs"]["b"][0])
    if dynamic_changed["failed_coordinates"] == 0 or dynamic_changed_candidate["failed_coordinates"] == 0:
        raise AssertionError("equal-shaped dynamic inputs did not alter the outputs")
    strict = all(item["passed"] for item in comparisons.values())
    emit(report_path, report, started, "fixture_complete", fixture_comparisons=comparisons,
         dynamic_input_effect={"unroll1": dynamic_changed, "unroll2": dynamic_changed_candidate},
         fixture_strict_passed=strict,
         fixture_scope="RolloutLoss public checkpointed interface, 30 dynamic steps, complete observation auxiliary and native parameter/initial-state gradients")
    if not strict:
        raise AssertionError("fixture strict comparison failed")


def stage_expert5(jax, jnp, RolloutLoss, args, report_path, report, started):
    from probe_shared_command_buffer_loops import original_component
    import importlib

    with args.payload.open("rb") as stream:
        source_dir = args.rollout_source_dir.resolve()
        source_modules = {
            "strong_baseline_architecture": source_dir / "strong_baseline_architecture.py",
            "shared_transformer": source_dir / "shared_transformer.py",
        }
        missing = [str(path) for path in source_modules.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"source overlay is incomplete: {missing}")
        importlib.import_module("research.daily_coarse_graining.daily_operator")
        package_name = "research.daily_coarse_graining.daily_operator"
        loaded_modules = {}
        for name, source_file in source_modules.items():
            module_name = f"{package_name}.{name}"
            spec = importlib.util.spec_from_file_location(module_name, source_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load source-reused module {source_file}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            setattr(sys.modules[package_name], name, module)
            loaded_modules[name] = module
        shared_file = source_modules["shared_transformer"]
        saved = pickle.load(stream)
    selected = next(row for row in saved["records"] if row["route"] == "expert5" and row["sample"] == 0)
    config = saved["config"]
    if (saved["schema"] != "shared_normalized_keys_real_v1"
            or selected["query"].shape != (7542, 1, 384) or selected["chunk"] != 128
            or (config.resolved_decoder_width, config.resolved_decoder_attention_heads,
                config.coordinate_blocks) != (384, 8, 2)):
        raise ValueError("saved checkpoint-CSE input no longer identifies original expert5 sample0")
    parameters = jax.device_put((saved["parameters"], selected["dynamic"], selected["banks"]))
    query = jax.device_put(selected["query"].reshape(-1))
    masks, indices = jax.device_put((selected["masks"], selected["indices"]))
    days = jax.device_put(np.arange(2, dtype=np.int32))
    discrete = jax.device_put(np.int32(0))
    cotangent = jnp.sin(jnp.arange(query.size, dtype=jnp.float32) * 0.11).reshape(selected["query"].shape)
    parameter_count = sum(np.size(value) for value in jax.tree_util.tree_leaves(parameters[0]))
    if parameter_count != 3_548_928 or any(value.dtype != np.float32 for value in jax.tree_util.tree_leaves((parameters, query))):
        raise ValueError("original expert5 parameter count or FP32 dtype changed")

    def step(packed, state, discrete_state, _day):
        blocks, dynamic, banks = packed
        current = state.reshape(selected["query"].shape)
        decoded = original_component(blocks, (current, dynamic, *banks), masks, indices, config, 128)
        return (current + jnp.float32(0.001) * decoded).reshape(-1), discrete_state + 1, {"decoded": decoded}

    def loss_with_aux(_packed, _initial, observations, _targets, _context):
        value = jnp.sum(observations["decoded"] * cotangent[None, ...], dtype=jnp.float32)
        return value, {"final_state": observations["state"][-1]}

    def unused_loss(*_):
        raise RuntimeError("the bound compact auxiliary objective is used")

    arms = {}
    for unroll in (1, 2):
        facade = TemporalUnrollJax(jax, unroll, checkpoint_prevent_cse=True)
        rollout = RolloutLoss(facade, jnp, step, unused_loss)
        bound = rollout.bind_checkpointed_loss_with_aux(loss_with_aux)

        def objective(p, state):
            return bound(p, state, discrete, days, None, None)

        native = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)
        emit(report_path, report, started, "expert5_lowering_started", unroll=unroll,
             parameters=parameter_count, query_shape=list(selected["query"].shape), temporal_steps=2,
             checkpoint_prevent_cse=True)
        began = time.monotonic()
        lowered = jax.jit(native).lower(parameters, query)
        module = lowered.compiler_ir(dialect="stablehlo")
        summary = compact_ir_summary(module)
        arms[unroll] = {"lower_seconds": time.monotonic() - began, "ir": summary,
                        "scan_calls": facade.lax.calls}
        del module, lowered, native
        if (facade.lax.calls != [{"length": 2, "unroll": unroll}]
                or facade.checkpoint_calls != [{"name": "transition", "prevent_cse": True, "policy": None}]):
            raise AssertionError("expert5 path did not execute exactly one two-step scan trace")
        emit(report_path, report, started, "expert5_arm_lowered", unroll=unroll,
             lower_seconds=arms[unroll]["lower_seconds"], ir=summary,
             scan_calls=facade.lax.calls, checkpoint_calls=facade.checkpoint_calls)
    baseline, candidate = arms[1]["ir"], arms[2]["ir"]
    body_growth = {}
    for key in ("operation_count", "tensor_result_shape_count", "distinct_tensor_result_shapes", "loop_count"):
        before, after = baseline[key], candidate[key]
        body_growth[key] = {"unroll1": before, "unroll2": after,
                            "delta": after - before,
                            "ratio": (after / before if before else None)}
    emit(report_path, report, started, "expert5_complete", expert5_identity=saved["identity"],
         shared_transformer_source_sha256=hashlib.sha256(shared_file.read_bytes()).hexdigest(),
         strong_baseline_source_sha256=hashlib.sha256(source_modules["strong_baseline_architecture"].read_bytes()).hexdigest(),
         expert5_scope={"route": "expert5", "sample": 0, "query_shape": list(selected["query"].shape),
                        "rows": 7542, "padded_rows": 7552, "chunks": 59, "chunk_size": 128,
                        "decoder_parameter_coordinates": parameter_count, "steps": 2,
                        "recurrence": "q_next=q+0.001*original_component(q,dynamic,raw_banks)"},
         checkpoint_policy="prevent_cse=True on both arms; default policy unchanged",
         expert5_ir={"unroll1": arms[1], "unroll2": arms[2], "comparison": body_growth},
         backend_compilation=False, large_hlo_exported=False, model_outputs_executed=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--rollout-source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.budget_seconds <= 120:
        raise ValueError("the remaining prescreen budget must be between 1 and 120 seconds")
    if args.output.exists():
        raise FileExistsError("preserve existing prescreen output")
    started = time.monotonic()
    report = {"status": "running", "scope": "CPU-only T30 temporal scan unroll 1 versus 2 prescreen",
              "backend": "cpu", "requested_budget_seconds": args.budget_seconds, "rtol": RTOL, "atol": ATOL,
              "gpu_count": 0, "paid_jobs": 0, "training_updates": 0,
              "source_adapter": "scripts/hpc/shared_temporal_scan_unroll_execution.py",
              "checkpoint_policy": "preserved; original default policy and prevent_cse=True in expert5 screen",
              "production_rollout_changed": False}
    atomic_json(args.output, report)
    try:
        import jax
        import jax.numpy as jnp
        if jax.default_backend() != "cpu" or jax.__version__ != "0.4.38":
            raise RuntimeError(f"requires CPU backend with JAX0.4.38, got {jax.default_backend()} {jax.__version__}")
        report.update(jax_version=jax.__version__, jaxlib_version=__import__("jaxlib").__version__,
                      devices=[str(device) for device in jax.devices()])
        source_dir = args.rollout_source_dir.resolve()
        rollout_file = source_dir / "rollout_loss.py"
        adjoint_file = source_dir / "streamed_adjoint.py"
        if not rollout_file.is_file() or not adjoint_file.is_file():
            raise FileNotFoundError("current RolloutLoss diagnostic source overlay is incomplete")
        package_name = "_temporal_rollout_source"
        package = type(sys)(package_name)
        package.__path__ = [str(source_dir)]
        sys.modules[package_name] = package
        for module_name, source_file in (("streamed_adjoint", adjoint_file), ("rollout_loss", rollout_file)):
            spec = importlib.util.spec_from_file_location(f"{package_name}.{module_name}", source_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load diagnostic source {source_file}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        RolloutLoss = sys.modules[f"{package_name}.rollout_loss"].RolloutLoss
        if not hasattr(RolloutLoss, "bind_checkpointed_loss_with_aux"):
            raise RuntimeError("staged RolloutLoss source lacks the compact checkpointed interface")
        emit(args.output, report, started, "runtime_verified", cpu_affinity=sorted(os.sched_getaffinity(0)),
             payload=str(args.payload), rollout_source={
                 "directory": str(source_dir),
                 "rollout_loss_sha256": hashlib.sha256(rollout_file.read_bytes()).hexdigest(),
                 "streamed_adjoint_sha256": hashlib.sha256(adjoint_file.read_bytes()).hexdigest(),
             })
        stage_fixture(jax, jnp, RolloutLoss, args.output, report, started)
        remaining = args.budget_seconds - (time.monotonic() - started)
        if remaining < 35:
            emit(args.output, report, started, "stage2_skipped_budget", remaining_seconds=remaining,
                 reason="preserve time for clean bounded CPU process exit")
            report["status"] = "fixture_passed_expert5_not_run_budget"
            atomic_json(args.output, report)
            return 0
        stage_expert5(jax, jnp, RolloutLoss, args, args.output, report, started)
        report["status"] = "completed"
        atomic_json(args.output, report)
        return 0
    except BaseException as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        atomic_json(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
