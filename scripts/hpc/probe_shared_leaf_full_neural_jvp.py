"""Independent CPU scalar public-loss full-neural state JVP at coordinate2243.

Recompute the historical original neural backbone/head from dynamic state.
JIT callable component boundaries remain transparent to AD; no compiled
executable calls, output replay, custom VJP or boundary collector is used.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import time

try:
    from scripts.hpc.probe_shared_leaf_full_state_cuda import adjacent, prepare, Journal, BASE_SHA, CORRECTED_SHA, install_turnover_overlay, read_pickle, validate_snapshot, sha
except ModuleNotFoundError:
    from probe_shared_leaf_full_state_cuda import adjacent, prepare, Journal, BASE_SHA, CORRECTED_SHA, install_turnover_overlay, read_pickle, validate_snapshot, sha

import numpy as np


def ad_transparent_component(jax, function, name, emit):
    """Close static metadata outside jit; dynamic arrays remain ordinary AD args."""
    cache = {}
    def wrapped(*args, **kwargs):
        key = tuple((key, id(value)) for key, value in sorted(kwargs.items()))
        if key not in cache:
            # Retain kwargs in the closure, so identity keys cannot be recycled.
            static = dict(kwargs)
            cache[key] = jax.jit(lambda *dynamic: function(*dynamic, **static))
            emit(name+"_AD_transparent_callable_bound", static_keys=list(static))
        with jax.disable_jit(False):
            return cache[key](*args)
    return wrapped


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "prepared-context", "old-snapshot", "new-snapshot", "field-adapter", "backbone-adapter", "turnover-overlay", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--turnover-base-sha256", default=BASE_SHA)
    p.add_argument("--turnover-sha256", "--turnover-corrected-sha256", dest="turnover_sha256", default=CORRECTED_SHA)
    p.add_argument("--domains", nargs="+", choices=("old:16", "old:17", "new:16", "new:17"), default=["old:17"])
    p.add_argument("--neural-execution", choices=("eager", "component_jit"), default="component_jit")
    p.add_argument("--wall-seconds", type=float, default=1200)
    return p


def run(args):
    if not 1 <= args.wall_seconds <= 3600 or len(set(args.domains)) != len(args.domains):
        raise ValueError("bounded wall≤3600 and unique actual domains required")
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    import jax
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "cpu": raise ValueError("independent CPU route required")
    adjacent("training_artifact_io")
    journal = Journal(args.output, args.wall_seconds)
    journal.report.update(schema_version="shared_leaf_independent_full_neural_cpu_jvp_v1", GPU_executed=False,
        network_state_AD=True, parameter_AD=False, direction_coordinate=2243,
        neural_execution=args.neural_execution, scalar_to_micro4_factor=.25,
        independent_route="scalar_public_objective_native_jvp_recomputed_neural")
    try:
        journal.emit("source_and_input_validation")
        _, owner, manifest, _, context, micro4, identity = prepare(jax, args)
        identity["independent_CPU_helper_sha256"] = sha(Path(__file__))
        journal.report["identity"] = identity
        from research.daily_coarse_graining.daily_operator import shared_process_head
        from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_joint_daily_objective
        from jax_orchidee.stomate import carbon_kernels
        adapter = owner.load_staged_adapter(args.field_adapter)
        assembler = owner.load_backbone_assembler(args.backbone_adapter)
        if args.neural_execution == "component_jit":
            adapter = ad_transparent_component(jax, adapter, "decoder", journal.emit)
            assembler = ad_transparent_component(jax, assembler, "backbone", journal.emit)
        scalar = make_shared_joint_daily_objective(context["runtime"], context["head"], context["template"],
            context["state_scales"], context["science_scales"], include_state_diagnostics=True)
        active_schedule, parameters = None, None
        with ExitStack() as patches:
            patches.enter_context(install_turnover_overlay(carbon_kernels, args.turnover_overlay, BASE_SHA))
            patches.enter_context(owner.candidate_process_head_patches(shared_process_head, adapter, assembler))
            patches.enter_context(jax.disable_jit())
            for domain in args.domains:
                schedule, row = domain.split(":")
                row = int(row)
                if schedule != active_schedule:
                    saved = read_pickle(getattr(args, schedule+"_snapshot"))
                    host_parameters, _ = validate_snapshot(jax, saved, manifest, schedule)
                    del saved
                    parameters = jax.device_put(host_parameters)
                    del host_parameters
                    active_schedule = schedule
                sample = jax.device_put(jax.tree_util.tree_map(lambda x: x[row-16], micro4))
                state = sample.inputs.continuous_state
                direction = jax.numpy.zeros_like(state).at[2243].set(1)
                def objective(dynamic_state):
                    changed = sample._replace(inputs=sample.inputs._replace(continuous_state=dynamic_state))
                    return scalar(parameters, changed)
                journal.emit("independent_scalar_JVP_entered", domain=domain)
                began = time.monotonic()
                loss, derivative, auxiliary = jax.jvp(objective, (state,), (direction,), has_aux=True)
                loss, derivative, auxiliary = jax.device_get(jax.block_until_ready((loss, derivative, auxiliary)))
                path = args.output.with_suffix("."+schedule+f".row{row}.jvp.pkl")
                if path.exists(): raise FileExistsError(path)
                owner.atomic_pickle(path, {"identity": identity, "domain": domain, "state": np.asarray(state),
                    "direction": np.asarray(direction), "loss": loss, "scalar_derivative": derivative,
                    "micro4_mean_row_derivative": derivative/4, "auxiliary": auxiliary})
                ordinary = {k: v for k, v in auxiliary.items() if k != "state_diagnostics"}
                hard = owner.hard_counts(auxiliary)
                result = {"domain": domain, "artifact": str(path), "seconds": time.monotonic()-began,
                    "loss": float(loss), "scalar_derivative": float(derivative), "micro4_mean_row_derivative": float(derivative/4),
                    "finite": bool(all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves((loss, derivative, ordinary)))),
                    "hard_counts": hard, "physical_hard_counts_zero": not any(hard.values())}
                journal.report["results"].append(result)
                journal.emit("independent_scalar_JVP_preserved", **result)
        journal.report.update(status="completed", finite_and_physical_passed=all(x["finite"] and x["physical_hard_counts_zero"] for x in journal.report["results"]))
        journal.emit("completed")
        return journal.report
    except Exception as error:
        journal.report.update(status="error", error_type=type(error).__name__, error=str(error))
        journal.emit("error")
        raise
    finally:
        journal.timer.cancel()


if __name__ == "__main__":
    run(parser().parse_args())
