"""Discarded four-window step using full streamed BPTT and the production loss."""

from __future__ import annotations

import json
import time

import numpy as np

from research.daily_coarse_graining.daily_operator.streamed_adjoint import StreamedAdjoint
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_science_window_sufficient_statistics,
    science_window_from_sufficient_statistics,
)
from scripts.hpc import audit_conditioned_attention_streamed_adjoint as streamed

window = streamed.window
EPSILONS = (1e-4, 1e-5)


def merge_statistics(items):
    total = items[0]
    for item in items[1:]:
        total = add_science_window_sufficient_statistics(total, item)
    return total


def trace_days(jax, micro, trace):
    templates = jax.tree_util.tree_map(lambda v: v[0], micro.inputs)
    retained = jax.tree_util.tree_map(lambda v: v[0], micro.day_inputs)
    if any(np.asarray(v).dtype.kind not in "biu" for v in jax.tree_util.tree_leaves(trace.result.next_discrete_state)):
        raise ValueError("differentiable discrete carry requires an additional adjoint")
    states, days = [], []
    for i in range(int(templates.continuous_state.shape[0])):
        states.append(templates.continuous_state[0] if i == 0 else trace.result.next_state.other[i - 1])
        discrete = (
            jax.tree_util.tree_map(lambda v: v[0], templates.discrete_state) if i == 0
            else jax.tree_util.tree_map(lambda v: v[i - 1], trace.result.next_discrete_state)
        )
        days.append((jax.tree_util.tree_map(lambda v: v[i], templates), discrete, retained))
    return states, days


def science_summary(statistics, flux_basis):
    result = science_window_from_sufficient_statistics(statistics, flux_basis=flux_basis)
    return {
        "loss": float(result.total),
        "components": {
            group: {
                name: {"mse": float(error.mse), "bias_squared": float(error.bias_squared), "count": float(error.count)}
                for name, error in getattr(result, group).items()
            }
            for group in ("flux_errors", "stock_errors")
        },
    }


def same_counts(left, right):
    return all(
        row["count"] == right["components"][group][name]["count"]
        for group, values in left["components"].items() for name, row in values.items()
    )


def classify(report):
    threshold = max(1e-8, 10 * abs(report["baseline_repeat"]["loss"] - report["baseline"]["loss"]))
    rows = report["evaluations"]
    complete = (
        len(rows) == 4 and {(row["epsilon"], row["sign"]) for row in rows} == {(e, s) for e in EPSILONS for s in (1, -1)}
        and len(report["windows"]) == 4 and {row["window"] for row in report["windows"]} == set(range(4))
        and len(report["finite_differences"]) == 2 and {row["epsilon"] for row in report["finite_differences"]} == set(EPSILONS)
    )
    hard = report["baseline"]["hard_failure_count"] + report["baseline_repeat"]["hard_failure_count"]
    finite_steps = (
        np.isfinite(report["baseline"]["loss"]) and np.isfinite(report["baseline_repeat"]["loss"])
        and all(np.isfinite(row["loss"]) and np.isfinite(row["predicted_change"]) for row in rows)
    )
    hard += sum(row["hard_failure_count"] for row in rows)
    counts = same_counts(report["baseline"], report["baseline_repeat"]) and all(
        same_counts(report["baseline"], row) for row in rows
    )
    agreement = all(
        np.isfinite(row["relative_derivative_error"]) and row["relative_derivative_error"] <= 0.01
        and row["central_secant"] < 0 and row["gradient_dot_actual_direction"] < 0
        for row in report["finite_differences"]
    )
    descent = all(row["actual_change"] < -threshold and row["predicted_change"] < 0 for row in rows if row["sign"] == 1)
    primal = all(row["linearized_score_agrees"] for row in report["windows"])
    passed = complete and finite_steps and hard == 0 and counts and agreement and descent and primal
    return {
        "classification": "streamed_step_admitted_for_multi_artifact_check" if passed else "streamed_step_not_admitted",
        "passed": bool(passed), "loss_decrease_floor": threshold, "hard_failures": hard,
        "science_counts_unchanged": counts, "derivatives_agree": agreement,
        "both_magnitudes_descend": descent, "observation_linearization_agrees": primal,
        "finite_step_values": finite_steps,
        "complete": complete,
    }


def experiment(bootstrap, values, artifact, *, emit):
    context = bootstrap["context"]
    jax, jnp = context["runtime"]["jax"], context["runtime"]["jnp"]
    fixed, basis = bootstrap["fixed_parameters"], bootstrap["flux_basis"]
    micros = [window._slice(artifact.batch.thirty_day, i, i + 1) for i in range(4)]
    report = {"windows": [], "evaluations": [], "finite_differences": []}

    def evaluate(current):
        parameters = window.replace_conditioned_coupled_gpp_trainable(fixed, current)
        outputs = [bootstrap["steps"]["thirty_forward"](parameters, micro) for micro in micros]
        jax.block_until_ready(outputs)
        total = merge_statistics([v[0] for v in outputs])
        summary = science_summary(total, basis)
        summary["window_losses"] = [science_summary(v[0], basis)["loss"] for v in outputs]
        summary["hard_failure_count"] = sum(window._hard_failure_count(window._hard_host(jax.device_get(v[1]))) for v in outputs)
        return summary, total, outputs

    emit("baseline_forward", report)
    baseline, total, outputs = evaluate(values)
    report["baseline"] = baseline
    emit("baseline_complete", report)
    if baseline["hard_failure_count"] or not np.isfinite(baseline["loss"]):
        raise RuntimeError("baseline physical/finite loss gate failed")
    trace_evaluator = window._make_trace_evaluator(bootstrap)
    reverse = StreamedAdjoint(jax, jnp, streamed.make_step(bootstrap))
    gradient = jax.tree_util.tree_map(lambda v: jnp.zeros_like(v, dtype=jnp.float64), values)
    parameters = window.replace_conditioned_coupled_gpp_trainable(fixed, values)
    for i, micro in enumerate(micros):
        emit(f"window_{i}_forward", report)
        trace, _ = trace_evaluator(parameters, micro)
        jax.block_until_ready(trace)
        _, score_gradient = streamed.make_observation_score(bootstrap, micro)
        loss, cotangents = score_gradient(
            streamed.observations(trace), trace.result.next_state_defined.other, trace.active_pft_mask, total,
        )
        states, days = trace_days(jax, micro, trace)
        cotangent_days = [jax.tree_util.tree_map(lambda v: v[k], cotangents) for k in range(len(days))]

        def progress(event, data):
            if event == "reverse_memory":
                report["one_day_reverse_memory"] = data
            report["reverse_progress"] = {"window": i, **data}
            emit(f"window_{i}_{event}", report)

        current = reverse.gradient(values, fixed, states, days, cotangent_days, callback=progress)
        gradient = reverse.add(gradient, current)
        reference = float(window.linearized_science_window_loss(outputs[i][0], total, flux_basis=basis))
        report["windows"].append({
            "window": i, "observation_linearized_score": float(loss), "production_linearized_score": reference,
            "linearized_score_agrees": bool(np.isclose(float(loss), reference, atol=1e-6, rtol=1e-5)),
            "gradient_l2": np.sqrt(streamed.dot(jax, current, current)),
        })
        emit(f"window_{i}_gradient_complete", report)
        del trace, current, states, days, cotangents, cotangent_days
    norm = np.sqrt(streamed.dot(jax, gradient, gradient))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("summed streamed gradient is nonfinite or zero")
    direction = jax.tree_util.tree_map(lambda g, v: (-g / norm).astype(v.dtype), gradient, values)
    report["gradient_l2"] = norm
    report["reverse_compile_count"] = reverse.compile_count
    report["baseline_repeat"], _, _ = evaluate(values)
    emit("baseline_repeat", report)
    for epsilon in EPSILONS:
        pair = []
        for sign in (1, -1):
            variant = window.displaced(jax, values, direction, sign * epsilon)
            row, _, _ = evaluate(variant)
            delta = jax.tree_util.tree_map(lambda a, b: a.astype(jnp.float64) - b.astype(jnp.float64), variant, values)
            row.update(
                epsilon=epsilon, sign=sign, actual_change=row["loss"] - baseline["loss"],
                predicted_change=streamed.dot(jax, gradient, delta),
                actual_delta_l2=np.sqrt(streamed.dot(jax, delta, delta)),
            )
            report["evaluations"].append(row)
            pair.append(row)
            emit(f"finite_step_{epsilon}_{sign}", report)
        actual_direction = (pair[0]["predicted_change"] - pair[1]["predicted_change"]) / (2 * epsilon)
        secant = (pair[0]["loss"] - pair[1]["loss"]) / (2 * epsilon)
        report["finite_differences"].append({
            "epsilon": epsilon, "central_secant": secant, "gradient_dot_actual_direction": actual_direction,
            "relative_derivative_error": abs(secant - actual_direction) / max(abs(secant), abs(actual_direction), 1e-30),
        })
    report["admission"] = classify(report)
    return report


def run(args):
    started = time.perf_counter()
    report = {
        "schema_version": "conditioned_attention_streamed_step_v1", "status": "initializing",
        "stage": "bootstrap", "artifact_update": 442, "horizon": 30, "window_count": 4,
        "policy": {"retained_updates": 0, "monitor_evaluated": False, "teacher_regenerated": False, "truncated_gradient_days": 0},
    }

    def emit(stage, updates):
        report.update(updates)
        report["stage"], report["elapsed_seconds"] = stage, time.perf_counter() - started
        window._atomic_json(args.output, report)
        print(f"streamed_step stage={stage} elapsed={report['elapsed_seconds']:.1f}s", flush=True)

    print("streamed_step stage=bootstrap", flush=True)
    try:
        parent = json.loads((args.conditioned_run_dir / "report.json").read_text())
        recovered = window.load_recovery(args.conditioned_run_dir / "recovery.pkl", identity=parent["identity"], parameter_count=window.TRAINABLE_PARAMETER_COUNT)
        if recovered is None or recovered["completed_logical_update"] != 3:
            raise ValueError("streamed step requires accepted update-3 recovery")
        _, original = window._load_parent_state(
            args.parent_persistent_training_report, args.parent_artifact_aware_recovery,
            persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())["contract_sha256"],
        )
        bootstrap = window._conditioned_bootstrap(
            args, recovered=original,
            conditioning_contract=window.load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract),
        )
        jax = bootstrap["context"]["runtime"]["jax"]
        report["status"], report["parent_identity"] = "running", parent["identity"]
        emit("load_artifact", {})
        artifact = window._load_artifact(args, bootstrap["context"], 442)
        result = experiment(bootstrap, jax.device_put(recovered["trainable"]), artifact, emit=emit)
        emit("completed", {**result, "status": "completed"})
    except Exception as exc:
        emit(report["stage"], {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    return report


def _parser():
    return window._parser()


if __name__ == "__main__":
    run(_parser().parse_args())
