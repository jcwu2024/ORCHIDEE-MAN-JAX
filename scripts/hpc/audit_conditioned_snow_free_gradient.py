"""Full corrected-head streamed/JVP/FD verification; no training or retention."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from scripts.hpc import audit_conditioned_snow_free_rollout as screen

integration = screen.integration
window = screen.window
streamed = integration.streamed
step_audit = integration.step_audit
EPSILONS = (1e-4, 1e-5)


def comparison(left, right):
    finite = bool(np.isfinite(left) and np.isfinite(right))
    gap = abs(left - right)
    relative = gap / max(abs(left), abs(right), 1e-30)
    return {"left": left, "right": right, "absolute_error": gap, "relative_error": relative,
            "passed": finite and (gap <= 1e-6 or relative <= 0.01)}


def effective_direction(jax, plus, minus, values, epsilon):
    return jax.tree_util.tree_map(
        lambda a, b, v: ((a.astype("float64") - b.astype("float64")) / (2 * epsilon)).astype(v.dtype),
        plus, minus, values,
    )


def tangent_projection(jax, jnp, tangent, values, direction, fixed, states, days, cotangents,
                       observations, defined, state_scales, *, emit):
    incoming = jnp.zeros_like(states[0])
    projection, nonfinite, max_gap = 0., 0, 0.
    rows = []
    for i, (state, day, cotangent) in enumerate(zip(states, days, cotangents, strict=True)):
        primal, response = tangent(values, state, direction, incoming, fixed, day)
        jax.block_until_ready((primal, response))
        projection += streamed.dot(jax, cotangent, response)
        active = np.asarray(defined[i])
        reference, actual = np.asarray(observations["state"][i]), np.asarray(primal["state"])
        state_tangent = np.asarray(response["state"])
        nonfinite += int(np.sum(active & (~np.isfinite(actual) | ~np.isfinite(state_tangent))))
        common = active & np.isfinite(reference) & np.isfinite(actual)
        delta = np.subtract(actual, reference, out=np.zeros_like(actual), where=common)
        scale = np.asarray(state_scales)
        scaled = np.divide(np.abs(delta), scale, out=np.zeros_like(delta),
                           where=common & np.isfinite(scale) & (scale > 0))
        index = int(np.argmax(scaled))
        max_gap = max(max_gap, float(scaled[index]))
        rows.append({"day": i + 1, "max_scaled_primal_gap": float(scaled[index]), "state_index": index})
        incoming = response["state"]
        emit("tangent_day", {"day": i + 1})
    return {"projection": projection, "nonfinite_active": nonfinite,
            "max_scaled_primal_gap": max_gap, "primal_days": rows}


def selected_days(bootstrap, micro, trace):
    jax = bootstrap["context"]["runtime"]["jax"]
    starts = np.concatenate((np.asarray(micro.inputs.continuous_state)[0, :1],
                             np.asarray(trace.result.next_state.other)[:-1]), axis=0)
    forcing = jax.tree_util.tree_map(lambda v: v[0], micro.inputs.forcing)
    return np.asarray(screen.warm_snow_free_regime(starts, forcing,
        bootstrap["context"]["runtime"]["contract"], bootstrap["head"].forcing_layout))


def same_masks(jax, left, right):
    return all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(left),
                                                  jax.tree_util.tree_leaves(right), strict=True))


def classify_case(row):
    checks = [pair for direction in row["directions"] for pair in direction["pairs"]]
    return bool(row["score_agrees"] and row["hard_failure_count"] == 0 and row["all_days_selected"]
        and row["gradient_finite"] and len(row["directions"]) == row["expected_direction_count"]
        and all(len(d["pairs"]) == len(EPSILONS) for d in row["directions"])
        and all(pair["passed"] for pair in checks))


def experiment(bootstrap, values, artifacts, historical_direction, *, emit):
    context = bootstrap["context"]
    jax, jnp = context["runtime"]["jax"], context["runtime"]["jnp"]
    fixed, basis = bootstrap["fixed_parameters"], bootstrap["flux_basis"]
    parameters = window.replace_conditioned_coupled_gpp_trainable(fixed, values)
    forward = {7: bootstrap["steps"]["seven_forward"], 30: bootstrap["steps"]["thirty_forward"]}
    trace_forward = integration.make_compact_trace_evaluator(bootstrap)
    observation_gradient = {h: integration.make_observation_gradient(bootstrap, h) for h in (7, 30)}
    step = streamed.make_step(bootstrap)
    reverse = integration.StreamedAdjoint(jax, jnp, step)
    tangent = streamed.make_day_kernels(jax, step)["tangent"]
    panel = {}
    report = {"cases": [], "warm_snow_free_boundary": bool(bootstrap["head"].warm_snow_free_boundary)}
    for update, horizon in ((322, 7), (442, 30)):
        batch = getattr(artifacts[update].batch, "seven_day" if horizon == 7 else "thirty_day")
        micros = [window._slice(batch, i, i + 1) for i in range(len(batch.inputs.continuous_state))]
        emit("corrected_panel_forward", report)
        outputs = [forward[horizon](parameters, micro) for micro in micros]
        jax.block_until_ready(outputs)
        panel[update] = (micros, outputs, step_audit.merge_statistics([v[0] for v in outputs]))
    for update, horizon, index in screen.CASES:
        micros, outputs, total = panel[update]
        micro = micros[index]
        def score(stats):
            return float(window.linearized_science_window_loss(stats, total, flux_basis=basis))
        production = score(outputs[index][0])
        row = {"artifact": update, "horizon": horizon, "window": index, "directions": [],
               "production_score": production, "expected_direction_count": 2 if horizon == 30 else 1,
               "science": screen.science_summary(outputs[index][0], basis),
               "hard_failure_count": window._hard_failure_count(window._hard_host(outputs[index][1]))}
        report["cases"].append(row)
        emit("canonical_trace", report)
        trace, _ = trace_forward(parameters, micro)
        observations = streamed.observations(trace)
        observed, cotangents = observation_gradient[horizon](observations, trace.result.next_state_defined.other,
                                                            trace.active_pft_mask, total, micro)
        row["observation_score"] = float(observed)
        row["score_agrees"] = bool(np.isclose(float(observed), production, atol=1e-6, rtol=1e-5))
        row["all_days_selected"] = bool(np.all(selected_days(bootstrap, micro, trace)))
        states, days = step_audit.trace_days(jax, micro, trace)
        cotangent_days = [jax.tree_util.tree_map(lambda v: v[i], cotangents) for i in range(horizon)]

        def progress(event, values):
            report["progress"] = {"artifact": update, "window": index, **values}
            emit(event, report)

        gradient = reverse.gradient(values, fixed, states, days, cotangent_days, callback=progress)
        norm = np.sqrt(streamed.dot(jax, gradient, gradient))
        row["gradient_l2"] = float(norm)
        row["gradient_finite"] = bool(np.isfinite(norm) and norm > 0)
        if not row["gradient_finite"]:
            raise RuntimeError("nonfinite or zero full corrected gradient")
        directions = [("negative_unit_gradient", jax.tree_util.tree_map(lambda g, p: (-g / norm).astype(p.dtype), gradient, values))]
        if horizon == 30:
            directions.append(("historical_442_probe", historical_direction))
        for name, direction in directions:
            direction_row = {"name": name, "pairs": []}
            row["directions"].append(direction_row)
            for epsilon in EPSILONS:
                variants, evaluations = [], []
                for sign in (1., -1.):
                    varied = window.displaced(jax, values, direction, sign * epsilon)
                    variants.append(varied)
                    parameter_variant = window.replace_conditioned_coupled_gpp_trainable(fixed, varied)
                    stats, hard = forward[horizon](parameter_variant, micro)
                    varied_trace, _ = trace_forward(parameter_variant, micro)
                    observation, _ = observation_gradient[horizon](streamed.observations(varied_trace),
                        varied_trace.result.next_state_defined.other, varied_trace.active_pft_mask, total, micro)
                    evaluation = {"sign": sign, "production_score": score(stats), "observation_score": float(observation),
                        "hard_failure_count": window._hard_failure_count(window._hard_host(hard)),
                        "masks_unchanged": same_masks(jax,
                            (trace.result.next_state_defined, trace.result.next_discrete_state, trace.active_pft_mask),
                            (varied_trace.result.next_state_defined, varied_trace.result.next_discrete_state, varied_trace.active_pft_mask)),
                        "all_days_selected": bool(np.all(selected_days(bootstrap, micro, varied_trace)))}
                    evaluation["score_agrees"] = bool(np.isclose(evaluation["production_score"], float(observation), atol=1e-6, rtol=1e-5))
                    evaluations.append(evaluation)
                    progress("finite_difference_forward", {"direction": name, "epsilon": epsilon, "sign": sign})
                effective = effective_direction(jax, *variants, values, epsilon)
                reverse_projection = streamed.dot(jax, gradient, effective)
                jvp = tangent_projection(jax, jnp, tangent, values, effective, fixed, states, days, cotangent_days,
                                        observations, trace.result.next_state_defined.other, context["scales"].state.other,
                                        emit=progress)
                secant = (evaluations[0]["production_score"] - evaluations[1]["production_score"]) / (2 * epsilon)
                pair = {"epsilon": epsilon, "evaluations": evaluations, "jvp": jvp,
                        "reverse_projection": reverse_projection, "central_secant": secant,
                        "jvp_reverse": comparison(jvp["projection"], reverse_projection),
                        "reverse_fd": comparison(reverse_projection, secant), "jvp_fd": comparison(jvp["projection"], secant),
                        "positive_step_change": evaluations[0]["production_score"] - production}
                pair["passed"] = bool(jvp["nonfinite_active"] == 0
                    and all(pair[k]["passed"] for k in ("jvp_reverse", "reverse_fd", "jvp_fd"))
                    and all(e["hard_failure_count"] == 0 and e["masks_unchanged"] and e["all_days_selected"] and e["score_agrees"] for e in evaluations)
                    and (name != "negative_unit_gradient" or (pair["positive_step_change"] < 0 and reverse_projection < 0)))
                direction_row["pairs"].append(pair)
                emit("direction_pair_completed", report)
        row["passed"] = classify_case(row)
        emit("case_completed", report)
    report.update(reverse_compile_count=reverse.compile_count, reverse_memory=reverse.memory,
                  passed=len(report["cases"]) == len(screen.CASES) and all(r["passed"] for r in report["cases"]))
    return report


def run(args):
    started = time.perf_counter()
    report = {"schema_version": "conditioned_snow_free_gradient_v1", "status": "initializing",
              "policy": {"retained_updates": 0, "monitor_evaluated": False, "teacher_regenerated": False,
                         "truncated_gradient_days": 0, "warm_snow_free_boundary": True}}

    def emit(stage, updates):
        report.update(updates)
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started)
        window._atomic_json(args.output, report)
        print(f"snow_gradient stage={stage} elapsed={report['elapsed_seconds']:.1f}s", flush=True)

    print("snow_gradient stage=bootstrap", flush=True)
    try:
        parent = json.loads((args.conditioned_run_dir / "report.json").read_text())
        recovered = window.load_recovery(args.conditioned_run_dir / "recovery.pkl", identity=parent["identity"],
                                         parameter_count=window.TRAINABLE_PARAMETER_COUNT)
        with args.probe.open("rb") as handle:
            probe = pickle.load(handle)
        if (recovered is None or recovered["completed_logical_update"] != 3
                or probe["parent_identity"] != parent["identity"]
                or (probe["completed_logical_update"], probe["artifact_update"], probe["window"]) != (3, 442, 0)):
            raise ValueError("corrected gradient audit requires accepted update 3 and its saved probe")
        _, original = window._load_parent_state(args.parent_persistent_training_report, args.parent_artifact_aware_recovery,
            persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())["contract_sha256"])
        bootstrap = window._conditioned_bootstrap(args, recovered=original, warm_snow_free_boundary=True,
            conditioning_contract=window.load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract))
        emit("load_artifacts", {"status": "running", "parent_identity": parent["identity"], "probe_path": str(args.probe)})
        artifacts = {u: window._load_artifact(args, bootstrap["context"], u) for u in (322, 442)}
        jax = bootstrap["context"]["runtime"]["jax"]
        result = experiment(bootstrap, jax.device_put(recovered["trainable"]), artifacts,
                            jax.device_put(probe["direction"]), emit=emit)
        emit("completed", {**result, "status": "completed", "classification": "corrected_network_derivative_gate_no_model_promotion"})
    except Exception as exc:
        emit(report.get("stage", "bootstrap"), {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    return report


def _parser():
    parser = window._parser()
    parser.add_argument("--probe", type=Path, default=streamed.PROBE)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
