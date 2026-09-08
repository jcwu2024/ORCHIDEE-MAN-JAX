"""Locate streamed/production score differences on two fixed seven-day windows."""

from __future__ import annotations

import json
import time

import numpy as np

from scripts.hpc import audit_conditioned_attention_streamed_integration as integration

window = integration.window
streamed = integration.streamed
step_audit = integration.step_audit
ARTIFACT = 322
WINDOWS = (5, 1)  # Original callback 37 fails most; callback 33 passes.
EPSILONS = (1e-4, 1e-5)


def statistics_rows(left, right):
    return {
        group: {
            name: {
                field: {"left": float(a), "right": float(b), "difference": float(a) - float(b)}
                for field, a, b in zip(value._fields, value, getattr(right, group)[name], strict=True)
            }
            for name, value in getattr(left, group).items()
        }
        for group in ("fluxes", "stocks")
    }


def trajectory_rows(left, right):
    rows = {}
    for group in ("fluxes", "stocks"):
        for name, a in getattr(left, group).items():
            b = getattr(right, group)[name]
            av, bv = np.asarray(a.value), np.asarray(b.value)
            am, bm = np.asarray(a.defined), np.asarray(b.defined)
            active = am & bm
            difference = np.where(active, av - bv, 0.0)
            rows[f"{group}.{name}"] = {
                "left": av.tolist(), "right": bv.tolist(),
                "left_defined": am.tolist(), "right_defined": bm.tolist(),
                "mask_mismatches": int(np.sum(am != bm)),
                "nonfinite_on_common_defined": int(np.sum(active & (~np.isfinite(av) | ~np.isfinite(bv)))),
                "max_abs_difference": float(np.max(np.abs(difference))),
            }
    return rows


def summarize_scores(scores):
    reference = scores["production"]
    return {name: {"value": value, "difference_from_production": value - reference,
                   "agrees_with_production": bool(np.isclose(value, reference, atol=1e-6, rtol=1e-5))}
            for name, value in scores.items()}


def make_extractors(bootstrap):
    context = bootstrap["context"]
    jax = context["runtime"]["jax"]
    scales = context["scales"]

    def statistics(predicted, expected):
        return window.science_window_sufficient_statistics(
            predicted.fluxes, expected.fluxes, predicted.stocks, expected.stocks,
            flux_scales=scales.fluxes, stock_scales=scales.stocks,
            horizon=7, flux_basis=bootstrap["flux_basis"],
        )

    def compact(parameters, micro):
        predicted, expected, hard = bootstrap["steps"]["window_trajectories"](
            parameters, micro, rematerialize=False,
        )
        return predicted, expected, statistics(predicted, expected), hard

    return jax.jit(compact), jax.jit(statistics)


def experiment(bootstrap, values, artifact, *, emit):
    jax, jnp = (bootstrap["context"]["runtime"][k] for k in ("jax", "jnp"))
    fixed, basis = bootstrap["fixed_parameters"], bootstrap["flux_basis"]
    parameters = window.replace_conditioned_coupled_gpp_trainable(fixed, values)
    micros = [window._slice(artifact.batch.seven_day, i, i + 1) for i in range(8)]
    forward = bootstrap["steps"]["seven_forward"]
    outputs = [forward(parameters, micro) for micro in micros]
    jax.block_until_ready(outputs)
    total = step_audit.merge_statistics([v[0] for v in outputs])
    report = {"windows": [], "artifact_update": ARTIFACT, "global_statistics_window_count": 8}
    trace_forward = window._make_trace_evaluator(bootstrap)
    compact, statistics = make_extractors(bootstrap)
    score_gradient = integration.make_observation_gradient(bootstrap, 7)
    reverse = integration.StreamedAdjoint(jax, jnp, streamed.make_step(bootstrap))

    def score(stats):
        return float(window.linearized_science_window_loss(stats, total, flux_basis=basis))

    for index in WINDOWS:
        micro = micros[index]
        row = {"window": index, "original_callback_ordinal": 32 + index}
        report["windows"].append(row)
        emit("window_start", report)
        trace, verbose_predicted = trace_forward(parameters, micro)
        compact_predicted, expected, compact_stats, compact_hard = compact(parameters, micro)
        verbose_stats = statistics(verbose_predicted, expected)
        compact_separated_stats = statistics(compact_predicted, expected)
        obs_loss, cotangents = score_gradient(
            streamed.observations(trace), trace.result.next_state_defined.other,
            trace.active_pft_mask, total, micro,
        )
        row["scores"] = summarize_scores({
            "production": score(outputs[index][0]), "observation_ad": float(obs_loss),
            "verbose_trace_separated": score(verbose_stats), "compact_fused": score(compact_stats),
            "compact_separated": score(compact_separated_stats),
        })
        row["verbose_vs_production_statistics"] = statistics_rows(verbose_stats, outputs[index][0])
        row["compact_vs_production_statistics"] = statistics_rows(compact_stats, outputs[index][0])
        row["verbose_vs_compact_trajectories"] = trajectory_rows(verbose_predicted, compact_predicted)
        row["trace_pft_fraction"] = np.asarray(trace.pft_fraction).tolist()
        row["trace_active_pft_mask"] = np.asarray(trace.active_pft_mask).tolist()
        row["hard_failure_count"] = sum(
            window._hard_failure_count(window._hard_host(jax.device_get(v))) for v in (outputs[index][1], compact_hard)
        )
        emit("window_score_paths", report)

        states, days = step_audit.trace_days(jax, micro, trace)
        cotangent_days = [jax.tree_util.tree_map(lambda v: v[i], cotangents) for i in range(7)]

        def progress(event, data):
            report["reverse_progress"] = {"window": index, **data}
            emit(event, report)

        gradient = reverse.gradient(values, fixed, states, days, cotangent_days, callback=progress)
        norm = np.sqrt(streamed.dot(jax, gradient, gradient))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("nonfinite or zero diagnostic gradient")
        direction = jax.tree_util.tree_map(lambda g, v: (-g / norm).astype(v.dtype), gradient, values)
        row["gradient_l2"] = norm
        row["finite_differences"] = []
        for epsilon in EPSILONS:
            pair = []
            for sign in (1, -1):
                displaced = window.displaced(jax, values, direction, sign * epsilon)
                variant = window.replace_conditioned_coupled_gpp_trainable(fixed, displaced)
                stats, hard = forward(variant, micro)
                varied_trace, _ = trace_forward(variant, micro)
                observed, _ = score_gradient(
                    streamed.observations(varied_trace), varied_trace.result.next_state_defined.other,
                    varied_trace.active_pft_mask, total, micro,
                )
                delta = jax.tree_util.tree_map(lambda a, b: a.astype(jnp.float64) - b.astype(jnp.float64), displaced, values)
                pair.append({"sign": sign, "production_score": score(stats), "observation_score": float(observed),
                             "gradient_dot_actual_delta": streamed.dot(jax, gradient, delta),
                             "hard_failure_count": window._hard_failure_count(window._hard_host(jax.device_get(hard)))})
            projected = (pair[0]["gradient_dot_actual_delta"] - pair[1]["gradient_dot_actual_delta"]) / (2 * epsilon)
            comparison = {"epsilon": epsilon, "evaluations": pair, "gradient_dot_actual_direction": projected}
            for name in ("production", "observation"):
                secant = (pair[0][f"{name}_score"] - pair[1][f"{name}_score"]) / (2 * epsilon)
                comparison[name] = {"central_secant": secant,
                                    "relative_derivative_error": abs(secant - projected) / max(abs(secant), abs(projected), 1e-30)}
            row["finite_differences"].append(comparison)
            emit("window_finite_difference", report)
        emit("window_completed", report)
    report["reverse_compile_count"] = reverse.compile_count
    return report


def run(args):
    started = time.perf_counter()
    report = {"schema_version": "conditioned_attention_seven_score_v1", "status": "initializing",
              "stage": "bootstrap", "policy": {"retained_updates": 0, "monitor_evaluated": False,
                                                "teacher_regenerated": False}}

    def emit(stage, updates):
        report.update(updates)
        report["stage"], report["elapsed_seconds"] = stage, time.perf_counter() - started
        window._atomic_json(args.output, report)
        print(f"seven_score stage={stage} elapsed={report['elapsed_seconds']:.1f}s", flush=True)

    print("seven_score stage=bootstrap", flush=True)
    try:
        parent = json.loads((args.conditioned_run_dir / "report.json").read_text())
        recovered = window.load_recovery(args.conditioned_run_dir / "recovery.pkl", identity=parent["identity"], parameter_count=window.TRAINABLE_PARAMETER_COUNT)
        if recovered is None or recovered["completed_logical_update"] != 3:
            raise ValueError("seven-day score audit requires accepted update 3")
        _, original = window._load_parent_state(
            args.parent_persistent_training_report, args.parent_artifact_aware_recovery,
            persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())["contract_sha256"],
        )
        bootstrap = window._conditioned_bootstrap(
            args, recovered=original, conditioning_contract=window.load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract),
        )
        emit("load_artifact", {"status": "running", "parent_identity": parent["identity"]})
        artifact = window._load_artifact(args, bootstrap["context"], ARTIFACT)
        jax = bootstrap["context"]["runtime"]["jax"]
        result = experiment(bootstrap, jax.device_put(recovered["trainable"]), artifact, emit=emit)
        emit("completed", {**result, "status": "completed", "classification": "diagnostic_complete_no_model_admission"})
    except Exception as exc:
        emit(report["stage"], {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    return report


if __name__ == "__main__":
    run(window._parser().parse_args())
