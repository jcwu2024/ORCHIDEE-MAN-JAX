"""Day/state localization of window 442/0; no parameter retention or training."""

from __future__ import annotations

import json
import time

import numpy as np

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention import (
    load_coupled_gpp_conditioned_attention_contract,
    replace_conditioned_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention_discarded_update import (
    TRAINABLE_PARAMETER_COUNT,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention_persistence import (
    load_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_science_window_sufficient_statistics,
    linearized_science_window_loss,
    science_window_sufficient_statistics,
)
from scripts.hpc.audit_conditioned_attention_update4_derivative import _parser, tree_dot
from scripts.hpc.audit_target_oriented_coupled_gpp_conditioned_attention_discarded_update import (
    _conditioned_bootstrap,
    _load_parent_state,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_localized_jump import _make_trace_evaluator, _state_location
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_full_step_preflight import _hard_failure_count, _slice
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import _hard_host

EPSILONS = (1e-3, 1e-4, 1e-5, 1e-6)


def displaced(jax, values, tangent, epsilon):
    return jax.tree_util.tree_map(
        lambda a, b: (a.astype("float64") + epsilon * b.astype("float64")).astype(a.dtype), values, tangent
    )


def derivative_comparison(tangent, plus, minus, epsilon, scale):
    tangent = np.asarray(tangent, dtype=np.float64)
    secant = (np.asarray(plus, dtype=np.float64) - np.asarray(minus, dtype=np.float64)) / (2 * epsilon)
    scale = np.asarray(scale, dtype=np.float64)
    error = (tangent - secant) / scale
    flat = int(np.argmax(np.abs(error)))
    return {
        "jvp_scaled_l2": float(np.linalg.norm(tangent / scale)),
        "secant_scaled_l2": float(np.linalg.norm(secant / scale)),
        "error_scaled_l2": float(np.linalg.norm(error)),
        "relative_error": float(
            np.linalg.norm(error) / max(np.linalg.norm(tangent / scale), np.linalg.norm(secant / scale), 1e-30)
        ),
        "worst_flat_index": flat,
        "worst_jvp": float(tangent.flat[flat]),
        "worst_secant": float(secant.flat[flat]),
        "worst_scaled_error": float(error.flat[flat]),
        "nonfinite_count": int(np.sum(~np.isfinite(tangent)) + np.sum(~np.isfinite(secant))),
    }


def select_local_days(rows):
    # Diagnostic ranking only: require both a relative and an absolute discrepancy.
    material = [i for i, r in enumerate(rows) if r["relative_error"] > 0.1 and r["error_scaled_l2"] > 1e-3]
    if not material:
        return [int(np.argmax([r["error_scaled_l2"] for r in rows]))]
    first = material[0]
    return sorted(set((max(0, first - 1), first, int(np.argmax([r["error_scaled_l2"] for r in rows])))))


def make_payload(bootstrap):
    from research.daily_coarse_graining.daily_operator.successor_training import (
        extract_biomass_productivity_trajectories,
    )

    context = bootstrap["context"]
    runtime, scales = context["runtime"], context["scales"]
    jax = runtime["jax"]
    trace_evaluator = _make_trace_evaluator(bootstrap)

    def evaluate(trainable, fixed, microbatch, total):
        parameters = replace_conditioned_coupled_gpp_trainable(fixed, trainable)
        trace, predicted = trace_evaluator(parameters, microbatch)
        templates = jax.tree_util.tree_map(lambda v: v[0], microbatch.inputs)
        expected = extract_biomass_productivity_trajectories(
            microbatch.expected_next_state.other[0],
            microbatch.expected_next_state_defined.other[0],
            daily_fast_gpp=microbatch.expected_fast_gpp[0],
            daily_fast_gpp_defined=microbatch.expected_fast_gpp_defined[0],
            pft_fraction=templates.pft.pft_fraction,
            active_pft_mask=templates.pft.active_pft_mask,
            contract=runtime["contract"],
        )
        statistics = science_window_sufficient_statistics(
            predicted.fluxes,
            expected.fluxes,
            predicted.stocks,
            expected.stocks,
            flux_scales=scales.fluxes,
            stock_scales=scales.stocks,
            horizon=30,
            flux_basis=bootstrap["flux_basis"],
        )
        return {
            "state": trace.result.next_state.other,
            "gpp": trace.result.prediction.named_outputs["daily.fast_gpp"],
            "score": linearized_science_window_loss(statistics, total, flux_basis=bootstrap["flux_basis"]),
            "fluxes": {k: v.value for k, v in predicted.fluxes.items()},
            "stocks": {k: v.value for k, v in predicted.stocks.items()},
        }

    forward = jax.jit(evaluate)
    tangent = jax.jit(
        lambda values, direction, fixed, batch, total: jax.jvp(
            lambda v: evaluate(v, fixed, batch, total),
            (values,),
            (direction,),
        )
    )
    return forward, tangent, trace_evaluator


def make_local_payload(bootstrap):
    from research.daily_coarse_graining.daily_operator.successor_training import rollout_input_from_template

    context = bootstrap["context"]
    runtime = context["runtime"]
    jax, jnp = runtime["jax"], runtime["jnp"]
    dynamic = bootstrap["steps"]["transition_builder"](context["template"])

    def evaluate(values, state, fixed, template, discrete, day_inputs):
        parameters = replace_conditioned_coupled_gpp_trainable(fixed, values)
        current = rollout_input_from_template(template, state, discrete, runtime["contract"])
        result = dynamic(parameters, current, day_inputs)
        return {
            "state": result.next_state.other,
            "process": {
                k: v for k, v in result.prediction.named_outputs.items() if jnp.issubdtype(v.dtype, jnp.inexact)
            },
        }

    forward = jax.jit(evaluate)
    tangent = jax.jit(
        lambda values, state, direction, state_direction, fixed, template, discrete, day_inputs: jax.jvp(
            lambda v, s: evaluate(v, s, fixed, template, discrete, day_inputs),
            (values, state),
            (direction, state_direction),
        )
    )
    return forward, tangent


def run(args):
    started = time.perf_counter()
    report = {
        "schema_version": "conditioned_attention_window442_v1",
        "status": "initializing",
        "stage": "bootstrap",
        "policy": {"retained_updates": 0, "teacher_regenerated": False, "monitor_evaluated": False},
        "artifact_update": 442,
        "window": 0,
        "probe_direction": "negative_unit_window0_gradient_not_original_update4",
        "finite_differences": [],
        "local_days": [],
    }

    def save(stage):
        report["stage"] = stage
        report["elapsed_seconds"] = time.perf_counter() - started
        _atomic_json(args.output, report)
        print(f"window442 stage={stage} elapsed={report['elapsed_seconds']:.1f}s", flush=True)

    # The inherited initialization-only entry point requires an unused output path.
    print("window442 stage=bootstrap", flush=True)
    try:
        parent = json.loads((args.conditioned_run_dir / "report.json").read_text())
        recovered = load_recovery(
            args.conditioned_run_dir / "recovery.pkl",
            identity=parent["identity"],
            parameter_count=TRAINABLE_PARAMETER_COUNT,
        )
        if recovered is None or recovered["completed_logical_update"] != 3:
            raise ValueError("window442 audit requires exact accepted update-3 recovery")
        _, original = _load_parent_state(
            args.parent_persistent_training_report,
            args.parent_artifact_aware_recovery,
            persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())[
                "contract_sha256"
            ],
        )
        bootstrap = _conditioned_bootstrap(
            args,
            recovered=original,
            conditioning_contract=load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract),
        )
        context = bootstrap["context"]
        jax, jnp = context["runtime"]["jax"], context["runtime"]["jnp"]
        contract = context["runtime"]["contract"]
        trainable, fixed = jax.device_put(recovered["trainable"]), bootstrap["fixed_parameters"]
        parameters = replace_conditioned_coupled_gpp_trainable(fixed, trainable)
        artifact = _load_artifact(args, context, 442)
        report["sample_identity"] = context["schedule"]["entries"][441]
        micro = _slice(artifact.batch.thirty_day, 0, 1)
        save("baseline_four_windows")
        outputs = [
            bootstrap["steps"]["thirty_forward"](parameters, _slice(artifact.batch.thirty_day, i, i + 1))
            for i in range(4)
        ]
        jax.block_until_ready(outputs)
        total = outputs[0][0]
        for value in outputs[1:]:
            total = add_science_window_sufficient_statistics(total, value[0])
        report["baseline_hard_failure_count"] = sum(
            _hard_failure_count(_hard_host(jax.device_get(v[1]))) for v in outputs
        )
        save("window0_reverse_gradient")
        primal, gradient = bootstrap["gradient_steps"]["thirty_gradient"](trainable, micro, total)
        jax.block_until_ready((primal, gradient))
        norm = np.sqrt(tree_dot(jax, gradient, gradient))
        direction = jax.tree_util.tree_map(lambda g, p: (-g / norm).astype(p.dtype), gradient, trainable)
        report["reverse"] = {
            "primal": float(primal),
            "gradient_l2_float64_reduction": norm,
            "gradient_dot_probe": tree_dot(jax, gradient, direction),
            "historical_gradient_l2_float32": 1097.91845703125,
        }
        if not np.isfinite(norm) or norm <= 0 or report["baseline_hard_failure_count"]:
            raise RuntimeError("nonfinite/zero derivative or baseline physical failure")
        save("window0_forward_tangent")
        forward, tangent, trace_evaluator = make_payload(bootstrap)
        baseline, derivatives = tangent(trainable, direction, fixed, micro, total)
        jax.block_until_ready((baseline, derivatives))
        trace, _ = trace_evaluator(parameters, micro)
        jax.block_until_ready(trace)
        baseline_host, derivative_host = jax.device_get((baseline, derivatives))
        report["tangent"] = {
            "primal": float(baseline_host["score"]),
            "score_jvp": float(derivative_host["score"]),
            "reverse_minus_forward_primal": float(primal) - float(baseline_host["score"]),
        }
        state_scale = np.asarray(context["scales"].state.other, dtype=np.float64)
        state_mask = np.asarray(trace.result.next_state_defined.other, dtype=bool)
        save("window0_finite_differences")
        selected_rows = None
        for epsilon in EPSILONS:
            plus_values, minus_values = (
                displaced(jax, trainable, direction, epsilon),
                displaced(jax, trainable, direction, -epsilon),
            )
            plus, minus = forward(plus_values, fixed, micro, total), forward(minus_values, fixed, micro, total)
            jax.block_until_ready((plus, minus))
            plus, minus = jax.device_get((plus, minus))
            delta = jax.tree_util.tree_map(
                lambda a, b: (a.astype(jnp.float64) - b.astype(jnp.float64)) / (2 * epsilon), plus_values, minus_values
            )
            rows = []
            for day in range(30):
                row = derivative_comparison(
                    np.where(state_mask[day], derivative_host["state"][day], 0),
                    np.where(state_mask[day], plus["state"][day], 0),
                    np.where(state_mask[day], minus["state"][day], 0),
                    epsilon,
                    state_scale,
                )
                row.update(
                    day=day + 1,
                    location=_state_location(contract, row["worst_flat_index"]),
                    baseline_state=float(baseline_host["state"][day, row["worst_flat_index"]]),
                    plus_state=float(plus["state"][day, row["worst_flat_index"]]),
                    minus_state=float(minus["state"][day, row["worst_flat_index"]]),
                    gpp=derivative_comparison(
                        derivative_host["gpp"][day], plus["gpp"][day], minus["gpp"][day], epsilon, 1
                    ),
                )
                rows.append(row)
            cell = {
                "epsilon": epsilon,
                "score": derivative_comparison(derivative_host["score"], plus["score"], minus["score"], epsilon, 1),
                "reverse_dot_actual_central_direction": tree_dot(jax, gradient, delta),
                "science": {
                    family: {
                        name: derivative_comparison(
                            derivative_host[family][name], plus[family][name], minus[family][name], epsilon, 1
                        )
                        for name in derivative_host[family]
                    }
                    for family in ("fluxes", "stocks")
                },
                "days": rows,
            }
            report["finite_differences"].append(cell)
            if epsilon == 1e-5:
                selected_rows = rows
            save(f"finite_difference_{epsilon}")
        local_forward, local_tangent = make_local_payload(bootstrap)
        templates = jax.tree_util.tree_map(lambda v: v[0], micro.inputs)
        day_inputs = jax.tree_util.tree_map(lambda v: v[0], micro.day_inputs)
        for day in select_local_days(selected_rows):
            template = jax.tree_util.tree_map(lambda v: v[day], templates)
            state = templates.continuous_state[0] if day == 0 else trace.result.next_state.other[day - 1]
            discrete = (
                jax.tree_util.tree_map(lambda v: v[0], templates.discrete_state)
                if day == 0
                else jax.tree_util.tree_map(lambda v: v[day - 1], trace.result.next_discrete_state)
            )
            state_direction = jnp.zeros_like(state) if day == 0 else derivatives["state"][day - 1]
            save(f"local_day_{day + 1}_tangent")
            local_base, local_derivative = local_tangent(
                trainable, state, direction, state_direction, fixed, template, discrete, day_inputs
            )
            jax.block_until_ready((local_base, local_derivative))
            cells = []
            for epsilon in EPSILONS:
                plus = local_forward(
                    displaced(jax, trainable, direction, epsilon),
                    state + epsilon * state_direction,
                    fixed,
                    template,
                    discrete,
                    day_inputs,
                )
                minus = local_forward(
                    displaced(jax, trainable, direction, -epsilon),
                    state - epsilon * state_direction,
                    fixed,
                    template,
                    discrete,
                    day_inputs,
                )
                jax.block_until_ready((plus, minus))
                d, p, m = jax.device_get((local_derivative, plus, minus))
                state_row = derivative_comparison(d["state"], p["state"], m["state"], epsilon, state_scale)
                state_row["location"] = _state_location(contract, state_row["worst_flat_index"])
                processes = {
                    k: derivative_comparison(d["process"][k], p["process"][k], m["process"][k], epsilon, 1)
                    for k in d["process"]
                    if np.asarray(d["process"][k]).size
                }
                cells.append({"epsilon": epsilon, "state": state_row, "process": processes})
            report["local_days"].append(
                {
                    "day": day + 1,
                    "state_primal_max_error_vs_scan": float(
                        jnp.max(jnp.abs(local_base["state"] - baseline["state"][day]))
                    ),
                    "state_jvp_max_error_vs_scan": float(
                        jnp.max(jnp.abs(local_derivative["state"] - derivatives["state"][day]))
                    ),
                    "comparisons": cells,
                }
            )
            save(f"local_day_{day + 1}_completed")
        report["status"] = "completed"
        save("completed")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save(report["stage"])
        raise
    return report


if __name__ == "__main__":
    run(_parser().parse_args())
