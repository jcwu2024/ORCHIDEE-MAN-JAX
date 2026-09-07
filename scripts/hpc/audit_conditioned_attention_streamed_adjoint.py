"""Full-horizon adjoint with one-day derivative graphs and stored daily carries."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from scripts.hpc import audit_conditioned_attention_window442 as window

PROBE = Path(
    "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/conditioned-derivative-modes-2cb51af.probe.pkl"
)


def dot(jax, cotangent, tangent):
    # Undefined constant outputs may be NaN, but their zero cotangent contributes nothing.
    return float(
        sum(
            np.sum(np.asarray(a, dtype=np.float64) * np.where(np.asarray(a) != 0, np.asarray(b, dtype=np.float64), 0.0))
            for a, b in zip(jax.tree_util.tree_leaves(cotangent), jax.tree_util.tree_leaves(tangent), strict=True)
        )
    )


def make_day_kernels(jax, step):
    def tangent(values, state, direction, state_direction, fixed, day):
        return jax.jvp(lambda v, s: step(v, s, fixed, day), (values, state), (direction, state_direction))

    def reverse(values, state, fixed, day, cotangent):
        primal, pullback = jax.vjp(lambda v, s: step(v, s, fixed, day), values, state)
        parameter_gradient, state_gradient = pullback(cotangent)
        return primal, parameter_gradient, state_gradient

    return {"forward": jax.jit(step), "tangent": jax.jit(tangent), "reverse": jax.jit(reverse)}


def memory_requirement(compiled, limit_bytes):
    memory = compiled.memory_analysis()
    if memory is None:
        raise RuntimeError("compiled one-day memory estimate unavailable")
    sizes = {
        k: int(getattr(memory, k))
        for k in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")
    }
    sizes["estimated_peak_bytes"] = (
        sizes["argument_size_in_bytes"]
        + sizes["output_size_in_bytes"]
        + sizes["temp_size_in_bytes"]
        - sizes["alias_size_in_bytes"]
    )
    if sizes["estimated_peak_bytes"] > limit_bytes:
        raise RuntimeError(
            f"one-day derivative requires {sizes['estimated_peak_bytes']} bytes; exceeds {limit_bytes} byte diagnostic envelope"
        )
    return sizes


def streamed_derivatives(
    jax, jnp, kernels, values, direction, fixed, states, days, cotangents, *, callback=None, limit_bytes=24 * 1024**3
):
    emit = callback or (lambda event, data: None)
    tangent_state = jnp.zeros_like(states[0])
    primals, tangents, input_tangents = [], [], []
    for day, state in enumerate(states):
        input_tangents.append(tangent_state)
        primal, tangent = kernels["tangent"](values, state, direction, tangent_state, fixed, days[day])
        jax.block_until_ready((primal, tangent))
        primals.append(primal)
        tangents.append(tangent)
        tangent_state = tangent["state"]
        emit("forward_day", {"day": day + 1})
    future = jnp.zeros_like(states[-1])
    accumulated = jax.tree_util.tree_map(lambda v: jnp.zeros_like(v, dtype=jnp.float64), values)
    add = jax.jit(lambda a, b: jax.tree_util.tree_map(lambda x, y: x + y.astype(x.dtype), a, b))
    records, adjoints = [], []
    emit("reverse_compile", {})
    compiled = kernels["reverse"].lower(values, states[-1], fixed, days[-1], cotangents[-1]).compile()
    memory = memory_requirement(compiled, limit_bytes)
    emit("reverse_memory", memory)
    for day in reversed(range(len(states))):
        cotangent = dict(cotangents[day], state=cotangents[day]["state"] + future)
        primal, gradient, incoming = compiled(values, states[day], fixed, days[day], cotangent)
        jax.block_until_ready((primal, gradient, incoming))
        parameter_projection = dot(jax, gradient, direction)
        forward_projection = dot(jax, cotangent, tangents[day])
        state_projection = dot(jax, incoming, input_tangents[day])
        accumulated = add(accumulated, gradient)
        row = {
            "day": day + 1,
            "parameter_projection": parameter_projection,
            "forward_projection": forward_projection,
            "incoming_state_projection": state_projection,
            "duality_gap": forward_projection - parameter_projection - state_projection,
            "parameter_gradient_l2": np.sqrt(dot(jax, gradient, gradient)),
            "incoming_state_gradient_l2": float(jnp.linalg.norm(incoming)),
            "nonfinite_state_gradient": int(jnp.sum(~jnp.isfinite(incoming))),
        }
        records.append(row)
        adjoints.append(incoming)
        future = incoming
        emit("reverse_day", row)
    jax.block_until_ready(accumulated)
    return {
        "gradient": accumulated,
        "records": records,
        "adjoints": list(reversed(adjoints)),
        "primals": primals,
        "tangents": tangents,
        "input_tangents": input_tangents,
        "forward_projection": sum(dot(jax, c, t) for c, t in zip(cotangents, tangents, strict=True)),
        "reverse_projection": dot(jax, accumulated, direction),
        "memory": memory,
    }


def make_step(bootstrap):
    from research.daily_coarse_graining.daily_operator.successor_training import rollout_input_from_template

    context = bootstrap["context"]
    dynamic = bootstrap["steps"]["transition_builder"](context["template"])

    def step(values, state, fixed, day):
        template, discrete, retained_inputs = day
        current = rollout_input_from_template(template, state, discrete, context["runtime"]["contract"])
        result = dynamic(window.replace_conditioned_coupled_gpp_trainable(fixed, values), current, retained_inputs)
        return {
            "state": result.next_state.other,
            "gpp": result.prediction.named_outputs["daily.fast_gpp"],
            "fraction": current.pft.pft_fraction,
        }

    return step


def observations(trace):
    return {
        "state": trace.result.next_state.other,
        "gpp": trace.result.prediction.named_outputs["daily.fast_gpp"],
        "fraction": trace.pft_fraction,
    }


def make_observation_score(bootstrap, micro):
    from research.daily_coarse_graining.daily_operator.successor_training import (
        extract_biomass_productivity_trajectories,
    )

    context = bootstrap["context"]
    jax, jnp = context["runtime"]["jax"], context["runtime"]["jnp"]
    contract, scales = context["runtime"]["contract"], context["scales"]
    templates = jax.tree_util.tree_map(lambda v: v[0], micro.inputs)
    expected = extract_biomass_productivity_trajectories(
        micro.expected_next_state.other[0],
        micro.expected_next_state_defined.other[0],
        daily_fast_gpp=micro.expected_fast_gpp[0],
        daily_fast_gpp_defined=micro.expected_fast_gpp_defined[0],
        pft_fraction=templates.pft.pft_fraction,
        active_pft_mask=templates.pft.active_pft_mask,
        contract=contract,
    )

    def score(obs, defined, active, total):
        predicted = extract_biomass_productivity_trajectories(
            obs["state"],
            defined,
            daily_fast_gpp=obs["gpp"],
            daily_fast_gpp_defined=jnp.broadcast_to(active[:, None, :], obs["gpp"].shape),
            pft_fraction=obs["fraction"],
            active_pft_mask=active,
            contract=contract,
        )
        current = window.science_window_sufficient_statistics(
            predicted.fluxes,
            expected.fluxes,
            predicted.stocks,
            expected.stocks,
            flux_scales=scales.fluxes,
            stock_scales=scales.stocks,
            horizon=30,
            flux_basis=bootstrap["flux_basis"],
        )
        return window.linearized_science_window_loss(current, total, flux_basis=bootstrap["flux_basis"])

    return jax.jit(score), jax.jit(jax.value_and_grad(score))


def run(args):
    started = time.perf_counter()
    report = {
        "schema_version": "conditioned_attention_streamed_adjoint_v1",
        "status": "initializing",
        "stage": "bootstrap",
        "policy": {"retained_updates": 0, "truncated_gradient_days": 0, "monitor_evaluated": False},
        "reverse_days": [],
    }

    def save(stage):
        report["stage"] = stage
        report["elapsed_seconds"] = time.perf_counter() - started
        window._atomic_json(args.output, report)
        print(f"streamed_adjoint stage={stage} elapsed={report['elapsed_seconds']:.1f}s", flush=True)

    print("streamed_adjoint stage=bootstrap", flush=True)
    try:
        parent = json.loads((args.conditioned_run_dir / "report.json").read_text())
        recovered = window.load_recovery(
            args.conditioned_run_dir / "recovery.pkl",
            identity=parent["identity"],
            parameter_count=window.TRAINABLE_PARAMETER_COUNT,
        )
        with args.probe.open("rb") as stream:
            probe = pickle.load(stream)
        if (
            recovered is None
            or recovered["completed_logical_update"] != 3
            or probe["parent_identity"] != parent["identity"]
            or (probe["completed_logical_update"], probe["artifact_update"], probe["window"]) != (3, 442, 0)
        ):
            raise ValueError("streamed adjoint requires matching update-3 window442/0 diagnostic probe")
        _, original = window._load_parent_state(
            args.parent_persistent_training_report,
            args.parent_artifact_aware_recovery,
            persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())[
                "contract_sha256"
            ],
        )
        bootstrap = window._conditioned_bootstrap(
            args,
            recovered=original,
            conditioning_contract=window.load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract),
        )
        context = bootstrap["context"]
        jax, jnp = context["runtime"]["jax"], context["runtime"]["jnp"]
        values, fixed = jax.device_put(recovered["trainable"]), bootstrap["fixed_parameters"]
        direction, total = jax.device_put((probe["direction"], probe["total"]))
        artifact = window._load_artifact(args, context, 442)
        micro = window._slice(artifact.batch.thirty_day, 0, 1)
        parameters = window.replace_conditioned_coupled_gpp_trainable(fixed, values)
        report["status"] = "running"
        report["probe_path"] = str(args.probe)
        save("canonical_forward")
        evaluator = window._make_trace_evaluator(bootstrap)
        trace, _ = evaluator(parameters, micro)
        statistics, hard = bootstrap["steps"]["thirty_forward"](parameters, micro)
        jax.block_until_ready((trace, statistics, hard))
        report["baseline_hard_failures"] = window._hard_failure_count(window._hard_host(jax.device_get(hard)))
        if report["baseline_hard_failures"]:
            raise RuntimeError("baseline physical gate failed")
        score, score_gradient = make_observation_score(bootstrap, micro)
        obs = observations(trace)
        loss, cotangents = score_gradient(obs, trace.result.next_state_defined.other, trace.active_pft_mask, total)
        jax.block_until_ready((loss, cotangents))
        report["baseline_score"] = float(loss)
        report["production_score"] = float(
            window.linearized_science_window_loss(statistics, total, flux_basis=bootstrap["flux_basis"])
        )
        templates = jax.tree_util.tree_map(lambda v: v[0], micro.inputs)
        retained_inputs = jax.tree_util.tree_map(lambda v: v[0], micro.day_inputs)
        if any(
            np.asarray(v).dtype.kind not in "biu" for v in jax.tree_util.tree_leaves(trace.result.next_discrete_state)
        ):
            raise ValueError("differentiable discrete carry would require an additional adjoint")
        states, days = [], []
        for day in range(30):
            states.append(templates.continuous_state[0] if day == 0 else trace.result.next_state.other[day - 1])
            discrete = (
                jax.tree_util.tree_map(lambda v: v[0], templates.discrete_state)
                if day == 0
                else jax.tree_util.tree_map(lambda v: v[day - 1], trace.result.next_discrete_state)
            )
            days.append((jax.tree_util.tree_map(lambda v: v[day], templates), discrete, retained_inputs))
        kernels = make_day_kernels(jax, make_step(bootstrap))
        cotangent_days = [jax.tree_util.tree_map(lambda v: v[day], cotangents) for day in range(30)]

        def progress(event, data):
            if event == "reverse_day":
                report["reverse_days"].append(data)
            if event == "reverse_memory":
                report["one_day_reverse_memory"] = data
            save(f"{event}_{data['day']}" if "day" in data else event)

        result = streamed_derivatives(
            jax, jnp, kernels, values, direction, fixed, states, days, cotangent_days, callback=progress
        )
        report["forward_projection"] = result["forward_projection"]
        report["reverse_projection"] = result["reverse_projection"]
        report["maximum_duality_gap"] = max(abs(r["duality_gap"]) for r in result["records"])
        report["largest_duality_gap_day"] = max(result["records"], key=lambda r: abs(r["duality_gap"]))["day"]
        state_scale = np.asarray(context["scales"].state.other, dtype=np.float64)
        report["state_adjoint_locations"] = []
        for day, adjoint in enumerate(result["adjoints"]):
            array = np.asarray(adjoint, dtype=np.float64)
            indices = np.argsort(np.abs(array * state_scale))[-3:][::-1]
            report["state_adjoint_locations"].append(
                {
                    "day": day + 1,
                    "coordinates": [
                        dict(
                            window._state_location(context["runtime"]["contract"], int(i)),
                            gradient=float(array[i]),
                            scaled_gradient=float(array[i] * state_scale[i]),
                        )
                        for i in indices
                    ],
                }
            )
        report["local_primal_state_errors"] = [
            float(
                jnp.max(jnp.where(trace.result.next_state_defined.other[i], jnp.abs(p["state"] - obs["state"][i]), 0.0))
            )
            for i, p in enumerate(result["primals"])
        ]
        trace_path = args.output.with_suffix(".trace.pkl")
        with trace_path.open("wb") as stream:
            pickle.dump(
                jax.device_get(
                    {
                        "states": states,
                        "days": days,
                        "observations": obs,
                        "cotangents": cotangents,
                        "tangents": result["tangents"],
                        "input_tangents": result["input_tangents"],
                        "adjoints": result["adjoints"],
                    }
                ),
                stream,
            )
        report["diagnostic_trace_path"] = str(trace_path)
        save("canonical_finite_differences")
        report["finite_differences"] = []
        for epsilon in (1e-4, 1e-5):
            losses = []
            masks = []
            variants = []
            for sign in (1.0, -1.0):
                variant = window.displaced(jax, values, direction, sign * epsilon)
                variants.append(variant)
                current, _ = evaluator(window.replace_conditioned_coupled_gpp_trainable(fixed, variant), micro)
                current_loss = score(
                    observations(current), current.result.next_state_defined.other, current.active_pft_mask, total
                )
                losses.append(float(current_loss))
                masks.append(
                    int(jnp.sum(current.result.next_state_defined.other != trace.result.next_state_defined.other))
                )
            report["finite_differences"].append(
                {
                    "epsilon": epsilon,
                    "central_secant": (losses[0] - losses[1]) / (2 * epsilon),
                    "state_mask_changes": masks,
                    "reverse_dot_actual_direction": dot(
                        jax,
                        result["gradient"],
                        jax.tree_util.tree_map(
                            lambda a, b: (a.astype(jnp.float64) - b.astype(jnp.float64)) / (2 * epsilon), *variants
                        ),
                    ),
                }
            )
            save(f"finite_difference_{epsilon}")
        report["status"] = "completed"
        save("completed")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save(report["stage"])
        raise
    return report


def _parser():
    parser = window._parser()
    parser.add_argument("--probe", type=Path, default=PROBE)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
