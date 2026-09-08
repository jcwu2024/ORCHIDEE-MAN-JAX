"""Same-input thermal decoder and recurrent sensitivity audit; no training."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import numpy as np

from scripts.hpc import audit_conditioned_attention_seven_score as seven

window = seven.window
FIELDS = tuple(f"thermal.hydrol_previous_step_state.{name}.normalized_tendency" for name in ("snowheat", "snowtemp"))
BOUNDS = dict(zip(FIELDS, (1e9, 100.0), strict=True))


def locations(contract):
    """Resolve canonical owners and mirrors from the actual state contract."""
    return {leaf.key: np.arange(leaf.start, leaf.stop) for leaf in contract.state_leaves
            if leaf.key in {f"{owner}_previous_step_state.{field}" for owner in ("hydrol", "diffuco")
                            for field in ("snowheat", "snowtemp")}}


def source_rows(start, result, captured, head, indices):
    """Reconstruct each learned affine thermal update independently on the host."""
    rows = {}
    thermal_offset = 0
    offsets = {}
    from research.daily_coarse_graining.constrained_daily_replay import THERMAL_FIELDS

    for component, field, _ in THERMAL_FIELDS:
        key = f"{component}.{field}"
        if key in head.input_assembler.thermal_keys:
            leaf = next(v for v in head.contract.state_leaves if v.key == key)
            offsets[key] = slice(thermal_offset, thermal_offset + leaf.stop - leaf.start)
            thermal_offset += leaf.stop - leaf.start
    for field in FIELDS:
        key = field.removeprefix("thermal.").removesuffix(".normalized_tendency")
        raw, decoded = (np.asarray(v, dtype=np.float64).reshape(-1) for v in captured[field])
        spec = next(v for v in head.learned_schema.fields if v.name == field)
        mean = np.asarray(head.output_normalization.mean[field]).reshape(-1)
        scale = np.asarray(head.output_normalization.scale[field]).reshape(-1)
        # These two fields are normalized signed tendencies, not day-end owners.
        if spec.constraint != "signed_unit" or spec.value_representation not in ("direct", "normalized_absolute_teacher_native"):
            raise ValueError(f"thermal source transform requires inspection: {spec.value_representation}")
        if spec.value_representation == "direct":
            expected = np.tanh(raw)
            slope = 1 - expected**2
        else:
            center = np.clip(mean, -1 + 1e-6, 1 - 1e-6)
            local_slope = np.maximum(1 - center**2, 1e-6)
            expected = np.tanh(np.arctanh(center) + raw * scale / local_slope)
            slope = (1 - expected**2) * scale / local_slope
        initial = np.asarray(start)[indices[key]]
        endpoint = np.asarray(result.next_state.other)[indices[key]]
        fast = np.asarray(result.fast_update.state.thermal)[offsets[key]]
        rows[field] = {
            "raw": raw.tolist(), "decoded": decoded.tolist(), "normalization_mean": mean.tolist(),
            "normalization_scale": scale.tolist(), "representation": spec.value_representation,
            "bound": BOUNDS[field], "start": initial.tolist(), "fast_endpoint": fast.tolist(),
            "tail_endpoint": endpoint.tolist(), "raw_to_endpoint_local_slope": (BOUNDS[field] * slope).tolist(),
            "decode_host_max_abs_error": float(np.max(np.abs(expected - decoded))),
            "affine_host_max_abs_error": float(np.max(np.abs(initial + BOUNDS[field] * decoded - fast))),
            "tail_minus_fast_max_abs_error": float(np.max(np.abs(endpoint - fast))),
        }
    return rows


def make_kernels(bootstrap):
    from research.daily_coarse_graining.daily_operator import target_oriented_attention as attention
    from research.daily_coarse_graining.daily_operator.successor_training import rollout_input_from_template

    context = bootstrap["context"]
    jax = context["runtime"]["jax"]
    dynamic = bootstrap["steps"]["transition_builder"](context["template"])
    decode = attention.normalized_output_to_physical

    def transition(parameters, state, day):
        template, discrete, retained = day
        inputs = rollout_input_from_template(template, state, discrete, context["runtime"]["contract"])
        return dynamic(parameters, inputs, retained)

    def detailed(parameters, state, day):
        captured = {}

        def record(raw, mask, **kwargs):
            value = decode(raw, mask, **kwargs)
            name = kwargs["field"].name
            if name in FIELDS:
                captured[name] = (raw, value)
            return value

        # Diagnostic-only tracing hook. Restored before execution; no formulas change.
        with patch.object(attention, "normalized_output_to_physical", record):
            result = transition(parameters, state, day)
        return result, captured

    def lean(parameters, state, day):
        return transition(parameters, state, day).next_state.other

    def tangent(parameters, state, direction, day):
        return jax.jvp(lambda s: lean(parameters, s, day), (state,), (direction,))

    return jax.jit(detailed), jax.jit(lean), jax.jit(tangent)


def experiment(bootstrap, values, artifact, *, emit):
    context = bootstrap["context"]
    jax, jnp = (context["runtime"][k] for k in ("jax", "jnp"))
    contract = context["runtime"]["contract"]
    indices = locations(contract)
    selected = np.concatenate(tuple(indices.values()))
    scales = np.asarray(context["scales"].state.other)
    parameters = window.replace_conditioned_coupled_gpp_trainable(bootstrap["fixed_parameters"], values)
    verbose_forward = window._make_trace_evaluator(bootstrap)
    compact_forward = seven.integration.make_compact_trace_evaluator(bootstrap)
    detailed, lean, tangent = make_kernels(bootstrap)
    report = {"windows": [], "state_locations": {k: v.tolist() for k, v in indices.items()}}

    def compare(a, b):
        a, b = np.asarray(a), np.asarray(b)
        finite = np.isfinite(a) & np.isfinite(b)
        delta = np.subtract(a, b, out=np.zeros_like(a), where=finite)
        ranked = np.abs(delta) / np.maximum(scales, 1e-30)
        i = int(np.argmax(ranked))
        return {"max_scaled_difference": float(ranked[i]), "worst": window._state_location(contract, i),
                "snow_difference": {k: delta[v].tolist() for k, v in indices.items()},
                "finite_mask_mismatches": int(np.sum(np.isfinite(a) != np.isfinite(b)))}

    for index in seven.WINDOWS:
        micro = window._slice(artifact.batch.seven_day, index, index + 1)
        row = {"window": index, "days": []}
        report["windows"].append(row)
        emit("trace_start", report)
        verbose, _ = verbose_forward(parameters, micro)
        compact, _ = compact_forward(parameters, micro)
        jax.block_until_ready((verbose, compact))
        states, days = seven.step_audit.trace_days(jax, micro, verbose)
        compact_states, _ = seven.step_audit.trace_days(jax, micro, compact)
        for day_index, (state, day) in enumerate(zip(states, days, strict=True)):
            result, captured = detailed(parameters, state, day)
            plain = lean(parameters, state, day)
            varied = lean(parameters, compact_states[day_index], day)
            jax.block_until_ready((result, captured, plain, varied))
            item = {"day": day_index + 1,
                    "sources": source_rows(state, result, captured, bootstrap["head"], indices),
                    "same_input_lean_minus_detailed": compare(plain, result.next_state.other),
                    "same_input_detailed_minus_verbose_scan": compare(result.next_state.other, verbose.result.next_state.other[day_index]),
                    "shared_kernel_full_input_difference": compare(varied, plain),
                    "actual_compact_minus_verbose_scan": compare(compact.result.next_state.other[day_index], verbose.result.next_state.other[day_index])}
            if index == seven.WINDOWS[0] and day_index in (1, 3, 6):
                # Isolate coupled snow fields, including both mirrors, under one executable.
                delta = jnp.where(jnp.isfinite(state) & jnp.isfinite(compact_states[day_index]),
                                  compact_states[day_index] - state, 0.0)
                snow_delta = jnp.zeros_like(delta).at[selected].set(delta[selected])
                snow_only = lean(parameters, state + snow_delta, day)
                other_only = lean(parameters, state + (delta - snow_delta), day)
                primal, response = tangent(parameters, state, snow_delta, day)
                checks = []
                for epsilon in (1.0, 0.5, 0.1):
                    plus = lean(parameters, state + epsilon * snow_delta, day)
                    minus = lean(parameters, state - epsilon * snow_delta, day)
                    secant = (plus - minus) / (2 * epsilon)
                    checks.append({"epsilon": epsilon, "jvp_minus_secant": compare(response, secant),
                                   "secant_snow": {k: np.asarray(secant)[v].tolist() for k, v in indices.items()}})
                item.update(snow_only_effect=compare(snow_only, plain), other_only_effect=compare(other_only, plain),
                            jvp_primal_minus_forward=compare(primal, plain),
                            input_snow_delta={k: np.asarray(snow_delta)[v].tolist() for k, v in indices.items()},
                            jvp_snow={k: np.asarray(response)[v].tolist() for k, v in indices.items()},
                            directional_checks=checks)
            row["days"].append(item)
            emit("day_completed", report)
    return report


def run(args):
    started = time.perf_counter()
    report = {"schema_version": "conditioned_attention_thermal_source_v1", "status": "initializing",
              "policy": {"retained_updates": 0, "monitor_evaluated": False, "teacher_regenerated": False}}

    def emit(stage, updates):
        report.update(updates)
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started)
        window._atomic_json(args.output, report)
        print(f"thermal_source stage={stage} elapsed={report['elapsed_seconds']:.1f}s", flush=True)

    emit("bootstrap", {})
    try:
        parent = json.loads((args.conditioned_run_dir / "report.json").read_text())
        recovered = window.load_recovery(args.conditioned_run_dir / "recovery.pkl", identity=parent["identity"],
                                         parameter_count=window.TRAINABLE_PARAMETER_COUNT)
        if recovered is None or recovered["completed_logical_update"] != 3:
            raise ValueError("thermal source audit requires accepted update 3")
        _, original = window._load_parent_state(
            args.parent_persistent_training_report, args.parent_artifact_aware_recovery,
            persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())["contract_sha256"],
        )
        bootstrap = window._conditioned_bootstrap(args, recovered=original,
            conditioning_contract=window.load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract))
        emit("load_artifact", {"status": "running", "parent_identity": parent["identity"]})
        artifact = window._load_artifact(args, bootstrap["context"], seven.ARTIFACT)
        values = bootstrap["context"]["runtime"]["jax"].device_put(recovered["trainable"])
        result = experiment(bootstrap, values, artifact, emit=emit)
        emit("completed", {**result, "status": "completed", "classification": "source_diagnostic_no_model_admission"})
    except Exception as exc:
        emit(report.get("stage", "bootstrap"), {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    return report


if __name__ == "__main__":
    run(window._parser().parse_args())
