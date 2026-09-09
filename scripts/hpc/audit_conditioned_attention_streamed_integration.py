"""Exercise the unchanged update-4 train-only protocol with streamed horizon gradients."""

from __future__ import annotations

import json
import time

import numpy as np

from research.daily_coarse_graining.daily_operator.streamed_adjoint import StreamedAdjoint, compact_adjoint_trace
from scripts.hpc import audit_conditioned_attention_streamed_step as step_audit
from scripts.hpc import run_target_oriented_coupled_gpp_conditioned_attention_persistent_training as persistent

streamed = step_audit.streamed
window = step_audit.window


def make_compact_trace_evaluator(bootstrap):
    """Record only adjoint inputs from the unchanged canonical daily rollout."""
    from research.daily_coarse_graining.daily_operator.successor_training import (
        rollout_daily_operator,
        trajectories_from_rollout,
    )

    context = bootstrap["context"]
    jax, contract = context["runtime"]["jax"], context["runtime"]["contract"]
    dynamic = bootstrap["steps"]["transition_builder"](context["template"])

    def evaluate(parameters, micro):
        templates = jax.tree_util.tree_map(lambda v: v[0], micro.inputs)
        retained = jax.tree_util.tree_map(lambda v: v[0], micro.day_inputs)
        trace = rollout_daily_operator(
            parameters, templates, transition=lambda v, inputs: dynamic(v, inputs, retained),
            contract=contract, rematerialize=False,
        )
        compact = compact_adjoint_trace(trace)
        return compact, trajectories_from_rollout(compact, contract)

    return jax.jit(evaluate)


def make_observation_gradient(bootstrap, horizon):
    """Dynamic targets keep the small science-loss executable reusable across windows."""
    from research.daily_coarse_graining.daily_operator.successor_training import (
        extract_biomass_productivity_trajectories,
    )

    context = bootstrap["context"]
    jax, jnp = context["runtime"]["jax"], context["runtime"]["jnp"]
    contract, scales = context["runtime"]["contract"], context["scales"]

    def score(obs, defined, active, total, micro):
        templates = jax.tree_util.tree_map(lambda v: v[0], micro.inputs)
        expected = extract_biomass_productivity_trajectories(
            micro.expected_next_state.other[0], micro.expected_next_state_defined.other[0],
            daily_fast_gpp=micro.expected_fast_gpp[0], daily_fast_gpp_defined=micro.expected_fast_gpp_defined[0],
            pft_fraction=templates.pft.pft_fraction, active_pft_mask=templates.pft.active_pft_mask, contract=contract,
        )
        predicted = extract_biomass_productivity_trajectories(
            obs["state"], defined, daily_fast_gpp=obs["gpp"],
            daily_fast_gpp_defined=jnp.broadcast_to(active[:, None, :], obs["gpp"].shape),
            pft_fraction=obs["fraction"], active_pft_mask=active, contract=contract,
        )
        current = window.science_window_sufficient_statistics(
            predicted.fluxes, expected.fluxes, predicted.stocks, expected.stocks,
            flux_scales=scales.fluxes, stock_scales=scales.stocks, horizon=horizon, flux_basis=bootstrap["flux_basis"],
        )
        return window.linearized_science_window_loss(current, total, flux_basis=bootstrap["flux_basis"])

    return jax.jit(jax.value_and_grad(score))


class StreamedHorizonCallbacks:
    def __init__(self, bootstrap, *, callback=None, compact_trace=False):
        self.bootstrap = bootstrap
        self.jax = bootstrap["context"]["runtime"]["jax"]
        self.jnp = bootstrap["context"]["runtime"]["jnp"]
        self.emit = callback or (lambda event, row: None)
        self.trace = (make_compact_trace_evaluator(bootstrap) if compact_trace
                      else window._make_trace_evaluator(bootstrap))
        self.reverse = StreamedAdjoint(self.jax, self.jnp, streamed.make_step(bootstrap))
        self.scores = {h: make_observation_gradient(bootstrap, h) for h in (7, 30)}
        self.records = []
        self.calls = {7: 0, 30: 0}

    def gradient(self, values, micro, total, *, horizon):
        started = time.perf_counter()
        jax = self.jax
        name = "seven" if horizon == 7 else "thirty"
        ordinal = self.calls[horizon]
        self.emit("horizon_start", {"horizon": horizon, "ordinal": ordinal})
        parameters = window.replace_conditioned_coupled_gpp_trainable(self.bootstrap["fixed_parameters"], values)
        trace, _ = self.trace(parameters, micro)
        loss, cotangents = self.scores[horizon](
            streamed.observations(trace), trace.result.next_state_defined.other, trace.active_pft_mask, total, micro,
        )
        states, days = step_audit.trace_days(jax, micro, trace)
        cotangent_days = [jax.tree_util.tree_map(lambda v: v[i], cotangents) for i in range(len(days))]

        def progress(event, row):
            if event != "reverse_day" or row["day"] % 10 == 0 or row["day"] == 1:
                self.emit(event, {"horizon": horizon, "ordinal": ordinal, **row})

        gradient = self.reverse.gradient(
            values, self.bootstrap["fixed_parameters"], states, days, cotangent_days, callback=progress,
        )
        statistics, _ = self.bootstrap["steps"][f"{name}_forward"](parameters, micro)
        reference = float(window.linearized_science_window_loss(statistics, total, flux_basis=self.bootstrap["flux_basis"]))
        record = {
            "horizon": horizon, "ordinal": ordinal, "observation_linearized_score": float(loss),
            "production_linearized_score": reference, "score_difference": float(loss) - reference,
            "linearized_score_agrees": bool(np.isclose(float(loss), reference, atol=1e-6, rtol=1e-5)),
            "seconds": time.perf_counter() - started,
        }
        self.records.append(record)
        self.calls[horizon] += 1
        self.emit("horizon_complete", record)
        return loss, gradient

    def attach(self):
        gradients = dict(self.bootstrap["gradient_steps"])
        gradients["seven_gradient"] = lambda v, b, t: self.gradient(v, b, t, horizon=7)
        gradients["thirty_gradient"] = lambda v, b, t: self.gradient(v, b, t, horizon=30)
        return dict(self.bootstrap, gradient_steps=gradients)


def classify(record, callbacks):
    complete = callbacks.calls == {7: 64, 30: 32}
    scores = len(callbacks.records) == 96 and all(r["linearized_score_agrees"] for r in callbacks.records)
    selected = bool(record["selection"]["accepted"])
    return {
        "passed": complete and scores and selected,
        "classification": "streamed_update4_ready_for_fixed_monitor" if complete and scores and selected else "streamed_update4_not_admitted",
        "all_scheduled_windows_differentiated": complete, "observation_linearizations_agree": scores,
        "original_train_only_gates_passed": selected, "selected_scale": record["selection"]["selected_scale"],
    }


def run(args, *, warm_snow_free_boundary=False):
    started = time.perf_counter()
    report = {
        "schema_version": "conditioned_attention_streamed_integration_v1", "status": "initializing", "stage": "bootstrap",
        "logical_update": 4, "policy": {"retained_updates": 0, "monitor_evaluated": False, "teacher_regenerated": False},
        "horizon_records": [],
    }
    report["policy"]["warm_snow_free_boundary"] = warm_snow_free_boundary
    if warm_snow_free_boundary:
        report["schema_version"] = "conditioned_snow_free_integration_v1"

    def emit(stage, updates):
        report.update(updates)
        report["stage"], report["elapsed_seconds"] = stage, time.perf_counter() - started
        window._atomic_json(args.output, report)
        print(f"streamed_integration stage={stage} elapsed={report['elapsed_seconds']:.1f}s", flush=True)

    print("streamed_integration stage=bootstrap", flush=True)
    try:
        parent = json.loads((args.conditioned_run_dir / "report.json").read_text())
        recovered = window.load_recovery(args.conditioned_run_dir / "recovery.pkl", identity=parent["identity"], parameter_count=window.TRAINABLE_PARAMETER_COUNT)
        if recovered is None or recovered["completed_logical_update"] != 3:
            raise ValueError("streamed integration requires accepted update 3")
        _, original = window._load_parent_state(
            args.parent_persistent_training_report, args.parent_artifact_aware_recovery,
            persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())["contract_sha256"],
        )
        bootstrap = window._conditioned_bootstrap(
            args, recovered=original, conditioning_contract=window.load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract),
            warm_snow_free_boundary=warm_snow_free_boundary,
        )
        contract = persistent.load_conditioned_attention_persistent_training_contract(args.conditioned_persistent_contract)
        index = contract.logical_updates.index(4)
        panel, registry = contract.train_panels[index], contract.source_registries[index]
        report.update(status="running", parent_identity=parent["identity"], train_only_panel=[list(g) for g in panel])
        emit("attach_streamed_callbacks", {})

        def progress(event, row):
            if event == "horizon_complete":
                report["horizon_records"].append(row)
            if event == "reverse_memory":
                report["one_day_reverse_memory"] = row
            emit(event, {"progress": row})

        callbacks = StreamedHorizonCallbacks(bootstrap, callback=progress, compact_trace=warm_snow_free_boundary)
        jax = bootstrap["context"]["runtime"]["jax"]
        result = persistent._train_only_update(
            args, callbacks.attach(), contract, trainable=jax.device_put(recovered["trainable"]), panel=panel,
            source_registry=registry, logical_update=4,
            flatten_gradient=persistent._make_flattened_unit_gradient(jax), gram_step=persistent._make_flat_gram(jax),
            combine_direction=persistent._make_flat_direction(jax), direct_virtual_step=persistent._make_dtype_preserving_direct_virtual_step(jax),
        )
        record = result["record"]
        del result  # Discard the candidate tree; do not enter any persistence or monitor path.
        emit("completed", {
            "status": "completed", "proposal": record, "admission": classify(record, callbacks),
            "horizon_callback_counts": {str(k): v for k, v in callbacks.calls.items()},
            "reverse_compile_count": callbacks.reverse.compile_count,
        })
    except Exception as exc:
        emit(report["stage"], {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    return report


def _parser():
    return window._parser()


if __name__ == "__main__":
    run(_parser().parse_args())
