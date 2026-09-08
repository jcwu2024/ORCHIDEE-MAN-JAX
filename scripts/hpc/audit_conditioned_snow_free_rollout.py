"""Matched update-3 snow boundary screen on three train windows; no training."""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np

from research.daily_coarse_graining.daily_operator.snow_free_boundary import HYDROL, warm_snow_free_regime
from research.daily_coarse_graining.daily_operator.successor_training import extract_biomass_productivity_trajectories
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    science_window_from_sufficient_statistics,
)
from scripts.hpc import audit_conditioned_attention_streamed_integration as integration
from scripts.hpc.run_target_oriented_full_step_preflight import _make_steps

window = integration.window
CASES = ((322, 7, 5), (322, 7, 1), (442, 30, 0))


def with_boundary(bootstrap, enabled):
    context = bootstrap["context"]
    head = replace(bootstrap["head"], warm_snow_free_boundary=enabled)
    steps = _make_steps(context["runtime"], head, context["template"], context["scales"],
                        context["target_contract"], science_flux_basis=bootstrap["flux_basis"])
    return dict(bootstrap, head=head, steps=steps)


def science_summary(stats, basis):
    value = science_window_from_sufficient_statistics(stats, flux_basis=basis)
    return {"total": float(value.total), "flux_mse": float(value.flux_mse), "stock_mse": float(value.stock_mse),
            "fluxes": {k: {n: float(x) for n, x in v._asdict().items()} for k, v in value.flux_errors.items()},
            "stocks": {k: {n: float(x) for n, x in v._asdict().items()} for k, v in value.stock_errors.items()}}


def make_trace_statistics(bootstrap, horizon):
    context = bootstrap["context"]
    jax, scales = context["runtime"]["jax"], context["scales"]

    def evaluate(predicted, micro):
        expected = extract_biomass_productivity_trajectories(
            micro.expected_next_state.other[0], micro.expected_next_state_defined.other[0],
            daily_fast_gpp=micro.expected_fast_gpp[0], daily_fast_gpp_defined=micro.expected_fast_gpp_defined[0],
            pft_fraction=micro.inputs.pft.pft_fraction[0], active_pft_mask=micro.inputs.pft.active_pft_mask[0],
            contract=context["runtime"]["contract"],
        )
        stats = window.science_window_sufficient_statistics(
            predicted.fluxes, expected.fluxes, predicted.stocks, expected.stocks,
            flux_scales=scales.fluxes, stock_scales=scales.stocks, horizon=horizon,
            flux_basis=bootstrap["flux_basis"],
        )
        return stats, expected

    return jax.jit(evaluate)


def experiment(bootstrap, values, artifacts, emit):
    jax = bootstrap["context"]["runtime"]["jax"]
    contract = bootstrap["context"]["runtime"]["contract"]
    parameters = window.replace_conditioned_coupled_gpp_trainable(bootstrap["fixed_parameters"], values)
    report = {"cases": [], "all_repair_forward_gates_pass": True}
    for enabled in (False, True):
        current = with_boundary(bootstrap, enabled)
        trace_forward = integration.make_compact_trace_evaluator(current)
        extract = {h: make_trace_statistics(current, h) for h in (7, 30)}
        for update, horizon, index in CASES:
            batch = getattr(artifacts[update].batch, "seven_day" if horizon == 7 else "thirty_day")
            micro = window._slice(batch, index, index + 1)
            row = {"arm": "repair" if enabled else "reference", "artifact": update,
                   "horizon": horizon, "window": index}
            report["cases"].append(row)
            emit("production_forward", report)
            production_stats, hard = current["steps"]["seven_forward" if horizon == 7 else "thirty_forward"](parameters, micro)
            jax.block_until_ready((production_stats, hard))
            row.update(science=science_summary(production_stats, bootstrap["flux_basis"]), hard=window._hard_host(hard))
            emit("recorded_forward", report)
            trace, predicted = trace_forward(parameters, micro)
            recorded_stats, expected = extract[horizon](predicted, micro)
            jax.block_until_ready((trace, recorded_stats))
            recorded = science_summary(recorded_stats, bootstrap["flux_basis"])
            row["recorded_science"] = recorded
            row["score_agrees"] = bool(np.isclose(recorded["total"], row["science"]["total"], atol=1e-6, rtol=1e-5))
            next_state = np.asarray(trace.result.next_state.other)
            starts = np.concatenate((np.asarray(micro.inputs.continuous_state)[0, :1], next_state[:-1]), axis=0)
            forcing = jax.tree_util.tree_map(lambda x: x[0], micro.inputs.forcing)
            selected = np.asarray(warm_snow_free_regime(starts, forcing, contract, current["head"].forcing_layout))
            row["selected_days"] = selected.tolist()
            row["snow"] = {}
            snow_pass = True
            for owner in (HYDROL, "diffuco_previous_step_state"):
                for name, reference, tolerance in (("snow", 0., 0.), ("snowdz", 0., 0.),
                                                    ("snowheat", 0., 1e-6), ("snowtemp", 273.15, 1e-10)):
                    key = f"{owner}.{name}"
                    leaf = next(v for v in contract.state_leaves if v.key == key)
                    vals = next_state[:, leaf.start:leaf.stop]
                    error = float(np.max(np.abs(vals[selected] - reference), initial=0.))
                    snow_pass &= bool(np.all(np.isfinite(vals[selected])) and error <= tolerance)
                    row["snow"][key] = {"next": vals.tolist(), "selected_max_abs_error": error}
            row["snow_pass"] = snow_pass
            row["trajectories"] = {group: {name: {
                "predicted": np.asarray(value.value).tolist(), "predicted_defined": np.asarray(value.defined).tolist(),
                "teacher": np.asarray(getattr(expected, group)[name].value).tolist(),
                "teacher_defined": np.asarray(getattr(expected, group)[name].defined).tolist(),
            } for name, value in getattr(predicted, group).items()} for group in ("fluxes", "stocks")}
            row["hard_failure_count"] = window._hard_failure_count(hard)
            if enabled:
                report["all_repair_forward_gates_pass"] &= bool(snow_pass and row["score_agrees"]
                    and row["hard_failure_count"] == 0 and np.all(selected) and np.isfinite(row["science"]["total"]))
            emit("case_completed", report)
    return report


def run(args):
    started = time.perf_counter()
    report = {"schema_version": "conditioned_snow_free_rollout_v1", "status": "initializing",
              "policy": {"retained_updates": 0, "teacher_regenerated": False, "monitor_evaluated": False}}

    def emit(stage, updates):
        report.update(updates)
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started)
        window._atomic_json(args.output, report)
        print(f"snow_free stage={stage} elapsed={report['elapsed_seconds']:.1f}s", flush=True)

    print("snow_free stage=bootstrap", flush=True)
    try:
        parent = json.loads((args.conditioned_run_dir / "report.json").read_text())
        recovered = window.load_recovery(args.conditioned_run_dir / "recovery.pkl", identity=parent["identity"],
                                         parameter_count=window.TRAINABLE_PARAMETER_COUNT)
        if recovered is None or recovered["completed_logical_update"] != 3:
            raise ValueError("snow boundary screen requires accepted update 3")
        _, original = window._load_parent_state(
            args.parent_persistent_training_report, args.parent_artifact_aware_recovery,
            persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())["contract_sha256"],
        )
        bootstrap = window._conditioned_bootstrap(args, recovered=original,
            conditioning_contract=window.load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract))
        emit("load_artifacts", {"status": "running", "parent_identity": parent["identity"]})
        artifacts = {u: window._load_artifact(args, bootstrap["context"], u) for u in (322, 442)}
        values = bootstrap["context"]["runtime"]["jax"].device_put(recovered["trainable"])
        result = experiment(bootstrap, values, artifacts, emit)
        emit("completed", {**result, "status": "completed", "classification": "forward_screen_no_model_promotion"})
    except Exception as exc:
        emit(report.get("stage", "bootstrap"), {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    return report


if __name__ == "__main__":
    run(window._parser().parse_args())
