"""Localize the rejected update-4 derivative without retaining parameter changes."""

from __future__ import annotations

import json
import time
from pathlib import Path

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
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    SEVEN_SOURCE,
    THIRTY_SOURCE,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_science_window_sufficient_statistics,
    linearized_science_window_loss,
    science_window_from_sufficient_statistics,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_conditioned_attention_discarded_update import (
    _conditioned_bootstrap,
    _load_parent_state,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import _add_gradient, _zero_gradient
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.probe_successor_multibatch_direction import _make_flat_direction, _make_flattened_unit_gradient
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_conditioned_attention_persistent_training import (
    _parser as _base_parser,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import _make_dtype_preserving_direct_virtual_step
from scripts.hpc.run_target_oriented_coupled_gpp_optimizer_preflight import _mean_gradient
from scripts.hpc.run_target_oriented_coupled_gpp_update2_geometry import _scaled_parameters
from scripts.hpc.run_target_oriented_full_step_preflight import _hard_failure_count, _slice
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import _direct_gradient, _hard_host

DEFAULT_RUN = Path(
    "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/"
    "target-oriented-coupled-gpp-conditioned-attention-persistent-45ab263"
)
SCALES = (0.0, 1.0, -1.0, 0.25, -0.25, 0.0625, -0.0625)


def tree_dot(jax, left, right):
    return float(
        sum(
            np.sum(np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64))
            for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True)
        )
    )


def summarize_artifact(record):
    cells = {float(value["scale"]): value for value in record["evaluations"]}
    baseline = cells[0.0]["loss"]
    pairs = []
    for scale in (1.0, 0.25, 0.0625):
        plus, minus = cells[scale], cells[-scale]
        pairs.append(
            {
                "scale": scale,
                "central_secant": (plus["loss"] - minus["loss"]) / (2.0 * scale),
                "forward_change": plus["loss"] - baseline,
                "reverse_change": minus["loss"] - baseline,
                "gradient_dot_actual_delta": plus["gradient_dot_actual_delta"],
                "gradient_minus_forward_change": plus["gradient_dot_actual_delta"] - (plus["loss"] - baseline),
            }
        )
    return {
        "artifact_update": record["artifact_update"],
        "baseline": baseline,
        "historical_baseline_error": baseline - record["historical_baseline"],
        "historical_full_step_error": cells[1.0]["loss"] - record["historical_full_step"],
        "pairs": pairs,
    }


def run(args):
    started = time.perf_counter()
    parent = json.loads((args.conditioned_run_dir / "report.json").read_text())
    rejection = json.loads((args.conditioned_run_dir / "rejected-proposal.json").read_text())
    if parent["training"]["completed_logical_update"] != 3 or rejection["proposal"]["logical_update"] != 4:
        raise ValueError("derivative audit requires accepted update 3 and rejected update 4")
    recovered = load_recovery(
        args.conditioned_run_dir / "recovery.pkl",
        identity=parent["identity"],
        parameter_count=TRAINABLE_PARAMETER_COUNT,
    )
    if recovered is None or recovered["completed_logical_update"] != 3:
        raise ValueError("update-3 recovery is unavailable")
    _, original = _load_parent_state(
        args.parent_persistent_training_report,
        args.parent_artifact_aware_recovery,
        persistent_contract_sha256=json.loads(args.artifact_aware_persistent_contract.read_text())["contract_sha256"],
    )
    bootstrap = _conditioned_bootstrap(
        args,
        recovered=original,
        conditioning_contract=load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract),
    )
    context = bootstrap["context"]
    jax, jnp = context["runtime"]["jax"], context["runtime"]["jnp"]
    from jax.flatten_util import ravel_pytree

    trainable = jax.device_put(recovered["trainable"])
    parameters = replace_conditioned_coupled_gpp_trainable(bootstrap["initialized_parameters"], trainable)
    proposal = rejection["proposal"]
    direction = proposal["direction"]
    weights = direction["selected_minimum_norm_direction"]["normalized_simplex_weights"]
    labels = direction["source_registry"]
    scheduled = proposal["train_only_panel"]["artifact_groups"][0]
    flatten = _make_flattened_unit_gradient(jax)
    unit_support, norm_checks, horizon_records, thirty_gradients = {}, {}, [], {}
    raw_thirty_gradients = {}
    report = {
        "schema_version": "conditioned_attention_update4_derivative_v1",
        "status": "running",
        "parent_git_head": parent["identity"]["git_head"],
        "policy": {"retained_updates": 0, "monitor_evaluated": False, "teacher_regenerated": False},
        "support_sources": [name for name in labels if weights[name] > 0.0],
        "horizon_gradient_records": horizon_records,
        "artifact_records": [],
    }

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        _atomic_json(args.output, report)

    def unit_record(name, gradient):
        unit, norm, nonfinite = flatten(gradient)
        jax.block_until_ready((unit, norm, nonfinite))
        if int(nonfinite) != 0:
            raise RuntimeError(f"nonfinite gradient: {name}")
        unit_support[name] = unit
        norm_checks[name] = {
            "observed": float(norm),
            "historical": direction["source_gradients"][name]["l2_norm"],
        }

    for name in report["support_sources"]:
        if ".artifact_" not in name:
            continue
        update = int(name.rsplit("_", 1)[1])
        artifact = _load_artifact(args, context, update)
        _, gradient, _, _ = _direct_gradient(jax, jnp, trainable, artifact.batch, bootstrap["direct_steps"])
        unit_record(name, gradient)
        del artifact, gradient
        print(f"derivative support direct artifact={update}", flush=True)

    # Preserve the production microbatch reduction and expose the previously ignored reverse primal.
    for horizon, source, name in ((7, SEVEN_SOURCE, "seven"), (30, THIRTY_SOURCE, "thirty")):
        accumulated = _zero_gradient(jax, jnp, trainable)
        for update in scheduled:
            artifact = _load_artifact(args, context, update)
            batch = artifact.batch.seven_day if horizon == 7 else artifact.batch.thirty_day
            micros = [_slice(batch, i, i + 1) for i in range(8 if horizon == 7 else 4)]
            outputs = [bootstrap["steps"][f"{name}_forward"](parameters, micro) for micro in micros]
            jax.block_until_ready(outputs)
            total = outputs[0][0]
            for output in outputs[1:]:
                total = add_science_window_sufficient_statistics(total, output[0])
            current_sum = _zero_gradient(jax, jnp, trainable)
            raw_sum = jax.tree_util.tree_map(lambda value: jnp.zeros_like(value, dtype=jnp.float64), trainable)
            for i, micro in enumerate(micros):
                primal, gradient = bootstrap["gradient_steps"][f"{name}_gradient"](trainable, micro, total)
                jax.block_until_ready((primal, gradient))
                expected = linearized_science_window_loss(outputs[i][0], total, flux_basis=bootstrap["flux_basis"])
                current_sum = _add_gradient(jax, jnp, current_sum, gradient)
                raw_sum = jax.tree_util.tree_map(lambda a, b: a + b.astype(jnp.float64), raw_sum, gradient)
                horizon_records.append(
                    {
                        "artifact_update": update,
                        "horizon": horizon,
                        "window": i,
                        "reverse_primal": float(primal),
                        "forward_linearized_primal": float(expected),
                        "primal_difference": float(primal - expected),
                        "gradient_l2": float(jnp.linalg.norm(ravel_pytree(gradient)[0])),
                        "hard_failure_count": _hard_failure_count(_hard_host(jax.device_get(outputs[i][1]))),
                    }
                )
            accumulated = _add_gradient(jax, jnp, accumulated, current_sum)
            if horizon == 30:
                thirty_gradients[update] = jax.device_get(current_sum)
                raw_thirty_gradients[update] = jax.device_get(raw_sum)
            print(f"derivative gradient horizon={horizon} artifact={update}", flush=True)
            save()
            del artifact, current_sum, raw_sum, gradient, outputs, micros
        mean = _mean_gradient(jax, accumulated, len(scheduled))
        unit_record(source, mean)
        del accumulated, mean

    zero = jnp.zeros_like(next(iter(unit_support.values())))
    matrix = jnp.stack([unit_support.get(name, zero) for name in labels])
    combined, norm, nonfinite = _make_flat_direction(jax)(
        matrix, jnp.asarray([weights[name] for name in labels], dtype=jnp.float64)
    )
    jax.block_until_ready((combined, norm, nonfinite))
    _, unravel = ravel_pytree(trainable)
    full_step = direction["calibration"]["full_parameter_delta_l2"]
    proposed, nonfinite = _make_dtype_preserving_direct_virtual_step(jax)(
        trainable,
        unravel(combined),
        norm,
        np.asarray(full_step, dtype=np.float64),
    )
    jax.block_until_ready((proposed, nonfinite))
    if int(nonfinite) != 0:
        raise RuntimeError("nonfinite reconstructed step")
    report["direction_reproduction"] = {
        "gradient_norms": norm_checks,
        "combined_norm": float(norm),
        "historical_combined_norm": direction["combined_gradient_norm"],
        "ideal_thirty_day_change": -full_step
        * direction["source_gradients"][THIRTY_SOURCE]["l2_norm"]
        * direction["selected_minimum_norm_direction"]["normalized_directional_derivatives"][THIRTY_SOURCE],
    }
    variants = {scale: _scaled_parameters(jax, trainable, proposed, scale) for scale in SCALES}
    host_before = jax.device_get(trainable)
    del matrix, combined
    unit_support.clear()
    baseline_by_update = {
        r["artifact_update"]: r["source_losses_before"][THIRTY_SOURCE]
        for r in proposal["baseline"]["scheduled_artifact_records"]
    }
    historical_after = {
        r["artifact_update"]: r["source_losses"][THIRTY_SOURCE]
        for r in proposal["scale_evaluations"][0]["scheduled_evaluation"]["evaluation"]["records"]
    }
    for update in scheduled:
        artifact = _load_artifact(args, context, update)
        cells = []
        for scale, variant in variants.items():
            current_parameters = replace_conditioned_coupled_gpp_trainable(bootstrap["initialized_parameters"], variant)
            outputs = [
                bootstrap["steps"]["thirty_forward"](current_parameters, _slice(artifact.batch.thirty_day, i, i + 1))
                for i in range(4)
            ]
            jax.block_until_ready(outputs)
            total = outputs[0][0]
            for output in outputs[1:]:
                total = add_science_window_sufficient_statistics(total, output[0])
            loss = science_window_from_sufficient_statistics(total, flux_basis=bootstrap["flux_basis"])
            delta = jax.tree_util.tree_map(
                lambda a, b: np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64),
                jax.device_get(variant),
                host_before,
            )
            cells.append(
                {
                    "scale": scale,
                    "loss": float(loss.total),
                    "gradient_dot_actual_delta": tree_dot(jax, thirty_gradients[update], delta),
                    "raw_gradient_dot_actual_delta": tree_dot(jax, raw_thirty_gradients[update], delta),
                    "actual_delta_l2": np.sqrt(tree_dot(jax, delta, delta)),
                    "hard_failure_count": sum(_hard_failure_count(_hard_host(jax.device_get(v[1]))) for v in outputs),
                }
            )
        record = {
            "artifact_update": update,
            "historical_baseline": baseline_by_update[update],
            "historical_full_step": historical_after[update],
            "evaluations": cells,
        }
        record["comparison"] = summarize_artifact(record)
        report["artifact_records"].append(record)
        print(f"derivative finite_difference artifact={update} comparison={record['comparison']}", flush=True)
        save()
        del artifact
    report["status"] = "completed"
    report["mean_full_step_actual_change"] = float(
        np.mean([v["comparison"]["pairs"][0]["forward_change"] for v in report["artifact_records"]])
    )
    report["mean_full_step_gradient_dot_actual_delta"] = float(
        np.mean([v["comparison"]["pairs"][0]["gradient_dot_actual_delta"] for v in report["artifact_records"]])
    )
    report["maximum_reverse_primal_difference"] = max(abs(v["primal_difference"]) for v in horizon_records)
    save()
    return report


def _parser():
    parser = _base_parser()
    parser.add_argument("--conditioned-run-dir", type=Path, default=DEFAULT_RUN)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
