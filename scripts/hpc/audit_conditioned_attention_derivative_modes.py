"""Compare JVP/VJP of the identical production scalar with and without remat."""

from __future__ import annotations

import json
import pickle
import time

import numpy as np

from scripts.hpc import audit_conditioned_attention_window442 as window


def make_kernels(jax, score):
    return {
        "forward": jax.jit(score),
        "reverse": jax.jit(jax.value_and_grad(score)),
        "tangent": jax.jit(
            lambda values, direction, fixed, batch, total: jax.jvp(
                lambda v: score(v, fixed, batch, total),
                (values,),
                (direction,),
            )
        ),
    }


def summarize_mode(row):
    reverse, tangent = row["reverse_dot_probe"], row["tangent"]
    error = abs(reverse - tangent) / max(abs(reverse), abs(tangent), 1e-30)
    return {
        "reverse_tangent_relative_disagreement": error,
        "same_sign": bool(np.sign(reverse) == np.sign(tangent)),
        "reverse_tangent_primal_difference": row["reverse_primal"] - row["tangent_primal"],
    }


def make_score(bootstrap, rematerialize):
    def score(v, f, b, t):
        return bootstrap["steps"]["window_linearized_score"](
            window.replace_conditioned_coupled_gpp_trainable(f, v),
            b,
            t,
            horizon=30,
            rematerialize=rematerialize,
        )

    return score


def run(args):
    started = time.perf_counter()
    report = {
        "schema_version": "conditioned_attention_derivative_modes_v1",
        "status": "initializing",
        "stage": "bootstrap",
        "artifact_update": 442,
        "window": 0,
        "policy": {"retained_updates": 0, "monitor_evaluated": False, "teacher_regenerated": False},
        "probe_direction": "negative_unit_production_remat_window0_gradient",
        "modes": [],
    }

    def save(stage):
        report["stage"] = stage
        report["elapsed_seconds"] = time.perf_counter() - started
        window._atomic_json(args.output, report)
        print(f"derivative_modes stage={stage} elapsed={report['elapsed_seconds']:.1f}s", flush=True)

    print("derivative_modes stage=bootstrap", flush=True)
    try:
        parent = json.loads((args.conditioned_run_dir / "report.json").read_text())
        recovered = window.load_recovery(
            args.conditioned_run_dir / "recovery.pkl",
            identity=parent["identity"],
            parameter_count=window.TRAINABLE_PARAMETER_COUNT,
        )
        if recovered is None or recovered["completed_logical_update"] != 3:
            raise ValueError("derivative modes require exact accepted update 3")
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
        parameters = window.replace_conditioned_coupled_gpp_trainable(fixed, values)
        artifact = window._load_artifact(args, context, 442)
        batch = window._slice(artifact.batch.thirty_day, 0, 1)
        report["sample_identity"] = context["schedule"]["entries"][441]["components"]["thirty_day"][0]
        report["parent_identity"] = parent["identity"]
        report["jax_version"] = jax.__version__
        report["status"] = "running"
        save("four_window_baseline")
        outputs = [
            bootstrap["steps"]["thirty_forward"](parameters, window._slice(artifact.batch.thirty_day, i, i + 1))
            for i in range(4)
        ]
        jax.block_until_ready(outputs)
        total = outputs[0][0]
        for item in outputs[1:]:
            total = window.add_science_window_sufficient_statistics(total, item[0])
        report["baseline_hard_failure_count"] = sum(
            window._hard_failure_count(window._hard_host(jax.device_get(v[1]))) for v in outputs
        )
        if report["baseline_hard_failure_count"]:
            raise RuntimeError("baseline physical gate failed")
        direction = None
        for rematerialize in (True, False):
            name = "remat" if rematerialize else "no_remat"

            kernels = make_kernels(jax, make_score(bootstrap, rematerialize))
            row = {"rematerialize": rematerialize, "finite_differences": []}
            report["modes"].append(row)
            save(f"{name}_reverse_compile")
            compiled = kernels["reverse"].lower(values, fixed, batch, total).compile()
            memory = compiled.memory_analysis()
            row["reverse_memory_bytes"] = (
                {
                    key: int(getattr(memory, key))
                    for key in (
                        "argument_size_in_bytes",
                        "output_size_in_bytes",
                        "temp_size_in_bytes",
                        "alias_size_in_bytes",
                    )
                }
                if memory is not None
                else None
            )
            save(f"{name}_reverse_execute")
            primal, gradient = compiled(values, fixed, batch, total)
            jax.block_until_ready((primal, gradient))
            norm = np.sqrt(window.tree_dot(jax, gradient, gradient))
            if not np.isfinite(norm) or norm <= 0:
                raise RuntimeError("nonfinite/zero reverse gradient")
            if direction is None:
                direction = jax.tree_util.tree_map(lambda g, p: (-g / norm).astype(p.dtype), gradient, values)
                # Derived diagnostic arrays only, so future probes need not repeat this reverse pass.
                capsule = args.output.with_suffix(".probe.pkl")
                with capsule.open("wb") as stream:
                    pickle.dump(
                        {
                            "schema_version": "window442_derivative_probe_v1",
                            "parent_identity": parent["identity"],
                            "completed_logical_update": 3,
                            "artifact_update": 442,
                            "window": 0,
                            "direction": jax.device_get(direction),
                            "total": jax.device_get(total),
                        },
                        stream,
                    )
                report["diagnostic_probe_path"] = str(capsule)
            row.update(
                reverse_primal=float(primal),
                reverse_gradient_l2=float(norm),
                reverse_dot_probe=window.tree_dot(jax, gradient, direction),
            )
            if rematerialize:
                row["historical_gradient_norm_error"] = float(norm - 1097.9184269904633)
                row["historical_reverse_primal_error"] = float(primal - 0.05346045656575092)
            save(f"{name}_tangent")
            tangent_primal, tangent = kernels["tangent"](values, direction, fixed, batch, total)
            jax.block_until_ready((tangent_primal, tangent))
            row.update(tangent_primal=float(tangent_primal), tangent=float(tangent))
            row["comparison"] = summarize_mode(row)
            save(f"{name}_finite_differences")
            for epsilon in (1e-4, 1e-5):
                plus_values = window.displaced(jax, values, direction, epsilon)
                minus_values = window.displaced(jax, values, direction, -epsilon)
                plus, minus = (
                    kernels["forward"](plus_values, fixed, batch, total),
                    kernels["forward"](minus_values, fixed, batch, total),
                )
                jax.block_until_ready((plus, minus))
                actual_direction = jax.tree_util.tree_map(
                    lambda a, b: (a.astype(jnp.float64) - b.astype(jnp.float64)) / (2 * epsilon),
                    plus_values,
                    minus_values,
                )
                row["finite_differences"].append(
                    {
                        "epsilon": epsilon,
                        "plus": float(plus),
                        "minus": float(minus),
                        "central_secant": float((plus - minus) / (2 * epsilon)),
                        "reverse_dot_actual_direction": window.tree_dot(jax, gradient, actual_direction),
                    }
                )
                save(f"{name}_finite_difference_{epsilon}")
            del compiled, kernels, gradient
        report["status"] = "completed"
        save("completed")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save(report["stage"])
        raise
    return report


if __name__ == "__main__":
    run(window._parser().parse_args())
