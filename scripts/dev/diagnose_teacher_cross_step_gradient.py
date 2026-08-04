from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402
from research.daily_coarse_graining.teacher_parameter_gradient_gate import (  # noqa: E402
    prepare_later_day_cross_step_state_objectives,
)

jax.config.update("jax_enable_x64", True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose a two-step Teacher carry VJP")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orchidee_man_250919.yaml")
    parser.add_argument("--run-def", type=Path, default=ROOT / "outputs/reference_mode/used_run.def")
    parser.add_argument("--parameter", default="g0")
    parser.add_argument("--output", default="gpp")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    configure_jax_compilation_cache(ROOT)
    first_state, second_output, baseline_state, parameter_value = (
        prepare_later_day_cross_step_state_objectives(
            args.config,
            used_run_def_path=args.run_def,
            parameter_id=args.parameter,
            output_id=args.output,
        )
    )
    value = jnp.asarray(parameter_value, dtype=jnp.float64)
    state, state_tangent = jax.jvp(
        first_state,
        (value,),
        (jnp.ones_like(value),),
    )
    second_primal, second_pullback = jax.vjp(
        second_output,
        state.values_by_component,
    )
    state_cotangent = second_pullback(jnp.ones_like(second_primal))[0]
    _, first_pullback = jax.vjp(first_state, value)
    parameter_cotangent = first_pullback(
        replace(state_tangent, values_by_component=state_cotangent)
    )[0]

    fields: list[dict[str, object]] = []
    projected_fields: list[tuple[dict[str, object], int, int]] = []
    directional_derivative = 0.0
    for component_index, (component, names, cotangents, tangents) in enumerate(zip(
        state.spec.components,
        state.spec.field_names_by_component,
        state_cotangent,
        state_tangent.values_by_component,
        strict=True,
    )):
        for field_index, (name, cotangent, tangent) in enumerate(
            zip(names, cotangents, tangents, strict=True)
        ):
            cotangent_array = np.asarray(cotangent)
            tangent_array = np.asarray(tangent)
            if not (
                np.issubdtype(cotangent_array.dtype, np.inexact)
                and np.issubdtype(tangent_array.dtype, np.inexact)
            ):
                continue
            finite_cotangent = bool(np.isfinite(cotangent_array).all())
            finite_tangent = bool(np.isfinite(tangent_array).all())
            cotangent_nonzero = bool(np.any(cotangent_array != 0.0))
            tangent_nonzero = bool(np.any(tangent_array != 0.0))
            if not cotangent_nonzero and finite_cotangent and finite_tangent:
                continue
            contribution = float(np.sum(cotangent_array * tangent_array))
            directional_derivative += contribution
            item = {
                "component": component,
                "field": name,
                "cotangent_finite": finite_cotangent,
                "tangent_finite": finite_tangent,
                "cotangent_nonzero": cotangent_nonzero,
                "tangent_nonzero": tangent_nonzero,
                "max_abs_cotangent": float(np.max(np.abs(cotangent_array))),
                "max_abs_tangent": float(np.max(np.abs(tangent_array))),
                "directional_contribution": contribution,
            }
            fields.append(item)
            if cotangent_nonzero:
                projected_fields.append((item, component_index, field_index))

    zero_cotangents = jax.tree_util.tree_map(
        lambda item: (
            jnp.zeros_like(item)
            if np.issubdtype(np.asarray(item).dtype, np.inexact)
            else item
        ),
        state_cotangent,
    )
    for item, component_index, field_index in projected_fields:
        projected = [list(component) for component in zero_cotangents]
        projected[component_index][field_index] = state_cotangent[component_index][field_index]
        projected_tuple = tuple(tuple(component) for component in projected)
        projected_parameter = first_pullback(
            replace(state_tangent, values_by_component=projected_tuple)
        )[0]
        item["first_step_reverse_projection"] = float(np.asarray(projected_parameter))
    fields.sort(key=lambda item: abs(float(item["directional_contribution"])), reverse=True)
    report = {
        "parameter": args.parameter,
        "output": args.output,
        "parameter_value": parameter_value,
        "second_step_primal": float(np.asarray(second_primal)),
        "state_jvp_dot_vjp": directional_derivative,
        "first_step_reverse_projection": float(np.asarray(parameter_cotangent)),
        "nonzero_cotangent_or_nonfinite_field_count": len(fields),
        "fields": fields,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="ascii")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
