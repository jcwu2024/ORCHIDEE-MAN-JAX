"""Field-level decomposition for successor output-locality audits."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-30)


def summarize_normalized_output_change(
    *,
    base: Any,
    tangent: Any,
    proposed: Any,
    scale: Any,
    step_norm: float,
) -> Mapping[str, Any]:
    """Summarize one normalized output tape's JVP and exact finite change."""

    base_array = np.asarray(base, dtype=np.float64)
    tangent_array = np.asarray(tangent, dtype=np.float64)
    proposed_array = np.asarray(proposed, dtype=np.float64)
    scale_array = np.asarray(scale, dtype=np.float64)
    if (
        step_norm <= 0.0
        or base_array.shape != tangent_array.shape
        or base_array.shape != proposed_array.shape
    ):
        raise ValueError("normalized output change shape or step drift")
    broadcast_scale = np.broadcast_to(scale_array, base_array.shape)
    if not np.all(np.isfinite(broadcast_scale) & (broadcast_scale > 0.0)):
        raise ValueError("normalized output change scale is invalid")
    linear = step_norm * tangent_array / broadcast_scale
    exact = (proposed_array - base_array) / broadcast_scale
    residual = exact - linear
    nonfinite = sum(
        np.count_nonzero(~np.isfinite(value))
        for value in (base_array, tangent_array, proposed_array)
    )
    return {
        "scalar_count": int(base_array.size),
        "linear_squared_sum": float(np.sum(linear * linear, dtype=np.float64)),
        "exact_squared_sum": float(np.sum(exact * exact, dtype=np.float64)),
        "residual_squared_sum": float(
            np.sum(residual * residual, dtype=np.float64)
        ),
        "linear_exact_dot": float(np.sum(linear * exact, dtype=np.float64)),
        "maximum_absolute_linear_change": float(
            np.max(np.abs(linear)) if linear.size else 0.0
        ),
        "maximum_absolute_exact_change": float(
            np.max(np.abs(exact)) if exact.size else 0.0
        ),
        "maximum_absolute_residual": float(
            np.max(np.abs(residual)) if residual.size else 0.0
        ),
        "nonfinite_value_count": int(nonfinite),
    }


def aggregate_normalized_output_changes(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Aggregate per-group normalized output changes without retaining tapes."""

    if not records:
        raise ValueError("normalized output aggregation requires records")
    count = sum(int(record["scalar_count"]) for record in records)
    linear_sq = sum(float(record["linear_squared_sum"]) for record in records)
    exact_sq = sum(float(record["exact_squared_sum"]) for record in records)
    residual_sq = sum(float(record["residual_squared_sum"]) for record in records)
    dot = sum(float(record["linear_exact_dot"]) for record in records)
    return {
        "group_count": len(records),
        "scalar_count": count,
        "normalized_linear_rms": float(np.sqrt(linear_sq / max(count, 1))),
        "normalized_exact_rms": float(np.sqrt(exact_sq / max(count, 1))),
        "normalized_residual_rms": float(
            np.sqrt(residual_sq / max(count, 1))
        ),
        "normalized_linearization_cosine": float(
            dot / max(np.sqrt(linear_sq * exact_sq), 1.0e-30)
        ),
        "maximum_absolute_linear_change": max(
            float(record["maximum_absolute_linear_change"]) for record in records
        ),
        "maximum_absolute_exact_change": max(
            float(record["maximum_absolute_exact_change"]) for record in records
        ),
        "maximum_absolute_residual": max(
            float(record["maximum_absolute_residual"]) for record in records
        ),
        "nonfinite_value_count": sum(
            int(record["nonfinite_value_count"]) for record in records
        ),
    }


def classify_maintenance_output_projection(
    *,
    safe_reference: Mapping[str, Any],
    raw_control: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    pass_action: str,
    fail_action: str,
) -> Mapping[str, Any]:
    """Select the smallest bounded maintenance correction passing every gate."""

    safe_checks = {
        "safe_reference_all_constraints_nonincreasing": bool(
            safe_reference["loss_summary"]["all_group_components_nonincreasing"]
        ),
        "safe_reference_hard_failures_zero": int(
            safe_reference["hard_failure_count"]
        )
        == int(gates["hard_failure_count"]),
        "safe_reference_nonfinite_values_zero": int(
            safe_reference["nonfinite_value_count"]
        )
        == int(gates["nonfinite_value_count"]),
    }
    raw_checks = {
        "raw_control_finite_step_failed": not bool(
            raw_control["loss_summary"]["all_group_components_nonincreasing"]
        ),
        "raw_control_exceeds_rms_cap": float(
            raw_control["maintenance_change"]["normalized_exact_rms"]
        )
        > float(raw_control["maintenance_cap"]["normalized_exact_rms"]),
        "raw_control_exceeds_maximum_cap": float(
            raw_control["maintenance_change"]["maximum_absolute_exact_change"]
        )
        > float(raw_control["maintenance_cap"]["maximum_absolute_exact_change"]),
    }
    expected_norms = tuple(
        float(value) for value in gates["candidate_correction_norms"]
    )
    observed_norms = tuple(
        float(candidate["correction_norm"]) for candidate in candidates
    )
    candidate_set_checks = {
        "candidate_count_exact": len(candidates)
        == int(gates["expected_candidate_count"]),
        "candidate_norms_exact": observed_norms == expected_norms,
    }
    candidate_records = []
    for candidate in candidates:
        medians = candidate["loss_summary"]["component_median_relative_changes"]
        checks = {
            "all_constraints_nonincreasing": bool(
                candidate["loss_summary"]["all_group_components_nonincreasing"]
            ),
            "npp_effect": float(medians["npp"])
            <= -float(gates["minimum_npp_median_relative_decrease"]),
            "agb_effect": float(medians["aboveground_biomass_tendency"])
            <= -float(gates["minimum_agb_median_relative_decrease"]),
            "bgb_effect": float(medians["belowground_biomass_tendency"])
            <= -float(gates["minimum_bgb_median_relative_decrease"]),
            "maintenance_rms_within_cap": float(
                candidate["maintenance_change"]["normalized_exact_rms"]
            )
            <= float(candidate["maintenance_cap"]["normalized_exact_rms"]),
            "maintenance_maximum_within_cap": float(
                candidate["maintenance_change"]["maximum_absolute_exact_change"]
            )
            <= float(
                candidate["maintenance_cap"]["maximum_absolute_exact_change"]
            ),
            "displacement_retained": float(candidate["displacement_fraction"])
            >= float(gates["minimum_displacement_fraction"]),
            "hard_failures_zero": int(candidate["hard_failure_count"])
            == int(gates["hard_failure_count"]),
            "nonfinite_values_zero": int(candidate["nonfinite_value_count"])
            == int(gates["nonfinite_value_count"]),
        }
        candidate_records.append(
            {
                "correction_norm": float(candidate["correction_norm"]),
                "passed": all(checks.values()),
                "checks": checks,
            }
        )
    selected = next(
        (record for record in candidate_records if record["passed"]), None
    )
    evidence_valid = (
        all(safe_checks.values())
        and all(raw_checks.values())
        and all(candidate_set_checks.values())
    )
    passed = evidence_valid and selected is not None
    return {
        "status": (
            "maintenance_output_projection_supported"
            if passed
            else "maintenance_output_projection_not_supported"
        ),
        "passed": passed,
        "evidence_valid": evidence_valid,
        "safe_reference_checks": safe_checks,
        "raw_control_checks": raw_checks,
        "candidate_set_checks": candidate_set_checks,
        "candidate_checks": candidate_records,
        "selected_correction_norm": (
            selected["correction_norm"] if selected is not None else None
        ),
        "next_action": pass_action if passed else fail_action,
    }


def summarize_output_locality_group(
    *,
    fields: Sequence[Any],
    field_labels: Mapping[str, str],
    scales: Mapping[str, Any],
    base_outputs: Mapping[str, Any],
    output_tangents: Mapping[str, Any],
    proposed_outputs: Mapping[str, Any],
    output_gradients: Mapping[str, Any],
    loss_labels: Sequence[str],
    base_losses: Sequence[float],
    proposed_losses: Sequence[float],
    end_to_end_base_losses: Sequence[float],
    end_to_end_proposed_losses: Sequence[float],
    end_to_end_first_order_changes: Sequence[float],
    step_norm: float,
) -> Mapping[str, Any]:
    """Decompose finite loss changes into output and downstream nonlinearities."""

    labels = tuple(str(label) for label in loss_labels)
    if step_norm <= 0.0 or len(labels) == 0:
        raise ValueError("output-locality loss labels and step norm must be positive")
    expected_names = tuple(field.name for field in fields)
    if set(field_labels) != set(expected_names):
        raise ValueError("output-locality field label registry drift")
    registries = (
        scales,
        base_outputs,
        output_tangents,
        proposed_outputs,
        output_gradients,
    )
    if any(set(registry) != set(expected_names) for registry in registries):
        raise ValueError("output-locality field registry drift")
    loss_arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            base_losses,
            proposed_losses,
            end_to_end_base_losses,
            end_to_end_proposed_losses,
            end_to_end_first_order_changes,
        )
    )
    if any(values.shape != (len(labels),) for values in loss_arrays):
        raise ValueError("output-locality loss vector shape drift")

    field_records = []
    nonfinite_value_count = int(
        sum(np.count_nonzero(~np.isfinite(values)) for values in loss_arrays)
    )
    contribution_totals = {
        label: {"first_order": 0.0, "exact_linearized": 0.0, "output_nonlinearity": 0.0}
        for label in labels
    }
    for field in fields:
        name = field.name
        base = np.asarray(base_outputs[name], dtype=np.float64)
        tangent = np.asarray(output_tangents[name], dtype=np.float64)
        proposed = np.asarray(proposed_outputs[name], dtype=np.float64)
        scale = np.asarray(scales[name], dtype=np.float64)
        gradient = np.asarray(output_gradients[name], dtype=np.float64)
        if base.shape != tangent.shape or base.shape != proposed.shape:
            raise ValueError(f"output-locality trajectory shape drift for {name}")
        if gradient.shape != (len(labels),) + base.shape:
            raise ValueError(f"output-locality gradient shape drift for {name}")
        if scale.shape != tuple(field.shape):
            raise ValueError(f"output-locality scale shape drift for {name}")
        if field.shape and base.shape[-len(field.shape) :] != tuple(field.shape):
            raise ValueError(f"output-locality field tail shape drift for {name}")
        broadcast_scale = np.broadcast_to(scale, base.shape)
        if not np.all(np.isfinite(broadcast_scale) & (broadcast_scale > 0.0)):
            raise ValueError(f"output-locality scale is invalid for {name}")
        linear = step_norm * tangent
        exact = proposed - base
        residual = exact - linear
        normalized_linear = linear / broadcast_scale
        normalized_exact = exact / broadcast_scale
        normalized_residual = residual / broadcast_scale
        nonfinite_value_count += int(
            sum(
                np.count_nonzero(~np.isfinite(value))
                for value in (base, tangent, proposed, gradient)
            )
        )
        contributions = {}
        for index, label in enumerate(labels):
            first_order = float(np.sum(gradient[index] * linear, dtype=np.float64))
            exact_linearized = float(np.sum(gradient[index] * exact, dtype=np.float64))
            output_nonlinearity = exact_linearized - first_order
            contributions[label] = {
                "first_order": first_order,
                "exact_linearized": exact_linearized,
                "output_nonlinearity": output_nonlinearity,
            }
            for key, value in contributions[label].items():
                contribution_totals[label][key] += value
        field_records.append(
            {
                "name": name,
                "label": str(field_labels[name]),
                "process_domain": field.process_domain,
                "source_route": field.source_route,
                "constraint": field.constraint,
                "value_representation": field.value_representation,
                "scalar_count": int(base.size),
                "normalized_linear_squared_sum": float(
                    np.sum(normalized_linear * normalized_linear, dtype=np.float64)
                ),
                "normalized_exact_squared_sum": float(
                    np.sum(normalized_exact * normalized_exact, dtype=np.float64)
                ),
                "normalized_residual_squared_sum": float(
                    np.sum(normalized_residual * normalized_residual, dtype=np.float64)
                ),
                "normalized_linear_exact_dot": float(
                    np.sum(normalized_linear * normalized_exact, dtype=np.float64)
                ),
                "maximum_absolute_normalized_exact_change": float(
                    np.max(np.abs(normalized_exact)) if normalized_exact.size else 0.0
                ),
                "maximum_absolute_normalized_residual": float(
                    np.max(np.abs(normalized_residual)) if normalized_residual.size else 0.0
                ),
                "loss_contributions": contributions,
            }
        )

    base_loss, proposed_loss, end_base, end_proposed, first_order = loss_arrays
    decomposition = {}
    for index, label in enumerate(labels):
        totals = contribution_totals[label]
        exact_change = float(proposed_loss[index] - base_loss[index])
        end_exact_change = float(end_proposed[index] - end_base[index])
        downstream = exact_change - totals["exact_linearized"]
        finite_minus_first = exact_change - float(first_order[index])
        decomposition[label] = {
            "base_loss": float(base_loss[index]),
            "proposed_loss": float(proposed_loss[index]),
            "exact_loss_change": exact_change,
            "end_to_end_exact_loss_change": end_exact_change,
            "end_to_end_first_order_change": float(first_order[index]),
            "output_first_order_change": totals["first_order"],
            "output_exact_linearized_change": totals["exact_linearized"],
            "output_nonlinearity_component": totals["output_nonlinearity"],
            "downstream_finite_response_component": downstream,
            "finite_minus_first_order": finite_minus_first,
            "chain_rule_relative_error": _relative_error(
                totals["first_order"], float(first_order[index])
            ),
            "base_replay_relative_error": _relative_error(
                float(base_loss[index]), float(end_base[index])
            ),
            "proposed_replay_relative_error": _relative_error(
                float(proposed_loss[index]), float(end_proposed[index])
            ),
            "finite_change_replay_relative_error": _relative_error(
                exact_change, end_exact_change
            ),
        }
    return {
        "field_count": len(field_records),
        "field_records": field_records,
        "loss_decomposition": decomposition,
        "nonfinite_value_count": nonfinite_value_count,
    }


def aggregate_output_locality(
    groups: Sequence[Mapping[str, Any]],
    *,
    loss_labels: Sequence[str],
    top_k: int,
) -> Mapping[str, Any]:
    """Aggregate group records and rank fields by nonlinear loss contribution."""

    labels = tuple(str(label) for label in loss_labels)
    if not groups or top_k < 1:
        raise ValueError("output-locality aggregation requires groups and top_k")
    names = tuple(record["name"] for record in groups[0]["field_records"])
    if any(tuple(record["name"] for record in group["field_records"]) != names for group in groups):
        raise ValueError("output-locality group field order drift")
    by_name: dict[str, dict[str, Any]] = {}
    for field_index, name in enumerate(names):
        records = tuple(group["field_records"][field_index] for group in groups)
        scalar_count = sum(int(record["scalar_count"]) for record in records)
        linear_sq = sum(float(record["normalized_linear_squared_sum"]) for record in records)
        exact_sq = sum(float(record["normalized_exact_squared_sum"]) for record in records)
        residual_sq = sum(float(record["normalized_residual_squared_sum"]) for record in records)
        dot = sum(float(record["normalized_linear_exact_dot"]) for record in records)
        contributions = {}
        for label in labels:
            values = [
                float(record["loss_contributions"][label]["output_nonlinearity"])
                for record in records
            ]
            regressing = [
                value
                for value, group in zip(values, groups, strict=True)
                if float(group["loss_decomposition"][label]["exact_loss_change"]) > 0.0
            ]
            contributions[label] = {
                "signed_sum": float(sum(values)),
                "absolute_sum": float(sum(abs(value) for value in values)),
                "regressing_group_signed_sum": float(sum(regressing)),
                "regressing_group_absolute_sum": float(sum(abs(value) for value in regressing)),
            }
        template = records[0]
        by_name[name] = {
            "name": name,
            "label": template["label"],
            "process_domain": template["process_domain"],
            "source_route": template["source_route"],
            "constraint": template["constraint"],
            "value_representation": template["value_representation"],
            "scalar_count": scalar_count,
            "normalized_linear_rms": float(np.sqrt(linear_sq / max(scalar_count, 1))),
            "normalized_exact_rms": float(np.sqrt(exact_sq / max(scalar_count, 1))),
            "normalized_residual_rms": float(np.sqrt(residual_sq / max(scalar_count, 1))),
            "normalized_linearization_cosine": float(
                dot / max(np.sqrt(linear_sq * exact_sq), 1.0e-30)
            ),
            "maximum_absolute_normalized_exact_change": max(
                float(record["maximum_absolute_normalized_exact_change"])
                for record in records
            ),
            "maximum_absolute_normalized_residual": max(
                float(record["maximum_absolute_normalized_residual"])
                for record in records
            ),
            "loss_contributions": contributions,
        }

    rankings = {}
    decomposition = {}
    for label in labels:
        regressing = tuple(
            group
            for group in groups
            if float(group["loss_decomposition"][label]["exact_loss_change"]) > 0.0
        )
        ordered = sorted(
            by_name.values(),
            key=lambda record: record["loss_contributions"][label][
                "regressing_group_absolute_sum"
            ],
            reverse=True,
        )
        total = sum(
            record["loss_contributions"][label]["regressing_group_absolute_sum"]
            for record in ordered
        )
        top = ordered[: min(top_k, len(ordered))]
        rankings[label] = {
            "regressing_group_count": len(regressing),
            "absolute_contribution_sum": float(total),
            "top_k": len(top),
            "top_k_absolute_coverage": float(
                sum(
                    record["loss_contributions"][label][
                        "regressing_group_absolute_sum"
                    ]
                    for record in top
                )
                / max(total, 1.0e-30)
            ),
            "fields": [
                {
                    "name": record["name"],
                    "absolute_contribution": record["loss_contributions"][label][
                        "regressing_group_absolute_sum"
                    ],
                    "signed_contribution": record["loss_contributions"][label][
                        "regressing_group_signed_sum"
                    ],
                }
                for record in top
            ],
        }
        output_magnitude = sum(
            abs(
                float(group["loss_decomposition"][label]["output_nonlinearity_component"])
            )
            for group in regressing
        )
        downstream_magnitude = sum(
            abs(
                float(
                    group["loss_decomposition"][label][
                        "downstream_finite_response_component"
                    ]
                )
            )
            for group in regressing
        )
        decomposition[label] = {
            "output_nonlinearity_absolute_sum": float(output_magnitude),
            "downstream_finite_response_absolute_sum": float(downstream_magnitude),
            "output_nonlinearity_share": float(
                output_magnitude
                / max(output_magnitude + downstream_magnitude, 1.0e-30)
            ),
        }
    return {
        "group_count": len(groups),
        "field_count": len(names),
        "fields": list(by_name.values()),
        "rankings": rankings,
        "nonlinearity_decomposition": decomposition,
        "nonfinite_value_count": sum(
            int(group["nonfinite_value_count"]) for group in groups
        ),
    }


def classify_output_locality(
    aggregate: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    localized_action: str,
    distributed_action: str,
    downstream_action: str,
) -> Mapping[str, Any]:
    """Choose field-local, all-output, or downstream-observable metric scope."""

    key_labels = tuple(str(label) for label in gates["key_loss_labels"])
    reference_errors = [
        float(record[key])
        for group in groups
        for record in group["loss_decomposition"].values()
        for key in (
            "base_replay_relative_error",
            "proposed_replay_relative_error",
            "finite_change_replay_relative_error",
        )
    ]
    chain_errors = [
        float(record["chain_rule_relative_error"])
        for group in groups
        for record in group["loss_decomposition"].values()
    ]
    evidence_checks = {
        "group_count_exact": len(groups) == int(gates["expected_group_count"]),
        "field_count_exact": int(aggregate["field_count"])
        == int(gates["expected_field_count"]),
        "replay_reference_closed": max(reference_errors, default=np.inf)
        <= float(gates["maximum_replay_relative_error"]),
        "chain_rule_closed": max(chain_errors, default=np.inf)
        <= float(gates["maximum_chain_rule_relative_error"]),
        "hard_failures_zero": sum(
            int(group.get("hard_failure_count", 0)) for group in groups
        )
        == int(gates["hard_failure_count"]),
        "nonfinite_values_zero": int(aggregate["nonfinite_value_count"])
        == int(gates["nonfinite_value_count"]),
        "key_regression_reproduced": all(
            int(aggregate["rankings"][label]["regressing_group_count"])
            >= int(gates["minimum_regressing_groups_per_key"])
            for label in key_labels
        ),
    }
    valid = all(evidence_checks.values())
    dominant = {
        label: float(
            aggregate["nonlinearity_decomposition"][label][
                "output_nonlinearity_share"
            ]
        )
        >= float(gates["minimum_output_nonlinearity_share"])
        for label in key_labels
    }
    localized = {
        label: dominant[label]
        and float(aggregate["rankings"][label]["top_k_absolute_coverage"])
        >= float(gates["minimum_top_k_absolute_coverage"])
        for label in key_labels
    }
    if not valid:
        status = "invalid_output_locality_evidence"
        next_action = "stop_and_repair_output_locality_evidence"
    elif all(localized.values()):
        status = "decoded_output_nonlinearity_localized"
        next_action = localized_action
    elif any(dominant.values()):
        status = "decoded_output_nonlinearity_distributed"
        next_action = distributed_action
    else:
        status = "downstream_finite_response_dominant"
        next_action = downstream_action
    return {
        "status": status,
        "evidence_valid": valid,
        "evidence_checks": evidence_checks,
        "output_nonlinearity_dominant": dominant,
        "top_k_localized": localized,
        "next_action": next_action,
    }
