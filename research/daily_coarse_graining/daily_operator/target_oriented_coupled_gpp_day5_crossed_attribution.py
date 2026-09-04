"""Day-5 parameter/state crossed attribution for the coupled GPP head."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_structured_gpp_state_feedback_attribution import (
    SOURCE_COMPONENTS,
    resolve_component_state_masks,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_day5_crossed_attribution_result_v1"
STATE_COMPONENT_COORDINATE_COUNTS = {
    "state_canopy": 52,
    "state_energy": 141,
    "state_water": 622,
    "state_thermal": 512,
    "state_carbon": 2_386,
    "state_driver": 11,
}
PARAMETER_FAMILY_COUNTS = {
    "structured_gpp": 12_879_363,
    "coupled_gpp_carbon_feedback": 3_080_832,
}
INTERPOLATION_FRACTIONS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
STATE_DOMINANT_RETAINED_FRACTION_MAXIMUM = 0.1
STATE_NOT_EXPLANATORY_RETAINED_FRACTION_MINIMUM = 0.75
INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM = 0.1
MINIMUM_EFFECT = 1.0e-12


def resolve_crossed_gpp_state_components(
    markov_contract: Any,
    layout: Any,
    components: Sequence[tuple[str, str]] = SOURCE_COMPONENTS,
) -> Mapping[str, Mapping[str, Any]]:
    """Resolve the six state-owner masks and verify their canopy-GPP routes."""

    resolved = resolve_component_state_masks(markov_contract, components)
    continuous = tuple(leaf for leaf in markov_contract.state_leaves if not leaf.discrete)
    canopy_index = layout.expert_names.index("canopy_gpp")
    records = {}
    masks = []
    for name, component in components:
        record = resolved[name]
        matching = tuple(
            index for index, leaf in enumerate(continuous) if leaf.component == component
        )
        if not matching:
            raise ValueError(f"crossed attribution state component is missing: {component}")
        family_indices = {
            int(layout.state_field_family_index[index]) for index in matching
        }
        family_names = {str(layout.source_family_names[index]) for index in family_indices}
        admitted = all(
            bool(layout.source_allowed[canopy_index, index]) for index in family_indices
        )
        count = int(record["state_coordinate_count"])
        if (
            family_names != {name}
            or not admitted
            or count != STATE_COMPONENT_COORDINATE_COUNTS[name]
        ):
            raise ValueError(f"crossed attribution source route drift: {name}")
        mask = np.asarray(record["mask"], dtype=np.bool_)
        masks.append(mask)
        records[name] = {
            **record,
            "source_family": name,
            "canopy_gpp_admitted": True,
        }
    occupancy = np.sum(np.stack(masks, axis=0), axis=0)
    if np.any(occupancy > 1):
        raise ValueError("crossed attribution state-owner masks overlap")
    return records


def classify_parameter_state_cross(
    *,
    gpp_cc: float,
    gpp_tc: float,
    gpp_ct: float,
    gpp_tt: float,
) -> Mapping[str, Any]:
    """Classify the four-cell parameter/state cross and return Shapley terms."""

    values = np.asarray((gpp_cc, gpp_tc, gpp_ct, gpp_tt), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("crossed attribution GPP values must be finite")
    jump = float(values[3] - values[0])
    if abs(jump) <= MINIMUM_EFFECT:
        raise ValueError("crossed attribution requires a nonzero free GPP jump")
    matched_state_retained = abs(float(values[1] - values[0])) / abs(jump)
    if matched_state_retained <= STATE_DOMINANT_RETAINED_FRACTION_MAXIMUM:
        classification = "state_path_dominant_for_day5_jump"
    elif matched_state_retained >= STATE_NOT_EXPLANATORY_RETAINED_FRACTION_MINIMUM:
        classification = "state_path_not_explanatory_for_day5_jump"
    else:
        classification = "mixed_parameter_state_path_for_day5_jump"

    parameter = 0.5 * float((values[1] - values[0]) + (values[3] - values[2]))
    state = 0.5 * float((values[2] - values[0]) + (values[3] - values[1]))
    interaction = float(values[3] - values[2] - values[1] + values[0])
    return {
        "classification": classification,
        "free_signed_gpp_jump": jump,
        "free_absolute_gpp_jump": abs(jump),
        "matched_state_retained_jump_fraction": matched_state_retained,
        "matched_state_removed_jump_fraction": 1.0 - matched_state_retained,
        "parameter_shapley_effect": parameter,
        "state_shapley_effect": state,
        "parameter_shapley_fraction": parameter / jump,
        "state_shapley_fraction": state / jump,
        "two_factor_interaction": interaction,
        "shapley_additivity_residual": jump - parameter - state,
        "state_effect_at_control_parameters": float(values[2] - values[0]),
        "state_effect_at_treatment_parameters": float(values[3] - values[1]),
        "parameter_effect_at_control_state": float(values[1] - values[0]),
        "parameter_effect_at_treatment_state": float(values[3] - values[2]),
    }


def classify_state_owner_controls(
    *,
    gpp_treatment_control_state: float,
    gpp_treatment_treatment_state: float,
    matched_owner_gpp: Mapping[str, float],
    transplanted_owner_gpp: Mapping[str, float],
    state_path_classification: str,
) -> Mapping[str, Any]:
    """Measure owner necessity and sufficiency at fixed treatment parameters."""

    if tuple(matched_owner_gpp) != tuple(STATE_COMPONENT_COORDINATE_COUNTS):
        raise ValueError("matched state-owner registry drift")
    if tuple(transplanted_owner_gpp) != tuple(STATE_COMPONENT_COORDINATE_COUNTS):
        raise ValueError("transplanted state-owner registry drift")
    baseline = float(gpp_treatment_control_state)
    treatment = float(gpp_treatment_treatment_state)
    full_effect = treatment - baseline
    finite = np.asarray(
        (baseline, treatment, *matched_owner_gpp.values(), *transplanted_owner_gpp.values()),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(finite)):
        raise ValueError("state-owner GPP values must be finite")
    eligible = abs(full_effect) > MINIMUM_EFFECT
    state_allowed = state_path_classification != "state_path_not_explanatory_for_day5_jump"
    records = {}
    selected = []
    for name in STATE_COMPONENT_COORDINATE_COUNTS:
        matched = float(matched_owner_gpp[name])
        transplanted = float(transplanted_owner_gpp[name])
        if eligible:
            residual = matched - baseline
            necessity = 1.0 - abs(residual) / abs(full_effect)
            sufficiency = abs(transplanted - baseline) / abs(full_effect)
        else:
            residual = necessity = sufficiency = None
        material = bool(
            state_allowed
            and eligible
            and max(float(necessity), float(sufficiency))
            >= INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM
        )
        if material:
            selected.append(name)
        records[name] = {
            "matched_owner_gpp": matched,
            "transplanted_owner_gpp": transplanted,
            "residual_state_effect_after_match": residual,
            "necessity_capture_fraction": necessity,
            "sufficiency_fraction": sufficiency,
            "individually_material": material,
        }
    return {
        "eligible": eligible,
        "full_state_effect_at_treatment_parameters": full_effect,
        "selected_components": selected,
        "components": records,
    }


def parameter_family_controls(
    *,
    control_gpp: float,
    treatment_gpp: float,
    treatment_family_only_gpp: Mapping[str, float],
) -> Mapping[str, Any]:
    """Resolve two-family necessity, sufficiency, and interaction at one state."""

    families = tuple(PARAMETER_FAMILY_COUNTS)
    if tuple(treatment_family_only_gpp) != families:
        raise ValueError("parameter-family control registry drift")
    values = np.asarray(
        (control_gpp, treatment_gpp, *treatment_family_only_gpp.values()),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("parameter-family GPP values must be finite")
    full_effect = float(treatment_gpp - control_gpp)
    eligible = abs(full_effect) > MINIMUM_EFFECT
    records = {}
    single_effects = {}
    for index, name in enumerate(families):
        other = families[1 - index]
        value = float(treatment_family_only_gpp[name])
        single = value - float(control_gpp)
        single_effects[name] = single
        records[name] = {
            "treatment_family_only_gpp": value,
            "single_family_effect": single,
            "signed_sufficiency_fraction": single / full_effect if eligible else None,
            "necessity_capture_fraction": (
                1.0
                - abs(float(treatment_family_only_gpp[other]) - float(control_gpp))
                / abs(full_effect)
                if eligible
                else None
            ),
        }
    interaction = full_effect - sum(single_effects.values())
    return {
        "eligible": eligible,
        "full_parameter_effect": full_effect,
        "two_family_interaction": interaction,
        "interaction_fraction": interaction / full_effect if eligible else None,
        "families": records,
    }


__all__ = [
    "INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM",
    "INTERPOLATION_FRACTIONS",
    "MINIMUM_EFFECT",
    "PARAMETER_FAMILY_COUNTS",
    "SCHEMA_VERSION",
    "STATE_COMPONENT_COORDINATE_COUNTS",
    "STATE_DOMINANT_RETAINED_FRACTION_MAXIMUM",
    "STATE_NOT_EXPLANATORY_RETAINED_FRACTION_MINIMUM",
    "classify_parameter_state_cross",
    "classify_state_owner_controls",
    "parameter_family_controls",
    "resolve_crossed_gpp_state_components",
]
