"""Matched snowheat intervention for the localized coupled-GPP jump."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .target_oriented_snowheat_recurrence import resolve_continuous_state_leaves

SCHEMA_VERSION = "target_oriented_coupled_gpp_snowheat_causal_intervention_result_v1"
TARGET_ARTIFACT_UPDATE = 550
TARGET_WINDOW_INDEX = 3
TARGET_DAY_INDEX = 4
SNOWHEAT_ALIAS_GROUP = (
    "hydrol_previous_step_state.snowheat",
    "diffuco_previous_step_state.snowheat",
)
CAUSAL_RETAINED_FRACTION_MAXIMUM = 0.1
NOT_EXPLANATORY_RETAINED_FRACTION_MINIMUM = 0.75
FACTORIZATION_FIELDS = (
    "normalization_scale",
    "potential",
    "limitation",
    "correction",
    "vcmax25_factor",
)


def resolve_coupled_gpp_snowheat_sources(
    markov_contract: Any,
    owner_contract: Any,
    layout: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """Resolve both aliases and verify that the GPP router admits each one."""

    leaves = resolve_continuous_state_leaves(markov_contract, SNOWHEAT_ALIAS_GROUP)
    owner_by_key = {leaf.key: leaf for leaf in owner_contract.leaves}
    canonical, mirror = SNOWHEAT_ALIAS_GROUP
    mirror_owner = owner_by_key.get(mirror)
    if (
        canonical not in owner_by_key
        or mirror_owner is None
        or mirror_owner.role != "mirror_of_canonical_owner"
        or mirror_owner.canonical_owner != canonical
    ):
        raise ValueError("coupled-GPP snowheat mirror ownership drift")

    continuous = tuple(leaf for leaf in markov_contract.state_leaves if not leaf.discrete)
    canopy_index = layout.expert_names.index("canopy_gpp")
    expected_families = {
        canonical: "state_water",
        mirror: "state_canopy",
    }
    records = {}
    for leaf in leaves:
        field_index = next(
            (index for index, candidate in enumerate(continuous) if candidate.key == leaf.key),
            None,
        )
        if field_index is None:
            raise ValueError(f"coupled-GPP snowheat state field is missing: {leaf.key}")
        family_index = int(layout.state_field_family_index[field_index])
        family = str(layout.source_family_names[family_index])
        admitted = bool(layout.source_allowed[canopy_index, family_index])
        if family != expected_families[leaf.key] or not admitted:
            raise ValueError(f"coupled-GPP snowheat source route drift: {leaf.key}")
        records[leaf.key] = {
            "start": int(leaf.start),
            "stop": int(leaf.stop),
            "shape": list(leaf.shape),
            "axis_names": list(leaf.axis_names),
            "source_family": family,
            "canopy_gpp_admitted": admitted,
        }
    return leaves, records


def classify_snowheat_day5_intervention(
    *,
    control_gpp: float,
    treatment_gpp: float,
    matched_snowheat_gpp: float,
) -> Mapping[str, Any]:
    values = np.asarray(
        (control_gpp, treatment_gpp, matched_snowheat_gpp),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("snowheat intervention GPP values must be finite")
    free = abs(float(values[1] - values[0]))
    cut = abs(float(values[2] - values[0]))
    if free <= 0.0:
        raise ValueError("snowheat intervention requires a nonzero free GPP jump")
    retained = cut / free
    if retained <= CAUSAL_RETAINED_FRACTION_MAXIMUM:
        classification = "snowheat_path_causal_for_day5_jump"
    elif retained >= NOT_EXPLANATORY_RETAINED_FRACTION_MINIMUM:
        classification = "snowheat_path_not_explanatory_for_day5_jump"
    else:
        classification = "snowheat_path_partial_for_day5_jump"
    return {
        "classification": classification,
        "free_absolute_gpp_jump": free,
        "matched_snowheat_absolute_gpp_jump": cut,
        "retained_jump_fraction": retained,
        "removed_jump_fraction": 1.0 - retained,
    }


def factor_log_attribution(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    pft_index: int,
) -> Mapping[str, Any]:
    """Decompose a positive routed-GPP ratio into existing multiplicative factors."""

    index = int(pft_index)
    contributions = {}
    for name in FACTORIZATION_FIELDS:
        before = float(np.asarray(control[name]).reshape(-1)[index])
        after = float(np.asarray(treatment[name]).reshape(-1)[index])
        if not np.isfinite(before) or not np.isfinite(after) or before <= 0.0 or after <= 0.0:
            raise ValueError(f"GPP factor {name} must be finite and positive")
        contributions[name] = {
            "control": before,
            "treatment": after,
            "log_ratio": float(np.log(after / before)),
        }
    routed_control = float(np.asarray(control["routed_gpp"]).reshape(-1)[index])
    routed_treatment = float(np.asarray(treatment["routed_gpp"]).reshape(-1)[index])
    if routed_control <= 0.0 or routed_treatment <= 0.0:
        raise ValueError("routed GPP must be positive for factor attribution")
    observed = float(np.log(routed_treatment / routed_control))
    reconstructed = float(sum(value["log_ratio"] for value in contributions.values()))
    dominant = max(contributions, key=lambda name: abs(contributions[name]["log_ratio"]))
    return {
        "pft_index": index,
        "routed_gpp_log_ratio": observed,
        "factor_log_ratio_sum": reconstructed,
        "factor_log_ratio_residual": observed - reconstructed,
        "dominant_absolute_log_ratio_factor": dominant,
        "factors": contributions,
    }


__all__ = [
    "CAUSAL_RETAINED_FRACTION_MAXIMUM",
    "FACTORIZATION_FIELDS",
    "NOT_EXPLANATORY_RETAINED_FRACTION_MINIMUM",
    "SCHEMA_VERSION",
    "SNOWHEAT_ALIAS_GROUP",
    "TARGET_ARTIFACT_UPDATE",
    "TARGET_DAY_INDEX",
    "TARGET_WINDOW_INDEX",
    "classify_snowheat_day5_intervention",
    "factor_log_attribution",
    "resolve_coupled_gpp_snowheat_sources",
]
