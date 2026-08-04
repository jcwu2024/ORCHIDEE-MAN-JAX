"""Non-neural Gate-C replay through a conservative daily boundary.

Teacher inventory endpoints are encoded as bounded outgoing fractions and
nonnegative incoming amounts.  Captured process terms remain separate labels:
an endpoint tendency is never renamed as a process flux.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from jax_orchidee.sechiba.hydrol import hydrol_initial_tmc_from_restart_mc
from jax_orchidee.stomate.soilcarbon_kernels import IDOCL, IDOCR
from research.daily_coarse_graining.daily_flux_capture import (
    daily_flux_capture_arrays,
    validate_daily_flux_capture,
)
from research.daily_coarse_graining.replay_ceiling import (
    PreDailyStomateReplayRecord,
)

WATER_INVENTORY_FIELDS = (
    "qsintveg",
    "mc",
    "mcl",
    "snow",
    "snow_nobio",
    "snowliq",
    "flood_res",
)
CARBON_INVENTORY_FIELDS = (
    "litter_above",
    "litter_below",
    "carbon_32l",
    "DOC",
    "interception_storage",
)
THERMAL_FIELDS: tuple[tuple[str, str, float], ...] = (
    ("enerbil_previous_step_state", "temp_sol", 100.0),
    ("thermosoil_previous_step_state", "ptn", 100.0),
    ("thermosoil_previous_step_state", "stempdiag", 100.0),
    ("hydrol_previous_step_state", "snowtemp", 100.0),
    ("hydrol_previous_step_state", "snowheat", 1.0e9),
)
ORCHIDEE_UNDEFINED_MAGNITUDE = 1.0e19


@dataclass(frozen=True)
class BoundedInventoryTransition:
    outgoing_fraction: Any
    incoming_amount: Any
    defined: Any
    reconstructed: Any
    numerical_zero_lower_bound: float


@dataclass(frozen=True)
class BoundedTendencyTransition:
    normalized_tendency: Any
    bound: float
    defined: Any
    reconstructed: Any


@dataclass(frozen=True)
class SignedInventoryTransition:
    positive: BoundedInventoryTransition
    debt: BoundedInventoryTransition
    reconstructed: Any


@dataclass(frozen=True)
class ConstrainedReplayBoundary:
    record: PreDailyStomateReplayRecord
    inventory_transitions: Mapping[str, BoundedInventoryTransition | SignedInventoryTransition]
    thermal_transitions: Mapping[str, BoundedTendencyTransition]
    report: Mapping[str, Any]


def apply_bounded_inventory_transition(
    start: Any,
    outgoing_fraction: Any,
    incoming_amount: Any,
) -> Any:
    """Apply one conservative stock update without post-hoc clipping."""

    start = jnp.asarray(start)
    outgoing_fraction = jnp.asarray(outgoing_fraction, dtype=start.dtype)
    incoming_amount = jnp.asarray(incoming_amount, dtype=start.dtype)
    return start * (1.0 - outgoing_fraction) + incoming_amount


def apply_bounded_tendency(
    start: Any,
    normalized_tendency: Any,
    bound: float,
) -> Any:
    """Apply a signed tendency constrained to a declared daily bound."""

    start = jnp.asarray(start)
    return start + jnp.asarray(normalized_tendency, dtype=start.dtype) * jnp.asarray(bound, dtype=start.dtype)


def encode_inventory_endpoints(
    start: Any,
    end: Any,
    *,
    name: str,
    atol: float = 1.0e-12,
) -> BoundedInventoryTransition:
    """Encode valid Teacher endpoints in the conservative stock parameterization."""

    start_array = np.asarray(start)
    end_array = np.asarray(end)
    if start_array.shape != end_array.shape:
        raise ValueError(f"{name} endpoint shape drift")
    start_defined = np.isfinite(start_array) & (np.abs(start_array) < ORCHIDEE_UNDEFINED_MAGNITUDE)
    end_defined = np.isfinite(end_array) & (np.abs(end_array) < ORCHIDEE_UNDEFINED_MAGNITUDE)
    if not np.array_equal(start_defined, end_defined):
        raise ValueError(f"{name} changed defined status inside an inventory update")
    if np.any(start_defined & (start_array < -atol)):
        minimum = float(np.min(start_array[start_defined]))
        raise ValueError(f"{name} has a negative day-start inventory: {minimum:g}")
    if np.any(end_defined & (end_array < -atol)):
        minimum = float(np.min(end_array[end_defined]))
        raise ValueError(f"{name} has a negative Teacher endpoint: {minimum:g}")

    shifted_start = start_array + float(atol)
    shifted_end = end_array + float(atol)
    decrease = start_defined & (shifted_end < shifted_start)
    denominator = np.where(decrease, shifted_start, 1.0)
    outgoing_fraction = np.where(
        decrease,
        (shifted_start - shifted_end) / denominator,
        0.0,
    )
    incoming_amount = np.where(
        start_defined & ~decrease,
        shifted_end - shifted_start,
        0.0,
    )
    if np.any(outgoing_fraction[start_defined] < 0.0) or np.any(outgoing_fraction[start_defined] > 1.0):
        raise ValueError(f"{name} outgoing fraction is outside [0, 1]")
    if np.any(incoming_amount[start_defined] < 0.0):
        raise ValueError(f"{name} incoming amount is negative")
    reconstructed = np.asarray(
        apply_bounded_inventory_transition(
            np.where(start_defined, shifted_start, 0.0),
            outgoing_fraction,
            incoming_amount,
        )
    )
    reconstructed = np.where(start_defined, reconstructed - float(atol), end_array)
    if not np.allclose(reconstructed, end_array, atol=atol, rtol=1.0e-12, equal_nan=True):
        raise ValueError(f"{name} bounded reconstruction does not recover its endpoint")
    return BoundedInventoryTransition(
        outgoing_fraction=outgoing_fraction,
        incoming_amount=incoming_amount,
        defined=start_defined,
        reconstructed=reconstructed,
        numerical_zero_lower_bound=-float(atol),
    )


def encode_thermal_endpoints(
    start: Any,
    end: Any,
    *,
    name: str,
    bound: float,
    atol: float = 1.0e-12,
) -> BoundedTendencyTransition:
    """Encode exact Teacher thermal endpoints as bounded daily tendencies."""

    start_array = np.asarray(start)
    end_array = np.asarray(end)
    if start_array.shape != end_array.shape:
        raise ValueError(f"{name} endpoint shape drift")
    start_defined = np.isfinite(start_array) & (np.abs(start_array) < ORCHIDEE_UNDEFINED_MAGNITUDE)
    end_defined = np.isfinite(end_array) & (np.abs(end_array) < ORCHIDEE_UNDEFINED_MAGNITUDE)
    if not np.array_equal(start_defined, end_defined):
        raise ValueError(f"{name} changed defined status")
    normalized = np.where(start_defined, (end_array - start_array) / float(bound), 0.0)
    if np.any(np.abs(normalized[start_defined]) > 1.0):
        maximum = float(np.max(np.abs(end_array[start_defined] - start_array[start_defined])))
        raise ValueError(f"{name} exceeds its {bound:g} daily bound: {maximum:g}")
    reconstructed = np.asarray(
        apply_bounded_tendency(
            np.where(start_defined, start_array, 0.0),
            normalized,
            bound,
        )
    )
    reconstructed = np.where(start_defined, reconstructed, end_array)
    if not np.allclose(reconstructed, end_array, atol=atol, rtol=1.0e-12, equal_nan=True):
        raise ValueError(f"{name} bounded tendency does not recover its endpoint")
    return BoundedTendencyTransition(
        normalized_tendency=normalized,
        bound=float(bound),
        defined=start_defined,
        reconstructed=reconstructed,
    )


def encode_signed_inventory_endpoints(
    start: Any,
    end: Any,
    *,
    name: str,
    atol: float = 1.0e-12,
) -> SignedInventoryTransition:
    """Represent a source-defined signed carry as two nonnegative inventories."""

    start_array = np.asarray(start)
    end_array = np.asarray(end)
    positive = encode_inventory_endpoints(
        np.maximum(start_array, 0.0),
        np.maximum(end_array, 0.0),
        name=f"{name}.storage",
        atol=0.0,
    )
    debt = encode_inventory_endpoints(
        np.maximum(-start_array, 0.0),
        np.maximum(-end_array, 0.0),
        name=f"{name}.evaporation_debt",
        atol=0.0,
    )
    reconstructed = np.asarray(positive.reconstructed) - np.asarray(debt.reconstructed)
    if not np.allclose(reconstructed, end_array, atol=atol, rtol=1.0e-12, equal_nan=True):
        raise ValueError(f"{name} signed inventory decomposition is not exact")
    return SignedInventoryTransition(
        positive=positive,
        debt=debt,
        reconstructed=reconstructed,
    )


def _packet_with_fields(packet: Any, fields: Mapping[str, Mapping[str, Any]]) -> Any:
    return teacher.DriverPreviousStepStatePacket(
        tstep=int(packet.tstep),
        fields_by_component=dict(fields),
        provenance_by_component=packet.provenance_by_component,
    )


def constrained_replay_boundary(
    previous_state: Any,
    record: PreDailyStomateReplayRecord,
    *,
    inventory_path: str,
) -> ConstrainedReplayBoundary:
    """Rebuild the captured fast boundary through constrained daily updates."""

    if record.daily_flux_labels is None:
        raise ValueError("constrained replay requires complete daily flux labels")
    capture_report = validate_daily_flux_capture(
        record.daily_flux_labels,
        inventory_path=inventory_path,
    )
    start_fields = previous_state.fields_by_component
    target_packet = record.half_hour_transition.current_state
    target_fields = target_packet.fields_by_component
    rebuilt_fields = {component: dict(values) for component, values in target_fields.items()}
    inventories: dict[str, BoundedInventoryTransition | SignedInventoryTransition] = {}

    start_water = start_fields["hydrol_previous_step_state"]
    target_water = target_fields["hydrol_previous_step_state"]
    for field in WATER_INVENTORY_FIELDS:
        transition = (
            encode_signed_inventory_endpoints(
                start_water[field],
                target_water[field],
                name=f"water.{field}",
            )
            if field == "qsintveg"
            else encode_inventory_endpoints(
                start_water[field],
                target_water[field],
                name=f"water.{field}",
            )
        )
        inventories[f"water.{field}"] = transition
        rebuilt_fields["hydrol_previous_step_state"][field] = transition.reconstructed

    rebuilt_updates = dict(record.ok_leak_updates)
    start_carbon = start_fields["slowproc_stomate_previous_step_state"]
    for field in CARBON_INVENTORY_FIELDS:
        transition = encode_inventory_endpoints(
            start_carbon[field],
            record.ok_leak_updates[field],
            name=f"carbon.{field}",
        )
        inventories[f"carbon.{field}"] = transition
        rebuilt_updates[field] = transition.reconstructed

    thermal: dict[str, BoundedTendencyTransition] = {}
    for component, field, bound in THERMAL_FIELDS:
        if field not in start_fields[component] or field not in target_fields[component]:
            continue
        transition = encode_thermal_endpoints(
            start_fields[component][field],
            target_fields[component][field],
            name=f"{component}.{field}",
            bound=bound,
        )
        thermal[f"{component}.{field}"] = transition
        rebuilt_fields[component][field] = transition.reconstructed

    rebuilt_packet = _packet_with_fields(target_packet, rebuilt_fields)
    rebuilt_transition = replace(
        record.half_hour_transition,
        current_state=rebuilt_packet,
    )
    rebuilt_record = replace(
        record,
        half_hour_transition=rebuilt_transition,
        ok_leak_updates=rebuilt_updates,
    )
    nonnegative_components = []
    for value in inventories.values():
        if isinstance(value, SignedInventoryTransition):
            nonnegative_components.extend(
                (
                    np.asarray(value.positive.reconstructed)[np.asarray(value.positive.defined)].reshape(-1),
                    np.asarray(value.debt.reconstructed)[np.asarray(value.debt.defined)].reshape(-1),
                )
            )
        else:
            nonnegative_components.append(np.asarray(value.reconstructed)[np.asarray(value.defined)].reshape(-1))
    all_inventory_values = np.concatenate(nonnegative_components)
    report = {
        "schema_version": "constrained_daily_replay_boundary_v1",
        "capture_inventory_complete": bool(capture_report["ok"]),
        "capture_label_count": int(capture_report["required_capture_label_count"]),
        "inventory_field_count": len(inventories),
        "thermal_field_count": len(thermal),
        "minimum_nonnegative_inventory_component": float(np.min(all_inventory_values)),
        "inventory_numerical_zero_lower_bound": -1.0e-12,
        "signed_source_carries": ["water.qsintveg"],
        "signed_source_carry_representation": ("nonnegative storage minus nonnegative evaporation debt"),
        "post_hoc_clipping_used": False,
        "stores_or_reconstructs_48_step_axis": False,
        "endpoint_differences_relabelled_as_process_fluxes": False,
    }
    return ConstrainedReplayBoundary(
        record=rebuilt_record,
        inventory_transitions=inventories,
        thermal_transitions=thermal,
        report=report,
    )


def _water_inventory(packet: Any, *, dz_mm: Any) -> np.ndarray:
    fields = packet.fields_by_component
    hydrol = fields["hydrol_previous_step_state"]
    slow = fields["slowproc_stomate_previous_step_state"]
    tmc = np.asarray(
        hydrol_initial_tmc_from_restart_mc(
            mc=hydrol["mc"],
            water2infilt=hydrol["water2infilt"],
            dz_mm=dz_mm,
        )
    )
    vegetated_fraction = np.sum(np.asarray(slow["veget_max"]), axis=1)
    soil = vegetated_fraction * np.sum(tmc * np.asarray(slow["soiltile"]), axis=1)
    return (
        soil
        + np.sum(np.asarray(hydrol["qsintveg"]), axis=1)
        + np.asarray(hydrol["snow"])
        + np.sum(np.asarray(hydrol["snow_nobio"]), axis=1)
    )


def audit_daily_water_budget(
    previous_state: Any,
    record: PreDailyStomateReplayRecord,
    *,
    dz_mm: Any,
) -> dict[str, Any]:
    """Audit the source HYDROL combined daily inventory equation."""

    water = record.daily_flux_labels.sechiba.water
    initial = _water_inventory(previous_state, dz_mm=dz_mm)
    final = _water_inventory(record.half_hour_transition.current_state, dz_mm=dz_mm)
    external_input = (
        np.sum(np.asarray(water.precip2canopy), axis=1)
        + np.sum(np.asarray(water.precip2ground), axis=1)
        + np.sum(np.asarray(water.returnflow), axis=1)
        + np.sum(np.asarray(water.reinfiltration), axis=1)
        + np.sum(np.asarray(water.irrigation), axis=1)
        + np.asarray(water.floodout)
    )
    external_output = (
        np.sum(np.asarray(water.vevapwet), axis=1)
        + np.sum(np.asarray(water.transpir), axis=1)
        + np.asarray(water.vevapnu)
        + np.asarray(water.vevapsno)
        + np.asarray(water.vevapflo)
        + np.asarray(water.runoff)
        + np.asarray(water.drainage)
    )
    residual = final - initial - external_input + external_output
    return {
        "initial_inventory": initial,
        "final_inventory": final,
        "external_input": external_input,
        "external_output": external_output,
        "residual": residual,
        "max_absolute_residual": float(np.max(np.abs(residual))),
        "source_equation": "HYDROL combined canopy+soil+snow inventory; internal transfers cancel",
    }


def _weighted_carbon_inventory(values: Mapping[str, Any], veget_max: Any) -> np.ndarray:
    veget = np.asarray(veget_max)
    litter = np.sum(np.asarray(values["litter_above"])[..., 0] * veget[:, None, :], axis=(1, 2))
    litter += np.sum(
        np.asarray(values["litter_below"])[..., 0] * veget[:, None, :, None],
        axis=(1, 2, 3),
    )
    particulate = np.sum(
        np.asarray(values["carbon_32l"]) * veget[:, None, :, None],
        axis=(1, 2, 3),
    )
    dissolved = np.sum(
        np.asarray(values["DOC"])[..., 0] * veget[:, :, None, None, None],
        axis=(1, 2, 3, 4),
    )
    canopy = np.sum(np.asarray(values["interception_storage"])[..., 0], axis=1)
    return litter + particulate + dissolved + canopy


def audit_daily_carbon_budget(
    previous_state: Any,
    record: PreDailyStomateReplayRecord,
    *,
    cue: float = 0.3,
) -> dict[str, Any]:
    """Audit the independent OK_LEAK carbon inventory from captured amounts."""

    start = previous_state.fields_by_component["slowproc_stomate_previous_step_state"]
    end = record.ok_leak_updates
    veget = np.asarray(start["veget_max"])
    transfers = record.daily_flux_labels.ok_leak
    initial = _weighted_carbon_inventory(start, veget)
    final = _weighted_carbon_inventory(end, veget)
    litter_input = np.sum(
        (np.asarray(start["turnover_daily"])[..., 0] + np.asarray(start["bm_to_litter"])[..., 0]) * veget[:, :, None],
        axis=(1, 2),
    )
    routing_input = (
        np.asarray(transfers.doc_to_topsoil)[:, IDOCL]
        + np.asarray(transfers.doc_to_topsoil)[:, IDOCR]
        + np.asarray(transfers.doc_to_subsoil)[:, IDOCL]
        + np.asarray(transfers.doc_to_subsoil)[:, IDOCR]
    )
    deposition_input = np.sum(
        np.asarray(transfers.doc_precip2ground)
        + np.asarray(transfers.doc_precip2canopy)
        + np.asarray(transfers.dry_dep_canopy),
        axis=(1, 2),
    )
    # wet_dep_ground/flood are destination diagnostics after canopy transfer,
    # not additional external input beside the captured deposition sources.
    external_input = litter_input + routing_input + deposition_input
    litter_loss = np.sum(
        (np.sum(np.asarray(transfers.litter_respiration), axis=2) + np.asarray(transfers.litter_flood_respiration))
        * veget,
        axis=1,
    )
    poc_loss = np.sum(
        (1.0 - float(cue))
        * (np.asarray(transfers.poc_gross_decomposition) + np.asarray(transfers.poc_flood_gross_decomposition))
        * veget[:, None, None, :, None],
        axis=(1, 2, 3, 4),
    )
    doc_loss = np.sum(
        (1.0 - float(cue))
        * (np.asarray(transfers.doc_gross_decomposition) + np.asarray(transfers.doc_flood_gross_decomposition))
        * veget[:, :, None, None],
        axis=(1, 2, 3),
    )
    atmospheric_loss = litter_loss + poc_loss + doc_loss
    export = np.sum(
        (np.asarray(transfers.doc_run) + np.asarray(transfers.doc_drain) + np.asarray(transfers.doc_flood))
        * veget[:, :, None, None],
        axis=(1, 2, 3),
    ) + np.sum(
        np.asarray(transfers.doc_run_2_peat) * veget[:, :, None, None, None],
        axis=(1, 2, 3, 4),
    )
    residual = initial + external_input - atmospheric_loss - export - final
    return {
        "initial_inventory": initial,
        "final_inventory": final,
        "external_input": external_input,
        "atmospheric_loss": atmospheric_loss,
        "lateral_export": export,
        "residual": residual,
        "max_absolute_residual": float(np.max(np.abs(residual))),
        "cue": float(cue),
        "source_equation": "combined independent litter+POC+DOC+canopy carbon inventory",
    }


def audit_daily_thermal_residual(record: PreDailyStomateReplayRecord) -> dict[str, Any]:
    """Declare the flux-side thermal residual without inventing heat capacity."""

    energy = record.daily_flux_labels.sechiba.energy
    net = np.asarray(energy.netrad)
    precipitation_heat = np.asarray(energy.precipitation_snow_heat)
    sensible = np.asarray(energy.fluxsens)
    latent = np.asarray(energy.fluxlat) + np.asarray(energy.fluxsubli)
    ground = np.asarray(energy.pgflux)
    phase = (
        np.asarray(energy.fusion)
        + np.asarray(energy.snow_melt_refreeze)
        + np.sum(np.asarray(energy.snow_liquid_excess), axis=1)
    )
    flux_side_residual = net + precipitation_heat - sensible - latent - ground - phase
    return {
        "net_radiation": net,
        "precipitation_snow_heat": precipitation_heat,
        "sensible": sensible,
        "latent": latent,
        "ground": ground,
        "phase_change": phase,
        "flux_side_residual": flux_side_residual,
        "max_absolute_flux_side_residual": float(np.max(np.abs(flux_side_residual))),
        "flux_identity_closure_required": True,
        "temperature_energy_closure_required": False,
        "semantics": (
            "diagnostic flux-side residual only; temperature endpoints are bounded "
            "tendencies and are not converted to energy without source heat-capacity state"
        ),
    }


def capture_defined_status(labels: Any) -> dict[str, np.ndarray]:
    """Return explicit defined masks for every captured daily process tensor."""

    return {name: np.isfinite(value) for name, value in daily_flux_capture_arrays(labels).items()}
