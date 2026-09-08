"""Source-backed warm, snow-free PFT14 boundary; no subdaily reconstruction.

Fortran provenance: condveg.f90::condveg_frac_snow 872-890;
diffuco.f90::diffuco_snow 1216-1275 and diffuco_comb 3135-3183;
enerbil.f90::enerbil_evapveg 1760; explicitsnow.f90::explicitsnow_main
222-235,259-315,395-402,459-500, explicitsnow_fall 1198-1266,
explicitsnow_profile 1681-1690, explicitsnow_melt_refrz 1441-1558.
All files are in fortran_source/ORCHIDEE/src_sechiba/.
"""

from __future__ import annotations

import jax.numpy as jnp

HYDROL = "hydrol_previous_step_state"
ZERO_STATE_KEYS = tuple(f"{HYDROL}.{name}" for name in
                        ("snow", "snowdz", "snowliq", "snow_nobio")) + (
    "slowproc_stomate_previous_step_state.frac_nobio",
)
THERMAL_REFERENCES = {"snowheat": (0.0, 1e9), "snowtemp": (273.15, 100.0)}
ZERO_OUTPUT_KEYS = tuple(
    f"inventory.water.{name}.{term}"
    for name in ("snow", "snow_nobio", "snowliq")
    for term in ("incoming_amount", "outgoing_fraction")
) + tuple(f"typed.water.{name}" for name in (
    "vevapsno", "subsnownobio", "snowmelt", "snowmelt_from_maxmass",
    "snow_melt_mass_by_layer", "snow_refreeze_mass_by_layer",
    "snow_liquid_percolation_by_interface",
)) + tuple(f"endpoint.fast_state.{HYDROL}.{name}" for name in (
    "snow_age", "snow_nobio_age", "snowdz", "snowgrain", "snowmelt", "subsnownobio",
))
DENSITY_KEY = f"endpoint.fast_state.{HYDROL}.snowrho"


def state_field(state, contract, key):
    leaf = next(leaf for leaf in contract.state_leaves if leaf.key == key)
    return jnp.asarray(state, dtype=jnp.float64)[..., leaf.start:leaf.stop]


def warm_snow_free_regime(state, forcing, contract, layout):
    """Sufficient no-snow condition, reducing only feature/record axes.

    Leading axes may represent independent days/windows for label auditing.
    No tolerance turns a real small snowpack into a zero-snow state. Strictly
    warm native endpoints also exclude freezing dew (diffuco_comb).
    """
    active = jnp.asarray(forcing.record_mask, dtype=bool)
    current = active & ~jnp.asarray(forcing.predecessor_mask, dtype=bool)
    values = jnp.asarray(forcing.values, dtype=jnp.float64)
    snow = values[..., layout.snow_start:layout.snow_stop]
    temp = values[..., layout.temperature_start:layout.temperature_stop]
    if layout.temperature_start < 0:
        raise ValueError("snow-free boundary requires native Tair coordinates")
    eligible = jnp.any(current, axis=-1)
    eligible &= jnp.all(jnp.where(current[..., None], snow == 0.0, True), axis=(-2, -1))
    eligible &= jnp.all(jnp.where(active[..., None], temp > 273.15, True), axis=(-2, -1))
    for key in ZERO_STATE_KEYS:
        eligible &= jnp.all(state_field(state, contract, key) == 0.0, axis=-1)
    for name, (reference, bound) in THERMAL_REFERENCES.items():
        delta = reference - state_field(state, contract, f"{HYDROL}.{name}")
        eligible &= jnp.all(jnp.isfinite(delta) & (jnp.abs(delta) <= bound), axis=-1)
    return eligible


def apply_warm_snow_free_boundary(outputs, inputs, contract, layout):
    """Replace only source-determined snow coordinates before the updater.

    Reset tendencies use float64, matching the conservative thermal update.
    They remove the incoming snow thermal derivative within this regime.
    The learned Attention outputs and derivatives are unchanged elsewhere.
    """
    state = inputs.canonical_input.continuous_state
    eligible = warm_snow_free_regime(state, inputs.forcing, contract, layout)
    result = dict(outputs)
    for key in ZERO_OUTPUT_KEYS:
        result[key] = jnp.where(eligible, jnp.zeros_like(outputs[key]), outputs[key])
    result[DENSITY_KEY] = jnp.where(eligible, jnp.full_like(outputs[DENSITY_KEY], 50.0), outputs[DENSITY_KEY])
    for name, (reference, bound) in THERMAL_REFERENCES.items():
        key = f"thermal.{HYDROL}.{name}.normalized_tendency"
        start = state_field(state, contract, f"{HYDROL}.{name}").reshape(outputs[key].shape)
        # Mask before arithmetic so inactive undefined state cannot poison AD.
        start = jnp.where(eligible, start, reference)
        result[key] = jnp.where(eligible, (reference - start) / bound, outputs[key])
    return result
