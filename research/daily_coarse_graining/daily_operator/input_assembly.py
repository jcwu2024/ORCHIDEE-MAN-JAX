"""Unique source-backed assembly of the Gate-E1 inference input views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from jax_orchidee.stomate.carbon_kernels import littercalc_litter_fractions
from jax_orchidee.stomate.soilcarbon_kernels import soilcarbon_leak_frac_carb
from research.daily_coarse_graining.constrained_daily_replay import (
    CARBON_INVENTORY_FIELDS,
    THERMAL_FIELDS,
    WATER_INVENTORY_FIELDS,
)
from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    reconstruct_state_fields_compiled,
)

from .conservative_update import split_signed_inventory
from .types import (
    AssembledDailyOperatorInput,
    DailyOperatorInput,
    DayStartProcessState,
    ProcessStaticConditions,
    StateDefinedMasks,
    StateFeatureGroups,
)

ORCHIDEE_UNDEFINED_THRESHOLD = 0.5e20
PFT_AXES = {
    "qsintveg": 1,
    "litter_above": 2,
    "litter_below": 2,
    "carbon_32l": 3,
    "DOC": 1,
    "interception_storage": 1,
}


def _get_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for name in path:
        current = current[name]
    return current


def _defined(value: Any) -> Any:
    array = jnp.asarray(value)
    return jnp.isfinite(array) & (jnp.abs(array) < ORCHIDEE_UNDEFINED_THRESHOLD)


@dataclass(frozen=True)
class CanonicalDailyOperatorInputAssembler:
    """Derive every overlapping process view from one canonical owner.

    The public :class:`DailyOperatorInput` has no grouped-state, litter,
    lignin, or process-static fields. This assembler is therefore the only
    path by which those cached views can enter the staged transition.
    """

    contract: DailyMarkovContract
    clay_start: int
    clay_stop: int
    clay_shape: tuple[int, ...]
    cue: float = 0.3
    sro_bottom: int = 5

    @classmethod
    def from_contract(
        cls,
        contract: DailyMarkovContract,
        *,
        cue: float = 0.3,
        sro_bottom: int = 5,
    ) -> CanonicalDailyOperatorInputAssembler:
        matches = tuple(
            leaf
            for leaf in contract.static_conditions.landpoint_static_leaves
            if leaf.name == "clay_frac"
        )
        if len(matches) != 1:
            raise ValueError("canonical static registry requires exactly one clay_frac owner")
        clay = matches[0]
        return cls(
            contract=contract,
            clay_start=int(clay.start),
            clay_stop=int(clay.stop),
            clay_shape=tuple(clay.shape),
            cue=float(cue),
            sro_bottom=int(sro_bottom),
        )

    @property
    def active_pft_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in self.contract.active_pft_indices)

    @property
    def thermal_keys(self) -> tuple[str, ...]:
        available = {leaf.key for leaf in self.contract.state_leaves}
        return tuple(
            f"{component}.{field}"
            for component, field, _bound in THERMAL_FIELDS
            if f"{component}.{field}" in available
        )

    def _select_pft(self, value: Any, axis: int) -> Any:
        return jnp.take(
            jnp.asarray(value),
            jnp.asarray(self.active_pft_indices, dtype=jnp.int32),
            axis=axis,
        )

    def _inventory_array(self, value: Any, field: str) -> Any:
        axis = PFT_AXES.get(field)
        return jnp.asarray(value) if axis is None else self._select_pft(value, axis)

    def _pack_water(self, fields: Mapping[str, Any]) -> Any:
        pieces = []
        for field in WATER_INVENTORY_FIELDS:
            value = self._inventory_array(fields[field], field)
            if field == "qsintveg":
                storage, debt = split_signed_inventory(value)
                pieces.extend((storage.reshape(-1), debt.reshape(-1)))
            else:
                pieces.append(value.reshape(-1))
        return jnp.concatenate(tuple(pieces))

    def _pack_carbon(self, fields: Mapping[str, Any]) -> Any:
        return jnp.concatenate(
            tuple(
                self._inventory_array(fields[field], field).reshape(-1)
                for field in CARBON_INVENTORY_FIELDS
            )
        )

    def _pack_thermal(self, fields: Mapping[str, Mapping[str, Any]]) -> Any:
        return jnp.concatenate(
            tuple(
                jnp.asarray(fields[component][field]).reshape(-1)
                for component, field in (key.split(".", 1) for key in self.thermal_keys)
            )
        )

    def assemble(self, value: DailyOperatorInput) -> AssembledDailyOperatorInput:
        """Build all named views from canonical state and static registries."""

        continuous = jnp.asarray(value.continuous_state, dtype=jnp.float64)
        fields = reconstruct_state_fields_compiled(
            continuous,
            value.discrete_state,
            self.contract,
        )
        hydrol = fields["hydrol_previous_step_state"]
        slow = fields["slowproc_stomate_previous_step_state"]
        state = StateFeatureGroups(
            canopy=self._select_pft(hydrol["qsintveg"], 1).reshape(-1),
            soil=jnp.asarray(hydrol["mc"]).reshape(-1),
            snow=jnp.concatenate(
                tuple(jnp.asarray(hydrol[name]).reshape(-1) for name in ("snow", "snow_nobio", "snowliq"))
            ),
            water=self._pack_water(hydrol),
            carbon=self._pack_carbon(slow),
            thermal=self._pack_thermal(fields),
            other=continuous,
        )
        masks = StateDefinedMasks(
            **{name: _defined(getattr(state, name)) for name in StateFeatureGroups._fields}
        )
        static_values = jnp.asarray(
            value.static_registry.context_features,
            dtype=jnp.float64,
        )
        clay = static_values[self.clay_start : self.clay_stop].reshape(self.clay_shape)
        day_start = DayStartProcessState(
            bm_to_litter=self._select_pft(slow["bm_to_litter"], 1),
            turnover_daily=self._select_pft(slow["turnover_daily"], 1),
            lignin_struc_above=self._select_pft(slow["lignin_struc_above"], 1),
            lignin_struc_below=self._select_pft(slow["lignin_struc_below"], 1),
        )
        process_static = ProcessStaticConditions(
            rprof=jnp.asarray(value.static_registry.rprof, dtype=jnp.float64),
            z_soil=jnp.asarray(value.static_registry.z_soil, dtype=jnp.float64),
            litterfrac=jnp.asarray(littercalc_litter_fractions(), dtype=jnp.float64),
            cue=jnp.asarray(self.cue, dtype=jnp.float64),
            frac_carb=soilcarbon_leak_frac_carb(clay),
            sro_bottom=jnp.asarray(self.sro_bottom, dtype=jnp.int32),
        )
        return AssembledDailyOperatorInput(
            state=state,
            state_defined=masks,
            discrete_state=value.discrete_state,
            forcing=value.forcing,
            pft=value.pft,
            day_start_process_state=day_start,
            process_static=process_static,
            static_conditions=static_values,
            annual_conditions=value.annual_conditions,
            year=value.year,
            day_index=value.day_index,
            canonical_input=value,
        )

    def validate_cached_views(
        self,
        source: DailyOperatorInput,
        cached: AssembledDailyOperatorInput,
    ) -> None:
        """Host-side fail-closed gate for persisted or externally cached views."""

        expected = self.assemble(source)

        def require_equal(label: str, left: Any, right: Any) -> None:
            left_leaves = jax.tree_util.tree_leaves(left)
            right_leaves = jax.tree_util.tree_leaves(right)
            if len(left_leaves) != len(right_leaves):
                raise ValueError(f"{label} cached-view structure drift")
            for actual, wanted in zip(left_leaves, right_leaves, strict=True):
                if not np.array_equal(
                    np.asarray(actual),
                    np.asarray(wanted),
                    equal_nan=True,
                ):
                    raise ValueError(f"{label} cached view contradicts its canonical owner")

        require_equal("grouped state", cached.state, expected.state)
        require_equal("state masks", cached.state_defined, expected.state_defined)
        require_equal(
            "day-start process state",
            cached.day_start_process_state,
            expected.day_start_process_state,
        )
        require_equal("process static", cached.process_static, expected.process_static)
        require_equal("canonical input", cached.canonical_input, source)
