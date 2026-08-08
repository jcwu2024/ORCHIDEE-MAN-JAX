"""Host-side v6 sample assembly for neural process-head training.

The anonymous fast-day vector is decoded here into named Teacher-owned fields.
It is never returned as an inference input or as an anonymous loss target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, NamedTuple

import numpy as np

from jax_orchidee.parameters.pft_catalog import PFTCatalog, load_pft_catalog
from jax_orchidee.sechiba.diffuco import rprof_from_humcste_use, z_soil_from_diaglev
from research.daily_coarse_graining.constrained_daily_replay import (
    CARBON_INVENTORY_FIELDS,
    THERMAL_FIELDS,
    WATER_INVENTORY_FIELDS,
    BoundedInventoryTransition,
    encode_inventory_endpoints,
    encode_signed_inventory_endpoints,
    encode_thermal_endpoints,
)
from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    reconstruct_fast_day_target,
    reconstruct_state_fields,
)
from research.daily_coarse_graining.daily_pft_interface import (
    DailyPFTInterfaceContract,
    load_daily_pft_interface_contract,
    pack_named_pft_channels,
)
from research.daily_coarse_graining.typed_sidecar import (
    COHERENT_CONTRACT_PATH,
    TypedSidecarContract,
    TypedSidecarIndex,
    load_typed_sidecar_contract,
    read_typed_sidecar_shard,
)

from .input_assembly import PFT_AXES, CanonicalDailyOperatorInputAssembler
from .types import (
    DailyOperatorInput,
    NativeForcingInput,
    PFTAxisInput,
    RetainedProcessBoundary,
    StateDefinedMasks,
    StateFeatureGroups,
    StaticRegistryInput,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PFT_CATALOG_PATH = ROOT / "configs" / "pft_catalogs" / "orchidee_man_paper_250919.json"


class MaskedProcessTarget(NamedTuple):
    value: Any
    defined: Any


class ProcessTrainingTarget(NamedTuple):
    """Named supervision plus state/lifecycle targets for one Teacher day."""

    supervision: Mapping[str, MaskedProcessTarget]
    retained_boundary: RetainedProcessBoundary
    next_state: StateFeatureGroups
    next_state_defined: StateDefinedMasks
    next_discrete_state: Mapping[str, Any]
    diagnostics: Any


class ProcessTrainingExample(NamedTuple):
    inputs: DailyOperatorInput
    targets: ProcessTrainingTarget


def _get_path(root: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = root
    for name in path:
        value = value[name]
    return value


def _condition_value(
    vector: Any,
    leaves: tuple[Any, ...],
    name: str,
) -> np.ndarray:
    matches = tuple(leaf for leaf in leaves if leaf.name == name)
    if len(matches) != 1:
        raise ValueError(f"condition registry requires exactly one {name!r} leaf")
    leaf = matches[0]
    return np.asarray(vector, dtype=np.float64)[leaf.start : leaf.stop].reshape(leaf.shape)


def _defined(value: Any) -> np.ndarray:
    array = np.asarray(value)
    return np.isfinite(array) & (np.abs(array) < 0.5e20)


def _broadcast_pft_mask(mask: np.ndarray, shape: tuple[int, ...], *, axis: int) -> np.ndarray:
    view = [1] * len(shape)
    view[axis] = mask.size
    return np.broadcast_to(mask.reshape(view), shape)


@dataclass(frozen=True)
class DailyProcessTrainingAdapter:
    """Bind frozen contracts and assemble one coherent v6 training example."""

    contract: DailyMarkovContract
    typed_contract: TypedSidecarContract
    pft_contract: DailyPFTInterfaceContract
    pft_catalog: PFTCatalog
    input_assembler: CanonicalDailyOperatorInputAssembler

    @classmethod
    def from_contract(
        cls,
        contract: DailyMarkovContract,
        *,
        typed_contract_path: str | Path = COHERENT_CONTRACT_PATH,
        pft_catalog_path: str | Path = DEFAULT_PFT_CATALOG_PATH,
    ) -> DailyProcessTrainingAdapter:
        typed = load_typed_sidecar_contract(typed_contract_path)
        if contract.sha256 != typed.parent_contract_sha256:
            raise ValueError("v6 Markov and typed contracts do not share one identity")
        pft_contract = load_daily_pft_interface_contract()
        catalog = load_pft_catalog(pft_catalog_path)
        active = tuple(int(index) for index in contract.active_pft_indices)
        if not active:
            raise ValueError("daily process training requires at least one active PFT slot")
        for index in active:
            matches = tuple(entry for entry in catalog.entries if entry.fortran_pft_id == index + 1)
            if len(matches) != 1:
                raise ValueError(f"PFT catalog cannot resolve Fortran PFT row {index + 1}")
        return cls(
            contract=contract,
            typed_contract=typed,
            pft_contract=pft_contract,
            pft_catalog=catalog,
            input_assembler=CanonicalDailyOperatorInputAssembler.from_contract(contract),
        )

    @property
    def active_pft_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in self.contract.active_pft_indices)

    def _pft_entries(self) -> tuple[Any, ...]:
        by_fortran_id = {entry.fortran_pft_id: entry for entry in self.pft_catalog.entries}
        return tuple(by_fortran_id[index + 1] for index in self.active_pft_indices)

    def _select_pft(self, value: Any, axis: int) -> np.ndarray:
        return np.take(np.asarray(value), np.asarray(self.active_pft_indices), axis=axis)

    def _pft_inputs(
        self,
        fields: Mapping[str, Mapping[str, Any]],
        parameters: Any,
    ) -> PFTAxisInput:
        slow = fields["slowproc_stomate_previous_step_state"]
        state_parts = []
        for name in ("veget_max", "lai", "biomass"):
            selected = self._select_pft(slow[name], 1)
            state_parts.append(np.moveaxis(selected, 1, 0).reshape((len(self.active_pft_indices), -1)))
        pft_state = np.concatenate(state_parts, axis=1).astype(np.float64, copy=False)

        leaves = self.contract.static_conditions.parameter_leaves
        alloc_min = _condition_value(parameters, leaves, "alloc_min")
        residence_time = _condition_value(parameters, leaves, "residence_time")
        vcmax25 = _condition_value(parameters, leaves, "vcmax25")
        maintenance = _condition_value(parameters, leaves, "maint_resp_slope")
        named_parameters = {
            "vcmax25": vcmax25,
            "maint_resp_slope_a": maintenance[:, 2],
            "maint_resp_slope_b": maintenance[:, 1],
            "maint_resp_slope_c": maintenance[:, 0],
            "alloc_min": alloc_min,
            "residence_time": residence_time,
        }
        named_parameters = {
            name: np.asarray(value, dtype=np.float64)[np.asarray(self.active_pft_indices)]
            for name, value in named_parameters.items()
        }
        pft_parameters = pack_named_pft_channels(
            self.pft_contract,
            named_parameters,
            channel_kind="parameter",
        )

        trait_values: dict[str, np.ndarray] = {}
        entries = self._pft_entries()
        for channel in self.pft_contract.trait_channels:
            values = []
            for entry in entries:
                if channel.name == "role_bare_soil":
                    value = entry.role == "bare_soil"
                elif channel.name == "managed":
                    value = not bool(entry.traits["natural"])
                elif channel.name.startswith("pref_soil_veg_"):
                    value = int(entry.traits["pref_soil_veg"]) == int(channel.name.rsplit("_", 1)[1])
                else:
                    value = entry.traits[channel.name]
                values.append(float(value))
            trait_values[channel.name] = np.asarray(values, dtype=np.float64)
        pft_traits = pack_named_pft_channels(
            self.pft_contract,
            trait_values,
            channel_kind="trait",
        )

        fraction = self._select_pft(slow["veget_max"], 1)
        if fraction.shape[0] != 1:
            raise ValueError("v6 process training currently requires one landpoint per sample")
        fraction = np.asarray(fraction[0], dtype=np.float64)
        return PFTAxisInput(
            pft_state=pft_state,
            pft_parameters=pft_parameters,
            pft_traits=pft_traits,
            pft_fraction=fraction,
            active_pft_mask=np.ones((fraction.size,), dtype=np.bool_),
        )

    def _static_registry(self, landpoint_static: Any) -> StaticRegistryInput:
        leaves = self.contract.static_conditions.landpoint_static_leaves
        humcste = _condition_value(landpoint_static, leaves, "hydrol_humcste")
        diaglev = _condition_value(landpoint_static, leaves, "diaglev")
        full_rprof = np.asarray(rprof_from_humcste_use(humcste[None, :]), dtype=np.float64)
        rprof = np.take(full_rprof, np.asarray(self.active_pft_indices), axis=1)
        z_soil = np.asarray(z_soil_from_diaglev(diaglev), dtype=np.float64)
        return StaticRegistryInput(
            context_features=np.asarray(landpoint_static, dtype=np.float64),
            rprof=rprof,
            z_soil=z_soil,
        )

    def _forcing(self, forcing_native: Any) -> NativeForcingInput:
        values = np.asarray(forcing_native, dtype=np.float64)
        spec = self.contract.native_forcing
        expected = (spec.source_records_per_day, spec.width)
        if values.shape != expected:
            raise ValueError(f"native forcing shape drift: {values.shape} != {expected}")
        count = values.shape[0]
        interval = float(spec.source_interval_seconds)
        predecessor = np.zeros((count,), dtype=np.bool_)
        predecessor[0] = True
        return NativeForcingInput(
            values=values,
            record_time_seconds=(np.arange(count, dtype=np.float64) - 1.0) * interval,
            duration_seconds=np.full((count,), interval, dtype=np.float64),
            predecessor_mask=predecessor,
            record_mask=np.ones((count,), dtype=np.bool_),
        )

    def _operator_input(
        self,
        sample: Mapping[str, Any],
        fields: Mapping[str, Mapping[str, Any]],
    ) -> DailyOperatorInput:
        return DailyOperatorInput(
            continuous_state=np.asarray(sample["state"], dtype=np.float64),
            discrete_state=sample["discrete_state"],
            forcing=self._forcing(sample["forcing_native"]),
            pft=self._pft_inputs(fields, sample["parameters"]),
            static_registry=self._static_registry(sample["landpoint_static"]),
            annual_conditions=np.asarray(sample["annual_conditions"], dtype=np.float64),
            year=np.asarray(sample["year"], dtype=np.int32),
            day_index=np.asarray(sample["day_index"], dtype=np.int32),
        )

    def assemble_input(self, parent_sample: Mapping[str, Any]) -> DailyOperatorInput:
        """Assemble inference-owned inputs without decoding a training target."""

        required = {
            "state",
            "forcing_native",
            "parameters",
            "landpoint_static",
            "annual_conditions",
            "year",
            "day_index",
            "discrete_state",
        }
        missing = sorted(required - set(parent_sample))
        if missing:
            raise ValueError(f"v6 process input sample is missing fields: {missing}")
        fields = reconstruct_state_fields(
            np.asarray(parent_sample["state"], dtype=np.float64),
            parent_sample["discrete_state"],
            self.contract,
        )
        return self._operator_input(parent_sample, fields)

    def _compact_inventory(self, value: Any, field: str) -> np.ndarray:
        axis = PFT_AXES.get(field)
        return np.asarray(value) if axis is None else self._select_pft(value, axis)

    @staticmethod
    def _add_transition(
        supervision: dict[str, MaskedProcessTarget],
        prefix: str,
        transition: BoundedInventoryTransition,
    ) -> None:
        supervision[f"{prefix}.outgoing_fraction"] = MaskedProcessTarget(
            np.asarray(transition.outgoing_fraction, dtype=np.float64),
            np.asarray(transition.defined, dtype=np.bool_),
        )
        supervision[f"{prefix}.incoming_amount"] = MaskedProcessTarget(
            np.asarray(transition.incoming_amount, dtype=np.float64),
            np.asarray(transition.defined, dtype=np.bool_),
        )

    def _inventory_supervision(
        self,
        start_fields: Mapping[str, Mapping[str, Any]],
        fast_fields: Mapping[str, Mapping[str, Any]],
        ok_leak: Mapping[str, Any],
    ) -> dict[str, MaskedProcessTarget]:
        result: dict[str, MaskedProcessTarget] = {}
        start_water = start_fields["hydrol_previous_step_state"]
        target_water = fast_fields["hydrol_previous_step_state"]
        for field in WATER_INVENTORY_FIELDS:
            start = self._compact_inventory(start_water[field], field)
            end = self._compact_inventory(target_water[field], field)
            if field == "qsintveg":
                signed = encode_signed_inventory_endpoints(start, end, name=f"water.{field}")
                self._add_transition(result, f"inventory.water.{field}.storage", signed.positive)
                self._add_transition(result, f"inventory.water.{field}.debt", signed.debt)
            else:
                transition = encode_inventory_endpoints(start, end, name=f"water.{field}")
                self._add_transition(result, f"inventory.water.{field}", transition)

        start_carbon = start_fields["slowproc_stomate_previous_step_state"]
        for field in CARBON_INVENTORY_FIELDS:
            start = self._compact_inventory(start_carbon[field], field)
            end = self._compact_inventory(ok_leak[field], field)
            transition = encode_inventory_endpoints(start, end, name=f"carbon.{field}")
            self._add_transition(result, f"inventory.carbon.{field}", transition)
        return result

    def _thermal_supervision(
        self,
        start_fields: Mapping[str, Mapping[str, Any]],
        fast_fields: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, MaskedProcessTarget]:
        result = {}
        available = {leaf.key for leaf in self.contract.state_leaves}
        for component, field, bound in THERMAL_FIELDS:
            if f"{component}.{field}" not in available:
                continue
            transition = encode_thermal_endpoints(
                start_fields[component][field],
                fast_fields[component][field],
                name=f"energy.{component}.{field}",
                bound=bound,
            )
            result[f"thermal.{component}.{field}.normalized_tendency"] = MaskedProcessTarget(
                np.asarray(transition.normalized_tendency, dtype=np.float64),
                np.asarray(transition.defined, dtype=np.bool_),
            )
        return result

    def _daily_supervision(
        self,
        daily: Mapping[str, Any],
    ) -> dict[str, MaskedProcessTarget]:
        result: dict[str, MaskedProcessTarget] = {}
        active = np.ones((len(self.active_pft_indices),), dtype=np.bool_)
        for source, key in (
            ("gpp_daily", "daily.fast_gpp"),
            ("resp_maint_part", "daily.fast_maintenance_respiration"),
        ):
            value = self._select_pft(daily[source], 1)
            mask = _defined(value) & _broadcast_pft_mask(active, value.shape, axis=1)
            result[key] = MaskedProcessTarget(np.asarray(value, dtype=np.float64), mask)
        for source, key in (
            ("tsurf_daily", "daily.temperature_context.tsurf"),
            ("tsoil_daily", "daily.temperature_context.tsoil"),
        ):
            value = np.asarray(daily[source], dtype=np.float64)
            result[key] = MaskedProcessTarget(value, _defined(value))
        return result

    def _supplemental_supervision(
        self,
        values: Mapping[str, Any],
        defined: Mapping[str, Any],
    ) -> dict[str, MaskedProcessTarget]:
        expected = set(self.typed_contract.fields_by_path)
        if set(values) != expected or set(defined) != expected:
            raise ValueError("supplemental fields do not match the frozen v6 typed contract")
        result = {}
        for field in self.typed_contract.fields:
            value = np.asarray(values[field.path])
            mask = np.asarray(defined[field.path])
            if value.shape != field.feature_shape or value.dtype != np.dtype(field.dtype):
                raise ValueError(f"supplemental value shape/dtype drift for {field.path}")
            if mask.shape != field.feature_shape or mask.dtype != np.dtype(np.bool_):
                raise ValueError(f"supplemental mask drift for {field.path}")
            if np.any(mask & ~_defined(value)):
                raise ValueError(f"supplemental defined value is invalid for {field.path}")
            result[f"typed.{field.path}"] = MaskedProcessTarget(value, mask)
        return result

    def field_labels(self) -> dict[str, str]:
        """Return the stable equal-macro loss ownership for every target leaf."""

        labels: dict[str, str] = {}
        water_labels = {
            "qsintveg": "water.canopy_inventory_endpoints",
            "mc": "water.soil_inventory_endpoints",
            "mcl": "water.soil_inventory_endpoints",
            "snow": "water.snow_inventory_endpoints",
            "snow_nobio": "water.snow_inventory_endpoints",
            "snowliq": "water.snow_inventory_endpoints",
            "flood_res": "water.flood_inventory_endpoints",
        }
        for field, label in water_labels.items():
            prefixes = (
                (f"inventory.water.{field}.storage", f"inventory.water.{field}.debt")
                if field == "qsintveg"
                else (f"inventory.water.{field}",)
            )
            for prefix in prefixes:
                labels[f"{prefix}.outgoing_fraction"] = label
                labels[f"{prefix}.incoming_amount"] = label
        for field in CARBON_INVENTORY_FIELDS:
            prefix = f"inventory.carbon.{field}"
            labels[f"{prefix}.outgoing_fraction"] = "carbon.ok_leak_inventory_endpoints"
            labels[f"{prefix}.incoming_amount"] = "carbon.ok_leak_inventory_endpoints"
        for component, field, _bound in THERMAL_FIELDS:
            label = (
                "energy.surface_temperature_tendency"
                if field == "temp_sol"
                else "energy.snow_thermal_tendency"
                if field in {"snowtemp", "snowheat"}
                else "energy.soil_temperature_tendency"
            )
            labels[f"thermal.{component}.{field}.normalized_tendency"] = label
        labels.update(
            {
                "daily.fast_gpp": "carbon.fast_gpp",
                "daily.fast_maintenance_respiration": "carbon.fast_maintenance_respiration",
                "daily.temperature_context.tsurf": "energy.daily_temperature_context",
                "daily.temperature_context.tsoil": "energy.daily_temperature_context",
            }
        )
        for field in self.typed_contract.fields:
            labels[f"typed.{field.path}"] = field.label_id
        return labels

    def assemble(
        self,
        parent_sample: Mapping[str, Any],
        *,
        supplemental_values: Mapping[str, Any],
        supplemental_defined: Mapping[str, Any],
    ) -> ProcessTrainingExample:
        """Assemble one sample without exposing its anonymous fast target."""

        required = {
            "state",
            "fast_day_target",
            "forcing_native",
            "parameters",
            "landpoint_static",
            "annual_conditions",
            "next_state",
            "diagnostics",
            "year",
            "day_index",
            "discrete_state",
            "next_discrete_state",
        }
        missing = sorted(required - set(parent_sample))
        if missing:
            raise ValueError(f"v6 parent sample is missing fields: {missing}")
        start_fields = reconstruct_state_fields(
            np.asarray(parent_sample["state"], dtype=np.float64),
            parent_sample["discrete_state"],
            self.contract,
        )
        reconstructed = reconstruct_fast_day_target(
            np.asarray(parent_sample["fast_day_target"], dtype=np.float64),
            self.contract.fast_day_target_leaves,
            template_fields=start_fields,
        )
        inputs = self._operator_input(parent_sample, start_fields)
        supervision = self._inventory_supervision(
            start_fields,
            reconstructed.fields_by_component,
            reconstructed.ok_leak_updates,
        )
        supervision.update(
            self._thermal_supervision(start_fields, reconstructed.fields_by_component)
        )
        supervision.update(self._daily_supervision(reconstructed.daily_fields))
        supervision.update(
            self._supplemental_supervision(supplemental_values, supplemental_defined)
        )
        expected_labels = self.field_labels()
        if set(supervision) != set(expected_labels):
            missing_targets = sorted(set(expected_labels) - set(supervision))
            extra_targets = sorted(set(supervision) - set(expected_labels))
            raise ValueError(
                "process supervision registry drift; "
                f"missing={missing_targets}, extra={extra_targets}"
            )

        next_input = inputs._replace(
            continuous_state=np.asarray(parent_sample["next_state"], dtype=np.float64),
            discrete_state=parent_sample["next_discrete_state"],
        )
        next_assembled = self.input_assembler.assemble(next_input)
        retained = RetainedProcessBoundary(
            fast_state_fields=reconstructed.fields_by_component,
            final_entry_diagnostics=reconstructed.final_diagnostics,
            daily_fields=reconstructed.daily_fields,
            ok_leak_result=reconstructed.ok_leak_updates,
        )
        target = ProcessTrainingTarget(
            supervision=supervision,
            retained_boundary=retained,
            next_state=next_assembled.state,
            next_state_defined=next_assembled.state_defined,
            next_discrete_state=parent_sample["next_discrete_state"],
            diagnostics=np.asarray(parent_sample["diagnostics"], dtype=np.float64),
        )
        return ProcessTrainingExample(inputs=inputs, targets=target)


def coherent_process_training_examples(
    index: TypedSidecarIndex,
    adapter: DailyProcessTrainingAdapter,
    *,
    spatial_split: str | None = None,
    temporal_split: str | None = None,
) -> Iterator[ProcessTrainingExample]:
    """Stream coherent base/typed shards without yielding anonymous targets."""

    if index.contract.sha256 != adapter.typed_contract.sha256:
        raise ValueError("typed dataset and process adapter contract mismatch")
    if index.parent_index.contract_sha256 != adapter.contract.sha256:
        raise ValueError("base dataset and process adapter Markov contract mismatch")
    for reference in index.select(
        spatial_split=spatial_split,
        temporal_split=temporal_split,
    ):
        from research.daily_coarse_graining.daily_markov_contract import load_markov_shard

        parent = load_markov_shard(reference.parent.path)
        values, defined, day_index = read_typed_sidecar_shard(
            reference.path,
            contract=index.contract,
            expected_days=reference.day_count,
        )
        if not np.array_equal(day_index, parent.day_index):
            raise ValueError("coherent typed/base day index mismatch")
        for row in range(parent.days):
            yield adapter.assemble(
                parent.sample(row),
                supplemental_values={name: value[row] for name, value in values.items()},
                supplemental_defined={name: value[row] for name, value in defined.items()},
            )
