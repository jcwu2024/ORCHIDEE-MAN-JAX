"""Daily Markov data contract for the coarse-grained ORCHIDEE teacher.

The contract stores one canonical cross-day trajectory and the native 6-hour
forcing records needed to reconstruct each teacher day.  It deliberately does
not persist expanded half-hour forcing, duplicate day-start/day-end arrays, or
finite masks.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
import yaml

from jax_orchidee.driver import domain as forcing_domain
from jax_orchidee.driver.bundle import ccanopy_from_co2
from jax_orchidee.driver.domain import read_annual_co2
from jax_orchidee.driver.orchestration import DriverCompiledHalfHourForcing
from jax_orchidee.sechiba.restart_io import (
    SECHIBA_RESTART_COMPONENT_FIELDS,
    SECHIBA_RESTART_TO_SOURCE_NAMES,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_SCHEMA_VERSION = "daily_markov_contract_v4"
SHARD_SCHEMA_VERSION = "daily_teacher_markov_year_v4"
NATIVE_FORCING_FIELDS = (
    "Tair",
    "PSurf",
    "Qair",
    "Wind_E",
    "Wind_N",
    "Rainf",
    "Snowf",
    "SWdown",
    "LWdown",
)
_SECHIBA_CONTRACT = ROOT / "docs/source_audits/sechiba_state_field_contract.yaml"
_RESTART_CONTRACT = ROOT / "docs/source_audits/restart_state_lifecycle.yaml"
_SLOW_COMPONENT = "slowproc_stomate_previous_step_state"
_FINALIZE_COMPONENT = "sechiba_finalize_state"
_SLOWPROC_DIFFUCO_MIRRORS = frozenset(
    {"lai", "frac_nobio", "veget_max", "veget", "tot_bare_soil"}
)
FAST_DAY_STATE_COMPONENTS = (
    "driver_previous_step_state",
    "diffuco_previous_step_state",
    "enerbil_previous_step_state",
    "hydrol_previous_step_state",
    "thermosoil_previous_step_state",
)
_FINALIZE_CROSS_DAY_FIELDS = frozenset(
    {
        "leaf_ci",
        "q_sol_pot",
        "temp_sol_pot",
        "free_drain_coef",
        "zwt_force",
        "resdist",
        "vegtot_old",
        "soilalb_bg",
        "refsoc",
        "e_soil_lat",
        *(
            SECHIBA_RESTART_TO_SOURCE_NAMES.get(name, name)
            for name in SECHIBA_RESTART_COMPONENT_FIELDS["slowproc"]
        ),
    }
)
_FINALIZE_AXIS_FALLBACKS = {
    "leaf_ci": ("npts", "nvm", "nlai"),
    "e_soil_lat": ("npts", "nvm"),
}
FAST_DAY_OK_LEAK_FIELDS = (
    "litter_above",
    "litter_below",
    "lignin_struc_above",
    "lignin_struc_below",
    "litterpart",
    "dead_leaves",
    "fuel_1hr",
    "fuel_10hr",
    "fuel_100hr",
    "fuel_1000hr",
    "carbon_32l",
    "DOC",
    "interception_storage",
    "deepC_peat",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class StateLeafSpec:
    component: str
    path: tuple[str, ...]
    shape: tuple[int, ...]
    full_shape: tuple[int, ...]
    dtype: str
    start: int | None
    stop: int | None
    classification: str
    source_ref: str
    missing_policy: str = "error"
    axis_names: tuple[str, ...] = ()
    selected_pft_indices: tuple[int, ...] = ()

    @property
    def key(self) -> str:
        return ".".join((self.component, *self.path))

    @property
    def discrete(self) -> bool:
        return self.start is None


@dataclass(frozen=True)
class NativeForcingSpec:
    fields: tuple[str, ...]
    field_shape: tuple[int, ...]
    source_records_per_day: int
    source_interval_seconds: float
    model_interval_seconds: float
    split: int
    precipitation_spread_steps: int

    @property
    def width(self) -> int:
        return len(self.fields) * int(np.prod(self.field_shape, dtype=np.int64))


@dataclass(frozen=True)
class ConditionLeafSpec:
    """One named slice in a flattened neural-condition vector."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    start: int
    stop: int
    temporal_role: str
    source: str


@dataclass(frozen=True)
class StaticConditionSpec:
    parameter_leaves: tuple[ConditionLeafSpec, ...]
    landpoint_static_leaves: tuple[ConditionLeafSpec, ...]
    annual_condition_leaves: tuple[ConditionLeafSpec, ...]
    includes: tuple[str, ...]

    @staticmethod
    def _width(leaves: tuple[ConditionLeafSpec, ...]) -> int:
        return max((leaf.stop for leaf in leaves), default=0)

    @property
    def parameter_width(self) -> int:
        return self._width(self.parameter_leaves)

    @property
    def landpoint_static_width(self) -> int:
        return self._width(self.landpoint_static_leaves)

    @property
    def annual_condition_width(self) -> int:
        return self._width(self.annual_condition_leaves)


@dataclass(frozen=True)
class DiagnosticSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    start: int
    stop: int
    owner: str
    axis_names: tuple[str, ...] = ()
    selected_pft_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class FastDayTargetLeafSpec:
    """One learned output owned by the replaced 48-step fast-day operator."""

    family: str
    component: str | None
    path: tuple[str, ...]
    shape: tuple[int, ...]
    full_shape: tuple[int, ...]
    dtype: str
    start: int
    stop: int
    owner: str
    axis_names: tuple[str, ...] = ()
    selected_pft_indices: tuple[int, ...] = ()

    @property
    def key(self) -> str:
        prefix = self.component if self.component is not None else self.family
        return ".".join((prefix, *self.path))


@dataclass(frozen=True)
class ReconstructedFastDayTarget:
    """Physical fast-day boundary rebuilt from the learned flat output."""

    fields_by_component: Mapping[str, Mapping[str, Any]]
    daily_fields: Mapping[str, Any]
    ok_leak_updates: Mapping[str, Any]
    final_diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class DailyMarkovContract:
    schema_version: str
    state_leaves: tuple[StateLeafSpec, ...]
    fast_day_target_leaves: tuple[FastDayTargetLeafSpec, ...]
    diagnostic_leaves: tuple[DiagnosticSpec, ...]
    native_forcing: NativeForcingSpec
    static_conditions: StaticConditionSpec
    source_hashes: tuple[tuple[str, str], ...]
    active_pft_indices: tuple[int, ...]

    @property
    def continuous_state_width(self) -> int:
        return max(
            (leaf.stop or 0 for leaf in self.state_leaves if not leaf.discrete),
            default=0,
        )

    @property
    def diagnostic_width(self) -> int:
        return max((leaf.stop for leaf in self.diagnostic_leaves), default=0)

    @property
    def fast_day_target_width(self) -> int:
        return max((leaf.stop for leaf in self.fast_day_target_leaves), default=0)

    @property
    def discrete_leaves(self) -> tuple[StateLeafSpec, ...]:
        return tuple(leaf for leaf in self.state_leaves if leaf.discrete)

    def metadata(self) -> dict[str, Any]:
        static_conditions = asdict(self.static_conditions) | {
            "parameter_width": self.static_conditions.parameter_width,
            "landpoint_static_width": self.static_conditions.landpoint_static_width,
            "annual_condition_width": self.static_conditions.annual_condition_width,
        }
        return {
            "schema_version": self.schema_version,
            "state_leaves": [asdict(leaf) | {"key": leaf.key} for leaf in self.state_leaves],
            "fast_day_target_leaves": [
                asdict(leaf) | {"key": leaf.key}
                for leaf in self.fast_day_target_leaves
            ],
            "diagnostic_leaves": [asdict(leaf) for leaf in self.diagnostic_leaves],
            "native_forcing": asdict(self.native_forcing) | {"width": self.native_forcing.width},
            "static_conditions": static_conditions,
            "source_hashes": dict(self.source_hashes),
            "active_pft_indices": list(self.active_pft_indices),
            "continuous_state_width": self.continuous_state_width,
            "fast_day_target_width": self.fast_day_target_width,
            "diagnostic_width": self.diagnostic_width,
        }

    @property
    def sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self.metadata()))


@dataclass(frozen=True)
class MarkovShard:
    state_trajectory: np.ndarray
    fast_day_target: np.ndarray
    forcing_native: np.ndarray
    forcing_record_indices: np.ndarray
    parameters: np.ndarray
    landpoint_static: np.ndarray
    annual_conditions: np.ndarray
    diagnostics: np.ndarray
    year: np.ndarray
    day_index: np.ndarray
    discrete_trajectories: Mapping[str, np.ndarray]

    @property
    def days(self) -> int:
        return int(self.day_index.size)

    def sample(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.days:
            raise IndexError(index)
        state = self.state_trajectory[index]
        next_state = self.state_trajectory[index + 1]
        forcing = self.forcing_native[index]
        diagnostics = self.diagnostics[index]
        return {
            "state": state,
            "state_finite": np.isfinite(state),
            "forcing_native": forcing,
            "forcing_finite": np.isfinite(forcing),
            "parameters": self.parameters,
            "landpoint_static": self.landpoint_static,
            "annual_conditions": self.annual_conditions,
            "next_state": next_state,
            "next_state_finite": np.isfinite(next_state),
            "diagnostics": diagnostics,
            "diagnostics_finite": np.isfinite(diagnostics),
            "fast_day_target": self.fast_day_target[index],
            "fast_day_target_finite": np.isfinite(self.fast_day_target[index]),
            "year": self.year,
            "day_index": self.day_index[index],
            "discrete_state": {
                key: value[index] for key, value in self.discrete_trajectories.items()
            },
            "next_discrete_state": {
                key: value[index + 1] for key, value in self.discrete_trajectories.items()
            },
        }


def _flatten_mapping(value: Any, prefix: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for name in sorted(value):
            yield from _flatten_mapping(value[name], (*prefix, str(name)))
        return
    if value is None:
        return
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"state leaf {'.'.join(prefix)} has unsupported dtype {array.dtype}")
    yield prefix, array


def _state_contract_entries() -> tuple[dict[str, Any], ...]:
    document = yaml.safe_load(_SECHIBA_CONTRACT.read_text(encoding="utf-8"))
    return tuple(document["contracts"])


def _state_axis_lookup() -> dict[tuple[str, str], tuple[str, ...]]:
    document = yaml.safe_load(_SECHIBA_CONTRACT.read_text(encoding="utf-8"))
    groups = document["axis_groups"]
    return {
        (entry["component"], entry["field"]): tuple(groups[entry["shape_group"]])
        for entry in document["contracts"]
    }


def _slow_axis_lookup() -> dict[str, tuple[str, ...]]:
    document = yaml.safe_load(_RESTART_CONTRACT.read_text(encoding="utf-8"))
    result = {}
    for group in document["axis_groups"].values():
        axes = tuple(group["axes"])
        for field in group["fields"]:
            result[str(field)] = axes
    return result


def _pft14_indices(packet) -> tuple[int, ...]:
    slow = packet.fields_by_component[_SLOW_COMPONENT]
    candidates = (slow.get("pft_present"), slow.get("veget_max"), slow.get("lai"))
    nvm = next(
        (
            int(np.asarray(value).shape[1])
            for value in candidates
            if value is not None and np.asarray(value).ndim >= 2
        ),
        None,
    )
    if nvm is None:
        raise ValueError("cannot identify the PFT axis from the day-end state")
    if nvm == 14:
        allowed = np.zeros(nvm, dtype=bool)
        allowed[[0, 13]] = True
        present = slow.get("pft_present")
        if present is not None and np.any(np.asarray(present, dtype=bool)[..., ~allowed]):
            raise ValueError("PFT14 contract cannot discard another present PFT slot")
        veget_max = slow.get("veget_max")
        if veget_max is not None and np.any(np.asarray(veget_max)[..., ~allowed] != 0.0):
            raise ValueError("PFT14 contract cannot discard another vegetated PFT slot")
        return (0, 13)
    return tuple(range(nvm))


def _select_pft_axes(
    value: np.ndarray,
    axis_names: tuple[str, ...],
    indices: tuple[int, ...],
) -> np.ndarray:
    result = np.asarray(value)
    for axis, name in enumerate(axis_names):
        if name == "nvm":
            result = np.take(result, indices, axis=axis)
    return result


def _canonical_state_inventory(
    packet,
) -> tuple[tuple[str, tuple[str, ...], np.ndarray, str, str, tuple[str, ...]], ...]:
    fields = packet.fields_by_component
    source_by_key = {
        (entry["component"], entry["field"]): str(entry["source_ref"])
        for entry in _state_contract_entries()
    }
    state_axes = _state_axis_lookup()
    slow_axes = _slow_axis_lookup()
    selected: list[
        tuple[str, tuple[str, ...], np.ndarray, str, str, tuple[str, ...]]
    ] = []
    for component, field_map in fields.items():
        for path, array in _flatten_mapping(field_map):
            top_name = path[0]
            if (
                component == _FINALIZE_COMPONENT
                and top_name not in _FINALIZE_CROSS_DAY_FIELDS
            ):
                continue
            if component == _SLOW_COMPONENT and top_name == "daily_accumulators":
                # The source day-end transition resets these fields. Same-day
                # accumulated values are Y[d], while next-day zeros are
                # deterministic and therefore not prognostic state.
                continue
            if component == _FINALIZE_COMPONENT:
                source_ref = (
                    "sechiba_main cross-day SAVE/restart carry consumed by "
                    "_sechiba_finalize_state_from_components"
                )
            elif component != _SLOW_COMPONENT and (component, top_name) not in source_by_key:
                if component == "thermosoil_previous_step_state" and top_name == "e_soil_lat":
                    source_ref = "thermosoil_main"
                else:
                    continue
            else:
                source_ref = source_by_key.get((component, top_name), "restart_state_lifecycle")
            if component == _SLOW_COMPONENT:
                axis_names = slow_axes.get(
                    path[-1], slow_axes.get(top_name, ())
                )
            elif component == _FINALIZE_COMPONENT:
                axis_names = slow_axes.get(
                    path[-1],
                    slow_axes.get(
                        top_name, _FINALIZE_AXIS_FALLBACKS.get(top_name, ())
                    ),
                )
            else:
                axis_names = state_axes.get((component, top_name), ())
            if axis_names and len(axis_names) != array.ndim:
                raise ValueError(
                    f"axis ledger drift for {component}.{'.'.join(path)}: "
                    f"{axis_names} vs shape {array.shape}"
                )
            if (
                component == "diffuco_previous_step_state"
                and top_name in _SLOWPROC_DIFFUCO_MIRRORS
                and top_name in fields.get(_SLOW_COMPONENT, {})
            ):
                slow_value = np.asarray(fields[_SLOW_COMPONENT][top_name])
                if slow_value.shape != array.shape or not np.array_equal(slow_value, array, equal_nan=True):
                    raise ValueError(f"day-end mirror drift for {component}.{top_name}")
                continue
            classification = "discrete" if array.dtype.kind in "biu" else "prognostic"
            selected.append(
                (component, path, array, classification, source_ref, axis_names)
            )
    return tuple(selected)


def _diagnostic_inventory(
    record,
) -> tuple[tuple[str, np.ndarray, str, tuple[str, ...]], ...]:
    slow_axes = _slow_axis_lookup()
    values: list[tuple[str, np.ndarray, str, tuple[str, ...]]] = []
    for name in sorted(record.daily_fold.daily_fields):
        values.append(
            (
                f"daily_fold.{name}",
                np.asarray(record.daily_fold.daily_fields[name]),
                "stomate_daily_process_fold_from_entries",
                slow_axes.get(name, ()),
            )
        )
    final = record.half_hour_transition.completed_entry_payloads[-1]
    for name in ("t2mdiag", "temp_sol"):
        values.append(
            (
                f"final_half_hour.{name}",
                np.asarray(final[name]),
                "compact_sechiba_entry_payload",
                (),
            )
        )
    return tuple(values)


def _fast_day_family(component: str) -> str:
    if component in {"diffuco_previous_step_state", "enerbil_previous_step_state"}:
        return "diffuco_enerbil"
    return component.removesuffix("_previous_step_state").removesuffix("_state")


def _fast_day_target_inventory(record):
    """Return the complete dynamic boundary handed to retained daily STOMATE."""

    state_axes = _state_axis_lookup()
    slow_axes = _slow_axis_lookup()
    values = []
    end_fields = record.half_hour_transition.current_state.fields_by_component
    for component in FAST_DAY_STATE_COMPONENTS:
        for path, array in _flatten_mapping(end_fields[component]):
            if array.dtype.kind != "f":
                continue
            top_name = path[0]
            values.append(
                (
                    _fast_day_family(component),
                    component,
                    path,
                    array,
                    "compiled half-hour SECHIBA state",
                    state_axes.get((component, top_name), ()),
                )
            )
    for name in sorted(record.daily_fold.daily_fields):
        array = np.asarray(record.daily_fold.daily_fields[name])
        values.append(
            (
                "daily_interface",
                None,
                (name,),
                array,
                "stomate_daily_process_fold_from_entries",
                slow_axes.get(name, ()),
            )
        )
    ok_values = dict(record.ok_leak_updates)
    perma_peat = record.ok_leak_result.soilcarbon.perma_peat
    if perma_peat is None:
        raise ValueError("PFT14 fast-day target requires deepC_peat")
    ok_values["deepC_peat"] = perma_peat.deepc_peat
    for name in FAST_DAY_OK_LEAK_FIELDS:
        array = np.asarray(ok_values[name])
        values.append(
            (
                "ok_leak",
                None,
                (name,),
                array,
                "stomate_lpj:OK_LEAK half-hour fold",
                slow_axes.get(name, ()),
            )
        )
    final = record.half_hour_transition.completed_entry_payloads[-1]
    for name in ("t2mdiag", "temp_sol"):
        values.append(
            (
                "final_diagnostics",
                None,
                (name,),
                np.asarray(final[name]),
                "compact_sechiba_entry_payload",
                (),
            )
        )
    return tuple(values)


def build_fast_day_target_leaves(
    record, active_pft_indices: tuple[int, ...]
) -> tuple[FastDayTargetLeafSpec, ...]:
    cursor = 0
    leaves = []
    for family, component, path, array, owner, axis_names in _fast_day_target_inventory(record):
        if axis_names and len(axis_names) != array.ndim:
            raise ValueError(
                f"fast-day target axis drift for {component or family}.{'.'.join(path)}: "
                f"{axis_names} vs shape {array.shape}"
            )
        selected = _select_pft_axes(array, axis_names, active_pft_indices)
        stop = cursor + int(selected.size)
        leaves.append(
            FastDayTargetLeafSpec(
                family=family,
                component=component,
                path=path,
                shape=tuple(selected.shape),
                full_shape=tuple(array.shape),
                dtype=str(array.dtype),
                start=cursor,
                stop=stop,
                owner=owner,
                axis_names=axis_names,
                selected_pft_indices=(
                    active_pft_indices if "nvm" in axis_names else ()
                ),
            )
        )
        cursor = stop
    return tuple(leaves)


def extract_fast_day_target(
    record, leaves: Sequence[FastDayTargetLeafSpec]
) -> np.ndarray:
    inventory = {
        (family, component, path): array
        for family, component, path, array, _owner, _axes in _fast_day_target_inventory(record)
    }
    width = max((leaf.stop for leaf in leaves), default=0)
    result = np.empty(width, dtype=np.float64)
    for leaf in leaves:
        array = _select_pft_axes(
            inventory[(leaf.family, leaf.component, leaf.path)],
            leaf.axis_names,
            leaf.selected_pft_indices,
        )
        if tuple(array.shape) != leaf.shape:
            raise ValueError(f"fast-day target shape drift for {leaf.key}")
        result[leaf.start : leaf.stop] = array.reshape(-1)
    return result


def fast_day_target_leaves_from_metadata(
    contract_metadata: Mapping[str, Any],
) -> tuple[FastDayTargetLeafSpec, ...]:
    """Parse and validate the learned-output layout stored in a manifest."""

    leaves = tuple(
        FastDayTargetLeafSpec(
            family=str(item["family"]),
            component=(
                None
                if item.get("component") is None
                else str(item["component"])
            ),
            path=tuple(str(name) for name in item["path"]),
            shape=tuple(int(size) for size in item["shape"]),
            full_shape=tuple(int(size) for size in item["full_shape"]),
            dtype=str(item["dtype"]),
            start=int(item["start"]),
            stop=int(item["stop"]),
            owner=str(item["owner"]),
            axis_names=tuple(str(name) for name in item.get("axis_names", ())),
            selected_pft_indices=tuple(
                int(index) for index in item.get("selected_pft_indices", ())
            ),
        )
        for item in contract_metadata["fast_day_target_leaves"]
    )
    cursor = 0
    for leaf in leaves:
        if leaf.start != cursor or leaf.stop <= leaf.start:
            raise ValueError(f"non-contiguous fast-day target leaf {leaf.key}")
        if leaf.stop - leaf.start != int(np.prod(leaf.shape, dtype=np.int64)):
            raise ValueError(f"fast-day target leaf width mismatch for {leaf.key}")
        cursor = leaf.stop
    if cursor != int(contract_metadata["fast_day_target_width"]):
        raise ValueError("fast-day target metadata width mismatch")
    return leaves


def reconstruct_fast_day_target(
    target: np.ndarray,
    leaves: Sequence[FastDayTargetLeafSpec],
    *,
    template_fields: Mapping[str, Mapping[str, Any]],
) -> ReconstructedFastDayTarget:
    """Inflate one compact PFT14 prediction into the retained-tail boundary."""

    target = np.asarray(target, dtype=np.float64)
    width = max((leaf.stop for leaf in leaves), default=0)
    if target.shape != (width,):
        raise ValueError(
            f"fast-day target width mismatch: {target.shape} != {(width,)}"
        )
    fields = deepcopy(dict(template_fields))
    groups: dict[str, dict[str, Any]] = {
        "daily_interface": {},
        "ok_leak": {},
        "final_diagnostics": {},
    }

    def get_path(root: Mapping[str, Any], path: tuple[str, ...]) -> Any | None:
        value: Any = root
        for name in path:
            if not isinstance(value, Mapping) or name not in value:
                return None
            value = value[name]
        return value

    def set_path(root: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
        current = root
        for name in path[:-1]:
            current = current.setdefault(name, {})
        current[path[-1]] = value

    for leaf in leaves:
        compact = target[leaf.start : leaf.stop].reshape(leaf.shape).astype(
            np.dtype(leaf.dtype), copy=False
        )
        if leaf.component is not None:
            component = fields.setdefault(leaf.component, {})
            full = _inflate_pft_axes(
                compact,
                leaf,
                base=get_path(component, leaf.path),
            )
            set_path(component, leaf.path, full)
        else:
            if leaf.family not in groups:
                raise ValueError(f"unknown fast-day target family {leaf.family}")
            full = _inflate_pft_axes(compact, leaf)
            set_path(groups[leaf.family], leaf.path, full)
    return ReconstructedFastDayTarget(
        fields_by_component=fields,
        daily_fields=groups["daily_interface"],
        ok_leak_updates=groups["ok_leak"],
        final_diagnostics=groups["final_diagnostics"],
    )


def _compiled_select_pft_axes(value, leaf):
    result = jnp.asarray(value)
    for axis, name in enumerate(leaf.axis_names):
        if name == "nvm":
            result = jnp.take(
                result,
                jnp.asarray(leaf.selected_pft_indices, dtype=jnp.int32),
                axis=axis,
            )
    return result


def _spec_value(values_by_component, spec, component: str, path: tuple[str, ...]):
    component_index = spec.components.index(component)
    field_names = spec.field_names_by_component[component_index]
    field_index = field_names.index(path[0])
    value = values_by_component[component_index][field_index]
    for name in path[1:]:
        value = value[name]
    return value


def make_compiled_training_output_projector(
    contract: DailyMarkovContract,
    *,
    day_start_state_spec,
    boundary_state_spec,
):
    """Build a trace-time projector from large runtime trees to compact rows."""

    continuous_leaves = tuple(
        leaf for leaf in contract.state_leaves if not leaf.discrete
    )
    discrete_leaves = contract.discrete_leaves

    def target_value(boundary, leaf: FastDayTargetLeafSpec):
        if leaf.component is not None:
            return _spec_value(
                boundary.half_hour_state_values,
                boundary_state_spec,
                leaf.component,
                leaf.path,
            )
        name = leaf.path[0]
        if leaf.family == "daily_interface":
            return boundary.daily_fields[name]
        if leaf.family == "ok_leak":
            return (
                boundary.deepc_peat
                if name == "deepC_peat"
                else boundary.ok_leak_updates[name]
            )
        return boundary.final_diagnostics[name]

    def projector(current_values, boundary):
        state = jnp.concatenate(
            tuple(
                _compiled_select_pft_axes(
                    _spec_value(
                        current_values,
                        day_start_state_spec,
                        leaf.component,
                        leaf.path,
                    ),
                    leaf,
                ).reshape(-1)
                for leaf in continuous_leaves
            )
        )
        discrete = tuple(
            _compiled_select_pft_axes(
                _spec_value(
                    current_values,
                    day_start_state_spec,
                    leaf.component,
                    leaf.path,
                ),
                leaf,
            )
            for leaf in discrete_leaves
        )
        target = jnp.concatenate(
            tuple(
                _compiled_select_pft_axes(target_value(boundary, leaf), leaf).reshape(-1)
                for leaf in contract.fast_day_target_leaves
            )
        )
        diagnostics = jnp.concatenate(
            tuple(
                _compiled_select_pft_axes(
                    (
                        boundary.daily_fields[leaf.name.removeprefix("daily_fold.")]
                        if leaf.name.startswith("daily_fold.")
                        else boundary.final_diagnostics[
                            leaf.name.removeprefix("final_half_hour.")
                        ]
                    ),
                    leaf,
                ).reshape(-1)
                for leaf in contract.diagnostic_leaves
            )
        )
        return state, discrete, target, diagnostics

    return projector


def build_daily_markov_contract(
    packet,
    record,
    *,
    parameter_leaves: tuple[ConditionLeafSpec, ...],
    landpoint_static_leaves: tuple[ConditionLeafSpec, ...],
    annual_condition_leaves: tuple[ConditionLeafSpec, ...],
    native_forcing_spec: NativeForcingSpec,
) -> DailyMarkovContract:
    for name, leaves, expected_role in (
        ("parameters", parameter_leaves, "landpoint_parameter"),
        ("landpoint_static", landpoint_static_leaves, "landpoint_static"),
        ("annual_conditions", annual_condition_leaves, "annual_exogenous"),
    ):
        cursor = 0
        seen = set()
        for leaf in leaves:
            if leaf.name in seen:
                raise ValueError(f"duplicate {name} condition leaf {leaf.name}")
            if leaf.start != cursor or leaf.stop - leaf.start != int(np.prod(leaf.shape)):
                raise ValueError(f"non-contiguous {name} condition leaf {leaf.name}")
            if leaf.temporal_role != expected_role:
                raise ValueError(f"invalid temporal role for {name}.{leaf.name}")
            seen.add(leaf.name)
            cursor = leaf.stop
    cursor = 0
    state_leaves = []
    end_packet = record.expected_result.day_end_state
    active_pft_indices = _pft14_indices(end_packet)
    for component, path, array, classification, source_ref, axis_names in _canonical_state_inventory(end_packet):
        selected_array = _select_pft_axes(array, axis_names, active_pft_indices)
        if classification == "discrete":
            start = stop = None
        else:
            start = cursor
            cursor += int(selected_array.size)
            stop = cursor
        state_leaves.append(
            StateLeafSpec(
                component=component,
                path=path,
                shape=tuple(selected_array.shape),
                full_shape=tuple(array.shape),
                dtype=str(array.dtype),
                start=start,
                stop=stop,
                classification=classification,
                source_ref=source_ref,
                missing_policy=(
                    "zero_after_year_start_rebase"
                    if component == "hydrol_previous_step_state" and path == ("nroot",)
                    else "error"
                ),
                axis_names=axis_names,
                selected_pft_indices=(
                    active_pft_indices if "nvm" in axis_names else ()
                ),
            )
        )
    diagnostic_cursor = 0
    diagnostic_leaves = []
    for name, array, owner, axis_names in _diagnostic_inventory(record):
        selected_array = _select_pft_axes(array, axis_names, active_pft_indices)
        stop = diagnostic_cursor + int(selected_array.size)
        diagnostic_leaves.append(
            DiagnosticSpec(
                name=name,
                shape=tuple(selected_array.shape),
                dtype=str(array.dtype),
                start=diagnostic_cursor,
                stop=stop,
                owner=owner,
                axis_names=axis_names,
                selected_pft_indices=(
                    active_pft_indices if "nvm" in axis_names else ()
                ),
            )
        )
        diagnostic_cursor = stop
    fast_day_target_leaves = build_fast_day_target_leaves(
        record, active_pft_indices
    )
    contract = DailyMarkovContract(
        schema_version=CONTRACT_SCHEMA_VERSION,
        state_leaves=tuple(state_leaves),
        fast_day_target_leaves=fast_day_target_leaves,
        diagnostic_leaves=tuple(diagnostic_leaves),
        native_forcing=native_forcing_spec,
        static_conditions=StaticConditionSpec(
            parameter_leaves=parameter_leaves,
            landpoint_static_leaves=landpoint_static_leaves,
            annual_condition_leaves=annual_condition_leaves,
            includes=(
                "PFT14 parameters",
                "soil and hydrology tables",
                "land geometry",
                "salinity",
                "tide",
                "year-varying atmospheric CO2",
            ),
        ),
        source_hashes=(
            (str(_SECHIBA_CONTRACT.relative_to(ROOT)), _sha256_file(_SECHIBA_CONTRACT)),
            (str(_RESTART_CONTRACT.relative_to(ROOT)), _sha256_file(_RESTART_CONTRACT)),
        ),
        active_pft_indices=active_pft_indices,
    )
    extract_state(packet, contract, allow_year_start_missing=True)
    return contract


def _value_at_path(
    packet, leaf: StateLeafSpec, *, allow_year_start_missing: bool = False
) -> np.ndarray:
    value: Any = packet.fields_by_component[leaf.component]
    for name in leaf.path:
        if (
            isinstance(value, Mapping)
            and name not in value
            and allow_year_start_missing
            and leaf.missing_policy == "zero_after_year_start_rebase"
        ):
            return np.zeros(leaf.shape, dtype=np.dtype(leaf.dtype))
        value = value[name]
    array = _select_pft_axes(
        np.asarray(value), leaf.axis_names, leaf.selected_pft_indices
    )
    if tuple(array.shape) != leaf.shape:
        raise ValueError(f"state shape drift for {leaf.key}: {array.shape} != {leaf.shape}")
    return array


def extract_state(
    packet,
    contract: DailyMarkovContract,
    *,
    allow_year_start_missing: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    continuous = np.empty(contract.continuous_state_width, dtype=np.float64)
    discrete: dict[str, np.ndarray] = {}
    for leaf in contract.state_leaves:
        value = _value_at_path(
            packet, leaf, allow_year_start_missing=allow_year_start_missing
        )
        if leaf.discrete:
            discrete[leaf.key] = value.copy()
        else:
            continuous[leaf.start : leaf.stop] = value.reshape(-1)
    return continuous, discrete


def build_state_trajectory(
    packets: Sequence[Any], contract: DailyMarkovContract
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    extracted = [
        extract_state(packet, contract, allow_year_start_missing=index == 0)
        for index, packet in enumerate(packets)
    ]
    trajectory = np.stack([value[0] for value in extracted])
    discrete = {
        leaf.key: np.stack([value[1][leaf.key] for value in extracted])
        for leaf in contract.discrete_leaves
    }
    return trajectory, discrete


def _inflate_pft_axes(
    value: np.ndarray,
    leaf: StateLeafSpec | FastDayTargetLeafSpec,
    *,
    base: Any | None = None,
) -> np.ndarray:
    if not leaf.selected_pft_indices:
        return np.asarray(value).reshape(leaf.full_shape)
    result = (
        np.zeros(leaf.full_shape, dtype=np.asarray(value).dtype)
        if base is None
        else np.asarray(base, dtype=np.asarray(value).dtype).reshape(leaf.full_shape).copy()
    )
    selected = np.asarray(value).reshape(leaf.shape)
    pft_axes = [axis for axis, name in enumerate(leaf.axis_names) if name == "nvm"]
    if len(pft_axes) != 1:
        raise ValueError(f"unsupported PFT-axis count for {leaf.key}: {pft_axes}")
    axis = pft_axes[0]
    index = [slice(None)] * result.ndim
    for source_index, target_index in enumerate(leaf.selected_pft_indices):
        index[axis] = target_index
        source = [slice(None)] * selected.ndim
        source[axis] = source_index
        result[tuple(index)] = selected[tuple(source)]
    return result


def _set_nested_field(fields: dict[str, Any], leaf: StateLeafSpec, value: np.ndarray) -> None:
    target = fields.setdefault(leaf.component, {})
    for name in leaf.path[:-1]:
        target = target.setdefault(name, {})
    target[leaf.path[-1]] = value


def _get_nested_field(fields: Mapping[str, Any], leaf: StateLeafSpec) -> Any | None:
    value: Any = fields.get(leaf.component)
    for name in leaf.path:
        if not isinstance(value, Mapping) or name not in value:
            return None
        value = value[name]
    return value


def _zero_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _zero_tree(item) for name, item in value.items()}
    return np.zeros_like(np.asarray(value))


def reconstruct_state_fields(
    continuous: np.ndarray,
    discrete: Mapping[str, np.ndarray],
    contract: DailyMarkovContract,
    *,
    template_fields: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Inflate canonical PFT14 state and rebuild deterministic packet mirrors."""

    continuous = np.asarray(continuous, dtype=np.float64)
    if continuous.shape != (contract.continuous_state_width,):
        raise ValueError("continuous state width does not match the Markov contract")
    fields: dict[str, dict[str, Any]] = (
        {} if template_fields is None else deepcopy(dict(template_fields))
    )
    for leaf in contract.state_leaves:
        if leaf.discrete:
            if leaf.key not in discrete:
                raise ValueError(f"missing discrete state leaf {leaf.key}")
            compact = np.asarray(discrete[leaf.key], dtype=np.dtype(leaf.dtype))
        else:
            compact = continuous[leaf.start : leaf.stop].reshape(leaf.shape)
        _set_nested_field(
            fields,
            leaf,
            _inflate_pft_axes(compact, leaf, base=_get_nested_field(fields, leaf)),
        )

    slow = fields.get(_SLOW_COMPONENT, {})
    if "daily_accumulators" in slow:
        slow["daily_accumulators"] = _zero_tree(slow["daily_accumulators"])
    diffuco = fields.setdefault("diffuco_previous_step_state", {})
    for name in _SLOWPROC_DIFFUCO_MIRRORS:
        if name in slow:
            diffuco[name] = np.asarray(slow[name]).copy()

    finalize = fields.get(_FINALIZE_COMPONENT)
    if finalize is not None:
        owner_order = (
            "hydrol_previous_step_state",
            "thermosoil_previous_step_state",
            "enerbil_previous_step_state",
            "diffuco_previous_step_state",
            "driver_previous_step_state",
            _SLOW_COMPONENT,
        )
        for name, previous in tuple(finalize.items()):
            for component in owner_order:
                candidate = fields.get(component, {}).get(name)
                if candidate is not None and np.shape(candidate) == np.shape(previous):
                    finalize[name] = np.asarray(candidate).copy()
                    break
    return fields


def assert_markov_continuity(
    day_start_packets: Sequence[Any], records: Sequence[Any], final_packet: Any, contract: DailyMarkovContract
) -> None:
    if len(day_start_packets) != len(records):
        raise ValueError("Teacher day-start and record counts differ")
    expected_next = [record.expected_result.day_end_state for record in records]
    observed_next = [*day_start_packets[1:], final_packet]
    for offset, (expected, observed) in enumerate(zip(expected_next, observed_next, strict=True)):
        expected_continuous, expected_discrete = extract_state(expected, contract)
        observed_continuous, observed_discrete = extract_state(observed, contract)
        if not np.array_equal(expected_continuous, observed_continuous, equal_nan=True):
            raise ValueError(f"Teacher state continuity failed after day offset {offset}")
        if expected_discrete.keys() != observed_discrete.keys() or any(
            not np.array_equal(expected_discrete[key], observed_discrete[key], equal_nan=True)
            for key in expected_discrete
        ):
            raise ValueError(f"Teacher discrete-state continuity failed after day offset {offset}")


def extract_diagnostics(record, contract: DailyMarkovContract) -> np.ndarray:
    inventory = {
        name: (value, axis_names)
        for name, value, _owner, axis_names in _diagnostic_inventory(record)
    }
    result = np.empty(contract.diagnostic_width, dtype=np.float64)
    for leaf in contract.diagnostic_leaves:
        raw_value, axis_names = inventory[leaf.name]
        value = _select_pft_axes(
            np.asarray(raw_value), axis_names, leaf.selected_pft_indices
        )
        if tuple(value.shape) != leaf.shape:
            raise ValueError(f"diagnostic shape drift for {leaf.name}")
        result[leaf.start : leaf.stop] = value.reshape(-1)
    return result


def forcing_record_indices_for_day(*, day_index: int, raw_steps: int, split: int) -> np.ndarray:
    if day_index < 1:
        raise ValueError("day_index is one-based and must be positive")
    first_model_tstep = (int(day_index) - 1) * 48
    first_read = first_model_tstep // int(split) + 1
    previous_read = first_read - 1 if first_read > 1 else int(round(86400.0 / (split * 1800.0)))
    reads = (previous_read, first_read, first_read + 1, first_read + 2, first_read + 3)
    return np.asarray(
        [forcing_domain._paper_forcing_raw_index_for_model_read(value, raw_steps) for value in reads],
        dtype=np.int32,
    )


def native_forcing_days(context, *, year: int, day_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray, NativeForcingSpec]:
    first = context.first_step_bundle
    if first is None:
        raise ValueError("native forcing capture requires a prepared first-step bundle")
    domain = first.domain
    land_indices_key = tuple(
        (int(i), int(j))
        for i, j in np.asarray(domain.forcing_indices_zero_based, dtype=np.int64)
    )
    cache = forcing_domain._read_forcing_land_cache(
        str(Path(context.config_path).resolve()), int(year), land_indices_key
    )
    raw_steps = int(cache.Tair.shape[0])
    indices = np.stack(
        [
            forcing_record_indices_for_day(
                day_index=int(day), raw_steps=raw_steps, split=int(context.forcing_split)
            )
            for day in day_indices
        ]
    )
    values = []
    for day_indices_raw in indices:
        columns = []
        for name in NATIVE_FORCING_FIELDS:
            value = np.asarray(getattr(cache, name), dtype=np.float64)[day_indices_raw]
            columns.append(value.reshape((5, -1)))
        values.append(np.concatenate(columns, axis=1))
    spec = NativeForcingSpec(
        fields=NATIVE_FORCING_FIELDS,
        field_shape=tuple(np.asarray(cache.Tair).shape[1:]),
        source_records_per_day=5,
        source_interval_seconds=float(context.forcing_split) * float(context.dt_sechiba),
        model_interval_seconds=float(context.dt_sechiba),
        split=int(context.forcing_split),
        precipitation_spread_steps=int(context.forcing_nb_spread),
    )
    return np.stack(values), indices, spec


def _native_columns(window: np.ndarray, spec: NativeForcingSpec) -> dict[str, jnp.ndarray]:
    width = int(np.prod(spec.field_shape, dtype=np.int64))
    if window.shape != (spec.source_records_per_day, spec.width):
        raise ValueError(f"native forcing shape {window.shape} does not match contract")
    return {
        name: jnp.asarray(window[:, index * width : (index + 1) * width], dtype=jnp.float64)
        for index, name in enumerate(spec.fields)
    }


def _shortwave_factors(context, *, day_index: int) -> np.ndarray:
    domain = context.first_step_bundle.domain
    local_i = (np.asarray(domain.kindex, dtype=np.int64) - 1) % int(domain.iim)
    local_j = (np.asarray(domain.kindex, dtype=np.int64) - 1) // int(domain.iim)
    local_land_indices = tuple((int(i), int(j)) for i, j in zip(local_i, local_j, strict=True))
    shape = (int(domain.nbindex), 1)
    factors = []
    start = (int(day_index) - 1) * 48
    for model_tstep in range(start, start + 48):
        factors.append(
            forcing_domain._paper_model_step_swdown_land_from_solarang(
                np.ones(shape, dtype=np.float64),
                model_tstep=model_tstep,
                split=int(context.forcing_split),
                dt_force=float(context.forcing_split) * float(context.dt_sechiba),
                lon=domain.lon,
                lat=domain.lat,
                land_indices_key=local_land_indices,
            ).reshape(-1)
        )
    return np.stack(factors)


def reconstruct_compiled_forcing_day(
    window: np.ndarray,
    spec: NativeForcingSpec,
    context,
    *,
    year: int,
    day_index: int,
    annual_co2_ppm: float | None = None,
) -> DriverCompiledHalfHourForcing:
    """Reconstruct all 48 Teacher forcing steps from five native records."""

    raw = _native_columns(np.asarray(window), spec)
    offsets = jnp.arange(48, dtype=jnp.int32)
    interval = offsets // int(spec.split)
    substep = offsets % int(spec.split) + 1
    previous_slot = interval
    current_slot = interval + 1
    weight = substep.astype(jnp.float64) / float(spec.split)

    def linear(name: str) -> jnp.ndarray:
        previous = raw[name][previous_slot]
        current = raw[name][current_slot]
        return previous + (current - previous) * weight[:, None]

    def spread(name: str) -> jnp.ndarray:
        current = raw[name][current_slot]
        scale = jnp.where(
            substep <= int(spec.precipitation_spread_steps),
            float(spec.split) / float(spec.precipitation_spread_steps),
            0.0,
        )
        return current * scale[:, None] * float(spec.model_interval_seconds)

    first = context.first_step_bundle
    npts = int(first.domain.nbindex)
    height_lev1 = float(first.forcing.Height_Lev1)
    height_levuv = float(first.forcing.Height_Levuv)
    if annual_co2_ppm is None:
        annual_co2_ppm = read_annual_co2(context.config_path, int(year))
    ccanopy = ccanopy_from_co2(float(annual_co2_ppm), npts)
    salinity = jnp.asarray(first.static_trace_fields.salinity)
    tide_height = jnp.asarray(first.static_trace_fields.tide_height)
    swdown = jnp.minimum(
        raw["SWdown"][current_slot] * jnp.asarray(_shortwave_factors(context, day_index=day_index)),
        2000.0,
    )
    return DriverCompiledHalfHourForcing(
        zlev=jnp.full((48, npts), height_lev1, dtype=jnp.float64),
        zlevuv=jnp.full((48, npts), height_levuv, dtype=jnp.float64),
        u=linear("Wind_N").reshape((48, *spec.field_shape)),
        v=linear("Wind_E").reshape((48, *spec.field_shape)),
        qair=linear("Qair"),
        temp_air=linear("Tair"),
        pb=linear("PSurf") / 100.0,
        precip_rain=spread("Rainf"),
        precip_snow=spread("Snowf"),
        lwdown=linear("LWdown"),
        swdown=swdown,
        ccanopy=jnp.broadcast_to(jnp.asarray(ccanopy), (48, *np.shape(ccanopy))),
        salinity=jnp.broadcast_to(salinity, (48, *salinity.shape)),
        tide_height=jnp.broadcast_to(tide_height, (48, *tide_height.shape)),
    )


def load_markov_shard(
    path: Path, *, contract: DailyMarkovContract | None = None
) -> MarkovShard:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    required = {
        "state_trajectory",
        "fast_day_target",
        "forcing_native",
        "forcing_record_indices",
        "parameters",
        "landpoint_static",
        "annual_conditions",
        "diagnostics",
        "year",
        "day_index",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"v3 Markov shard is missing arrays: {missing}")
    days = int(arrays["day_index"].size)
    if arrays["state_trajectory"].shape[0] != days + 1:
        raise ValueError("state_trajectory must contain S[0:T+1]")
    per_day = (
        "forcing_native",
        "forcing_record_indices",
        "fast_day_target",
        "diagnostics",
    )
    if any(arrays[name].shape[0] != days for name in per_day):
        raise ValueError("forcing and diagnostics must contain one row per day")
    if arrays["year"].ndim != 0 or arrays["day_index"].ndim != 1:
        raise ValueError("year must be scalar and day_index must be one-dimensional")
    if arrays["parameters"].ndim != 1 or arrays["landpoint_static"].ndim != 1:
        raise ValueError("parameter and landpoint-static conditions must be one-dimensional")
    if arrays["annual_conditions"].ndim != 1:
        raise ValueError("annual_conditions must be one vector per landpoint-year shard")
    if contract is not None:
        expected_widths = {
            "state_trajectory": contract.continuous_state_width,
            "fast_day_target": contract.fast_day_target_width,
            "forcing_native": contract.native_forcing.width,
            "parameters": contract.static_conditions.parameter_width,
            "landpoint_static": contract.static_conditions.landpoint_static_width,
            "annual_conditions": contract.static_conditions.annual_condition_width,
            "diagnostics": contract.diagnostic_width,
        }
        observed_widths = {
            "state_trajectory": arrays["state_trajectory"].shape[1],
            "fast_day_target": arrays["fast_day_target"].shape[1],
            "forcing_native": arrays["forcing_native"].shape[2],
            "parameters": arrays["parameters"].size,
            "landpoint_static": arrays["landpoint_static"].size,
            "annual_conditions": arrays["annual_conditions"].size,
            "diagnostics": arrays["diagnostics"].shape[1],
        }
        drift = {
            name: (observed_widths[name], expected)
            for name, expected in expected_widths.items()
            if observed_widths[name] != expected
        }
        if drift:
            raise ValueError(f"Markov shard contract width drift: {drift}")
    discrete = {
        name.removeprefix("state_discrete__"): value
        for name, value in arrays.items()
        if name.startswith("state_discrete__")
    }
    if any(value.shape[0] != days + 1 for value in discrete.values()):
        raise ValueError("discrete state trajectories must contain S[0:T+1]")
    return MarkovShard(
        state_trajectory=arrays["state_trajectory"],
        fast_day_target=arrays["fast_day_target"],
        forcing_native=arrays["forcing_native"],
        forcing_record_indices=arrays["forcing_record_indices"],
        parameters=arrays["parameters"],
        landpoint_static=arrays["landpoint_static"],
        annual_conditions=arrays["annual_conditions"],
        diagnostics=arrays["diagnostics"],
        year=arrays["year"],
        day_index=arrays["day_index"],
        discrete_trajectories=discrete,
    )


def estimated_uncompressed_bytes(arrays: Mapping[str, np.ndarray]) -> int:
    return sum(int(value.nbytes) for value in arrays.values())
