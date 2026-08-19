"""Axis-aligned seven-expert architecture for the approved strong baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .neural_architecture import (
    NeuralInputNormalization,
    NeuralOutputNormalization,
    NeuralOutputSchema,
)
from .output_transform import field_coordinates, normalized_output_to_physical
from .parameter_routing import apply_fast_parameter_routes
from .process_derivations import NativeForcingFieldLayout
from .state_features import assemble_local_pft_features
from .strong_baseline_contract import (
    StrongBaselineContract,
    StrongBaselineFieldOwnership,
    canonical_axis_aliases,
    resolve_strong_baseline_field_ledger,
)
from .types import AssembledDailyOperatorInput


@dataclass(frozen=True)
class StrongBaselineConfig:
    d_model: int = 384
    attention_heads: int = 8
    ffn_width: int = 1536
    expert_blocks: int = 3
    forcing_width: int = 256
    forcing_heads: int = 8
    forcing_ffn_width: int = 1024
    forcing_blocks: int = 2
    axial_blocks: int = 1
    head_width: int = 256
    graph_rounds: int = 2

    def validate(self, contract: StrongBaselineContract | None = None) -> None:
        if self.d_model % self.attention_heads or self.forcing_width % self.forcing_heads:
            raise ValueError("strong-baseline attention widths must divide their head counts")
        if self.ffn_width != 4 * self.d_model:
            raise ValueError("strong-baseline expert FFN expansion must remain four times width")
        if self.forcing_ffn_width != 4 * self.forcing_width:
            raise ValueError("strong-baseline forcing FFN expansion must remain four times width")
        if contract is None:
            return
        capacity = contract.payload["capacity"]
        observed = (
            self.d_model,
            self.attention_heads,
            self.expert_blocks,
            self.forcing_width,
            self.forcing_blocks,
            self.axial_blocks,
            self.graph_rounds,
        )
        declared = (
            int(capacity["expert_width"]),
            int(capacity["attention_heads"]),
            int(capacity["expert_blocks"]),
            int(capacity["forcing_width"]),
            int(capacity["forcing_blocks"]),
            int(capacity["axial_blocks"]),
            int(contract.payload["graph_messages"]["rounds"]),
        )
        if observed != declared:
            raise ValueError("strong-baseline executable configuration drift")


@dataclass(frozen=True, eq=False)
class StrongBaselineLayout:
    ledger: tuple[StrongBaselineFieldOwnership, ...]
    expert_names: tuple[str, ...]
    source_family_names: tuple[str, ...]
    axis_names: tuple[str, ...]
    axis_extents: tuple[int, ...]
    axis_offsets: tuple[int, ...]
    active_pft_indices: tuple[int, ...]
    pft_state_indices: np.ndarray
    field_expert_index: np.ndarray
    field_label_index: np.ndarray
    state_field_pool: np.ndarray
    state_field_family_index: np.ndarray
    state_axis_pool: np.ndarray
    state_element_pft_index: np.ndarray
    discrete_state_names: tuple[str, ...]
    discrete_state_field_pool: np.ndarray
    discrete_state_field_family_index: np.ndarray
    discrete_state_axis_pool: np.ndarray
    discrete_state_element_pft_index: np.ndarray
    static_field_pool: np.ndarray
    static_field_family_index: np.ndarray
    static_axis_pool: np.ndarray
    static_element_pft_index: np.ndarray
    annual_field_pool: np.ndarray
    annual_field_family_index: np.ndarray
    forcing_family_index: np.ndarray
    process_static_family_index: np.ndarray
    process_static_field_index: np.ndarray
    process_static_position_index: np.ndarray
    process_static_axis_pool: np.ndarray
    process_static_element_pft_index: np.ndarray
    source_allowed: np.ndarray
    graph_edge_mask: np.ndarray
    local_state_indices: tuple[np.ndarray, ...]
    t2m_longterm_start: int
    t2m_longterm_stop: int
    model_interval_seconds: float

    @property
    def axis_token_count(self) -> int:
        return int(sum(self.axis_extents))

    @property
    def max_output_rank(self) -> int:
        return max((len(field.shape) for field in self.ledger), default=0)


class StrongBaselineContext(NamedTuple):
    source_tokens: Any
    source_mask: Any
    source_family_index: Any
    expert_axis_tokens: Any
    expert_axis_mask: Any
    pft_tokens: Any
    pft_mask: Any


class StrongBaselineFieldContext(NamedTuple):
    features: Any
    active: Any
    axis_tokens: Any
    axis_mask: Any
    local_state_token: Any
    local_state_mask: Any


def _reverse_members(groups: Mapping[str, Sequence[str]], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for family, members in groups.items():
        for member in members:
            if member in result:
                raise ValueError(f"strong-baseline {label} source member overlaps: {member}")
            result[str(member)] = str(family)
    return result


def _output_state_owner(field: StrongBaselineFieldOwnership) -> str | None:
    if field.day_start_owner is not None:
        return field.day_start_owner
    if field.name.startswith("inventory.water."):
        source = field.name.removeprefix("inventory.water.").split(".", 1)[0]
        return f"hydrol_previous_step_state.{source}"
    if field.name.startswith("inventory.carbon."):
        source = field.name.removeprefix("inventory.carbon.").split(".", 1)[0]
        return f"slowproc_stomate_previous_step_state.{source}"
    if field.name.startswith("thermal."):
        _prefix, component, source, suffix = field.name.split(".")
        if suffix != "normalized_tendency":
            raise ValueError(f"strong-baseline thermal output suffix drift: {field.name}")
        return f"{component}.{source}"
    return None


def _local_state_indices(
    ledger: tuple[StrongBaselineFieldOwnership, ...],
    adapter: Any,
    contract: StrongBaselineContract,
) -> tuple[np.ndarray, ...]:
    leaves = {leaf.key: leaf for leaf in adapter.contract.state_leaves}
    aliases = canonical_axis_aliases(contract)
    active_pft_indices = tuple(int(value) for value in adapter.active_pft_indices)
    result = []
    for field in ledger:
        owner = _output_state_owner(field)
        if owner is None:
            result.append(np.zeros((0,), dtype=np.int32))
            continue
        leaf = leaves.get(owner)
        if leaf is None:
            raise ValueError(f"strong-baseline local output owner is absent: {owner}")
        if leaf.discrete:
            raise ValueError(f"strong-baseline local output owner is discrete: {owner}")
        source_axes = tuple(aliases[name] for name in leaf.axis_names)
        if source_axes != field.axes:
            raise ValueError(
                f"strong-baseline local output axes drift for {field.name}: {source_axes} versus {field.axes}"
            )
        output_coordinates = field_coordinates(field.shape)
        indices = np.full((field.scalar_count,), -1, dtype=np.int32)
        for output_index, output_coordinate in enumerate(output_coordinates):
            source_coordinate = []
            valid = True
            for axis, output_value, output_extent, source_extent in zip(
                field.axes,
                output_coordinate,
                field.shape,
                leaf.shape,
                strict=True,
            ):
                output_value = int(output_value)
                output_extent = int(output_extent)
                source_extent = int(source_extent)
                if axis == "pft" and source_extent == len(active_pft_indices):
                    if output_extent == source_extent:
                        source_value = output_value
                    else:
                        matches = tuple(
                            compact for compact, full in enumerate(active_pft_indices) if full == output_value
                        )
                        if not matches:
                            valid = False
                            break
                        source_value = matches[0]
                elif axis == "pft" and output_extent == len(active_pft_indices):
                    source_value = active_pft_indices[output_value]
                else:
                    source_value = output_value
                if source_value >= source_extent:
                    valid = False
                    break
                source_coordinate.append(source_value)
            if valid:
                local_index = np.ravel_multi_index(tuple(source_coordinate), leaf.shape)
                indices[output_index] = int(leaf.start) + int(local_index)
        result.append(indices)
    return tuple(result)


def _condition_pool(leaves: Sequence[Any], width: int) -> np.ndarray:
    rows = np.zeros((len(leaves), width), dtype=np.bool_)
    for index, leaf in enumerate(leaves):
        rows[index, int(leaf.start) : int(leaf.stop)] = True
    return rows


def _pft_state_indices(
    adapter: Any,
    contract: StrongBaselineContract,
) -> np.ndarray:
    """Map every compact PFT-state feature to its continuous-state statistic."""

    leaves = {leaf.key: leaf for leaf in adapter.contract.state_leaves}
    aliases = canonical_axis_aliases(contract)
    active = tuple(int(value) for value in adapter.active_pft_indices)
    parts = []
    for name in ("veget_max", "lai", "biomass"):
        key = f"slowproc_stomate_previous_step_state.{name}"
        leaf = leaves.get(key)
        if leaf is None:
            raise ValueError(f"strong-baseline PFT state owner is absent: {key}")
        canonical_axes = tuple(aliases[axis] for axis in leaf.axis_names)
        pft_axes = tuple(index for index, axis in enumerate(canonical_axes) if axis == "pft")
        if len(pft_axes) != 1:
            raise ValueError(f"strong-baseline PFT state owner has invalid axes: {key}")
        pft_axis = pft_axes[0]
        indices = np.arange(int(leaf.start), int(leaf.stop), dtype=np.int32).reshape(leaf.shape)
        coordinates = tuple(range(len(active))) if leaf.shape[pft_axis] == len(active) else active
        selected = np.take(indices, np.asarray(coordinates, dtype=np.int32), axis=pft_axis)
        parts.append(np.moveaxis(selected, pft_axis, 0).reshape((len(active), -1)))
    return np.concatenate(tuple(parts), axis=1)


def _source_axis_pool(
    *,
    field_names: tuple[str, ...],
    field_shapes: tuple[tuple[int, ...], ...],
    field_starts: tuple[int, ...],
    field_stops: tuple[int, ...],
    field_families: tuple[str, ...],
    bindings: Mapping[str, Sequence[Mapping[str, Any]]],
    axis_names: tuple[str, ...],
    axis_extents: tuple[int, ...],
    contract: StrongBaselineContract,
    active_pft_indices: tuple[int, ...],
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    experts = contract.expert_names
    axis_offsets = np.cumsum((0, *axis_extents[:-1]), dtype=np.int32)
    pool = np.zeros((len(experts), sum(axis_extents), width), dtype=np.bool_)
    element_pft_index = np.full((width,), -1, dtype=np.int32)
    for field_name, field_bindings in bindings.items():
        field_index = field_names.index(str(field_name))
        shape = field_shapes[field_index]
        start = field_starts[field_index]
        stop = field_stops[field_index]
        family = field_families[field_index]
        grid = (
            np.indices(shape, dtype=np.int32).reshape((len(shape), -1)).T if shape else np.zeros((1, 0), dtype=np.int32)
        )
        if grid.shape[0] != stop - start:
            raise ValueError(f"strong-baseline source-axis field width drift: {field_name}")
        for binding in field_bindings:
            dimension = int(binding["dimension"])
            axis = str(binding["axis"])
            mode = str(binding["mode"])
            if dimension >= len(shape):
                raise ValueError(f"strong-baseline source-axis rank drift: {field_name}")
            axis_index = axis_names.index(axis)
            extent = axis_extents[axis_index]
            source_extent = shape[dimension]
            if mode in {"full_pft", "compact_pft"}:
                element_pft_index[start:stop] = -2
            for coordinate in range(extent):
                if mode == "full_pft":
                    source_coordinates = (active_pft_indices[coordinate],)
                elif mode == "compact_pft":
                    source_coordinates = (coordinate,)
                elif mode == "interval_boundaries":
                    source_coordinates = tuple(value for value in (coordinate, coordinate + 1) if value < source_extent)
                else:
                    source_coordinates = (coordinate,)
                if not source_coordinates or max(source_coordinates) >= source_extent:
                    raise ValueError(f"strong-baseline source-axis extent drift: {field_name} {binding}")
                local = np.isin(grid[:, dimension], source_coordinates)
                if mode in {"full_pft", "compact_pft"}:
                    positions = start + np.flatnonzero(local)
                    existing = element_pft_index[positions]
                    if np.any((existing >= 0) & (existing != coordinate)):
                        raise ValueError(f"strong-baseline source PFT ownership overlaps: {field_name}")
                    element_pft_index[positions] = coordinate
                token_index = int(axis_offsets[axis_index] + coordinate)
                for expert_index, expert in enumerate(experts):
                    admitted = contract.payload["experts"][expert]["admitted_sources"]
                    if family in admitted:
                        pool[expert_index, token_index, start:stop] |= local
    return pool, element_pft_index


def _state_layout(
    adapter: Any,
    contract: StrongBaselineContract,
    *,
    axis_names: tuple[str, ...],
    axis_extents: tuple[int, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    source_members = contract.payload["source_members"]
    state_sources = _reverse_members(source_members["state_components"], "state")
    family_names = tuple(str(value) for value in contract.payload["source_families"])
    leaves = tuple(leaf for leaf in adapter.contract.state_leaves if not leaf.discrete)
    discrete_leaves = tuple(leaf for leaf in adapter.contract.state_leaves if leaf.discrete)
    width = int(adapter.contract.continuous_state_width)
    field_pool = np.zeros((len(leaves), width), dtype=np.bool_)
    field_family = np.zeros((len(leaves),), dtype=np.int32)
    aliases = canonical_axis_aliases(contract)
    active_full = tuple(int(value) for value in adapter.active_pft_indices)
    experts = contract.expert_names
    axis_offsets = np.cumsum((0, *axis_extents[:-1]), dtype=np.int32)
    axis_pool = np.zeros((len(experts), sum(axis_extents), width), dtype=np.bool_)
    element_pft_index = np.full((width,), -1, dtype=np.int32)

    for field_index, leaf in enumerate(leaves):
        family = state_sources.get(str(leaf.component))
        if family is None:
            raise ValueError(f"strong-baseline state component has no source: {leaf.component}")
        family_index = family_names.index(family)
        field_family[field_index] = family_index
        start = int(leaf.start)
        stop = int(leaf.stop)
        field_pool[field_index, start:stop] = True
        shape = tuple(int(value) for value in leaf.shape)
        axes = tuple(str(value) for value in leaf.axis_names)
        grid = (
            np.indices(shape, dtype=np.int32).reshape((len(shape), -1)).T if shape else np.zeros((1, 0), dtype=np.int32)
        )
        pft_axes = tuple(index for index, axis in enumerate(axes) if aliases.get(axis) == "pft")
        if pft_axes:
            if len(pft_axes) != 1:
                raise ValueError(f"strong-baseline state PFT axes overlap: {leaf.key}")
            pft_axis = pft_axes[0]
            raw_extent = shape[pft_axis]
            element_pft_index[start:stop] = -2
            for compact, full in enumerate(active_full):
                raw_coordinate = compact if raw_extent == len(active_full) else full
                if raw_coordinate < raw_extent:
                    local = grid[:, pft_axis] == raw_coordinate
                    element_pft_index[start:stop][local] = compact
        for raw_axis_index, raw_axis in enumerate(axes):
            canonical = aliases.get(raw_axis)
            if canonical not in axis_names:
                continue
            axis_index = axis_names.index(canonical)
            extent = axis_extents[axis_index]
            raw_extent = shape[raw_axis_index]
            for coordinate in range(extent):
                if canonical == "pft":
                    if raw_extent == len(active_full):
                        raw_coordinate = coordinate
                    elif coordinate < len(active_full) and raw_extent > active_full[coordinate]:
                        raw_coordinate = active_full[coordinate]
                    else:
                        continue
                elif coordinate < raw_extent:
                    raw_coordinate = coordinate
                else:
                    continue
                local = grid[:, raw_axis_index] == raw_coordinate
                token_index = int(axis_offsets[axis_index] + coordinate)
                for expert_index, expert in enumerate(experts):
                    admitted = contract.payload["experts"][expert]["admitted_sources"]
                    if family in admitted:
                        axis_pool[expert_index, token_index, start:stop] |= local
    discrete_offsets = np.cumsum(
        (0, *(int(np.prod(leaf.shape, dtype=np.int64)) for leaf in discrete_leaves)),
        dtype=np.int32,
    )
    discrete_width = int(discrete_offsets[-1])
    discrete_field_pool = np.zeros((len(discrete_leaves), discrete_width), dtype=np.bool_)
    discrete_field_family = np.zeros((len(discrete_leaves),), dtype=np.int32)
    discrete_axis_pool = np.zeros(
        (len(experts), sum(axis_extents), discrete_width),
        dtype=np.bool_,
    )
    discrete_element_pft_index = np.full((discrete_width,), -1, dtype=np.int32)
    for field_index, (leaf, start, stop) in enumerate(
        zip(discrete_leaves, discrete_offsets[:-1], discrete_offsets[1:], strict=True)
    ):
        family = state_sources.get(str(leaf.component))
        if family is None:
            raise ValueError(f"strong-baseline discrete component has no source: {leaf.component}")
        discrete_field_family[field_index] = family_names.index(family)
        discrete_field_pool[field_index, start:stop] = True
        shape = tuple(int(value) for value in leaf.shape)
        axes = tuple(str(value) for value in leaf.axis_names)
        grid = (
            np.indices(shape, dtype=np.int32).reshape((len(shape), -1)).T if shape else np.zeros((1, 0), dtype=np.int32)
        )
        pft_axes = tuple(index for index, axis in enumerate(axes) if aliases.get(axis) == "pft")
        if pft_axes:
            if len(pft_axes) != 1:
                raise ValueError(f"strong-baseline discrete PFT axes overlap: {leaf.key}")
            pft_axis = pft_axes[0]
            raw_extent = shape[pft_axis]
            discrete_element_pft_index[start:stop] = -2
            for compact, full in enumerate(active_full):
                raw_coordinate = compact if raw_extent == len(active_full) else full
                if raw_coordinate < raw_extent:
                    local = grid[:, pft_axis] == raw_coordinate
                    discrete_element_pft_index[start:stop][local] = compact
        for raw_axis_index, raw_axis in enumerate(axes):
            canonical = aliases.get(raw_axis)
            if canonical not in axis_names:
                continue
            axis_index = axis_names.index(canonical)
            extent = axis_extents[axis_index]
            raw_extent = shape[raw_axis_index]
            for coordinate in range(extent):
                if canonical == "pft":
                    if raw_extent == len(active_full):
                        raw_coordinate = coordinate
                    elif coordinate < len(active_full) and raw_extent > active_full[coordinate]:
                        raw_coordinate = active_full[coordinate]
                    else:
                        continue
                elif coordinate < raw_extent:
                    raw_coordinate = coordinate
                else:
                    continue
                local = grid[:, raw_axis_index] == raw_coordinate
                token_index = int(axis_offsets[axis_index] + coordinate)
                for expert_index, expert in enumerate(experts):
                    admitted = contract.payload["experts"][expert]["admitted_sources"]
                    if family in admitted:
                        discrete_axis_pool[expert_index, token_index, start:stop] |= local
    return (
        field_pool,
        field_family,
        axis_pool,
        element_pft_index,
        tuple(leaf.key for leaf in discrete_leaves),
        discrete_field_pool,
        discrete_field_family,
        discrete_axis_pool,
        discrete_element_pft_index,
    )


def build_strong_baseline_layout(
    *,
    adapter: Any,
    schema: NeuralOutputSchema,
    owner_contract: Any,
    contract: StrongBaselineContract,
    inputs: AssembledDailyOperatorInput,
    resolved_ledger: tuple[StrongBaselineFieldOwnership, ...] | None = None,
) -> StrongBaselineLayout:
    """Freeze all executable source, expert, axis, graph, and local-state maps."""

    ledger = (
        resolve_strong_baseline_field_ledger(
            contract,
            schema=schema,
            adapter=adapter,
            owner_contract=owner_contract,
        )
        if resolved_ledger is None
        else tuple(resolved_ledger)
    )
    if tuple(field.name for field in ledger) != tuple(field.name for field in schema.fields):
        raise ValueError("strong-baseline resolved ledger and schema drift")
    expert_names = contract.expert_names
    source_family_names = tuple(str(value) for value in contract.payload["source_families"])
    axis_names = tuple(str(value) for value in contract.payload["axis_registry"])
    axis_extents = []
    for axis in axis_names:
        if axis == "pft":
            axis_extents.append(len(adapter.active_pft_indices))
            continue
        axis_extents.append(
            max(
                (
                    int(extent)
                    for field in ledger
                    for name, extent in zip(field.axes, field.shape, strict=True)
                    if name == axis
                ),
                default=1,
            )
        )
    axis_extents_tuple = tuple(axis_extents)
    axis_offsets = tuple(int(value) for value in np.cumsum((0, *axis_extents_tuple[:-1]), dtype=np.int32))
    (
        state_field_pool,
        state_field_family,
        state_axis_pool,
        state_element_pft_index,
        discrete_state_names,
        discrete_state_field_pool,
        discrete_state_field_family,
        discrete_state_axis_pool,
        discrete_state_element_pft_index,
    ) = _state_layout(
        adapter,
        contract,
        axis_names=axis_names,
        axis_extents=axis_extents_tuple,
    )

    members = contract.payload["source_members"]
    static_sources = _reverse_members(members["static_fields"], "static")
    static_leaves = adapter.contract.static_conditions.landpoint_static_leaves
    if {leaf.name for leaf in static_leaves} != set(static_sources):
        raise ValueError("strong-baseline static source registry drift")
    static_family = np.asarray(
        [source_family_names.index(static_sources[leaf.name]) for leaf in static_leaves],
        dtype=np.int32,
    )
    static_pool = _condition_pool(
        static_leaves,
        int(adapter.contract.static_conditions.landpoint_static_width),
    )
    static_names = tuple(str(leaf.name) for leaf in static_leaves)
    static_axis_pool, static_element_pft_index = _source_axis_pool(
        field_names=static_names,
        field_shapes=tuple(tuple(int(value) for value in leaf.shape) for leaf in static_leaves),
        field_starts=tuple(int(leaf.start) for leaf in static_leaves),
        field_stops=tuple(int(leaf.stop) for leaf in static_leaves),
        field_families=tuple(static_sources[name] for name in static_names),
        bindings=contract.payload["source_axis_bindings"]["static_fields"],
        axis_names=axis_names,
        axis_extents=axis_extents_tuple,
        contract=contract,
        active_pft_indices=tuple(int(value) for value in adapter.active_pft_indices),
        width=int(adapter.contract.static_conditions.landpoint_static_width),
    )

    annual_sources = _reverse_members(members["annual_fields"], "annual")
    annual_leaves = adapter.contract.static_conditions.annual_condition_leaves
    if {leaf.name for leaf in annual_leaves} != set(annual_sources):
        raise ValueError("strong-baseline annual source registry drift")
    annual_family = np.asarray(
        [source_family_names.index(annual_sources[leaf.name]) for leaf in annual_leaves],
        dtype=np.int32,
    )
    annual_pool = _condition_pool(
        annual_leaves,
        int(adapter.contract.static_conditions.annual_condition_width),
    )

    forcing_sources = _reverse_members(members["forcing_fields"], "forcing")
    forcing_fields = tuple(adapter.contract.native_forcing.fields)
    if set(forcing_fields) != set(forcing_sources):
        raise ValueError("strong-baseline forcing source registry drift")
    forcing_family = np.asarray(
        [source_family_names.index(forcing_sources[name]) for name in forcing_fields],
        dtype=np.int32,
    )

    process_sources = _reverse_members(members["process_static_fields"], "process-static")
    process_names = tuple(inputs.process_static._fields)
    if set(process_names) != set(process_sources):
        raise ValueError("strong-baseline process-static source registry drift")
    process_family = []
    process_field = []
    process_position = []
    process_starts = []
    process_stops = []
    process_shapes = []
    process_width = 0
    for field_index, name in enumerate(process_names):
        value = np.asarray(getattr(inputs.process_static, name))
        size = int(value.size)
        process_starts.append(process_width)
        process_width += size
        process_stops.append(process_width)
        process_shapes.append(tuple(int(extent) for extent in value.shape))
        process_family.extend([source_family_names.index(process_sources[name])] * size)
        process_field.extend([field_index] * size)
        process_position.extend(range(size))
    process_axis_pool, process_element_pft_index = _source_axis_pool(
        field_names=process_names,
        field_shapes=tuple(process_shapes),
        field_starts=tuple(process_starts),
        field_stops=tuple(process_stops),
        field_families=tuple(process_sources[name] for name in process_names),
        bindings=contract.payload["source_axis_bindings"]["process_static_fields"],
        axis_names=axis_names,
        axis_extents=axis_extents_tuple,
        contract=contract,
        active_pft_indices=tuple(int(value) for value in adapter.active_pft_indices),
        width=process_width,
    )

    source_allowed = np.zeros((len(expert_names), len(source_family_names)), dtype=np.bool_)
    for expert_index, expert in enumerate(expert_names):
        admitted = contract.payload["experts"][expert]["admitted_sources"]
        source_allowed[expert_index, [source_family_names.index(name) for name in admitted]] = True

    graph = np.zeros((len(expert_names), len(expert_names)), dtype=np.bool_)
    for sender, receiver in contract.payload["graph_messages"]["directed_edges"]:
        graph[expert_names.index(sender), expert_names.index(receiver)] = True
    tlong = tuple(
        leaf
        for leaf in adapter.contract.state_leaves
        if leaf.key == "slowproc_stomate_previous_step_state.t2m_longterm"
    )
    if len(tlong) != 1 or tuple(tlong[0].shape) != (1,):
        raise ValueError("strong-baseline requires one scalar t2m_longterm owner")
    return StrongBaselineLayout(
        ledger=ledger,
        expert_names=expert_names,
        source_family_names=source_family_names,
        axis_names=axis_names,
        axis_extents=axis_extents_tuple,
        axis_offsets=axis_offsets,
        active_pft_indices=tuple(int(value) for value in adapter.active_pft_indices),
        pft_state_indices=_pft_state_indices(adapter, contract),
        field_expert_index=np.asarray([expert_names.index(field.expert) for field in ledger], dtype=np.int32),
        field_label_index=np.asarray([field.label_index for field in schema.fields], dtype=np.int32),
        state_field_pool=state_field_pool,
        state_field_family_index=state_field_family,
        state_axis_pool=state_axis_pool,
        state_element_pft_index=state_element_pft_index,
        discrete_state_names=discrete_state_names,
        discrete_state_field_pool=discrete_state_field_pool,
        discrete_state_field_family_index=discrete_state_field_family,
        discrete_state_axis_pool=discrete_state_axis_pool,
        discrete_state_element_pft_index=discrete_state_element_pft_index,
        static_field_pool=static_pool,
        static_field_family_index=static_family,
        static_axis_pool=static_axis_pool,
        static_element_pft_index=static_element_pft_index,
        annual_field_pool=annual_pool,
        annual_field_family_index=annual_family,
        forcing_family_index=forcing_family,
        process_static_family_index=np.asarray(process_family, dtype=np.int32),
        process_static_field_index=np.asarray(process_field, dtype=np.int32),
        process_static_position_index=np.asarray(process_position, dtype=np.int32),
        process_static_axis_pool=process_axis_pool,
        process_static_element_pft_index=process_element_pft_index,
        source_allowed=source_allowed,
        graph_edge_mask=graph,
        local_state_indices=_local_state_indices(ledger, adapter, contract),
        t2m_longterm_start=int(tlong[0].start),
        t2m_longterm_stop=int(tlong[0].stop),
        model_interval_seconds=float(adapter.contract.native_forcing.model_interval_seconds),
    )


def _normal(key: Any, shape: tuple[int, ...], fan_in: int) -> Any:
    scale = jnp.asarray(1.0 / np.sqrt(float(max(fan_in, 1))), dtype=jnp.float32)
    return jax.random.normal(key, shape, dtype=jnp.float32) * scale


def _linear_parameters(key: Any, in_width: int, out_width: int) -> dict[str, Any]:
    return {
        "kernel": _normal(key, (in_width, out_width), in_width),
        "bias": jnp.zeros((out_width,), dtype=jnp.float32),
    }


def _attention_parameters(key: Any, width: int) -> dict[str, Any]:
    keys = jax.random.split(key, 4)
    return {
        name: _linear_parameters(item, width, width)
        for name, item in zip(("query", "key", "value", "output"), keys, strict=True)
    }


def _ffn_parameters(key: Any, width: int, hidden: int) -> dict[str, Any]:
    first, second = jax.random.split(key)
    return {
        "first": _linear_parameters(first, width, hidden),
        "second": _linear_parameters(second, hidden, width),
    }


def _residual_block_parameters(key: Any, width: int, hidden: int) -> dict[str, Any]:
    attention, ffn = jax.random.split(key)
    return {
        "attention": _attention_parameters(attention, width),
        "ffn": _ffn_parameters(ffn, width, hidden),
        "attention_norm_scale": jnp.ones((width,), dtype=jnp.float32),
        "attention_norm_bias": jnp.zeros((width,), dtype=jnp.float32),
        "ffn_norm_scale": jnp.ones((width,), dtype=jnp.float32),
        "ffn_norm_bias": jnp.zeros((width,), dtype=jnp.float32),
    }


def initialize_strong_baseline_parameters(
    key: Any,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    inputs: AssembledDailyOperatorInput,
    contract: StrongBaselineContract,
    config: StrongBaselineConfig = StrongBaselineConfig(),
) -> Mapping[str, Any]:
    """Initialize the approved tree without historical neural parameters."""

    config.validate(contract)
    keys = iter(jax.random.split(key, 512))
    pft_width = int(np.asarray(assemble_local_pft_features(inputs.pft)).shape[-1])
    graph_logit = np.log(0.01 / 0.99)
    label_heads = []
    for _label in schema.labels:
        first = _linear_parameters(next(keys), config.d_model, config.head_width)
        second = _linear_parameters(next(keys), config.head_width, 1)
        second["kernel"] = second["kernel"] * 1.0e-3
        label_heads.append({"first": first, "second": second})
    return {
        "forcing_input": _linear_parameters(next(keys), 5, config.forcing_width),
        "forcing_blocks": tuple(
            _residual_block_parameters(next(keys), config.forcing_width, config.forcing_ffn_width)
            for _ in range(config.forcing_blocks)
        ),
        "axial_blocks": tuple(
            _residual_block_parameters(next(keys), config.d_model, config.ffn_width)
            for _ in range(config.axial_blocks)
        ),
        "forcing_to_model": _linear_parameters(next(keys), config.forcing_width, config.d_model),
        "summary_input": _linear_parameters(next(keys), 4, config.d_model),
        "pft_input": _linear_parameters(next(keys), pft_width, config.d_model),
        "process_static_input": _linear_parameters(next(keys), 1, config.d_model),
        "calendar_input": _linear_parameters(next(keys), 4, config.d_model),
        "local_state_input": _linear_parameters(next(keys), 2, config.d_model),
        "source_family_embedding": _normal(
            next(keys), (len(layout.source_family_names), config.d_model), config.d_model
        ),
        "forcing_field_embedding": _normal(
            next(keys), (layout.forcing_family_index.size, config.forcing_width), config.forcing_width
        ),
        "forcing_position_embedding": _normal(
            next(keys),
            (int(np.asarray(inputs.forcing.values).shape[0]), config.forcing_width),
            config.forcing_width,
        ),
        "state_field_embedding": _normal(
            next(keys), (layout.state_field_pool.shape[0], config.d_model), config.d_model
        ),
        "discrete_state_field_embedding": _normal(
            next(keys),
            (layout.discrete_state_field_pool.shape[0], config.d_model),
            config.d_model,
        ),
        "static_field_embedding": _normal(
            next(keys), (layout.static_field_pool.shape[0], config.d_model), config.d_model
        ),
        "annual_field_embedding": _normal(
            next(keys), (layout.annual_field_pool.shape[0], config.d_model), config.d_model
        ),
        "process_static_field_embedding": _normal(
            next(keys), (len(inputs.process_static._fields), config.d_model), config.d_model
        ),
        "process_static_position_embedding": _normal(
            next(keys),
            (int(np.max(layout.process_static_position_index)) + 1, config.d_model),
            config.d_model,
        ),
        "expert_embedding": _normal(next(keys), (len(layout.expert_names), config.d_model), config.d_model),
        "expert_query": _normal(next(keys), (len(layout.expert_names), config.d_model), config.d_model),
        "axis_role_embedding": _normal(next(keys), (len(layout.axis_names), config.d_model), config.d_model),
        "axis_position_embedding": tuple(
            None if axis == "pft" else _normal(next(keys), (extent, config.d_model), config.d_model)
            for axis, extent in zip(layout.axis_names, layout.axis_extents, strict=True)
        ),
        "axis_slot_projection": tuple(
            _linear_parameters(next(keys), config.d_model, config.d_model) for _ in range(layout.max_output_rank)
        ),
        "field_embedding": _normal(next(keys), (len(schema.fields), config.d_model), config.d_model),
        "expert_blocks": tuple(
            tuple(
                _residual_block_parameters(next(keys), config.d_model, config.ffn_width)
                for _ in range(config.expert_blocks)
            )
            for _ in layout.expert_names
        ),
        "graph_sender": tuple(
            tuple(_linear_parameters(next(keys), config.d_model, config.d_model) for _ in layout.expert_names)
            for _ in range(config.graph_rounds)
        ),
        "graph_gate_logits": jnp.full(
            (config.graph_rounds, len(layout.expert_names), len(layout.expert_names)),
            graph_logit,
            dtype=jnp.float32,
        ),
        "graph_norm_scale": jnp.ones(
            (config.graph_rounds, len(layout.expert_names), config.d_model),
            dtype=jnp.float32,
        ),
        "graph_norm_bias": jnp.zeros(
            (config.graph_rounds, len(layout.expert_names), config.d_model),
            dtype=jnp.float32,
        ),
        "pair_projection": tuple(
            _linear_parameters(next(keys), config.d_model, config.d_model) for _ in layout.expert_names
        ),
        "pair_gate_logits": jnp.full((len(layout.expert_names),), graph_logit, dtype=jnp.float32),
        "label_heads": tuple(label_heads),
    }


def strong_baseline_parameter_count(parameters: Mapping[str, Any]) -> int:
    return int(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(parameters)))


def _linear(parameters: Mapping[str, Any], value: Any) -> Any:
    return jnp.asarray(value, dtype=jnp.float32) @ parameters["kernel"] + parameters["bias"]


def _layer_norm(value: Any, scale: Any, bias: Any) -> Any:
    value = jnp.asarray(value, dtype=jnp.float32)
    mean = jnp.mean(value, axis=-1, keepdims=True)
    variance = jnp.mean((value - mean) ** 2, axis=-1, keepdims=True)
    return (value - mean) * jax.lax.rsqrt(variance + 1.0e-6) * scale + bias


def _ffn(parameters: Mapping[str, Any], value: Any) -> Any:
    return _linear(parameters["second"], jax.nn.gelu(_linear(parameters["first"], value)))


def _attention(
    parameters: Mapping[str, Any],
    query: Any,
    key_value: Any,
    *,
    heads: int,
    mask: Any,
) -> Any:
    query = _linear(parameters["query"], query)
    key = _linear(parameters["key"], key_value)
    value = _linear(parameters["value"], key_value)
    width = int(query.shape[-1])
    head_width = width // heads
    query = query.reshape(query.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
    key = key.reshape(key.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
    value = value.reshape(value.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
    logits = jnp.einsum("...hqd,...hkd->...hqk", query, key) / np.sqrt(float(head_width))
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    logits = jnp.where(mask[..., None, :, :], logits, -jnp.inf)
    row_valid = jnp.any(mask, axis=-1, keepdims=True)
    safe_logits = jnp.where(row_valid[..., None, :, :], logits, 0.0)
    weights = jax.nn.softmax(safe_logits, axis=-1)
    weights = jnp.where(mask[..., None, :, :], weights, 0.0)
    result = jnp.einsum("...hqk,...hkd->...hqd", weights, value)
    result = result.swapaxes(-3, -2).reshape(result.shape[:-3] + (result.shape[-2], width))
    return _linear(parameters["output"], result)


def _masked_summary(pool: np.ndarray, values: Any, defined: Any) -> tuple[Any, Any]:
    pool = jnp.asarray(pool, dtype=jnp.bool_)
    values = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    defined = jnp.asarray(defined, dtype=jnp.bool_).reshape(-1)
    values = jnp.where(defined, values, 0.0)
    selected = pool & defined[None, :]
    count = jnp.sum(selected, axis=1, dtype=jnp.float32)
    safe = jnp.where(selected, values[None, :], 0.0)
    mean = jnp.sum(safe, axis=1) / jnp.maximum(count, 1.0)
    centered = jnp.where(selected, values[None, :] - mean[:, None], 0.0)
    variance = jnp.sum(centered * centered, axis=1) / jnp.maximum(count, 1.0)
    minimum = jnp.min(jnp.where(selected, values[None, :], jnp.inf), axis=1)
    maximum = jnp.max(jnp.where(selected, values[None, :], -jnp.inf), axis=1)
    valid = count > 0.0
    summary = jnp.stack((mean, jnp.sqrt(jnp.maximum(variance, 0.0)), minimum, maximum), axis=1)
    return jnp.where(valid[:, None], summary, 0.0), valid


def _standardize(value: Any, mean: Any, scale: Any) -> Any:
    value = jnp.asarray(value, dtype=jnp.float32)
    mean = jnp.asarray(mean, dtype=jnp.float32)
    scale = jnp.asarray(scale, dtype=jnp.float32)
    safe_scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
    return jnp.where(jnp.isfinite(value), (value - mean) / safe_scale, 0.0)


def _mask_inactive_pft_elements(defined: Any, element_pft_index: np.ndarray, pft_mask: Any) -> Any:
    defined = jnp.asarray(defined, dtype=jnp.bool_).reshape(-1)
    element_pft_index = jnp.asarray(element_pft_index, dtype=jnp.int32)
    assigned = element_pft_index >= 0
    safe = jnp.maximum(element_pft_index, 0)
    structurally_valid = element_pft_index != -2
    return defined & structurally_valid & (~assigned | jnp.asarray(pft_mask, dtype=jnp.bool_)[safe])


def _forcing_tokens(
    parameters: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    normalization: NeuralInputNormalization,
    layout: StrongBaselineLayout,
    config: StrongBaselineConfig,
) -> tuple[Any, Any, Any]:
    forcing = inputs.forcing
    values = _standardize(forcing.values, normalization.forcing_mean, normalization.forcing_scale)
    records, fields = values.shape
    time = jnp.repeat(jnp.asarray(forcing.record_time_seconds, dtype=jnp.float32) / 86400.0, fields)
    duration = jnp.repeat(jnp.asarray(forcing.duration_seconds, dtype=jnp.float32) / 86400.0, fields)
    predecessor = jnp.repeat(jnp.asarray(forcing.predecessor_mask, dtype=jnp.float32), fields)
    valid = jnp.repeat(jnp.asarray(forcing.record_mask, dtype=jnp.bool_), fields)
    features = jnp.stack((values.reshape(-1), time, duration, predecessor, valid.astype(jnp.float32)), axis=1)
    field_index = jnp.tile(jnp.arange(fields, dtype=jnp.int32), records)
    record_index = jnp.repeat(jnp.arange(records, dtype=jnp.int32), fields)
    tokens = _linear(parameters["forcing_input"], features)
    tokens += parameters["forcing_field_embedding"][field_index]
    tokens += parameters["forcing_position_embedding"][record_index]
    family = jnp.tile(jnp.asarray(layout.forcing_family_index, dtype=jnp.int32), records)
    pair_mask = valid[:, None] & valid[None, :] & (family[:, None] == family[None, :])
    for block in parameters["forcing_blocks"]:
        message = _attention(block["attention"], tokens, tokens, heads=config.forcing_heads, mask=pair_mask)
        tokens = _layer_norm(
            tokens + message,
            block["attention_norm_scale"],
            block["attention_norm_bias"],
        )
        tokens = _layer_norm(
            tokens + _ffn(block["ffn"], tokens),
            block["ffn_norm_scale"],
            block["ffn_norm_bias"],
        )
        tokens = jnp.where(valid[:, None], tokens, 0.0)
    return _linear(parameters["forcing_to_model"], tokens), valid, family


def _factorized_axial_tokens(
    parameters: Mapping[str, Any],
    tokens: Any,
    mask: Any,
    *,
    layout: StrongBaselineLayout,
    config: StrongBaselineConfig,
) -> Any:
    """Mix coordinates within each physical axis using one shared block stack."""

    max_extent = max(layout.axis_extents)
    token_groups = []
    mask_groups = []
    for offset, extent in zip(layout.axis_offsets, layout.axis_extents, strict=True):
        token_groups.append(
            jnp.pad(
                tokens[:, offset : offset + extent],
                ((0, 0), (0, max_extent - extent), (0, 0)),
            )
        )
        mask_groups.append(
            jnp.pad(
                mask[:, offset : offset + extent],
                ((0, 0), (0, max_extent - extent)),
            )
        )
    grouped_tokens = jnp.stack(tuple(token_groups), axis=1)
    grouped_mask = jnp.stack(tuple(mask_groups), axis=1)
    pair_mask = grouped_mask[..., :, None] & grouped_mask[..., None, :]
    for block in parameters["axial_blocks"]:
        message = _attention(
            block["attention"],
            grouped_tokens,
            grouped_tokens,
            heads=config.attention_heads,
            mask=pair_mask,
        )
        grouped_tokens = _layer_norm(
            grouped_tokens + message,
            block["attention_norm_scale"],
            block["attention_norm_bias"],
        )
        grouped_tokens = _layer_norm(
            grouped_tokens + _ffn(block["ffn"], grouped_tokens),
            block["ffn_norm_scale"],
            block["ffn_norm_bias"],
        )
        grouped_tokens = jnp.where(grouped_mask[..., None], grouped_tokens, 0.0)
    return jnp.concatenate(
        tuple(
            grouped_tokens[:, axis_index, :extent]
            for axis_index, extent in enumerate(layout.axis_extents)
        ),
        axis=1,
    )


def assemble_strong_baseline_context(
    parameters: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    *,
    normalization: NeuralInputNormalization,
    layout: StrongBaselineLayout,
    config: StrongBaselineConfig = StrongBaselineConfig(),
) -> StrongBaselineContext:
    """Encode native forcing and axis-preserving source summaries in float32."""

    pft_mask = jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
    state = _standardize(
        inputs.state.other,
        normalization.continuous_mean,
        normalization.continuous_scale,
    )
    state_defined = _mask_inactive_pft_elements(
        inputs.state_defined.other,
        layout.state_element_pft_index,
        pft_mask,
    )
    state_summary, state_mask = _masked_summary(
        layout.state_field_pool,
        state,
        state_defined,
    )
    state_tokens = _linear(parameters["summary_input"], state_summary)
    state_family = jnp.asarray(layout.state_field_family_index, dtype=jnp.int32)
    state_tokens += parameters["state_field_embedding"]
    state_tokens += parameters["source_family_embedding"][state_family]

    if set(inputs.discrete_state) != set(layout.discrete_state_names):
        raise ValueError("strong-baseline discrete-state registry drift")
    discrete_values = jnp.concatenate(
        tuple(
            jnp.asarray(inputs.discrete_state[name], dtype=jnp.float32).reshape(-1)
            for name in layout.discrete_state_names
        )
    )
    discrete_mask = _mask_inactive_pft_elements(
        jnp.isfinite(discrete_values),
        layout.discrete_state_element_pft_index,
        pft_mask,
    )
    discrete_values = jnp.where(
        discrete_mask,
        discrete_values / (1.0 + jnp.abs(discrete_values)),
        0.0,
    )
    discrete_summary, discrete_field_mask = _masked_summary(
        layout.discrete_state_field_pool,
        discrete_values,
        discrete_mask,
    )
    discrete_tokens = _linear(parameters["summary_input"], discrete_summary)
    discrete_family = jnp.asarray(
        layout.discrete_state_field_family_index,
        dtype=jnp.int32,
    )
    discrete_tokens += parameters["discrete_state_field_embedding"]
    discrete_tokens += parameters["source_family_embedding"][discrete_family]

    axis_summaries = []
    axis_masks = []
    for expert_index in range(len(layout.expert_names)):
        summary, mask = _masked_summary(
            layout.state_axis_pool[expert_index],
            state,
            state_defined,
        )
        token = _linear(parameters["summary_input"], summary)
        discrete_axis_summary, discrete_axis_mask = _masked_summary(
            layout.discrete_state_axis_pool[expert_index],
            discrete_values,
            discrete_mask,
        )
        token += _linear(parameters["summary_input"], discrete_axis_summary)
        mask |= discrete_axis_mask
        token += parameters["expert_embedding"][expert_index]
        for axis_index, (offset, extent) in enumerate(zip(layout.axis_offsets, layout.axis_extents, strict=True)):
            addition = parameters["axis_role_embedding"][axis_index]
            position = parameters["axis_position_embedding"][axis_index]
            if position is not None:
                addition = addition + position
            token = token.at[offset : offset + extent].add(addition)
        axis_summaries.append(token)
        axis_masks.append(mask)

    static = _standardize(
        inputs.static_conditions,
        normalization.static_mean,
        normalization.static_scale,
    )
    static_defined = _mask_inactive_pft_elements(
        jnp.ones_like(static, dtype=jnp.bool_),
        layout.static_element_pft_index,
        pft_mask,
    )
    static_summary, static_mask = _masked_summary(
        layout.static_field_pool,
        static,
        static_defined,
    )
    static_tokens = _linear(parameters["summary_input"], static_summary)
    static_family = jnp.asarray(layout.static_field_family_index, dtype=jnp.int32)
    static_tokens += parameters["static_field_embedding"]
    static_tokens += parameters["source_family_embedding"][static_family]

    annual = _standardize(
        inputs.annual_conditions,
        normalization.annual_mean,
        normalization.annual_scale,
    )
    annual_summary, annual_mask = _masked_summary(
        layout.annual_field_pool,
        annual,
        jnp.ones_like(annual, dtype=jnp.bool_),
    )
    annual_tokens = _linear(parameters["summary_input"], annual_summary)
    annual_family = jnp.asarray(layout.annual_field_family_index, dtype=jnp.int32)
    annual_tokens += parameters["annual_field_embedding"]
    annual_tokens += parameters["source_family_embedding"][annual_family]

    process_values = jnp.concatenate(
        tuple(
            jnp.asarray(getattr(inputs.process_static, name), dtype=jnp.float32).reshape(-1)
            for name in inputs.process_static._fields
        )
    )
    process_mask = _mask_inactive_pft_elements(
        jnp.isfinite(process_values),
        layout.process_static_element_pft_index,
        pft_mask,
    )
    process_values = jnp.where(
        process_mask,
        process_values / (1.0 + jnp.abs(process_values)),
        0.0,
    )
    process_tokens = _linear(parameters["process_static_input"], process_values[:, None])
    process_field = jnp.asarray(layout.process_static_field_index, dtype=jnp.int32)
    process_position = jnp.asarray(layout.process_static_position_index, dtype=jnp.int32)
    process_family = jnp.asarray(layout.process_static_family_index, dtype=jnp.int32)
    process_tokens += parameters["process_static_field_embedding"][process_field]
    process_tokens += parameters["process_static_position_embedding"][process_position]
    process_tokens += parameters["source_family_embedding"][process_family]
    process_tokens = jnp.where(process_mask[:, None], process_tokens, 0.0)

    for expert_index in range(len(layout.expert_names)):
        static_axis_summary, static_axis_mask = _masked_summary(
            layout.static_axis_pool[expert_index],
            static,
            static_defined,
        )
        process_axis_summary, process_axis_mask = _masked_summary(
            layout.process_static_axis_pool[expert_index],
            process_values,
            process_mask,
        )
        axis_summaries[expert_index] += _linear(
            parameters["summary_input"],
            static_axis_summary,
        )
        axis_summaries[expert_index] += _linear(
            parameters["summary_input"],
            process_axis_summary,
        )
        axis_masks[expert_index] |= static_axis_mask | process_axis_mask

    pft_state_indices = jnp.asarray(layout.pft_state_indices, dtype=jnp.int32)
    pft_state = _standardize(
        inputs.pft.pft_state,
        jnp.asarray(normalization.continuous_mean)[pft_state_indices],
        jnp.asarray(normalization.continuous_scale)[pft_state_indices],
    )
    pft_features = jnp.concatenate(
        (
            pft_state,
            jnp.asarray(inputs.pft.pft_traits, dtype=jnp.float32),
            jnp.asarray(inputs.pft.pft_fraction, dtype=jnp.float32)[:, None],
        ),
        axis=1,
    )
    pft_features = jnp.where(jnp.isfinite(pft_features), pft_features, 0.0)
    pft_tokens = _linear(parameters["pft_input"], pft_features)
    pft_family_index = layout.source_family_names.index("pft_structure")
    pft_tokens += parameters["source_family_embedding"][pft_family_index]
    pft_tokens = jnp.where(pft_mask[:, None], pft_tokens, 0.0)

    day = jnp.asarray(inputs.day_index, dtype=jnp.float32)
    year = jnp.asarray(inputs.year, dtype=jnp.float32)
    phase = 2.0 * jnp.pi * day / 365.0
    calendar_features = jnp.stack((year / 1000.0, day / 365.0, jnp.sin(phase), jnp.cos(phase)))
    calendar = _linear(parameters["calendar_input"], calendar_features)[None, :]
    calendar_family = layout.source_family_names.index("calendar")
    calendar += parameters["source_family_embedding"][calendar_family]

    forcing_tokens, forcing_mask, forcing_family = _forcing_tokens(parameters, inputs, normalization, layout, config)
    source_tokens = jnp.concatenate(
        (
            forcing_tokens,
            state_tokens,
            discrete_tokens,
            static_tokens,
            annual_tokens,
            process_tokens,
            pft_tokens,
            calendar,
        ),
        axis=0,
    )
    source_mask = jnp.concatenate(
        (
            forcing_mask,
            state_mask,
            discrete_field_mask,
            static_mask,
            annual_mask,
            process_mask,
            pft_mask,
            jnp.ones((1,), dtype=jnp.bool_),
        )
    )
    source_family = jnp.concatenate(
        (
            forcing_family,
            state_family,
            discrete_family,
            static_family,
            annual_family,
            process_family,
            jnp.full((pft_tokens.shape[0],), pft_family_index, dtype=jnp.int32),
            jnp.asarray([calendar_family], dtype=jnp.int32),
        )
    )
    expert_axis_tokens = _factorized_axial_tokens(
        parameters,
        jnp.stack(tuple(axis_summaries)),
        jnp.stack(tuple(axis_masks)),
        layout=layout,
        config=config,
    )
    return StrongBaselineContext(
        source_tokens=source_tokens,
        source_mask=source_mask,
        source_family_index=source_family,
        expert_axis_tokens=expert_axis_tokens,
        expert_axis_mask=jnp.stack(tuple(axis_masks)),
        pft_tokens=pft_tokens,
        pft_mask=pft_mask,
    )


def encode_strong_baseline_experts(
    parameters: Mapping[str, Any],
    context: StrongBaselineContext,
    *,
    layout: StrongBaselineLayout,
    message_gate_scale: float = 1.0,
    config: StrongBaselineConfig = StrongBaselineConfig(),
) -> Any:
    """Run seven independent local experts, then optional bounded message rounds."""

    allowed = jnp.asarray(layout.source_allowed, dtype=jnp.bool_)
    family = jnp.asarray(context.source_family_index, dtype=jnp.int32)
    experts = []
    for expert_index, blocks in enumerate(parameters["expert_blocks"]):
        source_mask = allowed[expert_index, family] & context.source_mask
        tokens = jnp.concatenate((context.source_tokens, context.expert_axis_tokens[expert_index]), axis=0)
        mask = jnp.concatenate((source_mask, context.expert_axis_mask[expert_index]))
        query = (parameters["expert_query"][expert_index] + parameters["expert_embedding"][expert_index])[None, :]
        for block in blocks:
            message = _attention(
                block["attention"],
                query,
                tokens,
                heads=config.attention_heads,
                mask=mask[None, :],
            )
            query = _layer_norm(
                query + message,
                block["attention_norm_scale"],
                block["attention_norm_bias"],
            )
            query = _layer_norm(
                query + _ffn(block["ffn"], query),
                block["ffn_norm_scale"],
                block["ffn_norm_bias"],
            )
        experts.append(query[0])
    expert_tokens = jnp.stack(tuple(experts))

    if message_gate_scale == 0.0:
        return expert_tokens

    edge_mask = jnp.asarray(layout.graph_edge_mask, dtype=jnp.bool_)
    for round_index in range(config.graph_rounds):
        sent = jnp.stack(
            tuple(
                _linear(parameters["graph_sender"][round_index][sender], expert_tokens[sender])
                for sender in range(len(layout.expert_names))
            )
        )
        gates = jax.nn.sigmoid(parameters["graph_gate_logits"][round_index])
        gates = jnp.where(edge_mask, gates * jnp.asarray(message_gate_scale, jnp.float32), 0.0)
        received = jnp.einsum("sr,sd->rd", gates, sent)
        received /= jnp.maximum(jnp.sum(gates, axis=0)[:, None], 1.0)
        expert_tokens = _layer_norm(
            expert_tokens + received,
            parameters["graph_norm_scale"][round_index],
            parameters["graph_norm_bias"][round_index],
        )
    return expert_tokens


def strong_baseline_field_context(
    parameters: Mapping[str, Any],
    expert_tokens: Any,
    context: StrongBaselineContext,
    inputs: AssembledDailyOperatorInput,
    *,
    field: Any,
    ownership: StrongBaselineFieldOwnership,
    field_index: int,
    layout: StrongBaselineLayout,
    normalization: NeuralInputNormalization,
    include_expert_token: bool = True,
) -> StrongBaselineFieldContext:
    coordinates = jnp.asarray(field_coordinates(field.shape), dtype=jnp.int32)
    expert_index = int(layout.field_expert_index[field_index])
    field_token = parameters["field_embedding"][field_index]
    if include_expert_token:
        field_token = field_token + expert_tokens[expert_index]
    features = jnp.broadcast_to(
        field_token,
        (coordinates.shape[0], expert_tokens.shape[-1]),
    )
    active = jnp.ones((coordinates.shape[0],), dtype=jnp.bool_)
    axis_features = []
    coordinate_axis_tokens = []
    coordinate_axis_masks = []
    for axis_index, (axis_name, extent) in enumerate(zip(ownership.axes, ownership.shape, strict=True)):
        coordinate = coordinates[:, axis_index]
        canonical_index = layout.axis_names.index(axis_name)
        if axis_name == "pft":
            if field.full_pft_axis:
                matches = (
                    coordinate[:, None]
                    == jnp.asarray(
                        layout.active_pft_indices,
                        dtype=jnp.int32,
                    )[None, :]
                )
                compact = jnp.argmax(matches, axis=1)
                active &= jnp.any(matches, axis=1)
            else:
                compact = coordinate
                active &= compact < context.pft_tokens.shape[0]
            safe = jnp.minimum(compact, context.pft_tokens.shape[0] - 1)
            active &= context.pft_mask[safe]
            token = jnp.where(active[:, None], context.pft_tokens[safe], 0.0)
            dynamic_coordinate = safe
        else:
            token = jnp.zeros_like(features)
            dynamic_coordinate = coordinate
        offset = int(layout.axis_offsets[canonical_index])
        dynamic_index = offset + jnp.minimum(dynamic_coordinate, layout.axis_extents[canonical_index] - 1)
        dynamic = context.expert_axis_tokens[expert_index, dynamic_index]
        dynamic_mask = context.expert_axis_mask[expert_index, dynamic_index]
        token += jnp.where(dynamic_mask[:, None], dynamic, 0.0)
        token += parameters["axis_role_embedding"][canonical_index]
        position = parameters["axis_position_embedding"][canonical_index]
        if position is not None:
            token += position[jnp.minimum(dynamic_coordinate, layout.axis_extents[canonical_index] - 1)]
        token = _linear(parameters["axis_slot_projection"][axis_index], token)
        features += token
        if axis_name != "landpoint":
            coordinate_axis_tokens.append(token)
            coordinate_axis_masks.append(active & dynamic_mask)
        if axis_name != "landpoint" and int(extent) > 1:
            axis_features.append(token)

    indices = layout.local_state_indices[field_index]
    if indices.size:
        mapped = indices >= 0
        safe_indices = np.maximum(indices, 0)
        local_defined = (
            jnp.asarray(mapped, dtype=jnp.bool_)
            & inputs.state_defined.other[jnp.asarray(safe_indices, dtype=jnp.int32)]
        )
        local_mean = jnp.asarray(normalization.continuous_mean)[safe_indices]
        local_raw = inputs.state.other[jnp.asarray(safe_indices, dtype=jnp.int32)]
        local = _standardize(
            jnp.where(local_defined, local_raw, local_mean),
            local_mean,
            jnp.asarray(normalization.continuous_scale)[safe_indices],
        )
    else:
        local = jnp.zeros((coordinates.shape[0],), dtype=jnp.float32)
        local_defined = jnp.zeros((coordinates.shape[0],), dtype=jnp.bool_)
    local_state_token = _linear(
        parameters["local_state_input"],
        jnp.stack((local, local_defined.astype(jnp.float32)), axis=1),
    )
    features += local_state_token

    if len(axis_features) >= 2:
        pair = jnp.zeros_like(features)
        count = 0
        for left in range(len(axis_features)):
            for right in range(left + 1, len(axis_features)):
                pair += axis_features[left] * axis_features[right]
                count += 1
        pair /= float(count)
        features += jax.nn.sigmoid(parameters["pair_gate_logits"][expert_index]) * _linear(
            parameters["pair_projection"][expert_index], pair
        )
    if coordinate_axis_tokens:
        axis_tokens = jnp.stack(tuple(coordinate_axis_tokens), axis=1)
        axis_mask = jnp.stack(tuple(coordinate_axis_masks), axis=1)
    else:
        axis_tokens = jnp.zeros(
            (coordinates.shape[0], 0, features.shape[-1]),
            dtype=features.dtype,
        )
        axis_mask = jnp.zeros((coordinates.shape[0], 0), dtype=jnp.bool_)
    return StrongBaselineFieldContext(
        features=features,
        active=active,
        axis_tokens=axis_tokens,
        axis_mask=axis_mask,
        local_state_token=local_state_token,
        local_state_mask=local_defined & active,
    )


def _field_raw_output(
    parameters: Mapping[str, Any],
    expert_tokens: Any,
    context: StrongBaselineContext,
    inputs: AssembledDailyOperatorInput,
    *,
    field: Any,
    ownership: StrongBaselineFieldOwnership,
    field_index: int,
    layout: StrongBaselineLayout,
    normalization: NeuralInputNormalization,
) -> tuple[Any, Any]:
    field_context = strong_baseline_field_context(
        parameters,
        expert_tokens,
        context,
        inputs,
        field=field,
        ownership=ownership,
        field_index=field_index,
        layout=layout,
        normalization=normalization,
    )
    head = parameters["label_heads"][field.label_index]
    raw = _linear(
        head["second"],
        jax.nn.gelu(_linear(head["first"], field_context.features)),
    )[:, 0]
    return raw, field_context.active


def strong_baseline_process_outputs(
    parameters: Mapping[str, Any],
    context: StrongBaselineContext,
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    input_normalization: NeuralInputNormalization,
    output_normalization: NeuralOutputNormalization,
    forcing_layout: NativeForcingFieldLayout,
    message_gate_scale: float = 1.0,
    config: StrongBaselineConfig = StrongBaselineConfig(),
) -> Mapping[str, Any]:
    expert_tokens = encode_strong_baseline_experts(
        parameters,
        context,
        layout=layout,
        message_gate_scale=message_gate_scale,
        config=config,
    )
    outputs = {}
    for field_index, (field, ownership) in enumerate(zip(schema.fields, layout.ledger, strict=True)):
        raw, active = _field_raw_output(
            parameters,
            expert_tokens,
            context,
            inputs,
            field=field,
            ownership=ownership,
            field_index=field_index,
            layout=layout,
            normalization=input_normalization,
        )
        outputs[field.name] = normalized_output_to_physical(
            raw,
            active,
            field=field,
            normalization=output_normalization,
            inputs=inputs,
        )
    return apply_fast_parameter_routes(outputs, inputs, layout, forcing_layout)


__all__ = [
    "StrongBaselineConfig",
    "StrongBaselineContext",
    "StrongBaselineFieldContext",
    "StrongBaselineLayout",
    "assemble_strong_baseline_context",
    "build_strong_baseline_layout",
    "encode_strong_baseline_experts",
    "initialize_strong_baseline_parameters",
    "strong_baseline_field_context",
    "strong_baseline_parameter_count",
    "strong_baseline_process_outputs",
]
