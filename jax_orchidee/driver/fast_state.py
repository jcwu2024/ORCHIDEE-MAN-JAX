"""Fixed-shape driver state bundle for future compiled runtime loops.

The current driver state contract is the audited
``DriverPreviousStepStatePacket`` dictionary packet.  This module provides a
lossless intermediate representation that separates dynamic values from the
static component/field/provenance layout.  It is intentionally not wired into
the model runner yet; it is a safe staging point before replacing Python
half-hour orchestration with coarser JAX loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterator

from jax import tree_util

if TYPE_CHECKING:
    from jax_orchidee.driver.orchestration import DriverPreviousStepStatePacket


@lru_cache(maxsize=128)
def _index_by_name(names: tuple[str, ...]) -> dict[str, int]:
    return {name: index for index, name in enumerate(names)}


@dataclass(frozen=True)
class DriverFastStateSpec:
    """Static layout for a driver previous-step state packet."""

    components: tuple[str, ...]
    field_names_by_component: tuple[tuple[str, ...], ...]
    provenance_by_component: tuple[tuple[str, ...], ...]


class DriverFastComponentFields(Mapping[str, object]):
    """Read-only mapping view over one component of a fast driver state."""

    def __init__(self, field_names: tuple[str, ...], values: tuple[Any, ...]) -> None:
        self._field_names = field_names
        self._values = values

    def __getitem__(self, key: str) -> object:
        try:
            index = _index_by_name(self._field_names)[key]
        except ValueError as exc:
            raise KeyError(key) from exc
        except KeyError:
            raise
        return self._values[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self._field_names)

    def __len__(self) -> int:
        return len(self._field_names)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DriverFastStateBundle:
    """Pytree-friendly dynamic driver state plus static packet layout."""

    tstep: Any
    values_by_component: tuple[tuple[Any, ...], ...]
    spec: DriverFastStateSpec

    def tree_flatten(self):
        return (self.tstep, self.values_by_component), self.spec

    @classmethod
    def tree_unflatten(cls, spec: DriverFastStateSpec, children):
        tstep, values_by_component = children
        return cls(
            tstep=tstep,
            values_by_component=tuple(tuple(values) for values in values_by_component),
            spec=spec,
        )

    @property
    def components(self) -> tuple[str, ...]:
        return self.spec.components

    @property
    def field_names_by_component(self) -> tuple[tuple[str, ...], ...]:
        return self.spec.field_names_by_component

    def component_fields(self, component: str) -> Mapping[str, object]:
        try:
            index = _index_by_name(self.spec.components)[component]
        except ValueError:
            return {}
        except KeyError:
            return {}
        return DriverFastComponentFields(
            self.spec.field_names_by_component[index],
            self.values_by_component[index],
        )

    @property
    def fields_by_component(self) -> dict[str, Mapping[str, object]]:
        return {
            component: self.component_fields(component)
            for component in self.spec.components
        }

    def component_provenance(self, component: str) -> tuple[str, ...]:
        try:
            index = self.spec.components.index(component)
        except ValueError:
            return ()
        return self.spec.provenance_by_component[index]

    @property
    def provenance_by_component(self) -> dict[str, tuple[str, ...]]:
        return {
            component: provenance
            for component, provenance in zip(
                self.spec.components,
                self.spec.provenance_by_component,
                strict=True,
            )
        }

    @property
    def covered_fields(self) -> dict[str, tuple[str, ...]]:
        return {
            component: field_names
            for component, field_names in zip(
                self.spec.components,
                self.spec.field_names_by_component,
                strict=True,
            )
        }

    @property
    def empty(self) -> bool:
        return not any(self.spec.field_names_by_component)


def fast_state_from_previous_packet(
    packet: DriverPreviousStepStatePacket,
) -> DriverFastStateBundle:
    """Convert an audited previous-step packet to a fixed-layout state bundle."""

    components = tuple(packet.fields_by_component.keys())
    field_names_by_component = tuple(
        tuple(packet.fields_by_component[component].keys())
        for component in components
    )
    values_by_component = tuple(
        tuple(packet.fields_by_component[component][name] for name in field_names)
        for component, field_names in zip(components, field_names_by_component, strict=True)
    )
    provenance_by_component = tuple(
        tuple(packet.provenance_by_component.get(component, ()))
        for component in components
    )
    return DriverFastStateBundle(
        tstep=int(packet.tstep),
        values_by_component=values_by_component,
        spec=DriverFastStateSpec(
            components=components,
            field_names_by_component=field_names_by_component,
            provenance_by_component=provenance_by_component,
        ),
    )


def fast_state_from_previous_fields(
    *,
    tstep: int,
    fields_by_component: Mapping[str, Mapping[str, object]],
    provenance_by_component: Mapping[str, tuple[str, ...]],
) -> DriverFastStateBundle:
    """Build a fast state directly from component field mappings."""

    components = tuple(
        component
        for component, fields in fields_by_component.items()
        if fields
    )
    field_names_by_component = tuple(
        tuple(fields_by_component[component].keys())
        for component in components
    )
    values_by_component = tuple(
        tuple(fields_by_component[component][name] for name in field_names)
        for component, field_names in zip(components, field_names_by_component, strict=True)
    )
    return DriverFastStateBundle(
        tstep=int(tstep),
        values_by_component=values_by_component,
        spec=DriverFastStateSpec(
            components=components,
            field_names_by_component=field_names_by_component,
            provenance_by_component=tuple(
                tuple(provenance_by_component.get(component, ()))
                for component in components
            ),
        ),
    )


def previous_packet_from_fast_state(
    state: DriverFastStateBundle,
) -> DriverPreviousStepStatePacket:
    """Rebuild the audited previous-step packet from a fast state bundle."""

    from jax_orchidee.driver.orchestration import DriverPreviousStepStatePacket

    fields_by_component: dict[str, dict[str, object]] = {}
    provenance_by_component: dict[str, tuple[str, ...]] = {}
    for component, field_names, values, provenance in zip(
        state.spec.components,
        state.spec.field_names_by_component,
        state.values_by_component,
        state.spec.provenance_by_component,
        strict=True,
    ):
        fields_by_component[component] = {
            name: value
            for name, value in zip(field_names, values, strict=True)
        }
        provenance_by_component[component] = provenance

    return DriverPreviousStepStatePacket(
        tstep=int(state.tstep),
        fields_by_component=fields_by_component,
        provenance_by_component=provenance_by_component,
    )
