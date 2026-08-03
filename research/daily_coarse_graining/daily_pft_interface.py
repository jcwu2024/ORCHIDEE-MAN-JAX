"""Validated PFT-axis boundary for the future true-daily neural operator.

This module owns only interface assembly and validation. Stable PFT IDs remain
adapter metadata for remapping and provenance; they are deliberately excluded
from the arrays returned to a neural model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "daily_neural_pft_interface_v1"
OWNERSHIP_CLASSES = frozenset(
    {
        "retained_exact",
        "explicit_parameterized_fast_process_factor",
        "learned_conditional",
    }
)
NETWORK_INPUT_NAMES = (
    "pft_state",
    "pft_parameters",
    "pft_traits",
    "pft_fraction",
    "active_pft_mask",
)
IDENTITY_INPUT_NAMES = frozenset(
    {"landpoint_id", "pft_id", "fortran_pft_id", "mtc_id_embedding", "pft14_embedding"}
)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "manifests"
    / "coarse_graining"
    / "daily_neural_pft_interface_v1.json"
)


@dataclass(frozen=True)
class InterfaceField:
    name: str
    dtype: str
    shape: tuple[str, ...]
    axis_owner: str


@dataclass(frozen=True)
class ParameterChannel:
    index: int
    name: str
    dtype: str
    units: str
    axis_owner: str
    ownership_class: str
    source_owner: str
    jax_source: str
    consumers: tuple[str, ...]
    paper_calibrated: bool
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TraitChannel:
    index: int
    name: str
    dtype: str
    units: str
    axis_owner: str
    source_owner: str
    consumer: str


@dataclass(frozen=True)
class DailyPFTInterfaceContract:
    schema_version: str
    status: str
    pft_axis_name: str
    pft_axis_position: int
    minimum_n_pft: int
    network_inputs: tuple[InterfaceField, ...]
    forbidden_network_inputs: tuple[str, ...]
    parameter_channels: tuple[ParameterChannel, ...]
    trait_channels: tuple[TraitChannel, ...]
    source_path: Path

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.parameter_channels)

    @property
    def trait_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.trait_channels)


@dataclass(frozen=True)
class PFTAxisInputs:
    """One validated variable-length PFT batch with identity kept as metadata."""

    pft_state: np.ndarray
    pft_parameters: np.ndarray
    pft_traits: np.ndarray
    pft_fraction: np.ndarray
    active_pft_mask: np.ndarray
    pft_ids: tuple[str, ...]

    @property
    def n_pft(self) -> int:
        return int(self.pft_state.shape[0])

    def network_arrays(self) -> dict[str, np.ndarray]:
        """Return only legal numerical model inputs, never identity metadata."""

        return {
            "pft_state": self.pft_state,
            "pft_parameters": self.pft_parameters,
            "pft_traits": self.pft_traits,
            "pft_fraction": self.pft_fraction,
            "active_pft_mask": self.active_pft_mask,
        }

    def permuted(self, order: Sequence[int]) -> PFTAxisInputs:
        permutation = np.asarray(order, dtype=np.int64)
        if permutation.shape != (self.n_pft,) or set(permutation.tolist()) != set(range(self.n_pft)):
            raise ValueError("order must be a complete PFT-axis permutation")
        return PFTAxisInputs(
            pft_state=self.pft_state[permutation],
            pft_parameters=self.pft_parameters[permutation],
            pft_traits=self.pft_traits[permutation],
            pft_fraction=self.pft_fraction[permutation],
            active_pft_mask=self.active_pft_mask[permutation],
            pft_ids=tuple(self.pft_ids[index] for index in permutation),
        )


def _required_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _required_text(mapping: Mapping[str, Any], key: str, *, label: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"{label} requires nonempty {key}")
    return value


def _validate_contiguous_channels(channels: Sequence[object], *, label: str) -> None:
    indices = tuple(int(getattr(channel, "index")) for channel in channels)
    names = tuple(str(getattr(channel, "name")) for channel in channels)
    if indices != tuple(range(len(channels))):
        raise ValueError(f"{label} channel indices must be contiguous and ordered")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} channel names must be unique")
    if any(name in IDENTITY_INPUT_NAMES or "embedding" in name for name in names):
        raise ValueError(f"{label} channels cannot encode PFT or landpoint identity")


def load_daily_pft_interface_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> DailyPFTInterfaceContract:
    """Load and fail-closed validate the frozen v1 neural PFT interface."""

    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")

    pft_axis = _required_mapping(payload.get("pft_axis"), label="pft_axis")
    if pft_axis.get("name") != "n_pft" or int(pft_axis.get("position", -1)) != 0:
        raise ValueError("the neural PFT axis must be explicit at position zero")
    minimum_n_pft = int(pft_axis.get("minimum_size", 0))
    if minimum_n_pft < 1 or pft_axis.get("weights_shared_across_axis") is not True:
        raise ValueError("the PFT axis requires positive size and shared weights")

    network_inputs = tuple(
        InterfaceField(
            name=_required_text(raw, "name", label="network input"),
            dtype=_required_text(raw, "dtype", label="network input"),
            shape=tuple(str(item) for item in raw.get("shape", ())),
            axis_owner=_required_text(raw, "axis_owner", label="network input"),
        )
        for raw_value in payload.get("network_inputs", ())
        for raw in (_required_mapping(raw_value, label="network input"),)
    )
    if tuple(field.name for field in network_inputs) != NETWORK_INPUT_NAMES:
        raise ValueError("network inputs must exactly match the frozen PFT interface")
    for field in network_inputs:
        if not field.shape or field.shape[0] != "n_pft":
            raise ValueError(f"network input {field.name!r} has no leading n_pft axis")
    forbidden = tuple(str(item) for item in payload.get("forbidden_network_inputs", ()))
    if not IDENTITY_INPUT_NAMES.issubset(forbidden):
        raise ValueError("all PFT and landpoint identity shortcuts must be forbidden")

    parameter_channels = []
    for raw_value in payload.get("parameter_channels", ()):
        raw = _required_mapping(raw_value, label="parameter channel")
        ownership_class = _required_text(raw, "ownership_class", label="parameter channel")
        if ownership_class not in OWNERSHIP_CLASSES:
            raise ValueError(f"unknown parameter ownership class {ownership_class!r}")
        consumers = tuple(str(item).strip() for item in raw.get("consumers", ()) if str(item).strip())
        if not consumers:
            raise ValueError("every parameter channel requires at least one consumer")
        if ownership_class == "retained_exact" and not str(raw.get("exact_formula_owner", "")).strip():
            raise ValueError("retained_exact parameters require exact_formula_owner")
        if ownership_class == "explicit_parameterized_fast_process_factor" and not str(
            raw.get("explicit_factor_rule", "")
        ).strip():
            raise ValueError("explicit fast-process parameters require explicit_factor_rule")
        perturbation = _required_mapping(raw.get("controlled_perturbation"), label="controlled_perturbation")
        if ownership_class == "learned_conditional" and perturbation.get("status") not in {
            "available_in_paper_sensitivity_design",
            "available_in_controlled_teacher_data",
        }:
            raise ValueError("learned_conditional parameters require controlled perturbation data")
        parameter_channels.append(
            ParameterChannel(
                index=int(raw["index"]),
                name=_required_text(raw, "name", label="parameter channel"),
                dtype=_required_text(raw, "dtype", label="parameter channel"),
                units=_required_text(raw, "units", label="parameter channel"),
                axis_owner=_required_text(raw, "axis_owner", label="parameter channel"),
                ownership_class=ownership_class,
                source_owner=_required_text(raw, "source_owner", label="parameter channel"),
                jax_source=_required_text(raw, "jax_source", label="parameter channel"),
                consumers=consumers,
                paper_calibrated=bool(raw.get("paper_calibrated", False)),
                metadata=dict(raw),
            )
        )
    _validate_contiguous_channels(parameter_channels, label="parameter")

    trait_channels = []
    for raw_value in payload.get("trait_channels", ()):
        raw = _required_mapping(raw_value, label="trait channel")
        trait_channels.append(
            TraitChannel(
                index=int(raw["index"]),
                name=_required_text(raw, "name", label="trait channel"),
                dtype=_required_text(raw, "dtype", label="trait channel"),
                units=_required_text(raw, "units", label="trait channel"),
                axis_owner=_required_text(raw, "axis_owner", label="trait channel"),
                source_owner=_required_text(raw, "source_owner", label="trait channel"),
                consumer=_required_text(raw, "consumer", label="trait channel"),
            )
        )
    _validate_contiguous_channels(trait_channels, label="trait")
    if not parameter_channels or not trait_channels:
        raise ValueError("parameter and trait channel registries cannot be empty")
    if any(channel.dtype != "float64" or channel.axis_owner != "pft" for channel in parameter_channels):
        raise ValueError("all parameter channels must be float64 and PFT-axis owned")
    if any(channel.dtype != "float64" or channel.axis_owner != "pft" for channel in trait_channels):
        raise ValueError("all trait channels must be float64 and PFT-axis owned")

    return DailyPFTInterfaceContract(
        schema_version=SCHEMA_VERSION,
        status=_required_text(payload, "status", label="contract"),
        pft_axis_name="n_pft",
        pft_axis_position=0,
        minimum_n_pft=minimum_n_pft,
        network_inputs=network_inputs,
        forbidden_network_inputs=forbidden,
        parameter_channels=tuple(parameter_channels),
        trait_channels=tuple(trait_channels),
        source_path=source_path,
    )


def validate_pft_axis_inputs(
    contract: DailyPFTInterfaceContract,
    *,
    pft_state: object,
    pft_parameters: object,
    pft_traits: object,
    pft_fraction: object,
    active_pft_mask: object,
    pft_ids: Sequence[str],
) -> PFTAxisInputs:
    """Validate shapes, precision, mask semantics, and metadata separation."""

    state = np.asarray(pft_state)
    parameters = np.asarray(pft_parameters)
    traits = np.asarray(pft_traits)
    fraction = np.asarray(pft_fraction)
    active = np.asarray(active_pft_mask)
    if state.ndim < 2 or state.shape[0] < contract.minimum_n_pft:
        raise ValueError("pft_state must have a legal leading n_pft axis")
    n_pft = int(state.shape[0])
    expected_shapes = {
        "pft_parameters": (n_pft, len(contract.parameter_channels)),
        "pft_traits": (n_pft, len(contract.trait_channels)),
        "pft_fraction": (n_pft,),
        "active_pft_mask": (n_pft,),
    }
    actual_shapes = {
        "pft_parameters": parameters.shape,
        "pft_traits": traits.shape,
        "pft_fraction": fraction.shape,
        "active_pft_mask": active.shape,
    }
    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            raise ValueError(f"{name} shape {actual_shapes[name]} does not match {expected}")
    for name, values in {
        "pft_state": state,
        "pft_parameters": parameters,
        "pft_traits": traits,
        "pft_fraction": fraction,
    }.items():
        if values.dtype != np.dtype(np.float64):
            raise TypeError(f"{name} must use float64")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
    if active.dtype != np.dtype(np.bool_):
        raise TypeError("active_pft_mask must use exact bool dtype")
    if np.any(fraction < 0.0) or np.any(fraction > 1.0):
        raise ValueError("pft_fraction must lie in [0, 1]")
    if np.any(fraction[~active] != 0.0):
        raise ValueError("inactive PFT slots must have exactly zero fraction")
    ids = tuple(str(item).strip() for item in pft_ids)
    if len(ids) != n_pft or any(not item for item in ids) or len(set(ids)) != n_pft:
        raise ValueError("pft_ids metadata must contain one unique stable ID per slot")
    return PFTAxisInputs(state, parameters, traits, fraction, active, ids)


def pack_named_pft_channels(
    contract: DailyPFTInterfaceContract,
    values: Mapping[str, object],
    *,
    channel_kind: str,
) -> np.ndarray:
    """Pack complete named channels without exposing positional semantics."""

    if channel_kind == "parameter":
        channels: Sequence[ParameterChannel | TraitChannel] = contract.parameter_channels
    elif channel_kind == "trait":
        channels = contract.trait_channels
    else:
        raise ValueError("channel_kind must be 'parameter' or 'trait'")
    expected = tuple(channel.name for channel in channels)
    missing = sorted(set(expected) - set(values))
    extra = sorted(set(values) - set(expected))
    if missing or extra:
        raise ValueError(
            f"{channel_kind} channels do not match the frozen contract; "
            f"missing={missing}, extra={extra}"
        )
    arrays = [np.asarray(values[name]) for name in expected]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1 or not arrays or arrays[0].ndim != 1:
        raise ValueError(f"every named {channel_kind} channel must have the same one-dimensional PFT axis")
    if any(array.dtype != np.dtype(np.float64) for array in arrays):
        raise TypeError(f"every named {channel_kind} channel must use float64")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"every named {channel_kind} channel must be finite")
    return np.stack(arrays, axis=1)


def apply_active_pft_mask(values: object, active_pft_mask: object) -> np.ndarray:
    """Zero inactive PFT outputs without assuming any particular slot identity."""

    array = np.asarray(values)
    active = np.asarray(active_pft_mask)
    if array.ndim < 1 or active.dtype != np.dtype(np.bool_) or active.shape != (array.shape[0],):
        raise ValueError("values and exact bool mask must share the leading PFT axis")
    broadcast = active.reshape((active.size,) + (1,) * (array.ndim - 1))
    return np.where(broadcast, array, np.zeros((), dtype=array.dtype))


def pft_fraction_weighted_sum(values: object, inputs: PFTAxisInputs) -> np.ndarray:
    """Aggregate masked per-PFT outputs using the declared fractions."""

    masked = apply_active_pft_mask(values, inputs.active_pft_mask)
    if masked.shape[0] != inputs.n_pft:
        raise ValueError("values do not match the validated PFT axis")
    weights = inputs.pft_fraction.reshape((inputs.n_pft,) + (1,) * (masked.ndim - 1))
    return np.sum(masked * weights, axis=0)
