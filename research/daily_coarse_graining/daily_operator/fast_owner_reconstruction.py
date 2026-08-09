"""Contract-bound reconstruction of the complete successor fast boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax.numpy as jnp

from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    ReconstructedFastDayTarget,
    _compiled_inflate_pft_axes,
    _compiled_select_pft_axes,
)

from .process_derivations import (
    NativeForcingFieldLayout,
    derive_end_of_day_driver_fields,
    derive_native_daily_deterministic_fields,
)
from .types import NativeForcingInput

ROLE_EXISTING = "existing_process_endpoint"
ROLE_NEURAL = "independent_neural_endpoint"
ROLE_DERIVATION = "deterministic_source_exact_derivation"
ROLE_MIRROR = "mirror_of_canonical_owner"
ROLE_CARRY = "true_guarded_or_static_carry"
EXPECTED_CONTRACT_SHA256 = "4db5f828442212c58903096c36015f9d769c19aeaaa4eb9a4326fbea2fdfc4c9"
DEFAULT_UNDEF_SECHIBA = 1.0e20
ROUGHHEIGHT_PFT_KEY = "diffuco_previous_step_state.roughheight_pft"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "successor_fast_owner_reconstruction_v1.json"
)

_DERIVATION_KEYS = frozenset(
    {
        "driver_previous_step_state.albedo_nir",
        "driver_previous_step_state.albedo_vis",
        "driver_previous_step_state.old_eair",
        "driver_previous_step_state.old_qair",
        "driver_previous_step_state.old_zlev",
        "driver_previous_step_state.peqAcoef",
        "driver_previous_step_state.peqBcoef",
        "driver_previous_step_state.petAcoef",
        "driver_previous_step_state.petBcoef",
        "driver_previous_step_state.rau_old",
        "daily_interface.flood_root_radia",
        "daily_interface.precip_daily",
        "daily_interface.snowfall_daily",
        "daily_interface.t2m_daily",
        "daily_interface.t2m_max_daily",
        "daily_interface.t2m_min_daily",
        "daily_interface.tmc_topgrass_daily",
        "ok_leak.deepC_peat",
        "final_diagnostics.t2mdiag",
    }
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FastOwnerLeaf:
    key: str
    role: str
    canonical_owner: str
    shape: tuple[int, ...]
    start: int
    stop: int
    component: str | None = None
    family: str = ""
    path: tuple[str, ...] = ()
    axis_names: tuple[str, ...] = ()
    pft_axis: int | None = None
    output_key: str | None = None
    defined_output_key: str | None = None
    defined_policy: str = ""
    scientific_constraint: str = ""
    proposed_process_domain: str | None = None
    loss_group: str | None = None
    loss_mask_policy: str = ""
    value_representation: str | None = None


@dataclass(frozen=True)
class FastOwnerReconstructionContract:
    schema_version: str
    contract_sha256: str
    markov_contract_sha256: str
    leaves: tuple[FastOwnerLeaf, ...]

    @property
    def leaves_by_key(self) -> Mapping[str, FastOwnerLeaf]:
        return {leaf.key: leaf for leaf in self.leaves}


def load_fast_owner_reconstruction_contract(
    path: Path = DEFAULT_CONTRACT_PATH,
    *,
    markov_contract: DailyMarkovContract | None = None,
) -> FastOwnerReconstructionContract:
    """Load and identity-check the frozen 160-leaf owner contract."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    observed = _canonical_sha256(payload)
    if payload.get("contract_sha256") != observed:
        raise ValueError("fast-owner reconstruction contract SHA256 drift")
    if observed != EXPECTED_CONTRACT_SHA256:
        raise ValueError("fast-owner reconstruction contract is not the accepted v1")
    if payload.get("schema_version") != "successor_fast_owner_reconstruction_v1":
        raise ValueError("fast-owner reconstruction schema version drift")

    records = payload.get("leaves", ())
    leaves = tuple(
        FastOwnerLeaf(
            key=str(record["key"]),
            role=str(record["role"]),
            canonical_owner=str(record["canonical_owner"]),
            shape=tuple(int(value) for value in record["shape"]),
            start=int(record["start"]),
            stop=int(record["stop"]),
            component=record.get("component"),
            family=str(record.get("family", "")),
            path=tuple(str(value) for value in record.get("path", ())),
            axis_names=tuple(str(value) for value in record.get("axis_names", ())),
            pft_axis=(None if record.get("pft_axis") is None else int(record["pft_axis"])),
            output_key=record.get("output_key"),
            defined_output_key=record.get("defined_output_key"),
            defined_policy=str(record.get("defined_policy", "")),
            scientific_constraint=str(record.get("scientific_constraint", "")),
            proposed_process_domain=record.get("proposed_process_domain"),
            loss_group=record.get("loss_group"),
            loss_mask_policy=str(record.get("loss_mask_policy", "")),
            value_representation=record.get("value_representation"),
        )
        for record in records
    )
    if len(leaves) != 160 or len({leaf.key for leaf in leaves}) != 160:
        raise ValueError("fast-owner reconstruction contract must own 160 unique leaves")
    if max((leaf.stop for leaf in leaves), default=0) != 2855:
        raise ValueError("fast-owner reconstruction target width drift")
    if {leaf.key for leaf in leaves if leaf.role == ROLE_DERIVATION} != _DERIVATION_KEYS:
        raise ValueError("fast-owner deterministic derivation inventory drift")
    by_key = {leaf.key: leaf for leaf in leaves}
    for leaf in leaves:
        if leaf.role == ROLE_MIRROR:
            owner = by_key.get(leaf.canonical_owner)
            if owner is None or owner.role == ROLE_MIRROR:
                raise ValueError(f"mirror {leaf.key} does not have a direct canonical owner")

    markov_sha256 = str(payload["dataset"]["markov_contract_sha256"])
    if markov_contract is not None:
        if markov_contract.sha256 != markov_sha256:
            raise ValueError("fast-owner and Markov contract identities drift")
        markov_leaves = {leaf.key: leaf for leaf in markov_contract.fast_day_target_leaves}
        if set(markov_leaves) != set(by_key):
            raise ValueError("fast-owner and Markov leaf inventories drift")
        for key, leaf in by_key.items():
            source = markov_leaves[key]
            if leaf.shape != tuple(source.shape) or leaf.start != int(source.start) or leaf.stop != int(source.stop):
                raise ValueError(f"fast-owner and Markov slice drift for {key}")
    return FastOwnerReconstructionContract(
        schema_version=str(payload["schema_version"]),
        contract_sha256=observed,
        markov_contract_sha256=markov_sha256,
        leaves=leaves,
    )


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _copy_mapping(child) if isinstance(child, Mapping) else child for name, child in value.items()}


def _get_path(root: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = root
    for name in path:
        value = value[name]
    return value


def _set_path(root: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = root
    for name in path[:-1]:
        target = target.setdefault(name, {})
    target[path[-1]] = value


@dataclass(frozen=True, eq=False)
class SuccessorFastOwnerMaterializer:
    """Materialize 160 named fast leaves after conservative state updates.

    ``named_outputs`` are physical, denormalized endpoint values.  Any future
    normalized delta decoder remains upstream and is shared by both arms.
    """

    markov_contract: DailyMarkovContract
    owner_contract: FastOwnerReconstructionContract
    forcing_layout: NativeForcingFieldLayout
    forcing_indices: Mapping[str, int]
    undef_sechiba: float = DEFAULT_UNDEF_SECHIBA

    def __post_init__(self) -> None:
        if self.markov_contract.sha256 != self.owner_contract.markov_contract_sha256:
            raise ValueError("materializer contracts have different Markov identities")
        required_forcing = {"Tair", "PSurf", "Qair"}
        if not required_forcing <= set(self.forcing_indices):
            raise ValueError("materializer forcing indices are incomplete")

    @property
    def _markov_leaves(self) -> Mapping[str, Any]:
        return {leaf.key: leaf for leaf in self.markov_contract.fast_day_target_leaves}

    def _apply_static_defined_policy(self, leaf: FastOwnerLeaf, spec: Any, value: Any) -> Any:
        """Restore source-static undefined slots omitted from neural endpoints."""

        compact = jnp.asarray(value, dtype=jnp.dtype(spec.dtype))
        if leaf.key != ROUGHHEIGHT_PFT_KEY:
            return compact
        if leaf.defined_policy != "frozen_contract_static_mask" or leaf.pft_axis is None:
            raise ValueError("roughheight_pft static defined policy drift")
        if len(spec.selected_pft_indices) != compact.shape[int(leaf.pft_axis)]:
            raise ValueError("roughheight_pft compact PFT identity drift")
        for compact_index, source_index in enumerate(spec.selected_pft_indices):
            if int(source_index) != 0:
                continue
            selected = [slice(None)] * compact.ndim
            selected[int(leaf.pft_axis)] = compact_index
            compact = compact.at[tuple(selected)].set(jnp.asarray(jnp.nan, dtype=compact.dtype))
        return compact

    def materialize(
        self,
        *,
        fields_by_component: Mapping[str, Mapping[str, Any]],
        daily_fields: Mapping[str, Any],
        ok_leak_updates: Mapping[str, Any],
        final_diagnostics: Mapping[str, Any],
        named_outputs: Mapping[str, Any],
        forcing: NativeForcingInput,
        measurement_height: Any,
    ) -> ReconstructedFastDayTarget:
        fields = _copy_mapping(fields_by_component)
        groups: dict[str, dict[str, Any]] = {
            "daily_interface": _copy_mapping(daily_fields),
            "ok_leak": _copy_mapping(ok_leak_updates),
            "final_diagnostics": _copy_mapping(final_diagnostics),
        }
        markov = self._markov_leaves

        def root(spec: Any) -> dict[str, Any]:
            if spec.component is not None:
                return fields.setdefault(spec.component, {})
            return groups[spec.family]

        def get(key: str) -> Any:
            spec = markov[key]
            return _get_path(root(spec), spec.path)

        def set_full(key: str, value: Any) -> None:
            spec = markov[key]
            array = jnp.asarray(value, dtype=jnp.dtype(spec.dtype))
            if tuple(array.shape) != tuple(spec.full_shape):
                raise ValueError(f"full owner shape drift for {key}: {array.shape} != {spec.full_shape}")
            _set_path(root(spec), spec.path, array)

        for leaf in self.owner_contract.leaves:
            if leaf.role != ROLE_EXISTING:
                continue
            spec = markov[leaf.key]
            current = jnp.asarray(get(leaf.key), dtype=jnp.dtype(spec.dtype))
            if tuple(current.shape) == tuple(spec.full_shape):
                continue
            if tuple(current.shape) != leaf.shape:
                raise ValueError(f"existing process endpoint shape drift for {leaf.key}")
            set_full(leaf.key, _compiled_inflate_pft_axes(current, spec))

        for leaf in self.owner_contract.leaves:
            if leaf.role != ROLE_NEURAL:
                continue
            if leaf.output_key not in named_outputs:
                raise ValueError(f"missing successor endpoint {leaf.output_key!r}")
            spec = markov[leaf.key]
            compact = jnp.asarray(named_outputs[leaf.output_key], dtype=jnp.dtype(spec.dtype))
            if tuple(compact.shape) != leaf.shape:
                raise ValueError(f"successor endpoint shape drift for {leaf.key}: {compact.shape} != {leaf.shape}")
            compact = self._apply_static_defined_policy(leaf, spec, compact)
            if leaf.defined_output_key is not None:
                if leaf.defined_output_key not in named_outputs:
                    raise ValueError(f"missing successor defined logits {leaf.defined_output_key!r}")
                logits = jnp.asarray(named_outputs[leaf.defined_output_key])
                if tuple(logits.shape) != leaf.shape:
                    raise ValueError(f"defined-logit shape drift for {leaf.key}")
                compact = jnp.where(
                    logits >= 0.0,
                    compact,
                    jnp.asarray(self.undef_sechiba, dtype=compact.dtype),
                )
            base = get(leaf.key)
            set_full(
                leaf.key,
                _compiled_inflate_pft_axes(compact, spec, base=base),
            )

        albedo = jnp.asarray(get("driver_previous_step_state.albedo"))
        set_full("driver_previous_step_state.albedo_vis", albedo[:, 0])
        set_full("driver_previous_step_state.albedo_nir", albedo[:, 1])
        driver = derive_end_of_day_driver_fields(
            forcing,
            self.forcing_indices,
            scalar_measurement_height=measurement_height,
        )
        for name, value in driver._asdict().items():
            set_full(f"driver_previous_step_state.{name}", value)
        for name in ("peqAcoef", "petAcoef"):
            key = f"driver_previous_step_state.{name}"
            set_full(key, jnp.zeros_like(get(key)))
        set_full(
            "driver_previous_step_state.peqBcoef",
            get("driver_previous_step_state.old_qair"),
        )
        set_full(
            "driver_previous_step_state.petBcoef",
            get("driver_previous_step_state.old_eair"),
        )

        deterministic = derive_native_daily_deterministic_fields(
            forcing,
            self.forcing_layout,
            model_interval_seconds=float(self.markov_contract.native_forcing.model_interval_seconds),
        )
        for name in (
            "precip_daily",
            "snowfall_daily",
            "t2m_daily",
            "t2m_max_daily",
            "t2m_min_daily",
        ):
            set_full(f"daily_interface.{name}", getattr(deterministic, name))
        set_full("final_diagnostics.t2mdiag", deterministic.t2mdiag)
        set_full(
            "daily_interface.flood_root_radia",
            jnp.zeros_like(get("daily_interface.flood_root_radia")),
        )
        set_full(
            "daily_interface.tmc_topgrass_daily",
            jnp.zeros_like(get("daily_interface.tmc_topgrass_daily")),
        )

        carbon_32l = jnp.asarray(get("ok_leak.carbon_32l"))
        deep_carbon = jnp.transpose(jnp.sum(carbon_32l, axis=1), (0, 2, 1))
        set_full("ok_leak.deepC_peat", deep_carbon)

        for leaf in self.owner_contract.leaves:
            if leaf.role == ROLE_MIRROR:
                set_full(leaf.key, get(leaf.canonical_owner))

        for leaf in self.owner_contract.leaves:
            spec = markov[leaf.key]
            compact = _compiled_select_pft_axes(get(leaf.key), spec)
            if tuple(compact.shape) != leaf.shape:
                raise ValueError(f"materialized compact shape drift for {leaf.key}: {compact.shape} != {leaf.shape}")
        return ReconstructedFastDayTarget(
            fields_by_component=fields,
            daily_fields=groups["daily_interface"],
            ok_leak_updates=groups["ok_leak"],
            final_diagnostics=groups["final_diagnostics"],
        )


def build_successor_fast_owner_materializer(
    markov_contract: DailyMarkovContract,
    *,
    forcing_layout: NativeForcingFieldLayout,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> SuccessorFastOwnerMaterializer:
    """Bind the accepted owner schema to one accepted Markov contract."""

    width = 1
    for extent in markov_contract.native_forcing.field_shape:
        width *= int(extent)
    if width != 1:
        raise ValueError("successor scalar-forcing materializer requires one landpoint")
    forcing_indices = {name: index * width for index, name in enumerate(markov_contract.native_forcing.fields)}
    return SuccessorFastOwnerMaterializer(
        markov_contract=markov_contract,
        owner_contract=load_fast_owner_reconstruction_contract(
            contract_path,
            markov_contract=markov_contract,
        ),
        forcing_layout=forcing_layout,
        forcing_indices=forcing_indices,
    )
