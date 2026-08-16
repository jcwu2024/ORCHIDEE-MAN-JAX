"""Frozen ownership and axis ledger for the approved strong baseline."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fast_owner_reconstruction import FastOwnerReconstructionContract
from .neural_architecture import NeuralOutputSchema, OutputFieldSpec
from .training_data import DailyProcessTrainingAdapter

SCHEMA_VERSION = "strong_baseline_axis_aligned_process_expert_v1"
FROZEN_CONTRACT_SHA256 = "baf5791f0786f22e05e8db9115eeab9a77687e19250d4d9efcb5181d6a96351d"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "strong_baseline_axis_aligned_process_expert_v1.json"
)


def contract_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StrongBaselineContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def expert_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.payload["experts"])

    @property
    def label_ownership(self) -> Mapping[str, str]:
        return {str(label): str(expert) for label, expert in self.payload["label_ownership"].items()}


@dataclass(frozen=True)
class StrongBaselineFieldOwnership:
    name: str
    label: str
    expert: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    local_axes: tuple[str, ...]
    pft_axis: int | None
    value_representation: str
    day_start_owner: str | None

    @property
    def scalar_count(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))


def _require_sha256(value: Any, label: str) -> None:
    text = str(value)
    if len(text) != 64 or text.lower() != text or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"strong-baseline contract requires lowercase SHA256 {label}")


def _axis_aliases(payload: Mapping[str, Any]) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for canonical, record in payload["axis_registry"].items():
        canonical = str(canonical)
        aliases = tuple(str(value) for value in record["aliases"])
        if canonical not in aliases:
            raise ValueError(f"strong-baseline axis {canonical!r} must include its own alias")
        if record.get("mode") not in {
            "singleton_batch",
            "shared_permutation_equivariant",
            "ordered_factorized_axial",
            "categorical_shared",
        }:
            raise ValueError(f"strong-baseline axis {canonical!r} has an unknown mode")
        for alias in aliases:
            if alias in result:
                raise ValueError(f"strong-baseline axis alias is duplicated: {alias}")
            result[alias] = canonical
    return result


def _validate_experts(payload: Mapping[str, Any]) -> None:
    experts = payload["experts"]
    expected_counts = payload["expected_output_schema"]["expert_counts"]
    if set(experts) != set(expected_counts) or len(experts) != 7:
        raise ValueError("strong-baseline contract requires the frozen seven experts")

    source_families = tuple(str(value) for value in payload["source_families"])
    if len(source_families) != len(set(source_families)):
        raise ValueError("strong-baseline source families must be unique")
    used_sources: set[str] = set()
    for name, record in experts.items():
        admitted = tuple(str(value) for value in record["admitted_sources"])
        if not admitted or len(admitted) != len(set(admitted)):
            raise ValueError(f"strong-baseline expert {name!r} has invalid admitted sources")
        unknown = sorted(set(admitted) - set(source_families))
        if unknown:
            raise ValueError(f"strong-baseline expert {name!r} has unknown sources: {unknown}")
        used_sources.update(admitted)
    if used_sources != set(source_families):
        raise ValueError("strong-baseline source registry contains unused families")

    member_groups = payload["source_members"]
    member_families = set()
    for group_name, group in member_groups.items():
        if group_name == "structural":
            values = tuple(str(value) for value in group)
            if len(values) != len(set(values)):
                raise ValueError("strong-baseline structural sources overlap")
            member_families.update(values)
            continue
        flattened = tuple(str(value) for members in group.values() for value in members)
        if len(flattened) != len(set(flattened)):
            raise ValueError(f"strong-baseline {group_name} members overlap")
        member_families.update(str(value) for value in group)
    if member_families != set(source_families):
        raise ValueError("strong-baseline source members do not cover the family registry")

    ownership = payload["label_ownership"]
    if len(ownership) != 48 or set(ownership.values()) - set(experts):
        raise ValueError("strong-baseline label ownership must cover 48 labels and seven experts")


def _validate_graph(payload: Mapping[str, Any]) -> None:
    graph = payload["graph_messages"]
    experts = set(payload["experts"])
    edges = tuple(tuple(str(value) for value in edge) for edge in graph["directed_edges"])
    if graph.get("rounds") != 2 or len(edges) != 23 or len(set(edges)) != len(edges):
        raise ValueError("strong-baseline graph must contain two rounds and 23 unique edges")
    for edge in edges:
        if len(edge) != 2 or set(edge) - experts or edge[0] == edge[1]:
            raise ValueError(f"strong-baseline graph edge is invalid: {edge}")
    gate = graph["gate"]
    if gate.get("range") != [0.0, 1.0]:
        raise ValueError("strong-baseline message gate must be bounded to [0, 1]")
    if float(gate.get("initial_value")) != 0.01 or float(gate.get("qualification_value")) != 0.0:
        raise ValueError("strong-baseline message gate initialization drift")


def _validate_source_axis_bindings(payload: Mapping[str, Any]) -> None:
    bindings = payload["source_axis_bindings"]
    if set(bindings) != {"static_fields", "process_static_fields"}:
        raise ValueError("strong-baseline source-axis binding groups drift")
    axis_names = set(payload["axis_registry"])
    valid_modes = {"direct", "full_pft", "compact_pft", "interval_boundaries"}
    members = payload["source_members"]
    for group_name, records in bindings.items():
        admitted_fields = {str(field) for fields in members[group_name].values() for field in fields}
        if set(records) - admitted_fields:
            raise ValueError(f"strong-baseline {group_name} axis binding names drift")
        for field, field_bindings in records.items():
            dimensions = []
            for binding in field_bindings:
                dimension = int(binding["dimension"])
                axis = str(binding["axis"])
                mode = str(binding["mode"])
                if dimension < 0 or axis not in axis_names or mode not in valid_modes:
                    raise ValueError(f"strong-baseline source-axis binding is invalid: {field} {binding}")
                if mode in {"full_pft", "compact_pft"} and axis != "pft":
                    raise ValueError(f"strong-baseline PFT binding axis drift: {field}")
                dimensions.append(dimension)
            if not dimensions or len(dimensions) != len(set(dimensions)):
                raise ValueError(f"strong-baseline source-axis dimensions overlap: {field}")


def load_strong_baseline_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> StrongBaselineContract:
    """Load and fail closed on approved architecture or ownership drift."""

    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed:
        raise ValueError("strong-baseline contract SHA256 drift")
    if observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("strong-baseline contract is not the approved v1")

    identity = payload["data_identity"]
    for name in (
        "dataset_manifest_sha256",
        "acceptance_report_sha256",
        "markov_contract_sha256",
        "typed_contract_sha256",
        "fast_owner_contract_sha256",
    ):
        _require_sha256(identity[name], name)
    if identity.get("release_id") != "pft14-daily-teacher-v6":
        raise ValueError("strong-baseline contract is not bound to Teacher v6")
    if any(
        bool(identity.get(name))
        for name in ("historical_v5_allowed", "teacher_regeneration_allowed", "sealed_test_allowed")
    ):
        raise ValueError("strong-baseline contract opens a forbidden data source")

    expected = payload["expected_output_schema"]
    if (expected.get("field_count"), expected.get("label_count"), expected.get("scalar_count")) != (
        181,
        48,
        39371,
    ):
        raise ValueError("strong-baseline expected output schema drift")
    precision = payload["precision"]
    if (
        precision.get("neural_parameters") != "float32"
        or precision.get("neural_activations") != "float32"
        or precision.get("optimizer_state") != "float32"
        or any(
            precision.get(name) != "float64"
            for name in (
                "head_raw_to_constrained_cast",
                "output_normalization",
                "conservative_update",
                "retained_tail",
                "scientific_loss",
            )
        )
        or precision.get("reduced_precision_allowed") is not False
    ):
        raise ValueError("strong-baseline precision boundary drift")
    capacity = payload["capacity"]
    if (
        capacity.get("expert_width") != 384
        or capacity.get("attention_heads") != 8
        or capacity.get("expert_blocks") != 3
        or capacity.get("axial_blocks") != 1
        or capacity.get("parameter_count_minimum") != 35_000_000
        or capacity.get("parameter_count_target_maximum") != 50_000_000
        or capacity.get("parameter_count_hard_maximum") != 60_000_000
        or capacity.get("dense_all_output_to_all_input_attention_allowed") is not False
    ):
        raise ValueError("strong-baseline capacity envelope drift")
    if payload["loss_weighting"].get("process_fidelity") != ("equal_experts_then_equal_labels_within_expert"):
        raise ValueError("strong-baseline expert loss weighting drift")

    _validate_experts(payload)
    _axis_aliases(payload)
    _validate_source_axis_bindings(payload)
    _validate_graph(payload)
    return StrongBaselineContract(payload=payload, source_path=source_path, sha256=observed)


def _state_axes(
    field: OutputFieldSpec,
    adapter: DailyProcessTrainingAdapter,
) -> tuple[str, ...] | None:
    if field.name.startswith("inventory.water."):
        component = "hydrol_previous_step_state"
        remainder = field.name.removeprefix("inventory.water.")
        source_name = remainder.split(".", 1)[0]
    elif field.name.startswith("inventory.carbon."):
        component = "slowproc_stomate_previous_step_state"
        remainder = field.name.removeprefix("inventory.carbon.")
        source_name = remainder.split(".", 1)[0]
    elif field.name.startswith("thermal."):
        _prefix, component, source_name, suffix = field.name.split(".")
        if suffix != "normalized_tendency":
            raise ValueError(f"strong-baseline thermal field suffix drift: {field.name}")
    else:
        return None

    key = f"{component}.{source_name}"
    matches = tuple(leaf for leaf in adapter.contract.state_leaves if leaf.key == key)
    if len(matches) != 1:
        raise ValueError(f"strong-baseline field lacks one state-axis owner: {field.name}")
    return tuple(str(value) for value in matches[0].axis_names)


def _raw_field_axes(
    field: OutputFieldSpec,
    *,
    adapter: DailyProcessTrainingAdapter,
    owner_contract: FastOwnerReconstructionContract,
    contract: StrongBaselineContract,
) -> tuple[str, ...]:
    overrides = contract.payload["field_axis_overrides"]
    if field.name in overrides:
        return tuple(str(value) for value in overrides[field.name])

    typed_axes = {
        f"typed.{record.path}": tuple(str(value) for value in record.axes[1:])
        for record in adapter.typed_contract.fields
    }
    if field.name in typed_axes:
        return typed_axes[field.name]

    owner_axes: dict[str, tuple[str, ...]] = {}
    for leaf in owner_contract.leaves:
        if leaf.output_key is not None:
            owner_axes[str(leaf.output_key)] = tuple(str(value) for value in leaf.axis_names)
        if leaf.defined_output_key is not None:
            owner_axes[str(leaf.defined_output_key)] = tuple(str(value) for value in leaf.axis_names)
    if field.name in owner_axes:
        return owner_axes[field.name]

    axes = _state_axes(field, adapter)
    if axes is not None:
        return axes

    raise ValueError(f"strong-baseline field has no axis owner: {field.name}")


def resolve_strong_baseline_field_ledger(
    contract: StrongBaselineContract,
    *,
    schema: NeuralOutputSchema,
    adapter: DailyProcessTrainingAdapter,
    owner_contract: FastOwnerReconstructionContract,
) -> tuple[StrongBaselineFieldOwnership, ...]:
    """Resolve all 181 frozen outputs to one expert and canonical axis tuple."""

    identity = contract.payload["data_identity"]
    if adapter.contract.sha256 != identity["markov_contract_sha256"]:
        raise ValueError("strong-baseline and Markov contract identities drift")
    if adapter.typed_contract.sha256 != identity["typed_contract_sha256"]:
        raise ValueError("strong-baseline and typed contract identities drift")
    if owner_contract.contract_sha256 != identity["fast_owner_contract_sha256"]:
        raise ValueError("strong-baseline and fast-owner contract identities drift")

    ownership = contract.label_ownership
    if set(schema.labels) != set(ownership):
        missing = sorted(set(schema.labels) - set(ownership))
        extra = sorted(set(ownership) - set(schema.labels))
        raise ValueError(f"strong-baseline label ownership drift; missing={missing}, extra={extra}")

    aliases = _axis_aliases(contract.payload)
    axis_registry = contract.payload["axis_registry"]
    records: list[StrongBaselineFieldOwnership] = []
    for field in schema.fields:
        label = str(schema.labels[field.label_index])
        raw_axes = _raw_field_axes(
            field,
            adapter=adapter,
            owner_contract=owner_contract,
            contract=contract,
        )
        if len(raw_axes) != len(field.shape):
            raise ValueError(f"strong-baseline field axis rank drift for {field.name}: {raw_axes} versus {field.shape}")
        try:
            axes = tuple(aliases[name] for name in raw_axes)
        except KeyError as error:
            raise ValueError(f"strong-baseline field {field.name} uses an unknown axis {error.args[0]!r}") from error
        pft_positions = tuple(index for index, name in enumerate(axes) if name == "pft")
        pft_axis = contract.payload["field_pft_axis_overrides"].get(
            field.name,
            field.pft_axis,
        )
        if pft_axis is None:
            if pft_positions:
                raise ValueError(f"strong-baseline field lacks declared PFT semantics: {field.name}")
        elif pft_positions != (int(pft_axis),):
            raise ValueError(f"strong-baseline field PFT axis drift: {field.name}")
        local_axes = tuple(
            name
            for name, extent in zip(axes, field.shape, strict=True)
            if int(extent) > 1 and bool(axis_registry[name]["local_token_required"])
        )
        records.append(
            StrongBaselineFieldOwnership(
                name=field.name,
                label=label,
                expert=ownership[label],
                shape=tuple(int(value) for value in field.shape),
                axes=axes,
                local_axes=local_axes,
                pft_axis=None if pft_axis is None else int(pft_axis),
                value_representation=field.value_representation,
                day_start_owner=field.day_start_owner,
            )
        )

    if len(records) != len({record.name for record in records}):
        raise ValueError("strong-baseline field ledger contains duplicate outputs")
    expected = contract.payload["expected_output_schema"]
    if len(records) != int(expected["field_count"]):
        raise ValueError("strong-baseline field ledger count drift")
    if sum(record.scalar_count for record in records) != int(expected["scalar_count"]):
        raise ValueError("strong-baseline scalar output count drift")

    expected_experts = expected["expert_counts"]
    field_counts = Counter(record.expert for record in records)
    scalar_counts = Counter()
    labels_by_expert: dict[str, set[str]] = {name: set() for name in contract.expert_names}
    for record in records:
        scalar_counts[record.expert] += record.scalar_count
        labels_by_expert[record.expert].add(record.label)
    for expert, values in expected_experts.items():
        observed = (
            field_counts[expert],
            len(labels_by_expert[expert]),
            scalar_counts[expert],
        )
        declared = (int(values["fields"]), int(values["labels"]), int(values["scalars"]))
        if observed != declared:
            raise ValueError(
                f"strong-baseline expert ownership drift for {expert}: observed={observed}, declared={declared}"
            )
    return tuple(records)


def ledger_by_name(
    ledger: tuple[StrongBaselineFieldOwnership, ...],
) -> Mapping[str, StrongBaselineFieldOwnership]:
    result = {record.name: record for record in ledger}
    if len(result) != len(ledger):
        raise ValueError("strong-baseline ledger names are not unique")
    return result


def canonical_axis_aliases(contract: StrongBaselineContract) -> Mapping[str, str]:
    return _axis_aliases(contract.payload)


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "StrongBaselineContract",
    "StrongBaselineFieldOwnership",
    "canonical_axis_aliases",
    "contract_sha256",
    "ledger_by_name",
    "load_strong_baseline_contract",
    "resolve_strong_baseline_field_ledger",
]
