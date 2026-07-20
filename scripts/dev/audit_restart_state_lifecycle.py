from __future__ import annotations

import argparse
import importlib
import ast
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import (  # noqa: E402
    STOMATE_DAY_END_WRITEBACK_FIELDS,
    STOMATE_HALF_HOUR_CARRY_FIELDS,
)
from jax_orchidee.stomate.reference import (  # noqa: E402
    StomateDailyAccumulatorState,
    StomateOkPcRestartGasState,
    StomateReadstartRemainderState,
    StomateRestartEntryState,
    StomateRestartSeasonState,
)
from jax_orchidee.stomate.restart_io import (  # noqa: E402
    STOMATE_FIXED_PATH_CARRY_FIELDS,
    stomate_writerestart_field_ledger,
)
from scripts.dev.scan_stomate_io_restart_fields import scan_stomate_io_restart_fields  # noqa: E402


DEFAULT_CONTRACT = ROOT / "docs" / "source_audits" / "restart_state_lifecycle.yaml"
DEFAULT_SOURCE = ROOT / "fortran_source" / "ORCHIDEE" / "src_stomate" / "stomate_io.f90"

RESTART_READER_TYPES = (
    StomateRestartEntryState,
    StomateRestartSeasonState,
    StomateDailyAccumulatorState,
    StomateOkPcRestartGasState,
    StomateReadstartRemainderState,
)


def reader_contract_fields() -> tuple[str, ...]:
    """Return the unique public fields exposed by current STOMATE readers."""

    fields: list[str] = []
    for state_type in RESTART_READER_TYPES:
        fields.extend(name for name in state_type._fields if name != "provenance")
    return tuple(dict.fromkeys(fields))


def lifecycle_contract_fields() -> tuple[str, ...]:
    """Return the union of restart-reader fields and explicit day-end packet fields."""

    return tuple(
        dict.fromkeys((*reader_contract_fields(), *STOMATE_DAY_END_WRITEBACK_FIELDS))
    )


def _axis_declarations(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    declarations: dict[str, dict[str, Any]] = {}
    for group, declaration in document["axis_groups"].items():
        for field in declaration["fields"]:
            if field in declarations:
                raise ValueError(f"duplicate axis declaration for {field}")
            declarations[field] = {
                "shape_group": group,
                "axes": list(declaration["axes"]),
            }
    return declarations


def _active_source_records(source: Path) -> list[dict[str, Any]]:
    """Scan actual restart calls while excluding wholly commented call lines."""

    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    records = scan_stomate_io_restart_fields(source)["records"]
    return [
        record
        for record in records
        if not lines[int(record["line"]) - 1].lstrip().startswith("!")
    ]


def _fortran_calls_by_field(source: Path) -> dict[str, dict[str, list[int]]]:
    calls: dict[str, dict[str, list[int]]] = {}
    for record in _active_source_records(source):
        by_call = calls.setdefault(
            str(record["name"]), {"restget_p": [], "restput_p": []}
        )
        by_call[str(record["call"])].append(int(record["line"]))
    return calls


def _source_lines(
    aliases: list[str],
    calls: dict[str, dict[str, list[int]]],
    call: str,
) -> list[int]:
    return sorted(
        {line for alias in aliases for line in calls.get(alias, {}).get(call, [])}
    )


def _resolve_owner(owner: Any, *, field: str, role: str):
    """Validate a dotted executable owner declared by the YAML contract."""

    if not isinstance(owner, str) or "." not in owner:
        raise ValueError(f"{field}: missing YAML {role} executable owner")
    module_name, attribute = owner.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        resolved = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"{field}: YAML {role} owner does not resolve: {owner}"
        ) from exc
    if not callable(resolved):
        raise ValueError(f"{field}: YAML {role} owner is not executable: {owner}")
    return owner, resolved


def _require_owner(owner: Any, *, field: str, role: str) -> str:
    return _resolve_owner(owner, field=field, role=role)[0]


def _mapping_literal_keys(
    owner: str, mapping_names: list[str], *, field: str
) -> set[str]:
    """Return literal keys actually written into named mappings by an owner AST."""

    _, resolved = _resolve_owner(owner, field=field, role="production owner")
    try:
        tree = ast.parse(inspect.getsource(resolved))
    except (OSError, TypeError, IndentationError) as exc:
        raise ValueError(
            f"{field}: cannot inspect production owner AST: {owner}"
        ) from exc
    names = set(mapping_names)
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (
            "$return" in names
            and isinstance(node, ast.Return)
            and isinstance(node.value, ast.Dict)
        ):
            keys.update(
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in names
                    and isinstance(value, ast.Dict)
                ):
                    keys.update(
                        key.value
                        for key in value.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    )
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in names
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    keys.add(target.slice.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in names
            and node.func.attr == "update"
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            keys.update(
                key.value
                for key in node.args[0].keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
    return keys


def _owner_constant_fields(
    owner: str,
    constants: list[str],
    mapping_names: list[str],
    *,
    field: str,
) -> set[str]:
    """Resolve only constants that the inspected owner AST actually references."""

    _, resolved = _resolve_owner(owner, field=field, role="production owner")
    tree = ast.parse(inspect.getsource(resolved))
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    mapped_constants: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id in mapping_names
            for target in targets
        ):
            continue
        mapped_constants.update(
            child.id for child in ast.walk(node.value) if isinstance(child, ast.Name)
        )
    module = importlib.import_module(resolved.__module__)
    fields: set[str] = set()
    for name in constants:
        if name not in referenced:
            raise ValueError(
                f"{field}: production owner AST does not reference mapping constant {name}"
            )
        if name not in mapped_constants:
            raise ValueError(
                f"{field}: production owner AST does not map constant {name} "
                f"into {mapping_names}"
            )
        value = getattr(module, name, None)
        if not isinstance(value, tuple) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(
                f"{field}: mapping constant is not a tuple of field names: {name}"
            )
        fields.update(value)
    return fields


def _fixed_carry_mapped_fields() -> set[str]:
    """Return fields explicitly mapped by the executable fixed-carry owner."""

    _, owner = _resolve_owner(
        "jax_orchidee.stomate.restart_io.stomate_fixed_path_carry_state",
        field="fixed_path_carry_fields",
        role="fixed carry owner",
    )
    tree = ast.parse(inspect.getsource(owner))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "groups"
            for target in node.targets
        ):
            continue
        return {
            child.value
            for child in ast.walk(node.value)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
    raise ValueError("fixed carry owner AST has no explicit groups mapping")


def _require_text(mapping: dict[str, Any], key: str, *, field: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: missing non-empty YAML contract key {key}")
    return value


def _mask_contracts_by_field(
    document: dict[str, Any],
    axes: dict[str, dict[str, Any]],
    contract_fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    groups_by_axis = document.get("mask_contract_groups", {})
    resolved: dict[str, dict[str, Any]] = {field: {} for field in contract_fields}
    for axis in ("npts", "nvm"):
        expected = {field for field in contract_fields if axis in axes[field]["axes"]}
        assigned: set[str] = set()
        for group_name, declaration in groups_by_axis.get(axis, {}).items():
            fields = declaration.get("fields")
            if not isinstance(fields, list) or not fields:
                raise ValueError(
                    f"mask group {axis}/{group_name} requires explicit fields"
                )
            overlap = assigned & set(fields)
            if overlap:
                raise ValueError(f"mask groups for {axis} overlap: {sorted(overlap)}")
            owner = _require_owner(
                declaration.get("owner"), field=group_name, role="mask owner"
            )
            provenance = declaration.get("fortran_provenance")
            if not isinstance(provenance, dict) or not all(
                provenance.get(key) for key in ("file", "subroutine", "lines")
            ):
                raise ValueError(
                    f"mask group {axis}/{group_name} lacks Fortran provenance"
                )
            for key in ("mode", "applicability", "inactive_write_behavior"):
                _require_text(declaration, key, field=group_name)
            for field in fields:
                resolved.setdefault(field, {})[axis] = {
                    "group": group_name,
                    "mode": declaration["mode"],
                    "applicability": declaration["applicability"],
                    "inactive_write_behavior": declaration["inactive_write_behavior"],
                    "owner": owner,
                    "fortran_provenance": provenance,
                }
            assigned.update(fields)
        missing = expected - assigned
        extra = assigned - expected
        if missing or extra:
            raise ValueError(
                f"mask contract coverage mismatch for {axis}: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
    for field in contract_fields:
        if not resolved[field]:
            resolved[field] = {"structural": {"mode": "non_land_non_pft_structural"}}
    return resolved


def _disposition(
    field: str,
    *,
    derived: dict[str, Any],
    runtime: dict[str, Any],
    packet: dict[str, Any],
    source_local: dict[str, Any],
) -> str:
    matches = [
        name
        for name, fields in (
            ("derived", derived),
            ("runtime_only", runtime),
            ("packet_only", packet),
            ("source_local_control", source_local),
        )
        if field in fields
    ]
    if len(matches) > 1:
        raise ValueError(f"{field}: multiple field dispositions declared: {matches}")
    return matches[0] if matches else "restart_variable"


def build_audit(
    contract_path: Path = DEFAULT_CONTRACT,
    source_path: Path = DEFAULT_SOURCE,
) -> dict[str, Any]:
    document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    axes = _axis_declarations(document)
    reader_fields = set(reader_contract_fields())
    contract_fields = lifecycle_contract_fields()
    missing_axes = sorted(set(contract_fields) - set(axes))
    extra_axes = sorted(set(axes) - set(contract_fields))
    if missing_axes or extra_axes:
        raise ValueError(
            f"axis declaration mismatch: missing={missing_axes}, extra={extra_axes}"
        )

    calls = _fortran_calls_by_field(source_path)
    aliases_by_field = document.get("source_aliases", {})
    line_overrides = document.get("source_line_overrides", {})
    derived = document.get("derived_fields", {})
    runtime_initialized = document.get("runtime_initialized_fields", {})
    packet_only = document.get("packet_only_fields", {})
    source_local = document.get("source_local_fields", {})
    fixed_path_carry = document.get("fixed_path_carry_fields", {})
    restart_field_contracts = document.get("restart_field_contracts", {})
    restart_update_groups = document.get("restart_update_groups", {})
    disposition_contracts = document.get("disposition_contracts", {})
    required_stages = document.get("required_lifecycle_stages", {})
    expected_dispositions = {
        "restart_variable",
        "derived",
        "runtime_only",
        "source_local_control",
        "packet_only",
    }
    if set(disposition_contracts) != expected_dispositions:
        raise ValueError(
            "disposition contract mismatch: "
            f"missing={sorted(expected_dispositions - set(disposition_contracts))}, "
            f"extra={sorted(set(disposition_contracts) - expected_dispositions)}"
        )
    if set(required_stages) != expected_dispositions:
        raise ValueError("required_lifecycle_stages must be disposition-specific")
    boolean_fields = set(document.get("boolean_fields", ()))
    half_hour_carry = set(STOMATE_HALF_HOUR_CARRY_FIELDS)
    day_end_writeback = set(STOMATE_DAY_END_WRITEBACK_FIELDS)
    executable_fixed_carry = set(STOMATE_FIXED_PATH_CARRY_FIELDS)
    declared_fixed_carry = set(fixed_path_carry)
    if declared_fixed_carry != executable_fixed_carry:
        raise ValueError(
            "fixed-path carry declaration mismatch: "
            f"missing={sorted(executable_fixed_carry - declared_fixed_carry)}, "
            f"extra={sorted(declared_fixed_carry - executable_fixed_carry)}"
        )
    missing_executable_group_mapping = (
        declared_fixed_carry - _fixed_carry_mapped_fields()
    )
    if missing_executable_group_mapping:
        raise ValueError(
            "fixed carry owner does not map declared fields: "
            f"{sorted(missing_executable_group_mapping)}"
        )
    if (
        not executable_fixed_carry <= half_hour_carry
        or not executable_fixed_carry <= day_end_writeback
    ):
        raise ValueError(
            "fixed-path carry fields must be executable half-hour and day-end packet fields"
        )
    daily_accumulator_fields = set(StomateDailyAccumulatorState._fields) - {
        "provenance"
    }
    writer_ledger = stomate_writerestart_field_ledger()
    template_backed_serialized_fields = set(writer_ledger.serialized_fields)
    validated_not_serialized_fields = set(writer_ledger.validated_not_serialized_fields)
    template_backed_writer_fields = (
        template_backed_serialized_fields | validated_not_serialized_fields
    )

    special_fields = (
        set(derived) | set(runtime_initialized) | set(packet_only) | set(source_local)
    )
    restart_fields = set(contract_fields) - special_fields
    update_contract_by_field: dict[str, dict[str, Any]] = {}
    for group_name, group in restart_update_groups.items():
        group_fields = group.get("fields")
        if not isinstance(group_fields, list) or not group_fields:
            raise ValueError(
                f"restart update group {group_name} requires an explicit non-empty fields list"
            )
        duplicates = set(group_fields) & set(update_contract_by_field)
        if duplicates:
            raise ValueError(
                f"restart update groups overlap for fields: {sorted(duplicates)}"
            )
        disposition = group.get("disposition")
        owner = _require_owner(
            group.get("owner"), field=group_name, role="update owner"
        )
        mapping_names = list(group.get("ast_mapping_names", ()))
        proven_keys = _mapping_literal_keys(
            owner,
            mapping_names,
            field=group_name,
        ) | _owner_constant_fields(
            owner,
            list(group.get("ast_dynamic_constants", ())),
            mapping_names,
            field=group_name,
        )
        if disposition == "production_overwrite":
            unproven = set(group_fields) - proven_keys
            if unproven:
                raise ValueError(
                    f"restart update group {group_name} owner does not write fields: {sorted(unproven)}"
                )
        elif disposition != "unproven_previous_state_copy":
            raise ValueError(
                f"restart update group {group_name} has unsupported disposition {disposition}"
            )
        for field in group_fields:
            update_contract_by_field[field] = {
                "group": group_name,
                "disposition": disposition,
                "owner": owner,
                "ast_proven": field in proven_keys,
            }
    grouped_restart_fields = set(update_contract_by_field)
    if grouped_restart_fields & declared_fixed_carry:
        raise ValueError(
            "restart update groups overlap fixed guarded carry fields: "
            f"{sorted(grouped_restart_fields & declared_fixed_carry)}"
        )
    missing_update_contracts = (
        restart_fields - grouped_restart_fields - declared_fixed_carry
    )
    extra_update_contracts = grouped_restart_fields - restart_fields
    if missing_update_contracts or extra_update_contracts:
        raise ValueError(
            "restart update contract coverage mismatch: "
            f"missing={sorted(missing_update_contracts)}, extra={sorted(extra_update_contracts)}"
        )
    mask_contracts = _mask_contracts_by_field(document, axes, contract_fields)

    records: list[dict[str, Any]] = []
    for field in contract_fields:
        aliases = list(aliases_by_field.get(field, [field]))
        override = line_overrides.get(field, {})
        read_lines = list(
            override.get("read_lines", _source_lines(aliases, calls, "restget_p"))
        )
        write_lines = list(
            override.get("write_lines", _source_lines(aliases, calls, "restput_p"))
        )
        disposition = _disposition(
            field,
            derived=derived,
            runtime=runtime_initialized,
            packet=packet_only,
            source_local=source_local,
        )
        declaration = {
            "derived": derived,
            "runtime_only": runtime_initialized,
            "packet_only": packet_only,
            "source_local_control": source_local,
        }.get(disposition, {}).get(field, {})
        is_fixed_path_carry = field in fixed_path_carry
        direct_restart_read = bool(read_lines)
        fortran_restart_write = bool(write_lines)
        packet_writeback_eligible = (
            field in day_end_writeback or field in daily_accumulator_fields
        )

        if disposition == "derived":
            provenance = derived[field]["fortran_provenance"]
        elif disposition == "runtime_only":
            provenance = runtime_initialized[field]["fortran_provenance"]
        elif disposition == "packet_only":
            provenance = packet_only[field]["fortran_provenance"]
        elif disposition == "source_local_control":
            provenance = source_local[field]["fortran_provenance"]
        else:
            provenance = {
                "file": str(source_path.relative_to(ROOT)).replace("\\", "/"),
                "subroutine": "readstart/writerestart",
                "read_lines": read_lines,
                "write_lines": write_lines,
            }

        template_backed_write = field in template_backed_writer_fields
        template_serialized = field in template_backed_serialized_fields
        normalized_reader = field in reader_fields
        day_start_state = normalized_reader or field in half_hour_carry
        update_declared = field in half_hour_carry or packet_writeback_eligible
        next_restart_read_verified = bool(template_serialized and direct_restart_read)
        disposition_contract = disposition_contracts[disposition]
        freshness_contract = dict(disposition_contract.get("freshness_contract", {}))
        if not freshness_contract:
            raise ValueError(
                f"{field}: missing YAML freshness_contract for {disposition}"
            )
        for key in ("step_behavior", "cross_step", "cross_year"):
            _require_text(freshness_contract, key, field=field)
        mask_contract = mask_contracts[field]

        stages: dict[str, bool]
        owners: dict[str, str] = {}
        if disposition == "restart_variable":
            field_restart_contract = restart_field_contracts.get(field, {})
            for role in (
                "read_owner",
                "day_start_owner",
                "day_end_owner",
                "write_owner",
                "next_read_owner",
            ):
                owners[role] = _require_owner(
                    field_restart_contract.get(role, disposition_contract.get(role)),
                    field=field,
                    role=role,
                )
            if field == "resp_hetero":
                if (
                    field_restart_contract.get("fortran_state_symbol")
                    != "resp_hetero_d"
                ):
                    raise ValueError(
                        "resp_hetero restart field must declare Fortran SAVE state resp_hetero_d"
                    )
                if field_restart_contract.get("restart_variable_name") != "resp_hetero":
                    raise ValueError(
                        "resp_hetero_d must map to the Fortran restart variable resp_hetero"
                    )
            if is_fixed_path_carry:
                update_disposition = "guarded_carry"
                update_owner = (
                    "jax_orchidee.stomate.restart_io.stomate_fixed_path_carry_state"
                )
                update_group = "fixed_path_carry_fields"
                update_proven = True
                freshness_contract["guard"] = _require_text(
                    fixed_path_carry[field], "guard", field=field
                )
            else:
                update_contract = update_contract_by_field[field]
                update_disposition = update_contract["disposition"]
                update_owner = update_contract["owner"]
                update_group = update_contract["group"]
                update_proven = bool(
                    update_contract["ast_proven"]
                    and update_disposition == "production_overwrite"
                )
            owners["update_owner"] = _require_owner(
                update_owner, field=field, role="field-specific update owner"
            )
            freshness_contract["resolved_update_mode"] = update_disposition
            freshness_contract["update_group"] = update_group
            stages = {
                "fortran_restart_read": direct_restart_read,
                "day_start_state": normalized_reader and day_start_state,
                "update_or_guarded_carry": update_declared and update_proven,
                "day_end_packet": packet_writeback_eligible,
                "restart_write": fortran_restart_write and template_serialized,
                "next_restart_read": next_restart_read_verified,
            }
        elif disposition == "derived":
            owners["executable_owner"] = _require_owner(
                declaration.get("jax_owner"), field=field, role="jax_owner"
            )
            inputs = declaration.get("inputs")
            recompute_at = declaration.get("recompute_at")
            stages = {
                "source_inputs": isinstance(inputs, list)
                and bool(inputs)
                and all(item in contract_fields for item in inputs),
                "executable_owner": True,
                "recompute_point": isinstance(recompute_at, str)
                and bool(recompute_at.strip()),
                "explicitly_not_serialized": disposition_contract.get("serialization")
                == "not_serialized_recomputed",
            }
        elif disposition == "runtime_only":
            owners["initialization_owner"] = _require_owner(
                declaration.get("initialization_owner"),
                field=field,
                role="initialization_owner",
            )
            owners["update_owner"] = _require_owner(
                declaration.get("update_owner"), field=field, role="update_owner"
            )
            stages = {
                "initialization_owner": True,
                "update_owner": True,
                "explicitly_not_serialized": declaration.get("serialization")
                == "not_serialized_runtime",
            }
        elif disposition == "source_local_control":
            owners["read_owner"] = _require_owner(
                declaration.get("read_owner"), field=field, role="read_owner"
            )
            owners["use_owner"] = _require_owner(
                declaration.get("use_owner"), field=field, role="use_owner"
            )
            stages = {
                "read_owner": True,
                "use_owner": True,
                "explicitly_not_serialized": declaration.get("serialization")
                == "not_serialized_source_local",
            }
        else:
            owners["producer"] = _require_owner(
                declaration.get("producer"), field=field, role="producer"
            )
            owners["consumer"] = _require_owner(
                declaration.get("consumer"), field=field, role="consumer"
            )
            for name in ("day_start", "update", "day_end", "cross_step", "cross_year"):
                _require_text(declaration, name, field=field)
            stages = {
                "producer": True,
                "consumer": True,
                **{
                    name: True
                    for name in (
                        "day_start",
                        "update",
                        "day_end",
                        "cross_step",
                        "cross_year",
                    )
                },
                "explicitly_not_serialized": declaration.get("serialization")
                == "not_serialized_packet",
            }
            freshness_contract["cross_step"] = declaration["cross_step"]
            freshness_contract["cross_year"] = declaration["cross_year"]

        expected_stage_names = set(required_stages[disposition])
        if set(stages) != expected_stage_names:
            raise ValueError(
                f"{field}: lifecycle stages do not match YAML requirement for {disposition}: "
                f"missing={sorted(expected_stage_names - set(stages))}, extra={sorted(set(stages) - expected_stage_names)}"
            )
        gaps = [
            f"Missing {disposition} lifecycle stage: {name}."
            for name, passed in stages.items()
            if not passed
        ]
        lifecycle_closed = all(stages.values())

        records.append(
            {
                "field": field,
                "reader_contracts": [
                    state_type.__name__
                    for state_type in RESTART_READER_TYPES
                    if field in state_type._fields
                ],
                "source_aliases": aliases,
                **axes[field],
                "mask_contract": mask_contract,
                "mask": {
                    "value_encoding": "boolean_after_reader"
                    if field in boolean_fields
                    else "numeric",
                    "land_mask_explicit": "npts" not in axes[field]["axes"]
                    or "npts" in mask_contract,
                    "pft_mask_explicit": "nvm" not in axes[field]["axes"]
                    or "nvm" in mask_contract,
                },
                "freshness_contract": freshness_contract,
                "field_disposition": disposition,
                "owners": owners,
                "restart_field_contract": restart_field_contracts.get(field),
                "lifecycle": {
                    "fortran_restart_read": direct_restart_read,
                    "fortran_initialization_available": (
                        direct_restart_read
                        or disposition
                        in {
                            "derived",
                            "runtime_only",
                            "packet_only",
                            "source_local_control",
                        }
                    ),
                    "normalized_jax_reader": normalized_reader,
                    "day_start_state": day_start_state,
                    "half_hour_or_daily_update_declared": update_declared,
                    "day_end_packet_declared": packet_writeback_eligible,
                    "fortran_restart_write_exists": fortran_restart_write,
                    "jax_restart_write_exists": template_serialized,
                    "jax_template_backed_write": template_backed_write,
                    "next_restart_read_verified": next_restart_read_verified,
                    "required_stages": stages,
                    "closed": lifecycle_closed,
                },
                "day_end_transition_owner": (
                    {
                        "kind": "fixed_path_carry",
                        "guard": fixed_path_carry[field]["guard"],
                        "fortran_provenance": fixed_path_carry[field][
                            "fortran_provenance"
                        ],
                        "jax_owner": "jax_orchidee.stomate.restart_io.stomate_fixed_path_carry_state",
                    }
                    if is_fixed_path_carry
                    else None
                ),
                "fortran_provenance": provenance,
                "gaps": gaps,
            }
        )

    closed_fields = sum(record["lifecycle"]["closed"] for record in records)
    dispositions = {
        disposition: sum(
            record["field_disposition"] == disposition for record in records
        )
        for disposition in sorted({record["field_disposition"] for record in records})
    }
    state_transition_closed = closed_fields == len(records)
    restart_lifecycle_closed = (
        state_transition_closed and not document["known_global_gaps"]
    )
    return {
        "schema_version": document["schema_version"],
        "contract": str(contract_path.relative_to(ROOT)).replace("\\", "/"),
        "source": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "closed": restart_lifecycle_closed,
        "state_transition_closed": state_transition_closed,
        "restart_lifecycle_closed": restart_lifecycle_closed,
        "summary": {
            "contract_fields": len(records),
            "restart_reader_fields": len(reader_fields),
            "packet_only_fields": len(set(contract_fields) - reader_fields),
            "fields_with_direct_fortran_read": sum(
                bool(record["fortran_provenance"].get("read_lines"))
                for record in records
            ),
            "fields_with_fortran_writer": sum(
                record["lifecycle"]["fortran_restart_write_exists"]
                for record in records
            ),
            "day_end_packet_declared": sum(
                record["lifecycle"]["day_end_packet_declared"] for record in records
            ),
            "jax_restart_writer_fields": sum(
                record["lifecycle"]["jax_restart_write_exists"] for record in records
            ),
            "template_backed_writer_fields": sum(
                record["lifecycle"]["jax_template_backed_write"] for record in records
            ),
            "next_restart_read_verified_fields": sum(
                record["lifecycle"]["next_restart_read_verified"] for record in records
            ),
            "closed_fields": closed_fields,
            "fixed_path_carry_fields": len(executable_fixed_carry),
            "field_dispositions": dispositions,
            "writer_normalized_fields": len(writer_ledger.normalized_state_fields),
            "writer_source_local_fields": len(
                writer_ledger.source_local_not_restart_fields
            ),
        },
        "required_lifecycle_stages": document["required_lifecycle_stages"],
        "records": records,
        "limitations": document["known_global_gaps"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = build_audit(args.contract, args.source)
    text = json.dumps(audit, indent=2, sort_keys=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
