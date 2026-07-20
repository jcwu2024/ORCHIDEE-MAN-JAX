from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.fast_state import (  # noqa: E402
    DriverFastStateSpec,
    fast_state_from_previous_fields,
)
from jax_orchidee.driver.orchestration import (  # noqa: E402
    DAY1_PREVIOUS_STEP_STATE_GAPS,
    YEAR_HANDOFF_STATE_GAPS,
)


ORCHESTRATION_PATH = ROOT / "jax_orchidee" / "driver" / "orchestration.py"
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "reference_mode" / "sechiba_state_transition_contract.json"
)
DEFAULT_CONTRACT = ROOT / "docs" / "source_audits" / "sechiba_state_field_contract.yaml"

COMPONENTS = (
    "driver_previous_step_state",
    "diffuco_previous_step_state",
    "enerbil_previous_step_state",
    "hydrol_previous_step_state",
    "thermosoil_previous_step_state",
)
ADD_HELPERS = (
    "_add_enerbil_local_to_previous_fields",
    "_add_diffuco_local_to_previous_fields",
    "_add_diffuco_saved_controls_to_previous_fields",
    "_add_thermosoil_to_previous_fields",
    "_add_condveg_to_previous_fields",
    "_add_hydrol_to_previous_fields",
)
CARRY_HELPER = "_runtime_next_state_from_components"
CONTRACT_REQUIRED_KEYS = frozenset(
    {
        "component",
        "field",
        "shape_group",
        "dtype_kind",
        "applicability",
        "inactive_behavior",
        "freshness",
        "freshness_guard",
        "producer_phase",
        "consumer_phase",
        "jax_owner",
        "source_ref",
    }
)
SOURCE_REQUIRED_KEYS = frozenset({"file", "subroutine", "lines"})
DTYPE_KINDS = frozenset({"real", "integer", "logical"})


@dataclass(frozen=True)
class _ComponentAlias:
    names: frozenset[str]


def _string_values(node: ast.AST, env: dict[str, object]) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name):
        value = env.get(node.id, set())
        return set(value) if isinstance(value, (set, frozenset)) else set()
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return (
            set().union(*(_string_values(item, env) for item in node.elts))
            if node.elts
            else set()
        )
    if isinstance(node, ast.IfExp):
        body_env, else_env = _branch_envs(node.test, env)
        return _string_values(node.body, body_env) | _string_values(
            node.orelse, else_env
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Dict)
        and node.args
    ):
        mapping: dict[str, str] = {}
        for key, value in zip(
            node.func.value.keys, node.func.value.values, strict=True
        ):
            keys = _string_values(key, env) if key is not None else set()
            values = _string_values(value, env)
            if len(keys) == len(values) == 1:
                mapping[next(iter(keys))] = next(iter(values))
        source = _string_values(node.args[0], env)
        fallback = _string_values(node.args[1], env) if len(node.args) > 1 else set()
        return {
            mapping.get(name, name if name in fallback else name) for name in source
        }
    return set()


def _branch_envs(
    test: ast.AST, env: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    body_env = dict(env)
    else_env = dict(env)
    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and len(test.comparators) == 1
        and isinstance(test.left, ast.Name)
    ):
        return body_env, else_env
    name = test.left.id
    domain = env.get(name)
    if not isinstance(domain, (set, frozenset)):
        return body_env, else_env
    compared = _string_values(test.comparators[0], env)
    if not compared:
        return body_env, else_env
    if isinstance(test.ops[0], (ast.In, ast.Eq)):
        body_env[name] = set(domain) & compared
        else_env[name] = set(domain) - compared
    elif isinstance(test.ops[0], (ast.NotIn, ast.NotEq)):
        body_env[name] = set(domain) - compared
        else_env[name] = set(domain) & compared
    return body_env, else_env


def _component_alias(node: ast.AST, env: dict[str, object]) -> _ComponentAlias | None:
    if isinstance(node, ast.Name):
        value = env.get(node.id)
        return value if isinstance(value, _ComponentAlias) else None
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "fields"
    ):
        names = _string_values(node.slice, env)
        return _ComponentAlias(frozenset(names)) if names else None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "fields"
        and node.func.attr == "setdefault"
        and node.args
    ):
        names = _string_values(node.args[0], env)
        return _ComponentAlias(frozenset(names)) if names else None
    return None


def _record_target(
    target: ast.AST, env: dict[str, object], found: dict[str, set[str]]
) -> None:
    if not isinstance(target, ast.Subscript):
        return
    alias = _component_alias(target.value, env)
    if alias is None:
        return
    fields = _string_values(target.slice, env)
    for component in alias.names:
        found.setdefault(component, set()).update(fields)


def _loop_bindings(
    target: ast.AST, iterator: ast.AST, env: dict[str, object]
) -> dict[str, set[str]]:
    bindings: dict[str, set[str]] = {}
    if isinstance(target, ast.Name):
        bindings[target.id] = _string_values(iterator, env)
        return bindings
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
        iterator, (ast.Tuple, ast.List)
    ):
        rows = [row for row in iterator.elts if isinstance(row, (ast.Tuple, ast.List))]
        for index, item in enumerate(target.elts):
            if isinstance(item, ast.Name):
                bindings[item.id] = set().union(
                    *(
                        _string_values(row.elts[index], env)
                        for row in rows
                        if index < len(row.elts)
                    )
                )
    return bindings


def _scan_statements(
    statements: Iterable[ast.stmt], env: dict[str, object], found: dict[str, set[str]]
) -> None:
    for statement in statements:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            value = statement.value
            for target in targets:
                _record_target(target, env, found)
                if isinstance(target, ast.Name) and value is not None:
                    alias = _component_alias(value, env)
                    if alias is not None:
                        env[target.id] = alias
                    else:
                        values = _string_values(value, env)
                        if values:
                            env[target.id] = values
        elif isinstance(statement, ast.For):
            loop_env = dict(env)
            loop_env.update(_loop_bindings(statement.target, statement.iter, env))
            _scan_statements(statement.body, loop_env, found)
            _scan_statements(statement.orelse, dict(loop_env), found)
        elif isinstance(statement, ast.If):
            body_env, else_env = _branch_envs(statement.test, env)
            _scan_statements(statement.body, body_env, found)
            _scan_statements(statement.orelse, else_env, found)
        elif isinstance(statement, (ast.With, ast.AsyncWith, ast.While, ast.Try)):
            for child in ast.iter_child_nodes(statement):
                if isinstance(child, ast.stmt):
                    _scan_statements([child], dict(env), found)


def _function_nodes(source: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(source, filename=str(ORCHESTRATION_PATH))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def extract_declared_components(source: str) -> set[str]:
    function = _function_nodes(source)["_empty_previous_step_state_fields"]
    for node in ast.walk(function):
        if isinstance(node, ast.Dict):
            components = set().union(
                *(_string_values(key, {}) for key in node.keys if key is not None)
            )
            if components:
                return components
    return set()


def extract_producers(
    source: str,
) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    functions = _function_nodes(source)
    by_function: dict[str, dict[str, set[str]]] = {}
    combined: dict[str, set[str]] = {component: set() for component in COMPONENTS}
    for name in (*ADD_HELPERS, CARRY_HELPER):
        found: dict[str, set[str]] = {}
        _scan_statements(functions[name].body, {}, found)
        scoped = {
            component: fields
            for component, fields in found.items()
            if component in COMPONENTS
        }
        by_function[name] = scoped
        for component, fields in scoped.items():
            combined.setdefault(component, set()).update(fields)
    return combined, by_function


def _required_fields(gaps: Iterable[object]) -> dict[str, set[str]]:
    required = {component: set() for component in COMPONENTS}
    for gap in gaps:
        if gap.component in required:
            required[gap.component].update(gap.fields)
    return required


def _fast_spec(producers: dict[str, set[str]]) -> DriverFastStateSpec:
    state = fast_state_from_previous_fields(
        tstep=0,
        fields_by_component={
            component: {field: None for field in sorted(fields)}
            for component, fields in producers.items()
        },
        provenance_by_component={component: () for component in producers},
    )
    return state.spec


def _load_contract(
    contract_path: Path, contract_document: dict[str, object] | None
) -> dict[str, object]:
    if contract_document is not None:
        return contract_document
    document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("SECHIBA state-field contract must be a YAML mapping")
    return document


def _owner_exists(owner: object) -> bool:
    if not isinstance(owner, str) or "." not in owner:
        return False
    module_name, attribute = owner.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except (ImportError, AttributeError):
        return False
    return hasattr(module, attribute)


def _source_ref_error(source: object) -> str | None:
    if not isinstance(source, dict):
        return "source reference is not a mapping"
    missing = SOURCE_REQUIRED_KEYS - set(source)
    if missing:
        return f"missing keys: {sorted(missing)}"
    relative = source["file"]
    subroutine = source["subroutine"]
    line_span = source["lines"]
    if not isinstance(relative, str) or not relative.startswith(
        "fortran_source/ORCHIDEE/"
    ):
        return "file is not an ORCHIDEE source path"
    path = ROOT / relative
    if not path.is_file():
        return "Fortran source file does not exist"
    if not isinstance(subroutine, str) or not subroutine.strip():
        return "subroutine is empty"
    if not isinstance(line_span, str) or "-" not in line_span:
        return "lines must be an inclusive start-end string"
    try:
        first, last = (int(value) for value in line_span.split("-", 1))
    except ValueError:
        return "lines are not integers"
    source_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if first < 1 or last < first or last > len(source_lines):
        return "line span is outside the source file"
    span = "\n".join(source_lines[first - 1 : last]).lower()
    if subroutine.lower() not in span:
        return "subroutine name is absent from the cited line span"
    return None


def build_audit(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    contract_document: dict[str, object] | None = None,
) -> dict[str, object]:
    source = ORCHESTRATION_PATH.read_text(encoding="utf-8")
    declared = extract_declared_components(source)
    producers, producer_sources = extract_producers(source)
    day_required = _required_fields(DAY1_PREVIOUS_STEP_STATE_GAPS)
    year_required = _required_fields(YEAR_HANDOFF_STATE_GAPS)
    spec = _fast_spec(producers)
    fast_fields = {
        component: set(fields)
        for component, fields in zip(
            spec.components, spec.field_names_by_component, strict=True
        )
    }

    contract = _load_contract(contract_path, contract_document)
    axis_groups = contract.get("axis_groups", {})
    source_refs = contract.get("source_refs", {})
    raw_contracts = contract.get("contracts", [])
    if not isinstance(axis_groups, dict):
        axis_groups = {}
    if not isinstance(source_refs, dict):
        source_refs = {}
    if not isinstance(raw_contracts, list):
        raw_contracts = []

    contract_pairs: list[tuple[str, str]] = []
    incomplete_contracts: dict[str, list[str]] = {}
    unknown_shape_groups: dict[str, str] = {}
    invalid_dtype_kinds: dict[str, object] = {}
    invalid_owners: dict[str, object] = {}
    owner_field_mismatches: dict[str, dict[str, object]] = {}
    invalid_source_refs: dict[str, str] = {}
    owners_by_pair: dict[tuple[str, str], set[str]] = {}
    for function_name, by_component in producer_sources.items():
        owner = f"jax_orchidee.driver.orchestration.{function_name}"
        for component, fields in by_component.items():
            for field in fields:
                owners_by_pair.setdefault((component, field), set()).add(owner)
    contracts_by_component: dict[str, set[str]] = {
        component: set() for component in COMPONENTS
    }
    for index, entry in enumerate(raw_contracts):
        label = f"contract[{index}]"
        if not isinstance(entry, dict):
            incomplete_contracts[label] = sorted(CONTRACT_REQUIRED_KEYS)
            continue
        component = entry.get("component")
        field = entry.get("field")
        if isinstance(component, str) and isinstance(field, str):
            label = f"{component}.{field}"
            contract_pairs.append((component, field))
            contracts_by_component.setdefault(component, set()).add(field)
        missing = sorted(
            key
            for key in CONTRACT_REQUIRED_KEYS
            if key not in entry or entry[key] in (None, "")
        )
        if missing:
            incomplete_contracts[label] = missing
        shape_group = entry.get("shape_group")
        if shape_group not in axis_groups:
            unknown_shape_groups[label] = str(shape_group)
        dtype_kind = entry.get("dtype_kind")
        if dtype_kind not in DTYPE_KINDS:
            invalid_dtype_kinds[label] = dtype_kind
        owner = entry.get("jax_owner")
        if not _owner_exists(owner):
            invalid_owners[label] = owner
        elif isinstance(component, str) and isinstance(field, str):
            expected_owners = owners_by_pair.get((component, field), set())
            if owner not in expected_owners:
                owner_field_mismatches[label] = {
                    "declared": owner,
                    "extracted_production_owners": sorted(expected_owners),
                }
        source_ref = entry.get("source_ref")
        if source_ref not in source_refs:
            invalid_source_refs[label] = f"unknown source_ref: {source_ref}"

    source_ref_errors = {
        name: error
        for name, source in source_refs.items()
        if (error := _source_ref_error(source)) is not None
    }
    pair_counts: dict[tuple[str, str], int] = {}
    for pair in contract_pairs:
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    duplicate_contracts = sorted(
        f"{component}.{field}"
        for (component, field), count in pair_counts.items()
        if count != 1
    )
    expected_pairs = {
        (component, field)
        for component, fields in producers.items()
        for field in fields
    }
    actual_pairs = set(contract_pairs)
    producer_without_contract = sorted(
        f"{component}.{field}" for component, field in expected_pairs - actual_pairs
    )
    contract_without_producer = sorted(
        f"{component}.{field}" for component, field in actual_pairs - expected_pairs
    )

    components: dict[str, object] = {}
    for component in COMPONENTS:
        produced = producers.get(component, set())
        day = day_required[component]
        year = year_required[component]
        fast = fast_fields.get(component, set())
        missing_day = day - produced
        missing_year = year - produced
        missing_fast = produced - fast
        contract_fields = contracts_by_component.get(component, set())
        contract_missing = produced - contract_fields
        contract_extra = contract_fields - produced
        contract_labels = {f"{component}.{field}" for field in contract_fields}
        component_contract_errors = (
            bool(contract_missing)
            or bool(contract_extra)
            or any(label in contract_labels for label in incomplete_contracts)
            or any(label in contract_labels for label in unknown_shape_groups)
            or any(label in contract_labels for label in invalid_dtype_kinds)
            or any(label in contract_labels for label in invalid_owners)
            or any(label in contract_labels for label in owner_field_mismatches)
            or any(label in contract_labels for label in invalid_source_refs)
        )
        components[component] = {
            "producer_fields": sorted(produced),
            "day1_required_fields": sorted(day),
            "year_required_fields": sorted(year),
            "fast_spec_fields": sorted(fast),
            "producer_minus_day1_required": sorted(produced - day),
            "day1_required_minus_producer": sorted(missing_day),
            "producer_minus_year_required": sorted(produced - year),
            "year_required_minus_producer": sorted(missing_year),
            "producer_minus_fast_spec": sorted(missing_fast),
            "contract_fields": sorted(contract_fields),
            "producer_without_contract": sorted(contract_missing),
            "contract_without_producer": sorted(contract_extra),
            "closed": not missing_day
            and not missing_year
            and not missing_fast
            and not component_contract_errors,
        }

    closed_components = [
        name for name, report in components.items() if report["closed"]
    ]
    errors = {
        "undeclared_producer_components": sorted(
            set(producers) - declared - {"driver_previous_step_state"}
        ),
        "day1_required_without_producer": {
            name: report["day1_required_minus_producer"]
            for name, report in components.items()
            if report["day1_required_minus_producer"]
        },
        "year_required_without_producer": {
            name: report["year_required_minus_producer"]
            for name, report in components.items()
            if report["year_required_minus_producer"]
        },
        "producer_without_fast_spec_field": {
            name: report["producer_minus_fast_spec"]
            for name, report in components.items()
            if report["producer_minus_fast_spec"]
        },
        "duplicate_contracts": duplicate_contracts,
        "producer_without_contract": producer_without_contract,
        "contract_without_producer": contract_without_producer,
        "incomplete_contracts": incomplete_contracts,
        "unknown_shape_groups": unknown_shape_groups,
        "invalid_dtype_kinds": invalid_dtype_kinds,
        "invalid_owners": invalid_owners,
        "owner_field_mismatches": owner_field_mismatches,
        "invalid_source_refs": invalid_source_refs,
        "source_ref_errors": source_ref_errors,
    }
    return {
        "schema_version": 1,
        "scope": "SECHIBA driver/diffuco/enerbil/hydrol/thermosoil field-level previous-step transition contracts",
        "source": str(ORCHESTRATION_PATH.relative_to(ROOT)).replace("\\", "/"),
        "contract": str(contract_path.relative_to(ROOT)).replace("\\", "/"),
        "contract_count": len(raw_contracts),
        "source_ref_count": len(source_refs),
        "declared_empty_components": sorted(declared),
        "producer_functions": {
            name: {
                component: sorted(fields)
                for component, fields in sorted(by_component.items())
            }
            for name, by_component in producer_sources.items()
        },
        "fast_state_spec_type": type(spec).__name__,
        "components": components,
        "closed_components": closed_components,
        "errors": errors,
        "closed": len(closed_components) == len(COMPONENTS)
        and not any(errors.values()),
        "all_components_closed": len(closed_components) == len(COMPONENTS)
        and not any(errors.values()),
        "limitations": [
            "Closed applies only to driver, diffuco, enerbil, hydrol, and thermosoil; slowproc/STOMATE and other SECHIBA components are outside this audit.",
            "Producer extraction is a static union of conditional writes; field contracts therefore state the exact overwrite/carry guard for conditional producers.",
            "The imported fast-state spec is instantiated from statically extracted producer names; it checks layout preservation, not an independently declared required schema.",
            "This audit validates declared axes, dtype kind, applicability, inactive behavior, freshness, phases, owner existence, and Fortran provenance; runtime numeric values and restart serialization are separate gates.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the covered SECHIBA state transition field contract."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--require-closed", action="store_true")
    args = parser.parse_args(argv)
    audit = build_audit(contract_path=args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        f"WROTE {args.output.resolve()} closed={audit['closed']} components={audit['closed_components']}"
    )
    return int(args.require_closed and not audit["closed"])


if __name__ == "__main__":
    raise SystemExit(main())
