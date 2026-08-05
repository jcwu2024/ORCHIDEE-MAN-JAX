"""Machine audits for Gate-E1 graph boundaries and frozen label ownership."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import jax

from research.daily_coarse_graining.daily_flux_label_inventory import (
    load_daily_flux_label_inventory,
)

from .process_heads import PROCESS_LABEL_BINDINGS
from .types import (
    DailyOperatorInput,
    NativeForcingInput,
    PFTAxisInput,
    StateDefinedMasks,
    StateFeatureGroups,
    StaticRegistryInput,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
FORBIDDEN_INFERENCE_SYMBOLS = frozenset(
    {
        "encode_inventory_endpoints",
        "reconstruct_compiled_forcing_day",
        "reconstruct_compiled_forcing_window",
        "reconstruct_fast_day_target",
        "reconstruct_fast_day_target_compiled",
        "physical_fast_day_target",
        "paper_1961_driver_later_day_runtime_result",
        "_paper_1961_later_day_half_hour_transition",
    }
)
FORBIDDEN_MODEL_MODULE_FRAGMENTS = (
    "canonical_daily_model",
    "structured_canonical",
    "axis_process",
    "causal_carbon",
    "rollout_stability",
)
FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "landpoint_id",
        "pft_id",
        "pft_ids",
        "fortran_pft_id",
        "teacher_endpoint",
        "teacher_end_state",
        "next_state",
        "next_day_target",
        "physical_fast_day_target",
        "rain_input",
        "snowfall_input",
        "litter_input",
        "poc_respiration",
        "poc_to_doc",
        "doc_to_poc",
        "doc_respiration",
        "daily_temperature_context",
        "tsurf_daily",
        "tsoil_daily",
        "state",
        "state_defined",
        "day_start_process_state",
        "process_static",
    }
)
ALLOWED_DAILY_OPERATOR_INPUT_FIELDS = (
    "continuous_state",
    "discrete_state",
    "forcing",
    "pft",
    "static_registry",
    "annual_conditions",
    "year",
    "day_index",
)
ALLOWED_INPUT_SCHEMAS = {
    "DailyOperatorInput": ALLOWED_DAILY_OPERATOR_INPUT_FIELDS,
    "StateFeatureGroups": (
        "canopy",
        "soil",
        "snow",
        "water",
        "carbon",
        "thermal",
        "other",
    ),
    "StateDefinedMasks": (
        "canopy",
        "soil",
        "snow",
        "water",
        "carbon",
        "thermal",
        "other",
    ),
    "NativeForcingInput": (
        "values",
        "record_time_seconds",
        "duration_seconds",
        "predecessor_mask",
        "record_mask",
    ),
    "PFTAxisInput": (
        "pft_state",
        "pft_parameters",
        "pft_traits",
        "pft_fraction",
        "active_pft_mask",
    ),
    "StaticRegistryInput": (
        "context_features",
        "rprof",
        "z_soil",
    ),
}
INPUT_SCHEMA_TYPES = (
    DailyOperatorInput,
    StateFeatureGroups,
    StateDefinedMasks,
    NativeForcingInput,
    PFTAxisInput,
    StaticRegistryInput,
)


def audit_inference_source_graph(package_root: str | Path = PACKAGE_ROOT) -> dict[str, Any]:
    """Reject forbidden Teacher, endpoint, decoder, and historical-model calls."""

    root = Path(package_root)
    violations: list[dict[str, Any]] = []
    audited = []
    for path in sorted(root.glob("*.py")):
        if path.name == "audit.py":
            continue
        audited.append(path.name)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_INFERENCE_SYMBOLS:
                violations.append({"file": path.name, "line": node.lineno, "symbol": node.id})
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_INFERENCE_SYMBOLS:
                violations.append({"file": path.name, "line": node.lineno, "symbol": node.attr})
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                for name in names:
                    if any(fragment in name for fragment in FORBIDDEN_MODEL_MODULE_FRAGMENTS):
                        violations.append({"file": path.name, "line": node.lineno, "symbol": name})
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "range"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 48
            ):
                violations.append({"file": path.name, "line": node.lineno, "symbol": "range(48)"})
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == 13
            ):
                violations.append({"file": path.name, "line": node.lineno, "symbol": "PFT index 13"})
    actual_inputs = tuple(DailyOperatorInput._fields)
    allowed_inputs = set(ALLOWED_DAILY_OPERATOR_INPUT_FIELDS)
    missing_inputs = sorted(allowed_inputs - set(actual_inputs))
    extra_inputs = sorted(set(actual_inputs) - allowed_inputs)
    forbidden_inputs = sorted(set(actual_inputs) & FORBIDDEN_INPUT_FIELDS)
    if forbidden_inputs:
        violations.extend(
            {"file": "types.py", "line": None, "symbol": name}
            for name in forbidden_inputs
        )
    schema_drift = []
    input_schemas = {}
    for schema_type in INPUT_SCHEMA_TYPES:
        schema_name = schema_type.__name__
        actual = tuple(schema_type._fields)
        expected = ALLOWED_INPUT_SCHEMAS[schema_name]
        input_schemas[schema_name] = list(actual)
        if actual != expected:
            schema_drift.append(
                {
                    "schema": schema_name,
                    "expected": list(expected),
                    "actual": list(actual),
                }
            )
        nested_forbidden = sorted(set(actual) & FORBIDDEN_INPUT_FIELDS)
        violations.extend(
            {"file": "types.py", "line": None, "symbol": f"{schema_name}.{name}"}
            for name in nested_forbidden
        )
    return {
        "schema_version": "gate_e1_inference_source_graph_audit_v1",
        "audited_files": audited,
        "allowed_input_fields": list(ALLOWED_DAILY_OPERATOR_INPUT_FIELDS),
        "actual_input_fields": list(actual_inputs),
        "missing_input_fields": missing_inputs,
        "extra_input_fields": extra_inputs,
        "forbidden_input_fields": forbidden_inputs,
        "input_schemas": input_schemas,
        "input_schema_drift": schema_drift,
        "violations": violations,
        "passed": not violations and not missing_inputs and not extra_inputs and not schema_drift,
    }


def audit_process_label_bindings(inventory_path: str | Path | None = None) -> dict[str, Any]:
    """Require one owner binding for every non-budget state/supervision label."""

    inventory = (
        load_daily_flux_label_inventory()
        if inventory_path is None
        else load_daily_flux_label_inventory(inventory_path)
    )
    bound = [label for labels in PROCESS_LABEL_BINDINGS.values() for label in labels]
    duplicates = sorted({label for label in bound if bound.count(label) > 1})
    required = {
        label.label_id
        for label in inventory.labels
        if {"state_update", "supervision"}.intersection(label.required_for)
    }
    missing = sorted(required - set(bound))
    unknown = sorted(set(bound) - set(inventory.labels_by_id))
    return {
        "schema_version": "gate_e1_process_label_binding_audit_v1",
        "required_label_count": len(required),
        "bound_label_count": len(set(bound)),
        "missing": missing,
        "unknown": unknown,
        "duplicates": duplicates,
        "passed": not missing and not unknown and not duplicates,
    }


def audit_executable_jaxpr(transition, *args) -> dict[str, Any]:
    """Audit the staged graph for a forbidden 48-step recurrent scan."""

    closed = jax.make_jaxpr(transition)(*args)
    scans: list[int] = []
    primitives: list[str] = []

    def visit(value) -> None:
        if hasattr(value, "jaxpr") and hasattr(value, "consts"):
            visit(value.jaxpr)
        elif hasattr(value, "eqns") and hasattr(value, "invars"):
            for equation in value.eqns:
                name = str(equation.primitive)
                primitives.append(name)
                if name == "scan":
                    scans.append(int(equation.params.get("length", -1)))
                for parameter in equation.params.values():
                    visit(parameter)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(closed)
    forbidden = [length for length in scans if length == 48]
    callbacks = sorted(
        {
            name
            for name in primitives
            if "callback" in name or name in {"host_callback", "io_callback"}
        }
    )
    return {
        "schema_version": "gate_e1_executable_jaxpr_audit_v1",
        "scan_lengths": scans,
        "primitive_count": len(primitives),
        "forbidden_48_step_scans": forbidden,
        "callback_primitives": callbacks,
        "passed": not forbidden and not callbacks,
    }
