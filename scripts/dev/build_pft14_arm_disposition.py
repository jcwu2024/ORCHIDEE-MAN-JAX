from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "outputs" / "reference_mode" / "fortran_control_flow_inventory.json"
DEFAULT_RESOLUTION = ROOT / "outputs" / "reference_mode" / "pft14_control_flow_resolution.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "reference_mode" / "pft14_arm_disposition.json"
DEFAULT_CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
DEFAULT_STATIC_EDGE_LEDGER = ROOT / "docs" / "source_audits" / "paper_static_disabled_call_edges.yaml"
DEFAULT_MANUAL_RULE_GLOB = "pft14_arm_dispositions_*.yaml"
ALLOWED_MANUAL_DISPOSITIONS = {
    "pft14_active",
    "pft14_conditional",
    "paper_static_inactive",
    "outside_declared_workflow",
    "source_unreachable",
    "fortran_undefined_contract",
}


def load_manual_arm_rules(
    inventory: dict[str, Any], paths: list[Path]
) -> dict[str, dict[str, str]]:
    branches = {branch["id"]: branch for branch in inventory.get("branches", [])}
    arms = {arm["id"]: arm for arm in inventory.get("control_flow_arms", [])}
    reachable_arms = set(inventory.get("target_reachable_arm_ids", []))
    rules: dict[str, dict[str, str]] = {}
    for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1 or not isinstance(document.get("rules"), list):
            raise ValueError(f"{path}: expected schema_version 1 and a rules list")
        for rule in document["rules"]:
            branch_id = rule.get("branch_id")
            if branch_id not in branches:
                raise ValueError(f"{path}: unknown branch_id {branch_id!r}")
            decisions = rule.get("decisions")
            if not isinstance(decisions, dict) or not decisions:
                raise ValueError(f"{path}: {branch_id} requires decisions")
            for arm_name, decision in decisions.items():
                if isinstance(arm_name, bool):
                    arm_name = "true" if arm_name else "false"
                else:
                    arm_name = str(arm_name)
                arm_id = f"{branch_id}:{arm_name}"
                if arm_id not in arms:
                    raise ValueError(f"{path}: unknown arm {arm_id}")
                if arm_id not in reachable_arms:
                    raise ValueError(f"{path}: manual decision targets unreachable arm {arm_id}")
                if arm_id in rules:
                    raise ValueError(f"{path}: duplicate manual decision for {arm_id}")
                if not isinstance(decision, dict):
                    raise ValueError(f"{path}: {arm_id} decision must be a mapping")
                disposition = decision.get("disposition")
                reason = decision.get("reason")
                if disposition not in ALLOWED_MANUAL_DISPOSITIONS:
                    raise ValueError(f"{path}: {arm_id} invalid disposition {disposition!r}")
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(f"{path}: {arm_id} requires a source-backed reason")
                rules[arm_id] = {
                    "disposition": disposition,
                    "reason": reason,
                    "rule_file": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
                }
    return rules


def load_static_boolean_flags(
    path: Path = DEFAULT_CONFIG,
    *,
    static_edge_ledger: Path = DEFAULT_STATIC_EDGE_LEDGER,
) -> dict[str, bool]:
    flags, _ = load_static_boolean_context(path, static_edge_ledger=static_edge_ledger)
    return flags


def load_static_boolean_context(
    path: Path = DEFAULT_CONFIG,
    *,
    static_edge_ledger: Path = DEFAULT_STATIC_EDGE_LEDGER,
) -> tuple[dict[str, bool], dict[str, dict[str, bool]]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    flags: dict[str, bool] = {}
    for section in ("run_def_flags", "structural_overrides"):
        values = document.get(section, {})
        for name, value in values.items():
            if isinstance(value, bool):
                flags[str(name).lower()] = value
    edge_document = yaml.safe_load(static_edge_ledger.read_text(encoding="utf-8"))
    run_def = ROOT / edge_document["run_def"]
    conflicts: dict[str, dict[str, bool]] = {}
    for raw_line in run_def.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*([A-Za-z]+)", line)
        if match is None or match.group(2).lower() not in {"y", "n", "true", "false"}:
            continue
        name = match.group(1).lower()
        value = match.group(2).lower() in {"y", "true"}
        if name in flags and flags[name] != value:
            conflicts[name] = {"project_config": flags.pop(name), "materialized_run_def": value}
            continue
        if name in conflicts:
            continue
        flags[name] = value
    return flags, conflicts


def _if_condition(source: str) -> str | None:
    match = re.search(r"\b(?:else\s+)?if\s*\(", source, re.IGNORECASE)
    if match is None:
        return None
    start = match.end() - 1
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    return None


def evaluate_static_boolean_condition(source: str, flags: dict[str, bool]) -> bool | None:
    condition = _if_condition(source)
    if condition is None:
        return None
    expression = condition.lower()
    expression = re.sub(r"\.true\.", " True ", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\.false\.", " False ", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\.and\.", " and ", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\.or\.", " or ", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\.not\.", " not ", expression, flags=re.IGNORECASE)
    identifiers = set(re.findall(r"\b[a-z][a-z0-9_]*\b", expression, re.IGNORECASE))
    keywords = {"and", "or", "not", "true", "false"}
    if any(name not in flags and name not in keywords for name in identifiers):
        return None
    for name in sorted(flags, key=len, reverse=True):
        expression = re.sub(rf"\b{re.escape(name)}\b", str(flags[name]), expression)
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return None
    allowed = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.Constant)
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        return None
    value = eval(compile(tree, "<paper-static-condition>", "eval"), {"__builtins__": {}}, {})
    return value if isinstance(value, bool) else None


def _disposition(
    arm: dict[str, Any], row: dict[str, Any], static_flags: dict[str, bool]
) -> tuple[str, str, str]:
    resolution = row["resolution"]
    arm_name = arm["arm"]
    static_value = evaluate_static_boolean_condition(arm["source"], static_flags)
    if static_value is not None and arm_name in {"true", "false"}:
        arm_taken = (arm_name == "true") == static_value
        return (
            "classified",
            "paper_static_active" if arm_taken else "paper_static_inactive",
            "Pure boolean condition evaluated from the fixed paper run configuration.",
        )
    if resolution == "diagnostic_only":
        return (
            "classified",
            "diagnostic_taken" if arm_name in {"true", "case"} else "diagnostic_not_taken",
            "Source-text diagnostic guard; neither arm mutates the scientific state contract.",
        )
    if resolution == "valid_input_error_guard":
        return (
            "classified",
            "invalid_input_error" if arm_name in {"true", "case"} else "valid_input_continuation",
            "Source-text error guard; the error arm is outside valid scientific inputs and the other arm continues.",
        )
    if resolution == "process_inventory_classified":
        return (
            "classified",
            "valid_input_infrastructure",
            "The audited extension inventory has no open process gap for this infrastructure site.",
        )
    if resolution == "process_span_mapped":
        return (
            "candidate",
            "source_span_candidate",
            "A process span owns this site, but each arm still needs a source-backed reachability disposition.",
        )
    if resolution == "process_inventory_open_gap":
        return (
            "open",
            "inventory_gap",
            str(row.get("process_inventory_gap") or "The process inventory records an open gap."),
        )
    return (
        "open",
        "unmapped",
        "No process-span, infrastructure, diagnostic, or valid-input error classification owns this arm.",
    )


def build_arm_disposition(
    inventory: dict[str, Any],
    resolution: dict[str, Any],
    *,
    static_flags: dict[str, bool] | None = None,
    static_flag_conflicts: dict[str, dict[str, bool]] | None = None,
    manual_rules: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    static_flags = static_flags or {}
    manual_rules = manual_rules or {}
    arms = {arm["id"]: arm for arm in inventory.get("control_flow_arms", [])}
    rows = {row["control_flow_id"]: row for row in resolution.get("rows", [])}
    records: list[dict[str, Any]] = []
    missing_branch_rows: list[str] = []
    for arm_id in inventory.get("target_reachable_arm_ids", []):
        arm = arms[arm_id]
        row = rows.get(arm["branch_id"])
        if row is None:
            missing_branch_rows.append(arm_id)
            continue
        automatic_status, automatic_disposition, automatic_reason = _disposition(
            arm, row, static_flags
        )
        manual = manual_rules.get(arm_id)
        if manual is None:
            status, disposition, reason = (
                automatic_status,
                automatic_disposition,
                automatic_reason,
            )
            classification_source = "automatic"
        else:
            status = "classified"
            disposition = manual["disposition"]
            reason = manual["reason"]
            classification_source = manual["rule_file"]
        records.append(
            {
                "arm_id": arm_id,
                "branch_id": arm["branch_id"],
                "arm": arm["arm"],
                "file": arm["file"],
                "procedure": row.get("procedure"),
                "line": arm["line"],
                "source": arm["source"],
                "resolution": row["resolution"],
                "status": status,
                "disposition": disposition,
                "reason": reason,
                "classification_source": classification_source,
                "automatic_status": automatic_status,
                "automatic_disposition": automatic_disposition,
                "process_ledger_ids": row.get("process_ledger_ids", []),
                "process_inventory_id": row.get("process_inventory_id"),
                "scientific_source_owner_present": bool(row.get("process_ledger_ids"))
                or row["resolution"] in {"process_inventory_classified", "source_owner_mapped"},
            }
        )
    status_counts = Counter(record["status"] for record in records)
    disposition_counts = Counter(record["disposition"] for record in records)
    manual_original_status_counts = Counter(
        record["automatic_status"]
        for record in records
        if record["classification_source"] != "automatic"
    )
    denominator = len(inventory.get("target_reachable_arm_ids", []))
    return {
        "schema_version": 1,
        "reachable_arm_denominator": denominator,
        "recorded_arms": len(records),
        "missing_branch_rows": missing_branch_rows,
        "status_counts": dict(status_counts),
        "disposition_counts": dict(disposition_counts),
        "manual_rule_original_status_counts": dict(manual_original_status_counts),
        "closed": (
            len(records) == denominator
            and not missing_branch_rows
            and status_counts.get("candidate", 0) == 0
            and status_counts.get("open", 0) == 0
        ),
        "records": records,
        "note": (
            "Only narrow diagnostic, valid-input error, and gap-free infrastructure rules are "
            "automatically classified. Process-span ownership creates candidates, not closure."
        ),
        "static_boolean_flags": dict(sorted(static_flags.items())),
        "excluded_static_flag_conflicts": dict(sorted((static_flag_conflicts or {}).items())),
        "manual_rule_arms": len(manual_rules),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build source-backed PFT14 branch-arm dispositions.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--resolution", type=Path, default=DEFAULT_RESOLUTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    static_flags, static_flag_conflicts = load_static_boolean_context(args.config)
    manual_paths = sorted((ROOT / "docs" / "source_audits").glob(DEFAULT_MANUAL_RULE_GLOB))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    result = build_arm_disposition(
        inventory,
        json.loads(args.resolution.read_text(encoding="utf-8")),
        static_flags=static_flags,
        static_flag_conflicts=static_flag_conflicts,
        manual_rules=load_manual_arm_rules(inventory, manual_paths),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"WROTE {args.output.resolve()} denominator={result['reachable_arm_denominator']} "
        f"status={result['status_counts']} closed={result['closed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
