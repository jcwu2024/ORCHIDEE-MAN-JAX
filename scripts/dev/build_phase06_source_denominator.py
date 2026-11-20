"""Reconcile stable process identities with bounded, source-driven condition sites.

The lexical sites below are audit candidates, never a numerical-arm denominator.
The pure builder accepts supplied source text so portable controls need no local
Fortran checkout, scientific assets, JAX, compiler, or model execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
AUTHORITY = "outputs/reviews/main-physics-20261119/physics-branch-closeout/phase02-hydrol-thermal/coverage/current-coverage.json"
PHASE04 = "outputs/reviews/main-physics-20261119/physics-branch-closeout/phase04-hydrol-snow-boundaries/coverage.json"
PHASE05 = "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon"
OUTPUT = "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/denominator"
ID_SETS = (
    "process_entries", "source_owner_exact_spans", "process_extensions", "source_arms",
    "source_procedures", "source_branch_sites", "microcase_design_rows",
)
PROC_START = re.compile(r"^(?:RECURSIVE\s+|PURE\s+|ELEMENTAL\s+)?SUBROUTINE\s+(\w+)", re.I)
PROC_END = re.compile(r"^END\s*SUBROUTINE\b", re.I)
CALL = re.compile(r"\bCALL\s+(\w+)\s*(?:\(|$)", re.I)
IMPLICIT = re.compile(r"\b(MIN|MAX|MINVAL|MAXVAL|MINLOC|MAXLOC|MERGE|SIGN|ABS|SQRT|LOG10?|EXP)\s*\(", re.I)


def digest(value: Any) -> str:
    """Stable semantic identity, independent of JSON newline normalization."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def strip_comment(line: str) -> str:
    """Remove comments without treating an exclamation inside a string as one."""
    quote = None
    result = []
    i = 0
    while i < len(line):
        char = line[i]
        if quote:
            result.append(char)
            if char == quote:
                if i + 1 < len(line) and line[i + 1] == quote:
                    result.append(line[i + 1])
                    i += 1
                else:
                    quote = None
        elif char in ("'", '"'):
            quote = char
            result.append(char)
        elif char == "!":
            break
        else:
            result.append(char)
        i += 1
    return "".join(result)


def statements(text: str) -> list[dict[str, Any]]:
    """Join free-form continuations and keep exact physical source coordinates."""
    result = []
    pending = ""
    start = 0
    for number, raw in enumerate(text.splitlines(), 1):
        clean = strip_comment(raw).strip()
        if not clean or (clean.startswith("#") and not re.match(r"#\s*(?:if(?:def|ndef)?|elif|else|endif)\b", clean, re.I)):
            continue
        if not pending:
            start = number
        pending += " " + clean.lstrip("&").rstrip("&").strip()
        if clean.endswith("&"):
            continue
        # Semicolons outside literal strings separate executable statements.
        for part in re.split(r";(?=(?:[^'\"]|'[^']*'|\"[^\"]*\")*$)", pending.strip()):
            if part.strip():
                result.append({"start": start, "end": number, "text": part.strip()})
        pending = ""
    if pending:
        raise ValueError(f"unterminated source continuation at line {start}")
    return result


def procedures(text: str) -> dict[str, dict[str, Any]]:
    """Index bounded subroutines; source functions remain explicit call gaps."""
    result = {}
    active = None
    for statement in statements(text):
        begin = PROC_START.match(statement["text"])
        if begin:
            if active is not None:
                raise ValueError("nested subroutine needs an explicit source-scope decision")
            active = {"name": begin[1].lower(), "start": statement["start"], "statements": []}
        if active is not None:
            active["statements"].append(statement)
        if PROC_END.match(statement["text"]) and active is not None:
            active["end"] = statement["end"]
            if active["name"] in result:
                raise ValueError(f"duplicate local procedure {active['name']}")
            result[active["name"]] = active
            active = None
    if active is not None:
        raise ValueError(f"unterminated subroutine {active['name']}")
    return result


def spans(value: str | list[int]) -> list[tuple[int, int]]:
    if isinstance(value, list):
        return [(int(value[0]), int(value[-1]))]
    return [(int(pair[0]), int(pair[-1])) for part in value.split(",") if part.strip()
            for pair in [part.strip().split("-")]]


def intersects(first: int, last: int, ranges: list[tuple[int, int]]) -> bool:
    return any(first <= end and last >= start for start, end in ranges)


def without_strings(statement: str) -> str:
    return re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", "''", statement)


def site_kinds(statement: str) -> list[str]:
    """Explicit controls, loop domains, and hidden arithmetic selectors coexist."""
    text = without_strings(statement.strip())
    upper = text.upper()
    kinds = []
    if re.match(r"#\s*(?:if|ifdef|ifndef|elif|else)\b", upper, re.I):
        return ["compile_guard"]
    for pattern, kind in (
        (r"^ELSE\s*IF\s*\(", "elseif"), (r"^IF\s*\(", "if"),
        (r"^ELSEWHERE\b", "elsewhere"), (r"^WHERE\s*\(", "where"),
        (r"^SELECT\s*CASE\s*\(", "select_case"), (r"^CASE\b", "case"),
        (r"^ELSE\b", "else"), (r"^(?:\w+\s*:\s*)?DO\b", "loop_domain"),
    ):
        if re.search(pattern, upper):
            kinds.append(kind)
            break
    if IMPLICIT.search(text) and not re.match(r"^(?:END|CALL\s+(?:WRITE|XIOS)|WRITE|PRINT|IF.*WRITE)\b", upper):
        kinds.append("implicit_arithmetic")
    if "**" in text and "::" not in text and not upper.startswith(("WRITE", "PRINT")):
        kinds.append("power_domain")
    return kinds


def site_disposition(text: str, kind: str) -> tuple[str, str, list[str]]:
    """Conservative source classification; unknown guards stay required."""
    lower = text.lower()
    tags = []
    patterns = {
        "PFT_role_or_coordinate": r"\b(?:nvm|pft\w*|natural|is_tree|is_c4|is_peat|pasture|crop|jv|j)\b",
        "time_or_lifecycle": r"\b(?:firstcall\w*|lasttsday|last_ts\w*|day\w*|year\w*|month\w*|itime|itau\w*|istp|dt\w*)\b",
        "configuration": r"\b(?:ok_\w+|impose_\w+|read_\w+|river_routing|hydrol_cwrr|lpj_\w+|pheno_model|senescence_type)\b",
        "mask_or_compound": r"\.and\.|\.or\.|\bwhere\b|\bany\s*\(|\ball\s*\(",
        "arithmetic_boundary": r"\b(?:min|max|abs|sqrt|log|log10|exp|merge|sign)\s*\(|\*\*",
    }
    for tag, pattern in patterns.items():
        if re.search(pattern, lower):
            tags.append(tag)
    if re.match(r"^(?:if\s*\(\s*(?:printlev|long_print)|(?:write|print)\b)", lower):
        return "diagnostic_guard_source_only", "Confirm body has no scientific assignment before excluding execution; no numerical grade inferred.", tags
    if re.search(r"\b(?:allocated|associated)\s*\(|\bier\b|\bierr\b", lower):
        return "allocation_or_admission_guard", "Bind source allocation/error/initialization result; do not equate infrastructure with a numerical arm pass.", tags
    if kind == "compile_guard":
        return "compile_time_guard_not_runtime_arm", "Bind exact wrapper macro settings and preserve excluded fatal-contract build separately; no numerical branch inferred.", tags
    if kind == "loop_domain":
        return "loop_axis_domain", "Retain exact loop bounds/PFT/time order and use per-condition operands when claiming an internal arm.", tags
    return "source_condition_requires_scoped_witness", "Join exact source operands and admitted input boundary to a frozen oracle/component case; source mapping alone stays source_mapped.", tags


def guarded_statements(proc: dict[str, Any]) -> list[dict[str, Any]]:
    """Retain enclosing source gates; this is static scope, not taken-arm proof."""
    stack: list[dict[str, Any]] = []
    result = []
    for statement in proc["statements"]:
        text = without_strings(statement["text"]).upper()
        end = re.match(r"^END\s*(IF|WHERE|SELECT|DO)\b", text)
        end_block = end[1] if end else "CPP" if re.match(r"^#\s*ENDIF\b", text) else None
        if end_block:
            matching = [i for i, item in enumerate(stack) if item["block"] == end_block]
            if matching:
                stack = stack[:matching[-1]]
        branch = None
        if re.match(r"^ELSE\s*IF\b|^ELSE\b", text):
            branch = "IF"
        elif re.match(r"^ELSEWHERE\b", text):
            branch = "WHERE"
        elif re.match(r"^CASE\b", text):
            branch = "SELECT"
        elif re.match(r"^#\s*(?:ELIF|ELSE)\b", text):
            branch = "CPP"
        if branch:
            matching = [i for i, item in enumerate(stack) if item["block"] == branch]
            if matching:
                item = stack[matching[-1]]
                item.update(active_branch_line=statement["start"], active_branch_statement=statement["text"])
        result.append({**statement, "enclosing_source_controls": [dict(item) for item in stack]})
        block = None
        if re.match(r"^IF\s*\(", text) and re.search(r"\bTHEN\s*$", text):
            block = "IF"
        elif re.match(r"^WHERE\s*\(.*\)\s*$", text):
            block = "WHERE"
        elif re.match(r"^SELECT\s*CASE", text):
            block = "SELECT"
        elif re.match(r"^(?:\w+\s*:\s*)?DO\b", text):
            block = "DO"
        elif re.match(r"^#\s*IF(?:DEF|NDEF)?\b", text):
            block = "CPP"
        if block:
            stack.append({"block": block, "line": statement["start"], "statement": statement["text"],
                          "active_branch_line": statement["start"], "active_branch_statement": statement["text"]})
    return result


def coordinates(record: dict[str, Any], lane: str) -> list[tuple[str, int, int]]:
    """Extract explicit coordinates, never guess a broad condition from its label."""
    result = []
    for node in (record, record.get("source", {}), record.get("source_identity", {})):
        if not isinstance(node, dict):
            continue
        path = node.get("source_file", node.get("path", node.get("file")))
        pair = node.get("source_span", node.get("lines"))
        first = node.get("start_line", node.get("start"))
        last = node.get("end_line", node.get("end", first))
        if path and str(path).startswith("fortran_source/"):
            if isinstance(pair, (str, list)):
                result.extend((path, begin, end) for begin, end in spans(pair))
            elif isinstance(first, int):
                result.append((path, first, int(last)))
            elif isinstance(node.get("line"), int):
                result.append((path, node["line"], node["line"]))
    for field in ("id", "stable_id", "condition_id"):
        match = re.search(r"(fortran_source/[^:]+\.f90):(\d+)", str(record.get(field, "")))
        if match:
            result.append((match[1], int(match[2]), int(match[2])))
    if not result and lane == "daily-memory" and isinstance(record.get("source_span"), list):
        result.extend(("fortran_source/ORCHIDEE/src_stomate/stomate_season.f90", begin, end)
                      for begin, end in spans(record["source_span"]))
    return list(dict.fromkeys(result))


def validate_overlay(overlay: dict[str, Any], parent_ids: set[str], site_ids: set[str]) -> None:
    """A frozen scoped report cannot upgrade a route or use a dangling identity."""
    if not overlay.get("frozen") or not overlay.get("root_authorized"):
        raise ValueError("Phase06 overlay must be frozen and explicitly root-authorized")
    seen = set()
    for row in overlay.get("records", []):
        if row["id"] in seen:
            raise ValueError("duplicate overlay record")
        seen.add(row["id"])
        if not row.get("parent_ids") or not set(row["parent_ids"]) <= parent_ids:
            raise ValueError("overlay has unknown/missing stable parent")
        if not set(row.get("condition_ids", [])) <= site_ids:
            raise ValueError("overlay has unknown condition")
        if row.get("whole_route_closed") or row.get("all_arms_observed"):
            raise ValueError("finite overlay cannot promote whole route or all arms")
        if row.get("level") not in {"source_mapped", "jax_regression", "fortran_oracle", "production_verified"}:
            raise ValueError("unknown evidence grade")
        if not row.get("evidence") or not row.get("scope") or not row.get("remaining"):
            raise ValueError("overlay needs evidence, finite scope and remaining disposition")
        if row["level"] in {"fortran_oracle", "production_verified"}:
            if not row.get("same_input_boundary") or not row.get("independent_source_identity"):
                raise ValueError("source-only overlay cannot become numerical evidence")
        if row["level"] == "production_verified" and not row.get("actual_production_boundary"):
            raise ValueError("production evidence needs actual executed boundary")
        if row.get("condition_observed") and not (row.get("condition_ids") and row.get("predicate_operands") and row.get("case_ids")):
            raise ValueError("internal condition observation needs exact IDs, operands and cases")
        if row.get("natural_producer_verified") and not (row.get("natural_predecessor_identity") and row.get("actual_producer_boundary")):
            raise ValueError("component evidence cannot become natural-history evidence")


def validate_ledger(ledger: dict[str, Any], expected_ids: set[str], expected_sites: set[str]) -> None:
    """Detect dropped/duplicated identities and accidental source-to-pass upgrades."""
    parents = ledger["parents"]
    sites = ledger["conditions"]
    if len(parents) != len(expected_ids) or {row["stable_id"] for row in parents} != expected_ids:
        raise ValueError("parent denominator changed")
    if len(sites) != len(expected_sites) or {row["id"] for row in sites} != expected_sites:
        raise ValueError("source condition lost or duplicated")
    for row in parents:
        if not row.get("responsible_owner") or not row.get("next_minimum_action"):
            raise ValueError("parent lacks an owner/action")
        if row["route_wide_level"] != "source_mapped" or row["whole_route_closed"]:
            raise ValueError("finite evidence promoted a whole parent")
    for row in sites:
        if not row["parent_ids"] or not set(row["parent_ids"]) <= expected_ids:
            raise ValueError("source condition missing parent")
        if row["level"] != "source_mapped" or row["numerical_arm_observed"]:
            raise ValueError("source-only condition silently promoted")
        if not row.get("disposition") or not row.get("next_action"):
            raise ValueError("source condition is unclassified")
    candidates = ledger["historical_candidate_joins"]
    keys = {(row["parent_id"], row["candidate_site_id"]) for row in candidates}
    if len(keys) != len(candidates) or len(candidates) != ledger["summary"]["phase04_route_candidate_associations"]:
        raise ValueError("dropped or duplicate historical route/site association")
    for row in candidates:
        if row["parent_id"] not in expected_ids or not set(row["condition_ids"]) <= expected_sites:
            raise ValueError("orphan historical route/site association")
        if row["level"] != "source_mapped" or row["per_arm_execution_inferred"]:
            raise ValueError("historical route/site association falsely promoted")


def build_denominator(
    authority: dict[str, Any], phase04: dict[str, Any], phase05: dict[str, Any],
    reconciliation: dict[str, Any], source_texts: dict[str, str], *,
    expected_parent_count: int = 52, overlay: dict[str, Any] | None = None,
    phase05_remaining: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Join immutable authorities and scan their whole source procedures/callees."""
    parent_rows = authority["process_routes"]
    parent_ids = {row["stable_id"] for row in parent_rows}
    if len(parent_rows) != expected_parent_count or len(parent_ids) != expected_parent_count:
        raise ValueError("wrong or duplicate parent authority")
    histories = authority["all_seven_historical_stable_id_sets"]
    if set(histories) != set(ID_SETS):
        raise ValueError("historical seven ID sets changed")
    if set(histories["process_entries"]) != parent_ids:
        raise ValueError("historical process IDs disagree with authority")
    for ids in histories.values():
        if len(ids) != len(set(ids)):
            raise ValueError("historical ID set contains duplicates")
    old4 = {row["stable_id"]: row for row in phase04["pb03_records"]}
    old5 = {row["stable_id"]: row for row in phase05["process_routes"]}
    if not set(old4) <= parent_ids or not set(old5) <= parent_ids or set(old4) & set(old5):
        raise ValueError("phase04/05 stable route overlay mismatch")
    inherited_actions = (phase05_remaining or {}).get("actions", [])
    if len({row["id"] for row in inherited_actions}) != len(inherited_actions):
        raise ValueError("duplicate inherited remaining action")
    parsed = {path: procedures(text) for path, text in source_texts.items()}
    proc_parents: dict[tuple[str, str], set[str]] = defaultdict(set)
    source_gaps = []
    for row in parent_rows:
        path, name = row["fortran_file"], row["procedure"].lower()
        if path not in parsed or name not in parsed[path]:
            raise ValueError(f"missing original parent source procedure: {row['stable_id']}")
        proc_parents[path, name].add(row["stable_id"])
    # Resolve local source calls transitively. Unknown external/module calls are
    # retained separately; local source helper conditions cannot disappear merely
    # because the registered historical process span omitted their procedure.
    changed = True
    while changed:
        changed = False
        for (path, name), ids in list(proc_parents.items()):
            for statement in parsed[path][name]["statements"]:
                match = CALL.search(without_strings(statement["text"]))
                if match and match[1].lower() in parsed[path]:
                    target = proc_parents[path, match[1].lower()]
                    before = len(target)
                    target.update(ids)
                    changed |= len(target) != before
    sites = []
    calls = []
    for (path, name), route_ids in sorted(proc_parents.items()):
        proc = parsed[path][name]
        for statement in guarded_statements(proc):
            text = statement["text"]
            match = CALL.search(without_strings(text))
            if match:
                callee = match[1].lower()
                direct = [row["stable_id"] for row in parent_rows if row["fortran_file"] == path
                          and intersects(statement["start"], statement["end"], spans(row["source_span"]))]
                calls.append({"path": path, "line": statement["start"], "procedure": name,
                              "callee": callee, "parent_ids": sorted(set(direct) if direct else route_ids),
                              "registered_span_parent_ids": sorted(direct),
                              "enclosing_scope_review_parent_ids": sorted(route_ids),
                              "parent_join_role": "exact_registered_source_span" if direct else "source_review_responsibility_not_physical_route_membership",
                              "enclosing_source_controls": statement["enclosing_source_controls"],
                              "resolution": "same_file_source_closure" if callee in parsed[path] else "external_or_module_call_not_resolved_by_local_scan",
                              "next_action": "use existing exact owner evidence or resolve this callee before all-source closure; no coverage inferred from a call"})
            for kind in site_kinds(text):
                direct = [row["stable_id"] for row in parent_rows if row["fortran_file"] == path
                          and intersects(statement["start"], statement["end"], spans(row["source_span"]))]
                disposition, action, tags = site_disposition(text, kind)
                site = {"id": f"P06COND:{path}:{name}:{statement['start']}-{statement['end']}:{kind}",
                        "path": path, "procedure": name, "start": statement["start"], "end": statement["end"],
                        "kind": kind, "statement": text, "statement_semantic_sha256": digest(text),
                        "enclosing_source_controls": statement["enclosing_source_controls"],
                        "parent_ids": sorted(set(direct) if direct else route_ids), "registered_span_parent_ids": sorted(direct),
                        "enclosing_scope_review_parent_ids": sorted(route_ids),
                        "parent_join_role": "exact_registered_source_span" if direct else "source_review_responsibility_not_physical_route_membership",
                        "join": "registered_span" if direct else "whole_procedure_or_local_callee_expansion",
                        "tags": tags, "disposition": disposition, "next_action": action,
                        "level": "source_mapped", "numerical_arm_observed": False}
                sites.append(site)
    site_ids = {row["id"] for row in sites}
    if len(site_ids) != len(sites):
        raise ValueError("multiple source statements share an ambiguous stable site ID")
    condition_joins = []
    for inherited in reconciliation["records"]:
        original = inherited["original_record"]
        lane = inherited["view"].split("/")[0]
        coords = coordinates(original, lane)
        matches = [row for row in sites if any(row["path"] == path and intersects(row["start"], row["end"], [(first, last)])
                                             for path, first, last in coords)]
        explicit = [original.get(key) for key in ("process_id", "parent_process_id", "stable_process_id")]
        ids = {value for value in explicit if value in parent_ids}
        ids.update(route_id for row in matches for route_id in row["registered_span_parent_ids"])
        ids.update(row["stable_id"] for row in parent_rows for path, first, last in coords
                   if row["fortran_file"] == path and intersects(first, last, spans(row["source_span"])))
        review_ids = {route_id for (path, name), route_ids in proc_parents.items()
                      for coordinate_path, first, last in coords
                      if path == coordinate_path and intersects(first, last, [(parsed[path][name]["start"], parsed[path][name]["end"])])
                      for route_id in route_ids}
        exact_parent_join = bool(ids)
        if not ids:
            ids.update(route_id for row in matches for route_id in row["parent_ids"])
            ids.update(review_ids)
        if matches:
            join_status = "coordinate_overlap_not_arm_execution"
            action = "retain original finite report; add exact predicate operand witness before any arm claim"
        elif coords and all(re.match(r"^END\s*IF\b", source_texts[path].splitlines()[first - 1].strip(), re.I)
                            for path, first, last in coords):
            join_status = "legacy_END_IF_terminator_not_an_independent_predicate"
            action = "retain old lexical ID but do not invent a predicate/arm or numerical probe for this terminator"
        elif coords and ids:
            join_status = "finite_call_or_straight_line_boundary_no_direct_predicate"
            action = "retain original whole-tuple/caller boundary; no internal-arm observation inferred from this group"
        else:
            join_status = "group_or_legacy_identity_needs_exact_site_join"
            action = "reconcile the exact original record coordinates/procedure; preserve its grade without manufacturing a condition"
        condition_joins.append({"inherited_id": inherited["id"], "view": inherited["view"],
                                "source_coordinates": coords, "condition_ids": [row["id"] for row in matches],
                                "parent_ids": sorted(ids), "enclosing_review_only_parent_ids": sorted(review_ids - ids),
                                "parent_join_role": "exact_or_declared_source_association" if exact_parent_join else "source_review_responsibility_not_physical_route_membership",
                                "declared_disposition": inherited["declared_disposition"],
                                "evidence_or_exclusion": inherited.get("evidence_or_exclusion"),
                                "finite_missing_action": inherited.get("finite_missing_action"),
                                "join_status": join_status,
                                "per_arm_execution_inferred": False,
                                "next_action": action})
    historical_candidate_joins = []
    guarded_by_path = {path: [statement for proc in local.values() for statement in guarded_statements(proc)]
                       for path, local in parsed.items()}
    for route in authority.get("pb03_routes", []):
        pid, path = route["stable_id"], route["fortran_file"]
        if pid not in old4:
            raise ValueError("historical PB03 candidate route lacks Phase04 overlay")
        for candidate in route["source_candidate_sites"]:
            matches = [site for site in sites if site["path"] == path and site["start"] <= candidate["line"] <= site["end"]]
            current_statements = [statement for statement in guarded_by_path[path]
                                  if statement["start"] <= candidate["line"] <= statement["end"]]
            arm_ids = candidate["coincident_historical_arm_ids"]
            if not set(arm_ids) <= set(histories["source_arms"]):
                raise ValueError("historical PB03 candidate references unknown ARM identity")
            historical_candidate_joins.append({
                "parent_id": pid, "candidate_site_id": candidate["candidate_site_id"], "path": path,
                "line": candidate["line"], "historical_source_statement": candidate["source"],
                "historical_kinds": candidate["kinds"], "historical_arm_ids": arm_ids,
                "condition_ids": [site["id"] for site in matches],
                "current_source_statements": current_statements,
                "source_condition_record": old4[pid].get("source_condition_record"),
                "accepted_evidence_ids": old4[pid].get("accepted_evidence_ids", []),
                "join_status": "exact_source_coordinate_only" if matches else "ordered_assignment_or_mask_definition_not_standalone_scanned_control" if current_statements else "historical_site_needs_current_source_join",
                "level": "source_mapped", "per_arm_execution_inferred": False,
                "next_action": "Reuse Phase04 accepted finite cases and its current remaining action. Coordinate/ARM identity alone does not claim a taken arm; inspect the original exact witness before adding any probe.",
            })
    parents = []
    for row in parent_rows:
        pid = row["stable_id"]
        phase = "phase04" if pid in old4 else "phase05" if pid in old5 else "phase06_PB05_assignment"
        if pid in old4:
            evidence = {key: old4[pid].get(key) for key in ("route_wide_level", "cases", "accepted_evidence_ids", "phase04_evidence", "source_condition_record", "remaining_minimum_action")}
            current_action = old4[pid].get("remaining_minimum_action", row["next_minimum_action"])
        elif pid in old5:
            evidence = old5[pid]["phase05"]
            current_action = evidence.get("unclosed_target_action", row["next_minimum_action"])
        else:
            evidence = {"saved_evidence_level": row["saved_evidence_level"], "oracle_families": row["oracle_families"], "current_level_before": row["current_level"]}
            current_action = row["next_minimum_action"]
        if pid in old4:
            owner = "root/PB03 thermal or hydrol owner"
        elif pid in old5:
            owner = "root/" + str(old5[pid]["phase05"].get("lane", row["assignable_owner"]))
        elif any(word in pid for word in ("forcing_landpoint", "salinity_bbox", "tide_bbox", "paper_history_formula")):
            owner = "root/watchout_reference"
        else:
            owner = "root (restart/native-partial child collaborates with watchout_drag)"
        conditions = [site["id"] for site in sites if pid in site["parent_ids"]]
        parents.append({**{key: row[key] for key in ("stable_id", "group", "fortran_file", "procedure", "source_span", "source_condition", "jax_owner", "actual_caller", "inputs", "outputs", "writeback_phase")},
                        "historical_current_level": row["current_level"], "saved_evidence_level": row["saved_evidence_level"],
                        "route_overlay": phase, "responsible_owner": owner, "inherited_finite_evidence": evidence,
                        "route_wide_level": "source_mapped", "whole_route_closed": False,
                        "condition_ids": conditions, "next_minimum_action": current_action,
                        "current_action_authority": "phase04/remaining_minimum_action" if pid in old4 else "phase05/unclosed_target_action and inherited-remaining-actions.json" if pid in old5 else "phase06 PB05 assignment of unchanged authority action",
                        "remaining_action_context": {"role": "lane/common review context only, not per-route physical membership",
                                                     "lane_action_ids": [action["id"] for action in inherited_actions if action.get("lane") == evidence.get("lane")],
                                                     "common_action_ids": [action["id"] for action in inherited_actions if action.get("lane") == "inherited/common"]},
                        "source_only_not_numerical_denominator": True})
    joined = {site_id for row in condition_joins for site_id in row["condition_ids"]}
    joined4 = {site_id for row in historical_candidate_joins for site_id in row["condition_ids"]}
    source_gaps.extend({"id": row["id"], "parent_ids": row["parent_ids"], "path": row["path"],
                        "line": row["start"], "kind": row["kind"], "join": row["join"],
                        "gap_basis": "no Phase04 candidate or Phase05 record coordinate association; not absence of all old evidence",
                        "next_action": row["next_action"]} for row in sites if row["id"] not in joined | joined4)
    result = {"schema": "phase06-unified-source-denominator-v1", "parents": parents, "conditions": sites,
              "historical_id_sets": histories, "historical_id_sets_identity": {key: {"count": len(value), "semantic_sha256": digest(value)} for key, value in histories.items()},
              "historical_numeric_views_not_reestimated": {"numeric_arms": 4217, "associated_oracles": 824},
              "condition_joins": condition_joins, "historical_candidate_joins": historical_candidate_joins,
              "inherited_remaining_actions": inherited_actions, "source_scope_gaps": source_gaps, "call_edges": calls,
              "source_identities": {path: {"text_sha256": hashlib.sha256(text.encode()).hexdigest(), "lines": len(text.splitlines())} for path, text in source_texts.items()},
              "summary": {"parents": len(parents), "phase04": len(old4), "phase05": len(old5),
                          "previously_unassigned_parents": sorted(parent_ids - set(old4) - set(old5)),
                          "bounded_source_procedures": len(proc_parents), "source_candidate_sites": len(sites),
                          "sites_beyond_registered_parent_spans": sum(not row["registered_span_parent_ids"] for row in sites),
                          "inherited_reconciliation_records": len(condition_joins),
                          "inherited_records_without_site_coordinates_join": sum(not row["condition_ids"] for row in condition_joins),
                          "phase04_route_candidate_associations": len(historical_candidate_joins),
                          "phase04_unique_candidate_identities": len({row["candidate_site_id"] for row in historical_candidate_joins}),
                          "phase04_candidate_associations_without_current_site": sum(not row["condition_ids"] for row in historical_candidate_joins),
                          "phase04_assignment_or_mask_only_associations": sum(row["join_status"] == "ordered_assignment_or_mask_definition_not_standalone_scanned_control" for row in historical_candidate_joins),
                          "phase04_candidate_associations_without_current_statement": sum(not row["current_source_statements"] for row in historical_candidate_joins),
                          "candidate_sites_with_phase04_coordinate_join": len(joined4),
                          "candidate_sites_without_phase05_coordinate_join": len(sites) - len(joined),
                          "candidate_sites_without_inherited_coordinate_join": len(source_gaps),
                          "inherited_phase05_remaining_actions": len(inherited_actions),
                          "condition_kinds": dict(Counter(row["kind"] for row in sites)),
                          "source_denominator_complete": False, "numerical_acceptance": False,
                          "no_additive_denominator": True, "unassigned_parent_count": 0},
              "phase06_overlay": {"status": "await_root_frozen_reports", "records": []},
              "limits": ["local source-call closure is not cross-module caller completeness", "implicit arithmetic scan names and power operator do not enumerate every mathematical discontinuity", "coordinate overlap or full-tuple allclose cannot prove internal condition/arm observation", "old strict failures/external identities/natural producer limits are not overwritten"]}
    if overlay is not None:
        validate_overlay(overlay, parent_ids, site_ids)
        result["phase06_overlay"] = overlay
    validate_ledger(result, parent_ids, site_ids)
    return result


def remaining_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    """Group exact finite actions without converting lexical sites into arms."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    phase04_sites = {value for row in ledger["historical_candidate_joins"] for value in row["condition_ids"]}
    phase05_sites = {value for row in ledger["condition_joins"] for value in row["condition_ids"]}
    for row in ledger["conditions"]:
        groups[row["path"], row["procedure"], row["parent_join_role"], row["kind"], row["disposition"]].append(row)
    source_groups = []
    for key, rows in sorted(groups.items()):
        expanded = key[2] == "source_review_responsibility_not_physical_route_membership"
        action = ("First determine exact caller gates, PFT/axis role and configured-domain membership for the whole-procedure/local-helper expansion; responsibility IDs are not physical route reachability. "
                  if expanded else "Reuse accepted finite evidence and exact source-condition witnesses first. ")
        action += "Bind exact condition operands/cases only for a claimed internal arm, or document a source-backed exclusion. Do not rerun an accepted component solely because a lexical ID was added."
        source_groups.append({"id": "P06-SOURCE-GROUP:" + digest(key)[:16], "path": key[0], "procedure": key[1],
                              "parent_join_role": key[2], "kind": key[3], "disposition": key[4],
                              "responsible_parent_ids": sorted({pid for row in rows for pid in row["parent_ids"]}),
                              "condition_ids": [row["id"] for row in rows], "count": len(rows),
                              "source_line_bounds": [min(row["start"] for row in rows), max(row["end"] for row in rows)],
                              "phase04_coordinate_associated_count": sum(row["id"] in phase04_sites for row in rows),
                              "phase05_coordinate_associated_count": sum(row["id"] in phase05_sites for row in rows),
                              "level": "source_mapped", "next_action": action})
    calls: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ledger["call_edges"]:
        if row["resolution"] == "external_or_module_call_not_resolved_by_local_scan":
            calls[row["path"], row["procedure"], row["callee"]].append(row)
    call_groups = [{"path": key[0], "procedure": key[1], "callee": key[2],
                    "call_lines": [row["line"] for row in rows],
                    "review_parent_ids": sorted({pid for row in rows for pid in row["enclosing_scope_review_parent_ids"]}),
                    "next_action": "Join existing exact callee owner/producer evidence or source-backed infrastructure/exclusion contract. A lexical CALL is neither taken-call proof nor a missing numerical test by itself."}
                   for key, rows in sorted(calls.items())]
    return {"schema": "phase06-finite-remaining-source-actions-v1", "numerical_acceptance": False,
            "source_denominator_complete": False, "automatic_execution_authorized": False,
            "source_review_group_count": len(source_groups), "source_review_groups": source_groups,
            "unresolved_external_call_group_count": len(call_groups), "unresolved_external_call_groups": call_groups,
            "parent_current_actions": [{key: row[key] for key in ("stable_id", "responsible_owner", "current_action_authority", "next_minimum_action", "remaining_action_context")}
                                       for row in ledger["parents"]],
            "inherited_remaining_actions_file": "inherited-remaining-actions.json",
            "inherited_remaining_policy": "All32 Phase05 actions retain exact original statuses and bounded root resolutions; completed finite actions are not reopened by source expansion.",
            "infrastructure_and_compile_policy": "Confirm scientific writes, exact build macros, allocator/error contracts and external serialization identity before exclusion; preserve original undefined/fatal contracts.",
            "natural_producer_policy": "Finite synthetic component inputs, helper/factory creation and next-operand equality do not establish native producer or natural predecessor history."}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path(OUTPUT))
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    authority = read_json(root / AUTHORITY)
    phase04 = read_json(root / PHASE04)
    phase05 = read_json(root / PHASE05 / "coverage.json")
    reconciliation = read_json(root / PHASE05 / "source-condition-reconciliation.json")
    phase05_remaining = read_json(root / PHASE05 / "remaining.json")
    texts = {row["fortran_file"]: (root / row["fortran_file"]).read_text(encoding="utf-8", errors="replace")
             for row in authority["process_routes"]}
    overlay = read_json(args.overlay) if args.overlay else None
    result = build_denominator(authority, phase04, phase05, reconciliation, texts, overlay=overlay, phase05_remaining=phase05_remaining)
    for path, identity in result["source_identities"].items():
        identity.update(raw_sha256=hashlib.sha256((root / path).read_bytes()).hexdigest(),
                        decode_policy="UTF-8 errors=replace; raw byte identity retained independently")
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    artifact_keys = ("parents", "conditions", "historical_id_sets", "condition_joins", "historical_candidate_joins",
                     "inherited_remaining_actions", "source_scope_gaps", "call_edges", "source_identities")
    for key in artifact_keys:
        (output / f"{key.replace('_', '-')}.json").write_text(json.dumps(result[key], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "remaining.json").write_text(json.dumps(remaining_ledger(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {key: value for key, value in result.items() if key not in artifact_keys}
    compact["input_identities"] = [{"path": path, "semantic_sha256": digest(read_json(root / path))}
                                   for path in (AUTHORITY, PHASE04, PHASE05 + "/coverage.json", PHASE05 + "/source-condition-reconciliation.json", PHASE05 + "/remaining.json")]
    (output / "reconciliation.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
