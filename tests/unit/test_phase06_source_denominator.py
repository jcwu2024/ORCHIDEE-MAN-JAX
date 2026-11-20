"""Portable source/reconciliation controls, with no scientific or external data."""
from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.dev.build_phase06_source_denominator import (
    ID_SETS,
    build_denominator,
    digest,
    guarded_statements,
    procedures,
    remaining_ledger,
    site_kinds,
    statements,
    validate_ledger,
    validate_overlay,
)

PATH = "fortran_source/ORCHIDEE/src_sechiba/portable.f90"
SOURCE = """SUBROUTINE surface
! IF (lost_condition) q = MAX(q, 1.) is only a comment
IF (printlev > 0) WRITE(*,*) 'IF(x)! MAX(q) CALL fake()'
IF (active .AND. &
    & pft(14) > 0) THEN
  q = MAX(0., MIN(q, 1.))
  CALL helper(q)
ELSE
  WHERE (mask) q = 0.
END IF
END SUBROUTINE surface
SUBROUTINE helper(q)
#ifdef STRICT_CHECK
IF (q < 0.) CALL fatal('bad! value')
#endif
DO j = 2,nvm
 q = SQRT(MAX(q,0.))**2
END DO
END SUBROUTINE helper
SUBROUTINE observer
IF (LastTsDay) CALL helper(q)
END SUBROUTINE observer
"""


@pytest.fixture
def inputs():
    rows = []
    for name, procedure, span in (("surface", "surface", "4-6"), ("observer", "observer", "21-21")):
        rows.append({"stable_id": f"PROC:fixture.{name}", "group": "PB-05", "fortran_file": PATH,
                     "procedure": procedure, "source_span": span, "source_condition": "explicit fixture condition",
                     "jax_owner": "fixture.owner", "actual_caller": "fixture.caller", "inputs": ["q"],
                     "outputs": ["q"], "writeback_phase": "fixture-next-input", "current_level": "inventoried",
                     "saved_evidence_level": "fortran_oracle", "oracle_families": ["historical_fixture"],
                     "assignable_owner": "fixture", "next_minimum_action": "exact condition operands remain required"})
    histories = {name: [f"LEGACY:{name}:1"] for name in ID_SETS}
    histories["process_entries"] = [row["stable_id"] for row in rows]
    authority = {"process_routes": rows, "all_seven_historical_stable_id_sets": histories}
    p4 = {"pb03_records": [{"stable_id": rows[0]["stable_id"], "route_wide_level": "source_mapped", "cases": ["fixture-case"]}]}
    p5 = {"process_routes": []}
    reconciled = {"records": [{"id": "fixture:old-end-if", "view": "canopy/source_scan", "declared_disposition": "inventory_only",
                              "original_record": {"path": PATH, "line": 10}},
                             {"id": "fixture:old-component-pass", "view": "canopy/rows", "declared_disposition": "pass",
                              "evidence_or_exclusion": "only named whole-tuple output", "original_record": {"path": PATH, "line": 6}}]}
    return authority, p4, p5, reconciled, {PATH: SOURCE}


def build(inputs):
    return build_denominator(*inputs, expected_parent_count=2)


def test_continuation_strings_comments_and_compile_guards():
    rows = statements(SOURCE)
    condition = next(row for row in rows if row["start"] == 4)
    assert condition["end"] == 5 and ".AND." in condition["text"]
    assert "IF(x)! MAX(q) CALL fake()" in rows[1]["text"]
    assert site_kinds(rows[1]["text"]) == ["if"]
    assert site_kinds("END IF") == []
    assert site_kinds("#ifdef STRICT_CHECK") == ["compile_guard"]
    assert set(procedures(SOURCE)) == {"surface", "helper", "observer"}


def test_alternate_branches_and_compile_gates_keep_their_actual_source_selector():
    source = """SUBROUTINE controls
IF (read_lai) THEN
 q = 1
ELSEIF (other) THEN
 q = 2
ELSE
 q = 3
ENDIF
#ifdef STRICT_CHECK
 q = 4
#else
 q = 5
#endif
q = 6
END SUBROUTINE controls
"""
    rows = {row["text"]: row for row in guarded_statements(procedures(source)["controls"])}
    assert rows["q = 2"]["enclosing_source_controls"][0]["active_branch_statement"] == "ELSEIF (other) THEN"
    assert rows["q = 3"]["enclosing_source_controls"][0]["active_branch_statement"] == "ELSE"
    assert rows["q = 4"]["enclosing_source_controls"][0]["active_branch_statement"] == "#ifdef STRICT_CHECK"
    assert rows["q = 5"]["enclosing_source_controls"][0]["active_branch_statement"] == "#else"
    assert rows["q = 6"]["enclosing_source_controls"] == []


def test_all_histories_preserved_and_whole_component_does_not_promote_arms(inputs):
    ledger = build(inputs)
    assert ledger["historical_id_sets"] == inputs[0]["all_seven_historical_stable_id_sets"]
    assert all(row["level"] == "source_mapped" and not row["numerical_arm_observed"] for row in ledger["conditions"])
    assert all(not row["whole_route_closed"] for row in ledger["parents"])
    assert ledger["condition_joins"][0]["join_status"] == "legacy_END_IF_terminator_not_an_independent_predicate"
    assert ledger["condition_joins"][1]["declared_disposition"] == "pass"
    assert not ledger["condition_joins"][1]["per_arm_execution_inferred"]
    assert all(row["callee"] != "fake" for row in ledger["call_edges"])


def test_local_helper_hidden_bounds_and_axis_domain_cannot_disappear(inputs):
    ledger = build(inputs)
    helper = [row for row in ledger["conditions"] if row["procedure"] == "helper"]
    assert {"compile_guard", "if", "loop_domain", "implicit_arithmetic", "power_domain"} <= {row["kind"] for row in helper}
    assert all(row["parent_ids"] for row in helper)
    assert all(row["join"] == "whole_procedure_or_local_callee_expansion" for row in helper)
    call = next(row for row in ledger["call_edges"] if row["procedure"] == "surface" and row["callee"] == "helper")
    assert call["enclosing_source_controls"][0]["line"] == 4


def test_phase04_site_ids_preserved_without_taken_arm_or_stale_action(inputs):
    data = deepcopy(inputs)
    pid = data[0]["process_routes"][0]["stable_id"]
    arm = f"ARM:{PATH}:4:if:true"
    data[0]["all_seven_historical_stable_id_sets"]["source_arms"].append(arm)
    data[0]["pb03_routes"] = [{"stable_id": pid, "fortran_file": PATH, "source_candidate_sites": [
        {"candidate_site_id": f"SITE:{PATH}:4", "line": 4, "source": "IF (active .AND. &",
         "kinds": ["if_or_elseif_candidate"], "coincident_historical_arm_ids": [arm]},
    ]}]
    data[1]["pb03_records"][0].update(remaining_minimum_action="No further probe for accepted finite attribution.",
                                        source_condition_record={"report": "fixture-current-source-report", "stable_id": pid})
    result = build(data)
    join = result["historical_candidate_joins"][0]
    assert join["historical_arm_ids"] == [arm]
    assert join["candidate_site_id"] == f"SITE:{PATH}:4"
    assert join["condition_ids"] and not join["per_arm_execution_inferred"]
    assert join["source_condition_record"]["report"] == "fixture-current-source-report"
    assert result["summary"]["phase04_route_candidate_associations"] == 1
    assert result["parents"][0]["next_minimum_action"] == "No further probe for accepted finite attribution."
    data[0]["pb03_routes"][0]["source_candidate_sites"][0]["coincident_historical_arm_ids"] = ["ARM:unknown"]
    with pytest.raises(ValueError, match="unknown ARM"):
        build(data)


def test_coordinate_parent_is_separate_from_enclosing_review_responsibility(inputs):
    data = deepcopy(inputs)
    data[0]["process_routes"][1].update(procedure="surface", source_span="8-8")
    result = build(data)
    join = result["condition_joins"][1]
    assert join["parent_ids"] == [data[0]["process_routes"][0]["stable_id"]]
    assert join["enclosing_review_only_parent_ids"] == [data[0]["process_routes"][1]["stable_id"]]
    assert join["parent_join_role"] == "exact_or_declared_source_association"
    call = next(row for row in result["call_edges"] if row["procedure"] == "surface")
    assert call["parent_join_role"] == "source_review_responsibility_not_physical_route_membership"


def test_phase05_current_actions_keep_completed_finite_resolution(inputs):
    data = deepcopy(inputs)
    pid = data[0]["process_routes"][1]["stable_id"]
    data[2]["process_routes"] = [{"stable_id": pid, "phase05": {"lane": "fixture-lane", "unclosed_target_action": "Current scoped action"}}]
    remaining = {"actions": [
        {"id": "old-finite-action", "lane": "fixture-lane", "action": "Original finite action", "root_status": "bounded_action_completed"},
        {"id": "common-context", "lane": "inherited/common", "action": "Preserve exact limits"},
    ]}
    result = build_denominator(*data, expected_parent_count=2, phase05_remaining=remaining)
    assert result["parents"][1]["next_minimum_action"] == "Current scoped action"
    assert result["parents"][1]["remaining_action_context"]["lane_action_ids"] == ["old-finite-action"]
    assert result["inherited_remaining_actions"] == remaining["actions"]
    remaining["actions"].append(deepcopy(remaining["actions"][0]))
    with pytest.raises(ValueError, match="duplicate inherited remaining"):
        build_denominator(*data, expected_parent_count=2, phase05_remaining=remaining)


def test_finite_remaining_groups_preserve_every_site_and_source_role(inputs):
    ledger = build(inputs)
    remaining = remaining_ledger(ledger)
    grouped = [site for group in remaining["source_review_groups"] for site in group["condition_ids"]]
    assert set(grouped) == {row["id"] for row in ledger["conditions"]}
    assert len(grouped) == len(set(grouped))
    assert any(group["parent_join_role"] == "source_review_responsibility_not_physical_route_membership"
               for group in remaining["source_review_groups"])
    assert all(group["level"] == "source_mapped" for group in remaining["source_review_groups"])
    assert not remaining["automatic_execution_authorized"]
    assert not remaining["numerical_acceptance"]


def test_historical_ordered_assignment_is_not_manufactured_predicate(inputs):
    data = deepcopy(inputs)
    data[4][PATH] = SOURCE.replace("! IF (lost_condition) q = MAX(q, 1.) is only a comment", "q = q")
    pid = data[0]["process_routes"][0]["stable_id"]
    data[0]["pb03_routes"] = [{"stable_id": pid, "fortran_file": PATH, "source_candidate_sites": [
        {"candidate_site_id": f"SITE:{PATH}:2", "line": 2, "source": "q = q",
         "kinds": ["ordered_assignment_or_masked_array_candidate"], "coincident_historical_arm_ids": []},
    ]}]
    ledger = build(data)
    join = ledger["historical_candidate_joins"][0]
    assert not join["condition_ids"]
    assert join["current_source_statements"][0]["text"] == "q = q"
    assert join["join_status"] == "ordered_assignment_or_mask_definition_not_standalone_scanned_control"
    assert not join["per_arm_execution_inferred"]
    ledger["historical_candidate_joins"].clear()
    with pytest.raises(ValueError, match="dropped or duplicate historical"):
        validate_ledger(ledger, {row["stable_id"] for row in ledger["parents"]}, {row["id"] for row in ledger["conditions"]})


@pytest.mark.parametrize("mutation", ["drop_condition", "duplicate_condition", "promote_condition", "promote_parent", "orphan_condition", "drop_parent"])
def test_negative_controls_reject_lost_conditions_and_false_grades(inputs, mutation):
    ledger = build(inputs)
    parent_ids = {row["stable_id"] for row in ledger["parents"]}
    site_ids = {row["id"] for row in ledger["conditions"]}
    if mutation == "drop_condition":
        ledger["conditions"].pop()
    elif mutation == "duplicate_condition":
        ledger["conditions"].append(deepcopy(ledger["conditions"][0]))
    elif mutation == "promote_condition":
        ledger["conditions"][0]["level"] = "fortran_oracle"
    elif mutation == "promote_parent":
        ledger["parents"][0]["whole_route_closed"] = True
    elif mutation == "orphan_condition":
        ledger["conditions"][0]["parent_ids"] = ["PROC:unknown"]
    else:
        ledger["parents"].pop()
    with pytest.raises(ValueError):
        validate_ledger(ledger, parent_ids, site_ids)


@pytest.mark.parametrize("mutation", ["historical_set", "duplicate_parent", "missing_source", "overlay_overlap"])
def test_authority_identity_negative_controls(inputs, mutation):
    data = deepcopy(inputs)
    if mutation == "historical_set":
        data[0]["all_seven_historical_stable_id_sets"].pop("source_arms")
    elif mutation == "duplicate_parent":
        data[0]["process_routes"][1] = data[0]["process_routes"][0]
    elif mutation == "missing_source":
        data[4].clear()
    else:
        data[2]["process_routes"] = [{"stable_id": data[0]["process_routes"][0]["stable_id"]}]
    with pytest.raises(ValueError):
        build(data)


def overlay_record(ledger):
    return {"id": "frozen-fixture-case", "parent_ids": [ledger["parents"][0]["stable_id"]],
            "condition_ids": [ledger["conditions"][0]["id"]], "level": "fortran_oracle",
            "evidence": ["frozen-fixture-reference"], "scope": "same exact finite complete tuple",
            "remaining": "other internal arms and natural predecessors not observed",
            "same_input_boundary": "full component input/output tuple", "independent_source_identity": "fixture-original-source"}


def test_frozen_scoped_overlay_keeps_source_grade_separate(inputs):
    before = build(inputs)
    overlay = {"frozen": True, "root_authorized": True, "records": [overlay_record(before)]}
    after = build_denominator(*inputs, expected_parent_count=2, overlay=overlay)
    assert after["phase06_overlay"] == overlay
    assert after["conditions"] == before["conditions"]
    assert after["parents"] == before["parents"]


@pytest.mark.parametrize("mutation", ["unfrozen", "unapproved", "source_only", "whole_route", "dangling_parent", "dangling_condition", "fake_natural", "fake_condition_observation"])
def test_overlay_cannot_upgrade_source_only_or_fake_natural_history(inputs, mutation):
    ledger = build(inputs)
    overlay = {"frozen": True, "root_authorized": True, "records": [overlay_record(ledger)]}
    record = overlay["records"][0]
    if mutation == "unfrozen":
        overlay["frozen"] = False
    elif mutation == "unapproved":
        overlay["root_authorized"] = False
    elif mutation == "source_only":
        record.pop("independent_source_identity")
    elif mutation == "whole_route":
        record["whole_route_closed"] = True
    elif mutation == "dangling_parent":
        record["parent_ids"] = ["PROC:missing"]
    elif mutation == "dangling_condition":
        record["condition_ids"] = ["P06COND:missing"]
    elif mutation == "fake_natural":
        record["natural_producer_verified"] = True
    else:
        record["condition_observed"] = True
    with pytest.raises(ValueError):
        validate_overlay(overlay, {row["stable_id"] for row in ledger["parents"]}, {row["id"] for row in ledger["conditions"]})


def test_semantic_id_ignores_json_newline_normalization():
    assert digest({"a": 1, "b": [2, 3]}) == digest({"b": [2, 3], "a": 1})
    assert digest({"a": 1}) != digest({"a": 2})
