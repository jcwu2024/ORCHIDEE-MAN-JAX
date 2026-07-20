from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "scan_fortran_control_flow.py"
SPEC = importlib.util.spec_from_file_location("scan_fortran_control_flow", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SCAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCAN
SPEC.loader.exec_module(SCAN)


def test_comment_stripping_preserves_exclamation_in_strings():
    assert SCAN._code_without_comment("if (x == '!') call foo ! comment") == "if (x == '!') call foo "


def test_string_masking_prevents_false_call_records():
    assert "call" not in SCAN._code_without_strings("write(*,*) 'Cannot call STOMATE'").lower()


def test_external_call_provider_classifies_runtime_boundaries():
    assert SCAN._external_provider("random_number") == "fortran_intrinsic"
    assert SCAN._external_provider("restget") == "IOIPSL"
    assert SCAN._external_provider("mpi_bcast") == "MPI"
    assert SCAN._external_provider("xios_send_field") == "XIOS"
    assert SCAN._external_provider("missing_model_process") is None


def test_scan_builds_stable_procedure_call_and_branch_ids(tmp_path):
    source_root = tmp_path / "ORCHIDEE"
    source = source_root / "src_sechiba" / "sample.f90"
    source.parent.mkdir(parents=True)
    source.write_text(
        """subroutine outer(x)
  if (x > 0) call inner(x)
  where (x > 1) x = 1
  select case (x)
  case (1)
    call external_routine(x)
  end select
end subroutine outer
subroutine inner(x)
end subroutine inner
""",
        encoding="utf-8",
    )

    inventory = SCAN.scan_source(source_root, project_root=tmp_path, entry_names=("outer",))

    assert inventory["counts"]["procedures"] == 2
    assert inventory["counts"]["branches"] == 4
    assert inventory["counts"]["control_flow_arms"] == 6
    assert {call["callee_name"] for call in inventory["calls"]} == {"inner", "external_routine"}
    assert inventory["unresolved_callee_names"] == ["external_routine"]
    assert inventory["branches"][0]["id"] == "ORCHIDEE/src_sechiba/sample.f90:2:if"
    assert [arm["id"] for arm in inventory["control_flow_arms"]] == [
        "ORCHIDEE/src_sechiba/sample.f90:2:if:true",
        "ORCHIDEE/src_sechiba/sample.f90:2:if:false",
        "ORCHIDEE/src_sechiba/sample.f90:3:where:true",
        "ORCHIDEE/src_sechiba/sample.f90:3:where:false",
        "ORCHIDEE/src_sechiba/sample.f90:4:select_case:fallthrough",
        "ORCHIDEE/src_sechiba/sample.f90:5:case:case",
    ]
    assert all(
        arm["id"].startswith(f"{arm['branch_id']}:")
        for arm in inventory["control_flow_arms"]
    )
    assert inventory["counts"]["reachable_procedures"] == 2
    assert inventory["counts"]["reachable_branches"] == 4
    assert inventory["counts"]["target_reachable_arms"] == 6
    assert inventory["target_reachable_arm_ids"] == [
        arm["id"] for arm in inventory["control_flow_arms"]
    ]
    assert inventory["reachable_unresolved_callee_names"] == ["external_routine"]


def test_scan_emits_else_if_case_and_conservative_fallthrough_arms(tmp_path):
    source_root = tmp_path / "ORCHIDEE"
    source = source_root / "src_sechiba" / "arms.f90"
    source.parent.mkdir(parents=True)
    source.write_text(
        """subroutine entry(x)
  if (x > 1) then
  else if (x > 0) then
  end if
  elsewhere
  case default
end subroutine entry
""",
        encoding="utf-8",
    )

    inventory = SCAN.scan_source(source_root, project_root=tmp_path, entry_names=("entry",))

    arms_by_branch = {}
    for arm in inventory["control_flow_arms"]:
        arms_by_branch.setdefault(arm["branch_id"], []).append(arm["arm"])
    assert arms_by_branch == {
        "ORCHIDEE/src_sechiba/arms.f90:2:if": ["true", "false"],
        "ORCHIDEE/src_sechiba/arms.f90:3:else_if": ["true", "false"],
        "ORCHIDEE/src_sechiba/arms.f90:5:elsewhere": ["fallthrough"],
        "ORCHIDEE/src_sechiba/arms.f90:6:case": ["case"],
    }


def test_reachable_dynamic_dispatch_is_reported_as_call_graph_limitation(tmp_path):
    source_root = tmp_path / "ORCHIDEE"
    source = source_root / "src_driver" / "dynamic.f90"
    source.parent.mkdir(parents=True)
    source.write_text(
        """program driver
  procedure(step), pointer :: selected
  call object%advance()
end program driver
""",
        encoding="utf-8",
    )

    inventory = SCAN.scan_source(source_root, project_root=tmp_path, entry_names=("driver",))

    assert {site["kind"] for site in inventory["reachable_dynamic_dispatch_sites"]} == {
        "procedure_pointer",
        "type_bound_call",
    }
    assert inventory["call_graph_limitations"]


def test_scan_expands_continued_generic_module_procedure_interface(tmp_path):
    source_root = tmp_path / "ORCHIDEE"
    source = source_root / "src_driver" / "generic.f90"
    source.parent.mkdir(parents=True)
    source.write_text(
        """module generic_module
interface apply
  module procedure apply_scalar, &
                   & apply_vector
end interface
contains
subroutine apply_scalar(x)
end subroutine apply_scalar
subroutine apply_vector(x)
  if (x > 0) x = 0
end subroutine apply_vector
subroutine entry(x)
  call apply(x)
  write(*,*) 'do not call missing_from_string'
end subroutine entry
end module generic_module
""",
        encoding="utf-8",
    )

    inventory = SCAN.scan_source(source_root, project_root=tmp_path, entry_names=("entry",))

    assert inventory["generic_interfaces"]["apply"] == ["apply_scalar", "apply_vector"]
    assert inventory["counts"]["reachable_procedures"] == 3
    assert inventory["counts"]["reachable_branches"] == 1
    assert inventory["reachable_unresolved_callee_names"] == []


def test_disabled_call_edge_prunes_only_the_guarded_descendant(tmp_path):
    source_root = tmp_path / "ORCHIDEE"
    source = source_root / "src_driver" / "prune.f90"
    source.parent.mkdir(parents=True)
    source.write_text(
        """subroutine entry(x)
  if (x > 0) call disabled_child(x)
  call active_child(x)
end subroutine entry
subroutine disabled_child(x)
  if (x > 1) x = 1
end subroutine disabled_child
subroutine active_child(x)
  if (x > 2) x = 2
end subroutine active_child
""",
        encoding="utf-8",
    )
    edge = ("ORCHIDEE/src_driver/prune.f90", 2, "disabled_child")

    inventory = SCAN.scan_source(
        source_root,
        project_root=tmp_path,
        entry_names=("entry",),
        disabled_call_edges={edge},
    )

    reachable = set(inventory["reachable_procedure_ids"])
    assert any(value.endswith(":entry") for value in reachable)
    assert any(value.endswith(":active_child") for value in reachable)
    assert not any(value.endswith(":disabled_child") for value in reachable)
    assert inventory["applied_disabled_call_edges"] == [
        {"file": edge[0], "line": edge[1], "callee": edge[2]}
    ]
    assert inventory["unmatched_disabled_call_edges"] == []


def test_disabled_edge_loader_rejects_stale_run_def_assertion(tmp_path, monkeypatch):
    run_def = tmp_path / "used_run.def"
    run_def.write_text("SWITCH = n\n", encoding="utf-8")
    ledger = tmp_path / "edges.yaml"
    ledger.write_text(
        yaml.safe_dump(
            {
                "run_def": "used_run.def",
                "edges": [
                    {
                        "file": "a.f90",
                        "line": 1,
                        "callee": "child",
                        "config": "SWITCH=y",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(SCAN, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="disabled-edge config mismatch"):
        SCAN.load_disabled_call_edges(ledger)


def test_declared_function_reference_adds_function_to_reachable_graph(tmp_path):
    source_root = tmp_path / "ORCHIDEE"
    source = source_root / "src_driver" / "function_call.f90"
    source.parent.mkdir(parents=True)
    source.write_text(
        """subroutine entry(x)
  x = local_function(x)
end subroutine entry
function local_function(x)
  if (x > 0) x = 1
end function local_function
function unused_function(x)
  if (x > 1) x = 2
end function unused_function
""",
        encoding="utf-8",
    )

    inventory = SCAN.scan_source(source_root, project_root=tmp_path, entry_names=("entry",))

    reachable = set(inventory["reachable_procedure_ids"])
    assert any(value.endswith(":local_function") for value in reachable)
    assert not any(value.endswith(":unused_function") for value in reachable)
    assert inventory["counts"]["reachable_branches"] == 1
    assert inventory["counts"]["function_references"] == 1
