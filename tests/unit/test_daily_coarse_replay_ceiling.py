from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver import orchestration as teacher  # noqa: E402
from research.daily_coarse_graining.replay_ceiling import (  # noqa: E402
    _compare_trees,
    _minimal_daily_fold,
    _ok_leak_with_restart_shape,
    _pack_dynamic_boundary,
    _unpack_dynamic_boundary,
    capture_pre_daily_stomate_record,
    replay_pre_daily_stomate_record,
)


def test_capture_and_replay_replace_only_fast_day_owners(monkeypatch):
    values = {
        "transition": object(),
        "daily": object(),
        "boundary": object(),
        "maintenance": np.asarray([1.0]),
        "ok_result": object(),
        "ok_updates": {"DOC": np.asarray([2.0])},
    }
    owner_calls = {
        name: 0
        for name in ("transition", "daily", "boundary", "maintenance", "ok_result")
    }

    def owner(name):
        def run(*args, **kwargs):
            owner_calls[name] += 1
            if name == "ok_result":
                return values["ok_result"], values["ok_updates"]
            return values[name]

        return run

    monkeypatch.setattr(teacher, "_paper_1961_later_day_half_hour_transition", owner("transition"))
    monkeypatch.setattr(teacher, "_paper_later_day_daily_process_from_completed_entries", owner("daily"))
    monkeypatch.setattr(teacher, "_paper_ok_leak_pre_step_boundary_from_entry_state", owner("boundary"))
    monkeypatch.setattr(teacher, "stomate_maintenance_respiration_parts_from_stacks", owner("maintenance"))
    monkeypatch.setattr(teacher, "_paper_half_hour_ok_leak_fold_from_entries", owner("ok_result"))

    def fake_day(*args, **kwargs):
        return SimpleNamespace(
            ready_for_day_end_state=True,
            transition=teacher._paper_1961_later_day_half_hour_transition(),
            daily=teacher._paper_later_day_daily_process_from_completed_entries(),
            boundary=teacher._paper_ok_leak_pre_step_boundary_from_entry_state(),
            maintenance=teacher.stomate_maintenance_respiration_parts_from_stacks(),
            ok=teacher._paper_half_hour_ok_leak_fold_from_entries(),
        )

    monkeypatch.setattr(teacher, "paper_1961_driver_later_day_runtime_result", fake_day)
    state = SimpleNamespace(tstep=47)
    record = capture_pre_daily_stomate_record(
        "config.yaml",
        previous_state=state,
        year=1961,
        day_index=2,
        start_tstep=48,
        use_compiled_sechiba_day=True,
    )
    assert all(count == 1 for count in owner_calls.values())
    assert record.ok_leak_updates is not values["ok_updates"]

    replay = replay_pre_daily_stomate_record(
        "config.yaml",
        previous_state=state,
        record=record,
    )
    assert all(count == 1 for count in owner_calls.values())
    assert replay.transition is values["transition"]
    assert replay.daily is values["daily"]
    assert replay.boundary is values["boundary"]
    assert replay.maintenance is values["maintenance"]
    assert replay.ok[0] is values["ok_result"]
    np.testing.assert_array_equal(replay.ok[1]["DOC"], values["ok_updates"]["DOC"])


def test_tree_comparison_uses_tolerance_for_float_and_exact_discrete():
    expected = {"x": np.asarray([1.0]), "mask": np.asarray([True, False])}
    close = {"x": np.asarray([1.0 + 1.0e-9]), "mask": np.asarray([True, False])}
    wrong_mask = {"x": np.asarray([1.0]), "mask": np.asarray([False, False])}

    assert _compare_trees(expected, close, atol=1.0e-8, rtol=1.0e-10)["passed"]
    assert not _compare_trees(expected, wrong_mask, atol=1.0e-8, rtol=1.0e-10)["passed"]


def test_dynamic_boundary_codec_separates_static_graph_from_numeric_arrays():
    @dataclass(frozen=True)
    class Boundary:
        state: np.ndarray
        nested: SimpleNamespace
        provenance: tuple[str, ...]

    first = Boundary(
        state=np.asarray([1.0, 2.0]),
        nested=SimpleNamespace(mask=np.asarray([True, False]), label="daily"),
        provenance=("source",),
    )
    second = Boundary(
        state=np.asarray([3.0, 4.0]),
        nested=SimpleNamespace(mask=np.asarray([False, True]), label="daily"),
        provenance=("source",),
    )
    first_spec, first_leaves = _pack_dynamic_boundary(first)
    second_spec, second_leaves = _pack_dynamic_boundary(second)

    assert first_spec == second_spec
    rebuilt = _unpack_dynamic_boundary(first_spec, second_leaves)
    np.testing.assert_array_equal(rebuilt.state, second.state)
    np.testing.assert_array_equal(rebuilt.nested.mask, second.nested.mask)
    assert rebuilt.nested.label == "daily"
    assert rebuilt.provenance == ("source",)
    assert len(first_leaves) == 2


def test_minimal_daily_fold_rebuilds_only_the_retained_tail_contract():
    daily_fields = {
        "gpp_daily": np.asarray([1.0]),
        "resp_maint_part": np.asarray([2.0]),
        "resp_maint_radia": np.asarray([3.0]),
        "flood_root_radia": np.asarray([4.0]),
    }

    fold = _minimal_daily_fold(daily_fields)

    assert fold.ok
    assert fold.daily_fields == daily_fields
    assert fold.accumulator.fields == {}
    assert fold.accumulator.step_results == ()
    assert fold.maintenance.step_results == ()
    np.testing.assert_array_equal(fold.maintenance.resp_maint_part, daily_fields["resp_maint_part"])


def test_ok_leak_restart_shape_is_reconstructed_from_existing_carry():
    previous_resp_hetero = np.asarray([[7.0]])
    end_state = SimpleNamespace(
        fields_by_component={
            "slowproc_stomate_previous_step_state": {
                "resp_hetero": previous_resp_hetero,
            }
        }
    )
    minimal = SimpleNamespace(
        littercalc=object(),
        soilcarbon=SimpleNamespace(
            carbon_32l=np.asarray([1.0]),
            doc=np.asarray([2.0]),
            perma_peat=SimpleNamespace(deepc_peat=np.asarray([3.0])),
        ),
        interception_storage=np.asarray([4.0]),
    )

    rebuilt = _ok_leak_with_restart_shape(minimal, end_state)

    assert rebuilt.soilcarbon.resp_hetero_soil is previous_resp_hetero
    assert rebuilt.soilcarbon.perma_peat is minimal.soilcarbon.perma_peat
