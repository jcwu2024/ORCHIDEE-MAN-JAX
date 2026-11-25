"""Public R2 context handoff; physics and clock progression are delegates.

Real producer/consumer admission, return bookkeeping, source forcing reads and
SWnet phase mapping execute. No physical model step/day/year executes. The cold
delegate retains real one-entry admission/retirement, then supplies synthetic
successful daily fields solely for the public outer handoff contract.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import xarray as xr
from netCDF4 import Dataset

from jax_orchidee.driver import orchestration as o
from jax_orchidee.driver.sechiba_boundary import build_intersurf_first_step_payload
from tests.unit.test_watchout_paper_compiled_transport import _clock_payload, _clock_runtime_at, _prepared_case
from tests.unit.test_watchout_run_context_repair import _handoff_packet, _immutable, _restart_delegates

EVIDENCE = {"inputs": [], "chains": [], "missing_history": [], "producer_edges": [], "cold_empty": []}


class _Handoff:
    def __init__(self, tmp_path, monkeypatch, *, family="native", watchout=True, outcome="success"):
        self.family = family
        self.watchout = watchout
        self.outcome = outcome
        self.phase = "producer"
        self.prepares = []
        self.observed = []
        self.cold_admission = []
        fresh = _prepared_case(tmp_path, monkeypatch, watchout=watchout, coupled=True)
        fresh.cwrr_grid = NS(znt=np.ones(2), zlt=np.ones(2))
        fresh.run_def_values["STOMATE_OK_DGVM"] = "FALSE"
        if not watchout:
            # Source permits an explicit true flag with non-WATCHOUT zero drag.
            # These tests stop at context/mapping and never enter its division.
            fresh.run_def_values["CDRAG_from_GCM"] = "TRUE"
        config = fresh.config_path.read_text(encoding="utf-8")
        # The original helper already loaded its YAML cache. Use a distinct
        # immutable config identity rather than editing a cached input in place.
        fresh.config_path = tmp_path / "handoff_case.yaml"
        fresh.config_path.write_text(config.replace("forcing_1961.nc", "forcing_{year}.nc"), encoding="utf-8")
        original_path = tmp_path / "forcing_1961.nc"
        with xr.open_dataset(original_path) as opened:
            native = opened.load()
        native["cdrag"].values[:] = 0.
        native.to_netcdf(tmp_path / "forcing_1962.nc")
        native.close()
        (tmp_path / "forcing_1963.nc").write_bytes(original_path.read_bytes())
        for year in (1962, 1963):
            with Dataset(tmp_path / f"forcing_{year}.nc", "a") as dataset:
                dataset.variables["tstep"].calendar = "noleap"
                dataset.variables["tstep"].units = "seconds since 1961-01-01 00:00:00"
        self.fresh = _immutable(fresh)
        self.expected_mode = not watchout
        self.reference = replace(self.fresh, watchout_initialize_main_tstep=0,
            diffuco_pft14_static_kwargs={**self.fresh.diffuco_pft14_static_kwargs,
                "ldq_cdrag_from_gcm": self.expected_mode})
        candidate_step = o._paper_1961_step_bundle_from_context(self.fresh, year=1961, tstep=0)
        candidate_mode = bool(o._paper_initial_drag_mode(self.fresh.run_def_values, candidate_step.forcing))
        assert candidate_mode is True
        EVIDENCE["inputs"].append({"family": family, "watchout": watchout, "outcome": outcome,
            "forcing_sha256": {str(year): hashlib.sha256((tmp_path / f"forcing_{year}.nc").read_bytes()).hexdigest()
                for year in (1961, 1962, 1963)},
            "native1962_drag": 0., "historical_mode": self.expected_mode,
            "actual1961_initialize_drag": None if candidate_step.forcing.cdrag is None
                else np.asarray(candidate_step.forcing.cdrag).tolist(),
            "actual1961_candidate_mode": candidate_mode})
        _restart_delegates(monkeypatch)
        monkeypatch.setattr(o, "prepare_paper_1961_driver_context", self.prepare)
        monkeypatch.setattr(o, "paper_1961_driver_later_day_runtime_result", self.daily)
        monkeypatch.setattr(o, "_paper_run_compiled_later_day_blocks", self.block)
        monkeypatch.setattr(o, "paper_1961_driver_day_scaffold", self.reference_first)
        monkeypatch.setattr(o, "_driver_day_modelout", lambda first, **kw:
            NS(day_index=kw["day_index"]) if first.ready_for_first_day_end_state else None)
        monkeypatch.setattr(o, "_driver_modelout_from_outputs", lambda outputs, **kw:
            None if outputs is None else NS(day_index=kw["day_index"]))
        self._cold_delegates(monkeypatch)

    def prepare(self, *args, **kwargs):
        self.prepares.append(self.phase)
        return self.reference if self.phase == "producer" and self.family == "standard" else self.fresh

    def record(self, context, *, year, boundary):
        payload = build_intersurf_first_step_payload(o._paper_1961_step_bundle_from_context(context, year=year, tstep=0))
        fields = o._enerbil_driver_coverage_payload(payload,
            initializing=o._paper_watchout_initializing(context, 0),
            driver_albedo_payload={"albedo": np.array([[.3, .5]])})
        result = {"phase": self.phase, "boundary": boundary, "mode": o._paper_context_drag_mode(context),
            "marker": context.watchout_initialize_main_tstep, "watchout": payload.is_watchout,
            "actual_swnet": None if "swnet" not in fields else np.asarray(fields["swnet"]).tolist(),
            "file_swnet": None if payload.swnet is None else np.asarray(payload.swnet).tolist(),
            "actual_drag": None if payload.cdrag is None else np.asarray(payload.cdrag).tolist()}
        self.observed.append(result)
        return result

    def daily(self, *args, **kwargs):
        day = kwargs["day_index"]
        context = kwargs["prepared_context"]
        self.record(context, year=kwargs["year"], boundary=f"day{day}")
        fail = self.phase == "producer" and (
            self.outcome == "first-fail" or self.outcome == "later-fail" and day == 2)
        return o.DriverRuntimeDayResult(year=kwargs["year"], day_index=day,
            start_tstep=(day - 1) * 48, steps_per_stomate=48,
            daily_modelout=None if fail else NS(day_index=day),
            day_end_state=None if fail else replace(_handoff_packet(), tstep=day * 48 - 1,
                source_clock_runtime=_clock_runtime_at(context, local_tstep=day * 48 - 1, source_istp=day * 48)),
            stopped_at_tstep=0 if fail else None, missing_components=("delegated_science_gap",) if fail else ())

    def block(self, *args, **kwargs):
        self.record(kwargs["context"], year=kwargs["year"], boundary="block2-3")
        kwargs["daily_modelout"].extend((NS(day_index=2), NS(day_index=3)))
        return replace(_handoff_packet(), tstep=143,
            source_clock_runtime=_clock_runtime_at(kwargs["context"], local_tstep=143, source_istp=144)), 4

    def reference_first(self, *args, **kwargs):
        self.record(kwargs["prepared_context"], year=kwargs["year"], boundary="reference-first")
        ready = self.outcome != "first-fail"
        return NS(initial_state_mode=o.REFERENCE_CASE_INITIAL_STATE_MODE,
            first_day_end_state=replace(_handoff_packet(), source_clock_runtime=_clock_runtime_at(
                kwargs["prepared_context"], local_tstep=47, source_istp=48)) if ready else None,
            ready_for_first_day_end_state=ready, missing_components=() if ready else ("delegated_science_gap",))

    def _cold_delegates(self, monkeypatch):
        monkeypatch.setattr(o, "_require_source_day_alignment", lambda *a, **kw: None)
        monkeypatch.setattr(o, "_driver_source_clock_payload", _clock_payload)
        monkeypatch.setattr(o, "_filter_day_state_gaps", lambda *a: ())
        monkeypatch.setattr(o, "_calendar_snapshot_before_stomate", lambda *a: None)
        monkeypatch.setattr(o, "_paper_cold_start_day_daily_process_from_completed_entries", lambda **kw: None)

        def coverage(*args, **kwargs):
            self.cold_admission.append(self.record(kwargs["prepared_context"], year=kwargs["year"], boundary="cold-admission"))
            initialized = {} if self.outcome == "first-fail" else {
                "cold_start_first_step_previous_state": replace(_handoff_packet(), tstep=0),
                "cold_start_first_step_entry_payload": {}}
            return NS(initialized_payloads=initialized, missing_components=("delegated_science_gap",)
                if not initialized else (), provenance=())

        monkeypatch.setattr(o, "paper_1961_driver_cold_start_first_step_coverage", coverage)
        self.real_cold = o.paper_1961_driver_cold_start_day_scaffold

        def cold(*args, **kwargs):
            result = self.real_cold(*args, **kwargs, max_steps=1)
            if self.outcome == "first-fail":
                return result
            # Science completion is an explicit delegate. Admission, its first
            # entry and returned retired context above are production behavior.
            return replace(result, completed_steps=48, daily_process_ready=True,
                first_day_end_state=replace(_handoff_packet(), source_clock_runtime=_clock_runtime_at(
                    kwargs["prepared_context"], local_tstep=47, source_istp=48)), stomate_outputs=NS())

        monkeypatch.setattr(o, "paper_1961_driver_cold_start_day_scaffold", cold)

    def produce(self, ndays=1):
        common = dict(year=1962, ndays=ndays)
        if self.family == "standard":
            return o.paper_1961_driver_multiday_modelout_lite_run(self.fresh.config_path,
                **common, compact_later_days=True)
        if self.family == "cold":
            return o.paper_1961_driver_cold_start_multiday_modelout_lite_run(self.fresh.config_path, **common)
        return o.paper_1961_driver_restart_year_multiday_modelout_lite_run(self.fresh.config_path,
            **common, previous_year_end_state=replace(_handoff_packet(), source_clock_runtime=
                _clock_runtime_at(self.fresh, local_tstep=47, source_istp=48)))

    def consume(self, produced, entry):
        self.phase = "consumer"
        common = dict(year=1963, previous_year_end_state=produced.last_day_end_state,
            prepared_context=produced.prepared_context, run_mode="continuous")
        if entry == "year-start":
            return o.paper_1961_driver_restart_year_start_day_result(self.fresh.config_path, **common)
        return o.paper_1961_driver_restart_year_multiday_modelout_lite_run(self.fresh.config_path,
            **common, ndays=3 if entry == "outer-block" else 2,
            use_compiled_sechiba_day=entry == "outer-block", compiled_day_block_size=2 if entry == "outer-block" else 0)


@pytest.mark.parametrize("family", ["standard", "cold", "native"])
@pytest.mark.parametrize("watchout", [True, False], ids=["watchout-history-false", "nonwatch-history-true"])
@pytest.mark.parametrize("entry", ["year-start", "outer-day2", "outer-block"])
def test_real_public_producer_to_continuous_consumer(tmp_path, monkeypatch, family, watchout, entry):
    harness = _Handoff(tmp_path, monkeypatch, family=family, watchout=watchout)
    produced = harness.produce()
    prior = produced.prepared_context
    record = {"family": family, "entry": entry, "watchout": watchout,
        "producer_ready": produced.ready_for_requested_days,
        "producer_observations": list(harness.observed),
        "returned_history": prior is not None,
        "returned_mode": None if prior is None else o._paper_context_drag_mode(prior),
        "returned_marker": None if prior is None else prior.watchout_initialize_main_tstep}
    EVIDENCE["chains"].append(record)
    assert prior is not None
    assert o._paper_context_drag_mode(prior) is harness.expected_mode
    assert prior.watchout_initialize_main_tstep is None
    consumed = harness.consume(produced, entry)
    observations = [item for item in harness.observed if item["phase"] == "consumer"]
    record.update(observations=observations, actual_prepare_phases=harness.prepares,
        returned_consumer_mode=o._paper_context_drag_mode(consumed.prepared_context))
    assert harness.prepares == ["producer"]
    assert len(observations) == (1 if entry == "year-start" else 2)
    for actual in observations:
        assert actual["mode"] is harness.expected_mode
        assert actual["marker"] is None
        assert actual["watchout"] is watchout
        assert actual["actual_swnet"] == ([127.] if watchout else None)
        assert actual["actual_swnet"] == actual["file_swnet"]
    assert o._paper_context_drag_mode(consumed.prepared_context) is harness.expected_mode
    assert consumed.prepared_context.watchout_initialize_main_tstep is None


@pytest.mark.parametrize("entry", ["year-start", "outer-days", "outer-zero"])
def test_missing_history_rejects_before_prepare_or_physics(tmp_path, monkeypatch, entry):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("prepare_or_physics")
        raise AssertionError("Missing history reached fresh preparation or physical delegate")

    monkeypatch.setattr(o, "prepare_paper_1961_driver_context", forbidden)
    monkeypatch.setattr(o, "paper_1961_driver_later_day_runtime_result", forbidden)
    common = dict(year=1963, previous_year_end_state=_handoff_packet(), prepared_context=None, run_mode="continuous")
    try:
        with pytest.raises(ValueError, match="(?is)(?=.*continuous)(?=.*prepared_context)"):
            if entry == "year-start":
                o.paper_1961_driver_restart_year_start_day_result(tmp_path / "not-read.yaml", **common)
            else:
                o.paper_1961_driver_restart_year_multiday_modelout_lite_run(tmp_path / "not-read.yaml", **common,
                    ndays=0 if entry == "outer-zero" else 2)
    finally:
        EVIDENCE["missing_history"].append({"entry": entry, "actual_prepare_or_physics_calls": len(calls)})
    assert calls == []


@pytest.mark.parametrize("family", ["standard", "cold", "native"])
@pytest.mark.parametrize("outcome", ["zero", "first-fail", "later-fail"])
def test_producer_returns_only_actual_successful_history(tmp_path, monkeypatch, family, outcome):
    harness = _Handoff(tmp_path, monkeypatch, family=family, outcome=outcome)
    result = harness.produce(ndays=0 if outcome == "zero" else 2 if outcome == "later-fail" else 1)
    actual = result.prepared_context
    EVIDENCE["producer_edges"].append({"family": family, "outcome": outcome,
        "returned_history": actual is not None, "has_day_end_state": result.last_day_end_state is not None,
        "stopped_day": result.stopped_day_index, "completed_outputs": len(result.daily_modelout)})
    if outcome in ("zero", "first-fail"):
        assert actual is None
    else:
        assert result.stopped_day_index == 2
        assert len(result.daily_modelout) == 1
        assert actual is not None
        assert o._paper_context_drag_mode(actual) is False
        assert actual.watchout_initialize_main_tstep is None


@pytest.mark.parametrize("family", ["standard", "cold", "native"])
def test_continuous_empty_segment_keeps_public_produced_history(tmp_path, monkeypatch, family):
    harness = _Handoff(tmp_path, monkeypatch, family=family)
    produced = harness.produce()
    assert produced.prepared_context is not None
    harness.phase = "consumer"
    result = o.paper_1961_driver_restart_year_multiday_modelout_lite_run(harness.fresh.config_path,
        ndays=0, year=1963, previous_year_end_state=produced.last_day_end_state,
        prepared_context=produced.prepared_context, run_mode="continuous")
    assert result.prepared_context is produced.prepared_context
    assert result.last_day_end_state is produced.last_day_end_state
    assert harness.prepares == ["producer"]


@pytest.mark.parametrize("entry", ["cold-scaffold", "cold-runtime-day1"])
def test_zero_entry_cold_result_has_no_initialized_history(tmp_path, monkeypatch, entry):
    harness = _Handoff(tmp_path, monkeypatch, family="cold", outcome="first-fail")
    if entry == "cold-scaffold":
        result = harness.real_cold(harness.fresh.config_path, year=1962, max_steps=1, prepared_context=harness.fresh)
    else:
        result = o.paper_1961_driver_cold_start_runtime_day_result(harness.fresh.config_path, year=1962, day_index=1)
    EVIDENCE["cold_empty"].append({"entry": entry, "completed_entries": result.completed_steps,
        "has_history": result.prepared_context is not None, "has_day_end_state": result.first_day_end_state is not None})
    assert result.completed_steps == 0
    assert result.first_day_end_state is None
    assert result.prepared_context is None
