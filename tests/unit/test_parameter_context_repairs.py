"""Source-backed parameter adapter/cache regressions; every physical owner is a spy.

Fortran: control.f90:261-278, thermosoil.f90:338-374. These tests execute
actual Python adapters and cache construction, never a model or real JAX compile.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import gc
from pathlib import Path
from types import SimpleNamespace as NS
import weakref

import numpy as np
import pytest

from jax_orchidee.driver import orchestration as o
from jax_orchidee.driver.restart_state import restart_thermosoil_initialization_supported

ROOT = Path(__file__).resolve().parents[2]
COUNTS = {"fake_lowerings": 0, "fake_compilations": 0, "physical_delegate_spy_calls": 0}


def _materialized_values(pc=False, leak=True):
    """Explicit synthetic complete adapter input, independent of saved output files."""
    values = {name: "0.5" for name in (
        "MIN_WIND FR_DT FROZ_FRAC_CORR SMTOT_CORR MAX_FROZ_HYDRO DEPTH_MAX_H "
        "SECHIBA_QSINT CONTROL_SALINITY_MIN CONTROL_INUDATE_MIN DEPTH_MAX_T "
        "DEPTH_TOPTHICK DEPTH_CSTTHICK DEPTH_GEOM RATIO_GEOM_BELOW "
        "AGB_AGR_VEN_ALL_ST AGB_AGR_VEN_ALL_PN H_AGR_MAX_ST H_AGR_MAX_PN "
        "LAIMAX LAI_LEVEL_DEPTH SNOWCRI"
    ).split()}
    values.update({name: "FALSE" for name in (
        "STOMATE_OK_STOMATE STOMATE_OK_DGVM OK_EXPLICITSNOW ROUGH_DYN IMPOSE_AZE "
        "PEAT_HYDRO OK_FREEZE_CWRR OK_THERMODYNAMICAL_FREEZING DO_PONDS "
        "OK_RU2PEAT OK_WT_AB TIDES AGRI_PEAT dyn_nroot_larix satsoil "
        "RIVER_ROUTING TOPMODEL_NEW EROSION_MODULE FIRE_DISABLE DYN_PEAT "
        "PEAT_OCCUR GLUC_USE_AGE_CLASS OK_SNOWFACT"
    ).split()})
    values.update(NVM="2", NSTM="3", DT_SECHIBA="1800", DT_STOMATE="86400",
                  FR_DT="3.25", SLOPE="0.17", OK_PC=str(pc), OK_LEAK=str(leak))
    for name in ("EXT_COEFF", "EXT_COEFF_VEGETFRAC", "HYDROL_HUMCSTE", "VCMAX_FIX",
                 "SLOWPROC_HEIGHT", "PERCENT_THROUGHFALL_PFT", "RSTRUCT_CONST"):
        values.update({f"{name}__00014": "14.25", f"{name}__00002": "2.25"})
    values.update(PFT_TO_MTC__00014="2", PFT_TO_MTC__00002="3",
                  OK_LAIDEV__00014="FALSE", OK_LAIDEV__00002="TRUE")
    values.update({f"CWRR_KS__{i:05d}": str(i + 0.25) for i in range(1, 13)})
    return values


@pytest.fixture
def prepared_context(monkeypatch, tmp_path):
    scalars = NS(nvm=2, fortran_pft_ids=(14, 2), pft_to_mtc=np.array([2, 3]))
    grid = NS(znh=np.array([0.1, 0.3]), dnh=np.array([0.1, 0.2]), dlh=np.array([0.1, 0.2]))
    bundle = NS(run_scalars=scalars, domain=NS(lalo=np.zeros((1, 2))))
    delegates = {
        "read_run_scalars": scalars, "_paper_mangrove_pft_index": 0,
        "_paper_mangrove_fortran_pft_id": 14,
        "paper_case_cwrr_vertical_soil_grid_from_used_run_def": grid,
        "paper_case_vertical_grids_from_used_run_def": (np.ones(2), np.ones(2), np.ones(2)),
        "load_paper_1961_first_step_bundle": bundle, "load_case_config": {},
        "paper_forcing_split_from_config": 1, "paper_forcing_nb_spread": 1,
        "reference_case_first_step_restart_state": NS(),
        "paper_1961_first_step_stomate_boundary": NS(),
        "paper_case_stomate_static_input_kwargs": NS(parameters=NS()),
        "paper_case_stomate_boundary_input_kwargs": NS(),
        "assemble_pft14_control_inundation_first_step": NS(),
        "pft14_trans_co2_parameter_inputs_from_run_def": {},
    }
    for name, result in delegates.items():
        monkeypatch.setattr(o, name, lambda *args, _result=result, **kwargs: _result)

    def build(values):
        path = tmp_path / "materialized.def"
        path.write_text("\n".join(f"{k}={v}" for k, v in values.items()), encoding="utf-8")
        # Real cached body and parser; skip lru storage to avoid shared fixture state.
        return o._prepare_paper_1961_driver_context_cached.__wrapped__(
            str(tmp_path / "case.yaml"), str(path), None,
        )
    return build


@pytest.mark.parametrize("pc,leak", [(False, False), (True, False), (False, True), (True, True)])
def test_context_resolves_source_pc_override_and_preserves_other_fields(prepared_context, pc, leak):
    values = _materialized_values(pc, leak)
    context = prepared_context(values)
    assert context.hydrol_ok_pc is (pc and not leak)
    assert context.hydrol_ok_leak is leak
    assert context.run_def_values["OK_PC"] == str(pc)  # raw provenance is preserved
    assert context.hydrol_fr_dt == 3.25
    assert context.hydrol_reinf_slope == (0.17,)
    assert context.runtime.dt_sechiba == 1800
    np.testing.assert_array_equal(context.ext_coeff, [14.25, 2.25])


def test_context_retains_full_stomate_without_leak_guard(prepared_context):
    values = {**_materialized_values(True, False), "STOMATE_OK_STOMATE": "TRUE"}
    with pytest.raises(NotImplementedError, match="OK_LEAK=n"):
        prepared_context(values)


def test_materialized_pc_key_stays_required(prepared_context):
    values = _materialized_values()
    del values["OK_PC"]
    with pytest.raises(KeyError, match="OK_PC"):
        prepared_context(values)


@pytest.mark.parametrize("producer", [
    "paper_1961_next_step_hydrol_precall_scaffold",
    "paper_1961_driver_cold_start_first_step_coverage",
    "_paper_1961_next_step_runtime_result_compact",
    "paper_1961_driver_timestep_scaffold",
])
@pytest.mark.parametrize("pc,leak", [(False, False), (True, False), (False, True), (True, True)])
def test_actual_hydrol_caller_expressions_forward_effective_pc(producer, pc, leak):
    """Replay exact current kwargs into a spy, not scaffold or physical control flow."""
    tree = ast.parse((ROOT / "jax_orchidee/driver/orchestration.py").read_text(encoding="utf-8"))
    owner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == producer)
    call = next(n for n in ast.walk(owner) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "run_hydrol_first_step_module_from_precall")
    selected = [k for k in call.keywords if k.arg in {"ok_pc", "ok_leak", "fr_dt"}]
    assert {k.arg for k in selected} == {"ok_pc", "ok_leak", "fr_dt"}
    expression = ast.fix_missing_locations(ast.Expression(ast.Call(
        func=ast.Name(id="consumer_spy", ctx=ast.Load()), args=[], keywords=selected,
    )))
    values = _materialized_values(pc, leak)
    context = NS(hydrol_ok_pc=o._effective_hydrol_ok_pc(values),
                 hydrol_ok_leak=leak, hydrol_fr_dt=3.25)
    actual = eval(compile(expression, str(ROOT / "jax_orchidee/driver/orchestration.py"), "eval"),
                  {**vars(o), "consumer_spy": lambda **kw: kw,
                   "run_def_values": values, "context": context})
    assert actual == {"ok_pc": pc and not leak, "ok_leak": leak, "fr_dt": 3.25}


def _restart_context(**overrides):
    values = {"OK_EXPLICITSNOW": "TRUE", "USE_SOILC_TEMPDIFF": "TRUE"}
    pc = overrides.pop("pc", False); leak = overrides.pop("leak", True)
    values.update(overrides)
    return NS(run_def_values=values, hydrol_ok_pc=pc, hydrol_ok_leak=leak)


@pytest.mark.parametrize("raw,expected", [(None, True), ("TRUE", True), ("FALSE", False)])
def test_restart_refsoc_source_default_and_explicit_controls(raw, expected):
    context = _restart_context(**({} if raw is None else {"use_refSOC": raw}))
    assert restart_thermosoil_initialization_supported(context) is expected


@pytest.mark.parametrize("overrides,expected", [
    ({"OK_EXPLICITSNOW": "FALSE"}, False), ({"USE_SOILC_TEMPDIFF": "FALSE"}, False),
    ({"leak": False, "pc": False, "OK_FREEZE_THERMIX": "TRUE"}, False),
    ({"leak": False, "pc": True, "OK_FREEZE_THERMIX": "FALSE"}, False),
    ({"leak": False, "pc": True, "OK_FREEZE_THERMIX": "TRUE"}, True),
    ({"leak": False, "pc": True, "OK_FREEZE": "TRUE"}, True),
    ({"leak": False, "pc": True, "OK_FREEZE": "TRUE", "OK_FREEZE_THERMIX": "FALSE"}, False),
    ({"leak": True, "pc": False, "OK_FREEZE_THERMIX": "FALSE"}, True),
])
def test_restart_refsoc_retains_soilcarbon_admission_gates(overrides, expected):
    assert restart_thermosoil_initialization_supported(_restart_context(**overrides)) is expected


@pytest.mark.parametrize("missing", ["OK_EXPLICITSNOW", "USE_SOILC_TEMPDIFF"])
def test_restart_refsoc_other_missing_gates_still_reject(missing):
    context = _restart_context()
    del context.run_def_values[missing]
    assert not restart_thermosoil_initialization_supported(context)


@dataclass(frozen=True)
class _Context:
    config_path: Path
    run_def_path: Path
    run_def_values: dict
    hydrol_fr_dt: float
    runtime: object
    first_step_restart_state: object


class _CacheHarness:
    def __init__(self, monkeypatch, tmp_path, *, retain_transition):
        self.temp = tmp_path; self.observed = []; self.cache = {}
        self.spec = NS(components=("witness",), field_names_by_component=(("x",),))
        self.initial = NS(spec=self.spec, values_by_component=((np.zeros(1),),))
        self.tables = NS(mineral=NS(imin=1, imax=2)); self.arrays = NS(_asdict=lambda: {})
        self.created = 0
        monkeypatch.setattr(o, "_COMPILED_LATER_DAY_BLOCK_CACHE", self.cache)
        monkeypatch.setattr(o, "fast_state_from_previous_packet", lambda packet: self.initial)
        monkeypatch.setattr(o, "_compiled_hydrol_table_arrays", lambda tables: self.arrays)
        monkeypatch.setattr(o, "_compiled_landpoint_payload", lambda payload: {})
        monkeypatch.setattr(o, "_compiled_diffuco_parameter_values", lambda context: ())
        monkeypatch.setattr(o, "_paper_1961_later_day_transition_inputs",
                            lambda **kw: NS(compiled_base_payload_template={}))
        monkeypatch.setattr(o, "HydrolRuntimeStaticTables", lambda **kw: NS(**kw))
        monkeypatch.setattr(o, "MineralCWRRTables", lambda **kw: NS(**kw))
        monkeypatch.setattr(o, "StomateRestartSeasonState", lambda **kw: NS(**kw))
        monkeypatch.setattr(o, "paper_1961_driver_later_day_runtime_result", self.physical_spy)
        monkeypatch.setattr(o.jax, "lax", NS(scan=lambda body, carry, xs: body(carry, ((), np.int32(2)))))

        class SpyJit:
            def __init__(inner, transition): inner.transition = transition
            def lower(inner, *args):
                COUNTS["fake_lowerings"] += 1
                return inner
            def compile(inner):
                COUNTS["fake_compilations"] += 1
                if retain_transition:
                    return inner.transition
                # Crucial negative control: this object has no closure or context
                # reference. Only the production cache entry can retain context.
                return object()
        monkeypatch.setattr(o.jax, "jit", SpyJit)

    def context(self, fr_dt=2.0, *, path_name=None):
        self.created += 1
        path = self.temp / (path_name or f"immutable-{self.created}.def")
        contents = f"FR_DT={fr_dt}\n"
        if path.exists(): assert path.read_text() == contents
        else: path.write_text(contents, encoding="utf-8")
        season = NS(provenance=("synthetic",), _asdict=lambda: {"provenance": ("synthetic",)})
        return _Context(self.temp / "same-case.yaml", path, {"FR_DT": str(fr_dt)}, fr_dt,
                        NS(dt_stomate=86400., dt_sechiba=1800.),
                        NS(stomate=(), stomate_readstart=NS(season_state=season)))

    def build(self, context):
        executable, spec = o._paper_compiled_later_day_block_executable(
            context.config_path, context=context, initial_state=(), year=1961,
            block_forcing=(), block_day_numbers=np.array([2], dtype=np.int32),
            prebound_hydrol_runtime_static_tables=self.tables, daily_carbon_dispatch={},
            stomate_parameter_values=(),
        )
        assert spec is self.spec
        return executable

    def physical_spy(self, *args, **kwargs):
        COUNTS["physical_delegate_spy_calls"] += 1
        context = kwargs["prepared_context"]
        assert kwargs["used_run_def_path"] == context.run_def_path
        value = np.asarray([context.hydrol_fr_dt])
        self.observed.append((str(context.run_def_path), context.hydrol_fr_dt))
        return NS(day_end_state=NS(fields_by_component={"witness": {"x": value}}),
                  daily_modelout=NS(modelout_fields={"fr_dt": value}, modelout=value))

    def evaluate(self, executable):
        executable(self.initial.values_by_component, (), np.array([2]), (), self.arrays, {}, (), {}, ())
        return self.observed[-1]


def test_cache_same_instance_reuse_and_immutable_context_partition(monkeypatch, tmp_path):
    harness = _CacheHarness(monkeypatch, tmp_path, retain_transition=True)
    a = harness.context(2., path_name="A.def"); b = harness.context(3., path_name="B.def")
    first = harness.build(a); second = harness.build(b)
    assert first is harness.build(a)
    assert second is harness.build(b)
    assert first is not second
    assert harness.evaluate(first) == (str(a.run_def_path), 2.)
    cached_b = harness.evaluate(second)
    assert cached_b == (str(b.run_def_path), 3.)
    harness.cache.clear()
    assert harness.evaluate(harness.build(b)) == cached_b
    assert a.run_def_path.read_text() == "FR_DT=2.0\n"
    assert b.run_def_path.read_text() == "FR_DT=3.0\n"


def test_cache_equivalent_distinct_contexts_deliberately_compile_separately(monkeypatch, tmp_path):
    harness = _CacheHarness(monkeypatch, tmp_path, retain_transition=True)
    a = harness.context(2., path_name="same.def"); equivalent = harness.context(2., path_name="same.def")
    first = harness.build(a); second = harness.build(equivalent)
    assert first is not second
    assert harness.evaluate(first) == harness.evaluate(second)
    assert harness.build(a) is first and harness.build(equivalent) is second


def test_cache_explicit_owner_retains_context_until_clear_without_executable_closure(monkeypatch, tmp_path):
    harness = _CacheHarness(monkeypatch, tmp_path, retain_transition=False)
    context = harness.context(); reference = weakref.ref(context)
    executable = harness.build(context)
    assert type(executable) is object  # does not retain the transition or context
    assert harness.build(context) is executable
    assert next(iter(harness.cache.values())).context is context
    del context
    gc.collect()
    assert reference() is not None
    harness.cache.clear()
    gc.collect()
    assert reference() is None


def test_cache_bound_eight_fifo_eviction_releases_only_oldest_context(monkeypatch, tmp_path):
    harness = _CacheHarness(monkeypatch, tmp_path, retain_transition=False)
    def register():
        context = harness.context()
        harness.build(context)
        return weakref.ref(context)
    references = [register() for _ in range(8)]
    assert len(harness.cache) == 8
    oldest_key = next(iter(harness.cache))
    harness.build(references[0]())  # a hit preserves FIFO, not LRU semantics
    ninth = register()
    gc.collect()
    assert len(harness.cache) == 8 and oldest_key not in harness.cache
    assert references[0]() is None
    assert all(reference() is not None for reference in references[1:])
    assert ninth() is not None
    live_ids = {id(entry.context) for entry in harness.cache.values()}
    assert len(live_ids) == 8 and live_ids == {key[0] for key in harness.cache}
    harness.cache.clear()
    gc.collect()
    assert all(reference() is None for reference in references) and ninth() is None


def test_direct_cache_snapshot_reader_reports_executable_identity(monkeypatch):
    # Importing the dev audit configures a persistent project JAX cache. Execute
    # just its actual reader definition; no module initialization or audit work.
    path = ROOT / "scripts/dev/audit_compiled_fast_path_acceptance.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    reader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_scan_cache_snapshot")
    namespace = {"orchestration": o}
    exec(compile(ast.Module(body=[reader], type_ignores=[]), str(path), "exec"), namespace)
    context = object(); executable = object()
    entry = o._CompiledLaterDayBlockCacheEntry(context, executable)
    monkeypatch.setattr(o, "_COMPILED_SECHIBA_SCAN_CACHE", {})
    monkeypatch.setattr(o, "_COMPILED_LATER_DAY_BLOCK_CACHE", {("synthetic",): entry})
    snapshot = namespace["_scan_cache_snapshot"]()
    assert snapshot["block_entries"] == 1
    assert snapshot["block_executable_ids"] == [id(executable)]
    assert snapshot["block_executable_ids"] != [id(entry)]
