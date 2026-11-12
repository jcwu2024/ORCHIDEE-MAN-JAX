"""Restart entry must rebuild nonpersisted memory before any model step."""
from types import SimpleNamespace

import numpy as np
import ast
import inspect

import jax_orchidee.driver.orchestration as orch
import pytest


def test_first_restart_boundary_forwards_the_actual_restored_directory():
    """Execute the actual caller expression without any upstream process/IO."""
    tree = ast.parse(inspect.getsource(orch.paper_1961_driver_timestep_scaffold))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "paper_1961_first_step_stomate_boundary"]
    assert len(calls) == 1
    observed = {}
    def boundary(*args, **kwargs):
        observed.update(kwargs)
        return "no-model-no-io"
    declared = object()
    env = dict(paper_1961_first_step_stomate_boundary=boundary, config_path="config.yaml",
               step=object(), payload=object(), run_def_path="run.def", root=object(),
               first_step_restart_state=SimpleNamespace(run_dir=declared))
    result = eval(compile(ast.Expression(calls[0]), "<actual-first-step-boundary-call>", "eval"), env)
    assert result == "no-model-no-io"
    assert observed["run_dir"] is declared


def test_year_entry_initializes_from_saved_surface_without_mutating_old_year(monkeypatch):
    surface = np.asarray([275.25])
    old_temp = np.asarray([274.75])
    old_pcapa = np.full((1, 2, 1), 900.0)
    initial_pcapa = np.full((1, 2, 1), 1200.0)
    source = {"temp_sol": surface}
    state = orch.DriverPreviousStepStatePacket(
        tstep=17519,
        fields_by_component={
            "sechiba_finalize_state": source,
            "driver_previous_step_state": {"z0": np.asarray([0.1])},
            "hydrol_previous_step_state": {"nroot": np.ones((1, 1)), "temp_hydro": np.full((1, 2), 280.0)},
            "thermosoil_previous_step_state": {"temp_sol_beg": old_temp, "pcapa_en": old_pcapa, "cgrnd": np.ones((1, 1, 1))},
            "slowproc_stomate_previous_step_state": {"altmax": np.ones((1, 1))},
        },
        provenance_by_component={},
    )
    context = SimpleNamespace(runtime=SimpleNamespace(dt_stomate=86400, dt_sechiba=1800), dt_sechiba=1800, run_def_path="used_run.def", hydrol_ok_pc=False, hydrol_ok_leak=True, run_def_values={name: "TRUE" for name in ("OK_EXPLICITSNOW", "USE_SOILC_TEMPDIFF", "use_refSOC")})
    observed = {}
    monkeypatch.setattr(orch, "driver_year_handoff_state_gaps", lambda state: ())
    monkeypatch.setattr(orch, "_paper_1961_step_bundle_from_context", lambda context, **kw: SimpleNamespace(year=kw["year"]))
    monkeypatch.setattr(orch, "build_intersurf_first_step_payload", lambda step, **kw: SimpleNamespace(year=step.year))
    def initialize(restored, **kwargs):
        assert restored["temp_sol"] is surface
        assert kwargs["payload"].year == 1962
        return {"temp_sol_beg": restored["temp_sol"], "pcapa_en": initial_pcapa, "stempdiag": np.asarray([[273.0, 274.0]])}
    monkeypatch.setattr(orch, "restart_thermosoil_initial_fields", initialize)
    def next_day(*args, **kwargs):
        observed.update(kwargs)
        return "runtime-not-executed"
    monkeypatch.setattr(orch, "paper_1961_driver_later_day_runtime_result", next_day)

    result = orch.paper_1961_driver_restart_year_start_day_result("config.yaml", previous_year_end_state=state, year=1962, prepared_context=context)

    assert result == "runtime-not-executed"
    rebased = observed["previous_state"]
    assert rebased.tstep == -1 and observed["start_tstep"] == 0
    thermal = rebased.fields_by_component["thermosoil_previous_step_state"]
    np.testing.assert_array_equal(thermal["temp_sol_beg"], surface)
    np.testing.assert_array_equal(thermal["pcapa_en"], initial_pcapa)
    assert thermal["cgrnd"] is state.fields_by_component["thermosoil_previous_step_state"]["cgrnd"]
    assert "nroot" not in rebased.fields_by_component["hydrol_previous_step_state"]
    assert "temp_hydro" not in rebased.fields_by_component["hydrol_previous_step_state"]
    assert state.tstep == 17519
    assert state.fields_by_component["thermosoil_previous_step_state"]["temp_sol_beg"] is old_temp
    assert state.fields_by_component["thermosoil_previous_step_state"]["pcapa_en"] is old_pcapa
    assert "nroot" in state.fields_by_component["hydrol_previous_step_state"]
    assert "temp_hydro" in state.fields_by_component["hydrol_previous_step_state"]


def test_active_restart_initializes_all_derived_fields_from_their_owners(monkeypatch):
    """The active branch must reach the actual first DIFFUCO caller tuple."""
    from jax_orchidee.stomate.carbon_kernels import deadleaf_cover_from_litter
    shape=(1, 3)
    slow=SimpleNamespace(veget=np.full(shape, .2), veget_max=np.full(shape, .3),
                         lai=np.full(shape, 2.), height=np.full(shape, 4.), frac_nobio=np.asarray([[.1]]))
    entry=SimpleNamespace(assim_param=np.full((*shape, 1), 17.), leaf_age=np.ones((*shape, 4)),
                          leaf_frac=np.full((*shape, 4), .25), dead_leaves=np.ones((*shape, 2)), sla_calc=np.full(shape, .02))
    restart=SimpleNamespace(slowproc=slow, stomate_readstart=SimpleNamespace(
        entry_state=entry, season_state=SimpleNamespace(t2m_month=np.asarray([291.25]))))
    monkeypatch.setattr(orch, "_select_run_def_pft_vector", lambda values,name,scalars,**kw: np.full(3, 31. if name=="VCMAX_FIX" else 9.))
    values={"STOMATE_OK_STOMATE":"TRUE", "SECHIBA_QSINT":".1"}
    derived=orch._paper_first_step_slowproc_initial_fields(restart_state=restart,run_scalars=None,run_def_values=values)
    np.testing.assert_array_equal(derived.assim_param,entry.assim_param)
    np.testing.assert_array_equal(derived.height,slow.height)
    np.testing.assert_allclose(derived.temp_growth,[18.1],atol=1e-13,rtol=0)
    expected=deadleaf_cover_from_litter(entry.dead_leaves,slow.veget_max/.9,entry.sla_calc)
    np.testing.assert_array_equal(derived.deadleaf_cover,expected)
    np.testing.assert_allclose(derived.qsintmax,np.asarray([[0., .04, .04]]),atol=1e-16,rtol=0)
    # Execute the real first-scaffold helper and assembler call expressions.
    tree=ast.parse(inspect.getsource(orch.paper_1961_driver_timestep_scaffold))
    call=next(node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
              and node.func.id=="_paper_first_step_slowproc_initial_fields")
    actual=eval(compile(ast.Expression(call),"<actual-first-derived-call>","eval"),dict(
        _paper_first_step_slowproc_initial_fields=orch._paper_first_step_slowproc_initial_fields,
        first_step_restart_state=restart,step=SimpleNamespace(run_scalars=None),run_def_values=values))
    np.testing.assert_array_equal(actual.temp_growth,derived.temp_growth)
    inactive=orch._paper_first_step_slowproc_initial_fields(restart_state=restart,run_scalars=None,
        run_def_values={**values,"STOMATE_OK_STOMATE":"FALSE"})
    np.testing.assert_array_equal(inactive.temp_growth,[25.])
    np.testing.assert_array_equal(inactive.height,np.full(shape,9.))
    np.testing.assert_array_equal(inactive.assim_param,np.full((*shape,1),31.))
    np.testing.assert_array_equal(inactive.deadleaf_cover,[0.])


@pytest.mark.parametrize("field,value",[("njsc",[1.5]),("njsc",[0.]),("njsc",[13.]),
    ("veget_year",[np.nan]),("veget_year",[1.25]),("veget_year",[2**31])])
def test_restart_integer_owner_contract_rejects_invalid_wire_values(field,value):
    from jax_orchidee.sechiba.restart_io import restore_sechiba_integer_source_fields
    with pytest.raises(ValueError):restore_sechiba_integer_source_fields({field:np.asarray(value)})


def test_restart_integer_owner_contract_preserves_values_and_declares_kind():
    from jax_orchidee.sechiba.restart_io import restore_sechiba_integer_source_fields
    source={"njsc":np.asarray([12.]),"veget_year":np.asarray([-1.])}
    restored=restore_sechiba_integer_source_fields(source)
    for name in source:
        np.testing.assert_array_equal(restored[name],source[name])
        assert restored[name].dtype==np.int32
    assert source["njsc"].dtype==np.float64
    with pytest.raises(ValueError):restore_sechiba_integer_source_fields(source,njsc_class_count=3)




def test_first_step_thermal_alias_carries_nonzero_restored_memory():
    tree=ast.parse(inspect.getsource(orch.driver_previous_step_state_from_timestep_scaffold))
    assignment=next(node for node in ast.walk(tree) if isinstance(node,ast.Assign)
        and any(ast.unparse(target)=="fields['thermosoil_previous_step_state']['e_soil_lat']" for target in node.targets))
    restored=np.asarray([[7.25, -2.]])
    fields={"thermosoil_previous_step_state":{}}
    env=dict(fields=fields,scaffold=SimpleNamespace(first_step_restart_state=SimpleNamespace(
        thermosoil=SimpleNamespace(e_soil_lat=restored))))
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),"<actual-first-thermal-alias>","exec"),env)
    assert fields["thermosoil_previous_step_state"]["e_soil_lat"] is restored


def test_first_step_extractor_preserves_actual_growth_for_next_diffuco():
    """Execute the actual extractor branch then the next consumer assembler."""
    tree=ast.parse(inspect.getsource(orch.driver_previous_step_state_from_timestep_scaffold))
    branch=next(node for node in ast.walk(tree) if isinstance(node,ast.If)
        and '"temp_growth" in scaffold.diffuco_first_step_precall.payload' in ast.unparse(node.test).replace("'", '"'))
    growth=np.asarray([18.1])
    fields={"slowproc_stomate_previous_step_state":{}}
    env=dict(fields=fields,scaffold=SimpleNamespace(diffuco_first_step_precall=SimpleNamespace(payload={"temp_growth":growth})))
    exec(compile(ast.Module(body=[branch],type_ignores=[]),"<actual-first-growth-handoff>","exec"),env)
    assert fields["slowproc_stomate_previous_step_state"]["temp_growth"] is growth
    slow={**fields["slowproc_stomate_previous_step_state"],"veget":np.full((1,3),.2),"lai":np.full((1,3),2.)}
    _,payload=orch._paper_later_day_slowproc_derivvar_payload(slow,vcmax_fix=np.full(3,31.),height_presc=np.full(3,9.),qsintcst=.1)
    np.testing.assert_array_equal(payload["temp_growth"],growth)


@pytest.mark.parametrize("freeze,pc,leak,expected", [(False, False, False, False),
    (True, False, False, False), (True, True, False, True), (False, False, True, True)])
def test_thermal_initialization_obeys_source_soilcarbon_enable_gate(freeze,pc,leak,expected):
    from jax_orchidee.driver.restart_state import restart_thermosoil_initialization_supported
    values={name:"TRUE" for name in ("OK_EXPLICITSNOW","USE_SOILC_TEMPDIFF","use_refSOC","OK_WETDIAGLONG")}
    values["OK_FREEZE_THERMIX"]="TRUE" if freeze else "FALSE"
    context=SimpleNamespace(run_def_values=values,hydrol_ok_pc=pc,hydrol_ok_leak=leak)
    assert restart_thermosoil_initialization_supported(context) is expected




def test_first_extractor_carries_growth_and_thermal_alias_to_next_real_consumer(monkeypatch):
    """Execute the full extractor with orthogonal producer owners stubbed."""
    growth=np.asarray([18.1]); latent=np.asarray([[7.25,-2.]])
    for name in ("_add_enerbil_local_to_previous_fields","_add_diffuco_local_to_previous_fields",
        "_add_diffuco_saved_controls_to_previous_fields","_add_thermosoil_to_previous_fields",
        "_add_condveg_to_previous_fields","_add_hydrol_to_previous_fields","_write_maintenance_lai_to_previous_fields"):
        monkeypatch.setattr(orch,name,lambda *a,**kw:None)
    monkeypatch.setattr(orch,"stomate_fixed_path_carry_state",lambda **kw:SimpleNamespace(fields={}))
    monkeypatch.setattr(orch,"sechiba_finalize_source_state_from_restart",lambda state:{})
    monkeypatch.setattr(orch,"_sechiba_finalize_state_from_components",lambda **kw:{})
    slow=SimpleNamespace(**{name:np.ones((1,2)) for name in ("veget","veget_max","lai","frac_nobio","height","frac_age")})
    restart=SimpleNamespace(slowproc=slow,stomate=SimpleNamespace(),diffuco_precall=SimpleNamespace(soilalbedo_bg=np.zeros((1,2))),
        hydrol=SimpleNamespace(),thermosoil=SimpleNamespace(refsoc=None,e_soil_lat=latent),stomate_input=None,
        sechiba_restart_state=object(),stomate_readstart=SimpleNamespace(entry_state=object(),gas_state=object(),
        season_state=SimpleNamespace(dt_days_read=1.),remainder_state=SimpleNamespace(**{name:np.ones(1) for name in
        ("resp_hetero","tsurf_year","deepC_peat","depth_deepsoil","t2m_14")})))
    scaffold=SimpleNamespace(step=SimpleNamespace(tstep=0),first_step_restart_state=restart,
        diffuco_first_step_precall=SimpleNamespace(payload={"temp_growth":growth}),stomate_first_step_daily_accumulation=None,
        **{name:None for name in ("enerbil_first_step_local","diffuco_first_step_local_enerbil_precall",
        "diffuco_control_salinity","diffuco_control_assembly","thermosoil_first_step_module",
        "condveg_first_step_module","hydrol_first_step_module")})
    packet=orch.driver_previous_step_state_from_timestep_scaffold(scaffold)
    assert packet.fields_by_component["thermosoil_previous_step_state"]["e_soil_lat"] is latent
    slowfields=packet.fields_by_component["slowproc_stomate_previous_step_state"]
    _,payload=orch._paper_later_day_slowproc_derivvar_payload(slowfields,vcmax_fix=np.full(2,31.),height_presc=np.full(2,9.),qsintcst=.1)
    np.testing.assert_array_equal(payload["temp_growth"],growth)


def test_forwarded_directory_reaches_real_stomate_reader(tmp_path,monkeypatch):
    """Actual caller expression + boundary + NetCDF reader; no model processes."""
    from pathlib import Path
    from shutil import copyfile
    from netCDF4 import Dataset
    import jax_orchidee.driver.stomate_boundary as boundary
    from jax_orchidee.driver.init import read_run_scalars
    root=Path(__file__).resolve().parents[2]
    config=root/"configs/orchidee_man_250919.yaml"
    templates=sorted((root/"reference/OUT/orc_calibrate_250919_sen/arg2_1.0/001.0-071.0").rglob("stomate_start.nc"))
    if not templates:pytest.skip("real paper restart unavailable")
    restored=tmp_path/"stomate_start.nc";copyfile(templates[0],restored)
    with Dataset(restored,"r+") as nc:nc.variables["biomass"][:]=37.25
    scalars=read_run_scalars(config)
    monkeypatch.setattr(boundary,"parse_run_def",lambda path:{"NSTM":"3","RIVER_ROUTING":"FALSE"})
    monkeypatch.setattr(boundary,"paper_case_vertical_grids_from_used_run_def",lambda path:(np.ones(1),)*3)
    monkeypatch.setattr(boundary,"find_stomate_reference_files",lambda *a:pytest.fail("forwarding must suppress default historical reader"))
    seen={}
    def consumer(**kw):seen.update(kw);return "process-not-run"
    monkeypatch.setattr(boundary,"stomate_pre_step_ok_leak_boundary_inputs",consumer)
    tree=ast.parse(inspect.getsource(orch.paper_1961_driver_timestep_scaffold))
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id=="paper_1961_first_step_stomate_boundary")
    result=eval(compile(ast.Expression(call),"<actual-restored-reader-call>","eval"),dict(
        paper_1961_first_step_stomate_boundary=boundary.paper_1961_first_step_stomate_boundary,
        config_path=config,step=SimpleNamespace(run_scalars=scalars),payload=SimpleNamespace(kjpindex=1),
        run_def_path=tmp_path/"unused.def",root=root,first_step_restart_state=SimpleNamespace(run_dir=tmp_path)))
    assert result.stomate_files.start==restored
    np.testing.assert_array_equal(seen["state"].biomass,np.full_like(seen["state"].biomass,37.25))
