"""HYDROL static/cached/compiled parameter plumbing, without model timesteps.

Fortran: hydrol_init 2028-2252; hydrol_var_init 3932-4306;
hydrol_soil 6414-6470/6474-6562; hydrol_soil_coef 8248-8307.
Numerical consumers below are the actual local table and diagnostic owners.
AST probes check caller wiring only.
"""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from netCDF4 import Dataset

from jax_orchidee.driver import orchestration as o
from jax_orchidee.driver import restart_state as driver_restart
from jax_orchidee.sechiba import hydrol as h
from jax_orchidee.sechiba import restart_io, thermosoil

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FIELDS = {
    "nvan": "CWRR_N_VANGENUCHTEN", "avan": "CWRR_A_VANGENUCHTEN",
    "mcr": "VWC_RESIDUAL", "mcs_mineral": "VWC_SAT",
    "mcf_mineral": "VWC_FC", "mcw_mineral": "VWC_WP", "ks": "CWRR_KS",
    "pcent": "WETNESS_TRANSPIR_MAX", "mc_awet": "VWC_MIN_FOR_WET_ALB",
    "mc_adry": "VWC_MAX_FOR_DRY_ALB",
}
CUSTOM_A = dict(nvan=2.5, avan=.03, mcr=.04, mcs_mineral=.50,
                mcf_mineral=.30, mcw_mineral=.12, ks=100., pcent=.70,
                mc_awet=.32, mc_adry=.09)
CUSTOM_B = dict(nvan=2.7, avan=.04, mcr=.05, mcs_mineral=.55,
                mcf_mineral=.34, mcw_mineral=.14, ks=150., pcent=.65,
                mc_awet=.40, mc_adry=.10)
ZZ_MM = np.array([0., 1., 3., 7., 15., 31., 63.])
DEPTH_DRIVER_CUSTOM = {
    "CWRR_NKS_N0": .85, "CWRR_NKS_POWER": .45,
    "CWRR_AKS_A0": .0002, "CWRR_AKS_POWER": .65,
    "KFACT_DECAY_RATE": 1.5, "KFACT_STARTING_DEPTH": .25, "KFACT_MAX": 20.,
    "altmax_for_nroot_thresh": .6,
}


def _run_def_custom(custom):
    """Complete explicit vectors; change texture 2 and retain other textures."""
    defaults = h.hydrol_soil_parameters()
    values = {}
    for field, key in CONFIG_FIELDS.items():
        vector = np.asarray(defaults[field]).copy()
        vector[1] = custom[field]
        values.update({f"{key}__{i:05d}": str(value)
                       for i, value in enumerate(vector, start=1)})
    return values


def _parameters(custom):
    return h.hydrol_soil_parameters_from_run_def(_run_def_custom(custom))


def _static(parameters, *, refsoc=None, njsc=None):
    return h.hydrol_static_precall_template_from_slowproc(
        slowproc_restart=NS(veget=np.array([[.2, .6]]), veget_max=np.array([[.2, .6]]),
                            frac_nobio=np.array([[.2]]), soiltile=np.array([[.2, .3, .3]])),
        pref_soil_veg=np.array([1, 3]), nstm=3, ext_coeff_vegetfrac=np.array([.5, .5]),
        hydrol_njsc=np.array([2]) if njsc is None else njsc, zz_mm=ZZ_MM,
        refSOC_1d=refsoc, use_refSOC_hydrol=refsoc is not None,
        soil_parameters=parameters,
    )


@pytest.fixture
def context_builder(monkeypatch, tmp_path):
    """Real materialized parser/context body; unrelated IO boundaries are spies."""
    scalars = NS(nvm=2, nstm=3, pref_soil_veg=np.array([1, 3]),
                 fortran_pft_ids=(14, 2), pft_to_mtc=np.array([2, 3]))
    grid = NS(znh=ZZ_MM / 1000., dnh=np.ones(7) / 1000., dlh=np.ones(7) / 1000.)
    bundle = NS(run_scalars=scalars, domain=NS(lalo=np.zeros((1, 2))))
    delegates = {
        "read_run_scalars": scalars, "_paper_mangrove_pft_index": 0,
        "_paper_mangrove_fortran_pft_id": 14,
        "paper_case_cwrr_vertical_soil_grid_from_used_run_def": grid,
        "paper_case_vertical_grids_from_used_run_def": (np.ones(7),) * 3,
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
    values = {name: "0.5" for name in (
        "MIN_WIND FR_DT FROZ_FRAC_CORR SMTOT_CORR MAX_FROZ_HYDRO DEPTH_MAX_H "
        "SECHIBA_QSINT CONTROL_SALINITY_MIN CONTROL_INUDATE_MIN DEPTH_MAX_T "
        "DEPTH_TOPTHICK DEPTH_CSTTHICK DEPTH_GEOM RATIO_GEOM_BELOW "
        "AGB_AGR_VEN_ALL_ST AGB_AGR_VEN_ALL_PN H_AGR_MAX_ST H_AGR_MAX_PN "
        "LAIMAX LAI_LEVEL_DEPTH SNOWCRI"
    ).split()}
    values.update({name: "FALSE" for name in (
        "STOMATE_OK_STOMATE STOMATE_OK_DGVM OK_EXPLICITSNOW ROUGH_DYN IMPOSE_AZE "
        "PEAT_HYDRO OK_FREEZE_CWRR OK_THERMODYNAMICAL_FREEZING DO_PONDS OK_PC OK_LEAK "
        "OK_RU2PEAT OK_WT_AB TIDES AGRI_PEAT dyn_nroot_larix satsoil RIVER_ROUTING "
        "TOPMODEL_NEW EROSION_MODULE FIRE_DISABLE DYN_PEAT PEAT_OCCUR GLUC_USE_AGE_CLASS OK_SNOWFACT"
    ).split()})
    values.update(NVM="2", NSTM="3", DT_SECHIBA="1800", DT_STOMATE="86400")
    for name in ("EXT_COEFF", "EXT_COEFF_VEGETFRAC", "HYDROL_HUMCSTE", "VCMAX_FIX",
                 "SLOWPROC_HEIGHT", "PERCENT_THROUGHFALL_PFT", "RSTRUCT_CONST"):
        values.update({f"{name}__00014": "14.25", f"{name}__00002": "2.25"})
    values.update(PFT_TO_MTC__00014="2", PFT_TO_MTC__00002="3",
                  OK_LAIDEV__00014="FALSE", OK_LAIDEV__00002="TRUE")

    def build(custom, name, *, extra_run_def=None):
        declared = {**values, **_run_def_custom(custom), **(extra_run_def or {})}
        path = tmp_path / f"{name}.def"
        path.write_text("\n".join(f"{k}={v}" for k, v in declared.items()), encoding="utf-8")
        return o._prepare_paper_1961_driver_context_cached.__wrapped__(
            str(tmp_path / "case.yaml"), str(path), None)
    return build


def test_prepared_context_keeps_all_ten_custom_vectors_and_static_values(context_builder):
    a = context_builder(CUSTOM_A, "a")
    b = context_builder(CUSTOM_B, "b")
    for context, custom in ((a, CUSTOM_A), (b, CUSTOM_B)):
        parameters = context.hydrol_soil_parameters
        assert isinstance(parameters, h.HydrolSoilParameters)
        for field, value in custom.items():
            assert np.asarray(parameters[field]).shape == (12,)
            assert float(parameters[field][1]) == value
        np.testing.assert_array_equal(context.hydrol_cwrr_ks, parameters.ks)
        static = _static(parameters)
        for key, field in (("mcr", "mcr"), ("mcs", "mcs_mineral"),
                           ("mcf", "mcf_mineral"), ("mcw", "mcw_mineral")):
            np.testing.assert_array_equal(static[key], [custom[field]])
    assert a.hydrol_soil_parameters is not b.hydrol_soil_parameters
    np.testing.assert_array_equal(_static(a.hydrol_soil_parameters)["mcf"], [.30])


def test_runtime_cache_partitions_hydraulic_values_and_binds_diagnostic_only_values():
    """Unchanged coefficients may share; each runtime retains all ten parameters."""
    a = _parameters(CUSTOM_A)
    payload = _static(a)
    base = h.hydrol_runtime_static_tables_from_payload(payload, soil_parameters=a)
    assert h.hydrol_runtime_static_tables_from_payload(payload, soil_parameters=a).mineral is base.mineral
    for field in CONFIG_FIELDS:
        changed = _parameters({**CUSTOM_A, field: CUSTOM_B[field]})
        tables = h.hydrol_runtime_static_tables_from_payload(payload, soil_parameters=changed)
        np.testing.assert_array_equal(tables.soil_parameters[field], changed[field])
        if field in {"nvan", "avan", "mcr", "ks"}:
            assert tables.mineral is not base.mineral
            coefficient = {"nvan": "nvan_mod", "avan": "avan_mod", "mcr": "mc_lin", "ks": "k_lin"}[field]
            assert not np.array_equal(getattr(tables.mineral, coefficient), getattr(base.mineral, coefficient))
        else:
            assert tables.mineral is base.mineral


def test_real_day_static_binding_uses_each_prepared_context(context_builder, monkeypatch):
    monkeypatch.setattr(o, "_hydrol_state_namespace_from_previous_state",
                        lambda state: (NS(njsc=np.array([2])), ()))
    state = NS(fields_by_component={"slowproc_stomate_previous_step_state": {
        "lai": np.ones((1, 2)), "veget": np.array([[.2, .6]]),
        "veget_max": np.array([[.2, .6]]), "frac_nobio": np.array([[.2]]),
        "soiltile": np.array([[.2, .3, .3]]),
    }})
    for custom, name in ((CUSTOM_A, "day-a"), (CUSTOM_B, "day-b")):
        context = context_builder(custom, name)
        bundle = context.first_step_bundle
        context = replace(context, first_step_bundle=None)
        monkeypatch.setattr(o, "_paper_1961_step_bundle_from_context", lambda *args, **kwargs: bundle)
        result = o._paper_1961_later_day_transition_inputs(
            context=context, current_state=state, year=1961, start=48,
            steps_per_stomate=1, fixed_format_trace_dir=None,
            static_trace_fields=None, prebuild_day_payloads=False)
        np.testing.assert_array_equal(result.hydrol_static_template["mcf"], [custom["mcf_mineral"]])
        for field in CONFIG_FIELDS:
            np.testing.assert_array_equal(result.hydrol_runtime_static_tables.soil_parameters[field],
                                          context.hydrol_soil_parameters[field])


def test_default_static_and_runtime_semantics_match_explicit_usda_contract():
    implicit = _static(None)
    explicit = _static(h.hydrol_soil_parameters())
    for key in ("mcr", "mcs", "mcf", "mcw"):
        np.testing.assert_array_equal(implicit[key], explicit[key])
    first = h.hydrol_runtime_static_tables_from_payload(implicit)
    second = h.hydrol_runtime_static_tables_from_payload(explicit, soil_parameters=h.hydrol_soil_parameters())
    assert first.mineral is second.mineral
    for field in CONFIG_FIELDS:
        np.testing.assert_array_equal(first.soil_parameters[field], h.hydrol_soil_parameters()[field])


def test_refsoc_effective_thresholds_leave_custom_mineral_background_intact():
    parameters = _parameters(CUSTOM_A)
    payload = _static(parameters, refsoc=np.array([h.SOILC_MAX / 2]))
    tables = h.hydrol_runtime_static_tables_from_payload(payload, soil_parameters=parameters)
    assert float(payload["mcs"][0]) != CUSTOM_A["mcs_mineral"]
    assert float(tables.soil_parameters.mcs_mineral[1]) == .50
    assert float(tables.soil_parameters.mcf_mineral[1]) == .30
    assert float(tables.soil_parameters.mcw_mineral[1]) == .12


def _actual_call_keyword(producer, consumer, keyword):
    tree = ast.parse((ROOT / "jax_orchidee/driver/orchestration.py").read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == producer)
    calls = [node for node in ast.walk(owner) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == consumer]
    assert len(calls) == 1
    selected = next((kw.value for kw in calls[0].keywords if kw.arg == keyword), None)
    assert selected is not None, f"{producer} must forward {keyword} to {consumer}"
    return ast.fix_missing_locations(ast.Expression(selected))


@pytest.mark.parametrize("producer,consumer", [
    ("paper_1961_next_step_hydrol_precall_scaffold", "hydrol_static_precall_template_from_slowproc"),
    ("_paper_1961_next_step_runtime_result_compact", "hydrol_static_precall_template_from_slowproc"),
    ("paper_1961_driver_day_scaffold", "hydrol_static_precall_template_from_slowproc"),
    ("paper_1961_driver_later_day_scaffold", "hydrol_static_precall_template_from_slowproc"),
    ("_paper_1961_later_day_transition_inputs", "hydrol_static_precall_template_from_slowproc"),
    ("paper_1961_next_step_hydrol_precall_scaffold", "assemble_hydrol_first_step_precall_payload"),
    ("paper_1961_driver_cold_start_first_step_coverage", "assemble_hydrol_first_step_precall_payload"),
    ("_paper_1961_next_step_runtime_result_compact", "assemble_hydrol_first_step_precall_payload"),
    ("paper_1961_driver_timestep_scaffold", "assemble_hydrol_first_step_precall_payload"),
    ("paper_1961_next_step_hydrol_precall_scaffold", "run_hydrol_first_step_module_from_precall"),
    ("paper_1961_driver_cold_start_first_step_coverage", "run_hydrol_first_step_module_from_precall"),
    ("_paper_1961_next_step_runtime_result_compact", "run_hydrol_first_step_module_from_precall"),
    ("paper_1961_driver_timestep_scaffold", "run_hydrol_first_step_module_from_precall"),
    ("_paper_1961_later_day_transition_inputs", "hydrol_runtime_static_tables_from_payload"),
])
def test_actual_caller_parameter_expressions_keep_their_declared_context(producer, consumer):
    expression = _actual_call_keyword(producer, consumer, "soil_parameters")
    for custom in (CUSTOM_A, CUSTOM_B):
        parameters = _parameters(custom)
        environment = {**vars(o), "context": NS(hydrol_soil_parameters=parameters),
                       "run_def_values": _run_def_custom(custom), "hydrol_soil_parameters": parameters,
                       "hydrol_soil": parameters, "hydrol_runtime_static_tables": None}
        actual = eval(compile(expression, "<actual-soil-parameter-keyword>", "eval"), environment)
        for field in CONFIG_FIELDS:
            np.testing.assert_array_equal(actual[field], parameters[field])
        if any(isinstance(node, ast.Name) and node.id == "hydrol_runtime_static_tables"
               for node in ast.walk(expression)):
            # A compiled executable can capture A while its dynamic tables carry B.
            dynamic = _parameters(CUSTOM_B if custom is CUSTOM_A else CUSTOM_A)
            environment["hydrol_runtime_static_tables"] = NS(soil_parameters=dynamic)
            actual = eval(compile(expression, "<actual-dynamic-soil-keyword>", "eval"), environment)
            assert actual is dynamic


def test_compiled_dynamic_table_roundtrip_drives_actual_diagnostics_without_context_capture():
    """One narrow JIT receives both same-shape contexts as dynamic PyTree values."""
    def diagnose(flat):
        tables = o._hydrol_runtime_static_tables_from_compiled_arrays(flat, imin=1, imax=51)
        p = tables.soil_parameters
        def point(values):
            return jnp.asarray(values)[jnp.array([1])]
        mc = jnp.full((1, 7), .22)
        layer = h.hydrol_soil_layer_diagnostics(
            mc=mc, mcl=mc, dz_mm=jnp.ones(7), njsc=jnp.array([2]),
            mcs=point(p.mcs_mineral), mcf=point(p.mcf_mineral), mcw=point(p.mcw_mineral),
            mcs_mineral=p.mcs_mineral, mcf_mineral=p.mcf_mineral, mcw_mineral=p.mcw_mineral,
            pcent=p.pcent)
        litter = h.hydrol_litter_top_diagnostics(
            mc=jnp.full((1, 7, 3), .22), dz_mm=jnp.ones(7), dh_mm=jnp.ones(7), njsc=jnp.array([2]),
            mcr=p.mcr, mcs=point(p.mcs_mineral), mcf=point(p.mcf_mineral), mcw=point(p.mcw_mineral),
            mc_awet=p.mc_awet, mc_adry=p.mc_adry)
        return layer.sm_nostress, litter.soil_wet_litter, litter.tmc_litter_awet, tables.mineral.k_lin

    compiled = jax.jit(diagnose)
    observed = []
    for custom in (CUSTOM_A, CUSTOM_B, CUSTOM_A):
        p = _parameters(custom)
        tables = h.hydrol_runtime_static_tables_from_payload(_static(p), soil_parameters=p)
        flat = o._compiled_hydrol_table_arrays(tables)
        restored = o._hydrol_runtime_static_tables_from_compiled_arrays(flat, imin=1, imax=51)
        for field in tables.mineral._fields:
            np.testing.assert_array_equal(getattr(restored.mineral, field), getattr(tables.mineral, field))
        for field in CONFIG_FIELDS:
            np.testing.assert_array_equal(restored.soil_parameters[field], p[field])
        eager = diagnose(flat)
        actual = compiled(flat)
        for expected, result in zip(eager, actual, strict=True):
            np.testing.assert_allclose(result, expected, rtol=1.e-13, atol=1.e-13)
        # Source stencil: half first layer and three complete following layers.
        np.testing.assert_allclose(actual[1], (.22 - custom["mcw_mineral"]) /
                                   (custom["mcf_mineral"] - custom["mcw_mineral"]), rtol=1.e-13)
        np.testing.assert_allclose(actual[2], 3.5 * custom["mc_awet"], rtol=1.e-13)
        observed.append(tuple(np.asarray(value) for value in actual))
    assert not np.array_equal(observed[0][0], observed[1][0])
    assert not np.array_equal(observed[0][1], observed[1][1])
    assert not np.array_equal(observed[0][2], observed[1][2])
    assert not np.array_equal(observed[0][3], observed[1][3])
    for first, repeated in zip(observed[0], observed[2], strict=True):
        np.testing.assert_array_equal(first, repeated)


def test_small_real_file_init_preserves_legacy_schema_and_rebinds_soil_parameters(tmp_path):
    """Real partial NetCDF read only; this is not complete checkpoint validation."""
    path = tmp_path / "partial_init.nc"
    with Dataset(path, "w") as ds:
        for dim, size in (("time", 1), ("layer", 7), ("soil", 3), ("y", 1), ("x", 1)):
            ds.createDimension(dim, size)
        ds.createVariable("moistc", "f8", ("time", "layer", "soil", "y", "x"))[:] = .22
        ds.createVariable("njsc", "f8", ("time", "y", "x"))[:] = 2.
    raw = restart_io.read_sechiba_restart_state(path)
    initialized = restart_io.read_hydrol_init_restart_state(path, npts=1, nvm=2, nstm=3, nslm=7)
    assert raw.__class__ is restart_io.SechibaRestartState
    assert raw.__class__.__module__ == "jax_orchidee.sechiba.restart_io"
    assert set(raw.fields) == {"moistc", "njsc"}
    assert initialized["mc"].shape == (1, 3, 7)
    assert "soil_parameters" not in raw.fields and "soil_parameters" not in initialized
    for custom in (CUSTOM_A, CUSTOM_B):
        p = _parameters(custom)
        static = _static(p, njsc=raw.fields["njsc"])
        np.testing.assert_array_equal(static["njsc"], raw.fields["njsc"])
        np.testing.assert_array_equal(static["mcf"], [custom["mcf_mineral"]])


def _restart_initialization_case(*, with_refsoc):
    p = _parameters(CUSTOM_A)
    znh = np.geomspace(.001, 2., 7)
    dnh = np.diff(np.r_[0., znh])
    dlh = np.r_[(dnh[:-1] + dnh[1:]) / 2., dnh[-1] / 2.]
    context = NS(
        run_def_values={"OK_EXPLICITSNOW": "TRUE", "USE_SOILC_TEMPDIFF": "TRUE",
                        "use_refSOC": "TRUE", "use_refSOC_hydrol": "TRUE"},
        hydrol_soil_parameters=p, hydrol_ok_pc=False, hydrol_ok_leak=True,
        hydrol_soil_peat_hydro=False, hydrol_tides=False, hydrol_agri_peat=False,
        ok_freeze_cwrr=False, hydrol_depth_max_h=2., hydrol_humcste=np.array([1., 2.]),
        hydrol_dz_mm=dnh * 1000., hydrol_dh_mm=dlh * 1000., thermosoil_satsoil=False,
        first_step_bundle=NS(vegetation=NS(soiltile=np.array([[.2, .3, .3]]))),
        cwrr_grid=NS(znh=znh, dnh=dnh, dlh=dlh, znt=znh, zlt=znh, dz5=np.ones(7)),
        run_scalars=NS(pref_soil_veg=np.array([1, 3])),
    )
    source = dict(veget=np.array([[.2, .6]]), veget_max=np.array([[.2, .6]]),
                  njsc=np.array([2]), mc=np.full((1, 7, 3), .22), mcl=np.full((1, 7, 3), .22),
                  water2infilt=np.zeros((1, 3)), resdist=np.zeros((1, 3)),
                  vegtot_old=np.array([.8]), qsintveg=np.zeros((1, 2)), drysoil_frac=np.zeros(1))
    if with_refsoc:
        source["refSOC_1d"] = np.array([h.SOILC_MAX / 2.])
    return context, source


def test_actual_restart_initialization_retains_contract_and_separates_effective_thresholds(monkeypatch):
    context, source = _restart_initialization_case(with_refsoc=True)
    observed = {}
    original_hydro = h.hydrol_var_init
    original_moisture = h.hydrol_cold_start_thermosoil_moisture_inputs

    def hydro(**kwargs):
        observed["hydro_kwargs"] = kwargs
        observed["hydro"] = original_hydro(**kwargs)
        return observed["hydro"]

    def moisture(*args, **kwargs):
        observed["moisture_kwargs"] = kwargs
        observed["moisture"] = original_moisture(*args, **kwargs)
        return observed["moisture"]

    class ReachedThermalBoundary(Exception):
        pass

    def stop_before_thermal(**kwargs):
        raise ReachedThermalBoundary

    monkeypatch.setattr(h, "hydrol_var_init", hydro)
    monkeypatch.setattr(h, "hydrol_cold_start_thermosoil_moisture_inputs", moisture)
    monkeypatch.setattr(thermosoil, "thermosoil_humlev", stop_before_thermal)
    with pytest.raises(ReachedThermalBoundary):
        driver_restart.restart_thermosoil_initial_fields(
            source, context=context, payload=NS(), altmax=np.zeros((1, 2)))
    assert observed["hydro_kwargs"]["soil_parameters"] is context.hydrol_soil_parameters
    assert observed["hydro_kwargs"]["refSOC_1d"] is source["refSOC_1d"]
    assert observed["hydro_kwargs"]["use_refSOC_hydrol"] is True
    np.testing.assert_allclose(observed["hydro"].mcs, [.5 * h.POROS_ORG + .5 * .50], rtol=1.e-13)
    np.testing.assert_array_equal(observed["moisture_kwargs"]["mcs_mineral"],
                                  context.hydrol_soil_parameters.mcs_mineral)
    assert float(observed["hydro"].mcf[0]) != CUSTOM_A["mcf_mineral"]
    assert float(observed["hydro"].mcw[0]) != CUSTOM_A["mcw_mineral"]


def test_actual_restart_initialization_rejects_enabled_refsoc_without_the_field():
    context, source = _restart_initialization_case(with_refsoc=False)
    with pytest.raises(ValueError, match="refSOC_1d"):
        driver_restart.restart_thermosoil_initial_fields(
            source, context=context, payload=NS(), altmax=np.zeros((1, 2)))


@pytest.fixture(scope="module")
def depth_driver_tables():
    """Small real tables isolate depth changes while keeping soil unchanged."""
    soil = _parameters(CUSTOM_A)
    payload = {"njsc": [2], "mcs": [CUSTOM_A["mcs_mineral"]], "zz_mm": [1., 550., 2000.]}
    return tuple(h.hydrol_runtime_static_tables_from_payload(
        payload, soil_parameters=soil, depth_parameters=h.hydrol_depth_parameters(config),
    ) for config in (DEPTH_DRIVER_CUSTOM, {}))


def test_depth_driver_materialized_prepared_context_binds_nondefault_scalars(context_builder):
    custom = context_builder(
        CUSTOM_A, "depth-prepared", extra_run_def={**DEPTH_DRIVER_CUSTOM, "KFACT_DECAY_RATE": "1.5D0"},
    )
    baseline = context_builder(CUSTOM_A, "depth-default")
    assert isinstance(custom.hydrol_depth_parameters, h.HydrolDepthParameters)
    assert custom.hydrol_depth_parameters.as_config_values() == DEPTH_DRIVER_CUSTOM
    assert baseline.hydrol_depth_parameters.as_config_values() == h.hydrol_depth_parameters().as_config_values()
    assert o._prepared_hydrol_depth_parameters(custom) is custom.hydrol_depth_parameters
    assert o._prepared_hydrol_depth_parameters(baseline) is baseline.hydrol_depth_parameters


def test_depth_driver_legacy_context_fallback_and_runtime_priority(depth_driver_tables):
    custom, baseline = depth_driver_tables
    for legacy in (NS(run_def_values=DEPTH_DRIVER_CUSTOM),
                   NS(run_def_values=DEPTH_DRIVER_CUSTOM, hydrol_depth_parameters=None)):
        selected = o._prepared_hydrol_depth_parameters(legacy)
        assert selected.as_config_values() == DEPTH_DRIVER_CUSTOM
    context = NS(hydrol_depth_parameters=custom.depth_parameters, run_def_values={})
    assert o._prepared_hydrol_depth_parameters(context) is custom.depth_parameters
    assert o._prepared_hydrol_depth_parameters(
        context, runtime_static_tables=baseline,
    ) is baseline.depth_parameters
    assert o._prepared_hydrol_depth_parameters(
        context, runtime_static_tables=NS(),
    ) is custom.depth_parameters


def test_depth_driver_compiled_pack_tree_roundtrip_preserves_actual_consumer_values(depth_driver_tables):
    """Actual pack/restore and coefficient consumer; no JIT or physical step."""
    observed = []
    context = NS(hydrol_depth_parameters=depth_driver_tables[0].depth_parameters, run_def_values={})
    for runtime in depth_driver_tables:
        packed = o._compiled_hydrol_table_arrays(runtime)
        assert packed.depth_parameters is runtime.depth_parameters
        leaves, structure = jax.tree_util.tree_flatten(packed)
        unpacked = jax.tree_util.tree_unflatten(structure, leaves)
        restored = o._hydrol_runtime_static_tables_from_compiled_arrays(unpacked, imin=1, imax=51)
        for field in runtime.mineral._fields:
            np.testing.assert_array_equal(getattr(restored.mineral, field), getattr(runtime.mineral, field))
        assert restored.depth_parameters.as_config_values() == runtime.depth_parameters.as_config_values()
        assert o._prepared_hydrol_depth_parameters(
            context, runtime_static_tables=restored,
        ) is restored.depth_parameters

        def consume(tables):
            return h.hydrol_soil_coef_mineral_profile_from_tables(
                mc=[[.25, .28, .31]], profil_froz=[[0., .25, .5]], kfact_root=np.ones((1, 3)),
                tables=tables.mineral, mcr=[CUSTOM_A["mcr"]], mcs=[CUSTOM_A["mcs_mineral"]],
            )

        before, after = consume(runtime), consume(restored)
        for field in ("k", "d"):
            np.testing.assert_array_equal(getattr(after, field), getattr(before, field))
            assert np.isfinite(np.asarray(getattr(after, field))).all()
        observed.append(after)
    assert not np.array_equal(observed[0].k, observed[1].k)
    assert not np.array_equal(observed[0].d, observed[1].d)


def test_depth_driver_actual_later_day_binding_builds_nondefault_tables(context_builder, monkeypatch):
    # This fixture's last layer is 63 mm; use a 10 mm compacted depth so the
    # actual caller's numerical table can distinguish custom/default values.
    declared = {**DEPTH_DRIVER_CUSTOM, "KFACT_STARTING_DEPTH": .01}
    context = context_builder(CUSTOM_A, "depth-later-day", extra_run_def=declared)
    bundle = context.first_step_bundle
    context = replace(context, first_step_bundle=None)
    monkeypatch.setattr(o, "_paper_1961_step_bundle_from_context", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(o, "_hydrol_state_namespace_from_previous_state",
                        lambda state: (NS(njsc=np.array([2])), ()))
    state = NS(fields_by_component={"slowproc_stomate_previous_step_state": {
        "lai": np.ones((1, 2)), "veget": np.array([[.2, .6]]),
        "veget_max": np.array([[.2, .6]]), "frac_nobio": np.array([[.2]]),
        "soiltile": np.array([[.2, .3, .3]]),
    }})
    result = o._paper_1961_later_day_transition_inputs(
        context=context, current_state=state, year=1961, start=48,
        steps_per_stomate=1, fixed_format_trace_dir=None,
        static_trace_fields=None, prebuild_day_payloads=False,
    )
    runtime = result.hydrol_runtime_static_tables
    assert runtime.depth_parameters is context.hydrol_depth_parameters
    expected = np.clip(np.exp(-1.5 * (np.asarray(context.cwrr_grid.znh) - .01)), 1. / 20., 1.)
    np.testing.assert_allclose(runtime.mineral.kfact[:, 0], expected, rtol=1.e-14)
    default = h.hydrol_runtime_static_tables_from_payload(
        result.hydrol_static_template, soil_parameters=context.hydrol_soil_parameters,
    )
    assert not np.array_equal(runtime.mineral.kfact, default.mineral.kfact)
    assert not np.array_equal(runtime.mineral.k_lin, default.mineral.k_lin)


@pytest.mark.parametrize("producer,consumer", [
    ("paper_1961_next_step_hydrol_precall_scaffold", "run_hydrol_first_step_module_from_precall"),
    ("paper_1961_driver_cold_start_first_step_coverage", "run_hydrol_first_step_module_from_precall"),
    ("_paper_1961_next_step_runtime_result_compact", "run_hydrol_first_step_module_from_precall"),
    ("paper_1961_driver_timestep_scaffold", "run_hydrol_first_step_module_from_precall"),
    ("_paper_1961_later_day_transition_inputs", "hydrol_runtime_static_tables_from_payload"),
])
def test_depth_driver_actual_call_expressions_forward_the_declared_contract(producer, consumer, depth_driver_tables):
    """AST argument evidence only; actual numerical callers are tested above."""
    expression = _actual_call_keyword(producer, consumer, "depth_parameters")
    custom, baseline = depth_driver_tables
    environment = {**vars(o), "context": NS(hydrol_depth_parameters=custom.depth_parameters),
                   "run_def_values": DEPTH_DRIVER_CUSTOM, "hydrol_runtime_static_tables": None}
    actual = eval(compile(expression, "<actual-depth-parameter-keyword>", "eval"), environment)
    assert actual.as_config_values() == DEPTH_DRIVER_CUSTOM
    if any(isinstance(node, ast.Name) and node.id == "hydrol_runtime_static_tables"
           for node in ast.walk(expression)):
        environment["hydrol_runtime_static_tables"] = baseline
        selected = eval(compile(expression, "<actual-dynamic-depth-keyword>", "eval"), environment)
        assert selected is baseline.depth_parameters
