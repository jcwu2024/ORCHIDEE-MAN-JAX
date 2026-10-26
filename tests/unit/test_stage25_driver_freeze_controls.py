"""Stage25: execute the four driver HYDROL argument expressions.

Source: hydrol.f90:6884-6905 calls soil_froz after the normal solve;
hydrol_soil_froz:8336-8455 consumes the effective freezing controls.
These checks cover argument assembly only. A recording boundary avoids a
model run; the separate HYDROL regressions verify the physical composition.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@lru_cache(maxsize=1)
def _driver_tree():
    from jax_orchidee.driver import orchestration as orch

    return ast.parse(Path(orch.__file__).read_text(encoding="utf-8"))


@pytest.mark.parametrize("entry", [
    "paper_1961_next_step_hydrol_precall_scaffold",
    "paper_1961_driver_cold_start_first_step_coverage",
    "_paper_1961_next_step_runtime_result_compact",
    "paper_1961_driver_timestep_scaffold",
], ids=["ordinary", "cold", "compact", "timestep"])
def test_driver_assembles_actual_effective_freezing_controls(entry):
    from jax_orchidee.driver import orchestration as orch

    expected = dict(ok_thermodynamical_freezing=False, fr_dt=1.25,
                    froz_frac_corr=0.7, smtot_corr=3.0, max_froz_hydro=0.8)
    dh_mm = np.array([1.25, 3.75, 10.0, 20.0])
    values = {
        "OK_THERMODYNAMICAL_FREEZING": "FALSE", "FR_DT": "1.25",
        "FROZ_FRAC_CORR": "0.7", "SMTOT_CORR": "3.0",
        "MAX_FROZ_HYDRO": "0.8", "OK_FREEZE_CWRR": "TRUE",
        "DEPTH_MAX_H": "0.07", "DT_SECHIBA": "1800",
        **{name: "FALSE" for name in ("DO_PONDS", "OK_PC", "OK_LEAK",
            "OK_RU2PEAT", "OK_WT_AB", "TIDES", "AGRI_PEAT", "dyn_nroot_larix")},
        **{f"CWRR_KS__{i:05d}": "0.01" for i in range(1, 13)},
    }
    context = SimpleNamespace(
        hydrol_dz_mm=np.array([0.0, 2.5, 5.0, 10.0]), hydrol_dh_mm=dh_mm,
        hydrol_cwrr_ks=np.full(12, 0.01), hydrol_reinf_slope=[0.0],
        hydrol_depth_max_h=0.07, hydrol_do_ponds=False,
        hydrol_soil_peat_hydro=False, ok_freeze_cwrr=True,
        hydrol_ok_pc=False, hydrol_ok_leak=False, hydrol_ok_ru2peat=False,
        hydrol_ok_wt_ab=False, hydrol_tides=False, hydrol_agri_peat=False,
        hydrol_max_wt_ab=100.0, hydrol_dyn_nroot_larix=False,
        dt_sechiba=1800.0, diaglev=np.array([0.01, 0.02]),
        ok_thermodynamical_freezing=expected["ok_thermodynamical_freezing"],
        hydrol_fr_dt=expected["fr_dt"],
        hydrol_froz_frac_corr=expected["froz_frac_corr"],
        hydrol_smtot_corr=expected["smtot_corr"],
        hydrol_max_froz_hydro=expected["max_froz_hydro"],
    )
    payload = SimpleNamespace(temp_air=np.array([270.0]), pb=np.array([1013.0]),
                              u=np.array([2.0]), v=np.array([0.0]), kjpindex=1)
    scalars = SimpleNamespace(is_tree=np.zeros(14, dtype=bool), nstm=6)
    step = SimpleNamespace(run_scalars=scalars)
    hydrol_state = SimpleNamespace(nroot=None, run2peat=None, run2man=None)
    previous = SimpleNamespace(fields_by_component={
        "slowproc_stomate_previous_step_state": {"altmax": np.array([0.07])}})
    recorded = {}

    def record(precall, **kwargs):
        recorded.update(kwargs)
        return precall

    env = dict(vars(orch))
    env.update(
        run_hydrol_first_step_module_from_precall=record,
        # These unrelated ancillary inputs are not the boundary under test.
        _select_run_def_pft_vector=lambda *args, **kwargs: np.ones(14),
        paper_case_vertical_grids_from_used_run_def=lambda path: (context.diaglev, None),
        hydrol_precall=object(), hydrol_first_step_precall=object(),
        context=context, payload=payload,
        enerbil=SimpleNamespace(payload=payload, step=step), step=step,
        run_scalars=scalars, hydrol_state=hydrol_state,
        hydrol_cold=hydrol_state, previous_state=previous,
        slowproc_state=previous.fields_by_component["slowproc_stomate_previous_step_state"],
        first_step_restart_state=SimpleNamespace(
            stomate=SimpleNamespace(altmax=np.array([0.07])), hydrol=hydrol_state),
        stomate_cold_entry=SimpleNamespace(altmax=np.array([0.07])),
        cwrr_grid=SimpleNamespace(dnh=context.hydrol_dz_mm / 1000.0,
            dlh=dh_mm / 1000.0, znh=np.array([0.0, 0.0025, 0.0075, 0.0175])),
        hydrol_humcste=np.ones(14), hydrol_zz_mm=np.array([0.0, 2.5, 7.5, 17.5]),
        diaglev=context.diaglev, run_def_values=values, run_def_path="unused.def",
        nvm=14, module_jit=False, prepared_context=object(),
        hydrol_static_template={}, hydrol_runtime_static_tables=None,
        fixed_format_trace_dir=None, static_trace_fields=None,
    )
    function = next(node for node in _driver_tree().body
                    if isinstance(node, ast.FunctionDef) and node.name == entry)
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "run_hydrol_first_step_module_from_precall"]
    assert len(calls) == 1
    eval(compile(ast.Expression(calls[0]), orch.__file__, "eval"), env)
    actual = {name: recorded.get(name) for name in expected}
    assert actual == expected, "refresh must receive the actual configured controls"
    np.testing.assert_allclose(recorded["dh_mm"], dh_mm, rtol=0.0, atol=0.0)
