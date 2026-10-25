"""Actual OK_LEAK adapter DIC boundary, with independent source arithmetic."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.stomate import integration as adapter
from jax_orchidee.stomate.carbon_kernels import maintenance_respiration, NPARTS, IROOT, ICARBON
from jax_orchidee.stomate.daily import sum_resp_maint_radia
from jax_orchidee.stomate.soilcarbon_kernels import NPOOL, ICO2AQ

REFERENCE = json.loads((Path(__file__).parent / "fixtures/ok_leak_dic_units.json").read_text())
PFT = 13


def _inputs():
    nvm, deep, tiles = 14, 2, 4
    vp = np.zeros((1, nvm)); vp[0, PFT] = REFERENCE["cover"]
    parts = np.zeros((1, nvm, NPARTS, 1))
    litter = np.zeros((1, 2, nvm, 1)); litter[0, :, PFT, 0] = [3., 2.]
    below = np.zeros((1, 2, nvm, deep, 1)); below[0, :, PFT, :, 0] = 2.
    carbon = np.zeros((1, 3, nvm, deep)); carbon[0, :, PFT, :] = 20.
    doc = np.zeros((1, nvm, deep, 2, NPOOL, 1)); doc[0, PFT, :, 0, :, 0] = .1
    fb = np.zeros((1, deep, nvm)); fb[0, :, PFT] = .01
    natural = np.ones(nvm, bool)
    zero_pft = np.zeros((1, nvm)); zero_doc = np.zeros((1, nvm, 1))
    fuel = np.zeros((1, nvm, 2, 1))
    return dict(
        litter_above=litter, litter_below=below,
        lignin_struc_above=np.full((1,nvm), .25), lignin_struc_below=np.full((1,nvm,deep), .35),
        litterpart=np.zeros((1,nvm,2)), dead_leaves=np.zeros((1,nvm,2)),
        fuel_1hr=fuel.copy(), fuel_10hr=fuel.copy(), fuel_100hr=fuel.copy(), fuel_1000hr=fuel.copy(),
        bm_to_litter=parts.copy(), turnover=parts.copy(), rprof=np.ones((1,nvm)), veget_max=vp,
        sla_calc=np.full((1,nvm), .1), fbact_litter=fb, poor_soils=np.zeros(1),
        flood_frac=np.array([REFERENCE["flood_fraction"]]), carbon_32l=carbon, doc=doc,
        doc_precip2ground=zero_doc.copy(), doc_precip2canopy=zero_doc.copy(), dry_dep_canopy=zero_doc.copy(),
        interception_storage=zero_doc.copy(), canopy2ground=zero_pft.copy(),
        doc_to_topsoil=np.zeros((1,4)), doc_to_subsoil=np.zeros((1,4)), fbact_doc=fb,
        fbact_soilcarbon=fb, soil_mc=np.full((1,deep,tiles), .5), soil_mc_32l=np.full((1,deep,tiles), .5),
        wat_flux=np.zeros((1,deep,tiles)), soilwater_31mm=np.ones((1,tiles)),
        runoff_per_soil=np.full((1,tiles), REFERENCE["runoff"]),
        drainage_per_soil=np.full((1,tiles), REFERENCE["drainage"]), runoff2peat=np.zeros((1,tiles)),
        fastr=np.array([25.]), pref_soil_veg=np.zeros(nvm, int),
        tprof=np.full((1,deep,nvm), 280.), clay=np.array([.2]), bulk_dens=np.array([1650.]),
        z_soil=np.array([0.,1.,2.]), zf_soil_b=np.array([0.,1.,2.]), flux_red=np.ones(1),
        natural=natural, is_peat=np.zeros(nvm,bool), is_c4=np.zeros(nvm,bool),
        resp_maint_part_radia=np.zeros((1,nvm,NPARTS)), flood_root_radia=zero_pft.copy(),
        dt_days=REFERENCE["dt_days"], control_temp_above=np.ones((1,2)),
        control_moist_above=np.ones((1,nvm)), soil_mc_top_by_pft=np.full((1,nvm),.5),
        nslm=deep, ndeep=deep, sro_bottom=deep,
    )


def test_actual_adapter_matches_fixed_mixed_source_fixture(monkeypatch):
    inputs = _inputs()
    shape = inputs["veget_max"].shape
    def rate(value):
        x = np.zeros(shape); x[0,PFT] = value; return x
    lr = np.stack([rate(REFERENCE["litter_above_rate"]), rate(REFERENCE["litter_below_rate"])], axis=2)
    litter = SimpleNamespace(
        litter_above=inputs["litter_above"], litter_below=inputs["litter_below"],
        lignin_struc_above=inputs["lignin_struc_above"], lignin_struc_below=inputs["lignin_struc_below"],
        soilcarbon_input_doc=np.zeros((1,14,2,NPOOL,1)), floodcarbon_input=np.zeros((1,14,NPOOL,1)),
        resp_hetero_litter=lr, resp_hetero_flood=rate(REFERENCE["flood_litter_rate"]),
    )
    soil = SimpleNamespace(doc_exp=np.zeros((1,14,3,NPOOL,1)),
        resp_hetero_soil=rate(REFERENCE["soil_rate"]), resp_flood_soil=rate(REFERENCE["flood_soil_rate"]))
    monkeypatch.setattr(adapter, "littercalc_leak_core_with_controls_jit", lambda *a, **k: litter)
    monkeypatch.setattr(adapter, "soilcarbon_leak_core_step_jit", lambda *a, **k: soil)
    inputs["resp_maint_part_radia"][0,PFT,IROOT] = REFERENCE["root_step_mass"]
    inputs["flood_root_radia"][0,PFT] = REFERENCE["flood_root_step_mass"]
    result = adapter.stomate_ok_leak_explicit(**inputs)
    np.testing.assert_allclose(np.asarray(result.doc_export.doc_exp_agg)[0,:,ICO2AQ],
                               REFERENCE["expected_dic_run_flood_drain"], rtol=1e-12, atol=1e-14)
    # Conversion belongs only to aggregation; retained core contracts stay rates.
    np.testing.assert_array_equal(result.soilcarbon.resp_hetero_soil, soil.resp_hetero_soil)
    np.testing.assert_array_equal(result.littercalc.resp_hetero_litter, lr)


@pytest.mark.parametrize("flow", [True, False], ids=["mixed_nonzero", "no_flow"])
def test_actual_adapter_real_components_preserve_maintenance_step_mass(flow):
    inputs = _inputs()
    if not flow:
        inputs["flood_frac"][:] = 0.
        inputs["runoff_per_soil"][:] = 0.
        inputs["drainage_per_soil"][:] = 0.
    # Real maintenance producer includes dt in coeff_maint (Fortran resp:297-298).
    biomass = np.zeros((1,14,NPARTS,1)); biomass[0,PFT,IROOT,ICARBON] = 100.
    maintenance = maintenance_respiration(
        biomass, np.array([273.15]), np.array([273.15]), np.full((1,2),273.15),
        np.array([0.,1.,2.]), inputs["rprof"], inputs["sla_calc"], np.full((14,NPARTS),.01),
        np.zeros((14,3)), np.full(14,.5), np.ones(14,bool), dt_sechiba_days=inputs["dt_days"])
    _, flooded_root = sum_resp_maint_radia(maintenance.resp_maint_part, flood_frac=inputs["flood_frac"])
    inputs["resp_maint_part_radia"] = maintenance.resp_maint_part
    inputs["flood_root_radia"] = flooded_root
    result = adapter.stomate_ok_leak_explicit(**inputs)
    dt, cover, flood = inputs["dt_days"], inputs["veget_max"][0,PFT], inputs["flood_frac"][0]
    # Independent transcription of stomate 3314-3315,3409-3410,3442-3456.
    below_rate = float(result.littercalc.resp_hetero_litter[0,PFT,1])
    soil_rate = float(result.soilcarbon.resp_hetero_soil[0,PFT])
    root_mass = .01 * dt * 100.
    np.testing.assert_allclose(maintenance.resp_maint_part[0,PFT,IROOT], root_mass, rtol=1e-12)
    modifier = ((below_rate + soil_rate)*dt + root_mass)/(4.25*(1.-flood)*dt)
    flood_mass = (float(result.littercalc.resp_hetero_flood[0,PFT]) + float(result.soilcarbon.resp_flood_soil[0,PFT]))*dt + flood*root_mass
    expected = [inputs["runoff_per_soil"][0,0]*20e-4*cover*modifier,
                flood_mass*cover, inputs["drainage_per_soil"][0,0]*20e-3*cover*modifier]
    np.testing.assert_allclose(np.asarray(result.doc_export.doc_exp_agg)[0,:,ICO2AQ], expected, rtol=1e-12, atol=1e-14)
    if flow:
        assert below_rate > 0 and soil_rate > 0 and flood_mass > 0
    else:
        np.testing.assert_array_equal(expected, [0.,0.,0.])
