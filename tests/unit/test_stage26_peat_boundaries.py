"""Bounded component contracts; no multi-peat production admission or model run."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jax_orchidee.stomate import soilcarbon_kernels as sk


def _export_case(peat_pfts):
    nvm, ndeep, nstm = 14, 1, 4
    doc = np.zeros((1, nvm, ndeep, sk.NDOC, sk.NPOOL, 1))
    is_peat = np.zeros(nvm, dtype=bool)
    for pft, stock in peat_pfts.items():
        is_peat[pft] = True
        doc[0, pft, 0, sk.IFREE, sk.IACT, 0] = stock
    result = sk.soilcarbon_leak_doc_export(
        doc,
        soilwater_31mm=np.ones((1, nstm)),
        runoff_per_soil=np.zeros((1, nstm)),
        drainage_per_soil=np.zeros((1, nstm)),
        runoff2peat=np.full((1, nstm), 100.0),
        fastr=np.array([25.0]), flux_red=np.ones(1),
        pref_soil_veg=np.zeros(nvm, dtype=np.int32),
        veget_max=np.full((1, nvm), 1.0 / nvm),
        wet_dep_flood=np.zeros((1, nvm, 1)),
        floodcarbon_input=np.zeros((1, nvm, sk.NPOOL, 1)),
        fluxtot_flood=np.zeros((1, sk.NCARB, 1, nvm, ndeep)),
        lignin_struc_above=np.zeros((1, nvm)),
        lignin_struc_below=np.zeros((1, nvm, ndeep)),
        soil_mc=np.ones((1, ndeep, nstm)),
        soil_mc_32l=np.ones((1, ndeep, nstm)),
        zf_soil_b=np.array([0.0, 1.0]), z_soil=np.array([0.0, 1.0]),
        is_peat=is_peat, dt_days=1.0, sro_bottom=1, nslm=1,
    )
    # Scalar source transcription: buffer initialized once at 1126, reset by
    # a non-peat at 2130, accumulated at 2108, added to DOC at 2137.
    expected_buffer, expected_doc = np.zeros(nvm), np.zeros(nvm)
    buffer = 0.0
    for pft in range(nvm):
        if pft in peat_pfts:
            stock = float(peat_pfts[pft])
            # Each chosen stock <= DOCexp_max=20, so correction=1, water=1.
            buffer += min(stock, 100.0 * 1.0e-3 * stock)
            expected_buffer[pft] = buffer
            expected_doc[pft] = stock + buffer
        else:
            buffer = 0.0
    np.testing.assert_allclose(np.asarray(result.doc_exp), 0.0, rtol=0.0, atol=0.0)
    return result, expected_buffer, expected_doc


@pytest.mark.parametrize("peat_pfts", [{13: 10.0}, {12: 10.0, 13: 20.0}, {11: 10.0, 13: 20.0}],
                         ids=["single-current-layout", "consecutive-component", "nonpeat-resets"])
def test_export_peat_buffer_source_contract(peat_pfts):
    """Fortran soilcarbon_leak 1126, 2102-2141; generic helper is already supported."""
    result, expected_buffer, expected_doc = _export_case(peat_pfts)
    actual_buffer = np.asarray(result.doc_run_2_peat)[0, :, sk.IACT, 0, 0]
    actual_doc = np.asarray(result.doc)[0, :, 0, sk.IFREE, sk.IACT, 0]
    np.testing.assert_allclose(actual_buffer, expected_buffer, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual_doc, expected_doc, rtol=1e-14, atol=1e-14)


def _peat_case(stock):
    nvm, ndeep = 14, 2
    carbon = np.zeros((1, sk.NCARB, nvm, ndeep))
    carbon[0, sk.IACTIVE, 13, 0] = stock
    peat = np.zeros(nvm, dtype=bool)
    peat[13] = True
    return carbon, np.ones(ndeep), np.array([0.0, 1.0, 2.0]), peat, np.ones((1, nvm), dtype=bool)


@pytest.mark.parametrize("stock", [5.0e-9, 1.0e-8, 2.0e-8], ids=["below", "equal", "above"])
def test_peat_olt_component_default_uses_source_threshold(stock):
    """constantes_var:177 min_stomate=1e-8; soilcarbon_leak:1425-1436 uses strict >."""
    args = _peat_case(stock)
    result = sk.soilcarbon_perma_peat_redistribute(*args)
    # No layer exceeds .95*Cmax, so all inventories must be exactly unchanged.
    np.testing.assert_array_equal(np.asarray(result.carbon_32l), args[0])
    np.testing.assert_array_equal(np.asarray(result.carbon_transfer), np.zeros_like(args[0]))
    expected = stock if stock > 1.0e-8 else 0.0
    assert float(result.peat_olt[0, 13]) == expected


def test_peat_olt_explicit_threshold_retains_diagnostic_only_control():
    """Existing explicit component override is diagnostic only, never a stock clamp."""
    args = _peat_case(5.0e-9)
    zero = sk.soilcarbon_perma_peat_redistribute(*args, min_stomate=0.0)
    source = sk.soilcarbon_perma_peat_redistribute(*args, min_stomate=1.0e-8)
    assert float(zero.peat_olt[0, 13]) == 5.0e-9
    assert float(source.peat_olt[0, 13]) == 0.0
    for name in ("carbon_32l", "deepc_peat", "deepc_pt", "carbon_transfer"):
        np.testing.assert_array_equal(np.asarray(getattr(zero, name)), np.asarray(getattr(source, name)))


def test_peat_olt_real_core_default_uses_source_diagnostic_threshold():
    """Actual eager core, no stubs: diagnostic default differs, stock consumers do not."""
    carbon, cmax, zf, peat, mask = _peat_case(5.0e-9)
    nvm, ndeep, nstm = 14, 2, 4
    kwargs = dict(
        carbon_32l=carbon,
        doc=np.zeros((1, nvm, ndeep, sk.NDOC, sk.NPOOL, 1)),
        soilcarbon_input_doc=np.zeros((1, nvm, ndeep, sk.NPOOL, 1)),
        doc_to_topsoil=np.zeros((1, nstm)), doc_to_subsoil=np.zeros((1, nstm)),
        wet_dep_ground=np.zeros((1, nvm, 1)), wet_dep_flood=np.zeros((1, nvm, 1)),
        litter_above=np.zeros((1, sk.NLITT, nvm, 1)),
        litter_below=np.zeros((1, sk.NLITT, nvm, ndeep, 1)),
        lignin_struc_above=np.zeros((1, nvm)), lignin_struc_below=np.zeros((1, nvm, ndeep)),
        fbact_doc=np.zeros((1, ndeep, nvm)), fbact=np.zeros((1, ndeep, nvm)),
        soil_mc=np.ones((1, ndeep, nstm)), soil_mc_32l=np.ones((1, ndeep, nstm)),
        wat_flux=np.zeros((1, ndeep, nstm)), soilwater_31mm=np.ones((1, nstm)),
        runoff_per_soil=np.zeros((1, nstm)), drainage_per_soil=np.zeros((1, nstm)),
        runoff2peat=np.zeros((1, nstm)), fastr=np.array([25.0]),
        pref_soil_veg=np.zeros(nvm, dtype=np.int32), veget_max=np.full((1, nvm), 1.0 / nvm),
        flood_frac=np.zeros(1), floodcarbon_input=np.zeros((1, nvm, sk.NPOOL, 1)),
        tprof=np.full((1, ndeep, nvm), 280.0), clay=np.array([0.2]), bulk_dens=np.array([1650.0]),
        z_soil=zf, zf_soil_b=zf, flux_red=np.ones(1), natural=np.ones(nvm, dtype=bool),
        is_peat=peat, is_c4=np.zeros(nvm, dtype=bool), dt_days=1.0 / 48.0,
        dif_doc=np.zeros(1), nslm=2, sro_bottom=1, perma_peat=True,
        cmax_peat=cmax, perma_peat_veget_mask=mask,
    )
    result = sk.soilcarbon_leak_core_step(**kwargs)
    np.testing.assert_array_equal(np.asarray(result.perma_peat.carbon_32l), carbon)
    np.testing.assert_array_equal(np.asarray(result.carbon_32l), carbon)
    np.testing.assert_array_equal(np.asarray(result.doc), kwargs["doc"])
    assert float(result.perma_peat.peat_olt[0, 13]) == 0.0
