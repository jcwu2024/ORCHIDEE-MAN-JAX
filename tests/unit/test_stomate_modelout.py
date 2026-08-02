from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.parameters.pft_catalog import build_pft_run_layout, load_pft_catalog
from jax_orchidee.stomate.carbon_kernels import (
    IAGRHRTPN,
    IAGRHRTST,
    IAGRSAPPN,
    IAGRSAPST,
    ICARBON,
    IHEARTABOVE,
    IHEARTBELOW,
    ILEAF,
    IROOT,
    ISAPABOVE,
    ISAPBELOW,
    NPARTS,
)
from jax_orchidee.stomate.modelout import (
    MODEL_OUTPUT_FIELD_NAMES,
    MODEL_OUTPUT_FIELD_SOURCES,
    annual_history_mean_fields_from_daily_modelout,
    compute_modelout_from_fields,
    modelout_pft_selection,
    modelout_source_gaps,
    select_history_point_fields,
    select_history_point_fields_for_pft_id,
    stomate_lpj_history_fields_from_state,
)


def test_modelout_field_sources_cover_all_paper_inputs_with_fortran_and_xml_lines():
    assert modelout_source_gaps() == ()
    assert tuple(MODEL_OUTPUT_FIELD_SOURCES) == MODEL_OUTPUT_FIELD_NAMES

    npp = MODEL_OUTPUT_FIELD_SOURCES["NPP"]
    assert npp.history_name == "NPP"
    assert npp.xios_field_id == "NPP_STOMATE"
    assert npp.state_argument == "npp_daily"
    assert npp.fortran_expression == "npp_daily"
    assert npp.xios_send_line == 1677
    assert npp.history_write_lines == (2200, 2201)
    assert npp.xml_field_def_line == 634
    assert npp.xml_file_line == 762

    for name, source in MODEL_OUTPUT_FIELD_SOURCES.items():
        assert source.history_name == name
        assert source.fortran_expression
        assert source.xios_send_line > 0
        assert source.history_write_lines[0] <= source.history_write_lines[1]
        assert source.xml_field_def_line > 0
        assert source.xml_file_line > 0


def test_modelout_source_gaps_reports_missing_source_instead_of_filling():
    assert modelout_source_gaps(("LEAF_M", "NOT_A_STOMATE_HISTORY_FIELD")) == (
        "NOT_A_STOMATE_HISTORY_FIELD",
    )


def test_stomate_lpj_history_fields_from_state_feeds_paper_modelout_formula():
    npts, nvm = 1, 14
    pft = 13
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ILEAF, ICARBON] = 10.0
    biomass[0, pft, ISAPABOVE, ICARBON] = 20.0
    biomass[0, pft, IHEARTABOVE, ICARBON] = 30.0
    biomass[0, pft, IAGRSAPST, ICARBON] = 40.0
    biomass[0, pft, IAGRHRTST, ICARBON] = 50.0
    biomass[0, pft, IAGRSAPPN, ICARBON] = 60.0
    biomass[0, pft, IAGRHRTPN, ICARBON] = 70.0
    biomass[0, pft, ISAPBELOW, ICARBON] = 80.0
    biomass[0, pft, IHEARTBELOW, ICARBON] = 90.0
    biomass[0, pft, IROOT, ICARBON] = 100.0
    gpp = np.zeros((npts, nvm), dtype=np.float64)
    npp = np.zeros((npts, nvm), dtype=np.float64)
    gpp[0, pft] = 12.5
    npp[0, pft] = 7.5

    fields = stomate_lpj_history_fields_from_state(biomass=biomass, gpp_daily=gpp, npp_daily=npp)
    result = compute_modelout_from_fields({name: value[0, pft] for name, value in fields.items()})

    assert np.allclose(
        np.asarray(result.AGB_model),
        0.02 * (10.0 + 20.0 + 30.0 + 40.0 + 50.0 + 60.0 + 70.0),
    )
    assert np.allclose(np.asarray(result.BGB_model), 0.02 * (80.0 + 90.0 + 100.0))
    assert np.allclose(np.asarray(result.GPP_model), 12.5)
    assert np.allclose(np.asarray(result.NPP_model), 7.5)


def test_stomate_lpj_history_fields_from_state_maps_each_field_to_real_state_slot():
    npts, nvm = 2, 14
    biomass = np.full((npts, nvm, NPARTS, 1), -999.0, dtype=np.float64)
    gpp = np.full((npts, nvm), -111.0, dtype=np.float64)
    npp = np.full((npts, nvm), -222.0, dtype=np.float64)

    point, pft = 1, 13
    expected = {
        "LEAF_M": (ILEAF, 101.0),
        "SAP_M_AB": (ISAPABOVE, 102.0),
        "HEART_M_AB": (IHEARTABOVE, 103.0),
        "AGR_SAP_ST_M": (IAGRSAPST, 104.0),
        "AGR_HRT_ST_M": (IAGRHRTST, 105.0),
        "AGR_SAP_PN_M": (IAGRSAPPN, 106.0),
        "AGR_HRT_PN_M": (IAGRHRTPN, 107.0),
        "SAP_M_BE": (ISAPBELOW, 108.0),
        "HEART_M_BE": (IHEARTBELOW, 109.0),
        "ROOT_M": (IROOT, 110.0),
    }
    for _, (part, value) in expected.items():
        biomass[point, pft, part, ICARBON] = value
    gpp[point, pft] = 211.0
    npp[point, pft] = 222.0

    fields = stomate_lpj_history_fields_from_state(biomass=biomass, gpp_daily=gpp, npp_daily=npp)

    for name, (_, value) in expected.items():
        assert np.asarray(fields[name])[point, pft] == value
    assert np.asarray(fields["GPP"])[point, pft] == 211.0
    assert np.asarray(fields["NPP"])[point, pft] == 222.0

    result = compute_modelout_from_fields({name: value[point, pft] for name, value in fields.items()})
    belowground = {"SAP_M_BE", "HEART_M_BE", "ROOT_M"}
    assert np.asarray(result.AGB_model) == 0.02 * sum(
        value for name, (_, value) in expected.items() if name not in belowground
    )
    assert np.asarray(result.BGB_model) == 0.02 * (108.0 + 109.0 + 110.0)


def test_annual_history_mean_fields_from_daily_modelout_averages_and_exposes_history_axes():
    daily = []
    for day in range(3):
        fields = {}
        for offset, name in enumerate(MODEL_OUTPUT_FIELD_NAMES):
            value = np.zeros((1, 14), dtype=np.float64)
            value[0, 13] = 100.0 + 10.0 * day + offset
            fields[name] = value
        daily.append(fields)

    history = annual_history_mean_fields_from_daily_modelout(daily)
    selected = select_history_point_fields(history)
    result = compute_modelout_from_fields(selected)

    assert np.asarray(history["GPP"]).shape == (1, 14, 1, 1)
    assert np.asarray(selected["LEAF_M"]) == 110.0
    assert np.asarray(selected["NPP"]) == 121.0
    assert np.asarray(result.GPP_model) == 120.0
    assert np.asarray(result.NPP_model) == 121.0
    assert np.asarray(result.AGB_model) == 0.02 * sum([110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0])


def test_annual_history_mean_fields_from_daily_modelout_rejects_missing_and_empty_records():
    with np.testing.assert_raises_regex(ValueError, "at least one"):
        annual_history_mean_fields_from_daily_modelout(())

    complete = {name: np.zeros((1, 14), dtype=np.float64) for name in MODEL_OUTPUT_FIELD_NAMES}
    incomplete = dict(complete)
    incomplete.pop("GPP")
    with np.testing.assert_raises_regex(KeyError, "Missing GPP"):
        annual_history_mean_fields_from_daily_modelout((complete, incomplete))


def test_annual_history_mean_fields_from_daily_modelout_can_return_runtime_axes():
    daily = [
        {name: np.ones((2, 14), dtype=np.float64) * scale for name in MODEL_OUTPUT_FIELD_NAMES}
        for scale in (2.0, 4.0)
    ]

    averaged = annual_history_mean_fields_from_daily_modelout(daily, as_history_axes=False)

    assert np.asarray(averaged["LEAF_M"]).shape == (2, 14)
    assert np.allclose(np.asarray(averaged["LEAF_M"]), 3.0)


def test_modelout_selection_uses_stable_pft_identity_and_records_metadata():
    catalog = load_pft_catalog(
        ROOT / "configs" / "pft_catalogs" / "orchidee_man_paper_250919.json"
    )
    layout = build_pft_run_layout(
        catalog,
        layout_id="paper_250919_legacy14",
        fractions=[0.0] * 13 + [1.0],
    )
    fields = {
        name: np.arange(14, dtype=np.float64).reshape(1, 14, 1, 1)
        for name in MODEL_OUTPUT_FIELD_NAMES
    }

    selected, metadata = select_history_point_fields_for_pft_id(
        fields,
        layout=layout,
        pft_id="mangrove_pft14",
    )

    assert np.asarray(selected["GPP"]) == 13.0
    assert metadata == modelout_pft_selection(layout, "mangrove_pft14")
    assert metadata.metadata() == {
        "catalog_id": "orchidee_man_paper_250919",
        "layout_id": "paper_250919_legacy14",
        "pft_id": "mangrove_pft14",
        "pft_index": 13,
        "fortran_pft_id": 14,
        "mtc_id": 2,
    }
