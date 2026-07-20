from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.modelout import (
    PAPER_MODEL_PFT14_INDEX,
    compute_modelout_from_fields,
    history_filename_for_age,
    history_year_for_age,
    select_history_point_fields,
)
from jax_orchidee.stomate.parameters import load_pft14_parameters
from jax_orchidee.stomate.reference import (
    default_paper_modelout_csv_path,
    find_stomate_reference_files,
    pack_history_modelout_fields,
    paper_modelout_csv_targets_by_year,
    read_paper_modelout_csv,
)


def test_pft14_parameter_loader_applies_reference_override_over_base_run_def():
    params = load_pft14_parameters(ROOT)

    assert params.pft_index_fortran == 14
    assert params["VCMAX25"] == 63.2061836
    assert params["ARJV"] == 2.45712368323
    assert params["MAINT_RESP_SLOPE_C"] == 0.0876862
    assert params["RESIDENCE_TIME"] == 50.6583452
    assert params["ALLOC_MIN"] == 0.2019211

    assert params["SLA"] == 0.0153
    assert params["LAI_MAX"] == 12.0
    assert params["LEAFLIFE_TAB"] == 1.4
    assert params["ALLOC_AGR_ST"] == 0.25
    assert params["ALLOC_AGR_PN"] == 0.05

    assert params.mangrove_globals["ITIMETIDE"] == 584.0
    assert params.mangrove_globals["CONTROL_SALINITY_MIN"] == 0.5
    assert params.mangrove_globals["CONTROL_INUDATE_MIN"] == 0.5
    assert params.mangrove_globals["AGB_AGR_VEN_ALL_ST"] == 200.0
    assert params.mangrove_globals["AGB_AGR_VEN_ALL_PN"] == 40.0
    assert params.mangrove_globals["H_AGR_MAX_ST"] == 2.0
    assert params.mangrove_globals["H_AGR_MAX_PN"] == 0.4
    assert params.mangrove_globals["WEIGHT_VEN_A"] == 0.5
    assert params.mangrove_globals["WEIGHT_VEN_B"] == 0.5
    assert params.mangrove_globals["DEN_MAN"] == 1000.0


def test_modelout_formula_mapper_matches_paper_script_formula_for_arrays():
    fields = {
        "LEAF_M": np.asarray([[1.0, 2.0]]),
        "SAP_M_AB": np.asarray([[10.0, 20.0]]),
        "HEART_M_AB": np.asarray([[100.0, 200.0]]),
        "AGR_SAP_ST_M": np.asarray([[3.0, 4.0]]),
        "AGR_HRT_ST_M": np.asarray([[5.0, 6.0]]),
        "AGR_SAP_PN_M": np.asarray([[7.0, 8.0]]),
        "AGR_HRT_PN_M": np.asarray([[9.0, 10.0]]),
        "SAP_M_BE": np.asarray([[11.0, 12.0]]),
        "HEART_M_BE": np.asarray([[13.0, 14.0]]),
        "ROOT_M": np.asarray([[15.0, 16.0]]),
        "GPP": np.asarray([[4.25, 5.25]]),
        "NPP": np.asarray([[1.5, 2.5]]),
    }

    result = compute_modelout_from_fields(fields)

    expected_agb = 0.02 * np.asarray([[135.0, 250.0]])
    expected_bgb = 0.02 * np.asarray([[39.0, 42.0]])
    assert np.allclose(np.asarray(result.AGB_model), expected_agb)
    assert np.allclose(np.asarray(result.BGB_model), expected_bgb)
    assert np.allclose(np.asarray(result.GPP_model), fields["GPP"])
    assert np.allclose(np.asarray(result.NPP_model), fields["NPP"])


def test_modelout_formula_mapper_preserves_stomate_history_shape():
    shape = (1, 14, 1, 1)
    fields = {name: np.ones(shape, dtype=np.float64) for name in (
        "LEAF_M",
        "SAP_M_AB",
        "HEART_M_AB",
        "AGR_SAP_ST_M",
        "AGR_HRT_ST_M",
        "AGR_SAP_PN_M",
        "AGR_HRT_PN_M",
        "SAP_M_BE",
        "HEART_M_BE",
        "ROOT_M",
        "GPP",
        "NPP",
    )}

    result = compute_modelout_from_fields(fields)

    assert np.asarray(result.AGB_model).shape == shape
    assert np.asarray(result.BGB_model).shape == shape
    assert np.asarray(result.GPP_model).shape == shape
    assert np.asarray(result.NPP_model).shape == shape
    assert np.allclose(np.asarray(result.AGB_model), 0.14)
    assert np.allclose(np.asarray(result.BGB_model), 0.06)


def test_modelout_history_age_and_pft14_selector_match_paper_script():
    assert history_year_for_age(1) == 1961
    assert history_year_for_age(50) == 2010
    assert history_filename_for_age(50) == "stomate_history_2010.nc"
    assert PAPER_MODEL_PFT14_INDEX == 13


def test_modelout_history_selector_extracts_paper_script_point_without_hardcoding_formula():
    shape = (1, 14, 1, 1)
    fields = {}
    for offset, name in enumerate((
        "LEAF_M",
        "SAP_M_AB",
        "HEART_M_AB",
        "AGR_SAP_ST_M",
        "AGR_HRT_ST_M",
        "AGR_SAP_PN_M",
        "AGR_HRT_PN_M",
        "SAP_M_BE",
        "HEART_M_BE",
        "ROOT_M",
        "GPP",
        "NPP",
    )):
        array = np.zeros(shape, dtype=np.float64)
        array[0, 13, 0, 0] = 10.0 + offset
        fields[name] = array

    selected = select_history_point_fields(fields)
    result = compute_modelout_from_fields(selected)

    assert selected["LEAF_M"].shape == ()
    assert np.asarray(selected["LEAF_M"]) == 10.0
    assert np.asarray(selected["NPP"]) == 21.0
    assert np.asarray(result.GPP_model) == 20.0
    assert np.asarray(result.NPP_model) == 21.0
    assert np.asarray(result.AGB_model) == 0.02 * sum(range(10, 17))


def test_modelout_history_selector_rejects_non_history_axes():
    fields = {name: np.ones((14, 1, 1), dtype=np.float64) for name in (
        "LEAF_M",
        "SAP_M_AB",
        "HEART_M_AB",
        "AGR_SAP_ST_M",
        "AGR_HRT_ST_M",
        "AGR_SAP_PN_M",
        "AGR_HRT_PN_M",
        "SAP_M_BE",
        "HEART_M_BE",
        "ROOT_M",
        "GPP",
        "NPP",
    )}

    with np.testing.assert_raises_regex(ValueError, "time, vegetation, lat, lon"):
        select_history_point_fields(fields)


def test_reference_modelout_csv_matches_selected_2010_history_point():
    csv_path = default_paper_modelout_csv_path(ROOT)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    age = int(float(row["age"]))
    target_name = history_filename_for_age(age)
    histories = {path.name: path for path in find_stomate_reference_files(ROOT).histories}
    fields = pack_history_modelout_fields(histories[target_name])

    selected = select_history_point_fields(fields)
    result = compute_modelout_from_fields(selected)

    assert target_name == "stomate_history_2010.nc"
    assert np.asarray(result.AGB_model) == np.float32(row["AGB_model"])
    assert np.asarray(result.BGB_model) == np.float32(row["BGB_model"])
    assert np.asarray(result.GPP_model) == np.float32(row["GPP_model"])
    assert np.asarray(result.NPP_model) == np.float32(row["NPP_model"])


def test_paper_modelout_csv_reader_maps_age_to_target_history_year():
    rows = read_paper_modelout_csv(root=ROOT)
    targets = paper_modelout_csv_targets_by_year(root=ROOT)

    assert len(rows) == 1
    row = rows[0]
    assert row.i_grid == "001.0-071.0"
    assert row.age == 50
    assert row.target_year == 2010
    assert row.values == {
        "AGB_model": 96.398818359375,
        "BGB_model": 35.8754345703125,
        "GPP_model": 4.085646152496338,
        "NPP_model": 1.6014299392700195,
    }
    assert targets == {2010: (row,)}
