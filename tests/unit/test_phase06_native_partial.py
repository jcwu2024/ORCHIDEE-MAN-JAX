"""Portable original-backed native-like restart creation and next operands.

The NetCDF factory is synthetic. These tests certify its actual maintained
reader/initializer boundary, never an authentic native IOIPSL producer route.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from jax_orchidee.sechiba import thermosoil
from scripts.dev import oracle_phase06_native_partial as oracle

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def reference():
    return oracle.load_fixture()


@pytest.mark.parametrize("name,kind,field", oracle.CASES, ids=[case[0] for case in oracle.CASES])
def test_actual_partial_netcdf_to_initialize_matches_original(reference, tmp_path, name, kind, field):
    path = tmp_path / (name + ".nc")
    raw = oracle.canonical_restart(reference, kind, field)
    oracle.write_native_like(path, raw)
    normalized = oracle.read_native_like(path)
    assert normalized.keys() == raw.keys()
    for key, array in raw.items():
        np.testing.assert_array_equal(normalized[key], array, err_msg=key)
    actual, flags, initialized, state = oracle.run_chain(reference, path)
    assert flags["actual_NetCDF_reader"]
    assert flags["maintained_recurrence_adapter"]
    assert not flags["authentic_IOIPSL_producer"]
    assert bool(initialized.calculate_coef) is (kind != "retained")
    expected = {**reference["initial_expected"], **reference["case_expected_overrides"][name]}
    assert len(expected) == 43
    for key, encoded in expected.items():
        np.testing.assert_allclose(actual[key], oracle.codec.decode(encoded),
            rtol=reference["rtol"], atol=reference["atol"], err_msg=key)
    for key in oracle.FIELDS:
        array = np.asarray(getattr(initialized, key))
        assert np.isfinite(array).all()
        assert not np.any(array == oracle.VAL_EXP)
    if initialized.calculate_coef:
        for key in ("cgrnd", "dgrnd", "cgrnd_snow", "dgrnd_snow", "lambda_snow"):
            np.testing.assert_array_equal(getattr(state, key), getattr(initialized, key))


@pytest.mark.parametrize("permuted", [False, True], ids=["native_axes", "stable_PFT_remap"])
def test_actual_reader_selects_first_time_normalizes_axes_and_remaps_stable_ids(reference, tmp_path, permuted):
    raw = oracle.canonical_restart(reference, "retained", None)
    # Explicitly distinguish every PFT for this reader-only normalization test.
    raw["ptn"] = raw["ptn"] + .001 * np.arange(14)[None, None, :]
    raw["e_soil_lat"] = np.arange(14, dtype=np.float64)[None, :]
    path = tmp_path / "native_like.nc"
    oracle.write_native_like(path, raw, permuted=permuted)
    with xr.open_dataset(path, decode_times=False) as dataset:
        assert dataset.attrs["fixture_kind"] == "synthetic_native_like_NOT_IOIPSL_producer"
        assert dataset["cgrnd"].dims == ("time", "l_d", "z_g", "y", "x")
        assert dataset["ptn"].dims == ("time", "l_d", "z_f", "y", "x")
        assert dataset.sizes["time"] == 2
        np.testing.assert_array_equal(dataset["ptn"][1] - dataset["ptn"][0], 7.)
    actual = oracle.read_native_like(path, permuted=permuted)
    for key, value in raw.items():
        np.testing.assert_array_equal(actual[key], value, err_msg=key)
    if permuted:
        without_remap = oracle.read_native_like(path)
        assert not np.array_equal(without_remap["ptn"], raw["ptn"])


def test_retained_mixed_marker_is_rejected_after_actual_file_read(reference, tmp_path):
    path = tmp_path / "retained_mixed.nc"
    oracle.write_native_like(path, oracle.canonical_restart(reference, "retained_mixed", None))
    with pytest.raises(ValueError, match="restart soilcap_pft mixes val_exp"):
        oracle.run_chain(reference, path)


def test_strict_recurrence_reader_still_rejects_missing_field(reference, tmp_path):
    path = tmp_path / "missing_cgrnd.nc"
    oracle.write_native_like(path, oracle.canonical_restart(reference, "absent", "cgrnd"))
    assert "cgrnd" not in oracle.read_native_like(path)
    with pytest.raises(KeyError, match="missing THERMOSOIL restart variables:.*cgrnd"):
        thermosoil.read_thermosoil_restart_state(path)


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_created_state_reaches_exact_next_profile_energy_operands_and_source_guard(reference, tmp_path, mode):
    path = tmp_path / "missing_cgrnd.nc"
    oracle.write_native_like(path, oracle.canonical_restart(reference, "absent", "cgrnd"))
    arrays, _, initialized, state = oracle.run_chain(reference, path)
    payload, actual = oracle.next_component(reference, arrays, initialized, state, mode)
    assert actual.keys() == reference["component_expected"].keys()
    for name, encoded in reference["component_expected"].items():
        np.testing.assert_allclose(actual[name], oracle.codec.decode(encoded),
            rtol=reference["rtol"], atol=reference["atol"], err_msg=name)
    for name in ("cgrnd", "dgrnd", "cgrnd_snow", "dgrnd_snow", "lambda_snow"):
        np.testing.assert_array_equal(payload[name], getattr(initialized, name))
    for operand, field in (("soilcap_initial", "soilcap"), ("temp_sol_beg", "temp_sol_beg")):
        np.testing.assert_array_equal(payload[operand], getattr(initialized, field))
    np.testing.assert_array_equal(payload["pcapa_en_previous"], arrays["initial.pcapa_en"])
    assert reference["full_component_source_rejection"]["returncode"] != 0
    rejection = oracle.full_component_rejection(reference, arrays, initialized, state, mode)
    assert rejection["source_guard_preserved"]
    assert not rejection["downstream_published"]


@pytest.mark.external_data
def test_original_source_identity_is_separate_from_portable_numerical_gate(reference):
    missing = [name for name in reference["source_identity"] if not (ROOT / name).is_file()]
    if missing:
        pytest.skip("Original-source identity only; portable numerical checks remain active")
    for name, expected in reference["source_identity"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
