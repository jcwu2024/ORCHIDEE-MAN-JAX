from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from netCDF4 import Dataset

from jax_orchidee.driver.init import read_run_scalars
from jax_orchidee.parameters.pft_catalog import (
    build_pft_run_layout,
    build_selected_pft_run_layout,
    load_pft_catalog,
    pft_layout_netcdf_attributes,
    read_pft_layout_from_netcdf,
    remap_pft_axis,
    validate_active_capabilities,
)
from jax_orchidee.stomate.parameters import load_paper_case_stomate_parameter_bundle

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
CATALOG = ROOT / "configs" / "pft_catalogs" / "orchidee_man_paper_250919.json"


def _catalog():
    return load_pft_catalog(CATALOG)


def test_paper_catalog_has_stable_source_identities_and_named_layout() -> None:
    catalog = _catalog()

    assert catalog.catalog_id == "orchidee_man_paper_250919"
    assert len(catalog.entries) == 14
    assert catalog.entries[0].pft_id == "bare_soil"
    assert catalog.entries[0].fortran_pft_id == 1
    assert catalog.entries[-1].pft_id == "mangrove_pft14"
    assert catalog.entries[-1].fortran_pft_id == 14
    assert catalog.entries[-1].mtc_id == 2
    assert catalog.entries[-1].traits["is_peat"] is True
    assert catalog.layouts["paper_250919_legacy14"][-1] == "mangrove_pft14"


def test_paper_run_scalars_are_catalog_bound_without_numerical_change() -> None:
    scalars = read_run_scalars(CONFIG)

    assert scalars.pft_layout.layout_id == "paper_250919_legacy14"
    assert scalars.pft_ids[0] == "bare_soil"
    assert scalars.pft_ids[-1] == "mangrove_pft14"
    np.testing.assert_array_equal(scalars.fortran_pft_ids, np.arange(1, 15))
    np.testing.assert_array_equal(scalars.pft_to_mtc, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2])
    np.testing.assert_array_equal(
        scalars.active_pft_mask,
        [True, False, False, False, False, False, False, False, False, False, False, False, False, True],
    )
    np.testing.assert_array_equal(scalars.sechiba_vegmax, [0.0] * 13 + [1.0])
    assert scalars.pft_layout.index_for_id("mangrove_pft14") == 13


def test_stomate_parameter_bundle_retains_the_same_stable_layout() -> None:
    scalars = read_run_scalars(CONFIG)
    parameters = load_paper_case_stomate_parameter_bundle(CONFIG)

    assert parameters.pft_layout == scalars.pft_layout
    assert parameters.pft_ids == scalars.pft_ids
    np.testing.assert_array_equal(parameters.fortran_pft_ids, scalars.fortran_pft_ids)
    np.testing.assert_array_equal(parameters.pft_to_mtc, scalars.pft_to_mtc)


def test_permuted_layout_reads_each_parameter_from_its_fortran_source_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    source_ids = list(catalog_payload["layouts"]["paper_250919_legacy14"])
    source_ids[1], source_ids[-1] = source_ids[-1], source_ids[1]
    catalog_payload["layouts"]["paper_250919_permuted"] = source_ids
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")

    config_payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config_payload["pft_catalog"] = {
        "path": str(catalog_path),
        "layout_id": "paper_250919_permuted",
    }
    config_path = tmp_path / "permuted.yaml"
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    monkeypatch.setenv("ORCHIDEE_REPO_ROOT", str(ROOT))

    parameters = load_paper_case_stomate_parameter_bundle(config_path)

    assert parameters.pft_ids[1] == "mangrove_pft14"
    assert parameters.fortran_pft_ids[1] == 14
    assert parameters.vcmax25[1] == pytest.approx(63.2061836)
    assert parameters.pft_ids[-1] == "canonical_pft02"
    assert parameters.fortran_pft_ids[-1] == 2
    assert parameters.vcmax25[-1] == pytest.approx(50.0)


def test_removing_inactive_slot_preserves_active_state_by_stable_id() -> None:
    catalog = _catalog()
    full = build_pft_run_layout(
        catalog,
        layout_id="paper_250919_legacy14",
        fractions=[0.0] * 13 + [1.0],
    )
    selected_ids = tuple(pft_id for pft_id in full.pft_ids if pft_id != "canonical_pft13")
    compact = build_selected_pft_run_layout(
        catalog,
        layout_id="without_inactive_pft13",
        pft_ids=selected_ids,
        fractions=[0.0] * 12 + [1.0],
    )
    state = np.arange(2 * 14 * 3, dtype=np.float64).reshape(2, 14, 3)

    remapped = remap_pft_axis(state, source=full, target=compact, axis=1)

    np.testing.assert_array_equal(
        remapped[:, compact.index_for_id("mangrove_pft14"), :],
        state[:, full.index_for_id("mangrove_pft14"), :],
    )


def test_two_supported_pfts_coexist_and_permutation_roundtrip_by_id() -> None:
    catalog = _catalog()
    original = build_selected_pft_run_layout(
        catalog,
        layout_id="two_vegetated",
        pft_ids=("bare_soil", "canonical_pft02", "mangrove_pft14"),
        fractions=(0.0, 0.4, 0.6),
    )
    permuted = build_selected_pft_run_layout(
        catalog,
        layout_id="two_vegetated_permuted",
        pft_ids=("bare_soil", "mangrove_pft14", "canonical_pft02"),
        fractions=(0.0, 0.6, 0.4),
    )
    state = np.asarray([[0.0, 20.0, 140.0], [0.0, 21.0, 141.0]])

    reordered = remap_pft_axis(state, source=original, target=permuted, axis=1)
    restored = remap_pft_axis(reordered, source=permuted, target=original, axis=1)

    np.testing.assert_array_equal(restored, state)
    np.testing.assert_array_equal(reordered[:, permuted.index_for_id("canonical_pft02")], [20.0, 21.0])
    np.testing.assert_array_equal(reordered[:, permuted.index_for_id("mangrove_pft14")], [140.0, 141.0])


def test_structural_only_pft_cannot_be_promoted_by_shape_compatibility() -> None:
    catalog = _catalog()
    unvalidated = build_selected_pft_run_layout(
        catalog,
        layout_id="unvalidated_natural_pft",
        pft_ids=("bare_soil", "canonical_pft02"),
        fractions=(0.0, 1.0),
    )

    with pytest.raises(
        ValueError,
        match="canonical_pft02:scientific_support=structural_only",
    ):
        validate_active_capabilities(catalog, unvalidated)


def test_active_unvalidated_crop_capability_is_rejected() -> None:
    catalog = _catalog()
    crop = build_selected_pft_run_layout(
        catalog,
        layout_id="unsupported_crop",
        pft_ids=("bare_soil", "canonical_pft12"),
        fractions=(0.0, 1.0),
    )

    with pytest.raises(ValueError, match="canonical_pft12:stomate_crop=unvalidated"):
        validate_active_capabilities(catalog, crop)


def test_restart_layout_metadata_roundtrips_and_rejects_partial_identity(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    source = build_selected_pft_run_layout(
        catalog,
        layout_id="compact_paper",
        pft_ids=("bare_soil", "mangrove_pft14"),
        fractions=(0.0, 1.0),
    )
    path = tmp_path / "restart.nc"
    with Dataset(path, "w") as dataset:
        dataset.setncatts(pft_layout_netcdf_attributes(source))

    restored = read_pft_layout_from_netcdf(path, catalog)

    assert restored.layout_id == "compact_paper"
    assert restored.pft_ids == source.pft_ids
    assert restored.fortran_pft_ids == (1, 14)
    with Dataset(path, "r+") as dataset:
        dataset.delncattr("orchidee_jax_mtc_ids_json")
    with pytest.raises(ValueError, match="metadata is incomplete"):
        read_pft_layout_from_netcdf(path, catalog)


def test_legacy_restart_layout_requires_an_explicit_named_source(tmp_path: Path) -> None:
    catalog = _catalog()
    path = tmp_path / "legacy.nc"
    with Dataset(path, "w"):
        pass

    with pytest.raises(ValueError, match="explicit legacy_layout_id"):
        read_pft_layout_from_netcdf(path, catalog)

    restored = read_pft_layout_from_netcdf(
        path,
        catalog,
        legacy_layout_id="paper_250919_legacy14",
    )
    assert restored.pft_ids == catalog.layouts["paper_250919_legacy14"]


def test_restart_layout_metadata_rejects_catalog_identity_mismatch(tmp_path: Path) -> None:
    catalog = _catalog()
    source = build_pft_run_layout(
        catalog,
        layout_id="paper_250919_legacy14",
        fractions=[0.0] * 13 + [1.0],
    )
    attributes = pft_layout_netcdf_attributes(source)
    attributes["orchidee_jax_mtc_ids_json"] = json.dumps([1] * source.n_pft)
    path = tmp_path / "mismatched.nc"
    with Dataset(path, "w") as dataset:
        dataset.setncatts(attributes)

    with pytest.raises(ValueError, match="MTC identities disagree"):
        read_pft_layout_from_netcdf(path, catalog)
