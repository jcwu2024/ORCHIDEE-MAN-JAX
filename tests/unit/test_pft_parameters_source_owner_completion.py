from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_orchidee.stomate.parameters import (
    FORTRAN_UNDEF_INT,
    apply_configured_pft_constraints,
    build_age_class_layout,
    crop_rotation_sowing_keys,
    derive_pft_physiology_labels,
    load_pft14_parameters,
    resolve_age_class_config,
    resolve_pft_parameter_main,
    resolve_pft_parameter_section_switches,
    select_humcste_reference,
)


ROOT = Path(__file__).resolve().parents[2]


def test_pft_parameter_main_supports_dynamic_nvm_and_noop_reentry():
    mapping = resolve_pft_parameter_main(
        first_call=True,
        nvm=5,
        nvmc=18,
        pft_to_mtc=[1, 10, 11, 14, 18],
    )

    assert np.array_equal(mapping, [1, 10, 11, 14, 18])
    assert resolve_pft_parameter_main(first_call=False, nvm=5, nvmc=18) is None
    assert np.array_equal(
        resolve_pft_parameter_main(first_call=True, nvm=4, nvmc=4),
        [1, 2, 3, 4],
    )


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ([FORTRAN_UNDEF_INT, 2], "required"),
        ([1, 19], "outside"),
        ([2, 3], "first PFT"),
        ([1, 1], "only the first"),
    ],
)
def test_pft_parameter_main_rejects_each_fortran_guard(mapping, message):
    with pytest.raises(ValueError, match=message):
        resolve_pft_parameter_main(first_call=True, nvm=2, nvmc=18, pft_to_mtc=mapping)


@pytest.mark.parametrize(
    ("zmaxh", "expected"),
    [
        (2.0, [1.0, 3.0]),
        (4.0, [10.0, 30.0]),
        (3.5, [1.0, 3.0]),
    ],
)
def test_humcste_selection_covers_2m_4m_and_fortran_fallback(zmaxh, expected):
    result = select_humcste_reference(zmaxh, [1, 3], [1.0, 2.0, 3.0], [10.0, 20.0, 30.0])

    assert np.array_equal(result, expected)


def test_physiology_labels_cover_both_arms_for_arbitrary_nvm():
    labels = derive_pft_physiology_labels(
        leaf_tab=[1, 2, 3, 4, 1],
        pheno_model=["none", "ncdgdd", "moi", "none", "moi"],
    )

    assert np.array_equal(labels.is_tree, [True, True, False, False, True])
    assert np.array_equal(labels.is_deciduous, [False, True, False, False, True])
    assert np.array_equal(labels.is_evergreen, [True, False, False, False, False])
    assert np.array_equal(labels.is_needleleaf, [False, True, False, False, False])


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ((True, True, True, True, True), (True, True, True, True)),
        ((True, True, False, False, False), (True, False, False, False)),
        ((False, True, True, True, True), (False, False, True, True)),
    ],
)
def test_parameter_section_switches_preserve_nested_fortran_conditions(flags, expected):
    result = resolve_pft_parameter_section_switches(
        ok_sechiba=flags[0],
        hydrol_cwrr=flags[1],
        offline_mode=flags[2],
        ok_stomate=flags[3],
        ok_bvoc=flags[4],
    )

    assert (
        result.load_sechiba,
        result.zero_offline_cwrr_throughfall,
        result.load_stomate,
        result.load_bvoc,
    ) == expected


def test_config_constraints_apply_managed_mask_and_nstm_where_without_guessing_reset():
    natural, soil = apply_configured_pft_constraints(
        natural=[True, True, False, True],
        is_grassland_manag=[False, True, False, False],
        pref_soil_veg=[1, 4, 7, 2],
        nstm=4,
        use_age_class=True,
    )

    assert np.array_equal(natural, [True, False, False, True])
    assert np.array_equal(soil, [1, 4, 4, 2])

    _, invalid_high_soil = apply_configured_pft_constraints(
        natural=[True, True],
        is_grassland_manag=[False, True],
        pref_soil_veg=[3, 9],
        nstm=21,
        use_age_class=False,
    )
    assert np.array_equal(invalid_high_soil, [3, 9])

    with pytest.raises(ValueError, match="managed grassland"):
        apply_configured_pft_constraints(
            [True, True], [False, False], [1, 2], nstm=6, use_age_class=True
        )


@pytest.mark.parametrize(
    ("cycles", "keys"),
    [
        (1, ("SP_IPLT0",)),
        (2, ("SP_IPLT0", "SP_IPLT1")),
        (3, ("SP_IPLT0", "SP_IPLT1", "SP_IPLT2")),
    ],
)
def test_crop_rotation_sowing_key_selection(cycles, keys):
    assert crop_rotation_sowing_keys(cycles) == keys


def test_age_class_config_covers_disabled_enabled_warning_and_bound_selection():
    disabled = resolve_age_class_config(4, use_age_class=False)
    assert disabled.nvmap == 4
    assert np.array_equal(disabled.agec_group, [1, 2, 3, 4])
    assert not disabled.read_age_class_bound
    assert not disabled.no_age_class_warning

    enabled = resolve_age_class_config(
        6,
        use_age_class=True,
        nvmap=3,
        agec_group=[1, 2, 2, 3, 3, 3],
        use_bound_spa=False,
    )
    assert enabled.nvmap == 3
    assert enabled.read_age_class_bound
    assert not enabled.no_age_class_warning

    warning = resolve_age_class_config(3, use_age_class=True, single_age_class=False)
    assert warning.no_age_class_warning
    assert not resolve_age_class_config(
        3, use_age_class=True, single_age_class=True, use_bound_spa=True
    ).read_age_class_bound


def test_age_class_layout_matches_contiguous_fortran_groups_for_dynamic_nvm():
    layout = build_age_class_layout(
        [1, 2, 2, 2, 2, 3, 4, 5],
        nvmap=5,
        is_tree=[False, True, True, True, True, False, False, False],
        nagec_tree=4,
        nagec_herb=1,
    )

    assert np.array_equal(layout.start_index, [0, 1, 5, 6, 7])
    assert np.array_equal(layout.nagec_pft, [1, 4, 1, 1, 1])


def test_age_class_layout_rejects_missing_and_inconsistent_groups():
    with pytest.raises(ValueError, match="group 2 has no PFT"):
        build_age_class_layout(
            [1, 1, 3], nvmap=3, is_tree=[False, False, True], nagec_tree=1, nagec_herb=2
        )
    with pytest.raises(ValueError, match="expected 3"):
        build_age_class_layout(
            [1, 2, 2], nvmap=2, is_tree=[False, True, True], nagec_tree=3, nagec_herb=1
        )


def test_pft14_continuous_parameter_override_remains_float64_source_truth():
    params = load_pft14_parameters(ROOT)

    assert params.pft_index_fortran == 14
    assert isinstance(params["VCMAX25"], float)
    assert params["VCMAX25"] == 63.2061836
    assert params["ALLOC_MIN"] == 0.2019211

