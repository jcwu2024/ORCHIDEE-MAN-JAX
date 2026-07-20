import numpy as np
import pytest

from jax_orchidee.sticslai_completion import (
    STICS_FALSE_FIELDS,
    STICS_ONE_REAL_FIELDS,
    STICS_TRUE_FIELDS,
    STICS_ZERO_INTEGER_FIELDS,
    STICS_ZERO_REAL_FIELDS,
    SticsInitPftParameters,
    stics_init_source_routed,
)


PARAMETER_FIELDS = (
    "stpltger",
    "R_stamflax",
    "R_stlaxsen",
    "R_stsenlan",
    "R_stlevdrp",
    "R_stflodrp",
    "R_stdrpmat",
    "R_stdrpdes",
    "R_stlevamf",
)


def _parameters(nvm, *, ok_laidev=None, codlainet=None, nbox=None, vlaimax=None):
    sequence = np.arange(nvm, dtype=np.float64)
    return SticsInitPftParameters(
        ok_laidev=np.zeros(nvm, dtype=bool) if ok_laidev is None else ok_laidev,
        sp_codlainet=np.zeros(nvm, dtype=np.int32) if codlainet is None else codlainet,
        sp_stpltger=10.0 + sequence,
        sp_stamflax=20.0 + sequence,
        sp_stlaxsen=30.0 + sequence,
        sp_stsenlan=40.0 + sequence,
        sp_stlevdrp=50.0 + sequence,
        sp_stflodrp=5.0 + sequence,
        sp_stdrpmat=60.0 + sequence,
        sp_stdrpdes=70.0 + sequence,
        sp_stlevamf=80.0 + sequence,
        sp_nbox=np.zeros(nvm, dtype=np.int32) if nbox is None else nbox,
        sp_vlaimax=np.ones(nvm) if vlaimax is None else vlaimax,
    )


def _state(npts, nvm, nboxmax=5):
    state = {}
    for name in STICS_ZERO_REAL_FIELDS + STICS_ONE_REAL_FIELDS + PARAMETER_FIELDS + ("stlevflo",):
        state[name] = np.full((npts, nvm), 7.5)
    for name in STICS_ZERO_INTEGER_FIELDS + ("nrecbutoir",):
        state[name] = np.full((npts, nvm), 7, dtype=np.int32)
    for name in STICS_FALSE_FIELDS + STICS_TRUE_FIELDS:
        state[name] = np.full((npts, nvm), False)
    state.update(
        v_dltams=np.full((npts, nvm, 60), 7.5),
        histgrowthset=np.full((npts, nvm, 300, 5), 7.5),
        hist_sencourset=np.full((npts, nvm), 7, dtype=np.int32),
        hist_latestset=np.full((npts, nvm), 7, dtype=np.int32),
        doyhiststset=np.full((npts, nvm), 7, dtype=np.int32),
        box_ulai=np.full((nvm, nboxmax), 7.5),
    )
    for name in ("box_ndays", "box_lai", "box_lairem", "box_tdev", "box_biom", "box_biomrem", "box_durage", "box_somsenbase"):
        dtype = np.int32 if name == "box_ndays" else np.float64
        state[name] = np.full((npts, nvm, nboxmax), 7, dtype=dtype)
    return state


def test_stics_init_applies_global_or_recycle_mask_for_extensible_nvm():
    state = _state(2, 4)
    recycle = np.array([[False, True, False, False], [False, False, False, True]])
    result = stics_init_source_routed(
        state=state,
        f_crop_init=False,
        f_crop_recycle=recycle,
        parameters=_parameters(4),
    )

    selected = recycle
    np.testing.assert_array_equal(result.state["in_cycle"][selected], False)
    np.testing.assert_array_equal(result.state["f_sen_lai"][selected], True)
    np.testing.assert_allclose(result.state["swfac"][selected], 1.0)
    np.testing.assert_array_equal(result.state["nrecbutoir"][selected], 999)
    np.testing.assert_allclose(result.state["stpltger"][selected], [11.0, 13.0])
    np.testing.assert_allclose(result.state["stlevflo"][selected], [45.0, 45.0])
    np.testing.assert_allclose(result.state["v_dltams"][selected], 0.0)
    np.testing.assert_allclose(result.state["zrac"][~selected], 7.5)
    np.testing.assert_array_equal(result.f_crop_recycle, np.zeros((2, 4), dtype=bool))
    assert result.f_crop_init is False
    np.testing.assert_allclose(state["zrac"], 7.5)


def test_stics_init_global_init_resets_every_pft_without_pft14_shape():
    result = stics_init_source_routed(
        state=_state(3, 5),
        f_crop_init=True,
        f_crop_recycle=np.zeros((3, 5), dtype=bool),
        parameters=_parameters(5),
    )

    assert result.state["zrac"].shape == (3, 5)
    np.testing.assert_allclose(result.state["zrac"], 0.0)
    np.testing.assert_allclose(result.state["tursla"], 1.0)
    np.testing.assert_array_equal(result.state["nger"], 0)
    np.testing.assert_allclose(result.state["R_stlevamf"][0], np.arange(5) + 80.0)


def test_stics_init_resolves_history_box_and_box_ulai_source_arms():
    parameters = _parameters(
        3,
        ok_laidev=np.array([False, True, False]),
        codlainet=np.array([0, 2, 3], dtype=np.int32),
        nbox=np.array([1, 4, 3], dtype=np.int32),
        vlaimax=np.array([1.0, 2.0, 1.5]),
    )
    recycle = np.array([[False, True, False], [False, False, False]])
    result = stics_init_source_routed(
        state=_state(2, 3, nboxmax=5),
        f_crop_init=False,
        f_crop_recycle=recycle,
        parameters=parameters,
    )

    np.testing.assert_allclose(result.state["histgrowthset"][0, 1], 0.0)
    np.testing.assert_allclose(result.state["histgrowthset"][1, 1], 7.5)
    np.testing.assert_allclose(result.state["box_lai"][0, 1], 0.0)
    np.testing.assert_allclose(result.state["box_lai"][1, 1], 7.0)
    np.testing.assert_allclose(result.state["box_ulai"][0], 0.0)
    np.testing.assert_allclose(result.state["box_ulai"][1], [1.0, 1.5, 2.0, 2.5, 0.0])
    np.testing.assert_allclose(result.state["box_ulai"][2], [1.0, 1.5, 2.25, 0.0, 0.0])


def test_stics_init_requires_source_shapes_and_explicit_pft_parameters():
    state = _state(1, 3)
    with pytest.raises(ValueError, match=r"shape \(nvm,\)"):
        stics_init_source_routed(
            state=state,
            f_crop_init=True,
            f_crop_recycle=np.zeros((1, 3), dtype=bool),
            parameters=_parameters(2),
        )
    state["v_dltams"] = np.zeros((1, 3, 59))
    with pytest.raises(ValueError, match="v_dltams"):
        stics_init_source_routed(
            state=state,
            f_crop_init=True,
            f_crop_recycle=np.zeros((1, 3), dtype=bool),
            parameters=_parameters(3),
        )
