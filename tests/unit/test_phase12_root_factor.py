"""Source conductivity normalization in the two maintained root-factor paths."""
import jax
import numpy as np

from jax_orchidee.sechiba import hydrol as h

USDA_KS = np.array([7128., 3501.6, 1060.8, 108., 60., 249.6, 314.4, 16.8, 62.4, 28.8, 4.8, 48.])
RTOL, ATOL = 1e-12, 1e-14


def source_cases():
    single_override = USDA_KS.copy()
    single_override[3] = 16000.
    return {"default": USDA_KS, "whole_vector_doubled": USDA_KS * 2, "texture4_override": single_override}


def root_inputs(zz_mm):
    pref = np.array([2, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 3, 2, 4], dtype=np.int32)
    area = np.array([
        [.16, .15, .12, .25, .16, .16],
        [.0, 1e-8, np.nextafter(1e-8, np.inf), .35, .33, .31999998],
        [.25, .15, .10, .15, .15, .20],
        [.12, .15, .18, .22, .15, .18],
    ])
    pft_share = np.array([.37, .29, .16, .41, .49, .46, .22, .37, .35, .40, .21, .12, .15, .15])
    vegetation = area[:, pref - 1] * pft_share[None]
    vegetation[3] = 0.
    return dict(
        veget_max=vegetation, veget=.8 * vegetation, vegtot=vegetation.sum(axis=1),
        soiltile=area, pref=pref, humcste=np.array([1., .7, 1.4, 1., .9, 1.3, 1., .75, 1.1, .9, 1.6, .8, 1.2, .65]),
        zz_mm=np.asarray(zz_mm), njsc=np.array([4, 8, 1, 11], dtype=np.int32),
    )


def independent_vegetmax(inputs):
    area = inputs["soiltile"]
    result = np.zeros((area.shape[0], len(inputs["pref"]), area.shape[1]))
    for pft, tile in enumerate(inputs["pref"] - 1):
        active = (area[:, tile] > 1e-8) & (inputs["vegtot"] > 1e-8)
        result[active, pft, tile] = inputs["veget_max"][active, pft] / area[active, tile]
    return result


def source_expression(inputs, ks, peat_hydro):
    vegetation = independent_vegetmax(inputs)
    area = inputs["soiltile"]
    result = np.ones((area.shape[0], len(inputs["zz_mm"]), area.shape[1]))
    # hydrol.f90:1210–1229, constantes_soil_var.f90:167,270–273.
    mineral = USDA_KS.max() / ks[inputs["njsc"] - 1]
    for pft, tile in enumerate(inputs["pref"][1:] - 1, start=1):
        base = USDA_KS.max() / 2120. if peat_hydro and tile in (3, 4) else mineral
        exponent = -vegetation[:, pft, tile, None] / 2.
        exponent = exponent * (inputs["humcste"][pft] * inputs["zz_mm"] / 1000. - 1.) / 2.
        factor = np.maximum(np.power(np.broadcast_to(base, (area.shape[0],))[:, None], exponent), 1.)
        result[:, :, tile] *= factor
    return result


def evaluate_root(module, inputs, ks, peat_hydro):
    params = {key: inputs[key] for key in ("soiltile", "humcste", "zz_mm", "njsc")}
    veg_args = {key: inputs[key] for key in ("veget", "veget_max", "vegtot")}
    veg = module.hydrol_vegupd_static_state(
        **veg_args, soiltile=inputs["soiltile"], pref_soil_veg=inputs["pref"],
    )
    eager = module.hydrol_kfact_root_from_vegupd(
        **params, ks=ks, peat_hydro=peat_hydro, pref_soil_veg=inputs["pref"], vegetmax_soil=veg.vegetmax_soil,
    )
    compiled = module._hydrol_vegupd_kfact_root_jit(
        **params, **veg_args, ks=ks, peat_hydro=peat_hydro, pref_indices=tuple(map(int, inputs["pref"] - 1)),
    )
    return np.asarray(eager), np.asarray(compiled[-1])


def test_root_conductivity_uses_fixed_usda_maximum_with_configured_ks():
    with jax.enable_x64():
        inputs = root_inputs([0., 300., 1000., 1800.])
        for name, ks in source_cases().items():
            for peat in (False, True):
                expected = source_expression(inputs, ks, peat)
                for actual in evaluate_root(h, inputs, ks, peat):
                    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, err_msg=f"{name}, peat={peat}")

    # Tile 6 uses the mineral root factor while tiles 4 and 5 use peat.
    default = source_expression(inputs, USDA_KS, False)
    peat = source_expression(inputs, USDA_KS, True)
    np.testing.assert_array_equal(default[:, :, 5], peat[:, :, 5])
    assert np.all(np.max(np.abs(default[:, :, 3:5] - peat[:, :, 3:5]), axis=(0, 1)) > 0.)
    np.testing.assert_array_equal(peat[1, :, :2], 1.)
    np.testing.assert_array_equal(peat[3], 1.)
    assert np.max(peat[1, :, 2]) > 1.
