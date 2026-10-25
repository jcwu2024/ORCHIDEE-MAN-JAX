"""Source-order ae carry through real HYDROL components and shared writer.

Fortran hydrol.f90:5843 adds subsinksoil; 9014 masks tiles; 9066
weights bare fractions; 8665-8723 reads that state at the next split.
The vector includes positive/negative/zero residual, bare fractions 0/1,
partial cover, an absent tile, and a no-vegetation point. No trace assets.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_orchidee.driver.fast_state import fast_state_from_previous_fields
from jax_orchidee.driver.orchestration import (
    DriverPreviousStepStatePacket,
    _previous_state_component_fields,
    _runtime_next_state_from_components,
)
from jax_orchidee.sechiba.hydrol import (
    HydrolFirstStepModuleClosure,
    build_mineral_cwrr_tables,
    hydrol_module_diagnostics,
    hydrol_module_explicit_step,
    hydrol_module_outputs,
    hydrol_split_soil_fluxes,
)


@pytest.fixture(scope="module")
def actual_chain():
    npts, nvm, nstm, nslm = 7, 4, 6, 7
    pft = np.tile([0.1, 0.3, 0.2, 0.4], (npts, 1))
    cover = pft * np.array([1.0, 0.6, 0.8, 0.75])
    cover[3] = pft[3]  # covered non-bare PFTs: frac_bare = 0
    cover[4] = 0.0  # frac_bare = 1
    pft[5, 1] += pft[5, 2]
    pft[5, 2] = 0.0
    cover[5] = pft[5] * np.array([1.0, 0.6, 0.8, 0.75])
    pft[6] = cover[6] = 0.0
    tiles = np.tile([0.4, 0.2, 0.0, 0.4, 0.0, 0.0], (npts, 1))
    tiles[5] = [0.6, 0.0, 0.0, 0.4, 0.0, 0.0]
    point = lambda value: np.full(npts, value, dtype=np.float64)
    tile = lambda value: np.full((npts, nstm), value, dtype=np.float64)
    mc = np.broadcast_to(np.linspace(0.30, 0.42, nslm)[None, :, None],
                         (npts, nslm, nstm)).copy()
    inputs = dict(
        precip_rain=point(0.4), vevapwet=np.zeros_like(pft),
        veget=cover, veget_max=pft, qsintmax=np.zeros_like(pft),
        qsintveg=np.zeros_like(pft), tot_melt=point(0.0),
        vegtot=pft.sum(axis=1), throughfall_by_pft=np.zeros(nvm),
        pref_soil_veg=np.array([1, 1, 2, 4], dtype=np.int32),
        soiltile=tiles, vevapflo=point(0.0), flood_frac=point(0.0),
        flood_res=point(0.0), subsinksoil=np.array([0.04, -0.02, 0., .04, .04, .04, .04]),
        vevapnu=point(0.05), vevapnu_pft=np.zeros_like(pft),
        transpir=np.zeros_like(pft), humrel=np.ones_like(pft),
        humrelv=np.zeros((npts, nvm, nstm)),
        us=np.zeros((npts, nvm, nstm, nslm)), ae_ns=tile(0.0),
        evap_bare_lim=point(0.2), evap_bare_lim_ns=tile(0.02),
        water2infilt=tile(0.02), reinfiltration_soil=tile(0.0),
        mc=mc, mcl=mc.copy(), profil_froz=np.zeros_like(mc),
        mcr=point(0.057), mcs=point(0.50),
        dz_mm=np.array([0., 3., 6., 8., 10., 12., 14.]),
        zz_mm=np.array([1., 4., 10., 18., 28., 40., 54.]),
        dt_days=1.0 / 48.0, free_drain_coef=tile(1.0),
        resolv=np.ones((npts, nstm), dtype=bool), kfact_root=np.ones_like(mc),
        ks=point(10.0), kfact=np.ones(nslm), tmc=tile(0.0),
        njsc=np.full(npts, 2, dtype=np.int32), reinf_slope=point(0.25),
        zmaxh_m=2.0, zwt_force=tile(1.0e20),
    )
    tables = build_mineral_cwrr_tables(
        njsc=inputs["njsc"], mcs=inputs["mcs"], z_m=inputs["zz_mm"] / 1000.0)
    module = hydrol_module_explicit_step(**inputs, mineral_tables=tables)
    nroot = np.zeros((npts, nvm, nslm))
    nroot[:, :, 1:] = 1.0 / (nslm - 1)
    diagnostic_inputs = dict(
        nroot=nroot, dz_mm=inputs["dz_mm"],
        dh_mm=np.array([1., 3., 6., 8., 10., 12., 14.]),
        njsc=inputs["njsc"], soiltile=tiles, vegtot=inputs["vegtot"],
        mcs=inputs["mcs"], mineral_tables=tables,
        vevapnu=inputs["vevapnu"], tot_melt=inputs["tot_melt"],
    )
    diagnostics = hydrol_module_diagnostics(module, **diagnostic_inputs)
    closure = HydrolFirstStepModuleClosure(
        module=module, outputs=hydrol_module_outputs(module, soiltile=tiles),
        diagnostics=diagnostics, thermosoil_moisture=None,
        snow_state=None, snow_step=None, module_inputs=inputs,
        diagnostic_inputs=diagnostic_inputs, missing_inputs=(), provenance=(), notes=(),
    )
    # Source hydrol_vegupd:5191-5248, computed independently from inputs.
    bare = np.zeros_like(pft)
    bare[:, 0] = (pft[:, 0] > 1e-8).astype(float)
    np.divide(cover[:, 1:], pft[:, 1:], out=bare[:, 1:], where=pft[:, 1:] > 1e-8)
    bare[:, 1:] = np.where(pft[:, 1:] > 1e-8, 1.0 - bare[:, 1:], 0.)
    frac = np.zeros_like(tiles)
    for jv, jst in enumerate(inputs["pref_soil_veg"] - 1):
        denom = tiles[:, jst] * inputs["vegtot"]
        frac[:, jst] += np.divide(pft[:, jv] * bare[:, jv], denom,
                                  out=np.zeros(npts), where=denom > 1e-8)
    np.testing.assert_allclose(module.vegupd.frac_bare_ns, frac, rtol=0., atol=1e-15)
    # Independent source equation, not a manufactured split/diagnostic seam.
    surface = np.asarray(module.split.ae_ns) + np.asarray(module.flood.subsinksoil)[:, None]
    expected = (surface * (tiles > 1e-8) * frac
                * (inputs["vegtot"] > 1.0e-8)[:, None])
    return inputs, closure, surface, expected


def _carry(closure, representation):
    if representation == "fast":
        previous = fast_state_from_previous_fields(
            tstep=0, fields_by_component={}, provenance_by_component={})
    else:
        previous = DriverPreviousStepStatePacket(0, {}, {})
    state = _runtime_next_state_from_components(
        tstep=1, previous_state=previous, enerbil_local=None,
        diffuco_local=None, thermosoil_module=None, condveg_module=None,
        hydrol_module=closure,
    )
    return _previous_state_component_fields(state, "hydrol_previous_step_state")


def test_actual_surface_accounts_residual_and_water_extraction(actual_chain):
    inputs, closure, expected_surface, _ = actual_chain
    module = closure.module
    np.testing.assert_allclose(module.flood.subsinksoil, inputs["subsinksoil"], rtol=0., atol=0.)
    for jst, tile in enumerate(module.soil.tile_results):
        np.testing.assert_allclose(tile.surface.ae_ns, expected_surface[:, jst], rtol=0., atol=1e-15)
        raw_extract = (np.maximum(np.asarray(module.split.ae_ns)[:, jst], 0.)
                       + np.maximum(inputs["subsinksoil"], 0.))
        np.testing.assert_allclose(tile.surface.water2extract,
                                   raw_extract - np.asarray(tile.surface.temp), atol=1e-15)
    assert np.any(np.asarray(module.vegupd.frac_bare_ns) == 0.)
    assert np.any(np.asarray(module.vegupd.frac_bare_ns) == 1.)


def test_actual_module_diagnostic_retains_post_surface_ae(actual_chain):
    _, closure, _, expected = actual_chain
    np.testing.assert_allclose(closure.diagnostics.ae_ns, expected, rtol=0., atol=1e-15)


@pytest.mark.parametrize("representation", ["packet", "fast"])
def test_shared_runtime_writer_retains_post_diagnostic_ae(actual_chain, representation):
    _, closure, _, expected = actual_chain
    fields = _carry(closure, representation)
    np.testing.assert_allclose(fields["ae_ns"], expected, rtol=0., atol=1e-15)
    np.testing.assert_allclose(fields["ae_ns"], closure.diagnostics.ae_ns, rtol=0., atol=0.)


@pytest.mark.parametrize("representation", ["packet", "fast"])
@pytest.mark.parametrize("compiled", [False, True])
def test_real_carry_reaches_next_split_old_evaporation(actual_chain, representation, compiled):
    inputs, closure, _, expected = actual_chain
    fields = _carry(closure, representation)
    module = closure.module
    # Disable the alternate beta branch so the actual previous ae is read.
    def next_split(ae):
        return hydrol_split_soil_fluxes(
            veget_max=inputs["veget_max"], soiltile=inputs["soiltile"],
            precisol=module.flood.precisol, vevapnu=inputs["vevapnu"],
            vevapnu_pft=inputs["vevapnu_pft"], transpir=inputs["transpir"],
            humrel=closure.diagnostics.humrel, humrelv=closure.diagnostics.humrelv,
            us=closure.diagnostics.us, ae_ns=ae,
            evap_bare_lim=np.zeros_like(inputs["vegtot"]),
            evap_bare_lim_ns=np.zeros_like(inputs["soiltile"]),
            frac_bare_ns=module.vegupd.frac_bare_ns,
            tot_bare_soil=np.sum(np.asarray(module.vegupd.frac_bare) * inputs["veget_max"], axis=1),
            vegetmax_soil=module.vegupd.vegetmax_soil, vegtot=inputs["vegtot"],
            pref_soil_veg=inputs["pref_soil_veg"],
        )
    result = (jax.jit(next_split) if compiled else next_split)(jnp.asarray(fields["ae_ns"]))
    old = np.sum(expected * inputs["soiltile"] * inputs["vegtot"][:, None], axis=1)
    np.testing.assert_allclose(result.vevapnu_old, old, rtol=0., atol=1e-15)
    # Also verify the source fallback for negative old evaporation and no veg.
    expected_next = expected.copy()
    frac = np.asarray(module.vegupd.frac_bare_ns)
    total_bare = np.sum(np.asarray(module.vegupd.frac_bare) * inputs["veget_max"], axis=1)
    for ji in range(expected.shape[0]):
        for jst in range(expected.shape[1]):
            if old[ji] > 1e-8:
                expected_next[ji, jst] = (0. if expected[ji, jst] > 1000. * old[ji]
                    else expected[ji, jst] * inputs["vevapnu"][ji] / old[ji])
            elif frac[ji, jst] > 1e-8:
                expected_next[ji, jst] = (inputs["vevapnu"][ji] * frac[ji, jst] / total_bare[ji]
                                          if total_bare[ji] > 1e-8 else 0.)
    np.testing.assert_allclose(result.ae_ns, expected_next, rtol=0., atol=1e-15)


def test_grid_total_adds_residual_once(actual_chain):
    inputs, closure, _, _ = actual_chain
    np.testing.assert_allclose(
        closure.diagnostics.vevapnu,
        inputs["vevapnu"] + inputs["subsinksoil"] * inputs["vegtot"],
        rtol=0., atol=1e-15,
    )
