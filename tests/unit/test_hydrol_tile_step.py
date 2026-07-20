from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (  # noqa: E402
    build_mineral_cwrr_tables,
    hydrol_mc_to_mcl,
    hydrol_soil_explicit_solve_step,
    hydrol_soil_coef_mineral_profile_from_tables,
    hydrol_soil_infilt_explicit,
    hydrol_soil_setup_coefficients,
    hydrol_soil_surface_water_setup,
    hydrol_soil_tile_explicit_step,
)


def _tile_inputs():
    return {
        "water2infilt": np.asarray([0.10], dtype=np.float64),
        "ae_ns_tile": np.asarray([0.06], dtype=np.float64),
        "subsinksoil": np.asarray([0.02], dtype=np.float64),
        "precisol_ns_tile": np.asarray([0.50], dtype=np.float64),
        "reinfiltration_soil": np.asarray([0.03], dtype=np.float64),
        "is_crop_soil": False,
        "mc": np.asarray([[0.22, 0.25, 0.28, 0.31]], dtype=np.float64),
        "mcl": np.asarray([[0.22, 0.25, 0.28, 0.31]], dtype=np.float64),
        "profil_froz": np.asarray([[0.0, 0.05, 0.10, 0.15]], dtype=np.float64),
        "mcr": np.asarray([0.057], dtype=np.float64),
        "mcs": np.asarray([0.50], dtype=np.float64),
        "a": np.asarray([[0.10, 0.09, 0.08, 0.07]], dtype=np.float64),
        "b": np.asarray([[0.10, 0.12, 0.14, 0.16]], dtype=np.float64),
        "d": np.asarray([[4.0, 4.5, 5.0, 5.5]], dtype=np.float64),
        "k": np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=np.float64),
        "dz_mm": np.asarray([0.0, 3.0, 6.0, 6.0], dtype=np.float64),
        "dt_days": 1.0 / 48.0,
        "free_drain_coef": np.asarray([1.0], dtype=np.float64),
        "rootsink": np.asarray([[0.0, 0.01, 0.01, 0.0]], dtype=np.float64),
        "resolv": np.asarray([True]),
        "mask_soiltile": np.asarray([1.0], dtype=np.float64),
        "kfact_root": np.asarray([[1.0, 1.1, 1.2, 1.3]], dtype=np.float64),
        "ks": np.asarray([10.0], dtype=np.float64),
        "kfact": np.asarray([1.0, 0.8, 0.6, 0.4], dtype=np.float64),
        "reinf_slope": np.asarray([0.25], dtype=np.float64),
    }


def test_hydrol_soil_tile_explicit_step_composes_existing_source_kernels():
    inputs = _tile_inputs()
    result = hydrol_soil_tile_explicit_step(**inputs)

    surface = hydrol_soil_surface_water_setup(
        water2infilt=inputs["water2infilt"],
        ae_ns_tile=inputs["ae_ns_tile"],
        subsinksoil=inputs["subsinksoil"],
        precisol_ns_tile=inputs["precisol_ns_tile"],
        reinfiltration_soil=inputs["reinfiltration_soil"],
        is_crop_soil=inputs["is_crop_soil"],
    )
    infilt = hydrol_soil_infilt_explicit(
        mc=inputs["mc"],
        flux_infilt=surface.flux_infilt,
        k=inputs["k"],
        dz_mm=inputs["dz_mm"],
        dt_days=inputs["dt_days"],
        mcs=inputs["mcs"],
        ks=inputs["ks"],
        kfact=inputs["kfact"],
        kfact_root=inputs["kfact_root"],
    )
    setup = hydrol_soil_setup_coefficients(
        a=inputs["a"],
        d=inputs["d"],
        dz_mm=inputs["dz_mm"],
        dt_days=inputs["dt_days"],
        free_drain_coef=inputs["free_drain_coef"],
    )
    solve = hydrol_soil_explicit_solve_step(
        setup=setup,
        mcl_before=hydrol_mc_to_mcl(infilt.mc, inputs["profil_froz"], inputs["mcr"]),
        mc_before=infilt.mc,
        profil_froz=inputs["profil_froz"],
        mcr=inputs["mcr"],
        b=inputs["b"],
        dz_mm=inputs["dz_mm"],
        dt_days=inputs["dt_days"],
        free_drain_coef=inputs["free_drain_coef"],
        flux_top=surface.flux_top,
        rootsink=inputs["rootsink"],
        resolv=inputs["resolv"],
        mask_soiltile=inputs["mask_soiltile"],
        k_bottom=inputs["k"][:, -1],
    )

    np.testing.assert_allclose(np.asarray(result.surface.flux_infilt), np.asarray(surface.flux_infilt))
    np.testing.assert_allclose(np.asarray(result.infilt.mc), np.asarray(infilt.mc))
    np.testing.assert_allclose(np.asarray(result.setup.f), np.asarray(setup.f))
    np.testing.assert_allclose(np.asarray(result.solve.rhs), np.asarray(solve.rhs))
    np.testing.assert_allclose(np.asarray(result.solve.mc_after_mcl_update), np.asarray(solve.mc_after_mcl_update))
    expected_reinfiltration = inputs["reinf_slope"] * np.asarray(infilt.ru_infilt)
    np.testing.assert_allclose(
        np.asarray(result.water2infilt_after_reinf),
        expected_reinfiltration,
    )
    np.testing.assert_allclose(
        np.asarray(result.ru_ns_after_reinf),
        np.asarray(infilt.ru_infilt) - expected_reinfiltration,
    )


def test_hydrol_soil_tile_explicit_step_handles_doponds_and_requires_reinf_slope():
    inputs = _tile_inputs()
    no_ponds = dict(inputs)
    no_ponds.pop("reinf_slope")
    with pytest.raises(ValueError, match="reinf_slope"):
        hydrol_soil_tile_explicit_step(**no_ponds)

    ponds = dict(no_ponds)
    ponds["doponds"] = True
    result = hydrol_soil_tile_explicit_step(**ponds)
    np.testing.assert_allclose(np.asarray(result.water2infilt_after_reinf), np.zeros(1))


def test_hydrol_soil_tile_explicit_step_can_rebuild_mineral_coefficients_in_source_order():
    inputs = _tile_inputs()
    z_m = np.asarray([0.00098, 0.00391, 0.00978, 0.02151], dtype=np.float64)
    tables = build_mineral_cwrr_tables(
        njsc=np.asarray([2]),
        mcs=inputs["mcs"],
        z_m=z_m,
    )

    coef_before = hydrol_soil_coef_mineral_profile_from_tables(
        mc=inputs["mc"],
        profil_froz=inputs["profil_froz"],
        kfact_root=inputs["kfact_root"],
        tables=tables,
        mcr=inputs["mcr"],
        mcs=inputs["mcs"],
        ok_freeze_cwrr=False,
    )
    surface = hydrol_soil_surface_water_setup(
        water2infilt=inputs["water2infilt"],
        ae_ns_tile=inputs["ae_ns_tile"],
        subsinksoil=inputs["subsinksoil"],
        precisol_ns_tile=inputs["precisol_ns_tile"],
        reinfiltration_soil=inputs["reinfiltration_soil"],
        is_crop_soil=inputs["is_crop_soil"],
    )
    infilt = hydrol_soil_infilt_explicit(
        mc=inputs["mc"],
        flux_infilt=surface.flux_infilt,
        k=coef_before.k,
        dz_mm=inputs["dz_mm"],
        dt_days=inputs["dt_days"],
        mcs=inputs["mcs"],
        ks=inputs["ks"],
        kfact=inputs["kfact"],
        kfact_root=inputs["kfact_root"],
    )
    coef_after = hydrol_soil_coef_mineral_profile_from_tables(
        mc=infilt.mc,
        profil_froz=inputs["profil_froz"],
        kfact_root=inputs["kfact_root"],
        tables=tables,
        mcr=inputs["mcr"],
        mcs=inputs["mcs"],
        ok_freeze_cwrr=False,
    )

    explicit = dict(inputs)
    explicit.update(a=coef_after.a, b=coef_after.b, d=coef_after.d, k=coef_before.k, k_after_infilt=coef_after.k)
    rebuilt = dict(inputs)
    for name in ("a", "b", "d", "k"):
        rebuilt.pop(name)
    rebuilt["mineral_tables"] = tables
    rebuilt["ok_freeze_cwrr"] = False

    explicit_result = hydrol_soil_tile_explicit_step(**explicit)
    rebuilt_result = hydrol_soil_tile_explicit_step(**rebuilt)

    assert rebuilt_result.coef_before_infilt is not None
    assert rebuilt_result.coef_after_infilt is not None
    np.testing.assert_allclose(np.asarray(rebuilt_result.coef_before_infilt.k), np.asarray(coef_before.k))
    np.testing.assert_allclose(np.asarray(rebuilt_result.coef_after_infilt.a), np.asarray(coef_after.a))
    np.testing.assert_allclose(np.asarray(rebuilt_result.infilt.mc), np.asarray(explicit_result.infilt.mc))
    np.testing.assert_allclose(np.asarray(rebuilt_result.setup.f), np.asarray(explicit_result.setup.f))
    np.testing.assert_allclose(np.asarray(rebuilt_result.solve.rhs), np.asarray(explicit_result.solve.rhs))
    np.testing.assert_allclose(
        np.asarray(rebuilt_result.solve.mc_after_mcl_update),
        np.asarray(explicit_result.solve.mc_after_mcl_update),
    )
