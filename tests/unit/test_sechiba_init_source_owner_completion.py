from __future__ import annotations

import numpy as np

from jax_orchidee.sechiba.initialize import (
    SechibaInitDimensions,
    sechiba_init_index_tables,
    sechiba_init_state,
    sechiba_prepare_thermosoil_pft_state,
)


def _dimensions() -> SechibaInitDimensions:
    return SechibaInitDimensions(
        npts=2,
        nvm=16,
        nslm=3,
        nstm=4,
        nnobio=1,
        itimetide=4,
        nflow=3,
        nexp=2,
        nctext=3,
        ncarb=3,
        nparts=2,
        nelements=2,
        nlitt=2,
        ndeep=2,
        ndoc=2,
        npool=2,
        nleafages=4,
        nlai=3,
        ngrnd=5,
        nsnow=3,
        npco2=3,
    )


def test_thirteen_active_allocation_success_arms_have_exact_shapes_without_fabricated_values():
    result = sechiba_init_state(l_first=True, dimensions=_dimensions())
    expected = {
        "tq_cdrag_pft": (2, 16),
        "mc_peat_above": (2,),
        "mc_croppeat_above": (2,),
        "mc_man_above": (2,),
        "soil_mc": (2, 3, 4),
        "wat_flux": (2, 3, 4),
        "drainage_per_soil": (2, 4),
        "runoff_per_soil": (2, 4),
        "runoff2peat": (2, 4),
        "erodepth": (2, 16),
        "precip2canopy": (2, 16),
        "precip2ground": (2, 16),
        "canopy2ground": (2, 16),
    }

    assert {name: result.allocation_shapes[name] for name in expected} == expected
    assert all(name in result.allocation_only_fields for name in expected)
    assert all(name not in result.state for name in expected)


def test_index_tables_follow_fortran_level_major_flattening_and_offsets():
    tables = sechiba_init_index_tables(
        index=np.array([2, 7], dtype=np.int32),
        kjpij=10,
        nvm=3,
        nstm=2,
        nnobio=1,
        nlai=2,
        ngrnd=2,
        nsnow=2,
        nslm=2,
        offset_omp=4,
        offset_mpi=1,
    )

    np.testing.assert_array_equal(np.asarray(tables["indexveg"]), [5, 10, 15, 20, 25, 30])
    np.testing.assert_array_equal(np.asarray(tables["indexlai0"]), [5, 10, 15, 20])
    np.testing.assert_array_equal(np.asarray(tables["indexlai"]), [5, 10, 15, 20, 25, 30])
    np.testing.assert_array_equal(np.asarray(tables["indexalb"]), [5, 10, 15, 20])


def test_hydrol_tile_mapping_is_dynamic_in_nvm_and_independent_of_vegetation_cover():
    mc = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    mcl = mc + 100.0
    soilmoist = np.arange(6, dtype=np.float64).reshape(2, 3)
    preferences = np.array([1, 4, 2, 4, 3], dtype=np.int32)
    ok_laidev = np.array([False, True, False, False, True])

    result = sechiba_prepare_thermosoil_pft_state(
        mc_layh_s=mc,
        mcl_layh_s=mcl,
        soilmoist=soilmoist,
        pref_soil_veg=preferences,
        ok_laidev=ok_laidev,
    )

    np.testing.assert_array_equal(np.asarray(result["mc_layh_pft"]), mc[:, :, preferences - 1])
    np.testing.assert_array_equal(np.asarray(result["mcl_layh_pft"]), mcl[:, :, preferences - 1])
    np.testing.assert_array_equal(
        np.asarray(result["soilmoist_pft"]), np.broadcast_to(soilmoist[:, :, None], (2, 3, 5))
    )
    np.testing.assert_array_equal(np.asarray(result["is_crop_soil"]), [False, False, True, True])
