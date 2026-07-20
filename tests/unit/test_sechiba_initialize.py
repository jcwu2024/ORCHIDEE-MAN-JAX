from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.sechiba.initialize import (
    SechibaInitDimensions,
    SechibaInitializationBoundaryError,
    SechibaInitializationError,
    SechibaInitializationOwners,
    SechibaInitializePFT14Switches,
    sechiba_initialize_full_irrigation_restart_transition,
    sechiba_init_state,
    sechiba_initialize_pft14_explicit,
)


def _dimensions(*, npts=2, nvm=16):
    return SechibaInitDimensions(
        npts=npts,
        nvm=nvm,
        nslm=3,
        nstm=6,
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
        ngrnd=4,
        nsnow=3,
        npco2=3,
    )


def _pft14(npts=2, nvm=16):
    value = np.zeros((npts, nvm), dtype=np.float64)
    value[:, 13] = np.linspace(0.35, 0.85, npts)
    return value


def test_owned_span_materializes_only_source_assigned_values_and_shapes():
    result = sechiba_init_state(l_first=True, dimensions=_dimensions())
    state = result.state

    assert result.l_first is False
    assert result.assignment_order.index("flood_res") < result.assignment_order.index("salinity")
    assert result.assignment_order.index("njsc") < result.assignment_order.index("temp_sol")
    assert result.assignment_order.index("gpp") < result.assignment_order.index("salinity")
    assert result.assignment_order.index("salinity") < result.assignment_order.index("DOC")
    np.testing.assert_array_equal(np.asarray(state["vegstress_old"]), np.ones((2, 16)))
    np.testing.assert_array_equal(np.asarray(state["salinity"]), np.full(2, 30.0))
    np.testing.assert_array_equal(np.asarray(state["tide_height"]), np.zeros((2, 4)))
    np.testing.assert_array_equal(np.asarray(state["shumdiag_man"]), np.ones((2, 3)))
    np.testing.assert_array_equal(np.asarray(state["DOC"]), np.zeros((2, 16, 2, 2, 2, 2)))
    np.testing.assert_array_equal(np.asarray(state["returnflow"]), np.zeros((2, 3)))
    assert "erodepth" in result.allocation_only_fields
    assert "erodepth" not in state
    assert "drysoil_frac" not in state
    assert "sechiba_init lines 1983-2553" in result.provenance[0]


def test_l_first_repeated_call_matches_fatal_fortran_branch():
    with pytest.raises(SechibaInitializationError, match="repeated call"):
        sechiba_init_state(l_first=False, dimensions=_dimensions())


def test_source_order_dispatch_updates_state_between_exact_owners_and_writes_outputs():
    calls: list[str] = []
    npts, nvm = 2, 16

    def owner(name, writes):
        def run(**kwargs):
            calls.append(name)
            assert kwargs["marker"] == len(calls)
            return writes

        return run

    slowproc = owner(
        "slowproc_initialize",
        {
            "veget_max": _pft14(npts, nvm),
            "co2_flux": np.arange(npts * nvm, dtype=np.float64).reshape(npts, nvm),
            "rot_cmd_max": 3,
        },
    )
    owners = SechibaInitializationOwners(
        slowproc_initialize=slowproc,
        diffuco_initialize=owner("diffuco_initialize", {"rstruct": np.ones((npts, nvm))}),
        enerbil_initialize=owner("enerbil_initialize", {"qsurf": np.asarray([0.01, 0.02])}),
        hydrol_initialize=owner(
            "hydrol_initialize",
            {
                "snow": np.asarray([1.0, 2.0]),
                "mc_layh_s": np.arange(2 * 3 * 3, dtype=np.float64).reshape(2, 3, 3),
                "mcl_layh_s": np.arange(2 * 3 * 3, dtype=np.float64).reshape(2, 3, 3) + 100.0,
                "soilmoist": np.arange(6, dtype=np.float64).reshape(2, 3),
            },
        ),
        condveg_initialize=owner(
            "condveg_initialize",
            {
                "z0m": np.asarray([0.1, 0.2]),
                "z0h": np.asarray([0.01, 0.02]),
                "emis": np.asarray([0.96, 0.97]),
            },
        ),
        thermosoil_initialize=owner("thermosoil_initialize", {"soilcap": np.asarray([10.0, 11.0])}),
    )

    owner_inputs = {}
    for position, name in enumerate(
        (
            "slowproc_initialize",
            "diffuco_initialize",
            "enerbil_initialize",
            "hydrol_initialize",
            "condveg_initialize",
            "thermosoil_initialize",
        ),
        start=1,
    ):
        owner_inputs[name] = lambda state, results, marker=position: {"marker": marker}

    result = sechiba_initialize_pft14_explicit(
        l_first=True,
        dimensions=_dimensions(),
        forcing={"qair": np.asarray([0.005, 0.006]), "temp_air": np.asarray([280.0, 281.0])},
        pft_parameters={
            "veget_max_default": _pft14(),
            "pref_soil_veg": np.array([1] * 13 + [3] + [2] * 2),
            "ok_LAIdev": np.array([False] * 13 + [True] + [False] * 2),
        },
        owner_inputs=owner_inputs,
        owners=owners,
    )

    assert calls == [
        "slowproc_initialize",
        "diffuco_initialize",
        "enerbil_initialize",
        "hydrol_initialize",
        "condveg_initialize",
        "thermosoil_initialize",
    ]
    assert result.l_first is False
    assert result.process_order.index("netco2flux_writeback") < result.process_order.index("rot_cmd_allocation")
    assert result.process_order.index("rot_cmd_allocation") < result.process_order.index("diffuco_initialize")
    assert result.process_order.index("hydrol_initialize") < result.process_order.index("hydrol_soil_tiles_to_pft")
    assert result.process_order.index("hydrol_soil_tiles_to_pft") < result.process_order.index(
        "thermosoil_initialize"
    )
    assert result.process_order.index("thermosoil_initialize") < result.process_order.index(
        "routing_initialize[single-point-no-call]"
    )
    expected_flux = np.sum(
        np.arange(npts * nvm, dtype=np.float64).reshape(npts, nvm)[:, 1:] * _pft14(npts, nvm)[:, 1:],
        axis=1,
    )
    np.testing.assert_allclose(np.asarray(result.state["netco2flux"]), expected_flux)
    np.testing.assert_array_equal(np.asarray(result.state["rot_cmd"]), np.zeros((npts, 3), dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(result.state["riverflow"]), np.zeros((npts, 3)))
    np.testing.assert_array_equal(np.asarray(result.state["z0m_out"]), [0.1, 0.2])
    np.testing.assert_array_equal(np.asarray(result.state["qsurf_out"]), [0.01, 0.02])
    np.testing.assert_array_equal(np.asarray(result.state["is_crop_soil"]), [False, False, True])
    np.testing.assert_array_equal(
        np.asarray(result.state["soilmoist_pft"]),
        np.broadcast_to(np.arange(6, dtype=np.float64).reshape(2, 3, 1), (2, 3, 16)),
    )


def test_dynamic_land_and_extensible_nvm_contract_allows_zero_pft14_cover_and_validates_forcing():
    dimensions = _dimensions(npts=3, nvm=18)
    with pytest.raises(SechibaInitializationBoundaryError, match="missing explicit inputs"):
        sechiba_initialize_pft14_explicit(
            l_first=True,
            dimensions=dimensions,
            forcing={"qair": np.ones(3)},
            pft_parameters={"veget_max_default": np.zeros((3, 18))},
            owner_inputs={},
        )

    with pytest.raises(ValueError, match="leading land dimension 3"):
        sechiba_initialize_pft14_explicit(
            l_first=True,
            dimensions=dimensions,
            forcing={"qair": np.ones(1)},
            pft_parameters={"veget_max_default": _pft14(3, 18)},
            owner_inputs={},
        )


def test_unowned_component_and_nonpaper_switches_are_explicit_gaps():
    common = dict(
        l_first=True,
        dimensions=_dimensions(),
        forcing={"qair": np.ones(2)},
        pft_parameters={"veget_max_default": _pft14()},
        owner_inputs={"slowproc_initialize": {}},
        owners=SechibaInitializationOwners(slowproc_initialize=lambda: {}),
    )
    with pytest.raises(SechibaInitializationBoundaryError, match="co2_flux and veget_max"):
        sechiba_initialize_pft14_explicit(**common)

    with pytest.raises(SechibaInitializationBoundaryError, match="co2_flux and veget_max"):
        sechiba_initialize_pft14_explicit(
            **(common | {"switches": SechibaInitializePFT14Switches(do_fullirr=True)})
        )


def test_full_irrigation_restart_transition_matches_fortran_uniform_and_varying_cases():
    uniform = sechiba_initialize_full_irrigation_restart_transition(
        restart_irrigation=np.asarray([[2.0], [2.0]])
    )
    varying = sechiba_initialize_full_irrigation_restart_transition(
        restart_irrigation=np.asarray([[1.0], [3.0]])
    )
    np.testing.assert_array_equal(np.asarray(uniform), 0.0)
    np.testing.assert_array_equal(np.asarray(varying), [[1.0], [3.0]])
