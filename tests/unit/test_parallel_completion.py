from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from jax_orchidee.parallel import (
    XiosDimensions,
    XiosInitControls,
    init_orchidee_mpi,
    xios_orchidee_init,
    xios_orchidee_send_field,
)


ROOT = Path(__file__).resolve().parents[2]


def _dimensions() -> XiosDimensions:
    return XiosDimensions(4, 3, 2, 2, 3, 14, 6, 3, 4, 2, 32, 5, 7, 3, 4)


def _controls(**updates: bool) -> XiosInitControls:
    values = {
        "off_line_mode": True,
        "river_routing": False,
        "hydrol_cwrr": True,
        "ok_freeze_cwrr": False,
        "check_cwrr2": False,
        "do_floodplains": False,
        "ok_stomate": False,
        "do_irrigation": False,
        "ok_co2": False,
        "ok_bvoc": False,
        "ok_radcanopy": False,
        "ok_multilayer": False,
        "ok_bbgfertil_nox": False,
        "ok_cropsfertil_nox": False,
        "check_waterbal": False,
        "impaze": True,
        "ok_explicitsnow": False,
    }
    values.update(updates)
    return XiosInitControls(**values)


def test_init_orchidee_mpi_preserves_source_communicator_selection_and_serial_disable():
    calls: list[str] = []
    state = init_orchidee_mpi(
        xios_orchidee_ok=True,
        cpp_para=True,
        communicator="ignored",
        mpi_init=lambda: calls.append("mpi_init") or "world",
        xios_comm_init=lambda: calls.append("xios_comm") or "xios",
        communicator_info=lambda comm: (4, 2) if comm == "xios" else (1, 0),
    )
    assert calls == ["mpi_init", "xios_comm"]
    assert state.communicator == "xios" and (state.mpi_size, state.mpi_rank) == (4, 2)
    assert state.mpi_rank_root == 0 and not state.is_mpi_root and state.is_ok_mpi

    supplied = init_orchidee_mpi(
        xios_orchidee_ok=False,
        cpp_para=True,
        communicator="coupler",
        communicator_info=lambda comm: (2, 0),
    )
    assert supplied.communicator == "coupler" and supplied.is_mpi_root
    serial = init_orchidee_mpi(xios_orchidee_ok=True, cpp_para=False)
    assert (serial.mpi_rank, serial.mpi_size, serial.is_ok_mpi) == (0, 1, False)
    assert not serial.xios_orchidee_ok


def test_xios_init_disabled_has_no_xios_transport_but_keeps_calendar_conversion():
    converted: list[float] = []
    result = xios_orchidee_init(
        communicator="comm",
        date0=12.5,
        year=2001,
        month=2,
        day=3,
        lon_mpi=np.empty((4, 2)),
        lat_mpi=np.empty((4, 2)),
        soilth_lev=np.empty((3,)),
        kindex_mpi=np.array([1, 2, 3]),
        dimensions=_dimensions(),
        controls=_controls(),
        calendar="360d",
        dt_sechiba=1800.0,
        xios_orchidee_ok=False,
        xios_compiled=False,
        is_omp_root=True,
        almaoutput=False,
        julian_to_date=lambda value: converted.append(value) or (1900, 1, 12, 0.0),
    )
    assert converted == [12.5]
    assert result.calendar == "d360" and result.time_origin == (1900, 1, 12)
    assert result.plan is None and not result.xios_orchidee_ok


def test_xios_init_preserves_domain_axes_deactivation_and_alma_broadcast():
    plans = []
    broadcasts = []
    lon = np.arange(8.0).reshape(4, 2)
    lat = np.arange(20.0, 28.0).reshape(4, 2)
    result = xios_orchidee_init(
        communicator="orch",
        date0=0.0,
        year=2000,
        month=1,
        day=2,
        lon_mpi=lon,
        lat_mpi=lat,
        soilth_lev=np.array([0.1, 0.4, 1.0]),
        kindex_mpi=np.array([1, 6, 11]),
        dimensions=_dimensions(),
        controls=_controls(),
        calendar="gregorian",
        dt_sechiba=1800.0,
        xios_orchidee_ok=True,
        xios_compiled=True,
        is_omp_root=True,
        almaoutput=False,
        julian_to_date=lambda value: (1999, 12, 31, 43200.0),
        transport=lambda plan: (
            plans.append(plan)
            or {
                "delsoilmoist": False,
                "delintercept": False,
                "delswe": False,
                "soilwet": True,
                "twbr": False,
            }
        ),
        broadcast_almaoutput=lambda value: broadcasts.append(value) or value,
    )
    plan = plans[0]
    assert plan.communicator == "orch" and plan.timestep_seconds == 1800.0
    assert plan.start_date == (2000, 1, 2, 0, 0, 0)
    assert plan.time_origin == (1999, 12, 31, 0, 0, 0)
    assert (plan.domain.ibegin, plan.domain.jbegin, plan.domain.data_ibegin) == (
        0,
        1,
        0,
    )
    np.testing.assert_array_equal(plan.domain.data_i_index, [0, 5, 10])
    np.testing.assert_array_equal(plan.domain.lonvalue_1d, lon[:, 0])
    np.testing.assert_array_equal(plan.domain.latvalue_1d, lat[0, :])
    axes = {axis.axis_id: axis for axis in plan.axes}
    np.testing.assert_array_equal(axes["nvm"].values, np.arange(1.0, 15.0))
    np.testing.assert_array_equal(axes["ngrnd"].values, [0.1, 0.4, 1.0])
    assert axes["nsnow"].n_glo == 1
    assert "q2m" in plan.disabled_fields and "basinmap" in plan.disabled_fields
    assert "dss" in plan.disabled_fields and "frac_bare" not in plan.disabled_fields
    assert "PARsuntab" in plan.disabled_fields and "PARsun" in plan.disabled_fields
    assert broadcasts == [True] and result.almaoutput


def test_xios_init_rejects_enabled_without_compiled_support_and_bad_fortran_shape():
    common = dict(
        communicator="orch",
        date0=0.0,
        year=2000,
        month=1,
        day=1,
        lon_mpi=np.zeros((4, 2)),
        lat_mpi=np.zeros((4, 2)),
        soilth_lev=np.ones(3),
        kindex_mpi=np.array([1, 2, 3]),
        dimensions=_dimensions(),
        controls=_controls(),
        calendar="noleap",
        dt_sechiba=1800.0,
        xios_orchidee_ok=True,
        is_omp_root=True,
        almaoutput=False,
        julian_to_date=lambda value: (2000, 1, 1, 0.0),
        transport=lambda plan: {
            "delsoilmoist": False,
            "delintercept": False,
            "delswe": False,
            "soilwet": False,
            "twbr": False,
        },
        broadcast_almaoutput=lambda value: value,
    )
    with pytest.raises(RuntimeError, match="compiled with XIOS"):
        xios_orchidee_init(**common, xios_compiled=False)
    with pytest.raises(ValueError, match="local-domain shape"):
        xios_orchidee_init(
            **(common | {"lon_mpi": np.zeros((2, 4))}), xios_compiled=True
        )


@pytest.mark.parametrize("rank", range(1, 6))
def test_send_field_rank_dispatch_gathers_exact_shape_and_transports_only_on_root(
    rank: int,
):
    shape = (2, *range(3, rank + 2))
    field = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    sent = []
    status = xios_orchidee_send_field(
        "field",
        field,
        nbp_mpi=3,
        xios_orchidee_ok=True,
        is_omp_root=True,
        gather_omp=lambda value: np.concatenate((value, value[:1]), axis=0),
        transport=lambda field_id, value: sent.append((field_id, value.copy())),
    )
    assert status.rank == rank and status.gathered_shape == (3, *shape[1:])
    assert status.transported and sent[0][0] == "field"

    nonroot = xios_orchidee_send_field(
        "field",
        field,
        nbp_mpi=3,
        xios_orchidee_ok=True,
        is_omp_root=False,
        gather_omp=lambda value: np.concatenate((value, value[:1]), axis=0),
        transport=None,
    )
    assert not nonroot.transported
    disabled = xios_orchidee_send_field(
        "field",
        field,
        nbp_mpi=999,
        xios_orchidee_ok=False,
        is_omp_root=True,
        gather_omp=None,
        transport=None,
    )
    assert disabled.gathered_shape == () and not disabled.transported


def test_parallel_completion_ledger_has_exact_17_implemented_arms():
    ledger = yaml.safe_load(
        (
            ROOT / "docs/source_audits/pft14_source_owners_parallel_completion.yaml"
        ).read_text(encoding="utf-8")
    )
    assert ledger["totals"] == {"procedures": 7, "arms": 17}
    assert sum(entry["arm_count"] for entry in ledger["entries"]) == 17
    assert len({entry["id"] for entry in ledger["entries"]}) == 7
    for entry in ledger["entries"]:
        assert entry["status"] == "implemented"
        assert entry["jax_owner"].startswith("jax_orchidee.parallel.")
        assert entry["gap"] is None
        assert entry["state_contract"]
        assert entry["validation"] == ["tests/unit/test_parallel_completion.py"]
