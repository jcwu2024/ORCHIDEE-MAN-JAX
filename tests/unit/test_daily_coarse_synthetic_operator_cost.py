from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver import orchestration as teacher  # noqa: E402
from jax_orchidee.driver.orchestration import (  # noqa: E402
    _add_hydrol_to_previous_fields,
    _empty_previous_step_state_fields,
)
from research.daily_coarse_graining.boundary_adapter import (  # noqa: E402
    packet_from_projected_values,
    project_packet_values,
    promote_runtime_state_spec,
)
from research.daily_coarse_graining.synthetic_operator_cost import (  # noqa: E402
    _operator_output_width,
    _synthetic_nroot,
    initialize_operator_parameters,
    synthetic_boundary,
)


def _state(*, include_nroot: bool = False):
    hydrol = {
        "mc": np.full((1, 11, 6), 0.4),
        "mcl": np.full((1, 11, 6), 0.3),
        "us": np.linspace(0.1, 1.0, 1 * 14 * 6 * 11).reshape((1, 14, 6, 11)),
        "snow": np.asarray([0.2]),
        "humrel": np.full((1, 14), 0.5),
        "soil_mc": np.full((1, 11, 6), 0.4),
    }
    if include_nroot:
        hydrol["nroot"] = np.full((1, 14, 11), 1.0 / 11.0)
    slow = {
        "biomass": np.ones((1, 14, 12, 1)),
        "daily_accumulators": {
            "resp_maint_radia": np.ones((1, 14)),
            "flood_root_radia": np.zeros((1, 14)),
        },
        "npp_daily": np.full((1, 14), 2.0),
        "resp_maint": np.full((1, 14), 1.0),
        "resp_growth": np.full((1, 14), 0.5),
        "resp_hetero": np.zeros((1, 14)),
        "litter_above": np.ones((1, 2, 14, 1)),
        "litter_below": np.ones((1, 2, 14, 32, 1)),
        "lignin_struc_above": np.ones((1, 14)),
        "lignin_struc_below": np.ones((1, 14, 32)),
        "litterpart": np.ones((1, 14, 2)),
        "dead_leaves": np.ones((1, 14, 2)),
        "fuel_1hr": np.ones((1, 14, 2, 1)),
        "fuel_10hr": np.ones((1, 14, 2, 1)),
        "fuel_100hr": np.ones((1, 14, 2, 1)),
        "fuel_1000hr": np.ones((1, 14, 2, 1)),
        "carbon_32l": np.ones((1, 3, 14, 32)),
        "DOC": np.ones((1, 14, 32, 2, 7, 1)),
        "deepC_peat": np.ones((1, 32, 14)),
        "interception_storage": np.ones((1, 14, 1)),
    }
    fields = {
        "driver_previous_step_state": {"fluxsens": np.asarray([1.0])},
        "diffuco_previous_step_state": {
            "temp_sol": np.asarray([290.0]),
            "gpp": np.ones((1, 14)),
        },
        "enerbil_previous_step_state": {"temp_sol": np.asarray([290.0])},
        "hydrol_previous_step_state": hydrol,
        "thermosoil_previous_step_state": {
            "ptn": np.full((1, 32, 14), 288.0),
            "stempdiag": np.full((1, 11), 288.0),
        },
        "slowproc_stomate_previous_step_state": slow,
        "sechiba_finalize_state": {"temp_sol": np.asarray([290.0])},
    }
    return teacher.DriverPreviousStepStatePacket(
        tstep=-1,
        fields_by_component=fields,
        provenance_by_component={name: ("test",) for name in fields},
    )


def _forcing(temp_offset: float = 0.0):
    common = {
        name: np.full((48, 1), 0.2)
        for name in (
            "zlev",
            "zlevuv",
            "u",
            "v",
            "qair",
            "pb",
            "precip_rain",
            "precip_snow",
            "lwdown",
            "swdown",
            "ccanopy",
            "salinity",
            "tide_height",
        )
    }
    common["temp_air"] = np.linspace(280.0 + temp_offset, 290.0 + temp_offset, 48)[:, None]
    return SimpleNamespace(**common)


def _metadata():
    return teacher.DriverRuntimeStepMetadata(
        pref_soil_veg=np.asarray([1]),
        nstm=6,
        is_tree=np.asarray([False] * 14),
        is_peat=np.asarray([False] * 14),
        lalo=np.asarray([[0.0, 0.0]]),
        npts=1,
    )


def test_nroot_runtime_spec_is_promoted_only_from_a_dynamic_producer():
    start = _state(include_nroot=False)
    produced = _state(include_nroot=True)
    dynamic_nroot = np.linspace(0.0, 1.0, 154).reshape((1, 14, 11))
    produced.fields_by_component["hydrol_previous_step_state"]["nroot"] = dynamic_nroot

    spec = promote_runtime_state_spec(start, produced)
    values = project_packet_values(produced, spec)
    restored = packet_from_projected_values(values_by_component=values, spec=spec, tstep=47)

    assert "nroot" not in start.fields_by_component["hydrol_previous_step_state"]
    np.testing.assert_array_equal(
        restored.fields_by_component["hydrol_previous_step_state"]["nroot"],
        dynamic_nroot,
    )
    with pytest.raises(ValueError, match="did not emit hydrol.nroot"):
        promote_runtime_state_spec(start, start)


def test_teacher_hydrol_owner_writes_the_exact_dynamic_nroot():
    token = np.linspace(0.0, 1.0, 154).reshape((1, 14, 11))
    scalar = np.asarray([0.0])
    hydrol = SimpleNamespace(
        ok=True,
        module=SimpleNamespace(
            soil=SimpleNamespace(
                mc=scalar,
                mcl=scalar,
                water2infilt=scalar,
                run2peat=scalar,
                run2man=scalar,
                wt_ab=scalar,
                wt_ab_tide=scalar,
            ),
            split=SimpleNamespace(ae_ns=scalar),
            canop=SimpleNamespace(qsintveg=scalar),
            flood=SimpleNamespace(flood_res=scalar),
        ),
        diagnostics=SimpleNamespace(nroot=token, mc_layh_s=scalar),
        outputs=SimpleNamespace(
            drainage_per_soil=scalar,
            runoff_per_soil=scalar,
            wat_flux=scalar,
            runoff2peat=scalar,
            canopy2ground=scalar,
            precip2ground=scalar,
            precip2canopy=scalar,
        ),
        module_inputs={},
        snow_state=None,
        snow_step=None,
    )
    fields = _empty_previous_step_state_fields()

    _add_hydrol_to_previous_fields(fields, hydrol)

    assert fields["hydrol_previous_step_state"]["nroot"] is token


def test_synthetic_nroot_uses_dynamic_state_and_parameter_signal():
    state = _state(include_nroot=False)
    first = _synthetic_nroot(state, np.asarray(0.0))
    second = _synthetic_nroot(state, np.asarray(2.0))

    assert first.shape == (1, 14, 11)
    np.testing.assert_allclose(np.asarray(first).sum(axis=-1), 1.0)
    assert not np.array_equal(np.asarray(first), np.asarray(second))


def test_synthetic_boundary_covers_runtime_daily_and_ok_leak_outputs():
    state = _state(include_nroot=False)
    width = _operator_output_width(state)
    first_parameters = initialize_operator_parameters(width, seed=1)
    second_parameters = initialize_operator_parameters(width, seed=2)

    first = synthetic_boundary(
        state, _forcing(), _metadata(), first_parameters, min_wind=0.1
    )
    second = synthetic_boundary(
        state, _forcing(), _metadata(), second_parameters, min_wind=0.1
    )

    transition, daily, ok_leak, checksum = first
    assert "nroot" in transition.current_state.fields_by_component["hydrol_previous_step_state"]
    assert len(daily.daily_fields) == 17
    assert set(teacher._paper_half_hour_ok_leak_state_updates(ok_leak)) == set(
        teacher._paper_half_hour_ok_leak_state_updates(second[2])
    )
    assert np.isfinite(np.asarray(checksum))
    assert not np.array_equal(np.asarray(checksum), np.asarray(second[3]))


def test_compiled_boundary_keeps_parameters_as_dynamic_inputs():
    state = _state(include_nroot=False)
    width = _operator_output_width(state)
    first_parameters = initialize_operator_parameters(width, seed=3)
    second_parameters = initialize_operator_parameters(width, seed=4)

    compiled = jax.jit(
        lambda parameters: synthetic_boundary(
            state, _forcing(), _metadata(), parameters, min_wind=0.1
        )[3]
    )
    first = compiled(first_parameters)
    second = compiled(second_parameters)

    assert not np.array_equal(np.asarray(first), np.asarray(second))
