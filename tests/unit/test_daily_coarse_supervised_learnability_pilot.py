from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver import orchestration as teacher  # noqa: E402
from research.daily_coarse_graining.persistence_baseline import (  # noqa: E402
    COMMON_OK_LEAK_FIELDS,
)
from research.daily_coarse_graining.supervised_learnability_pilot import (  # noqa: E402
    PilotSample,
    _exact_discrete_metrics,
    _normalization,
    _unpack_prediction,
    build_boundary_vector_spec,
)

DAILY_FIELDS = (
    "humrel_daily",
    "litterhum_daily",
    "t2m_daily",
    "tsurf_daily",
    "tsoil_daily",
    "soilhum_daily",
    "precip_daily",
    "gpp_daily",
    "wspeed_daily",
    "snowfall_daily",
    "snowmass_daily",
    "tmc_topgrass_daily",
    "t2m_min_daily",
    "t2m_max_daily",
    "resp_maint_part",
    "resp_maint_radia",
    "flood_root_radia",
)


def _record_and_state():
    components = {
        "driver_previous_step_state": {
            "continuous": np.asarray([1.0]),
            "stage": np.asarray([2], dtype=np.int32),
        },
        "diffuco_previous_step_state": {"continuous": np.asarray([2.0])},
        "enerbil_previous_step_state": {"continuous": np.asarray([3.0])},
        "hydrol_previous_step_state": {
            "nroot": np.arange(6.0).reshape((1, 2, 3)),
            "stage": np.asarray([4], dtype=np.int32),
        },
        "thermosoil_previous_step_state": {"continuous": np.asarray([5.0])},
        "sechiba_finalize_state": {
            "continuous": np.asarray([6.0]),
            "veget_year": np.asarray([0], dtype=np.int64),
        },
        "slowproc_stomate_previous_step_state": {"resp_hetero": np.asarray([[0.0]])},
    }
    state = teacher.DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component=components,
        provenance_by_component={name: ("test",) for name in components},
    )
    ok_updates = {
        name: np.arange(6.0).reshape((2, 3)) if name == "DOC" else np.asarray([1.0])
        for name in COMMON_OK_LEAK_FIELDS
    }
    record = SimpleNamespace(
        year=1962,
        day_index=1,
        input_state_tstep=47,
        half_hour_transition=SimpleNamespace(
            current_state=state,
            completed_entry_payloads=(
                {"t2mdiag": np.asarray([280.0]), "temp_sol": np.asarray([281.0])},
            ),
        ),
        daily_fold=SimpleNamespace(
            daily_fields={name: np.asarray([float(index)]) for index, name in enumerate(DAILY_FIELDS)}
        ),
        ok_leak_updates=ok_updates,
        ok_leak_result=SimpleNamespace(
            soilcarbon=SimpleNamespace(
                perma_peat=SimpleNamespace(deepc_peat=np.arange(4.0).reshape((2, 2)))
            )
        ),
    )
    return record, state


def test_boundary_vector_spec_counts_every_continuous_element():
    record, _ = _record_and_state()

    spec = build_boundary_vector_spec(record)

    expected = 11 + 17 + (12 + 6 + 4) + 2
    assert spec.total_size == expected
    assert len(spec.leaves) == 6 + 17 + 14 + 2
    doc = next(leaf for leaf in spec.leaves if leaf.family == "ok_leak" and leaf.name == "DOC")
    assert doc.stop - doc.start == 6


def test_unpack_prediction_reconstructs_distinct_element_values():
    record, state = _record_and_state()
    spec = build_boundary_vector_spec(record)
    vector = np.arange(spec.total_size, dtype=np.float64)

    end_state, daily, ok_leak, final = _unpack_prediction(vector, state, spec)

    nroot = np.asarray(end_state.fields_by_component["hydrol_previous_step_state"]["nroot"])
    assert np.unique(nroot).size == nroot.size
    assert np.unique(np.asarray(ok_leak.soilcarbon.doc)).size == 6
    assert len(daily) == 17
    assert set(final) == {"t2mdiag", "temp_sol"}


def test_discrete_owner_rule_is_exact_carry_including_veget_year_zero():
    record, state = _record_and_state()
    spec = build_boundary_vector_spec(record)
    sample = PilotSample(
        year=1962,
        day_index=1,
        day_start_state=state,
        forcing=None,
        record=record,
        reference=np.zeros(spec.total_size),
        target=np.zeros(spec.total_size),
        finite_mask=np.ones(spec.total_size, dtype=bool),
    )

    result = _exact_discrete_metrics((sample,), spec)

    assert result["passed"]
    assert result["checked_elements"] == 3


def test_normalization_handles_constant_training_elements_without_holdout_data():
    training = np.asarray([[1.0, 3.0], [1.0, 5.0], [1.0, 7.0]])

    mean, scale = _normalization(training)

    np.testing.assert_array_equal(mean, [1.0, 5.0])
    assert scale[0] == 1.0
    np.testing.assert_allclose(scale[1], np.std(training[:, 1]))
