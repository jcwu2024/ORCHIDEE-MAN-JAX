from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_orchidee.driver.orchestration import DriverPreviousStepStatePacket
from jax_orchidee.stomate import soilcarbon_kernels
from scripts.dev.probe_ok_leak_outer_block_reentry import (
    _configure_doc_sqrt_mode,
    _normalize_daily_accumulator_schema,
    build_parser,
)


def _packet(accumulators):
    return DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component={
            "slowproc_stomate_previous_step_state": {
                "daily_accumulators": dict(accumulators),
                "retained": np.asarray([2.0]),
            }
        },
        provenance_by_component={
            "slowproc_stomate_previous_step_state": ("source",)
        },
    )


def test_normalize_outer_block_accumulator_schema_only_changes_audited_fields():
    packet = _packet(
        {
            "humrel_daily": np.asarray([0.0]),
            "fwet_daily": np.asarray([0.0]),
            "liqwt_daily": np.asarray([0.0]),
        }
    )
    produced = _packet(
        {
            "humrel_daily": np.asarray([4.0]),
            "flood_root_radia": np.asarray([3.0]),
            "resp_maint_part": np.asarray([5.0]),
            "resp_maint_radia": np.asarray([6.0]),
        }
    )

    normalized, changes = _normalize_daily_accumulator_schema(packet, produced)
    values = normalized.fields_by_component[
        "slowproc_stomate_previous_step_state"
    ]["daily_accumulators"]

    assert tuple(values) == (
        "humrel_daily",
        "flood_root_radia",
        "resp_maint_part",
        "resp_maint_radia",
    )
    assert all(np.array_equal(value, [0.0]) for value in values.values())
    assert changes == {
        "zero_filled": (
            "flood_root_radia",
            "resp_maint_part",
            "resp_maint_radia",
        ),
        "dropped": ("fwet_daily", "liqwt_daily"),
    }


def test_normalize_outer_block_accumulator_schema_rejects_unknown_drift():
    packet = _packet({"humrel_daily": np.asarray([0.0]), "unknown": np.asarray([0.0])})
    produced = _packet({"humrel_daily": np.asarray([1.0])})

    with pytest.raises(ValueError, match="non-reset dropped"):
        _normalize_daily_accumulator_schema(packet, produced)


def test_raw_doc_sqrt_mode_restores_the_teacher_primal_graph():
    original = soilcarbon_kernels._sqrt_with_finite_zero_tangent
    try:
        _configure_doc_sqrt_mode("raw_sqrt")
        values = jnp.asarray([0.0, 4.0], dtype=jnp.float64)
        expected = jax.make_jaxpr(jnp.sqrt)(values)
        actual = jax.make_jaxpr(
            soilcarbon_kernels._sqrt_with_finite_zero_tangent
        )(values)

        assert str(actual) == str(expected)
    finally:
        soilcarbon_kernels._sqrt_with_finite_zero_tangent = original


def test_doc_sqrt_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported DOC sqrt mode"):
        _configure_doc_sqrt_mode("unknown")


def test_outer_probe_parser_accepts_the_teacher_seven_day_block():
    args = build_parser().parse_args(
        [
            "--dataset-manifest",
            "dataset.json",
            "--plan",
            "plan.json",
            "--landpoint-id",
            "069.0-119.0",
            "--year",
            "1963",
            "--day-index",
            "184",
            "--block-days",
            "7",
            "--output",
            "report.json",
        ]
    )

    assert args.block_days == 7
