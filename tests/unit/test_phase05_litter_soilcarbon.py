"""Independent original-Fortran finite litter/soilcarbon reference tuples.

The fixture is generated from source procedures, never from JAX expected
values. These tests execute the maintained component adapter and its real
half-hour writer; they do not advance a whole model step or day.
"""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from scripts.dev.oracle_phase05_litter_soilcarbon import actual_outputs, case_inputs

jax.config.update("jax_enable_x64", True)
REFERENCE = json.loads(
    (Path(__file__).parents[1] / "fixtures/phase05_litter_soilcarbon_reference.json").read_text(encoding="utf-8")
)


def _assert_reference(actual, expected):
    for field, encoded in expected.items():
        f = np.asarray(encoded["values"], dtype=np.float64).reshape(encoded["shape"])
        j = np.asarray(actual[field])
        assert np.all(np.isfinite(j)), field
        np.testing.assert_allclose(j, f, rtol=REFERENCE["rtol"], atol=REFERENCE["atol"], err_msg=field)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["spec"]["id"])
def test_original_litter_tf_doc_soilcarbon_tuple_and_actual_writer(case):
    args, raw = case_inputs(case["spec"])
    actual, result, writer, next_inputs = actual_outputs(args)
    _assert_reference(actual, case["expected"])
    from jax_orchidee.coupled import stomate_active_layer_ok_leak_inputs

    active = stomate_active_layer_ok_leak_inputs(
        tprof=args["tprof"],
        zi_soil=raw["zz"],
        restart_altmax=raw["altmax"],
        restart_fixed_cryoturbation_depth=raw["fixed"],
        dayno=1,
        firstcall_soilcarbon=True,
        newaltcalc=raw["newaltcalc"],
        soilc_isspinup=raw["spinup"],
    )
    _assert_reference(active.state_updates, case["active_layer_expected"])
    assert len(writer) == 13
    # This checks source units at the real adapter's respiration/export seam.
    np.testing.assert_allclose(
        result.transfers.litter_respiration,
        result.littercalc.resp_hetero_litter * args["dt_days"],
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_array_equal(next_inputs["doc"], writer["DOC"])
    if "next_expected" in case:
        second, _, _, _ = actual_outputs(next_inputs)
        _assert_reference(second, case["next_expected"])


def test_source_floor_guard_is_strict_and_litter_fluxes_are_not_reconstructed():
    for case in REFERENCE["cases"]:
        spec = case["spec"]
        if not spec.get("floor"):
            continue
        args, _ = case_inputs(spec)
        actual, result, _, _ = actual_outputs(args)
        value = spec.get("floor_value", 1e-13)
        expected_lignin = 0.2 if value > 1e-8 else 0.0
        np.testing.assert_allclose(actual["lignin_struc_above"][:, 13], expected_lignin, rtol=1e-12, atol=1e-14)
        # The saved transfer is the actual process output in step-mass units.
        np.testing.assert_allclose(
            result.transfers.litter_to_doc, actual["soilcarbon_input_doc"] / 48, rtol=1e-12, atol=1e-14
        )


def test_tf_doc_observed_storage_has_source_deposition_budget():
    for case in REFERENCE["cases"]:
        if case["spec"]["id"] not in {"tf_storage_limited", "tf_absent_vegetation"}:
            continue
        args, _ = case_inputs(case["spec"])
        actual, result, _, _ = actual_outputs(args)
        active = args["veget_max"][:, 13] > 0
        release = np.minimum(args["canopy2ground"][:, 13] * 0.1, args["interception_storage"][:, 13, 0])
        expected = np.where(
            active,
            args["interception_storage"][:, 13, 0]
            + args["dry_dep_canopy"][:, 13, 0]
            + args["doc_precip2canopy"][:, 13, 0]
            - release,
            0.0,
        )
        np.testing.assert_allclose(actual["interception_storage"][:, 13, 0], expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(
            result.wet_dep_ground + result.wet_dep_flood,
            args["doc_precip2ground"]
            + np.where(
                (np.arange(14) > 0)[None, :, None] & (args["veget_max"] > 0)[:, :, None],
                np.minimum(args["canopy2ground"][:, :, None] * 0.1, args["interception_storage"]),
                0.0,
            ),
            rtol=1e-12,
            atol=1e-14,
        )


def test_ok_pc_local_helper_support_keeps_full_driver_source_guard():
    from jax_orchidee.driver.orchestration import _validate_stomate_main_supported_switches

    with pytest.raises(NotImplementedError, match="OK_LEAK=n"):
        _validate_stomate_main_supported_switches({"STOMATE_OK_STOMATE": "TRUE", "OK_LEAK": "FALSE", "OK_PC": "TRUE"})
    _validate_stomate_main_supported_switches({"STOMATE_OK_STOMATE": "TRUE", "OK_LEAK": "TRUE", "OK_PC": "TRUE"})
