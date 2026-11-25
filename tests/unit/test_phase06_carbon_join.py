"""Nonzero original cryoturbation return through both remaining real writers.

This finite composition checks state ownership and diagnostics. It is not a
new original oracle for a combined natural daily trajectory.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.driver import orchestration as driver
from jax_orchidee.stomate import integration
from scripts.dev import oracle_phase05_growth_conditions as growth
from scripts.dev import oracle_phase06_soilcarbon_diffusion as soil


def first_two(value):
    if isinstance(value, dict):
        return {key: first_two(item) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.shape and value.shape[0] == 6:
        return value[:2].copy()
    return value


@pytest.fixture(scope="module")
def actual_join():
    spec = next(item for item in soil.SPECS if item["route"] == "cryoturbation")
    inputs, _ = soil.inputs(spec)
    _, _, second = soil.actual_chain(inputs, cryo=True)
    _, result, _ = soil.actual_chain(second, cryo=True)
    reference = np.load(soil.FIXTURE)
    expected = reference[spec["id"] + "/call1/soil_litter_below"]
    np.testing.assert_allclose(result.soilcarbon.litter_below, expected, rtol=soil.RTOL, atol=soil.ATOL)
    # This case distinguishes the actual post-soil owner from littercalc.
    assert np.max(np.abs(expected - np.asarray(result.littercalc.litter_below))) > 1e-4
    carbon_inputs = first_two(growth.chain_inputs())
    for group in ("prescribe_inputs", "post_npp_inputs"):
        carbon_inputs[group]["veget_max"] = inputs["veget_max"]
    carbon = growth.run_current_chain(carbon_inputs, public_wrapper=True)
    envelope = integration.ExplicitOkLeakFromPostNppResult(prep=None, ok_leak=result, requires_trace=())
    outputs = integration.stomate_lpj_outputs_from_post_npp_ok_leak_explicit(
        post_npp=carbon.post_npp, ok_leak=envelope,
        veget_max=inputs["veget_max"], z_soil=inputs["z_soil"], zf_soil=inputs["zf_soil_b"],
        carb_mass_total_old=np.zeros(2), gpp_daily=carbon_inputs["post_npp_inputs"]["gpp_daily"],
    )
    return inputs, carbon_inputs, carbon, envelope, outputs, expected


def test_nonzero_original_post_soil_litter_enters_daily_packet_and_end_carry(actual_join):
    inputs, carbon_inputs, carbon, envelope, outputs, expected = actual_join
    previous = driver.DriverPreviousStepStatePacket(
        tstep=47, fields_by_component={"slowproc_stomate_previous_step_state": {
            "deepC_peat": np.zeros((2, 14, 32)), "depth_deepsoil": inputs["zf_soil_b"],
        }}, provenance_by_component={},
    )
    packet = driver._paper_day_first_stomate_state_packet(
        daily_carbon=carbon, ok_leak=envelope, outputs=outputs,
        daily_reset_fields={
            "gpp_daily": np.zeros_like(carbon_inputs["post_npp_inputs"]["gpp_daily"]),
            "resp_maint_part": np.zeros_like(carbon_inputs["post_npp_inputs"]["resp_maint_part"]),
        }, previous_state=previous, dt_days=1.,
    )
    end = driver._paper_day_end_state_packet(previous_state=previous, stomate_state=packet)
    for actual in (packet.fields["litter_below"],
                   end.fields_by_component["slowproc_stomate_previous_step_state"]["litter_below"]):
        np.testing.assert_allclose(actual, expected, rtol=soil.RTOL, atol=soil.ATOL)


def test_nonzero_original_post_soil_litter_enters_actual_lpj_diagnostic(actual_join):
    _, _, _, envelope, outputs, expected = actual_join
    # Original StomateLpj:1578-1591 adds structural/metabolic stock, with
    # aboveground stock in the first layer and bare PFT excluded.
    expected_total = expected[:, 0, :, :, 0] + expected[:, 1, :, :, 0]
    above = np.asarray(envelope.ok_leak.littercalc.litter_above)
    expected_total[:, :, 0] += above[:, 0, :, 0] + above[:, 1, :, 0]
    expected_total[:, 0, :] = 0.
    np.testing.assert_allclose(outputs.output_diagnostics.tot_litter_carb, expected_total,
                               rtol=soil.RTOL, atol=soil.ATOL)


def test_unavailable_litter_uses_pre_decay_and_legacy_pools_at_day_end(actual_join):
    inputs, carbon_inputs, carbon, envelope, outputs, _ = actual_join
    # Distinct pool values exercise littercalc_leak 2325–2348 and LPJ 1218–1221.
    pre_decay = np.full((2, 2, 14), 4.0)
    littercalc = envelope.ok_leak.littercalc._replace(litter_above_before_decay_carbon=pre_decay)
    envelope = envelope._replace(ok_leak=envelope.ok_leak._replace(littercalc=littercalc))
    grass = np.arange(14) == 9
    flags_off = np.zeros(14, dtype=bool)
    parameters = SimpleNamespace(
        is_tree=np.arange(14) == 13, natural=grass, is_grassland_manag=flags_off,
        is_grassland_grazed=flags_off, is_grassland_cut=flags_off,
    )
    previous = driver.DriverPreviousStepStatePacket(
        tstep=47, fields_by_component={"slowproc_stomate_previous_step_state": {
            "litter": np.full((2, 2, 14, 2, 1), 8.0),
            "litter_not_avail": np.ones_like(pre_decay),
            "deepC_peat": np.zeros((2, 14, 32)), "depth_deepsoil": inputs["zf_soil_b"],
        }}, provenance_by_component={},
    )
    packet = driver._paper_day_first_stomate_state_packet(
        daily_carbon=carbon, ok_leak=envelope, outputs=outputs, previous_state=previous,
        stomate_parameters=parameters, dt_days=1.0, daily_reset_fields={
            "gpp_daily": np.zeros_like(carbon_inputs["post_npp_inputs"]["gpp_daily"]),
            "resp_maint_part": np.zeros_like(carbon_inputs["post_npp_inputs"]["resp_maint_part"]),
        },
    )
    end = driver._paper_day_end_state_packet(previous_state=previous, stomate_state=packet)
    expected = np.zeros_like(pre_decay)
    expected[:, :, 0] = 8.0  # Bare soil has zero available fraction.
    expected[:, :, 9] = 2.0  # Natural grass retains 8 × (1 − (4 − 1) / 4).
    np.testing.assert_array_equal(packet.fields["litter_not_avail"], expected)
    np.testing.assert_array_equal(
        end.fields_by_component["slowproc_stomate_previous_step_state"]["litter_not_avail"],
        expected,
    )
