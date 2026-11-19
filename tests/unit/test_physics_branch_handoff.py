"""Real PB-03 component chain and selected next-call/writeback contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jax_orchidee.sechiba import hydrol
from scripts.dev import oracle_physics_branch_handoff as seam
from scripts.dev import oracle_physics_branch_hydrol as hydro_oracle
from scripts.dev import oracle_physics_branch_thermal as thermal_oracle

THERMAL_REFERENCE = thermal_oracle.load_fixture()
JOINT_GATES = [(gate["field"], index) for gate in THERMAL_REFERENCE["strict_gates"]
              if gate["component"] == "thermal" and gate["case"] == seam.THERMAL_CASE
              for index in gate["failed_flat_indices"]]


@pytest.fixture(scope="module", params=("eager", "jit"))
def components(request):
    hcase = next(c for c in hydro_oracle.load_fixture()["cases"] if c["name"] == seam.HYDROL_CASE)
    tcase = next(c for c in thermal_oracle.load_fixture()["cases"] if c["name"] == seam.THERMAL_CASE)
    tinputs = thermal_oracle.decode_inputs(tcase)
    hresult, tresult, entry = seam.run_components(hcase["inputs"], tinputs, request.param)
    return request.param, hcase, tcase, tinputs, hresult, tresult, entry


def test_actual_hydrol_moisture_enters_full_thermal_component(components):
    mode, _, tcase, tinputs, hresult, tresult, entry = components
    assert hresult.ok and tresult.ok
    # The fixed reference stores candidate01's eager producer tuple. Current
    # eager/compiled exports retain the original strict continuity contract;
    # they need not be bitwise identical to that historical producer.
    for name in seam.MOISTURE_FIELDS:
        value = getattr(hresult.thermosoil_moisture, name)
        np.testing.assert_array_equal(entry[name], value)
        np.testing.assert_allclose(value, tinputs[name], rtol=1e-12, atol=1e-14, err_msg=name)
    masks = THERMAL_REFERENCE["strict_failure_masks"].get(tcase["name"], {}).get(mode, {})
    actual = thermal_oracle.flatten_result(tresult)
    for field, expected in tcase["expected"].items():
        value, reference = actual[field].ravel(), thermal_oracle.decode(expected).ravel()
        passing = np.ones(value.size, dtype=bool)
        passing[masks.get(field, [])] = False
        np.testing.assert_allclose(value[passing], reference[passing],
            rtol=thermal_oracle.RTOL, atol=thermal_oracle.ATOL, err_msg=field)


@pytest.mark.parametrize("field,index", JOINT_GATES)
def test_joint_thermal_known_strict_coordinate(components, request, field, index):
    mode, _, tcase, _, _, tresult, _ = components
    if mode == "jit":
        request.applymarker(pytest.mark.xfail(strict=True,
            reason="Preserved joint actual HYDROL to full thermal JIT flux discrepancy; cause unresolved"))
    actual = thermal_oracle.flatten_result(tresult)[field].ravel()[index]
    expected = thermal_oracle.decode(tcase["expected"][field]).ravel()[index]
    np.testing.assert_allclose(actual, expected, rtol=thermal_oracle.RTOL, atol=thermal_oracle.ATOL,
        err_msg=f"{field}[{index}]")


def test_joint_original_thermal_input_is_bound_to_same_hydrol_tuple(components):
    _, hcase, tcase, _, _, _, _ = components
    identity = hashlib.sha256(json.dumps(hcase["inputs"], sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert identity == tcase["origin"]["producer_inputs_canonical_sha256"]
    assert hcase["inputs"] == tcase["origin"]["producer_inputs"]


def test_real_moisture_mapping_and_compiled_export(components):
    _, hcase, _, _, hresult, _, _ = components
    eager = hydrol.hydrol_to_thermosoil_moisture_inputs(hresult.diagnostics,
        pref_soil_veg=hcase["inputs"]["pref_soil_veg"])
    _, compiled = hydrol.hydrol_module_outputs_and_moisture_inputs_jit(
        hresult.module, hresult.diagnostics, soiltile=hcase["inputs"]["soiltile"],
        pref_soil_veg=hcase["inputs"]["pref_soil_veg"])
    for name in seam.MOISTURE_FIELDS:
        value = getattr(eager, name)
        np.testing.assert_array_equal(getattr(hresult.thermosoil_moisture, name), value)
        np.testing.assert_array_equal(getattr(compiled, name), value)


def test_actual_writers_packet_and_next_diffuco_selection(components):
    _, _, _, _, hresult, tresult, _ = components
    packet, mapped, used, missing = seam.write_and_map(hresult, tresult)
    assert missing  # Other components and a complete next DIFFUCO call are absent.
    for name, expected in (("qsintveg", hresult.module.canop.qsintveg),
                           ("humrel", hresult.diagnostics.humrel),
                           ("evap_bare_lim", hresult.diagnostics.evap_bare_lim),
                           ("drysoil_frac", hresult.diagnostics.drysoil_frac)):
        assert name in used
        np.testing.assert_array_equal(mapped[name], expected)
    water = packet.fields_by_component["hydrol_previous_step_state"]
    for name, expected in (("mc", hresult.module.soil.mc), ("mcl", hresult.module.soil.mcl),
                           ("ae_ns", hresult.diagnostics.ae_ns), ("wat_flux", hresult.outputs.wat_flux),
                           ("profil_froz_hydro_ns", hresult.module.soil.profil_froz_hydro_ns)):
        np.testing.assert_array_equal(water[name], expected)
    temperature = packet.fields_by_component["thermosoil_previous_step_state"]
    energy = packet.fields_by_component["enerbil_previous_step_state"]
    np.testing.assert_array_equal(temperature["ptn"], tresult.module.profile.ptn)
    for name in ("cgrnd", "dgrnd", "soilcap", "soilflx"):
        np.testing.assert_array_equal(temperature[name], tresult.downstream_payload[name])
        np.testing.assert_array_equal(energy[name], tresult.downstream_payload[name])
    for name in ("soilcap_pft", "soilflx_pft"):
        np.testing.assert_array_equal(energy[name], tresult.downstream_payload[name])
    for name in ("snow", "snowdz", "snowrho", "snowtemp", "snowheat", "snowgrain", "snow_age", "snow_nobio", "snow_nobio_age"):
        np.testing.assert_array_equal(water[name], getattr(hresult.snow_state, name))
        np.testing.assert_array_equal(packet.fields_by_component["diffuco_previous_step_state"][name], getattr(hresult.snow_state, name))
        if name not in ("snowheat", "snowgrain"):
            np.testing.assert_array_equal(mapped[name], getattr(hresult.snow_state, name))


def test_actual_stomate_entry_rename_and_water_thermal_arrays(components):
    _, hcase, _, tinputs, hresult, tresult, _ = components
    packet, _, _, _ = seam.write_and_map(hresult, tresult)
    entry = seam.stomate_entry(hresult, tresult, packet, hcase["inputs"], tinputs)
    assert entry is not None
    for name, expected in (("humrel", hresult.diagnostics.vegstress),
                           ("shumdiag", hresult.diagnostics.shumdiag),
                           ("soil_mc", hresult.diagnostics.mc_layh_s),
                           ("wat_flux", hresult.outputs.wat_flux),
                           ("runoff_per_soil", hresult.outputs.runoff_per_soil),
                           ("drainage_per_soil", hresult.outputs.drainage_per_soil),
                           ("stempdiag", tresult.module.profile.stempdiag),
                           ("tdeep", tresult.module.final.deeptemp_prof),
                           ("hsdeep", tresult.module.final.deephum_prof)):
        np.testing.assert_array_equal(entry[name], expected)


def test_source_boundary_rename_and_moisture_provenance():
    """Only this source identity gate may skip in a source-free checkout."""
    source = Path(__file__).resolve().parents[2] / "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"
    if not source.is_file():
        pytest.skip("Original source identity gate only; actual component checks remain active")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "c902a886f7a1c0837dec3f6d11b3d93516792e5c4b95e6873bb92e6461864526"
    lines = source.read_text(encoding="utf-8").splitlines()
    transfer = "\n".join(lines[1094:1103])
    assert "mc_layh_s(:,:,pref_soil_veg(jv))" in transfer
    assert "soilmoist_pft(:,:,jv) = soilmoist(:, :)" in transfer
    assert "vegstress" in "\n".join(lines[1183:1189])
