"""Portable numerical gates separated from optional original-source identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from scripts.dev import oracle_phase06_canopy_predecessors as canopy

REFERENCE = json.loads(
    (Path(__file__).parents[1] / "fixtures/phase06_canopy_predecessors_reference.json").read_text(encoding="utf-8")
)
FIELDS = [(g, k) for g, fields in REFERENCE["expected"].items() for k in fields]


@pytest.fixture(scope="module", params=(False, True), ids=("eager", "jit"))
def actual(request):
    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    return canopy.actual_values(use_jit=request.param)


@pytest.mark.parametrize("group,field", FIELDS)
def test_phase06_original_numeric_field(actual, group, field):
    np.testing.assert_allclose(
        actual[group][field], REFERENCE["expected"][group][field], rtol=REFERENCE["rtol"], atol=REFERENCE["atol"]
    )


@pytest.mark.parametrize("use_jit", (False, True), ids=("eager", "jit"))
@pytest.mark.parametrize("maintenance", (False, True), ids=("restart-held", "maintenance-source-order"))
def test_real_next_diffuco_uses_carried_and_surface_operands(use_jit, maintenance):
    fixture, result, missing = canopy.run_next(use_jit=use_jit, maintenance=maintenance)
    key = "maintenance" if maintenance else "restart_held"
    expected = REFERENCE["inputs"][key]["next_diffuco"]
    assert missing  # Unrelated backgrounds are declared, not invented by mapper.
    for name in (
        "lai",
        "veget",
        "veget_max",
        "roughheight",
        "roughheight_pft",
        "z0h",
        "z0m",
        "qsintmax",
        "frac_snow_veg",
    ):
        np.testing.assert_allclose(
            result.kwargs[name], expected[name], rtol=REFERENCE["rtol"], atol=REFERENCE["atol"], err_msg=name
        )
    np.testing.assert_array_equal(result.kwargs["passthrough"]["height"], expected["height"])
    assert not np.any(result.kwargs["ok_laidev"])
    np.testing.assert_array_equal(result.result.payload.payload["lai"], fixture.diffuco_precall.payload["lai"])


@pytest.mark.parametrize("use_jit", (False, True), ids=("eager", "jit"))
def test_full_present_restart_and_surface_hold_do_not_impose_fc_lai_relation(use_jit):
    x, initialized, _, held = canopy.run_predecessors(use_jit=use_jit)
    assert initialized.found_restart
    for key in ("lai", "veget", "veget_max", "frac_nobio"):
        np.testing.assert_array_equal(getattr(initialized, key), x[key])
        np.testing.assert_array_equal(held[key], getattr(initialized, key))
    # Demonstrated independent counterexample within the safe finite domain.
    assert float(np.asarray(held["veget"])[0, 13]) == 0.27
    assert float(np.asarray(held["lai"])[0, 13]) == 0.0


@pytest.mark.parametrize("use_jit", (False, True), ids=("eager", "jit"))
def test_high_fc_producer_is_preserved_without_claiming_consumer_admission(use_jit):
    _, initialized, _, held = canopy.run_predecessors(use_jit=use_jit, high_fc=True)
    assert initialized.found_restart
    np.testing.assert_array_equal(np.asarray(held["veget"])[:, 13], np.full(8, 0.6))
    np.testing.assert_array_equal(np.asarray(held["lai"])[:, 13], np.full(8, 1.01e-8))
    # The original consumer SIGFPE is a separate retained source diagnostic;
    # these producer equalities do not admit this tuple as CONDVEG success.


def test_declared_independent_inputs():
    def normalize(value):
        return json.loads(json.dumps(value, default=lambda item: item.tolist()))

    for maintenance, key in ((False, "restart_held"), (True, "maintenance")):
        x = canopy.make_inputs(maintenance=maintenance)
        expected = REFERENCE["inputs"][key]
        assert normalize(x) == expected["surface"]
        assert normalize(canopy.restart_inputs(x)) == expected["restart"]
        if maintenance:
            biomass, sla = canopy.maintenance_inputs()
            assert biomass.tolist() == expected["maintenance_biomass"]
            assert sla.tolist() == expected["maintenance_sla_calc"]


@pytest.mark.external_data
def test_optional_original_source_identity():
    checks = []
    for group, identities in REFERENCE["reference_identity"].items():
        for name, identity in identities.items():
            if isinstance(identity, list):
                checks.extend((canopy.ROOT / item["source"], item) for item in identity)
                continue
            if name.startswith("enerbil_"):
                continue  # Compiled scaffolding, not an executed owner here.
            if name.startswith("slowproc_"):
                filename = "slowproc.f90"
            elif name.startswith("condveg_"):
                filename = "condveg.f90"
            elif name in ("qsfrict_init", "qsatcalc", "dev_qsatcalc"):
                filename = "qsat_moisture.f90"
            else:
                filename = "diffuco.f90"
            checks.append((canopy.ROOT / "fortran_source/ORCHIDEE/src_sechiba" / filename, identity))
    missing = sorted({str(path) for path, _ in checks if not path.is_file()})
    if missing:
        pytest.skip("Original-source identity NOT EXECUTED; missing: " + ", ".join(missing))
    for path, identity in checks:
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == identity["source_sha256"]
        block = b"".join(raw.splitlines(keepends=True)[identity["start_line"] - 1 : identity["end_line"]])
        assert hashlib.sha256(block).hexdigest() == identity.get("span_sha256", identity.get("sha256"))
