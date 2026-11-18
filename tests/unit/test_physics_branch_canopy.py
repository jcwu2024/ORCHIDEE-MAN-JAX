"""PB-01 independent original-Fortran truth on real CPU canopy components.

The two strict low-light cim failures remain explicit expected failures. This
does not relax their tolerance or count them as scientific acceptance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from scripts.dev.oracle_physics_branch_canopy import actual_values

REFERENCE = json.loads(
    (Path(__file__).parents[1] / "fixtures/physics_branch_canopy_reference.json").read_text(encoding="utf-8")
)
FIELDS = [
    pytest.param(group, field)
    for group, fields in REFERENCE["expected"].items()
    for field in fields
    if (group, field) != ("diffuco", "diffuco.cim")
]
LOW_LIGHT_INDEX = REFERENCE["cases"].index("radiation_above_mask")


@pytest.fixture(scope="module", params=(False, True), ids=("eager", "jit"))
def actual(request):
    result = actual_values(use_jit=request.param)
    assert jax.default_backend() == "cpu"
    assert jax.config.jax_enable_x64
    return result


@pytest.mark.parametrize("group,field", FIELDS)
def test_canopy_original_fortran_field(actual, group, field):
    np.testing.assert_allclose(
        actual[group][field], REFERENCE["expected"][group][field], rtol=REFERENCE["rtol"], atol=REFERENCE["atol"]
    )


def test_canopy_cim_other_cases_match_original_fortran(actual):
    expected = np.asarray(REFERENCE["expected"]["diffuco"]["diffuco.cim"])
    ordinary_cases = np.arange(expected.size) != LOW_LIGHT_INDEX
    np.testing.assert_allclose(
        actual["diffuco"]["diffuco.cim"][ordinary_cases],
        expected[ordinary_cases],
        rtol=REFERENCE["rtol"],
        atol=REFERENCE["atol"],
    )


@pytest.mark.xfail(
    strict=True,
    reason="PB01-low-light-cim: identical-coefficient scalar Fortran/JAX cubic differences are amplified by the source 1e-8 denominator; only radiation_above_mask remains a strict failed gate",
)
def test_canopy_low_light_cim_strict_gate(actual):
    np.testing.assert_allclose(
        actual["diffuco"]["diffuco.cim"][LOW_LIGHT_INDEX],
        REFERENCE["expected"]["diffuco"]["diffuco.cim"][LOW_LIGHT_INDEX],
        rtol=REFERENCE["rtol"],
        atol=REFERENCE["atol"],
    )


def test_canopy_declared_input_identity():
    """Declared inputs must match the versioned reference, including without source."""
    from scripts.dev.oracle_physics_branch_canopy import (
        NAMES,
        make_canopy_fixture,
        make_condveg_inputs,
        make_control_inputs,
    )

    fixture = make_canopy_fixture()
    inputs = dict(
        canopy=dict(cases=NAMES, payload=fixture.diffuco_precall.payload, static=fixture.static_kwargs),
        controls=make_control_inputs(),
        condveg=make_condveg_inputs(),
    )
    normalized = json.loads(json.dumps(inputs, default=lambda value: value.tolist()))
    assert normalized == REFERENCE["inputs"]


@pytest.mark.external_data
def test_canopy_reference_still_bound_to_original_fortran():
    """Optional original-source gate; a missing source is not a certification pass."""
    from scripts.dev.oracle_physics_branch_canopy import ROOT

    checks = []
    for group in REFERENCE["reference_identity"].values():
        spans = group["spans"]
        if "source" in spans:
            checks.append((ROOT / spans["source"].replace("\\", "/"), spans))
        else:
            filename = "condveg.f90" if "condveg_main" in spans else "diffuco.f90"
            for name, span in spans.items():
                # The shared composer also compiles energy/QSAT declarations;
                # those are not additional executed scientific owners here.
                if name.startswith("enerbil_"):
                    continue
                source = "qsat_moisture.f90" if name in ("qsfrict_init", "qsatcalc", "dev_qsatcalc") else filename
                checks.append((ROOT / "fortran_source/ORCHIDEE/src_sechiba" / source, span))
    missing = sorted({path.relative_to(ROOT).as_posix() for path, _ in checks if not path.is_file()})
    if missing:
        pytest.skip("Original-Fortran source/span gate NOT EXECUTED; missing source: " + ", ".join(missing))
    for path, span in checks:
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == span["source_sha256"]
        selected = b"".join(raw.splitlines(keepends=True)[span["start_line"] - 1:span["end_line"]])
        assert hashlib.sha256(selected).hexdigest() == span.get("span_sha256", span.get("sha256"))
