"""Current original-source snow maxmass boundaries and owner return state.

The fixture is from the new21-case333700 reference, not the historical333600
snow oracle. Complete compiled grain at the extreme-deposition boundary is
an explicit strict xfail: original O0 versus O3/FMA also changes the ==0 arm.
No physical natural-reachability or full-model acceptance is implied.
"""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from jax_orchidee.sechiba import hydrol
from scripts.dev.oracle_snow_maxmass_domain import _path_inputs, snow_owner_arrays

REFERENCE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "snow_maxmass_domain_current_source.json").read_text(
        encoding="utf-8"
    )
)
CASES = REFERENCE["cases"]
HELPERS = [c for c in CASES if c["inputs"]["mode"] == "helper"]
PATHS = [c for c in CASES if c["inputs"]["mode"] == "path"]
RTOL, ATOL = 1e-12, 1e-14


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "narrow_jit"])
def test_current_source_maxmass_all_locations_caps_ties_and_return_heat(compiled):
    rho = np.array([c["inputs"]["rho"] for c in HELPERS])
    depth = np.array([c["inputs"]["dz"] for c in HELPERS])
    capacity = np.array([c["inputs"]["soilcap"] for c in HELPERS])
    temperature = np.array([c["inputs"]["temperature"] for c in HELPERS])
    liquid = np.array([c["inputs"]["liquid"] for c in HELPERS])

    def component(cap, rho, depth, temp, liq):
        mass = hydrol.explicitsnow_maxmass_step(soilcap=cap, snowrho=rho, snowdz=depth)
        excess = hydrol.explicitsnow_liquid_heat_excess_step(
            snowliq=liq, snowdz=mass.snowdz, snowrho=rho, grndflux=np.zeros(cap.shape)
        )
        heat = hydrol.snow3lheat_explicit(excess.snowliq, rho, mass.snowdz, temp)
        return mass.snow, mass.snowdz, mass.snowmelt_from_maxmass, excess.snowliq, excess.grndflux, heat

    values = (jax.jit(component) if compiled else component)(capacity, rho, depth, temperature, liquid)
    for i, case in enumerate(HELPERS):
        for field, value in zip(("mass", "snowdz", "removal", "snowliq", "grndflux", "snowheat"), values):
            np.testing.assert_allclose(
                np.atleast_1d(np.asarray(value)[i]),
                case["expected"][field],
                rtol=RTOL,
                atol=ATOL,
                err_msg=case["inputs"]["id"] + "/" + field,
            )
        np.testing.assert_allclose(
            np.sum(rho[i] * depth[i]) - np.asarray(values[0])[i], np.asarray(values[2])[i], rtol=RTOL, atol=ATOL
        )


COMPILED_OWNER = jax.jit(snow_owner_arrays)


class KnownSnowGrainMismatch(AssertionError):
    """Only the documented finite grain residual may be an expected failure."""


OWNER_CASES = []
for c in PATHS:
    for compiled in (False, True):
        marks = []
        if compiled and c["inputs"]["id"] == "negative_sublimation_crossing":
            marks = [
                pytest.mark.xfail(
                    strict=True,
                    raises=KnownSnowGrainMismatch,
                    reason="Original-source ==0 liquid/grain arm is sensitive to FMA; full JIT grain differs from strict O0 reference. No tolerance or source algorithm changed.",
                )
            ]
        OWNER_CASES.append(
            pytest.param(c, compiled, marks=marks, id=c["inputs"]["id"] + ("-whole-jit" if compiled else "-eager"))
        )


@pytest.mark.parametrize("case,compiled", OWNER_CASES)
def test_current_source_complete_snow_owner_return(case, compiled):
    values = _path_inputs(case["inputs"])
    # Both sides consume the identical emitted source entry enthalpy.
    values["snowheat"] = np.array([case["entry_heat"]])
    actual = (COMPILED_OWNER if compiled else snow_owner_arrays)(values)
    known_grain_case = compiled and case["inputs"]["id"] == "negative_sublimation_crossing"
    for field, expected in case["expected"].items():
        if known_grain_case and field == "snowgrain":
            continue
        name = "snow" if field == "mass" else field
        np.testing.assert_allclose(np.asarray(actual[name]).ravel(), expected, rtol=RTOL, atol=ATOL, err_msg=field)
    if known_grain_case:
        # Check every other field first: an unrelated failure must remain red.
        grain = np.asarray(actual["snowgrain"]).ravel()
        assert np.all(np.isfinite(grain)), "Nonfinite grain is not the known finite residual"
        try:
            np.testing.assert_allclose(grain, case["expected"]["snowgrain"], rtol=RTOL, atol=ATOL, err_msg="snowgrain")
        except AssertionError as exc:
            raise KnownSnowGrainMismatch(str(exc)) from exc
