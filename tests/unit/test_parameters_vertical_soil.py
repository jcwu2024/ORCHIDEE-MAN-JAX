from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path

import numpy as np
import pytest

from jax_orchidee.parameters.vertical_soil import VerticalSoilParameterError, vertical_soil_init


ROOT = Path(__file__).resolve().parents[2]
ORACLE_SOURCE = ROOT / "tests" / "fortran_oracles" / "vertical_soil_geometry_oracle.f90"
GFORTRAN = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")


@pytest.fixture(scope="module")
def vertical_oracle(tmp_path_factory):
    compiler = GFORTRAN if GFORTRAN.exists() else shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran is unavailable")
    executable = tmp_path_factory.mktemp("vertical_soil_oracle") / "vertical_soil_oracle.exe"
    env = os.environ.copy()
    if GFORTRAN.exists():
        env["PATH"] = rf"{GFORTRAN.parent};C:\msys64\usr\bin;{env['PATH']}"
    subprocess.run(
        [str(compiler), "-O0", str(ORACLE_SOURCE), "-o", str(executable)],
        check=True,
        env=env,
    )
    return executable, env


def _oracle(oracle, values: dict[str, float]):
    executable, env = oracle
    args = [values[key] for key in (
        "DEPTH_MAX_T", "DEPTH_MAX_H", "DEPTH_TOPTHICK", "DEPTH_CSTTHICK",
        "DEPTH_GEOM", "RATIO_GEOM_BELOW",
    )]
    completed = subprocess.run(
        [str(executable), *(str(value) for value in args)], check=True,
        capture_output=True, text=True, env=env,
    )
    lines = completed.stdout.strip().splitlines()
    nslm, ngrnd = (int(value) for value in lines[0].split())
    return nslm, ngrnd, [np.fromstring(line, sep=" ") for line in lines[1:]]


@pytest.mark.parametrize("values", [
    {"DEPTH_MAX_T": 38.0, "DEPTH_MAX_H": 2.0, "DEPTH_TOPTHICK": 9.77517107e-4,
     "DEPTH_CSTTHICK": 2.0, "DEPTH_GEOM": 2.0, "RATIO_GEOM_BELOW": 1.05},
    {"DEPTH_MAX_T": 12.0, "DEPTH_MAX_H": 3.0, "DEPTH_TOPTHICK": 0.002,
     "DEPTH_CSTTHICK": 0.4, "DEPTH_GEOM": 4.0, "RATIO_GEOM_BELOW": 1.2},
    {"DEPTH_MAX_T": 8.0, "DEPTH_MAX_H": 1.5, "DEPTH_TOPTHICK": 0.0015,
     "DEPTH_CSTTHICK": 1.0, "DEPTH_GEOM": 1.5, "RATIO_GEOM_BELOW": 1.1},
])
def test_dynamic_geometry_matches_source_extracted_fortran_oracle(vertical_oracle, values):
    expected_nslm, expected_ngrnd, expected = _oracle(vertical_oracle, values)
    result = vertical_soil_init(values)
    assert (result.nslm, result.ngrnd) == (expected_nslm, expected_ngrnd)
    actual = (result.znh, result.dnh, result.dlh, result.zlh, result.znt, result.dlt, result.zlt)
    for jax_value, oracle_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(jax_value, oracle_value, rtol=2e-14, atol=2e-14)


def test_defaults_are_dynamic_and_geometry_identities_hold():
    result = vertical_soil_init()
    assert result.znh.shape == (result.nslm,)
    assert result.znt.shape == (result.ngrnd,)
    np.testing.assert_allclose(result.zlh, result.zlt[: result.nslm])
    np.testing.assert_allclose(result.dlh[:-1], (result.dnh[:-1] + result.dnh[1:]) / 2.0)
    assert result.znh[-1] == result.depth_max_h
    assert result.zlt[-1] == result.depth_max_t


def test_depth_cstthick_warning_arm_applies_fortran_correction():
    result = vertical_soil_init({"DEPTH_MAX_H": 2.0, "DEPTH_CSTTHICK": 1.5})
    assert result.depth_cstthickness == 2.0


def test_vertical_reads_fortran_d_exponents():
    result = vertical_soil_init({
        "DEPTH_MAX_T": "3.8D1", "DEPTH_MAX_H": "2d0",
        "DEPTH_TOPTHICK": "9.77517107D-4", "DEPTH_CSTTHICK": "2D0",
        "DEPTH_GEOM": "2D0", "RATIO_GEOM_BELOW": "1.05D0",
    })
    assert result.depth_max_t == 38.0
    assert result.depth_topthickness == 9.77517107e-4


@pytest.mark.parametrize("values,match", [
    ({"DEPTH_MAX_H": 3.0, "DEPTH_MAX_T": 2.0}, "DEPTH_MAX_H"),
    ({"DEPTH_MAX_H": 2.0, "DEPTH_GEOM": 1.0}, "DEPTH_GEOM"),
])
def test_fatal_source_guards(values, match):
    with pytest.raises(VerticalSoilParameterError, match=match):
        vertical_soil_init(values)


def test_nblayermax_rejects_silent_float_truncation():
    with pytest.raises(TypeError, match="nblayermax"):
        vertical_soil_init(nblayermax=20.5)
