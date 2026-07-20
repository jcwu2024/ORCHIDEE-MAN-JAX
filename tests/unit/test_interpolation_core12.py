from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from jax_orchidee.driver.interpolation_core12 import (
    AggregatePacket,
    FortranUndefinedVariableError,
    InterpolationSource,
    InterpolationTarget,
    interpweight_1d,
    interpweight_2d,
)


GFORTRAN = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")


def _target(*, resolution=10.0, contfrac=(0.5, 1.0)):
    return InterpolationTarget(
        lalo=np.array([[45.0, 2.0], [46.0, 3.0]]),
        resolution=np.full((2, 2), resolution, dtype=np.float64),
        neighbours=np.arange(16, dtype=np.int32).reshape(2, 8) + 1,
        contfrac=np.asarray(contfrac, dtype=np.float64),
    )


def _packet(request, *, one_dimensional=False, no_overlap_second=True):
    index_shape = (2, request.nbvmax) if one_dimensional else (2, request.nbvmax, 2)
    index = np.zeros(index_shape, dtype=np.int32)
    area = np.zeros((2, request.nbvmax), dtype=np.float64)
    if one_dimensional:
        index[0, :2] = [1, 2]
    else:
        index[0, :2] = [[1, 1], [2, 1]]
    area[0, :2] = [10.0, 15.0]
    if not no_overlap_second:
        if one_dimensional:
            index[1, 0] = 3
        else:
            index[1, 0] = [1, 2]
        area[1, 0] = 20.0
    return AggregatePacket(index, area, True)


def test_interpweight_1d_preserves_order_mask_geometry_retry_and_scaling():
    seen = []

    def aggregate(request):
        seen.append(request)
        if len(seen) == 1:
            return AggregatePacket(
                np.zeros((2, request.nbvmax), dtype=np.int32),
                np.zeros((2, request.nbvmax)),
                False,
            )
        return _packet(request, one_dimensional=True)

    result = interpweight_1d(
        InterpolationSource(
            values=np.array([-1.0, 2.0, 1.0]),
            longitude=np.array([1.0, 2.0, 3.0]),
            latitude=np.array([40.0, 41.0, 42.0]),
            variable_name="olson",
        ),
        _target(),
        [-1.0, 99.0],
        aggregate,
        varmin=8.0,
        varmax=9.0,
        noneg=True,
        masktype="mabove",
        maskvalues=[0.5, 0.0, 0.0],
        max_resolution_lon=5.0,
        max_resolution_lat=5.0,
    )

    np.testing.assert_allclose(result.fractions, [[0.0, 0.6], [1.0, 0.0]])
    np.testing.assert_allclose(result.availability, [0.5, -1.0])
    np.testing.assert_array_equal(result.mask, [0, 1, 1])
    np.testing.assert_array_equal(result.variable_use_types, [1.0, 2.0])
    assert (result.varmin, result.varmax) == (1.0, 2.0)
    assert result.nbvmax_attempts == (200, 400)
    np.testing.assert_array_equal(seen[0].neighbours, _target().neighbours)
    assert seen[0].mask is None  # The 1D aggregate_vec_p signature has no mask.
    assert seen[0].callsign == "olson map"
    assert seen[0].nbvmax == 200 and seen[1].nbvmax == 400


@pytest.mark.parametrize(
    ("masktype", "mask_variable", "expected"),
    [
        ("nomask", None, [1, 1, 1]),
        ("mbelow", None, [1, 0, 0]),
        ("mabove", None, [0, 1, 1]),
        ("var", np.array([0.0, 3.0, -2.0]), [0, 1, 0]),
    ],
)
def test_interpweight_1d_mask_dispatch(masktype, mask_variable, expected):
    def aggregate(request):
        return _packet(request, one_dimensional=True, no_overlap_second=False)

    result = interpweight_1d(
        InterpolationSource([0.0, 2.0, 1.0], [1.0, 2.0, 3.0], [40.0, 41.0, 42.0], mask_variable, "olson"),
        _target(),
        [1.0, 2.0],
        aggregate,
        varmin=1.0,
        varmax=2.0,
        noneg=False,
        masktype=masktype,
        maskvalues=[0.5, 0.0, 0.0],
        max_resolution_lon=5.0,
        max_resolution_lat=5.0,
    )
    np.testing.assert_array_equal(result.mask, expected)


def test_interpweight_1d_undefined_and_fatal_guards_are_explicit():
    source = InterpolationSource([1.0], [1.0], [40.0], variable_name="olson")
    with pytest.raises(ValueError, match="number of dimensions"):
        interpweight_1d(
            InterpolationSource([[1.0]], [1.0], [40.0], variable_name="olson"), _target(), [1.0], lambda _: None,
            varmin=1.0, varmax=1.0, noneg=False, masktype="nomask", maskvalues=[0, 0, 0],
            max_resolution_lon=5.0, max_resolution_lat=5.0,
        )
    with pytest.raises(FortranUndefinedVariableError, match="rank-2 resolution"):
        interpweight_1d(
            source, _target(), [1.0], lambda _: None,
            varmin=1.0, varmax=1.0, noneg=False, masktype="nomask", maskvalues=[0, 0, 0],
        )
    with pytest.raises(FortranUndefinedVariableError, match="mask_variable"):
        interpweight_1d(
            source, _target(), [1.0], lambda _: None,
            varmin=1.0, varmax=1.0, noneg=False, masktype="var", maskvalues=[0, 0, 0],
            max_resolution_lon=5.0, max_resolution_lat=5.0,
        )
    with pytest.raises(RuntimeError, match="no sens"):
        interpweight_1d(
            source, _target(), [1.0], lambda _: None,
            varmin=1.0, varmax=1.0, noneg=False, masktype="msumrange", maskvalues=[0, 0, 0],
            max_resolution_lon=5.0, max_resolution_lat=5.0,
        )


def test_interpweight_2d_axis_coordinates_mask_fraction_and_calculated_largegrid():
    seen = []

    def aggregate(request):
        seen.append(request)
        return _packet(request)

    source = InterpolationSource(
        values=np.array([[1.0, -3.0], [2.0, 1.0]]),
        longitude=np.array([0.0, 0.001]),
        latitude=np.array([45.0, 45.001]),
        variable_name="soilcolor",
    )
    result = interpweight_2d(
        source,
        _target(resolution=1_500.0),
        [1.0, 2.0],
        aggregate,
        varmin=1.0,
        varmax=2.0,
        noneg=True,
        masktype="mbelow",
        maskvalues=[0.5, 0.0, 0.0],
    )

    np.testing.assert_allclose(result.fractions[0], [0.4, 0.6])
    np.testing.assert_allclose(result.availability[0], 25.0 / 1_500.0**2 / 0.5)
    np.testing.assert_array_equal(result.mask, [[0, 1], [0, 0]])
    assert result.nbvmax_attempts[0] > 200
    np.testing.assert_allclose(seen[0].longitude[:, 0], [0.0, 0.001])
    np.testing.assert_allclose(seen[0].latitude[0, :], [45.0, 45.001])
    np.testing.assert_array_equal(seen[0].mask, result.mask)


@pytest.mark.parametrize(
    ("masktype", "values", "thresholds", "expected"),
    [
        ("nomask", [[1.0, 2.0], [3.0, 4.0]], [0.0, 0.0, 0.0], [[1, 1], [1, 1]]),
        ("mbelow", [[-1.0, 2.0], [3.0, 4.0]], [0.0, 0.0, 0.0], [[1, 0], [0, 0]]),
        ("mabove", [[1.0, 2.0], [3.0, 4.0]], [3.5, 0.0, 0.0], [[0, 0], [0, 1]]),
        ("msumrange", [[1.0, 3.0], [1.0, 3.0]], [3.0, 1.0, 5.0], [[1, 1], [1, 1]]),
    ],
)
def test_interpweight_2d_mask_dispatch_on_full_coordinates(masktype, values, thresholds, expected):
    def aggregate(request):
        return _packet(request, no_overlap_second=False)

    result = interpweight_2d(
        InterpolationSource(
            values=np.asarray(values),
            longitude=np.array([[0.0, 0.0], [1.0, 1.0]]),
            latitude=np.array([[40.0, 41.0], [40.0, 41.0]]),
            variable_name="soilcolor",
        ),
        _target(), [1.0, 2.0, 3.0, 4.0], aggregate,
        varmin=1.0, varmax=4.0, noneg=False, masktype=masktype, maskvalues=thresholds,
        max_resolution_lon=5.0, max_resolution_lat=5.0,
    )
    np.testing.assert_array_equal(result.mask, expected)


def test_interpweight_2d_var_mask_uses_explicit_io_value():
    captured = []

    def aggregate(request):
        captured.append(request.mask)
        return _packet(request, no_overlap_second=False)

    result = interpweight_2d(
        InterpolationSource(
            [[1.0, 2.0], [2.0, 1.0]], [0.0, 1.0], [40.0, 41.0],
            [[0.0, 2.0], [-1.0, 1.0]],
            "soilcolor",
        ),
        _target(), [1.0, 2.0], aggregate,
        varmin=1.0, varmax=2.0, noneg=False, masktype="var", maskvalues=[0, 0, 0],
        max_resolution_lon=5.0, max_resolution_lat=5.0,
    )
    np.testing.assert_array_equal(result.mask, [[0, 1], [0, 1]])
    np.testing.assert_array_equal(captured[0], result.mask)


@pytest.mark.parametrize(
    ("shape", "initime", "expected_rank"),
    [((2, 2, 2), -1, 3), ((2, 2, 2, 3), -1, 4), ((2, 2, 2, 3), 0, 3), ((2, 2, 2, 3), 2, 3), ((2, 2, 2, 3), 99, 3)],
)
def test_interpweight_2d_rank3_rank4_and_time_dispatch_reach_source_undefined_boundary(shape, initime, expected_rank):
    seen = []

    def aggregate(request):
        seen.append(request.source_rank)
        return _packet(request, no_overlap_second=False)

    with pytest.raises(FortranUndefinedVariableError, match="invar2D"):
        interpweight_2d(
            InterpolationSource(np.ones(shape), [0.0, 1.0], [40.0, 41.0], variable_name="ranked"),
            _target(), [1.0], aggregate,
            varmin=1.0, varmax=1.0, noneg=True, masktype="nomask", maskvalues=[0, 0, 0],
            initime=initime, max_resolution_lon=5.0, max_resolution_lat=5.0,
        )
    assert seen == [expected_rank]


def test_interpweight_2d_shape_dispatch_and_guards():
    kwargs = dict(
        varmin=1.0, varmax=1.0, noneg=False, masktype="nomask", maskvalues=[0, 0, 0],
        max_resolution_lon=5.0, max_resolution_lat=5.0,
    )
    with pytest.raises(ValueError, match="number of dimensions"):
        interpweight_2d(InterpolationSource([1.0], [0.0], [40.0], variable_name="bad"), _target(), [1.0], lambda _: None, **kwargs)
    with pytest.raises(ValueError, match="axis coordinates"):
        interpweight_2d(
            InterpolationSource(np.ones((2, 2)), [0.0], [40.0, 41.0], variable_name="bad"),
            _target(), [1.0], lambda _: None, **kwargs,
        )
    with pytest.raises(ValueError, match="not ready"):
        interpweight_2d(
            InterpolationSource(np.ones((2, 2)), [0.0, 1.0], [40.0, 41.0], variable_name="bad"),
            _target(), [1.0], lambda _: None, **{**kwargs, "masktype": "unknown"},
        )
    with pytest.raises(ValueError, match="variable_name"):
        interpweight_2d(
            InterpolationSource(np.ones((2, 2)), [0.0, 1.0], [40.0, 41.0]),
            _target(), [1.0], lambda request: _packet(request), **kwargs,
        )


@pytest.fixture(scope="module")
def interpolation_oracle(tmp_path_factory):
    compiler = GFORTRAN if GFORTRAN.exists() else shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran is unavailable")
    directory = tmp_path_factory.mktemp("interpolation_core12_oracle")
    source = directory / "oracle.f90"
    executable = directory / "oracle.exe"
    source.write_text(textwrap.dedent("""
        program oracle
          implicit none
          integer :: nix,njx,nbvmax,ip,jv,idi
          real(8) :: resolution(2),maxres(2),ivar(2),sarea(2),types(2)
          real(8) :: ovar(2),aout,scaled
          integer :: sindex(2)
          resolution=(/10d0,10d0/); maxres=(/5d0,5d0/)
          nix=INT(resolution(1)/maxres(1))+2
          njx=INT(resolution(2)/maxres(2))+2
          nbvmax=nix*njx
          IF (nbvmax < 200) nbvmax=200
          ivar=(/1d0,2d0/); sarea=(/10d0,15d0/); sindex=(/1,2/)
          types=(/1d0,2d0/); ovar=0d0; aout=0d0
          idi=COUNT(sarea > 0d0)
          IF (idi > 0) THEN
            DO ip=1,idi
              DO jv=1,2
                IF (ivar(sindex(ip)) == types(jv)) ovar(jv)=ovar(jv)+sarea(ip)
              ENDDO
              aout=aout+sarea(ip)
            ENDDO
            ovar=ovar/aout
          ENDIF
          scaled=aout/(resolution(1)*resolution(2))/0.5d0
          write(*,'(I0,1X,4(ES25.16,1X))') nbvmax,ovar(1),ovar(2),aout,scaled
          aout=0d0; ovar=0d0
          idi=COUNT((/0d0,0d0/) > 0d0)
          IF (idi > 0) THEN
            stop 2
          ELSE
            aout=-1d0
            ovar(INT((2d0+1d0)/2d0))=1d0
          ENDIF
          write(*,'(3(ES25.16,1X))') ovar(1),ovar(2),aout
        end program oracle
    """), encoding="ascii")
    env = os.environ.copy()
    if GFORTRAN.exists():
        env["PATH"] = rf"{GFORTRAN.parent};C:\msys64\usr\bin;{env['PATH']}"
    compiled = subprocess.run(
        [str(compiler), "-O0", str(source), "-o", str(executable)],
        capture_output=True, text=True, env=env,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    completed = subprocess.run([str(executable)], capture_output=True, text=True, env=env)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout.splitlines()


def test_source_extracted_gfortran_oracle_matches_numerical_chain(interpolation_oracle):
    weighted = np.fromstring(interpolation_oracle[0], sep=" ")
    fallback = np.fromstring(interpolation_oracle[1], sep=" ")
    assert int(weighted[0]) == 200
    np.testing.assert_allclose(weighted[1:], [0.4, 0.6, 25.0, 0.5], rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(fallback, [1.0, 0.0, -1.0], rtol=0.0, atol=0.0)
