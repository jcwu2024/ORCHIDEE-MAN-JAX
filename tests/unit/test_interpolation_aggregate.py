from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

import jax_orchidee.driver.interpolation_aggregate as module
from jax_orchidee.driver.interpolation_aggregate import (
    AggregationVectorResult,
    MPIBackendRequiredError,
    R_EARTH,
    aggregate_2d,
    aggregate_2d_p,
    aggregate_vec,
    aggregate_vec_p,
)


# Source-extracted from interpol_help.f90 lines 296-304, 386-390 and
# 651-658, 734-738. Kinds, constants, stdin, and stdout are adapted only.
ORACLE_SOURCE = r"""
program aggregate_overlap_oracle
  implicit none
  real(8), parameter :: pi=3.1415926535897932384626433832795d0
  real(8), parameter :: R_Earth=6378000.d0, mincos=0.001d0
  real(8) :: target_lat, target_lon, resolution_x, resolution_y
  real(8) :: source_lat, source_lon, lolowrel, louprel, lalowrel, lauprel
  real(8) :: lon_up, lon_low, lat_up, lat_low, coslat, ax, ay, areaoverlap
  read(*,*) target_lat, target_lon, resolution_x, resolution_y
  read(*,*) source_lat, source_lon, lolowrel, louprel, lalowrel, lauprel
  coslat = max(cos(target_lat*pi/180.d0),mincos)*pi/180.d0*R_Earth
  lon_up = target_lon + resolution_x/(2.d0*coslat)
  lon_low = target_lon - resolution_x/(2.d0*coslat)
  coslat = pi/180.d0*R_Earth
  lat_up = target_lat + resolution_y/(2.d0*coslat)
  lat_low = target_lat - resolution_y/(2.d0*coslat)
  areaoverlap = -1.d0
  if (source_lat > lat_low .and. source_lat < lat_up .or. &
      lalowrel < lat_low .and. lauprel > lat_low .or. &
      lalowrel < lat_up .and. lauprel > lat_up) then
    if (source_lon > lon_low .and. source_lon < lon_up .or. &
        lolowrel < lon_low .and. louprel > lon_low .or. &
        lolowrel < lon_up .and. louprel > lon_up) then
      coslat = max(cos(source_lat*pi/180.d0),mincos)
      ax = (min(lon_up,louprel)-max(lon_low,lolowrel))*pi/180.d0*R_Earth*coslat
      ay = (min(lat_up,lauprel)-max(lat_low,lalowrel))*pi/180.d0*R_Earth
      areaoverlap = ax*ay
    endif
  endif
  write(*,'(ES25.16)') areaoverlap
end program aggregate_overlap_oracle
"""


@pytest.fixture(scope="session")
def aggregate_oracle(tmp_path_factory):
    env = os.environ.copy()
    ucrt = r"C:\msys64\ucrt64\bin"
    env["PATH"] = ucrt + os.pathsep + env.get("PATH", "")
    compiler = shutil.which("gfortran", path=env["PATH"])
    if compiler is None:
        pytest.skip("gfortran is required for the source-extracted aggregate oracle")
    directory = tmp_path_factory.mktemp("interpolation_aggregate_oracle")
    source = directory / "aggregate_overlap_oracle.f90"
    executable = directory / "aggregate_overlap_oracle.exe"
    source.write_text(ORACLE_SOURCE, encoding="ascii")
    subprocess.run(
        [compiler, str(source), "-O0", "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return executable, env


def _oracle(executable_and_env, target, source):
    executable, env = executable_and_env
    payload = " ".join(map(str, target)) + "\n" + " ".join(map(str, source)) + "\n"
    completed = subprocess.run(
        [str(executable)], input=payload, check=True, capture_output=True, text=True, env=env
    )
    return float(completed.stdout.strip())


def _scale(latitude=0.0):
    return np.pi / 180.0 * R_EARTH * np.cos(np.deg2rad(latitude))


def _common_target(center_lon=0.0, width_deg=2.0):
    yscale = np.pi / 180.0 * R_EARTH
    return dict(
        nbpt=1,
        lalo=np.asarray([[0.0, center_lon]]),
        neighbours=np.zeros((1, 4), dtype=np.int32),
        resolution=np.asarray([[width_deg * _scale(), width_deg * yscale]]),
        contfrac=np.asarray([0.37]),
    )


def _grid_2d(mask_center=1):
    axis = np.asarray([-1.0, 0.0, 1.0])
    lon = np.repeat(axis[:, None], 3, axis=1)
    lat = np.repeat(axis[None, :], 3, axis=0)
    mask = np.zeros((3, 3), dtype=np.int32)
    mask[1, 1] = mask_center
    return lon, lat, mask


@pytest.mark.parametrize(
    ("center_lon", "width_deg", "fraction"),
    [(0.0, 2.0, 1.0), (0.5, 1.0, 0.5)],
)
def test_aggregate_2d_full_and_partial_overlap_match_gfortran(
    aggregate_oracle, center_lon, width_deg, fraction
):
    inputs = _common_target(center_lon, width_deg)
    lon, lat, mask = _grid_2d()
    result = aggregate_2d(
        **inputs, iml=3, jml=3, lon_rel=lon, lat_rel=lat, mask=mask,
        callsign="oracle", incmax=3,
    )
    target = (0.0, center_lon, inputs["resolution"][0, 0], inputs["resolution"][0, 1])
    expected = _oracle(aggregate_oracle, target, (0.0, 0.0, -0.5, 0.5, -0.5, 0.5))
    assert result.ok
    np.testing.assert_array_equal(result.indinc[0], [[2, 2], [0, 0], [0, 0]])
    assert result.areaoverlap[0, 0] == pytest.approx(expected, rel=2e-14)
    assert result.areaoverlap[0, 0] == pytest.approx(fraction * _scale() ** 2)
    np.testing.assert_array_equal(result.areaoverlap[0, 1:], [-1.0, -1.0])
    np.testing.assert_allclose(result.weights[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.counts, [1])


def test_aggregate_2d_mask_no_overlap_and_undefined_tail():
    inputs = _common_target(0.0, 2.0)
    lon, lat, mask = _grid_2d(mask_center=0)
    masked = aggregate_2d(
        **inputs, iml=3, jml=3, lon_rel=lon, lat_rel=lat, mask=mask,
        callsign="masked", incmax=2,
    )
    np.testing.assert_array_equal(masked.indinc, 0)
    np.testing.assert_array_equal(masked.areaoverlap, -1.0)
    np.testing.assert_array_equal(masked.weights, 0.0)

    far = _common_target(20.0, 1.0)
    mask[1, 1] = 1
    absent = aggregate_2d(
        **far, iml=3, jml=3, lon_rel=lon, lat_rel=lat, mask=mask,
        callsign="absent", incmax=2,
    )
    np.testing.assert_array_equal(absent.indinc, 0)
    np.testing.assert_array_equal(absent.areaoverlap, -1.0)


def test_no_overlap_and_degenerate_polygon_match_gfortran(aggregate_oracle):
    yscale = np.pi / 180.0 * R_EARTH
    far_target = (0.0, 20.0, _scale(), yscale)
    source = (0.0, 0.0, -0.5, 0.5, -0.5, 0.5)
    assert _oracle(aggregate_oracle, far_target, source) == -1.0

    full_target = (0.0, 0.0, 2.0 * _scale(), 2.0 * yscale)
    degenerate_source = (0.0, 0.0, 0.0, 0.0, -0.5, 0.5)
    expected = _oracle(aggregate_oracle, full_target, degenerate_source)
    actual = module._rectangle_overlap_area(-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, -0.5, 0.5, 0.0)
    assert actual == expected == 0.0


def test_aggregate_2d_rejects_degenerate_source_and_reports_slot_overflow():
    inputs = _common_target()
    with pytest.raises(ValueError, match="iml>=2 and jml>=2"):
        aggregate_2d(
            **inputs, iml=1, jml=1, lon_rel=[[0.0]], lat_rel=[[0.0]], mask=[[1]],
            callsign="degenerate", incmax=1,
        )
    lon, lat, mask = _grid_2d()
    overflow = aggregate_2d(
        **inputs, iml=3, jml=3, lon_rel=lon, lat_rel=lat, mask=np.ones_like(mask),
        callsign="overflow", incmax=0,
    )
    assert not overflow.ok


@pytest.mark.parametrize(
    ("center_lon", "width_deg", "fraction"),
    [(0.0, 2.0, 1.0), (0.25, 1.0, 0.75)],
)
def test_aggregate_vec_full_and_partial_overlap_match_gfortran(
    aggregate_oracle, center_lon, width_deg, fraction
):
    inputs = _common_target(center_lon, width_deg)
    result = aggregate_vec(
        **inputs, iml=1, lon_rel=[0.0], lat_rel=[0.0],
        resol_lon=_scale(), resol_lat=_scale(), callsign="vector", incmax=3,
    )
    target = (0.0, center_lon, inputs["resolution"][0, 0], inputs["resolution"][0, 1])
    expected = _oracle(aggregate_oracle, target, (0.0, 0.0, -0.5, 0.5, -0.5, 0.5))
    assert result.ok
    np.testing.assert_array_equal(result.indinc[0], [1, 0, 0])
    assert result.areaoverlap[0, 0] == pytest.approx(expected, rel=2e-14)
    assert result.areaoverlap[0, 0] == pytest.approx(fraction * _scale() ** 2)
    np.testing.assert_allclose(result.weights[0], [1.0, 0.0, 0.0])


def test_aggregate_vec_strict_boundary_touch_is_not_an_overlap():
    inputs = _common_target(0.5, 1.0)
    result = aggregate_vec(
        **inputs, iml=1, lon_rel=[0.0], lat_rel=[0.0],
        resol_lon=_scale(), resol_lat=_scale(), callsign="boundary", incmax=1,
    )
    np.testing.assert_array_equal(result.indinc, 0)
    np.testing.assert_array_equal(result.areaoverlap, -1.0)


def test_aggregate_vec_preserves_source_slot_order_normalization_and_no_overlap():
    inputs = _common_target(0.0, 4.0)
    result = aggregate_vec(
        **inputs, iml=2, lon_rel=[-0.5, 0.5], lat_rel=[0.0, 0.0],
        resol_lon=_scale(), resol_lat=_scale(), callsign="slots", incmax=3,
    )
    np.testing.assert_array_equal(result.indinc[0], [1, 2, 0])
    np.testing.assert_allclose(result.weights[0], [0.5, 0.5, 0.0], rtol=2e-14)
    far = _common_target(20.0, 1.0)
    absent = aggregate_vec(
        **far, iml=1, lon_rel=[0.0], lat_rel=[0.0],
        resol_lon=_scale(), resol_lat=_scale(), callsign="absent", incmax=2,
    )
    np.testing.assert_array_equal(absent.indinc, 0)
    np.testing.assert_array_equal(absent.areaoverlap, -1.0)
    np.testing.assert_array_equal(absent.weights, 0.0)


def test_unread_contfrac_and_neighbour_values_do_not_affect_overlap():
    base = _common_target()
    reference = aggregate_vec(
        **base, iml=1, lon_rel=[0.0], lat_rel=[0.0], resol_lon=_scale(),
        resol_lat=_scale(), callsign="reference", incmax=1,
    )
    base["contfrac"] = np.asarray([np.nan])
    base["neighbours"] = np.asarray([[-99, 0, 2**30, -7]], dtype=np.int64)
    unread_values = aggregate_vec(
        **base, iml=1, lon_rel=[0.0], lat_rel=[0.0], resol_lon=_scale(),
        resol_lat=_scale(), callsign="unread", incmax=1,
    )
    np.testing.assert_array_equal(unread_values.indinc, reference.indinc)
    np.testing.assert_array_equal(unread_values.areaoverlap, reference.areaoverlap)

    lon, lat, mask = _grid_2d()
    base_2d = _common_target()
    reference_2d = aggregate_2d(
        **base_2d, iml=3, jml=3, lon_rel=lon, lat_rel=lat, mask=mask,
        callsign="reference_2d", incmax=1,
    )
    base_2d["contfrac"] = np.asarray([2.5])
    base_2d["neighbours"] = np.asarray([[9, -3, 0, 44]])
    unread_2d = aggregate_2d(
        **base_2d, iml=3, jml=3, lon_rel=lon, lat_rel=lat, mask=mask,
        callsign="unread_2d", incmax=1,
    )
    np.testing.assert_array_equal(unread_2d.indinc, reference_2d.indinc)
    np.testing.assert_array_equal(unread_2d.areaoverlap, reference_2d.areaoverlap)


def test_parallel_serial_root_calls_same_core(monkeypatch):
    sentinel = AggregationVectorResult(
        np.asarray([[7]], dtype=np.int32), np.asarray([[2.0]]), True
    )
    seen = {}

    def fake_core(*args, **kwargs):
        seen["call"] = (args, kwargs)
        return sentinel

    monkeypatch.setattr(module, "aggregate_vec", fake_core)
    result = aggregate_vec_p("payload", marker=3, grid_type="RegLonLat", size=1)
    assert result is sentinel
    assert seen["call"] == (("payload",), {"marker": 3})


def test_parallel_boundaries_are_explicit():
    with pytest.raises(MPIBackendRequiredError, match="MPI backend"):
        aggregate_vec_p(size=2)
    with pytest.raises(MPIBackendRequiredError, match="MPI backend"):
        aggregate_2d_p(size=2)
    with pytest.raises(ValueError, match="RegXY requires explicit"):
        aggregate_vec_p(grid_type="RegXY")
    with pytest.raises(ValueError, match="only RegLonLat or RegXY"):
        aggregate_2d_p(grid_type="UnStruct")


def test_aggregate_2d_p_serial_matches_direct_core():
    inputs = _common_target()
    lon, lat, mask = _grid_2d()
    kwargs = dict(
        **inputs, iml=3, jml=3, lon_rel=lon, lat_rel=lat, mask=mask,
        callsign="serial", incmax=2,
    )
    direct = aggregate_2d(**kwargs)
    wrapped = aggregate_2d_p(**kwargs, grid_type="RegLonLat", size=1)
    np.testing.assert_array_equal(wrapped.indinc, direct.indinc)
    np.testing.assert_array_equal(wrapped.areaoverlap, direct.areaoverlap)
