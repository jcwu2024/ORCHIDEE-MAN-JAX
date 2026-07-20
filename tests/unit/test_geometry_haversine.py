from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from jax_orchidee.driver.geometry_haversine import (
    HaversineGeometryError,
    haversine_clockwise,
    haversine_distance,
    haversine_heading,
    haversine_laloarea,
    haversine_laloseglen,
    haversine_polyheadings,
    haversine_polysort,
    haversine_radialdis,
    haversine_reglatlontoploy,
    haversine_regxytoploy,
    haversine_singlepointploy,
)


GFORTRAN = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")

# Direct extraction of haversine.f90 lines 933-1013 with only module kinds and
# constants replaced. This is deliberately not an independently derived oracle.
HAVERSINE_ORACLE = r"""
program haversine_oracle
  implicit none
  real(8), parameter :: pi=acos(-1.0d0), r_earth=6378000.0d0
  real(8) :: ls,as,le,ae,dlon,dlat,lat1,lat2,a,c,y,x,head,dis
  real(8) :: lon1,tc,lat,lon,radial,rdlon
  character(64) :: arg
  call get_command_argument(1,arg); read(arg,*) ls
  call get_command_argument(2,arg); read(arg,*) as
  call get_command_argument(3,arg); read(arg,*) le
  call get_command_argument(4,arg); read(arg,*) ae
  dlon=(le-ls)*pi/180.0d0; lat1=as*pi/180.0d0; lat2=ae*pi/180.0d0
  y=sin(dlon)*cos(lat2)
  x=cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(dlon)
  head=mod(atan2(y,x)*180.0d0/pi+360.0d0,360.0d0)
  dlat=(ae-as)*pi/180.0d0
  a=sin(dlat/2.0d0)**2+sin(dlon/2.0d0)**2*cos(lat1)*cos(lat2)
  c=2.0d0*atan2(sqrt(a),sqrt(1.0d0-a)); dis=c*r_earth
  radial=500000.0d0; tc=270.0d0*pi/180.0d0; lon1=ls*pi/180.0d0
  lat=asin(sin(lat1)*cos(radial/r_earth)+cos(lat1)*sin(radial/r_earth)*cos(tc))
  rdlon=atan2(sin(tc)*sin(radial/r_earth)*cos(lat1), &
       cos(radial/r_earth)-sin(lat1)*sin(lat))
  lon=mod(lon1+rdlon+pi,2.0d0*pi)-pi
  write(*,'(4(ES25.17E3,1X))') head,dis,lon*180.0d0/pi,lat*180.0d0/pi
end program haversine_oracle
"""


@pytest.fixture(scope="module")
def haversine_oracle(tmp_path_factory):
    compiler = GFORTRAN if GFORTRAN.exists() else shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran is unavailable")
    work = tmp_path_factory.mktemp("haversine_micro_oracle")
    source = work / "haversine_oracle.f90"
    source.write_text(HAVERSINE_ORACLE, encoding="ascii")
    executable = work / "haversine_oracle.exe"
    env = os.environ.copy()
    if GFORTRAN.exists():
        env["PATH"] = rf"{GFORTRAN.parent};C:\msys64\usr\bin;{env['PATH']}"
    subprocess.run([str(compiler), "-O0", str(source), "-o", str(executable)], check=True, env=env)
    return executable, env


@pytest.mark.parametrize(
    "values",
    [(0.0, 0.0, 90.0, 0.0), (12.3, -45.0, -130.0, 62.0), (-179.0, 15.0, 179.0, -20.0)],
)
def test_spherical_primitives_match_source_extracted_gfortran(haversine_oracle, values):
    executable, env = haversine_oracle
    completed = subprocess.run(
        [str(executable), *(str(value) for value in values)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    expected = np.fromstring(completed.stdout, sep=" ")
    radial = haversine_radialdis(values[0], values[1], 270.0, 500000.0)
    actual = np.asarray([
        haversine_heading(*values), haversine_distance(*values), radial[0], radial[1]
    ])
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-12)


def _regular_grid():
    lon = np.repeat(np.asarray([[0.0], [10.0], [20.0]]), 3, axis=1)
    lat = np.repeat(np.asarray([[-10.0, 0.0, 10.0]]), 3, axis=0)
    return lon, lat


def test_regular_lonlat_polygon_and_global_neighbours_preserve_fortran_order():
    lon, lat = _regular_grid()
    result = haversine_reglatlontoploy(lon, lat, np.arange(1, 10), global_grid=True)
    np.testing.assert_array_equal(result.iorig, [1, 2, 3, 1, 2, 3, 1, 2, 3])
    np.testing.assert_array_equal(result.jorig, [1, 1, 1, 2, 2, 2, 3, 3, 3])
    np.testing.assert_allclose(result.lonpoly[4], [5, 10, 15, 15, 15, 10, 5, 5])
    np.testing.assert_allclose(result.latpoly[4], [-5, -5, -5, 0, 5, 5, 5, 0])
    np.testing.assert_array_equal(result.neighb_loc[4], [1, 2, 3, 6, 9, 8, 7, 4])
    assert result.neighb_loc[3, 7] == 6  # cyclic west neighbour at i=1


def test_regular_grid_degenerate_guards_and_projection_code():
    with pytest.raises(HaversineGeometryError, match="longitude"):
        haversine_reglatlontoploy(np.zeros((1, 3)), np.zeros((1, 3)), [1])
    with pytest.raises(HaversineGeometryError, match="latitude"):
        haversine_reglatlontoploy(np.zeros((3, 1)), np.zeros((3, 1)), [1])
    lon, lat = _regular_grid()
    with pytest.raises(HaversineGeometryError, match="projection"):
        haversine_regxytoploy(lon, lat, [5], projection_code=203, ij_to_latlon=lambda i, j: (j, i))


def test_projected_regular_grid_uses_fortran_half_index_vertices():
    lon, lat = _regular_grid()
    result = haversine_regxytoploy(
        lon, lat, [5], projection_code=1, ij_to_latlon=lambda i, j: (100.0 + j, 200.0 + i)
    )
    np.testing.assert_allclose(result.latpoly[0], [101.5, 101.5, 101.5, 102, 102.5, 102.5, 102.5, 102])
    np.testing.assert_allclose(result.lonpoly[0], [201.5, 202, 202.5, 202.5, 202.5, 202, 201.5, 201.5])


def test_single_point_source_guard_keeps_fortran_and_condition():
    result = haversine_singlepointploy(np.asarray([[10.0, 20.0]]), np.asarray([[45.0, 46.0]]))
    assert result.center.tolist() == [[10.0, 45.0]]
    with pytest.raises(HaversineGeometryError, match="iim=jjm=1"):
        haversine_singlepointploy(np.zeros((2, 2)), np.zeros((2, 2)))


def test_heading_sort_regular_segments_and_area_cover_orientation_arms():
    lon, lat = _regular_grid()
    polygon = haversine_reglatlontoploy(lon, lat, [5])
    headings = haversine_polyheadings(polygon.lonpoly, polygon.latpoly, polygon.center)
    order = haversine_clockwise(4, headings[0], 0.0)
    assert sorted(order.tolist()) == list(range(1, 9))
    slon, slat, _, _ = haversine_polysort(
        polygon.lonpoly, polygon.latpoly, headings, polygon.neighb_loc
    )
    lengths = haversine_laloseglen(slon, slat)
    expected_ns = 10.0 * np.pi / 180.0 * 6_378_000.0
    assert np.count_nonzero(np.isclose(lengths[0], expected_ns)) == 2
    area = haversine_laloarea(lengths)
    np.testing.assert_allclose(area, (lengths[:, 0] + lengths[:, 2]) / 2 * (lengths[:, 1] + lengths[:, 3]) / 2)
    with pytest.raises(HaversineGeometryError, match="regular"):
        haversine_laloseglen([[0, 1, 2, 3, 4, 5, 6, 7]], [[0, 1, 3, 4, 5, 6, 7, 8]])
    with pytest.raises(HaversineGeometryError, match="four"):
        haversine_laloarea(np.ones((1, 3)))
