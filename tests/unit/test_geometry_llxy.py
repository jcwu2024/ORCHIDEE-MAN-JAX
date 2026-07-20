from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from jax_orchidee.driver.geometry_llxy import (
    HH,
    PROJ_CASSINI,
    PROJ_CYL,
    PROJ_GAUSS,
    PROJ_LATLON,
    PROJ_MERC,
    PROJ_PS,
    PROJ_ROTLL,
    GeometryError,
    ProjectionInfo,
    ij_to_latlon,
    ijll_cyl,
    ijll_merc,
    latlon_to_ij,
    llij_cyl,
    llij_gauss,
    llij_merc,
    rotate_coords,
)


def test_init_and_unknown_projection_guards() -> None:
    with pytest.raises(GeometryError, match="not initialized"):
        latlon_to_ij(ProjectionInfo(PROJ_LATLON), 0.0, 0.0)
    with pytest.raises(GeometryError, match="unrecognized"):
        ij_to_latlon(ProjectionInfo(999, init=True), 1.0, 1.0)


def test_latlon_dispatch_is_one_based_and_broadcasts_dynamic_arrays() -> None:
    proj = ProjectionInfo(
        PROJ_LATLON,
        init=True,
        lat1=-10.0,
        lon1=100.0,
        latinc=0.5,
        loninc=2.0,
        knowni=1.0,
        knownj=1.0,
    )
    i, j = latlon_to_ij(proj, np.array([[-10.0], [-9.5]]), np.array([100.0, 102.0]))
    np.testing.assert_array_equal(i, [[1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_array_equal(j, [[1.0, 1.0], [2.0, 2.0]])
    lat, lon = ij_to_latlon(proj, i, j)
    np.testing.assert_allclose(lat, [[-10.0, -10.0], [-9.5, -9.5]])
    np.testing.assert_allclose(lon, [[100.0, 102.0], [100.0, 102.0]])


def test_longitude_wrap_preserves_fortran_boundaries() -> None:
    proj = ProjectionInfo(
        PROJ_CYL,
        init=True,
        lat1=-90.0,
        lon1=-180.0,
        latinc=1.0,
        loninc=1.0,
        knowni=1.0,
        knownj=1.0,
    )
    i, j = llij_cyl(np.array([-90.0, -90.0]), np.array([-180.0, 180.0]), proj)
    np.testing.assert_array_equal(i, [361.0, 361.0])
    lat, lon = ijll_cyl(i, j, proj)
    np.testing.assert_array_equal(lat, [-90.0, -90.0])
    np.testing.assert_array_equal(lon, [-180.0, -180.0])


def test_mercator_degrees_radians_and_wrap_round_trip() -> None:
    proj = ProjectionInfo(
        PROJ_MERC,
        init=True,
        lon1=170.0,
        knowni=3.0,
        knownj=4.0,
        dlon=0.02,
        rsw=-2.5,
    )
    lat = np.array([-30.0, 0.0, 45.0])
    lon = np.array([179.0, -179.0, 160.0])
    i, j = llij_merc(lat, lon, proj)
    out_lat, out_lon = ijll_merc(i, j, proj)
    np.testing.assert_allclose(out_lat, lat, atol=1e-12)
    np.testing.assert_allclose(out_lon, lon, atol=1e-12)


def test_spherical_ps_pole_guard_uses_projection_hemisphere() -> None:
    proj = ProjectionInfo(
        PROJ_PS,
        init=True,
        stdlon=10.0,
        truelat1=-60.0,
        hemi=-1.0,
        rebydx=100.0,
        polei=7.0,
        polej=8.0,
    )
    lat, lon = ij_to_latlon(proj, 7.0, 8.0)
    assert lat == -90.0
    assert lon == 100.0


def test_gaussian_interpolation_retains_fortran_one_based_j() -> None:
    proj = ProjectionInfo(
        PROJ_GAUSS,
        init=True,
        nlat=2,
        lon1=0.0,
        loninc=90.0,
        gauss_lat=np.array([80.0, 30.0, -30.0, -80.0]),
    )
    i, j = llij_gauss(np.array([90.0, 55.0, 0.0, -90.0]), 90.0, proj)
    np.testing.assert_array_equal(i, [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(j, [1.0, 1.5, 2.5, 4.0])
    with pytest.raises(GeometryError, match=r"2\*nlat"):
        llij_gauss(0.0, 0.0, ProjectionInfo(PROJ_GAUSS, nlat=2, gauss_lat=np.array([1.0])))


def test_rotation_and_rotated_grid_support_arrays() -> None:
    lat, lon = rotate_coords(np.array([0.0, 10.0]), np.array([0.0, 20.0]), 45.0, 10.0, 0.0)
    assert lat.shape == (2,)
    assert lon.shape == (2,)
    proj = ProjectionInfo(
        PROJ_ROTLL,
        init=True,
        ixdim=9,
        jydim=9,
        stagger=HH,
        phi=40.0,
        lambda_=80.0,
        lat1=45.0,
        lon1=0.0,
    )
    i, j = latlon_to_ij(proj, np.array([45.0, 50.0]), np.array([0.0, 5.0]))
    assert np.asarray(i).shape == (2,)
    assert np.asarray(j).shape == (2,)
    out_lat, out_lon = ij_to_latlon(proj, i, j)
    assert np.all(np.isfinite(out_lat))
    assert np.all(np.isfinite(out_lon))


GFORTRAN = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")
if not GFORTRAN.is_file():
    discovered_gfortran = shutil.which("gfortran")
    GFORTRAN = Path(discovered_gfortran) if discovered_gfortran else GFORTRAN


@pytest.mark.skipif(not GFORTRAN.is_file(), reason="gfortran is unavailable")
def test_source_extracted_gfortran_micro_oracle(tmp_path) -> None:
    # Formula statements are extracted from module_llxy.f90 lines 1355-1360,
    # 1379-1382, 1474-1489, 1630-1667, and 2163-2221.
    source = textwrap.dedent(
        """
        program llxy_oracle
          implicit none
          real(8), parameter :: pi=3.141592653589793d0, dpr=180d0/pi, rpd=pi/180d0
          real(8) :: lat,lon,i,j,dlon,rsw,knowni,knownj,lon1,deltalon
          real(8) :: ilat,ilon,olat,olon,lat_np,lon_np,lon_0,rlat,rlon
          real(8) :: phi_np,lam_np,lam_0,dlam,sinphi,cosphi,coslam,sinlam
          real(8) :: glat(4),diff1,diffn
          integer :: n,nlow
          logical :: found

          dlon=.02d0; rsw=-2.5d0; knowni=3d0; knownj=4d0; lon1=170d0
          lat=45d0; lon=-179d0
          deltalon=lon-lon1
          if (deltalon < -180d0) deltalon=deltalon+360d0
          if (deltalon > 180d0) deltalon=deltalon-360d0
          i=knowni+deltalon/(dlon*dpr)
          j=knownj+log(tan(.5d0*((lat+90d0)*rpd)))/dlon-rsw
          lat=2d0*atan(exp(dlon*(rsw+j-knownj)))*dpr-90d0
          lon=(i-knowni)*dlon*dpr+lon1
          if (lon > 180d0) lon=lon-360d0
          if (lon < -180d0) lon=lon+360d0
          write(*,'(4(ES25.16,1X))') i,j,lat,lon

          ilat=10d0; ilon=20d0; lat_np=45d0; lon_np=10d0; lon_0=0d0
          phi_np=lat_np*rpd; lam_np=lon_np*rpd; lam_0=lon_0*rpd
          rlat=ilat*rpd; rlon=ilon*rpd; dlam=lam_np
          sinphi=cos(phi_np)*cos(rlat)*cos(rlon-dlam)+sin(phi_np)*sin(rlat)
          cosphi=sqrt(1d0-sinphi*sinphi)
          coslam=sin(phi_np)*cos(rlat)*cos(rlon-dlam)-cos(phi_np)*sin(rlat)
          sinlam=cos(rlat)*sin(rlon-dlam)
          if (cosphi /= 0d0) then
            coslam=coslam/cosphi; sinlam=sinlam/cosphi
          end if
          olat=dpr*asin(sinphi)
          olon=dpr*(atan2(sinlam,coslam)-dlam-lam_0+lam_np)
          do; if (olon >= -180d0) exit; olon=olon+360d0; end do
          do; if (olon <= 180d0) exit; olon=olon-360d0; end do
          write(*,'(2(ES25.16,1X))') olat,olon

          glat=(/80d0,30d0,-30d0,-80d0/); lat=0d0; found=.false.
          do n=1,3
            if ((glat(n)-lat)*(glat(n+1)-lat) <= 0d0) then
              found=.true.; nlow=n; exit
            end if
          end do
          if (.not.found) error stop
          j=((glat(nlow)-lat)*(nlow+1)+(lat-glat(nlow+1))*nlow)/(glat(nlow)-glat(nlow+1))
          i=(90d0-0d0)/90d0+1d0
          write(*,'(2(ES25.16,1X))') i,j
        end program llxy_oracle
        """
    )
    source_path = tmp_path / "llxy_oracle.f90"
    executable = tmp_path / "llxy_oracle.exe"
    source_path.write_text(source, encoding="ascii")
    compile_env = os.environ.copy()
    compile_env["PATH"] = f"{GFORTRAN.parent}{os.pathsep}{compile_env['PATH']}"
    compile_result = None
    for _ in range(3):
        compile_result = subprocess.run(
            [str(GFORTRAN), str(source_path), "-o", str(executable)],
            capture_output=True,
            env=compile_env,
            text=True,
        )
        if compile_result.returncode == 0:
            break
        time.sleep(0.1)
    assert compile_result is not None
    compile_result.check_returncode()
    lines = subprocess.run(
        [str(executable)],
        check=True,
        capture_output=True,
        env=compile_env,
        text=True,
    ).stdout.splitlines()
    merc_oracle = np.fromstring(lines[0], sep=" ")
    rotation_oracle = np.fromstring(lines[1], sep=" ")
    gauss_oracle = np.fromstring(lines[2], sep=" ")

    merc = ProjectionInfo(PROJ_MERC, dlon=0.02, rsw=-2.5, knowni=3.0, knownj=4.0, lon1=170.0)
    i, j = llij_merc(45.0, -179.0, merc)
    lat, lon = ijll_merc(i, j, merc)
    np.testing.assert_allclose([i, j, lat, lon], merc_oracle, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(rotate_coords(10.0, 20.0, 45.0, 10.0, 0.0), rotation_oracle, rtol=2e-14, atol=2e-14)
    gauss = ProjectionInfo(PROJ_GAUSS, nlat=2, lon1=0.0, loninc=90.0, gauss_lat=np.array([80.0, 30.0, -30.0, -80.0]))
    np.testing.assert_allclose(llij_gauss(0.0, 90.0, gauss), gauss_oracle, rtol=0.0, atol=0.0)


def test_cassini_unrotated_dispatch_is_exact_cylindrical_path() -> None:
    proj = ProjectionInfo(
        PROJ_CASSINI,
        init=True,
        lat0=90.0,
        lat1=-90.0,
        lon1=-180.0,
        latinc=1.0,
        loninc=1.0,
        knowni=1.0,
        knownj=1.0,
    )
    assert latlon_to_ij(proj, -89.0, -179.0) == (2.0, 2.0)
    assert ij_to_latlon(proj, 2.0, 2.0) == (-89.0, -179.0)
