"""Exact numerical ports of the assigned ``module_llxy`` geometry routines.

Angles are degrees at the public boundary, as in the Fortran module. Grid
coordinates retain Fortran's one-based, real-valued index convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


PI_M = 3.141592653589793
DEG_PER_RAD = 180.0 / PI_M
RAD_PER_DEG = PI_M / 180.0
A_WGS84 = 6_378_137.0
E_WGS84 = 0.081819192
A_NAD83 = 6_378_137.0
E_NAD83 = 0.0818187034

PROJ_CASSINI = 0
PROJ_LC = 1
PROJ_PS = 2
PROJ_PS_WGS84 = 102
PROJ_MERC = 3
PROJ_GAUSS = 4
PROJ_CYL = 5
PROJ_LATLON = 6
PROJ_ALBERS_NAD83 = 105
PROJ_ROTLL = 203
HH = 4
VV = 5


class GeometryError(ValueError):
    """Equivalent of a fatal ``module_llxy`` input/projection error."""


@dataclass(frozen=True)
class ProjectionInfo:
    """Python state contract for the Fortran ``proj_info`` derived type."""

    code: int
    init: bool = False
    nlat: int = 0
    nlon: int = 0
    ixdim: int = 0
    jydim: int = 0
    stagger: int = 0
    phi: float = 0.0
    lambda_: float = 0.0
    lat1: float = 0.0
    lon1: float = 0.0
    lat0: float = 0.0
    lon0: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    latinc: float = 0.0
    loninc: float = 0.0
    dlon: float = 0.0
    stdlon: float = 0.0
    truelat1: float = 0.0
    truelat2: float = 0.0
    hemi: float = 1.0
    cone: float = 0.0
    polei: float = 0.0
    polej: float = 0.0
    rsw: float = 0.0
    rebydx: float = 0.0
    re_m: float = 0.0
    knowni: float = 0.0
    knownj: float = 0.0
    rho0: float = 0.0
    nc: float = 0.0
    bigc: float = 0.0
    wrap: bool = False
    gauss_lat: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))


def _arrays(a: object, b: object) -> tuple[np.ndarray, np.ndarray]:
    return np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def _result(a: np.ndarray, b: np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    if a.ndim == 0:
        return float(a), float(b)
    return a, b


def _wrap_180(value: np.ndarray) -> np.ndarray:
    value = np.where(value < -180.0, value + np.ceil((-180.0 - value) / 360.0) * 360.0, value)
    return np.where(value > 180.0, value - np.ceil((value - 180.0) / 360.0) * 360.0, value)


def _fmod_int(value: int, divisor: int) -> int:
    return value - int(value / divisor) * divisor


def _trunc_div(value: int, divisor: int) -> int:
    return int(value / divisor)


def _llij_latlon(lat: object, lon: object, proj: ProjectionInfo):
    lat, lon = _arrays(lat, lon)
    i = (lon - proj.lon1) / proj.loninc + proj.knowni
    j = (lat - proj.lat1) / proj.latinc + proj.knownj
    return _result(i, j)


def _ijll_latlon(i: object, j: object, proj: ProjectionInfo):
    i, j = _arrays(i, j)
    lat = proj.lat1 + (j - proj.knownj) * proj.latinc
    lon = proj.lon1 + (i - proj.knowni) * proj.loninc
    return _result(lat, lon)


def _llij_ps(lat: object, lon: object, proj: ProjectionInfo):
    lat, lon = _arrays(lat, lon)
    reflon = proj.stdlon + 90.0
    scale_top = 1.0 + proj.hemi * np.sin(proj.truelat1 * RAD_PER_DEG)
    ala = lat * RAD_PER_DEG
    rm = proj.rebydx * np.cos(ala) * scale_top / (1.0 + proj.hemi * np.sin(ala))
    alo = (lon - reflon) * RAD_PER_DEG
    return _result(proj.polei + rm * np.cos(alo), proj.polej + proj.hemi * rm * np.sin(alo))


def ijll_ps(i: object, j: object, proj: ProjectionInfo):
    """Port of ``module_llxy.f90:786-845``, spherical PS inverse."""
    i, j = _arrays(i, j)
    reflon = proj.stdlon + 90.0
    scale_top = 1.0 + proj.hemi * np.sin(proj.truelat1 * RAD_PER_DEG)
    xx = i - proj.polei
    yy = (j - proj.polej) * proj.hemi
    r2 = xx**2 + yy**2
    gi2 = (proj.rebydx * scale_top) ** 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        lat_regular = DEG_PER_RAD * proj.hemi * np.arcsin((gi2 - r2) / (gi2 + r2))
        arccos = np.arccos(xx / np.sqrt(r2))
    lon_regular = np.where(yy > 0.0, reflon + DEG_PER_RAD * arccos, reflon - DEG_PER_RAD * arccos)
    at_pole = r2 == 0.0
    lat = np.where(at_pole, proj.hemi * 90.0, lat_regular)
    lon = _wrap_180(np.where(at_pole, reflon, lon_regular))
    return _result(lat, lon)


def _ps_wgs84_terms(proj: ProjectionInfo) -> tuple[float, float]:
    sine = np.sin(proj.hemi * proj.truelat1 * RAD_PER_DEG)
    mc = np.cos(proj.hemi * proj.truelat1 * RAD_PER_DEG) / np.sqrt(1.0 - (E_WGS84 * sine) ** 2.0)
    tc = np.sqrt((1.0 - sine) / (1.0 + sine) * ((1.0 + E_WGS84 * sine) / (1.0 - E_WGS84 * sine)) ** E_WGS84)
    return float(mc), float(tc)


def _llij_ps_wgs84(lat: object, lon: object, proj: ProjectionInfo):
    lat, lon = _arrays(lat, lon)
    h = proj.hemi
    mc, tc = _ps_wgs84_terms(proj)
    sine = np.sin(h * lat * RAD_PER_DEG)
    t = np.sqrt((1.0 - sine) / (1.0 + sine) * ((1.0 + E_WGS84 * sine) / (1.0 - E_WGS84 * sine)) ** E_WGS84)
    rho = (A_WGS84 / proj.dx) * mc * t / tc
    angle = (h * lon - h * proj.stdlon) * RAD_PER_DEG
    x = h * rho * np.sin(angle)
    y = -h * rho * np.cos(angle)
    return _result(proj.knowni + x - proj.polei, proj.knownj + y - proj.polej)


def _ijll_ps_wgs84(i: object, j: object, proj: ProjectionInfo):
    i, j = _arrays(i, j)
    h = proj.hemi
    x = i - proj.knowni + proj.polei
    y = j - proj.knownj + proj.polej
    mc, tc = _ps_wgs84_terms(proj)
    rho = np.sqrt((x * proj.dx) ** 2.0 + (y * proj.dx) ** 2.0)
    t = rho * tc / (A_WGS84 * mc)
    lon = h * proj.stdlon * RAD_PER_DEG + h * np.arctan2(h * x, -h * y)
    chi = PI_M / 2.0 - 2.0 * np.arctan(t)
    a = E_WGS84**2 / 2 + 5 * E_WGS84**4 / 24 + E_WGS84**6 / 40 + 73 * E_WGS84**8 / 2016
    b = 7 * E_WGS84**4 / 24 + 29 * E_WGS84**6 / 120 + 54113 * E_WGS84**8 / 40320
    c = 7 * E_WGS84**6 / 30 + 81 * E_WGS84**8 / 280
    d = 4279 * E_WGS84**8 / 20160
    lat = h * (chi + np.sin(2 * chi) * (a + np.cos(2 * chi) * (b + np.cos(2 * chi) * (c + d * np.cos(2 * chi)))))
    return _result(lat * DEG_PER_RAD, lon * DEG_PER_RAD)


def _albers_q(sinphi: np.ndarray) -> np.ndarray:
    return (1.0 - E_NAD83**2) * (sinphi / (1.0 - (E_NAD83 * sinphi) ** 2) - np.log((1.0 - E_NAD83 * sinphi) / (1.0 + E_NAD83 * sinphi)) / (2.0 * E_NAD83))


def _llij_albers(lat: object, lon: object, proj: ProjectionInfo):
    lat, lon = _arrays(lat, lon)
    h = proj.hemi
    q = _albers_q(np.sin(h * lat * RAD_PER_DEG))
    rho = h * (A_NAD83 / proj.dx) * np.sqrt(proj.bigc - proj.nc * q) / proj.nc
    theta = proj.nc * (h * lon - h * proj.stdlon) * RAD_PER_DEG
    x = h * rho * np.sin(theta)
    y = h * proj.rho0 - h * rho * np.cos(theta)
    return _result(proj.knowni + x - proj.polei, proj.knownj + y - proj.polej)


def _ijll_albers(i: object, j: object, proj: ProjectionInfo):
    i, j = _arrays(i, j)
    x = i - proj.knowni + proj.polei
    y = j - proj.knownj + proj.polej
    rho = np.sqrt(x**2 + (proj.rho0 - y) ** 2)
    theta = np.arctan2(x, proj.rho0 - y)
    q = (proj.bigc - (rho * proj.nc * proj.dx / A_NAD83) ** 2) / proj.nc
    denom = 1.0 - np.log((1.0 - E_NAD83) / (1.0 + E_NAD83)) * (1.0 - E_NAD83**2) / (2.0 * E_NAD83)
    beta = np.arcsin(q / denom)
    a = E_NAD83**2 / 3 + 31 * E_NAD83**4 / 180 + 517 * E_NAD83**6 / 5040
    b = 23 * E_NAD83**4 / 360 + 251 * E_NAD83**6 / 3780
    c = 761 * E_NAD83**6 / 45360
    lat = proj.hemi * (beta + a * np.sin(2 * beta) + b * np.sin(4 * beta) + c * np.sin(6 * beta)) * DEG_PER_RAD
    lon = proj.stdlon + theta * DEG_PER_RAD / proj.nc
    return _result(lat, lon)


def ijll_lc(i: object, j: object, proj: ProjectionInfo):
    """Port of ``module_llxy.f90:1183-1256``, Lambert inverse."""
    i, j = _arrays(i, j)
    chi1 = (90.0 - proj.hemi * proj.truelat1) * RAD_PER_DEG
    chi2 = (90.0 - proj.hemi * proj.truelat2) * RAD_PER_DEG
    xx = proj.hemi * i - proj.polei
    yy = proj.polej - proj.hemi * j
    r2 = xx**2 + yy**2
    r = np.sqrt(r2) / proj.rebydx
    lon = proj.stdlon + DEG_PER_RAD * np.arctan2(proj.hemi * xx, yy) / proj.cone
    lon = np.fmod(lon + 360.0, 360.0)
    if chi1 == chi2:
        chi = 2.0 * np.arctan((r / np.tan(chi1)) ** (1.0 / proj.cone) * np.tan(chi1 * 0.5))
    else:
        chi = 2.0 * np.arctan((r * proj.cone / np.sin(chi1)) ** (1.0 / proj.cone) * np.tan(chi1 * 0.5))
    lat = (90.0 - chi * DEG_PER_RAD) * proj.hemi
    at_pole = r2 == 0.0
    return _result(np.where(at_pole, proj.hemi * 90.0, lat), _wrap_180(np.where(at_pole, proj.stdlon, lon)))


def llij_lc(lat: object, lon: object, proj: ProjectionInfo):
    """Port of ``module_llxy.f90:1259-1313``, Lambert forward."""
    lat, lon = _arrays(lat, lon)
    deltalon = np.where(lon - proj.stdlon > 180.0, lon - proj.stdlon - 360.0, lon - proj.stdlon)
    deltalon = np.where(deltalon < -180.0, deltalon + 360.0, deltalon)
    ctl1r = np.cos(proj.truelat1 * RAD_PER_DEG)
    rm = proj.rebydx * ctl1r / proj.cone * (np.tan((90 * proj.hemi - lat) * RAD_PER_DEG / 2) / np.tan((90 * proj.hemi - proj.truelat1) * RAD_PER_DEG / 2)) ** proj.cone
    arg = proj.cone * deltalon * RAD_PER_DEG
    i = proj.hemi * (proj.polei + proj.hemi * rm * np.sin(arg))
    j = proj.hemi * (proj.polej - rm * np.cos(arg))
    return _result(i, j)


def llij_merc(lat: object, lon: object, proj: ProjectionInfo):
    """Port of ``module_llxy.f90:1343-1364``, Mercator forward."""
    lat, lon = _arrays(lat, lon)
    delta = np.where(lon - proj.lon1 < -180.0, lon - proj.lon1 + 360.0, lon - proj.lon1)
    delta = np.where(delta > 180.0, delta - 360.0, delta)
    i = proj.knowni + delta / (proj.dlon * DEG_PER_RAD)
    j = proj.knownj + np.log(np.tan(0.5 * (lat + 90.0) * RAD_PER_DEG)) / proj.dlon - proj.rsw
    return _result(i, j)


def ijll_merc(i: object, j: object, proj: ProjectionInfo):
    """Port of ``module_llxy.f90:1367-1385``, Mercator inverse."""
    i, j = _arrays(i, j)
    lat = 2.0 * np.arctan(np.exp(proj.dlon * (proj.rsw + j - proj.knownj))) * DEG_PER_RAD - 90.0
    lon = _wrap_180((i - proj.knowni) * proj.dlon * DEG_PER_RAD + proj.lon1)
    return _result(lat, lon)


def llij_cyl(lat: object, lon: object, proj: ProjectionInfo):
    """Port of ``module_llxy.f90:1459-1491``, cyclic cylindrical forward."""
    lat, lon = _arrays(lat, lon)
    delta = np.where(lon - proj.lon1 < 0.0, lon - proj.lon1 + 360.0, lon - proj.lon1)
    delta = np.where(delta > 360.0, delta - 360.0, delta)
    i = delta / proj.loninc
    period = 360.0 / proj.loninc
    i = np.where(i <= 0.0, i + period, i)
    i = np.where(i > period, i - period, i)
    return _result(i + proj.knowni, (lat - proj.lat1) / proj.latinc + proj.knownj)


def ijll_cyl(i: object, j: object, proj: ProjectionInfo):
    """Port of ``module_llxy.f90:1494-1525``, cyclic cylindrical inverse."""
    i, j = _arrays(i, j)
    period = 360.0 / proj.loninc
    iw = np.where(i - proj.knowni < 0.0, i - proj.knowni + period, i - proj.knowni)
    iw = np.where(iw >= period, iw - period, iw)
    lat = (j - proj.knownj) * proj.latinc + proj.lat1
    lon = _wrap_180(iw * proj.loninc + proj.lon1)
    return _result(lat, lon)


def rotate_coords(ilat: object, ilon: object, lat_np: float, lon_np: float, lon_0: float, direction: int = 1):
    """Port of ``module_llxy.f90:1615-1669`` in degrees."""
    rlat, rlon = _arrays(ilat, ilon)
    phi_np = lat_np * RAD_PER_DEG
    lam_np = lon_np * RAD_PER_DEG
    lam_0 = lon_0 * RAD_PER_DEG
    rlat = rlat * RAD_PER_DEG
    rlon = rlon * RAD_PER_DEG
    dlam = PI_M - lam_0 if direction < 0 else lam_np
    sinphi = np.cos(phi_np) * np.cos(rlat) * np.cos(rlon - dlam) + np.sin(phi_np) * np.sin(rlat)
    cosphi = np.sqrt(1.0 - sinphi * sinphi)
    coslam = np.sin(phi_np) * np.cos(rlat) * np.cos(rlon - dlam) - np.cos(phi_np) * np.sin(rlat)
    sinlam = np.cos(rlat) * np.sin(rlon - dlam)
    coslam = np.where(cosphi != 0.0, coslam / cosphi, coslam)
    sinlam = np.where(cosphi != 0.0, sinlam / cosphi, sinlam)
    olat = DEG_PER_RAD * np.arcsin(sinphi)
    olon = DEG_PER_RAD * (np.arctan2(sinlam, coslam) - dlam - lam_0 + lam_np)
    return _result(olat, _wrap_180(olon))


def llij_cassini(lat: object, lon: object, proj: ProjectionInfo):
    """Port of ``module_llxy.f90:1558-1581``, Cassini forward."""
    comp_lat, comp_lon = (rotate_coords(lat, lon, proj.lat0, proj.lon0, proj.stdlon, -1) if abs(proj.lat0) != 90.0 else (lat, lon))
    return llij_cyl(comp_lat, comp_lon, proj)


def ijll_cassini(i: object, j: object, proj: ProjectionInfo):
    """Port of ``module_llxy.f90:1584-1607``, Cassini inverse."""
    comp_lat, comp_lon = ijll_cyl(i, j, proj)
    if abs(proj.lat0) != 90.0:
        return rotate_coords(comp_lat, comp_lon, proj.lat0, proj.lon0, proj.stdlon, 1)
    return comp_lat, comp_lon


def _rot_forward_scalar(lat: float, lon: float, proj: ProjectionInfo) -> tuple[float, float]:
    if proj.stagger not in {HH, VV}:
        raise GeometryError(f"rotated-grid stagger must be HH ({HH}) or VV ({VV})")
    dphd = proj.phi / int((proj.jydim - 1) / 2)
    dlmd = proj.lambda_ / (proj.ixdim - 1)
    d2r = PI_M / 180.0
    jmt = int(proj.jydim / 2) + 1
    glat = lat * d2r
    glon = -lon * d2r
    tph0 = proj.lat1 * d2r
    tlm0 = -proj.lon1 * d2r
    x = np.cos(tph0) * np.cos(glat) * np.cos(glon - tlm0) + np.sin(tph0) * np.sin(glat)
    y = -np.cos(glat) * np.sin(glon - tlm0)
    z = np.cos(tph0) * np.sin(glat) - np.sin(tph0) * np.cos(glat) * np.cos(glon - tlm0)
    tlat_deg = DEG_PER_RAD * np.arctan(z / np.sqrt(x * x + y * y))
    tlon_deg = DEG_PER_RAD * np.arctan(y / x)
    row = tlat_deg / dphd + jmt
    col = tlon_deg / dlmd + proj.ixdim
    if row - int(row) > 0.999:
        row += 0.0002
    elif col - int(col) > 0.999:
        col += 0.0002
    nrow, ncol = int(row), int(col)
    i_real = col / 2.0 + (0.5 if (proj.stagger == HH and _fmod_int(nrow, 2) != 0) or (proj.stagger == VV and _fmod_int(nrow, 2) == 0) else 0.0)
    j_real = row
    tlat = tlat_deg * d2r
    tlon = tlon_deg * d2r
    dph = dphd * d2r
    dlm = dlmd * d2r
    same_parity = abs(_fmod_int(nrow, 2)) == abs(_fmod_int(ncol, 2))
    if proj.stagger == VV:
        same_parity = not same_parity
    if same_parity:
        tlat1 = (nrow - jmt) * dph
        tlat2 = tlat1 + dph
    else:
        tlat1 = (nrow + 1 - jmt) * dph
        tlat2 = tlat1 - dph
    tlon1 = (ncol - proj.ixdim) * dlm
    tlon2 = tlon1 + dlm
    d1 = np.arccos(
        np.cos(tlat) * np.cos(tlat1) * np.cos(tlon - tlon1)
        + np.sin(tlat) * np.sin(tlat1)
    )
    d2 = np.arccos(
        np.cos(tlat) * np.cos(tlat2) * np.cos(tlon - tlon2)
        + np.sin(tlat) * np.sin(tlat2)
    )
    if same_parity:
        if d1 > d2:
            nrow += 1
            ncol += 1
    elif d1 < d2:
        nrow += 1
    else:
        ncol += 1
    if ncol <= 0:
        ncol -= 1
    ii = _trunc_div(ncol, 2)
    if (proj.stagger == HH and abs(_fmod_int(nrow, 2)) == 1) or (
        proj.stagger == VV and _fmod_int(nrow, 2) == 0
    ):
        ii += 1
    # Lines 1837-1838 assign this nearest point to local i/j, not to the
    # i_real/j_real OUT arguments. Evaluate it, then retain the OUTs.
    _nearest_local_ij = float(ii), float(nrow)
    return float(i_real), float(j_real)


def llij_rotlatlon(lat: object, lon: object, proj: ProjectionInfo):
    """Port of ``module_llxy.f90:1672-1840``, rotated E-grid forward."""
    lat_arr, lon_arr = _arrays(lat, lon)
    kernel = np.vectorize(lambda a, b: _rot_forward_scalar(float(a), float(b), proj), otypes=[float, float])
    i, j = kernel(lat_arr, lon_arr)
    return _result(i, j)


def _rot_inverse_scalar(i: float, j: float, proj: ProjectionInfo) -> tuple[float, float]:
    if proj.stagger not in {HH, VV}:
        raise GeometryError(f"rotated-grid stagger must be HH ({HH}) or VV ({VV})")
    j_work = j + 0.0002 if j - int(j) > 0.999 else j
    jh = int(j_work)
    dphd = proj.phi / int((proj.jydim - 1) / 2)
    dlmd = proj.lambda_ / (proj.ixdim - 1)
    tph0 = proj.lat1 * RAD_PER_DEG
    tlm0 = -proj.lon1 * RAD_PER_DEG
    midrow = int((proj.jydim + 1) / 2)
    col = 2.0 * i - 1.0 + abs(_fmod_int(jh + 1, 2))
    tlatd = (j_work - midrow) * dphd
    tlond = (col - proj.ixdim) * dlmd
    if proj.stagger == VV:
        tlond += -dlmd if _fmod_int(jh, 2) == 0 else dlmd
    tlatr, tlonr = tlatd * RAD_PER_DEG, tlond * RAD_PER_DEG
    arg1 = np.sin(tlatr) * np.cos(tph0) + np.cos(tlatr) * np.sin(tph0) * np.cos(tlonr)
    glatr = np.arcsin(arg1)
    arg2 = np.cos(tlatr) * np.cos(tlonr) / (np.cos(glatr) * np.cos(tph0)) - np.tan(glatr) * np.tan(tph0)
    if abs(arg2) > 1.0:
        arg2 = abs(arg2) / arg2
    fctr = -1.0 if tlond > 0.0 else 1.0
    glond = tlm0 * DEG_PER_RAD + fctr * np.arccos(arg2) * DEG_PER_RAD
    lon = float(_wrap_180(np.asarray(glond * -1.0)))
    return float(glatr * DEG_PER_RAD), lon


def ijll_rotlatlon(i: object, j: object, proj: ProjectionInfo):
    """Port of ``module_llxy.f90:1843-1914``, rotated E-grid inverse."""
    i_arr, j_arr = _arrays(i, j)
    kernel = np.vectorize(lambda a, b: _rot_inverse_scalar(float(a), float(b), proj), otypes=[float, float])
    lat, lon = kernel(i_arr, j_arr)
    return _result(lat, lon)


def llij_gauss(lat: object, lon: object, proj: ProjectionInfo):
    """Port of ``module_llxy.f90:2146-2225`` with dynamic Gaussian arrays."""
    lat_arr, lon_arr = _arrays(lat, lon)
    gauss = np.asarray(proj.gauss_lat, dtype=float)
    count = proj.nlat * 2
    if gauss.ndim != 1 or gauss.size < count or count < 2:
        raise GeometryError("gauss_lat must contain at least 2*nlat values")
    i = (lon_arr - proj.lon1) / proj.loninc + 1.0

    def latitude_index(value: float) -> float:
        if abs(value) > abs(gauss[0]):
            return 1.0 if abs(value - gauss[0]) < abs(value - gauss[count - 1]) else float(count)
        products = (gauss[: count - 1] - value) * (gauss[1:count] - value)
        hits = np.flatnonzero(products <= 0.0)
        if hits.size == 0:
            raise GeometryError("no bounding Gaussian latitudes found")
        low = int(hits[0])
        n_low = low + 1.0
        return ((gauss[low] - value) * (n_low + 1.0) + (value - gauss[low + 1]) * n_low) / (gauss[low] - gauss[low + 1])

    j = np.vectorize(latitude_index, otypes=[float])(lat_arr)
    return _result(i, j)


def latlon_to_ij(proj: ProjectionInfo, lat: object, lon: object):
    """Port of dispatch and init guard in ``module_llxy.f90:593-649``."""
    if not proj.init:
        raise GeometryError("map projection is not initialized")
    functions = {
        PROJ_LATLON: _llij_latlon,
        PROJ_MERC: llij_merc,
        PROJ_PS: _llij_ps,
        PROJ_PS_WGS84: _llij_ps_wgs84,
        PROJ_ALBERS_NAD83: _llij_albers,
        PROJ_LC: llij_lc,
        PROJ_GAUSS: llij_gauss,
        PROJ_CYL: llij_cyl,
        PROJ_CASSINI: llij_cassini,
        PROJ_ROTLL: llij_rotlatlon,
    }
    try:
        function = functions[proj.code]
    except KeyError as exc:
        raise GeometryError(f"unrecognized map projection code: {proj.code}") from exc
    return function(lat, lon, proj)


def ij_to_latlon(proj: ProjectionInfo, i: object, j: object):
    """Port of dispatch and init guard in ``module_llxy.f90:652-702``."""
    if not proj.init:
        raise GeometryError("map projection is not initialized")
    functions = {
        PROJ_LATLON: _ijll_latlon,
        PROJ_MERC: ijll_merc,
        PROJ_PS: ijll_ps,
        PROJ_PS_WGS84: _ijll_ps_wgs84,
        PROJ_ALBERS_NAD83: _ijll_albers,
        PROJ_LC: ijll_lc,
        PROJ_CYL: ijll_cyl,
        PROJ_CASSINI: ijll_cassini,
        PROJ_ROTLL: ijll_rotlatlon,
    }
    try:
        function = functions[proj.code]
    except KeyError as exc:
        raise GeometryError(f"unrecognized map projection code: {proj.code}") from exc
    return function(i, j, proj)
