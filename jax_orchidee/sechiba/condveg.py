"""Source-backed CONDVEG kernels after ENERBIL.

This module covers only closed algebra from ``condveg_main`` and helper code
whose Fortran inputs are explicit. Inputs that are module state in Fortran,
such as soil/leaf albedo arrays and PFT roughness parameters, remain explicit
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple

from jax import config, jit

config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


MIN_SECHIBA = 1.0e-8
SNOWCRI_ALB = 10.0
SN_DENS = 330.0
CT_KARMAN = 0.41
PB_STD = 1013.0
ZERO_CELSIUS = 273.15
MIN_WIND = 0.1
HEIGHT_DISPLACEMENT = 0.66
Z0_BARE = 0.01
Z0_ICE = 0.001
CDRAG_FOLIAGE = 0.2
PRANDTL = 0.71
CT_LEAF = 0.01
CANOPY_C1 = 0.32
CANOPY_C2 = 0.264
CANOPY_C3 = 15.1

CONDVEG_MAIN_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_main lines 332-480",
)
CONDVEG_FRAC_SNOW_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_frac_snow lines 858-898",
)
CONDVEG_Z0CDRAG_DYN_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_z0cdrag_dyn lines 1519-1720",
)
CONDVEG_Z0CDRAG_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_z0cdrag lines 1288-1474",
)
CONDVEG_CONSTANT_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_parameters/constantes_var.f90 lines 371-392,460,688-723",
    "fortran_source/ORCHIDEE/src_parameters/constantes_soil_var.f90 lines 97-99",
)
CONDVEG_ALBEDO_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_albedo lines 612-839",
)
CONDVEG_SOILALB_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_soilalb lines 932-1131",
    "fortran_source/ORCHIDEE/src_parameters/constantes_var.f90 lines 736-761",
)
CONDVEG_INITIALIZE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_initialize lines 102-310",
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_frac_snow lines 858-898",
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_z0cdrag_dyn lines 1519-1720",
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_albedo lines 612-839",
)


class CondvegInitializeBoundaryError(NotImplementedError):
    """A reachable initializer branch requires target-external structure."""


@dataclass(frozen=True)
class CondvegInitializeResult:
    """State written by the audited PFT14 ``condveg_initialize`` path."""

    soilalb_bg: jnp.ndarray
    emis: jnp.ndarray
    albedo: jnp.ndarray
    z0m: jnp.ndarray
    z0h: jnp.ndarray
    roughheight: jnp.ndarray
    roughheight_pft: jnp.ndarray
    frac_snow_veg: jnp.ndarray
    frac_snow_nobio: jnp.ndarray
    l_first: bool
    recomputed_roughness: bool
    provenance: tuple[str, ...] = CONDVEG_INITIALIZE_PROVENANCE

    def as_payload(self) -> dict[str, object]:
        return {
            "soilalb_bg": self.soilalb_bg,
            "emis": self.emis,
            "albedo": self.albedo,
            "z0m": self.z0m,
            "z0h": self.z0h,
            "roughheight": self.roughheight,
            "roughheight_pft": self.roughheight_pft,
            "frac_snow_veg": self.frac_snow_veg,
            "frac_snow_nobio": self.frac_snow_nobio,
        }


class CondvegSnowFractions(NamedTuple):
    """Snow cover fractions on vegetated and non-vegetated surfaces."""

    frac_snow_veg: jnp.ndarray
    frac_snow_nobio: jnp.ndarray


class CondvegRoughness(NamedTuple):
    """CONDVEG roughness outputs."""

    z0m: jnp.ndarray
    z0h: jnp.ndarray
    roughheight: jnp.ndarray
    roughheight_pft: jnp.ndarray


class CondvegMainMinimalResult(NamedTuple):
    """Source-equivalent subset of ``condveg_main`` outputs."""

    frac_snow_veg: jnp.ndarray
    frac_snow_nobio: jnp.ndarray
    emis: jnp.ndarray
    z0m: jnp.ndarray
    z0h: jnp.ndarray
    roughheight: jnp.ndarray
    roughheight_pft: jnp.ndarray
    albedo: jnp.ndarray | None = None
    albedo_snow: jnp.ndarray | None = None
    alb_bare: jnp.ndarray | None = None
    alb_veget: jnp.ndarray | None = None
    albedo_snow_mean: jnp.ndarray | None = None


class CondvegAlbedoResult(NamedTuple):
    """CONDVEG two-band albedo outputs and diagnostics."""

    albedo: jnp.ndarray
    albedo_snow: jnp.ndarray
    alb_bare: jnp.ndarray
    alb_veget: jnp.ndarray
    snowa_veg: jnp.ndarray
    snowa_nobio: jnp.ndarray
    albedo_snow_mean: jnp.ndarray


class CondvegSoilAlbedoResult(NamedTuple):
    """Bare-soil bands derived from interpolated soil-colour fractions."""

    soilalb_dry: jnp.ndarray
    soilalb_wet: jnp.ndarray
    soilalb_moy: jnp.ndarray
    asoilcol: jnp.ndarray
    fallback_count: jnp.ndarray


class CondvegMainOutputPacket(NamedTuple):
    """Scientific diagnostic fields before XIOS/IOIPSL serialization."""

    xios: tuple[tuple[str, jnp.ndarray], ...]
    history_primary: tuple[tuple[str, jnp.ndarray], ...]
    history_secondary: tuple[tuple[str, jnp.ndarray], ...]


@dataclass(frozen=True)
class CondvegCoverage:
    """Audit status for fields intentionally covered or left unported."""

    ok: bool
    covered_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    branch: str
    provenance: tuple[str, ...]
    notes: tuple[str, ...] = ()


class CondvegFirstStepModuleClosure(NamedTuple):
    """First-step CONDVEG execution and downstream snow-fraction boundary."""

    module: CondvegMainMinimalResult | None
    inputs: Mapping[str, object]
    missing_inputs: tuple[str, ...]
    covered_fields: tuple[str, ...]
    provenance: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when CONDVEG supplied the THERMOSOIL-required snow fractions."""

        return (
            self.module is not None
            and "frac_snow_veg" in self.covered_fields
            and "frac_snow_nobio" in self.covered_fields
        )


def _as_float64(value):
    return jnp.asarray(value, dtype=jnp.float64)


def condveg_main_emissivity(kjpindex: int, *, emis_scal):
    """Fill ``emis(:)=emis_scal`` from ``condveg_main``.

    Fortran provenance: ``condveg.f90::condveg_main`` lines 402-403. The
    caller must pass the active module value ``emis_scal``; this helper does
    not assume ``IMPOSE_AZE`` initialization state.
    """

    return jnp.full((int(kjpindex),), jnp.asarray(emis_scal, dtype=jnp.float64))


def condveg_frac_snow(
    *,
    snow,
    snow_nobio,
    snowrho,
    snowdz,
    ok_explicitsnow,
    min_sechiba=MIN_SECHIBA,
    snowcri_alb=SNOWCRI_ALB,
    sn_dens=SN_DENS,
) -> CondvegSnowFractions:
    """Compute CONDVEG snow-cover fractions.

    Fortran provenance: ``condveg.f90::condveg_frac_snow`` lines 878-894.
    The explicit-snow branch uses ``snowdz``/``snowrho``; the default branch
    uses snow mass and the explicit ``snowcri_alb`` and ``sn_dens`` constants.
    """

    snow = _as_float64(snow)
    snow_nobio = _as_float64(snow_nobio)

    if ok_explicitsnow:
        if snowrho is None or snowdz is None:
            raise ValueError(
                "snowrho and snowdz are required when ok_explicitsnow is true"
            )
        snowrho = _as_float64(snowrho)
        snowdz = _as_float64(snowdz)
        snowdepth = jnp.sum(snowdz, axis=1)
        snowrho_snowdz = jnp.sum(snowrho * snowdz, axis=1)
        safe_depth = jnp.where(snowdepth < min_sechiba, 1.0, snowdepth)
        snowrho_ave = snowrho_snowdz / safe_depth
        explicit_frac = jnp.tanh(snowdepth / (0.025 * (snowrho_ave / 50.0)))
        frac_snow_veg = jnp.where(snowdepth < min_sechiba, 0.0, explicit_frac)
    else:
        positive_snow = jnp.maximum(snow, 0.0)
        frac_snow_veg = jnp.minimum(
            positive_snow / (positive_snow + snowcri_alb * sn_dens / 100.0),
            1.0,
        )

    positive_nobio_snow = jnp.maximum(snow_nobio, 0.0)
    frac_snow_nobio = jnp.minimum(
        positive_nobio_snow / (positive_nobio_snow + snowcri_alb),
        1.0,
    )
    return CondvegSnowFractions(frac_snow_veg, frac_snow_nobio)


def condveg_prescribed_roughness(
    *,
    kjpindex: int,
    nvm: int,
    z0_scal,
    roughheight_scal,
) -> CondvegRoughness:
    """Return the ``IMPOSE_AZE=true`` prescribed roughness branch.

    Fortran provenance: ``condveg.f90::condveg_main`` lines 407-415.
    """

    z0_scal = jnp.asarray(z0_scal, dtype=jnp.float64)
    roughheight_scal = jnp.asarray(roughheight_scal, dtype=jnp.float64)
    return CondvegRoughness(
        z0m=jnp.full((int(kjpindex),), z0_scal),
        z0h=jnp.full((int(kjpindex),), z0_scal),
        roughheight=jnp.full((int(kjpindex),), roughheight_scal),
        roughheight_pft=jnp.full((int(kjpindex), int(nvm)), roughheight_scal),
    )


def condveg_z0cdrag_dyn(
    *,
    veget,
    veget_max,
    frac_nobio,
    totfrac_nobio,
    zlev,
    height,
    temp_air,
    pb,
    u,
    v,
    lai,
    frac_snow_veg,
    min_sechiba=MIN_SECHIBA,
    ct_karman=CT_KARMAN,
    z0_bare=Z0_BARE,
    z0_ice=Z0_ICE,
    min_wind=MIN_WIND,
    pb_std=PB_STD,
    zero_celsius=ZERO_CELSIUS,
    height_displacement=HEIGHT_DISPLACEMENT,
    cdrag_foliage=CDRAG_FOLIAGE,
    prandtl=PRANDTL,
    ct_leaf=CT_LEAF,
    c1=CANOPY_C1,
    c2=CANOPY_C2,
    c3=CANOPY_C3,
) -> CondvegRoughness:
    """Dynamic roughness branch used when ``IMPOSE_AZE=false`` and ``ROUGH_DYN=true``.

    Fortran provenance: ``condveg.f90::condveg_z0cdrag_dyn`` lines 1578-1718.
    This port supports the source configuration ``nnobio=1`` / ``iice=1``.
    The Fortran raises a fatal error for any other non-vegetated surface type.
    """

    veget = _as_float64(veget)
    veget_max = _as_float64(veget_max)
    frac_nobio = _as_float64(frac_nobio)
    totfrac_nobio = _as_float64(totfrac_nobio)
    zlev = _as_float64(zlev)
    height = _as_float64(height)
    temp_air = _as_float64(temp_air)
    pb = _as_float64(pb)
    u = _as_float64(u)
    v = _as_float64(v)
    lai = _as_float64(lai)
    frac_snow_veg = _as_float64(frac_snow_veg)

    if frac_nobio.ndim != 2 or frac_nobio.shape[1] != 1:
        raise ValueError(
            "condveg_z0cdrag_dyn is source-backed only for nnobio=1/iice=1"
        )

    ztmp = jnp.maximum(10.0, zlev)
    z0_ground = (1.0 - frac_snow_veg) * z0_bare + frac_snow_veg * z0_bare / 10.0
    z0m = veget_max[:, 0] * (ct_karman / jnp.log(ztmp / z0_ground)) ** 2

    wind = jnp.sqrt(u * u + v * v)
    u_star = ct_karman * jnp.maximum(min_wind, wind) / jnp.log(zlev / z0_ground)
    reynolds = (
        z0_ground
        * u_star
        / (1.327e-5 * (pb_std / pb) * (temp_air / zero_celsius) ** 1.81)
    )
    kbs_m1 = 2.46 * reynolds ** (1.0 / 4.0) - jnp.log(7.4)
    z0h = (
        veget_max[:, 0]
        * (ct_karman / jnp.log(ztmp / (z0_ground / jnp.exp(kbs_m1)))) ** 2
    )

    sumveg = veget_max[:, 0]
    ave_height = jnp.zeros_like(zlev)
    nvm = veget_max.shape[1]
    roughheight_pft = jnp.full_like(height, jnp.nan)

    for jv in range(1, nvm):
        active = veget_max[:, jv] > 0.0
        eta = c1 - c2 * jnp.exp(-c3 * cdrag_foliage * lai[:, jv])
        z0m_pft = (
            height[:, jv]
            * (1.0 - height_displacement)
            * (jnp.exp(-ct_karman / eta) - jnp.exp(-ct_karman / (c1 - c2)))
        ) + z0_ground

        z0m_add = veget_max[:, jv] * (ct_karman / jnp.log(ztmp / z0m_pft)) ** 2
        z0m = jnp.where(active, z0m + z0m_add, z0m)

        safe_veget_max = jnp.where(active, veget_max[:, jv], 1.0)
        fc = veget[:, jv] / safe_veget_max
        fs = 1.0 - fc
        eta_ec = (cdrag_foliage * lai[:, jv]) / (2.0 * eta * eta)
        u_star = (
            ct_karman
            * jnp.maximum(min_wind, wind)
            / jnp.log((zlev + height[:, jv] * (1.0 - height_displacement)) / z0m_pft)
        )
        reynolds = (
            z0_ground
            * u_star
            / (1.327e-5 * (pb_std / pb) * (temp_air / zero_celsius) ** 1.81)
        )
        kbs_m1 = 2.46 * reynolds ** (1.0 / 4.0) - jnp.log(7.4)
        ct_star = prandtl ** (-2.0 / 3.0) * jnp.sqrt(1.0 / reynolds)
        canopy_kb = (
            (ct_karman * cdrag_foliage)
            / (4.0 * ct_leaf * eta * (1.0 - jnp.exp(-eta_ec / 2.0)))
            * fc**2.0
            + 2.0 * fc * fs * (ct_karman * eta * z0m_pft / height[:, jv]) / ct_star
            + kbs_m1 * fs**2.0
        )
        soil_kb = kbs_m1 * fs**2.0
        kb_m1 = jnp.where(lai[:, jv] > min_sechiba, canopy_kb, soil_kb)
        z0h_pft = z0m_pft / jnp.exp(kb_m1)
        z0h_add = veget_max[:, jv] * (ct_karman / jnp.log(ztmp / z0h_pft)) ** 2
        z0h = jnp.where(active, z0h + z0h_add, z0h)
        sumveg = jnp.where(active, sumveg + veget_max[:, jv], sumveg)
        ave_height = jnp.where(
            active, ave_height + veget_max[:, jv] * height[:, jv], ave_height
        )
        roughheight_pft = roughheight_pft.at[:, jv].set(
            height[:, jv] * (1.0 - height_displacement)
        )

    has_veg = sumveg > 0.0
    z0h = jnp.where(has_veg, z0h / sumveg, z0h)
    z0m = jnp.where(has_veg, z0m / sumveg, z0m)
    z0h = (1.0 - totfrac_nobio) * z0h
    z0m = (1.0 - totfrac_nobio) * z0m

    z0m = z0m + frac_nobio[:, 0] * (ct_karman / jnp.log(ztmp / z0_ice)) ** 2
    u_star = ct_karman * jnp.maximum(min_wind, wind) / jnp.log(zlev / z0_ice)
    reynolds = (
        z0_ice * u_star / (1.327e-5 * (pb_std / pb) * (temp_air / zero_celsius) ** 1.81)
    )
    kbs_m1 = 2.46 * reynolds ** (1.0 / 4.0) - jnp.log(7.4)
    z0h = (
        z0h
        + frac_nobio[:, 0]
        * (ct_karman / jnp.log(ztmp / (z0_ice / jnp.exp(kbs_m1)))) ** 2
    )

    z0h = ztmp / jnp.exp(ct_karman / jnp.sqrt(z0h))
    z0m = ztmp / jnp.exp(ct_karman / jnp.sqrt(z0m))
    roughheight = ave_height * (1.0 - height_displacement)
    return CondvegRoughness(z0m, z0h, roughheight, roughheight_pft)


def condveg_z0cdrag(
    *,
    veget,
    veget_max,
    frac_nobio,
    totfrac_nobio,
    zlev,
    height,
    tot_bare_soil,
    is_tree,
    z0_over_height,
    ratio_z0m_z0h,
    ct_karman=CT_KARMAN,
    z0_bare=Z0_BARE,
    z0_ice=Z0_ICE,
    height_displacement=HEIGHT_DISPLACEMENT,
) -> CondvegRoughness:
    """Static roughness branch used when ``IMPOSE_AZE=false`` and ``ROUGH_DYN=false``.

    Fortran provenance: ``condveg.f90::condveg_z0cdrag`` lines 1325-1474.
    This port supports the source configuration ``nnobio=1`` / ``iice=1`` and
    keeps PFT parameters (`is_tree`, `z0_over_height`, `ratio_z0m_z0h`)
    explicit because they are module state in Fortran.
    """

    veget = _as_float64(veget)
    veget_max = _as_float64(veget_max)
    frac_nobio = _as_float64(frac_nobio)
    totfrac_nobio = _as_float64(totfrac_nobio)
    zlev = _as_float64(zlev)
    height = _as_float64(height)
    tot_bare_soil = _as_float64(tot_bare_soil)
    is_tree = jnp.asarray(is_tree, dtype=bool)
    z0_over_height = _as_float64(z0_over_height)
    ratio_z0m_z0h = _as_float64(ratio_z0m_z0h)

    if veget.ndim != 2:
        raise ValueError("veget must have shape (npts, nvm)")
    npts, nvm = veget.shape
    if veget_max.shape != (npts, nvm) or height.shape != (npts, nvm):
        raise ValueError("veget_max and height must share shape (npts, nvm)")
    if frac_nobio.ndim != 2 or frac_nobio.shape != (npts, 1):
        raise ValueError("condveg_z0cdrag is source-backed only for nnobio=1/iice=1")
    for name, arr in (
        ("totfrac_nobio", totfrac_nobio),
        ("zlev", zlev),
        ("tot_bare_soil", tot_bare_soil),
    ):
        if arr.shape != (npts,):
            raise ValueError(f"{name} must have shape (npts,)")
    for name, arr in (
        ("is_tree", is_tree),
        ("z0_over_height", z0_over_height),
        ("ratio_z0m_z0h", ratio_z0m_z0h),
    ):
        if arr.shape != (nvm,):
            raise ValueError(f"{name} must have shape (nvm,)")

    ztmp = jnp.maximum(10.0, zlev)
    z0m = tot_bare_soil * (ct_karman / jnp.log(ztmp / z0_bare)) ** 2
    z0h = (
        tot_bare_soil * (ct_karman / jnp.log(ztmp / (z0_bare / ratio_z0m_z0h[0]))) ** 2
    )
    sumveg = tot_bare_soil
    ave_height = jnp.zeros_like(zlev)
    roughheight_pft = jnp.full_like(height, jnp.nan)

    for jv in range(1, nvm):
        d_veg = jnp.where(is_tree[jv], veget_max[:, jv], veget[:, jv])
        z0_pft = jnp.maximum(height[:, jv] * z0_over_height[jv], z0_bare)
        z0m = z0m + d_veg * (ct_karman / jnp.log(ztmp / z0_pft)) ** 2
        z0h = (
            z0h
            + d_veg * (ct_karman / jnp.log(ztmp / (z0_pft / ratio_z0m_z0h[jv]))) ** 2
        )
        sumveg = sumveg + d_veg
        ave_height = ave_height + veget_max[:, jv] * height[:, jv]

    has_veg = sumveg > 0.0
    z0m = jnp.where(has_veg, z0m / sumveg, z0m)
    z0h = jnp.where(has_veg, z0h / sumveg, z0h)
    z0m = (1.0 - totfrac_nobio) * z0m
    z0h = (1.0 - totfrac_nobio) * z0h

    z0m = z0m + frac_nobio[:, 0] * (ct_karman / jnp.log(ztmp / z0_ice)) ** 2
    z0h = (
        z0h
        + frac_nobio[:, 0]
        * (ct_karman / jnp.log(ztmp / z0_ice / ratio_z0m_z0h[0])) ** 2
    )
    z0m = ztmp / jnp.exp(ct_karman / jnp.sqrt(z0m))
    z0h = ztmp / jnp.exp(ct_karman / jnp.sqrt(z0h))

    roughheight = ave_height * (1.0 - height_displacement)
    for jv in range(1, nvm):
        roughheight_pft = roughheight_pft.at[:, jv].set(
            height[:, jv] * (1.0 - height_displacement)
        )
    return CondvegRoughness(z0m, z0h, roughheight, roughheight_pft)


def condveg_albedo_snow_mean(*, snow, albedo_snow):
    """Mean snow albedo diagnostic sent by ``condveg_main``.

    Fortran provenance: ``condveg.f90::condveg_main`` lines 447-455.
    """

    snow = _as_float64(snow)
    albedo_snow = _as_float64(albedo_snow)
    return jnp.where(snow > 0.0, (albedo_snow[:, 0] + albedo_snow[:, 1]) / 2.0, 0.0)


def condveg_soilalb_from_interpolation(
    *,
    soilcolrefrac,
    asoilcol,
    vis_dry=(0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.27),
    nir_dry=(0.48, 0.44, 0.40, 0.36, 0.32, 0.28, 0.24, 0.20, 0.55),
    vis_wet=(0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.15),
    nir_wet=(0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.31),
    albsoil_vis=(0.18, 0.16, 0.16, 0.15, 0.12, 0.105, 0.09, 0.075, 0.25),
    albsoil_nir=(0.36, 0.34, 0.34, 0.33, 0.30, 0.25, 0.20, 0.15, 0.45),
    min_sechiba=MIN_SECHIBA,
) -> CondvegSoilAlbedoResult:
    """Convert interpolated soil-colour fractions to two-band albedos.

    This owns the pure numerical part of ``condveg_soilalb`` lines 1045-1120.
    ``soilcolrefrac`` and ``asoilcol`` are the outputs of the source
    ``interpweight_2D`` call at lines 1039-1042. Reading ``SOILALB_FILE`` and
    performing that interpolation remain an explicit IO/interpolation
    boundary.
    """

    fractions = _as_float64(soilcolrefrac)
    availability = _as_float64(asoilcol)
    parameter_arrays = tuple(
        _as_float64(value)
        for value in (vis_dry, nir_dry, vis_wet, nir_wet, albsoil_vis, albsoil_nir)
    )
    if fractions.ndim != 2:
        raise ValueError("soilcolrefrac must have shape (npts, classnb)")
    npts, classnb = fractions.shape
    if availability.shape != (npts,):
        raise ValueError("asoilcol must have shape (npts,)")
    if classnb == 0 or any(value.shape != (classnb,) for value in parameter_arrays):
        raise ValueError("all soil-albedo parameter arrays must have shape (classnb,)")

    (
        vis_dry_arr,
        nir_dry_arr,
        vis_wet_arr,
        nir_wet_arr,
        albsoil_vis_arr,
        albsoil_nir_arr,
    ) = parameter_arrays
    nonzero_count = jnp.count_nonzero(fractions != 0.0, axis=1)
    fallback = (nonzero_count == 0) | (availability < float(min_sechiba))

    # Lines 1090-1107 accumulate in soil-class order. Values below the source
    # threshold are set to zero only on the interpolated-data branch.
    usable = jnp.where(fractions < float(min_sechiba), 0.0, fractions)
    soilalb_dry = jnp.zeros((npts, 2), dtype=jnp.float64)
    soilalb_wet = jnp.zeros((npts, 2), dtype=jnp.float64)
    soilalb_moy = jnp.zeros((npts, 2), dtype=jnp.float64)
    for soil_class in range(classnb):
        weight = usable[:, soil_class]
        soilalb_dry = soilalb_dry.at[:, 0].add(vis_dry_arr[soil_class] * weight)
        soilalb_dry = soilalb_dry.at[:, 1].add(nir_dry_arr[soil_class] * weight)
        soilalb_wet = soilalb_wet.at[:, 0].add(vis_wet_arr[soil_class] * weight)
        soilalb_wet = soilalb_wet.at[:, 1].add(nir_wet_arr[soil_class] * weight)
        soilalb_moy = soilalb_moy.at[:, 0].add(albsoil_vis_arr[soil_class] * weight)
        soilalb_moy = soilalb_moy.at[:, 1].add(albsoil_nir_arr[soil_class] * weight)

    fallback_dry = jnp.asarray(
        [
            (jnp.sum(vis_dry_arr) / classnb + jnp.sum(vis_wet_arr) / classnb) / 2.0,
            (jnp.sum(nir_dry_arr) / classnb + jnp.sum(nir_wet_arr) / classnb) / 2.0,
        ],
        dtype=jnp.float64,
    )
    fallback_moy = jnp.asarray(
        [jnp.sum(albsoil_vis_arr) / classnb, jnp.sum(albsoil_nir_arr) / classnb],
        dtype=jnp.float64,
    )
    soilalb_dry = jnp.where(fallback[:, None], fallback_dry[None, :], soilalb_dry)
    soilalb_wet = jnp.where(fallback[:, None], fallback_dry[None, :], soilalb_wet)
    soilalb_moy = jnp.where(fallback[:, None], fallback_moy[None, :], soilalb_moy)
    return CondvegSoilAlbedoResult(
        soilalb_dry=soilalb_dry,
        soilalb_wet=soilalb_wet,
        soilalb_moy=soilalb_moy,
        asoilcol=availability,
        fallback_count=jnp.sum(fallback),
    )


def condveg_main_output_packet(
    *,
    alb_bare,
    alb_veget,
    albedo,
    albedo_snow,
    snow,
    almaoutput=False,
    hist2_id=-1,
) -> CondvegMainOutputPacket:
    """Build ``condveg_main`` diagnostics without performing file IO.

    Fortran provenance: ``condveg.f90::condveg_main`` lines 439-476. Field
    order, ALMA selection and the secondary-history gate match the source.
    """

    alb_bare = _as_two_band("alb_bare", alb_bare)
    npts = alb_bare.shape[0]
    alb_veget = _as_two_band("alb_veget", alb_veget, npts)
    albedo = _as_two_band("albedo", albedo, npts)
    albedo_snow = _as_two_band("albedo_snow", albedo_snow, npts)
    snow = _as_float64(snow)
    if snow.shape != (npts,):
        raise ValueError("snow must have shape (npts,)")

    snow_mean = condveg_albedo_snow_mean(snow=snow, albedo_snow=albedo_snow)
    xios = (
        ("soilalb_vis", alb_bare[:, 0]),
        ("soilalb_nir", alb_bare[:, 1]),
        ("vegalb_vis", alb_veget[:, 0]),
        ("vegalb_nir", alb_veget[:, 1]),
        ("albedo_vis", albedo[:, 0]),
        ("albedo_nir", albedo[:, 1]),
        ("albedo_snow", snow_mean),
    )
    if bool(almaoutput):
        history_primary = (
            ("Albedo", (albedo[:, 0] + albedo[:, 1]) / 2.0),
            ("SAlbedo", (albedo_snow[:, 0] + albedo_snow[:, 1]) / 2.0),
        )
    else:
        history_primary = (
            ("soilalb_vis", alb_bare[:, 0]),
            ("soilalb_nir", alb_bare[:, 1]),
            ("vegalb_vis", alb_veget[:, 0]),
            ("vegalb_nir", alb_veget[:, 1]),
        )
    history_secondary = history_primary if int(hist2_id) > 0 else ()
    return CondvegMainOutputPacket(xios, history_primary, history_secondary)


def _as_two_band(name: str, value, npts: int | None = None):
    arr = _as_float64(value)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (npts, 2)")
    if npts is not None and arr.shape[0] != npts:
        raise ValueError(f"{name} must have shape (npts, 2)")
    return arr


def _as_pft_band(name: str, value, nvm: int):
    arr = _as_float64(value)
    if arr.ndim != 1 or arr.shape[0] != nvm:
        raise ValueError(f"{name} must have shape (nvm,)")
    return arr


def _condveg_restart_array(restart_fields, name, shape, *, val_exp):
    value = restart_fields.get(name)
    if value is None:
        return jnp.full(shape, val_exp, dtype=jnp.float64)
    value = jnp.asarray(value, dtype=jnp.float64)
    if value.shape != shape:
        raise ValueError(f"restart {name} must have shape {shape}, got {value.shape}")
    return value


def condveg_initialize(
    *,
    l_first: bool,
    veget,
    veget_max,
    frac_nobio,
    totfrac_nobio,
    zlev,
    drysoil_frac,
    height,
    snowdz,
    snowrho,
    tot_bare_soil,
    snow,
    snow_age,
    snow_nobio,
    snow_nobio_age,
    temp_air,
    pb,
    u,
    v,
    lai,
    restart_fields: Mapping[str, object] | None = None,
    alb_leaf_vis=None,
    alb_leaf_nir=None,
    snowa_aged_vis=None,
    snowa_aged_nir=None,
    snowa_dec_vis=None,
    snowa_dec_nir=None,
    fixed_snow_albedo=1.0e20,
    tcst_snowa=10.0,
    alb_ice=(0.60, 0.20),
    ok_explicitsnow=True,
    alb_bg_modis=True,
    impaze=False,
    rough_dyn=True,
    val_exp=999999.0,
    undef_sechiba=1.0e20,
) -> CondvegInitializeResult:
    """Execute ``condveg_initialize`` for the source-audited PFT14 path.

    Fortran provenance: ``condveg.f90::condveg_initialize`` lines 102-310.
    The paper configuration selects MODIS background albedo, calculated
    emissivity, and dynamic roughness. Missing MODIS restart state reaches
    map interpolation at lines 190-193 and is rejected as an external owner.
    """

    if not bool(l_first):
        raise RuntimeError("condveg_initialize repeated call is fatal at lines 170-172")
    if not bool(alb_bg_modis):
        raise CondvegInitializeBoundaryError(
            "alb_bg_modis=False reaches condveg_soilalb/map structure outside the PFT14 paper target (lines 195-240)"
        )
    if bool(impaze):
        raise CondvegInitializeBoundaryError(
            "impaze=True is outside the audited paper structure (condveg_initialize lines 262-270, 283-288)"
        )
    if not bool(rough_dyn):
        raise CondvegInitializeBoundaryError(
            "rough_dyn=False reaches the target-external static roughness branch (lines 294-297)"
        )

    veget = _as_float64(veget)
    veget_max = _as_float64(veget_max)
    if veget.ndim != 2 or veget.shape != veget_max.shape or veget.shape[1] < 14:
        raise ValueError(
            "veget and veget_max must share shape (npts, nvm) with nvm >= 14"
        )
    npts, nvm = veget.shape
    restart_fields = {} if restart_fields is None else dict(restart_fields)

    # Lines 180-193: only an entirely missing field invokes interpolation;
    # partially sentinel-valued restart arrays pass through unchanged.
    soilalb_bg = _condveg_restart_array(
        restart_fields, "soilalbedo_bg", (npts, 2), val_exp=val_exp
    )
    if bool(jnp.all(soilalb_bg == val_exp)):
        raise CondvegInitializeBoundaryError(
            "missing soilalbedo_bg requires condveg_background_soilalb map interpolation (lines 190-193)"
        )

    z0m = _condveg_restart_array(restart_fields, "z0m", (npts,), val_exp=val_exp)
    z0h = _condveg_restart_array(restart_fields, "z0h", (npts,), val_exp=val_exp)
    roughheight = _condveg_restart_array(
        restart_fields, "roughheight", (npts,), val_exp=val_exp
    )
    roughheight_pft = _condveg_restart_array(
        restart_fields, "roughheight_pft", (npts, nvm), val_exp=val_exp
    )

    # Lines 260-275 precede the restart fallback test.
    emis = jnp.ones((npts,), dtype=jnp.float64)
    snow_fractions = condveg_frac_snow(
        snow=snow,
        snow_nobio=snow_nobio,
        snowrho=snowrho,
        snowdz=snowdz,
        ok_explicitsnow=ok_explicitsnow,
    )

    recomputed_roughness = bool(
        jnp.all(z0m == val_exp)
        | jnp.all(z0h == val_exp)
        | jnp.all(roughheight == val_exp)
    )
    if recomputed_roughness:
        roughness = condveg_z0cdrag_dyn(
            veget=veget,
            veget_max=veget_max,
            frac_nobio=frac_nobio,
            totfrac_nobio=totfrac_nobio,
            zlev=zlev,
            height=height,
            temp_air=temp_air,
            pb=pb,
            u=u,
            v=v,
            lai=lai,
            frac_snow_veg=snow_fractions.frac_snow_veg,
        )
        z0m, z0h = roughness.z0m, roughness.z0h
        roughheight, roughheight_pft = roughness.roughheight, roughness.roughheight_pft

    albedo_result = condveg_albedo_explicit(
        veget=veget,
        veget_max=veget_max,
        drysoil_frac=drysoil_frac,
        frac_nobio=frac_nobio,
        totfrac_nobio=totfrac_nobio,
        snow=snow,
        snow_age=snow_age,
        snow_nobio=snow_nobio,
        snow_nobio_age=snow_nobio_age,
        tot_bare_soil=tot_bare_soil,
        frac_snow_veg=snow_fractions.frac_snow_veg,
        frac_snow_nobio=snow_fractions.frac_snow_nobio,
        impaze=False,
        alb_bg_modis=True,
        soilalb_bg=soilalb_bg,
        alb_leaf_vis=alb_leaf_vis,
        alb_leaf_nir=alb_leaf_nir,
        snowa_aged_vis=snowa_aged_vis,
        snowa_aged_nir=snowa_aged_nir,
        snowa_dec_vis=snowa_dec_vis,
        snowa_dec_nir=snowa_dec_nir,
        fixed_snow_albedo=fixed_snow_albedo,
        undef_sechiba=undef_sechiba,
        tcst_snowa=tcst_snowa,
        alb_ice=alb_ice,
    )
    return CondvegInitializeResult(
        soilalb_bg=soilalb_bg,
        emis=emis,
        albedo=albedo_result.albedo,
        z0m=z0m,
        z0h=z0h,
        roughheight=roughheight,
        roughheight_pft=roughheight_pft,
        frac_snow_veg=snow_fractions.frac_snow_veg,
        frac_snow_nobio=snow_fractions.frac_snow_nobio,
        l_first=False,
        recomputed_roughness=recomputed_roughness,
    )


def condveg_albedo_explicit(
    *,
    veget,
    veget_max,
    drysoil_frac,
    frac_nobio,
    totfrac_nobio,
    snow,
    snow_age,
    snow_nobio,
    snow_nobio_age,
    tot_bare_soil,
    frac_snow_veg,
    frac_snow_nobio,
    impaze=False,
    albedo_scal=None,
    alb_bg_modis=False,
    alb_bare_model=False,
    soilalb_bg=None,
    soilalb_wet=None,
    soilalb_dry=None,
    soilalb_moy=None,
    alb_leaf_vis=None,
    alb_leaf_nir=None,
    snowa_aged_vis=None,
    snowa_aged_nir=None,
    snowa_dec_vis=None,
    snowa_dec_nir=None,
    fixed_snow_albedo=1.0e20,
    undef_sechiba=1.0e20,
    tcst_snowa=10.0,
    alb_ice=(0.60, 0.20),
    min_sechiba=MIN_SECHIBA,
) -> CondvegAlbedoResult:
    """Compute CONDVEG albedo outputs.

    Fortran provenance: ``condveg.f90::condveg_albedo`` lines 612-839.
    This explicit port supports ``nnobio=1``/``iice=1`` and keeps all
    module-state albedo arrays explicit instead of inferring them.
    """

    veget = _as_float64(veget)
    veget_max = _as_float64(veget_max)
    drysoil_frac = _as_float64(drysoil_frac)
    frac_nobio = _as_float64(frac_nobio)
    totfrac_nobio = _as_float64(totfrac_nobio)
    snow = _as_float64(snow)
    snow_age = _as_float64(snow_age)
    snow_nobio = _as_float64(snow_nobio)
    snow_nobio_age = _as_float64(snow_nobio_age)
    tot_bare_soil = _as_float64(tot_bare_soil)
    frac_snow_veg = _as_float64(frac_snow_veg)
    frac_snow_nobio = _as_float64(frac_snow_nobio)

    if veget.ndim != 2:
        raise ValueError("veget must have shape (npts, nvm)")
    pft_shape = veget.shape
    npts, nvm = pft_shape
    if veget_max.shape != pft_shape:
        raise ValueError(f"veget_max must have shape {pft_shape}")
    if (
        frac_nobio.ndim != 2
        or frac_nobio.shape != snow_nobio.shape
        or frac_nobio.shape != frac_snow_nobio.shape
    ):
        raise ValueError(
            "frac_nobio, snow_nobio, and frac_snow_nobio must have shape (npts, nnobio)"
        )
    if frac_nobio.shape[0] != npts or frac_nobio.shape[1] != 1:
        raise ValueError(
            "condveg_albedo_explicit is source-backed only for nnobio=1/iice=1"
        )
    if snow_nobio_age.ndim == 1:
        if snow_nobio_age.shape[0] != npts:
            raise ValueError("snow_nobio_age must have shape (npts,) or (npts, nnobio)")
        snow_nobio_age = snow_nobio_age[:, None]
    elif snow_nobio_age.shape != frac_nobio.shape:
        raise ValueError("snow_nobio_age must have shape (npts,) or (npts, nnobio)")
    for name, value in (
        ("drysoil_frac", drysoil_frac),
        ("totfrac_nobio", totfrac_nobio),
        ("snow", snow),
        ("snow_age", snow_age),
        ("tot_bare_soil", tot_bare_soil),
        ("frac_snow_veg", frac_snow_veg),
    ):
        if value.ndim != 1 or value.shape[0] != npts:
            raise ValueError(f"{name} must have shape (npts,)")

    albedo = jnp.zeros((npts, 2), dtype=jnp.float64)
    alb_bare = jnp.zeros((npts, 2), dtype=jnp.float64)
    alb_veget = jnp.zeros((npts, 2), dtype=jnp.float64)

    if bool(impaze):
        if albedo_scal is None:
            raise ValueError("albedo_scal is required when impaze=True")
        albedo_scal = jnp.asarray(albedo_scal, dtype=jnp.float64)
        if albedo_scal.shape != (2,):
            raise ValueError("albedo_scal must have shape (2,)")
        albedo = jnp.broadcast_to(albedo_scal[None, :], (npts, 2))
    else:
        alb_leaf_vis = _as_pft_band("alb_leaf_vis", alb_leaf_vis, nvm)
        alb_leaf_nir = _as_pft_band("alb_leaf_nir", alb_leaf_nir, nvm)
        alb_leaf = jnp.stack((alb_leaf_vis, alb_leaf_nir), axis=1)

        if bool(alb_bg_modis):
            if soilalb_bg is None:
                raise ValueError("soilalb_bg is required when alb_bg_modis=True")
            alb_bare = _as_two_band("soilalb_bg", soilalb_bg, npts)
        elif bool(alb_bare_model):
            if soilalb_wet is None or soilalb_dry is None:
                raise ValueError(
                    "soilalb_wet and soilalb_dry are required when alb_bare_model=True"
                )
            soilalb_wet = _as_two_band("soilalb_wet", soilalb_wet, npts)
            soilalb_dry = _as_two_band("soilalb_dry", soilalb_dry, npts)
            alb_bare = soilalb_wet + drysoil_frac[:, None] * (soilalb_dry - soilalb_wet)
        else:
            if soilalb_moy is None:
                raise ValueError(
                    "soilalb_moy is required when alb_bg_modis=False and alb_bare_model=False"
                )
            alb_bare = _as_two_band("soilalb_moy", soilalb_moy, npts)

        albedo = tot_bare_soil[:, None] * alb_bare
        if nvm > 1:
            veg_leaf = veget[:, 1:] @ alb_leaf[1:, :]
            albedo = albedo + veg_leaf
            alb_veget = alb_veget + veg_leaf

    use_fixed_snow = abs(float(fixed_snow_albedo) - float(undef_sechiba)) > float(
        jnp.finfo(jnp.float64).eps
    )
    fixed_snow_albedo = jnp.asarray(fixed_snow_albedo, dtype=jnp.float64)
    if use_fixed_snow:
        snowa_veg = jnp.full((npts, 2), fixed_snow_albedo, dtype=jnp.float64)
        snowa_nobio = jnp.full((npts, 1, 2), fixed_snow_albedo, dtype=jnp.float64)
    else:
        snowa_aged_vis = _as_pft_band("snowa_aged_vis", snowa_aged_vis, nvm)
        snowa_aged_nir = _as_pft_band("snowa_aged_nir", snowa_aged_nir, nvm)
        snowa_dec_vis = _as_pft_band("snowa_dec_vis", snowa_dec_vis, nvm)
        snowa_dec_nir = _as_pft_band("snowa_dec_nir", snowa_dec_nir, nvm)
        snowa_aged = jnp.stack((snowa_aged_vis, snowa_aged_nir), axis=1)
        snowa_dec = jnp.stack((snowa_dec_vis, snowa_dec_nir), axis=1)

        agefunc_veg = jnp.exp(-snow_age / jnp.asarray(tcst_snowa, dtype=jnp.float64))
        agefunc_nobio = jnp.exp(
            -snow_nobio_age / jnp.asarray(tcst_snowa, dtype=jnp.float64)
        )
        fraction_veg = 1.0 - totfrac_nobio
        safe_fraction = jnp.where(fraction_veg > min_sechiba, fraction_veg, 1.0)
        pft_snow_albedo = (
            snowa_aged[None, :, :] + snowa_dec[None, :, :] * agefunc_veg[:, None, None]
        )
        snowa_veg = jnp.sum(
            veget_max[:, :, None] / safe_fraction[:, None, None] * pft_snow_albedo,
            axis=1,
        )
        snowa_veg = jnp.where((fraction_veg > min_sechiba)[:, None], snowa_veg, 0.0)
        snowa_nobio = (
            snowa_aged[0, :][None, None, :]
            + snowa_dec[0, :][None, None, :] * agefunc_nobio[:, :, None]
        )

    fraction_veg = 1.0 - totfrac_nobio
    updated = fraction_veg[:, None] * (
        (1.0 - frac_snow_veg[:, None]) * albedo + frac_snow_veg[:, None] * snowa_veg
    )
    alb_ice = jnp.asarray(alb_ice, dtype=jnp.float64)
    if alb_ice.shape != (2,):
        raise ValueError("alb_ice must have shape (2,)")
    nobio_term = frac_nobio[:, :, None] * (
        (1.0 - frac_snow_nobio[:, :, None]) * alb_ice[None, None, :]
        + frac_snow_nobio[:, :, None] * snowa_nobio
    )
    albedo = updated + jnp.sum(nobio_term, axis=1)
    albedo_snow = fraction_veg[:, None] * frac_snow_veg[:, None] * snowa_veg
    albedo_snow = albedo_snow + jnp.sum(
        frac_nobio[:, :, None] * frac_snow_nobio[:, :, None] * snowa_nobio, axis=1
    )
    albedo_snow_mean = condveg_albedo_snow_mean(snow=snow, albedo_snow=albedo_snow)

    return CondvegAlbedoResult(
        albedo=albedo,
        albedo_snow=albedo_snow,
        alb_bare=alb_bare,
        alb_veget=alb_veget,
        snowa_veg=snowa_veg,
        snowa_nobio=snowa_nobio,
        albedo_snow_mean=albedo_snow_mean,
    )


def condveg_main_minimal(
    *,
    snow,
    snow_age=None,
    snow_nobio,
    snow_nobio_age=None,
    snowrho,
    snowdz,
    veget,
    veget_max,
    frac_nobio,
    totfrac_nobio,
    zlev,
    height,
    temp_air,
    pb,
    u,
    v,
    lai,
    emis_scal,
    impaze=False,
    rough_dyn=True,
    ok_explicitsnow=True,
    z0_scal=None,
    roughheight_scal=None,
    drysoil_frac=None,
    tot_bare_soil=None,
    albedo_scal=None,
    alb_bg_modis=False,
    alb_bare_model=False,
    soilalb_bg=None,
    soilalb_wet=None,
    soilalb_dry=None,
    soilalb_moy=None,
    alb_leaf_vis=None,
    alb_leaf_nir=None,
    snowa_aged_vis=None,
    snowa_aged_nir=None,
    snowa_dec_vis=None,
    snowa_dec_nir=None,
    fixed_snow_albedo=1.0e20,
    undef_sechiba=1.0e20,
    tcst_snowa=10.0,
    alb_ice=(0.60, 0.20),
    is_tree=None,
    z0_over_height=None,
    ratio_z0m_z0h=None,
    almaoutput=False,
    hist2_id=-1,
) -> CondvegMainMinimalResult:
    """Run the source-backed subset of ``condveg_main``.

    Covered order: snow fraction, emissivity, and roughness branch dispatch
    from ``condveg_main`` lines 398-428. The active paper-case roughness path
    calls ``condveg_z0cdrag_dyn``. The non-dynamic roughness and albedo
    helpers run only when their explicit module-state inputs are supplied.
    """

    snow_fraction = condveg_frac_snow(
        snow=snow,
        snow_nobio=snow_nobio,
        snowrho=snowrho,
        snowdz=snowdz,
        ok_explicitsnow=ok_explicitsnow,
    )
    kjpindex = int(_as_float64(snow).shape[0])
    emis = condveg_main_emissivity(kjpindex, emis_scal=emis_scal)

    if impaze:
        if z0_scal is None or roughheight_scal is None:
            raise ValueError(
                "impaze=True requires explicit z0_scal and roughheight_scal"
            )
        roughness = condveg_prescribed_roughness(
            kjpindex=kjpindex,
            nvm=int(_as_float64(veget_max).shape[1]),
            z0_scal=z0_scal,
            roughheight_scal=roughheight_scal,
        )
    elif rough_dyn:
        roughness = condveg_z0cdrag_dyn(
            veget=veget,
            veget_max=veget_max,
            frac_nobio=frac_nobio,
            totfrac_nobio=totfrac_nobio,
            zlev=zlev,
            height=height,
            temp_air=temp_air,
            pb=pb,
            u=u,
            v=v,
            lai=lai,
            frac_snow_veg=snow_fraction.frac_snow_veg,
        )
    else:
        missing = tuple(
            name
            for name, value in (
                ("is_tree", is_tree),
                ("z0_over_height", z0_over_height),
                ("ratio_z0m_z0h", ratio_z0m_z0h),
                ("tot_bare_soil", tot_bare_soil),
            )
            if value is None
        )
        if missing:
            raise ValueError(
                f"rough_dyn=False requires explicit condveg_z0cdrag inputs: {missing}"
            )
        roughness = condveg_z0cdrag(
            veget=veget,
            veget_max=veget_max,
            frac_nobio=frac_nobio,
            totfrac_nobio=totfrac_nobio,
            zlev=zlev,
            height=height,
            tot_bare_soil=tot_bare_soil,
            is_tree=is_tree,
            z0_over_height=z0_over_height,
            ratio_z0m_z0h=ratio_z0m_z0h,
        )

    albedo_result = None
    albedo_fields = (
        drysoil_frac,
        snow_age,
        snow_nobio_age,
        albedo_scal,
        soilalb_bg,
        soilalb_wet,
        soilalb_dry,
        soilalb_moy,
        alb_leaf_vis,
        alb_leaf_nir,
    )
    if any(value is not None for value in albedo_fields):
        missing = tuple(
            name
            for name, value in (
                ("drysoil_frac", drysoil_frac),
                ("tot_bare_soil", tot_bare_soil),
                ("snow_age", snow_age),
                ("snow_nobio_age", snow_nobio_age),
            )
            if value is None
        )
        if missing:
            raise ValueError(f"missing condveg_albedo inputs: {missing}")
        albedo_result = condveg_albedo_explicit(
            veget=veget,
            veget_max=veget_max,
            drysoil_frac=drysoil_frac,
            frac_nobio=frac_nobio,
            totfrac_nobio=totfrac_nobio,
            snow=snow,
            snow_age=snow_age,
            snow_nobio=snow_nobio,
            snow_nobio_age=snow_nobio_age,
            tot_bare_soil=tot_bare_soil,
            frac_snow_veg=snow_fraction.frac_snow_veg,
            frac_snow_nobio=snow_fraction.frac_snow_nobio,
            impaze=impaze,
            albedo_scal=albedo_scal,
            alb_bg_modis=alb_bg_modis,
            alb_bare_model=alb_bare_model,
            soilalb_bg=soilalb_bg,
            soilalb_wet=soilalb_wet,
            soilalb_dry=soilalb_dry,
            soilalb_moy=soilalb_moy,
            alb_leaf_vis=alb_leaf_vis,
            alb_leaf_nir=alb_leaf_nir,
            snowa_aged_vis=snowa_aged_vis,
            snowa_aged_nir=snowa_aged_nir,
            snowa_dec_vis=snowa_dec_vis,
            snowa_dec_nir=snowa_dec_nir,
            fixed_snow_albedo=fixed_snow_albedo,
            undef_sechiba=undef_sechiba,
            tcst_snowa=tcst_snowa,
            alb_ice=alb_ice,
        )

    return CondvegMainMinimalResult(
        frac_snow_veg=snow_fraction.frac_snow_veg,
        frac_snow_nobio=snow_fraction.frac_snow_nobio,
        emis=emis,
        z0m=roughness.z0m,
        z0h=roughness.z0h,
        roughheight=roughness.roughheight,
        roughheight_pft=roughness.roughheight_pft,
        albedo=None if albedo_result is None else albedo_result.albedo,
        albedo_snow=None if albedo_result is None else albedo_result.albedo_snow,
        alb_bare=None if albedo_result is None else albedo_result.alb_bare,
        alb_veget=None if albedo_result is None else albedo_result.alb_veget,
        albedo_snow_mean=None
        if albedo_result is None
        else albedo_result.albedo_snow_mean,
    )


def condveg_main_minimal_coverage(
    *,
    impaze,
    rough_dyn,
    has_z0_scal=False,
    has_roughheight_scal=False,
    has_albedo_inputs=False,
    has_static_roughness_inputs=False,
) -> CondvegCoverage:
    """Report which ``condveg_main`` outputs this minimal module can cover."""

    covered = ["frac_snow_veg", "frac_snow_nobio", "emis"]
    missing_fields: list[str] = []
    missing_inputs: list[str] = []
    notes: list[str] = []

    if impaze:
        if has_z0_scal and has_roughheight_scal:
            covered.extend(["z0m", "z0h", "roughheight", "roughheight_pft"])
            branch = "impaze_prescribed"
        else:
            missing_fields.extend(["z0m", "z0h", "roughheight", "roughheight_pft"])
            if not has_z0_scal:
                missing_inputs.append("z0_scal")
            if not has_roughheight_scal:
                missing_inputs.append("roughheight_scal")
            branch = "impaze_prescribed_missing_scalars"
    elif rough_dyn:
        covered.extend(["z0m", "z0h", "roughheight", "roughheight_pft"])
        branch = "dynamic_roughness"
    else:
        if has_static_roughness_inputs:
            covered.extend(["z0m", "z0h", "roughheight", "roughheight_pft"])
            branch = "static_roughness"
        else:
            missing_fields.extend(["z0m", "z0h", "roughheight", "roughheight_pft"])
            missing_inputs.append("condveg_z0cdrag explicit PFT parameters")
            branch = "static_roughness_missing_inputs"

    if has_albedo_inputs:
        covered.extend(["albedo", "albedo_snow", "alb_bare", "alb_veget"])
        notes.append(
            "condveg_albedo is covered when explicit soil, leaf, snow-age, and snow-albedo module state is supplied."
        )
    else:
        missing_fields.extend(["albedo", "albedo_snow", "alb_bare", "alb_veget"])
        missing_inputs.append("condveg_albedo explicit constants/state")

    ok = not missing_fields and not missing_inputs
    provenance = (
        *CONDVEG_MAIN_PROVENANCE,
        *CONDVEG_FRAC_SNOW_PROVENANCE,
        *CONDVEG_Z0CDRAG_PROVENANCE,
        *CONDVEG_Z0CDRAG_DYN_PROVENANCE,
        *CONDVEG_ALBEDO_PROVENANCE,
    )
    return CondvegCoverage(
        ok=ok,
        covered_fields=tuple(covered),
        missing_fields=tuple(dict.fromkeys(missing_fields)),
        missing_inputs=tuple(dict.fromkeys(missing_inputs)),
        branch=branch,
        provenance=provenance,
        notes=tuple(notes),
    )


CONDVEG_FIRST_STEP_MODULE_CLOSURE_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1085-1091",
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_main lines 398-435",
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_frac_snow lines 858-898",
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_z0cdrag lines 1288-1474",
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_z0cdrag_dyn lines 1519-1720",
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_albedo lines 612-839",
)


def run_condveg_first_step_module(
    *,
    restart_payload: Mapping[str, object],
    driver_payload: Mapping[str, object],
    slowproc_payload: Mapping[str, object],
    hydrol_diagnostics=None,
    run_def_values: Mapping[str, str] | None = None,
    ok_explicitsnow=True,
    impaze=False,
    rough_dyn=True,
    use_jit=False,
) -> CondvegFirstStepModuleClosure:
    """Run first-step ``condveg_main`` from explicit upstream state.

    HYDROL supplies ``drysoil_frac`` after ``hydrol_main``. If it is absent,
    this helper still closes the same-step snow-fraction boundary needed by
    THERMOSOIL, but it reports the albedo update as missing instead of
    substituting a cold-start value.
    """

    values = {} if run_def_values is None else dict(run_def_values)
    restart = dict(restart_payload)
    driver = dict(driver_payload)
    slowproc = dict(slowproc_payload)
    nvm = int(getattr(restart.get("veget_max"), "shape", (0, 0))[1])

    def indexed(name: str, default=None):
        if f"{name}__00001" not in values:
            if default is None:
                raise KeyError(name)
            return default
        import numpy as np

        return np.asarray(
            [float(values[f"{name}__{index:05d}"]) for index in range(1, nvm + 1)],
            dtype=float,
        )

    inputs = {
        "snow": restart.get("snow"),
        "snow_nobio": restart.get("snow_nobio"),
        "snowrho": restart.get("snowrho"),
        "snowdz": restart.get("snowdz"),
        "veget": restart.get("veget"),
        "veget_max": restart.get("veget_max"),
        "frac_nobio": restart.get("frac_nobio"),
        "totfrac_nobio": slowproc.get("totfrac_nobio"),
        "zlev": driver.get("zlev"),
        "height": restart.get("height"),
        "temp_air": driver.get("temp_air"),
        "pb": driver.get("pb"),
        "u": driver.get("u"),
        "v": driver.get("v"),
        "lai": restart.get("lai"),
        "emis_scal": 1.0,
        "impaze": impaze,
        "rough_dyn": rough_dyn,
        "ok_explicitsnow": ok_explicitsnow,
    }
    if (not bool(impaze)) and (not bool(rough_dyn)):
        inputs.update(
            {
                "is_tree": slowproc.get("is_tree"),
                "z0_over_height": slowproc.get("z0_over_height"),
                "ratio_z0m_z0h": slowproc.get("ratio_z0m_z0h"),
            }
        )
    if "totfrac_nobio" not in inputs or inputs["totfrac_nobio"] is None:
        import numpy as np

        if inputs["frac_nobio"] is not None:
            inputs["totfrac_nobio"] = np.sum(np.asarray(inputs["frac_nobio"]), axis=1)

    albedo_inputs = {
        "drysoil_frac": None
        if hydrol_diagnostics is None
        else hydrol_diagnostics.drysoil_frac,
        "tot_bare_soil": slowproc.get("tot_bare_soil"),
        "snow_age": restart.get("snow_age"),
        "snow_nobio_age": restart.get("snow_nobio_age"),
        "soilalb_bg": restart.get("soilalbedo_bg"),
    }
    has_albedo_inputs = all(value is not None for value in albedo_inputs.values())
    if has_albedo_inputs:
        inputs.update(albedo_inputs)
        if values:
            inputs.update(
                {
                    "alb_bg_modis": str(values.get("ALB_BG_MODIS", "FALSE"))
                    .strip()
                    .strip(".")
                    .upper()
                    in {"T", "TRUE", "Y", "YES", "1"},
                    "alb_bare_model": str(values.get("ALB_BARE_MODEL", "FALSE"))
                    .strip()
                    .strip(".")
                    .upper()
                    in {"T", "TRUE", "Y", "YES", "1"},
                    "fixed_snow_albedo": float(values.get("CONDVEG_SNOWA", 1.0e20)),
                    "tcst_snowa": float(values.get("TCST_SNOWA", 10.0)),
                    "alb_ice": tuple(
                        float(values[f"ALB_ICE__{index:05d}"]) for index in range(1, 3)
                    ),
                    "alb_leaf_vis": indexed("ALB_LEAF_VIS"),
                    "alb_leaf_nir": indexed("ALB_LEAF_NIR"),
                    "snowa_aged_vis": indexed("SNOWA_AGED_VIS"),
                    "snowa_aged_nir": indexed("SNOWA_AGED_NIR"),
                    "snowa_dec_vis": indexed("SNOWA_DEC_VIS"),
                    "snowa_dec_nir": indexed("SNOWA_DEC_NIR"),
                }
            )

    required = [
        "snow",
        "snow_nobio",
        "veget",
        "veget_max",
        "frac_nobio",
        "totfrac_nobio",
        "zlev",
        "height",
        "temp_air",
        "pb",
        "u",
        "v",
        "lai",
    ]
    if bool(ok_explicitsnow):
        required.extend(("snowrho", "snowdz"))
    if (not bool(impaze)) and (not bool(rough_dyn)):
        required.extend(("tot_bare_soil", "is_tree", "z0_over_height", "ratio_z0m_z0h"))
    missing = [name for name in required if inputs.get(name) is None]
    if missing:
        return CondvegFirstStepModuleClosure(
            module=None,
            inputs={name: value for name, value in inputs.items() if value is not None},
            missing_inputs=tuple(dict.fromkeys(missing)),
            covered_fields=(),
            provenance=CONDVEG_FIRST_STEP_MODULE_CLOSURE_PROVENANCE,
            notes=(
                "CONDVEG first-step execution was skipped because strict inputs are missing.",
            ),
        )

    if bool(use_jit) and has_albedo_inputs:
        module = _condveg_main_minimal_jit(**inputs)
    else:
        module = condveg_main_minimal(**inputs)
    covered = [
        "frac_snow_veg",
        "frac_snow_nobio",
        "emis",
        "z0m",
        "z0h",
        "roughheight",
        "roughheight_pft",
    ]
    missing_albedo = ()
    if module.albedo is not None:
        covered.extend(
            ("albedo", "albedo_snow", "alb_bare", "alb_veget", "albedo_snow_mean")
        )
    else:
        missing_albedo = ("albedo_explicit_inputs",)
    return CondvegFirstStepModuleClosure(
        module=module,
        inputs=inputs,
        missing_inputs=missing_albedo,
        covered_fields=tuple(covered),
        provenance=CONDVEG_FIRST_STEP_MODULE_CLOSURE_PROVENANCE,
        notes=(
            "Runs CONDVEG first-step snow fractions and roughness from explicit source-backed inputs.",
            "Albedo is reported as missing unless HYDROL drysoil_frac, snow ages, and soil albedo restart state are all supplied.",
        ),
    )


CONDVEG_MAIN_MINIMAL_JIT_STATIC_ARGNAMES = (
    "impaze",
    "rough_dyn",
    "ok_explicitsnow",
    "alb_bg_modis",
    "alb_bare_model",
    "fixed_snow_albedo",
    "tcst_snowa",
    "undef_sechiba",
)


_condveg_main_minimal_jit = jit(
    condveg_main_minimal,
    static_argnames=CONDVEG_MAIN_MINIMAL_JIT_STATIC_ARGNAMES,
)


def condveg_background_soilalb_source_routed(
    *, bg_alb_vis, bg_alb_nir, target, aggregate, undef_sechiba=1.0e20
):
    """Production entry point for background-soil-albedo interpolation."""
    from .science_completion import condveg_background_soilalb_source_routed as owner

    return owner(
        bg_alb_vis=bg_alb_vis,
        bg_alb_nir=bg_alb_nir,
        target=target,
        aggregate=aggregate,
        undef_sechiba=undef_sechiba,
    )


def condveg_finalize_restart_packet(**kwargs):
    """Production entry point for ``condveg_finalize`` restart selection."""
    from .science_completion import condveg_finalize_restart_packet as owner

    return owner(**kwargs)
