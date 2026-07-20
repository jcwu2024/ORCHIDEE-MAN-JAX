"""Modelout formula mapper for PFT14 STOMATE history fields."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jax_orchidee.stomate.carbon_kernels import (
    IAGRHRTST,
    IAGRHRTPN,
    IAGRSAPST,
    IAGRSAPPN,
    ICARBON,
    ICARBRES,
    IFRUIT,
    IHEARTABOVE,
    IHEARTBELOW,
    ILEAF,
    IROOT,
    ISAPABOVE,
    ISAPBELOW,
)


MODEL_OUTPUT_FIELD_NAMES = (
    "LEAF_M",
    "SAP_M_AB",
    "HEART_M_AB",
    "AGR_SAP_ST_M",
    "AGR_HRT_ST_M",
    "AGR_SAP_PN_M",
    "AGR_HRT_PN_M",
    "SAP_M_BE",
    "HEART_M_BE",
    "ROOT_M",
    "GPP",
    "NPP",
)

MODEL_OUTPUT_PROVENANCE = (
    "Fortran writes: src_stomate/stomate_lpj.f90, subroutine StomateLpj, "
    "lines 2200-2246. XML enabled fields: "
    "fortran_run_scripts/paper_250919/peat-leak_xml_zz/"
    "file_def_orchidee_pods_year_240405.xml, lines 747 and 762-783. "
    "Paper modelout formula: "
    "fortran_run_scripts/paper_250919/c2.4_Model_run_functions_sensitivity.py, "
    "lines 50, 60-76, and 115-118."
)

PAPER_MODEL_START_YEAR = 1961
PAPER_MODEL_PFT14_INDEX = 13


class ModeloutResult(NamedTuple):
    """Paper modelout columns derived from selected STOMATE fields."""

    AGB_model: jnp.ndarray
    BGB_model: jnp.ndarray
    GPP_model: jnp.ndarray
    NPP_model: jnp.ndarray


class ModeloutFieldSource(NamedTuple):
    """Field-level source proof for one paper modelout input.

    Provenance: `src_stomate/stomate_lpj.f90`, subroutine `StomateLpj`,
    XIOS sends lines 1677-1699 and history writes lines 2200-2246; XML field
    definitions are in `fortran_run_scripts/paper_250919/peat-leak_xml_zz/`
    `field_def_orchidee.xml`, lines 634-647, and yearly file selection is in
    `file_def_orchidee_pods_year_240405.xml`, lines 762-782.
    """

    history_name: str
    xios_field_id: str
    state_argument: str
    fortran_expression: str
    xios_send_line: int
    history_write_lines: tuple[int, int]
    xml_field_def_line: int
    xml_file_line: int


MODEL_OUTPUT_FIELD_SOURCES: dict[str, ModeloutFieldSource] = {
    "LEAF_M": ModeloutFieldSource(
        "LEAF_M",
        "LEAF_M",
        "biomass",
        "biomass(:,:,ileaf,icarbon)",
        1684,
        (2212, 2213),
        636,
        771,
    ),
    "SAP_M_AB": ModeloutFieldSource(
        "SAP_M_AB",
        "SAP_M_AB",
        "biomass",
        "biomass(:,:,isapabove,icarbon)",
        1685,
        (2214, 2215),
        637,
        772,
    ),
    "HEART_M_AB": ModeloutFieldSource(
        "HEART_M_AB",
        "HEART_M_AB",
        "biomass",
        "biomass(:,:,iheartabove,icarbon)",
        1687,
        (2218, 2219),
        639,
        774,
    ),
    "AGR_SAP_ST_M": ModeloutFieldSource(
        "AGR_SAP_ST_M",
        "AGR_SAP_ST_M",
        "biomass",
        "biomass(:,:,iagrsapst,icarbon)",
        1696,
        (2239, 2240),
        644,
        779,
    ),
    "AGR_HRT_ST_M": ModeloutFieldSource(
        "AGR_HRT_ST_M",
        "AGR_HRT_ST_M",
        "biomass",
        "biomass(:,:,iagrhrtst,icarbon)",
        1698,
        (2243, 2244),
        646,
        781,
    ),
    "AGR_SAP_PN_M": ModeloutFieldSource(
        "AGR_SAP_PN_M",
        "AGR_SAP_PN_M",
        "biomass",
        "biomass(:,:,iagrsappn,icarbon)",
        1697,
        (2241, 2242),
        645,
        780,
    ),
    "AGR_HRT_PN_M": ModeloutFieldSource(
        "AGR_HRT_PN_M",
        "AGR_HRT_PN_M",
        "biomass",
        "biomass(:,:,iagrhrtpn,icarbon)",
        1699,
        (2245, 2246),
        647,
        782,
    ),
    "SAP_M_BE": ModeloutFieldSource(
        "SAP_M_BE",
        "SAP_M_BE",
        "biomass",
        "biomass(:,:,isapbelow,icarbon)",
        1686,
        (2216, 2217),
        638,
        773,
    ),
    "HEART_M_BE": ModeloutFieldSource(
        "HEART_M_BE",
        "HEART_M_BE",
        "biomass",
        "biomass(:,:,iheartbelow,icarbon)",
        1688,
        (2220, 2221),
        640,
        775,
    ),
    "ROOT_M": ModeloutFieldSource(
        "ROOT_M",
        "ROOT_M",
        "biomass",
        "biomass(:,:,iroot,icarbon)",
        1689,
        (2222, 2223),
        641,
        776,
    ),
    "GPP": ModeloutFieldSource(
        "GPP",
        "GPP",
        "gpp_daily",
        "gpp_daily",
        1678,
        (2202, 2203),
        635,
        763,
    ),
    "NPP": ModeloutFieldSource(
        "NPP",
        "NPP_STOMATE",
        "npp_daily",
        "npp_daily",
        1677,
        (2200, 2201),
        634,
        762,
    ),
}


def modelout_source_gaps(field_names: Iterable[str] = MODEL_OUTPUT_FIELD_NAMES) -> tuple[str, ...]:
    """Return paper modelout fields without audited STOMATE output provenance.

    Fortran provenance: `src_stomate/stomate_lpj.f90`, subroutine
    `StomateLpj`, output writes lines 1677-1699 and 2200-2246; paper XML
    enables the corresponding yearly fields in
    `file_def_orchidee_pods_year_240405.xml`, lines 762-782. This helper is a
    guardrail only; it reports missing source proof and never fills data.
    """

    return tuple(name for name in field_names if name not in MODEL_OUTPUT_FIELD_SOURCES)


def history_year_for_age(age: int, *, start_year: int = PAPER_MODEL_START_YEAR) -> int:
    """Return the STOMATE history year selected by the paper modelout script.

    Provenance: ``fortran_run_scripts/paper_250919/``
    ``c2.4_Model_run_functions_sensitivity.py``, function ``extract_file``,
    line 48 uses ``stomate_history_(1961 + Iage - 1).nc``.
    """

    return int(start_year) + int(age) - 1


def history_filename_for_age(age: int, *, start_year: int = PAPER_MODEL_START_YEAR) -> str:
    """Return the paper-script STOMATE history filename for an observation age."""

    return f"stomate_history_{history_year_for_age(age, start_year=start_year)}.nc"


def select_history_point_fields(
    fields: dict[str, object],
    *,
    time_index: int = 0,
    pft_index: int = PAPER_MODEL_PFT14_INDEX,
    lat_index: int = 0,
    lon_index: int = 0,
) -> dict[str, jnp.ndarray]:
    """Select the paper-script point/PFT slice from STOMATE history fields.

    Provenance: ``c2.4_Model_run_functions_sensitivity.py`` lines 57-71 read
    each required variable at ``[0, 13, 0, 0]``. The default ``pft_index=13``
    is the zero-based PFT14 selector; callers may supply another PFT index for
    later multi-PFT validation without changing the formula mapper.
    """

    missing = [name for name in MODEL_OUTPUT_FIELD_NAMES if name not in fields]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"Missing modelout input fields: {joined}")

    selected: dict[str, jnp.ndarray] = {}
    for name in MODEL_OUTPUT_FIELD_NAMES:
        value = jnp.asarray(fields[name])
        if value.ndim != 4:
            raise ValueError(f"{name} must have shape (time, vegetation, lat, lon)")
        selected[name] = value[time_index, pft_index, lat_index, lon_index]
    return selected


def compute_modelout_from_fields(fields: dict[str, object]) -> ModeloutResult:
    """Compute paper modelout values from audited STOMATE output fields.

    Formula provenance:
    `c2.4_Model_run_functions_sensitivity.py`, lines 50, 60-76, 115-118.
    STOMATE history provenance:
    `stomate_lpj.f90`, subroutine `StomateLpj`, lines 2200-2246; yearly XML
    file `file_def_orchidee_pods_year_240405.xml`, lines 747 and 762-783.

    Required inputs preserve normal STOMATE axes, for example
    `(time, vegetation, lat, lon)` or any broadcast-compatible array shape.
    """

    missing = [name for name in MODEL_OUTPUT_FIELD_NAMES if name not in fields]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"Missing modelout input fields: {joined}")

    values = {name: jnp.asarray(fields[name]) for name in MODEL_OUTPUT_FIELD_NAMES}
    agb = 0.02 * (
        values["LEAF_M"]
        + values["SAP_M_AB"]
        + values["HEART_M_AB"]
        + values["AGR_SAP_ST_M"]
        + values["AGR_HRT_ST_M"]
        + values["AGR_SAP_PN_M"]
        + values["AGR_HRT_PN_M"]
    )
    bgb = 0.02 * (values["SAP_M_BE"] + values["HEART_M_BE"] + values["ROOT_M"])
    return ModeloutResult(
        AGB_model=agb,
        BGB_model=bgb,
        GPP_model=values["GPP"],
        NPP_model=values["NPP"],
    )


def annual_history_mean_fields_from_daily_modelout(
    daily_fields: Sequence[Mapping[str, object]],
    *,
    field_names: Iterable[str] = MODEL_OUTPUT_FIELD_NAMES,
    as_history_axes: bool = True,
) -> dict[str, jnp.ndarray]:
    """Aggregate daily STOMATE output sends into a yearly history record.

    Fortran provenance:
    `src_stomate/stomate_lpj.f90::StomateLpj` sends the paper modelout fields
    through `xios_orchidee_send_field` at lines 1677-1699, and the legacy
    `histwrite_p` path writes the same values at lines 2200-2246. The paper
    XML file
    `fortran_run_scripts/paper_250919/peat-leak_xml_zz/file_def_orchidee_pods_year_240405.xml`
    enables `stomate_history` with `output_freq="1y"` at line 747 and the
    modelout fields at lines 762-782. Local NetCDF metadata for
    `reference/case_001_071/.../stomate_history_1961.nc` records
    `online_operation = average`, `interval_operation = 1800 s`, and
    `interval_write = 1 yr` for these fields.

    This helper implements only that output-layer averaging contract. It does
    not infer process state, tune values, or apply paper CSV formulas.

    Input daily fields must use the local runtime shape `(npts, nvm)` for each
    requested field. When `as_history_axes` is true, the returned arrays use
    the paper single-point history shape `(time, vegetation, lat, lon)`, i.e.
    `(1, nvm, 1, npts)`, so they can feed `select_history_point_fields`.
    """

    records = tuple(daily_fields)
    if not records:
        raise ValueError("daily_fields must contain at least one completed daily record")

    requested = tuple(field_names)
    gaps = modelout_source_gaps(requested)
    if gaps:
        joined = ", ".join(gaps)
        raise NotImplementedError(f"Missing source-backed annual history field mappings: {joined}")

    result: dict[str, jnp.ndarray] = {}
    expected_shape: tuple[int, ...] | None = None
    for name in requested:
        missing_days = [index + 1 for index, fields in enumerate(records) if name not in fields]
        if missing_days:
            raise KeyError(f"Missing {name} in daily modelout records: {missing_days}")
        stacked = jnp.stack([jnp.asarray(fields[name]) for fields in records], axis=0)
        if stacked.ndim != 3:
            raise ValueError(f"{name} daily fields must stack to shape (day, npts, nvm)")
        shape = tuple(stacked.shape[1:])
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(f"{name} daily fields have shape {shape}, expected {expected_shape}")
        mean = jnp.mean(stacked, axis=0)
        if as_history_axes:
            # Local single-point history uses (time, veget, lat, lon). For a
            # one-dimensional landpoint list, keep landpoints on the lon axis.
            mean = jnp.swapaxes(mean, 0, 1)[None, :, None, :]
        result[name] = mean
    return result


def stomate_lpj_history_fields_from_state(*, biomass, gpp_daily, npp_daily) -> dict[str, jnp.ndarray]:
    """Build paper modelout history fields from explicit ``StomateLpj`` state.

    Fortran provenance: ``src_stomate/stomate_lpj.f90``, subroutine
    ``StomateLpj``, XIOS sends for biomass pools and carbon fluxes lines
    1677-1699 and history writes lines 2200-2246. The returned field names are
    the inputs consumed by ``compute_modelout_from_fields`` and the paper
    modelout script.
    """

    gaps = modelout_source_gaps()
    if gaps:
        joined = ", ".join(gaps)
        raise NotImplementedError(f"Missing source-backed modelout field mappings: {joined}")

    biomass = jnp.asarray(biomass)
    gpp_daily = jnp.asarray(gpp_daily)
    npp_daily = jnp.asarray(npp_daily)
    if biomass.ndim != 4:
        raise ValueError("biomass must have shape (npts, nvm, nparts, nelements)")
    if gpp_daily.shape != biomass.shape[:2] or npp_daily.shape != biomass.shape[:2]:
        raise ValueError("gpp_daily and npp_daily must share (npts, nvm) with biomass")

    return {
        "LEAF_M": biomass[:, :, ILEAF, ICARBON],
        "SAP_M_AB": biomass[:, :, ISAPABOVE, ICARBON],
        "HEART_M_AB": biomass[:, :, IHEARTABOVE, ICARBON],
        "AGR_SAP_ST_M": biomass[:, :, IAGRSAPST, ICARBON],
        "AGR_HRT_ST_M": biomass[:, :, IAGRHRTST, ICARBON],
        "AGR_SAP_PN_M": biomass[:, :, IAGRSAPPN, ICARBON],
        "AGR_HRT_PN_M": biomass[:, :, IAGRHRTPN, ICARBON],
        "SAP_M_BE": biomass[:, :, ISAPBELOW, ICARBON],
        "HEART_M_BE": biomass[:, :, IHEARTBELOW, ICARBON],
        "ROOT_M": biomass[:, :, IROOT, ICARBON],
        "GPP": gpp_daily,
        "NPP": npp_daily,
        "FRUIT_M": biomass[:, :, IFRUIT, ICARBON],
        "RESERVE_M": biomass[:, :, ICARBRES, ICARBON],
    }
