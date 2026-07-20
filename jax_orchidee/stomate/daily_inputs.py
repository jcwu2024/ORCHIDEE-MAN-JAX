"""Local STOMATE daily-accumulation increments.

These helpers only implement local algebra that is explicit in
``stomate_main`` before the ``stomate_accu`` calls. They do not infer missing
state from sentinels or output targets.
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp


ONE_DAY_SECONDS = 86400.0
MIN_SECHIBA = 1.0e-8
MIN_STOMATE = 1.0e-8

DAILY_INPUTS_PROVENANCE = (
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines 2917-2918 "
    "computes precip from precip_rain and precip_snow.",
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines 2921-2931 "
    "computes veget_cov_max from veget_max and totfrac_nobio.",
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines 3020-3032 "
    "computes gpp_d before daily accumulation.",
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines 3207-3208 "
    "passes precip and gpp_d into stomate_accu.",
)


TOTFRAC_NOBIO_PROVENANCE = (
    "Fortran source truth: src_sechiba/slowproc.f90, slowproc_main lines "
    "937-940 computes totfrac_nobio_lastyear as SUM(frac_nobio_lastyear).",
    "Fortran source truth: src_sechiba/slowproc.f90, slowproc_main lines "
    "973-985 passes totfrac_nobio_lastyear as stomate_main's totfrac_nobio.",
    "Fortran source truth: src_sechiba/slowproc.f90, slowproc_veget lines "
    "2899-2902 computes totfrac_nobio as SUM(frac_nobio).",
    "Fortran source truth: src_sechiba/slowproc.f90, slowproc_main lines "
    "1116-1121 computes tot_bare_soil from veget_max and veget; it is not "
    "the same contract as totfrac_nobio.",
)


@dataclass(frozen=True)
class DailyInputAvailability:
    """Entry-payload coverage for local STOMATE daily increments."""

    precip_status: str
    gpp_status: str
    precip_fields: tuple[str, ...]
    gpp_fields: tuple[str, ...]
    missing_for_gpp: tuple[str, ...]
    provenance: tuple[str, ...] = DAILY_INPUTS_PROVENANCE
    totfrac_nobio_provenance: tuple[str, ...] = TOTFRAC_NOBIO_PROVENANCE


@dataclass(frozen=True)
class EntryNormalizationAvailability:
    """Entry-payload coverage for STOMATE vegetation-fraction normalization."""

    current_status: str
    new_firstday_status: str
    new_lcchange_status: str
    current_fields: tuple[str, ...]
    new_firstday_fields: tuple[str, ...]
    new_lcchange_fields: tuple[str, ...]
    missing_for_current: tuple[str, ...]
    missing_for_new_firstday: tuple[str, ...]
    missing_for_new_lcchange: tuple[str, ...]
    provenance: tuple[str, ...] = DAILY_INPUTS_PROVENANCE
    totfrac_nobio_provenance: tuple[str, ...] = TOTFRAC_NOBIO_PROVENANCE


@dataclass(frozen=True)
class StomateEntryLocalPrepResult:
    """Source-backed local values built at the start of ``stomate_main``."""

    precip: jnp.ndarray
    veget_cov: jnp.ndarray
    veget_cov_max: jnp.ndarray
    gpp_d: jnp.ndarray
    glccNetLCC: jnp.ndarray | None = None
    glccSecondShift: jnp.ndarray | None = None
    glccPrimaryShift: jnp.ndarray | None = None
    harvest_matrix: jnp.ndarray | None = None
    vegetnew_firstday: jnp.ndarray | None = None
    veget_cov_max_new: jnp.ndarray | None = None
    provenance: tuple[str, ...] = DAILY_INPUTS_PROVENANCE


def stomate_precip_increment(precip_rain, precip_snow, *, dt_sechiba, one_day=ONE_DAY_SECONDS):
    """Build Fortran's local ``precip`` increment for ``precip_daily``.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 2917-2918:
    ``precip = (precip_rain + precip_snow) * one_day / dt_sechiba``.
    The result is the increment passed to ``stomate_accu`` at line 3207.
    """

    return (jnp.asarray(precip_rain) + jnp.asarray(precip_snow)) * (
        jnp.asarray(one_day) / jnp.asarray(dt_sechiba)
    )


def stomate_normalize_by_bio_fraction(field, totfrac_nobio, *, min_sechiba=MIN_SECHIBA):
    """Normalize a STOMATE entry field by biological grid-cell fraction.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 2921-2931, 2935-2977, and 2980-3016. Pixels with
    ``1 - totfrac_nobio > min_sechiba`` use
    ``field / (1 - totfrac_nobio)``; other pixels are zero.
    """

    field = jnp.asarray(field)
    denominator = 1.0 - jnp.asarray(totfrac_nobio)
    safe_denominator = jnp.where(denominator > min_sechiba, denominator, 1.0)
    return jnp.where(denominator[..., None] > min_sechiba, field / safe_denominator[..., None], 0.0)


def stomate_veget_cover_fractions(veget, veget_max, totfrac_nobio, *, min_sechiba=MIN_SECHIBA):
    """Build STOMATE's local ``veget_cov`` and ``veget_cov_max`` arrays.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 2921-2931.
    """

    return (
        stomate_normalize_by_bio_fraction(veget, totfrac_nobio, min_sechiba=min_sechiba),
        stomate_normalize_by_bio_fraction(veget_max, totfrac_nobio, min_sechiba=min_sechiba),
    )


def stomate_veget_cov_max(veget_max, totfrac_nobio, *, min_sechiba=MIN_SECHIBA):
    """Build STOMATE's local maximum vegetation fraction.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 2921-2931. Pixels with
    ``1 - totfrac_nobio > min_sechiba`` use
    ``veget_max / (1 - totfrac_nobio)``; other pixels are zero.
    """

    return stomate_normalize_by_bio_fraction(veget_max, totfrac_nobio, min_sechiba=min_sechiba)


def stomate_vegetnew_firstday(vegetnew_firstday, totfrac_nobio_new, *, min_sechiba=MIN_SECHIBA):
    """Normalize first-day next-year vegetation fractions.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 2980-2990. This uses ``totfrac_nobio_new``, not
    current ``totfrac_nobio``.
    """

    return stomate_normalize_by_bio_fraction(vegetnew_firstday, totfrac_nobio_new, min_sechiba=min_sechiba)


def stomate_veget_cov_max_new(veget_max_new, totfrac_nobio_new, *, min_sechiba=MIN_SECHIBA):
    """Build STOMATE's local next-year maximum vegetation fraction.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 2993-3016. This requires ``totfrac_nobio_new``.
    """

    return stomate_normalize_by_bio_fraction(veget_max_new, totfrac_nobio_new, min_sechiba=min_sechiba)



def stomate_gpp_daily_increment(
    gpp,
    veget_max,
    totfrac_nobio,
    *,
    dt_sechiba,
    one_day=ONE_DAY_SECONDS,
    min_sechiba=MIN_SECHIBA,
    min_stomate=MIN_STOMATE,
):
    """Build Fortran's local ``gpp_d`` increment for ``gpp_daily``.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 2921-2931 construct ``veget_cov_max`` and lines
    3020-3032 construct ``gpp_d``. PFT1/bare soil is zero, and PFTs 2..nvm use
    ``gpp / veget_cov_max * one_day / dt_sechiba`` only where
    ``veget_cov_max > min_stomate``. The result is the increment passed to
    ``stomate_accu`` at line 3208.
    """

    gpp = jnp.asarray(gpp)
    veget_cov_max = stomate_veget_cov_max(veget_max, totfrac_nobio, min_sechiba=min_sechiba)
    scale = jnp.asarray(one_day) / jnp.asarray(dt_sechiba)
    safe_veget = jnp.where(veget_cov_max > min_stomate, veget_cov_max, 1.0)
    gpp_d = jnp.where(veget_cov_max > min_stomate, gpp / safe_veget * scale, 0.0)
    return gpp_d.at[:, 0].set(0.0)


def stomate_gpp_daily_increment_from_veget_cov_max(
    gpp,
    veget_cov_max,
    *,
    dt_sechiba,
    one_day=ONE_DAY_SECONDS,
    min_stomate=MIN_STOMATE,
):
    """Build ``gpp_d`` from an already source-normalized ``veget_cov_max``.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 3020-3032. This is the same algebra as
    :func:`stomate_gpp_daily_increment` after lines 2921-2931 have already
    built ``veget_cov_max``.
    """

    gpp = jnp.asarray(gpp)
    veget_cov_max = jnp.asarray(veget_cov_max)
    scale = jnp.asarray(one_day) / jnp.asarray(dt_sechiba)
    safe_veget = jnp.where(veget_cov_max > min_stomate, veget_cov_max, 1.0)
    gpp_d = jnp.where(veget_cov_max > min_stomate, gpp / safe_veget * scale, 0.0)
    return gpp_d.at[:, 0].set(0.0)


def _require_all_for_branch(branch_name: str, values: dict[str, object | None]) -> None:
    missing = tuple(name for name, value in values.items() if value is None)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{branch_name} requires explicit source-backed inputs: {joined}")


def stomate_entry_local_prep_explicit(
    *,
    precip_rain,
    precip_snow,
    dt_sechiba,
    veget,
    veget_max,
    totfrac_nobio,
    gpp,
    glccNetLCC=None,
    glccSecondShift=None,
    glccPrimaryShift=None,
    harvest_matrix=None,
    date: int | None = None,
    vegetnew_firstday=None,
    totfrac_nobio_new=None,
    veget_max_new=None,
    do_now_stomate_lcchange: bool | None = None,
    dyn_peat: bool | None = None,
    update_peatfrac: bool | None = None,
    one_day=ONE_DAY_SECONDS,
    min_sechiba=MIN_SECHIBA,
    min_stomate=MIN_STOMATE,
) -> StomateEntryLocalPrepResult:
    """Build the exact local STOMATE entry values from explicit inputs.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 2917-3032. This function groups only the local
    algebra in section 3: precipitation scaling, current biological-fraction
    normalization, GLCC matrix normalization, optional first-day/new-cover
    branches, and ``gpp_d`` scaling. Branch inputs must be supplied explicitly;
    nearby fields such as ``tot_bare_soil`` are not accepted.
    """

    precip = stomate_precip_increment(
        precip_rain,
        precip_snow,
        dt_sechiba=dt_sechiba,
        one_day=one_day,
    )
    veget_cov, veget_cov_max = stomate_veget_cover_fractions(
        veget,
        veget_max,
        totfrac_nobio,
        min_sechiba=min_sechiba,
    )
    gpp_d = stomate_gpp_daily_increment_from_veget_cov_max(
        gpp,
        veget_cov_max,
        dt_sechiba=dt_sechiba,
        one_day=one_day,
        min_stomate=min_stomate,
    )

    normalized_glcc = {
        name: (
            None
            if value is None
            else stomate_normalize_by_bio_fraction(value, totfrac_nobio, min_sechiba=min_sechiba)
        )
        for name, value in (
            ("glccNetLCC", glccNetLCC),
            ("glccSecondShift", glccSecondShift),
            ("glccPrimaryShift", glccPrimaryShift),
            ("harvest_matrix", harvest_matrix),
        )
    }

    normalized_firstday = None
    if date is None:
        if vegetnew_firstday is not None:
            _require_all_for_branch(
                "date == 1 vegetnew_firstday normalization",
                {"date": date, "totfrac_nobio_new": totfrac_nobio_new},
            )
    elif int(date) == 1:
        _require_all_for_branch(
            "date == 1 vegetnew_firstday normalization",
            {
                "vegetnew_firstday": vegetnew_firstday,
                "totfrac_nobio_new": totfrac_nobio_new,
            },
        )
        normalized_firstday = stomate_vegetnew_firstday(
            vegetnew_firstday,
            totfrac_nobio_new,
            min_sechiba=min_sechiba,
        )
    elif vegetnew_firstday is not None:
        normalized_firstday = jnp.asarray(vegetnew_firstday)

    normalized_new_cover = None
    new_cover_active = bool(do_now_stomate_lcchange) or bool(dyn_peat and update_peatfrac)
    if new_cover_active:
        _require_all_for_branch(
            "veget_cov_max_new normalization",
            {
                "veget_max_new": veget_max_new,
                "totfrac_nobio_new": totfrac_nobio_new,
            },
        )
        normalized_new_cover = stomate_veget_cov_max_new(
            veget_max_new,
            totfrac_nobio_new,
            min_sechiba=min_sechiba,
        )

    return StomateEntryLocalPrepResult(
        precip=precip,
        veget_cov=veget_cov,
        veget_cov_max=veget_cov_max,
        gpp_d=gpp_d,
        glccNetLCC=normalized_glcc["glccNetLCC"],
        glccSecondShift=normalized_glcc["glccSecondShift"],
        glccPrimaryShift=normalized_glcc["glccPrimaryShift"],
        harvest_matrix=normalized_glcc["harvest_matrix"],
        vegetnew_firstday=normalized_firstday,
        veget_cov_max_new=normalized_new_cover,
    )


def entry_normalization_availability_from_record(record: dict[str, object]) -> EntryNormalizationAvailability:
    """Report entry coverage for the normalization block before ``gpp_d``.

    ``totfrac_nobio`` supports current-year ``veget_cov`` and
    ``veget_cov_max``. ``totfrac_nobio_new`` supports ``vegetnew_firstday`` and
    ``veget_cov_max_new``. ``tot_bare_soil`` is not accepted for either role.
    """

    current_fields = ("veget", "veget_max", "totfrac_nobio")
    new_firstday_fields = ("vegetnew_firstday", "totfrac_nobio_new", "date")
    new_lcchange_fields = ("veget_max_new", "totfrac_nobio_new", "do_now_stomate_lcchange")
    missing_current = tuple(name for name in current_fields if name not in record)
    missing_new_firstday = tuple(name for name in new_firstday_fields if name not in record)
    missing_new_lcchange = tuple(name for name in new_lcchange_fields if name not in record)
    return EntryNormalizationAvailability(
        current_status="covered" if not missing_current else "partial",
        new_firstday_status="covered" if not missing_new_firstday else "partial",
        new_lcchange_status="covered" if not missing_new_lcchange else "partial",
        current_fields=current_fields,
        new_firstday_fields=new_firstday_fields,
        new_lcchange_fields=new_lcchange_fields,
        missing_for_current=missing_current,
        missing_for_new_firstday=missing_new_firstday,
        missing_for_new_lcchange=missing_new_lcchange,
    )


def entry_normalization_availability_with_supplied_totfrac_nobio(
    record: dict[str, object],
    *,
    totfrac_nobio,
) -> EntryNormalizationAvailability:
    """Report current-year normalization coverage with audited ``totfrac_nobio``."""

    augmented = dict(record)
    augmented["totfrac_nobio"] = totfrac_nobio
    return entry_normalization_availability_from_record(augmented)


def entry_normalization_availability_with_supplied_totfrac_nobio_new(
    record: dict[str, object],
    *,
    totfrac_nobio_new,
) -> EntryNormalizationAvailability:
    """Report next-year normalization coverage with audited ``totfrac_nobio_new``."""

    augmented = dict(record)
    augmented["totfrac_nobio_new"] = totfrac_nobio_new
    return entry_normalization_availability_from_record(augmented)


def daily_input_availability_from_entry(record: dict[str, object]) -> DailyInputAvailability:
    """Report which local daily increments can be built from an entry record.

    ``precip`` is closed by the bridge trace fields and ``dt_sechiba``.
    ``gpp_d`` remains partial for the current trace because ``totfrac_nobio`` is
    required by the audited Fortran formula but is not present in
    ``before_stomate_main``. A similarly named ``tot_bare_soil`` field must not
    satisfy this contract: Fortran computes it from bare soil and uncovered
    vegetation fractions after the STOMATE call path, while ``totfrac_nobio`` is
    the sum of non-biological surface fractions passed into ``stomate_main``.
    """

    precip_required = ("precip_rain", "precip_snow", "dt_sechiba")
    gpp_required = ("gpp", "veget_max", "dt_sechiba")
    missing_precip = tuple(name for name in precip_required if name not in record)
    missing_gpp_base = tuple(name for name in gpp_required if name not in record)
    missing_gpp = tuple(name for name in (*gpp_required, "totfrac_nobio") if name not in record)
    if missing_gpp_base:
        gpp_status = "missing"
    elif missing_gpp:
        gpp_status = "partial"
    else:
        gpp_status = "covered"
    return DailyInputAvailability(
        precip_status="covered" if not missing_precip else "missing",
        gpp_status=gpp_status,
        precip_fields=precip_required,
        gpp_fields=(*gpp_required, "totfrac_nobio"),
        missing_for_gpp=missing_gpp,
    )


def daily_input_availability_with_supplied_totfrac_nobio(
    record: dict[str, object],
    *,
    totfrac_nobio,
) -> DailyInputAvailability:
    """Report daily-input coverage when audited ``totfrac_nobio`` is supplied.

    This helper is for read-only static/restart coverage paths that already
    have a source-audited ``totfrac_nobio`` vector, for example the Phase 1C
    imposed-vegetation initialization in ``jax_orchidee.driver.init``. It does
    not derive ``totfrac_nobio`` from ``tot_bare_soil`` or any other nearby
    field.
    """

    augmented = dict(record)
    augmented["totfrac_nobio"] = totfrac_nobio
    return daily_input_availability_from_entry(augmented)
