"""Trace schema validation for STOMATE PFT14 process cuts.

The schema is a contract for future Fortran trace files. Validators here only
check supplied CSV headers or mapping-like data; they do not read project trace
outputs by default and do not infer process values.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


TRACE_COLUMN_GROUPS: dict[str, tuple[str, ...]] = {
    "gpp_accumulation": (
        "itime",
        "date",
        "day",
        "sec",
        "do_slow",
        "dt_sechiba",
        "dt_stomate",
        "dt_days",
        "gpp_d_before_accu",
        "gpp_daily_before_accu",
        "gpp_daily_after_accu",
    ),
    "maintenance_respiration": (
        "biomass_before_maint",
        "lai_before_maint",
        "lai_after_maint",
        "t2m",
        "t2m_longterm",
        "stempdiag",
        "rprof",
        "sla_calc",
        "resp_maint_part_radia_after",
        "resp_maint_radia_after_sum",
    ),
    "resp_maint_part_accumulation": (
        "resp_maint_part_before_accum",
        "resp_maint_part_radia_after",
        "resp_maint_part_after_accum",
    ),
    "alloc_input_output": (
        "biomass_before_alloc",
        "biomass_after_alloc",
        "lai_before_alloc",
        "senescence",
        "moiavail_week",
        "tsoil_month",
        "soilhum_month",
        "age",
        "leaf_age",
        "leaf_frac",
        "when_growthinit",
        "rprof",
        "sla_calc",
        "LtoLSR",
        "StoLSR",
        "RtoLSR",
        "alloc_sap_above",
        "f_alloc",
    ),
    "npp_calc_input_output": (
        "biomass_before_npp",
        "biomass_after_npp",
        "gpp_daily",
        "f_alloc",
        "resp_maint_part",
        "PFTpresent",
        "bm_alloc_after",
        "resp_maint_after",
        "resp_growth_after",
        "npp_daily_after",
        "leaf_age_before",
        "leaf_age_after",
        "leaf_frac_before",
        "leaf_frac_after",
        "age_before",
        "age_after",
    ),
    "biomass_agr_before_after": (
        "sap_m_ab_before",
        "sap_m_ab_after",
        "heart_m_ab_before",
        "heart_m_ab_after",
        "agr_sap_st_m_before",
        "agr_sap_st_m_after",
        "agr_sap_pn_m_before",
        "agr_sap_pn_m_after",
        "agr_hrt_st_m_before",
        "agr_hrt_st_m_after",
        "agr_hrt_pn_m_before",
        "agr_hrt_pn_m_after",
    ),
    "modelout_history": (
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
    ),
}

TRACE_SCHEMA_PROVENANCE: dict[str, str] = {
    "gpp_accumulation": (
        "src_stomate/stomate.f90, stomate_main lines 3198-3208; "
        "stomate_accu_r1d/r2d/r3d lines 9341-9411."
    ),
    "maintenance_respiration": (
        "src_stomate/stomate.f90, stomate_main lines 3244-3263; "
        "src_stomate/stomate_resp.f90, maint_respiration lines 122-170, "
        "203-232, 234-303, 319-376."
    ),
    "resp_maint_part_accumulation": "src_stomate/stomate.f90, stomate_main lines 3265-3267.",
    "alloc_input_output": (
        "src_stomate/stomate_lpj.f90, StomateLpj lines 1093-1102; "
        "src_stomate/stomate_alloc.f90, alloc lines 146-152, 187-201, 335-834."
    ),
    "npp_calc_input_output": (
        "src_stomate/stomate_lpj.f90, StomateLpj lines 1115-1131; "
        "src_stomate/stomate_npp.f90, npp_calc lines 116-180, 230-247, "
        "280-295, 299-386, 447-531."
    ),
    "biomass_agr_before_after": (
        "src_parameters/constantes_var.f90 pool indices lines 196-208; "
        "src_stomate/stomate_lpj.f90 history sends lines 1684-1699."
    ),
    "modelout_history": (
        "src_stomate/stomate_lpj.f90 history sends lines 1675-1707; "
        "paper modelout script lines 50, 60-76, 115-118."
    ),
}


@dataclass(frozen=True)
class TraceValidationResult:
    """Grouped trace-column validation result."""

    columns: tuple[str, ...]
    groups: tuple[str, ...]
    missing_by_group: dict[str, tuple[str, ...]]
    present_by_group: dict[str, tuple[str, ...]]
    unknown_columns: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return all(not missing for missing in self.missing_by_group.values())


def columns_from_trace(trace: object) -> tuple[str, ...]:
    """Extract columns from supplied CSV path, mapping, row mappings, or frame.

    Fortran provenance: schema-only validator for the process cuts documented
    in `docs/source_audits/stomate_daily_trace_contract.md`. This function is a
    validation boundary and does not interpret values.
    """

    if isinstance(trace, (str, Path)):
        path = Path(trace)
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                return tuple(next(reader))
            except StopIteration:
                return ()

    if hasattr(trace, "columns"):
        return tuple(str(column) for column in trace.columns)

    if isinstance(trace, Mapping):
        return tuple(str(key) for key in trace.keys())

    if isinstance(trace, Iterable):
        items = list(trace)
        if not items:
            return ()
        if all(isinstance(item, str) for item in items):
            return tuple(str(item) for item in items)
        columns: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, Mapping):
                raise TypeError("Iterable trace data must contain mappings or column strings")
            for key in item.keys():
                text = str(key)
                if text not in seen:
                    seen.add(text)
                    columns.append(text)
        return tuple(columns)

    raise TypeError("Trace must be a CSV path, mapping, iterable of mappings, column list, or object with columns")


def validate_trace_columns(
    trace: object,
    groups: tuple[str, ...] | None = None,
) -> TraceValidationResult:
    """Validate supplied trace columns against required schema groups.

    Fortran provenance is recorded per group in `TRACE_SCHEMA_PROVENANCE`.
    Missing columns are reported by group so the Fortran trace can be extended
    without guessing state in JAX.
    """

    columns = columns_from_trace(trace)
    column_set = set(columns)
    selected_groups = groups if groups is not None else tuple(TRACE_COLUMN_GROUPS)
    unknown_groups = [group for group in selected_groups if group not in TRACE_COLUMN_GROUPS]
    if unknown_groups:
        joined = ", ".join(unknown_groups)
        raise KeyError(f"Unknown trace schema groups: {joined}")

    missing_by_group = {}
    present_by_group = {}
    expected_columns: set[str] = set()
    for group in selected_groups:
        expected = TRACE_COLUMN_GROUPS[group]
        expected_columns.update(expected)
        missing_by_group[group] = tuple(column for column in expected if column not in column_set)
        present_by_group[group] = tuple(column for column in expected if column in column_set)

    unknown_columns = tuple(column for column in columns if column not in expected_columns)
    return TraceValidationResult(
        columns=columns,
        groups=tuple(selected_groups),
        missing_by_group=missing_by_group,
        present_by_group=present_by_group,
        unknown_columns=unknown_columns,
    )


def require_trace_columns(trace: object, groups: tuple[str, ...] | None = None) -> TraceValidationResult:
    """Validate trace columns and raise a grouped error if required columns are missing."""

    result = validate_trace_columns(trace, groups)
    if result.ok:
        return result
    parts = []
    for group, missing in result.missing_by_group.items():
        if missing:
            parts.append(f"{group}: {', '.join(missing)}")
    raise ValueError("Missing required trace columns by group: " + " | ".join(parts))
