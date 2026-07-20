"""STOMATE-facing readers for the 1961 fixed-format trace package.

These helpers preserve explicit trace values only. They do not infer missing
STOMATE process state or compute any carbon process.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jax_orchidee.stomate.carbon_kernels import NPARTS
from jax_orchidee.trace.fixed_format import FixedTraceTagSchema, parse_record, stream_records


STOMATE_TRACE_ROOT = Path("outputs/server_1961_trace_full_20260623/traces")

MAINT_AFTER_SCHEMA = FixedTraceTagSchema(
    tag="maint_after",
    fields=(
        "itime",
        "ik",
        "pft",
        "part",
        "do_slow",
        "dt_sechiba",
        "dt_stomate",
        "dt_days",
        "lai",
        "t2m",
        "t2m_longterm",
        "height",
        "sla_calc",
        "biomass",
        "resp_maint_part_radia",
        "resp_maint_part_after_accum",
        "resp_maint_radia_after_sum",
    ),
    provenance=(
        "outputs/server_1961_trace_full_20260623/traces/orchjax_stomate_maint_trace.txt:maint_after",
        "src_stomate/stomate.f90, stomate_main lines 3244-3267.",
        "src_stomate/stomate_resp.f90, maint_respiration lines 122-170, 203-232, 234-303, 319-376.",
    ),
)

AFTER_ALLOC_SCHEMA = FixedTraceTagSchema(
    tag="after_alloc",
    fields=(
        "ip",
        "pft",
        "part",
        "dt_days",
        "f_alloc",
        "biomass_after_alloc",
        "lai",
        "slai",
        "senescence",
        "rprof",
        "sla_calc",
    ),
    provenance=(
        "outputs/server_1961_trace_full_20260623/traces/orchjax_stomate_lpj_trace.txt:after_alloc",
        "src_stomate/stomate_lpj.f90, StomateLpj lines 1093-1102.",
        "src_stomate/stomate_alloc.f90, alloc lines 146-152, 187-201, 335-834.",
    ),
    notes=(
        "This trace exposes f_alloc and after-allocation biomass by part. It "
        "does not expose the full upstream allocation input state."
    ),
)


@dataclass(frozen=True)
class MaintAfterGroup:
    """One PFT's 12-part `maint_after` trace group."""

    itime: int
    ik: int
    pft: int
    rows: tuple[dict[str, object], ...]

    @property
    def resp_maint_part_radia(self) -> np.ndarray:
        return np.asarray([row["resp_maint_part_radia"] for row in self.rows], dtype=np.float64)

    @property
    def resp_maint_part_after_accum(self) -> np.ndarray:
        return np.asarray([row["resp_maint_part_after_accum"] for row in self.rows], dtype=np.float64)

    @property
    def resp_maint_radia_after_sum(self) -> float:
        totals = {float(row["resp_maint_radia_after_sum"]) for row in self.rows}
        if len(totals) != 1:
            raise ValueError("maint_after group has inconsistent summed respiration values")
        return totals.pop()


@dataclass(frozen=True)
class AfterAllocGroup:
    """One PFT's 12-part `after_alloc` trace group."""

    ip: int
    pft: int
    rows: tuple[dict[str, object], ...]

    @property
    def f_alloc(self) -> np.ndarray:
        return np.asarray([row["f_alloc"] for row in self.rows], dtype=np.float64)

    @property
    def biomass_after_alloc(self) -> np.ndarray:
        return np.asarray([row["biomass_after_alloc"] for row in self.rows], dtype=np.float64)


def _trace_path(filename: str, *, root: str | Path = STOMATE_TRACE_ROOT) -> Path:
    return Path(root) / filename


def read_first_pft14_maint_after_group(
    *,
    require_nonzero: bool = False,
    root: str | Path = STOMATE_TRACE_ROOT,
) -> MaintAfterGroup:
    """Read the first PFT14 `maint_after` group, optionally nonzero.

    Trace provenance: `orchjax_stomate_maint_trace.txt`, tag `maint_after`.
    Fortran provenance is recorded in `MAINT_AFTER_SCHEMA`. The function
    streams records and stops once one complete 12-part PFT14 group is found.
    """

    path = _trace_path("orchjax_stomate_maint_trace.txt", root=root)
    rows: list[dict[str, object]] = []
    current_key: tuple[int, int, int] | None = None

    for record in stream_records(path, tags="maint_after"):
        row = parse_record(record, MAINT_AFTER_SCHEMA)
        if row["pft"] != 14:
            continue
        key = (int(row["itime"]), int(row["ik"]), int(row["pft"]))
        if int(row["part"]) == 1 and len(rows) == NPARTS:
            group = MaintAfterGroup(*(current_key or key), rows=tuple(rows))
            if not require_nonzero or np.any(group.resp_maint_part_radia != 0.0):
                return group
            rows = []
        if current_key is None:
            current_key = key
        if key != current_key:
            if len(rows) == NPARTS:
                group = MaintAfterGroup(*current_key, rows=tuple(rows))
                if not require_nonzero or np.any(group.resp_maint_part_radia != 0.0):
                    return group
            rows = []
            current_key = key
        rows.append(row)

    if current_key is not None and len(rows) == NPARTS:
        group = MaintAfterGroup(*current_key, rows=tuple(rows))
        if not require_nonzero or np.any(group.resp_maint_part_radia != 0.0):
            return group
    raise ValueError("No complete PFT14 maint_after group found")


def read_first_pft14_after_alloc_group(*, root: str | Path = STOMATE_TRACE_ROOT) -> AfterAllocGroup:
    """Read the first PFT14 `after_alloc` group with explicit `f_alloc`.

    Trace provenance: `orchjax_stomate_lpj_trace.txt`, tag `after_alloc`.
    Fortran provenance is recorded in `AFTER_ALLOC_SCHEMA`. The trace exposes
    `f_alloc` as an explicit truth input for downstream NPP boundary tests, but
    does not close the full allocation process by itself.
    """

    path = _trace_path("orchjax_stomate_lpj_trace.txt", root=root)
    rows: list[dict[str, object]] = []
    current_key: tuple[int, int] | None = None

    for record in stream_records(path, tags="after_alloc"):
        row = parse_record(record, AFTER_ALLOC_SCHEMA)
        if row["pft"] != 14:
            continue
        key = (int(row["ip"]), int(row["pft"]))
        if int(row["part"]) == 1 and len(rows) == NPARTS:
            return AfterAllocGroup(*(current_key or key), rows=tuple(rows))
        if current_key is None:
            current_key = key
        if key != current_key:
            if len(rows) == NPARTS:
                return AfterAllocGroup(*current_key, rows=tuple(rows))
            rows = []
            current_key = key
        rows.append(row)

    if current_key is not None and len(rows) == NPARTS:
        return AfterAllocGroup(*current_key, rows=tuple(rows))
    raise ValueError("No complete PFT14 after_alloc group found")
