"""HYDROL full-solve trace schema helpers.

The groups here describe future Fortran CSV instrumentation needed to unlock
full `hydrol_soil` parity. They validate headers only; they do not infer
missing HYDROL state or read traces unless a CSV path is explicitly supplied.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping


@dataclass(frozen=True)
class HydrolTraceGroup:
    """Named HYDROL trace group with required CSV columns."""

    name: str
    columns: tuple[str, ...]
    fortran_provenance: tuple[str, ...]
    consumer: str


@dataclass(frozen=True)
class HydrolTraceValidation:
    """Grouped HYDROL trace-column validation result."""

    available_columns: frozenset[str]
    missing_by_group: dict[str, tuple[str, ...]]

    @property
    def is_valid(self) -> bool:
        """Return whether all requested HYDROL trace columns are present.

        Fortran provenance: schema-only boundary for trace insertion spans in
        `fortran_source/ORCHIDEE/src_sechiba/hydrol.f90`; this property
        computes no process state.
        """

        return all(not missing for missing in self.missing_by_group.values())


@dataclass(frozen=True)
class HydrolListDirectedRecord:
    """One Fortran list-directed HYDROL trace record."""

    label: str
    values: Mapping[str, object]


@dataclass(frozen=True)
class HydrolFullStepTraceSlice:
    """Minimal first-match HYDROL full-step trace slice.

    The records come from the 2026-06-23 list-directed trace patch around
    `hydrol.f90` lines 5899-5996, 6007-6052, 6063-6075, and 6971-7044.
    """

    pre_layers: tuple[HydrolListDirectedRecord, ...]
    post_tile: HydrolListDirectedRecord
    post_layers: tuple[HydrolListDirectedRecord, ...]
    update_layers: tuple[HydrolListDirectedRecord, ...]
    alt_first_solve: HydrolListDirectedRecord | None
    alt_residual_layers: tuple[HydrolListDirectedRecord, ...]
    requires_trace: tuple[str, ...]

    def pre_columns(self) -> dict[str, tuple[object, ...]]:
        """Return pre-solve layer records as column tuples.

        Fortran provenance: values are emitted immediately before the main
        solve, after `hydrol_soil` lines 5899-5996. This helper reshapes trace
        records only; it computes no HYDROL process state.
        """

        return records_to_columns(self.pre_layers)

    def post_layer_columns(self) -> dict[str, tuple[object, ...]]:
        """Return post-tridiagonal layer records as column tuples.

        Fortran provenance: values are emitted after drainage correction around
        `hydrol_soil` lines 6007-6052 and before the total-moisture update.
        """

        return records_to_columns(self.post_layers)

    def update_columns(self) -> dict[str, tuple[object, ...]]:
        """Return post-update layer records as column tuples.

        Fortran provenance: values are emitted after `hydrol_soil` lines
        6063-6075, before over-saturation smoothing at line 6083.
        """

        return records_to_columns(self.update_layers)


@dataclass(frozen=True)
class HydrolAltResidualTraceSlice:
    """Minimal alternate residual-boundary trace slice."""

    alt_first_solve: HydrolListDirectedRecord
    alt_residual_layers: tuple[HydrolListDirectedRecord, ...]
    requires_trace: tuple[str, ...]

    def residual_columns(self) -> dict[str, tuple[object, ...]]:
        """Return residual-boundary layer records as column tuples.

        Fortran provenance: values are emitted after `hydrol_soil` line 7042,
        with trigger logic at lines 7011-7018 and boundary reset at lines
        7031-7039.
        """

        return records_to_columns(self.alt_residual_layers)


SETUP_COLUMNS = (
    "kjit",
    "ji",
    "jst",
    "jsl",
    "njsc",
    "mcr",
    "mcs",
    "z_m",
    "dz_mm",
    "mc_before_coef",
    "mcl_before_coef",
    "profil_froz_hydro_ns",
    "kfact_root",
    "mc_used",
    "bin_i",
    "a_raw",
    "b_raw",
    "d_raw",
    "k_floor_raw",
    "k_eval_raw",
    "a",
    "b",
    "d",
    "k",
    "free_drain_coef",
    "e",
    "f",
    "g1",
    "ep",
    "fp",
    "gp",
)

PRE_SOLVE_COLUMNS = (
    "kjit",
    "ji",
    "jst",
    "jsl",
    "mask_soiltile",
    "resolv",
    "mc_before_solve",
    "mcl_before_solve",
    "mclint",
    "tmci",
    "flux_top",
    "rootsink_by_layer",
    "rootsink_sum",
    "b",
    "rhs",
    "tmat_e",
    "tmat_f",
    "tmat_g1",
    "srhs",
    "stmat_e",
    "stmat_f",
    "stmat_g1",
)

SOLVE_COLUMNS = (
    "kjit",
    "ji",
    "jst",
    "jsl",
    "resolv",
    "rhs",
    "tmat_e",
    "tmat_f",
    "tmat_g1",
    "mcl_before_tridiag",
    "mcl_after_tridiag",
    "bet",
    "gam",
)

DRAINAGE_CONSERVATION_COLUMNS = (
    "kjit",
    "ji",
    "jst",
    "k_bottom",
    "free_drain_coef",
    "dt_days",
    "mask_soiltile",
    "resolv",
    "dr_ns_before_corr",
    "tmci",
    "tmcf",
    "flux_top",
    "rootsink_sum",
    "check_tr_ns",
    "dr_corrnum_ns",
    "dr_ns_after_corr",
)

POST_SOLVE_COLUMNS = (
    "kjit",
    "ji",
    "jst",
    "jsl",
    "mc_before_update",
    "mcl_after_tridiag",
    "profil_froz_hydro_ns",
    "mcr",
    "mcs",
    "mc_after_mcl_update",
    "mc_after_over_mcs2",
    "ru_corr_ns",
    "dr_corr_ns",
    "is_under_mcr",
    "check_under_ns",
    "mc_final",
    "mcl_final",
    "tmc_final",
)

ALTERNATE_PATH_COLUMNS = (
    "kjit",
    "ji",
    "jst",
    "jsl",
    "mcl_top_after_first_solve",
    "mcr",
    "flux_top",
    "min_sechiba",
    "resolv_alternate",
    "srhs",
    "stmat_e",
    "stmat_f",
    "stmat_g1",
    "rhs_residual",
    "tmat_residual_e",
    "tmat_residual_f",
    "tmat_residual_g1",
    "mcl_after_residual_solve",
    "mc_after_residual_update",
)

PRE_LIST_DIRECTED_COLUMNS = (
    "kjit",
    "ji",
    "jst",
    "jsl",
    "njsc",
    "resolv",
    "mask_soiltile",
    "mc",
    "mcl",
    "mclint",
    "profil_froz_hydro_ns",
    "mcr",
    "mcs",
    "a",
    "b",
    "d",
    "k",
    "e",
    "f",
    "g1",
    "ep",
    "fp",
    "gp",
    "rhs",
    "tmat_e",
    "tmat_f",
    "tmat_g1",
    "rootsink",
    "tmci",
    "flux_top",
    "free_drain_coef",
    "dt_days",
)

POST_TILE_LIST_DIRECTED_COLUMNS = (
    "kjit",
    "ji",
    "jst",
    "resolv",
    "tmci",
    "tmcf",
    "flux_top",
    "rootsink_sum",
    "dr_ns",
    "dr_corrnum_ns",
    "check_tr_ns",
    "k_bottom",
    "free_drain_coef",
    "dt_days",
)

POST_LAYER_LIST_DIRECTED_COLUMNS = (
    "kjit",
    "ji",
    "jst",
    "jsl",
    "mcl",
    "mc",
    "profil_froz_hydro_ns",
    "mcr",
    "mcs",
)

UPDATE_LIST_DIRECTED_COLUMNS = (
    "kjit",
    "ji",
    "jst",
    "jsl",
    "mc",
    "mcl",
    "profil_froz_hydro_ns",
    "mcr",
    "mcs",
)

ALT_FIRST_SOLVE_LIST_DIRECTED_COLUMNS = (
    "kjit",
    "ji",
    "jst",
    "mcl_top_after_first_solve",
    "flux_top",
    "mcr",
    "min_sechiba",
)

ALT_RESIDUAL_LIST_DIRECTED_COLUMNS = (
    "kjit",
    "ji",
    "jst",
    "jsl",
    "resolv",
    "rhs",
    "tmat_e",
    "tmat_f",
    "tmat_g1",
    "mcl_after_residual_solve",
)

HYDROL_LIST_DIRECTED_TRACE_SCHEMAS = {
    "pre": PRE_LIST_DIRECTED_COLUMNS,
    "post_tile": POST_TILE_LIST_DIRECTED_COLUMNS,
    "post_layer": POST_LAYER_LIST_DIRECTED_COLUMNS,
    "mc_after_update": UPDATE_LIST_DIRECTED_COLUMNS,
    "alt_first_solve": ALT_FIRST_SOLVE_LIST_DIRECTED_COLUMNS,
    "alt_residual": ALT_RESIDUAL_LIST_DIRECTED_COLUMNS,
}

_INTEGER_COLUMNS = frozenset(("kjit", "ji", "jst", "jsl", "njsc"))
_BOOLEAN_COLUMNS = frozenset(("resolv",))

FULL_STEP_TRACE_REQUIRES_TRACE = (
    "setup/source-table rebuild is blocked for this list-directed trace because "
    "`z_m`, `dz_mm`, `kfact_root`, `mc_used`, `bin_i`, `a_raw`, `b_raw`, "
    "`d_raw`, `k_floor_raw`, and `k_eval_raw` are not emitted",
    "tridiagonal forward-sweep `bet` and `gam` are not emitted; current parity "
    "compares `mcl_after_tridiag` from post-layer trace records",
    "`dr_ns_before_corr` is not emitted; current parity reconstructs it from "
    "`hydrol.f90` lines 6007-6016 and validates the emitted corrected `dr_ns`",
    "post-over-saturation and under-residual routing remains blocked because "
    "`ru_corr_ns`, `dr_corr_ns`, `check_over_ns`, `is_under_mcr`, and "
    "`check_under_ns` are not emitted",
    "alternate residual branch wiring remains partial because saved equations "
    "and first alternate per-layer solve outputs are not emitted",
)

ALT_RESIDUAL_REQUIRES_TRACE = (
    "inactive residual samples cannot validate all layers because the trace "
    "does not emit first-alternate-solve `mcl` for layers 2..nslm",
    "residual post-update `mc` cannot be checked because `mc_after_residual_update` "
    "is not emitted in the 2026-06-23 list-directed trace",
    "saved alternate equations `srhs/stmat_*` are not emitted, so residual "
    "boundary validation starts from the already-reset `rhs/tmat` records",
)

DEFAULT_FULL_STEP_TRACE_SAMPLES = (
    {"kjit": 1, "ji": 1, "jst": 4},
    {"kjit": 2, "ji": 1, "jst": 4},
)

DEFAULT_ALT_RESIDUAL_TRACE_SAMPLES = (
    {"jst": 4, "resolv": False},
    {"jst": 4, "resolv": True},
)

TRACE_GROUPS = (
    HydrolTraceGroup(
        name="setup",
        columns=SETUP_COLUMNS,
        fortran_provenance=(
            "hydrol.f90:4067-4069",
            "hydrol.f90:4150-4159",
            "hydrol.f90:4172-4244",
            "hydrol.f90:5931-5944",
            "hydrol.f90:8248-8280",
            "hydrol.f90:8480-8568",
        ),
        consumer="jax_orchidee.sechiba.hydrol coefficient/setup kernels",
    ),
    HydrolTraceGroup(
        name="pre_solve",
        columns=PRE_SOLVE_COLUMNS,
        fortran_provenance=(
            "hydrol.f90:5899-5944",
            "hydrol.f90:5915-5929",
            "hydrol.f90:5950-5996",
        ),
        consumer="jax_orchidee.sechiba.hydrol hydrol_soil_rhs_main",
    ),
    HydrolTraceGroup(
        name="solve",
        columns=SOLVE_COLUMNS,
        fortran_provenance=("hydrol.f90:6004", "hydrol.f90:8147-8198"),
        consumer="jax_orchidee.sechiba.hydrol hydrol_soil_tridiag_solve",
    ),
    HydrolTraceGroup(
        name="drainage_conservation",
        columns=DRAINAGE_CONSERVATION_COLUMNS,
        fortran_provenance=("hydrol.f90:6007-6052",),
        consumer="jax_orchidee.sechiba.hydrol drainage/check kernels",
    ),
    HydrolTraceGroup(
        name="post_solve",
        columns=POST_SOLVE_COLUMNS,
        fortran_provenance=(
            "hydrol.f90:6063-6160",
            "hydrol.f90:6300-6328",
            "hydrol.f90:7623-7770",
            "hydrol.f90:7935-8026",
        ),
        consumer="jax_orchidee.sechiba.hydrol post-solve kernels",
    ),
    HydrolTraceGroup(
        name="alternate_path",
        columns=ALTERNATE_PATH_COLUMNS,
        fortran_provenance=("hydrol.f90:6971-7044", "hydrol.f90:7044-7108"),
        consumer="jax_orchidee.sechiba.hydrol residual-boundary kernels",
    ),
)


def trace_group_names() -> tuple[str, ...]:
    """Return available HYDROL trace group names.

    Fortran provenance: schema groups mirror audited insertion spans in
    `hydrol.f90`; this function exposes metadata only.
    """

    return tuple(group.name for group in TRACE_GROUPS)


def trace_group(group_name: str) -> HydrolTraceGroup:
    """Return one HYDROL trace group definition by name.

    Fortran provenance: this metadata maps future trace files to audited
    `hydrol.f90` insertion spans and JAX consumers without computing state.
    """

    for group in TRACE_GROUPS:
        if group.name == group_name:
            return group
    raise KeyError(f"unknown HYDROL trace group: {group_name}")


def required_columns(group_name: str) -> tuple[str, ...]:
    """Return required columns for one HYDROL trace group.

    Fortran provenance: column groups correspond to the trace insertion spans
    listed in each `HydrolTraceGroup.fortran_provenance`; no HYDROL state is
    inferred here.
    """

    return trace_group(group_name).columns


def csv_header(path: str | Path) -> tuple[str, ...]:
    """Read the header columns from an explicitly supplied CSV path.

    Fortran provenance: reference/history reader validation boundary only;
    this helper reads no data rows and computes no HYDROL process values.
    """

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            return tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"CSV file has no header: {path}") from exc


def _columns_from_source(source: str | Path | Mapping[str, object] | Iterable[str]) -> tuple[str, ...]:
    if isinstance(source, Mapping):
        return tuple(str(key) for key in source.keys())
    if isinstance(source, (str, Path)):
        return csv_header(source)
    return tuple(str(item) for item in source)


def validate_columns(
    source: str | Path | Mapping[str, object] | Iterable[str],
    *,
    groups: Iterable[str] | None = None,
) -> HydrolTraceValidation:
    """Validate CSV/Mapping-like columns against HYDROL trace groups.

    Fortran provenance: schema-only validation for audited `hydrol.f90`
    insertion spans. The function reads a CSV header only when `source` is an
    explicit path and never invents missing columns.
    """

    available = frozenset(_columns_from_source(source))
    group_names = tuple(groups) if groups is not None else trace_group_names()
    missing_by_group: dict[str, tuple[str, ...]] = {}
    for group_name in group_names:
        group = trace_group(group_name)
        missing_by_group[group.name] = tuple(column for column in group.columns if column not in available)
    return HydrolTraceValidation(available_columns=available, missing_by_group=missing_by_group)


def _parse_list_directed_value(column: str, token: str) -> object:
    if column in _BOOLEAN_COLUMNS:
        if token == "T":
            return True
        if token == "F":
            return False
        raise ValueError(f"expected Fortran logical token for {column}, got {token!r}")
    if column in _INTEGER_COLUMNS:
        return int(token)
    return float(token.replace("D", "E").replace("d", "e"))


def _record_from_tokens(label: str, tokens: list[str]) -> HydrolListDirectedRecord:
    columns = HYDROL_LIST_DIRECTED_TRACE_SCHEMAS[label]
    if len(tokens) != len(columns):
        raise ValueError(f"{label} record has {len(tokens)} fields, expected {len(columns)}")
    return HydrolListDirectedRecord(
        label=label,
        values={column: _parse_list_directed_value(column, token) for column, token in zip(columns, tokens)},
    )


def iter_list_directed_records(
    path: str | Path,
    *,
    labels: Iterable[str] | None = None,
) -> Iterator[HydrolListDirectedRecord]:
    """Stream Fortran list-directed HYDROL trace records.

    Fortran provenance: this reader decodes diagnostic records emitted by the
    2026-06-23 patch around `hydrol.f90` lines 5899-5996, 6007-6052,
    6063-6075, and 6971-7044. It never infers missing process fields.
    """

    wanted = None if labels is None else frozenset(labels)
    current_label: str | None = None
    current_tokens: list[str] = []

    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            if parts[0] in HYDROL_LIST_DIRECTED_TRACE_SCHEMAS:
                if current_label is not None:
                    expected = len(HYDROL_LIST_DIRECTED_TRACE_SCHEMAS[current_label])
                    raise ValueError(
                        f"incomplete {current_label} record before line {line_number}: "
                        f"{len(current_tokens)} of {expected} fields"
                    )
                current_label = parts[0]
                current_tokens = parts[1:]
            elif current_label is not None:
                current_tokens.extend(parts)
            else:
                raise ValueError(f"continuation line without HYDROL trace label at {path}:{line_number}")

            if current_label is not None:
                expected = len(HYDROL_LIST_DIRECTED_TRACE_SCHEMAS[current_label])
                if len(current_tokens) > expected:
                    raise ValueError(
                        f"{current_label} record has extra fields at {path}:{line_number}: "
                        f"{len(current_tokens)} > {expected}"
                    )
                if len(current_tokens) == expected:
                    record = _record_from_tokens(current_label, current_tokens)
                    if wanted is None or record.label in wanted:
                        yield record
                    current_label = None
                    current_tokens = []

    if current_label is not None:
        expected = len(HYDROL_LIST_DIRECTED_TRACE_SCHEMAS[current_label])
        raise ValueError(
            f"incomplete {current_label} record at end of {path}: "
            f"{len(current_tokens)} of {expected} fields"
        )


def first_matching_records(
    path: str | Path,
    label: str,
    *,
    match: Mapping[str, object],
    count: int,
) -> tuple[HydrolListDirectedRecord, ...]:
    """Return the first `count` list-directed records matching exact fields.

    Fortran provenance: streaming trace reader only; no HYDROL process values
    are computed or guessed.
    """

    if label not in HYDROL_LIST_DIRECTED_TRACE_SCHEMAS:
        raise KeyError(f"unknown HYDROL list-directed trace label: {label}")
    records: list[HydrolListDirectedRecord] = []
    for record in iter_list_directed_records(path, labels=(label,)):
        if all(record.values.get(key) == value for key, value in match.items()):
            records.append(record)
            if len(records) == count:
                return tuple(records)
    raise ValueError(f"found {len(records)} {label} records matching {match}, expected {count}")


def records_to_columns(records: Iterable[HydrolListDirectedRecord]) -> dict[str, tuple[object, ...]]:
    """Stack homogeneous list-directed records into column tuples.

    Fortran provenance: trace reshaping only; this helper deliberately does not
    fill absent HYDROL fields.
    """

    rows = tuple(records)
    if not rows:
        return {}
    label = rows[0].label
    if any(record.label != label for record in rows):
        raise ValueError("records_to_columns requires records with one label")
    columns = HYDROL_LIST_DIRECTED_TRACE_SCHEMAS[label]
    return {column: tuple(record.values[column] for record in rows) for column in columns}


def discover_hydrol_full_step_sample_keys(
    trace_dir: str | Path,
    *,
    jst: int = 4,
    ji: int | None = 1,
    max_samples: int = 2,
) -> tuple[dict[str, int], ...]:
    """Stream first active HYDROL full-step profile keys for a tiny sample set.

    Fortran provenance: samples come from `pre` records emitted around
    `hydrol.f90` lines 5899-5996. The sampler only inspects first layer
    (`jsl=1`) active profiles, stops after `max_samples`, and leaves full
    profile loading to `load_first_hydrol_full_step_slice`.
    """

    if max_samples < 1:
        return ()

    trace_dir = Path(trace_dir)
    path = trace_dir / "orchjax_hydrol_main_trace.txt"
    keys: list[dict[str, int]] = []
    seen: set[tuple[int, int, int]] = set()

    for record in iter_list_directed_records(path, labels=("pre",)):
        values = record.values
        if values["jst"] != jst:
            continue
        if ji is not None and values["ji"] != ji:
            continue
        if values["jsl"] != 1 or values["resolv"] is not True:
            continue

        key_tuple = (int(values["kjit"]), int(values["ji"]), int(values["jst"]))
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        keys.append({"kjit": key_tuple[0], "ji": key_tuple[1], "jst": key_tuple[2]})
        if len(keys) == max_samples:
            return tuple(keys)

    raise ValueError(f"found {len(keys)} active HYDROL profile keys for jst={jst}, expected {max_samples}")


def load_first_hydrol_full_step_slice(
    trace_dir: str | Path,
    *,
    kjit: int = 1,
    ji: int = 1,
    jst: int = 4,
    nslm: int = 11,
) -> HydrolFullStepTraceSlice:
    """Load the first minimal HYDROL full-step trace slice by streaming files.

    Fortran provenance: combines the list-directed records emitted near
    `hydrol.f90` lines 5899-5996 (`pre`), 6007-6052 (`post_tile` and
    `post_layer`), 6063-6075 (`mc_after_update`), and 6971-7044
    (`alt_first_solve`/`alt_residual`). Missing fields are reported through
    `requires_trace` instead of inferred.
    """

    trace_dir = Path(trace_dir)
    layer_match = {"kjit": kjit, "ji": ji, "jst": jst}
    tile_match = {"kjit": kjit, "ji": ji, "jst": jst}

    pre_layers = first_matching_records(
        trace_dir / "orchjax_hydrol_main_trace.txt",
        "pre",
        match=layer_match,
        count=nslm,
    )
    post_tile = first_matching_records(
        trace_dir / "orchjax_hydrol_post_trace.txt",
        "post_tile",
        match=tile_match,
        count=1,
    )[0]
    post_layers = first_matching_records(
        trace_dir / "orchjax_hydrol_post_trace.txt",
        "post_layer",
        match=layer_match,
        count=nslm,
    )
    update_layers = first_matching_records(
        trace_dir / "orchjax_hydrol_update_trace.txt",
        "mc_after_update",
        match=layer_match,
        count=nslm,
    )

    try:
        alt_first_solve = first_matching_records(
            trace_dir / "orchjax_hydrol_alt_trace.txt",
            "alt_first_solve",
            match=tile_match,
            count=1,
        )[0]
    except FileNotFoundError:
        alt_first_solve = None

    try:
        alt_residual_layers = first_matching_records(
            trace_dir / "orchjax_hydrol_alt_residual_trace.txt",
            "alt_residual",
            match=layer_match,
            count=nslm,
        )
    except FileNotFoundError:
        alt_residual_layers = ()

    expected_layers = tuple(range(1, nslm + 1))
    for name, records in (
        ("pre", pre_layers),
        ("post_layer", post_layers),
        ("mc_after_update", update_layers),
        ("alt_residual", alt_residual_layers),
    ):
        if records and tuple(record.values["jsl"] for record in records) != expected_layers:
            raise ValueError(f"{name} records are not the first contiguous {nslm} HYDROL layers")

    return HydrolFullStepTraceSlice(
        pre_layers=pre_layers,
        post_tile=post_tile,
        post_layers=post_layers,
        update_layers=update_layers,
        alt_first_solve=alt_first_solve,
        alt_residual_layers=alt_residual_layers,
        requires_trace=FULL_STEP_TRACE_REQUIRES_TRACE,
    )


def load_hydrol_full_step_sample_slices(
    trace_dir: str | Path,
    *,
    samples: Iterable[Mapping[str, int]] | None = None,
    nslm: int = 11,
) -> tuple[HydrolFullStepTraceSlice, ...]:
    """Load a small configured set of HYDROL full-step trace slices.

    Fortran provenance: this is a sampler over the same list-directed records
    as `load_first_hydrol_full_step_slice`; it performs first-match streaming
    for each requested `(kjit, ji, jst)` and does not materialize full files.
    """

    if samples is None:
        samples = discover_hydrol_full_step_sample_keys(trace_dir)

    return tuple(
        load_first_hydrol_full_step_slice(
            trace_dir,
            kjit=int(sample["kjit"]),
            ji=int(sample["ji"]),
            jst=int(sample["jst"]),
            nslm=nslm,
        )
        for sample in samples
    )


def load_hydrol_alt_residual_sample_slices(
    trace_dir: str | Path,
    *,
    samples: Iterable[Mapping[str, object]] = DEFAULT_ALT_RESIDUAL_TRACE_SAMPLES,
    nslm: int = 11,
) -> tuple[HydrolAltResidualTraceSlice, ...]:
    """Load the tiny active/inactive alternate residual trace sample set.

    Fortran provenance: this wraps first-match streaming over
    `alt_first_solve` after `hydrol.f90` line 7005 and `alt_residual` after
    line 7042. The trace-emitted `resolv` value is checked against the source
    trigger at lines 7011-7018 by `hydrol_trace_backed_alt_residual`.
    """

    return tuple(
        load_first_alt_residual_slice(
            trace_dir,
            jst=int(sample.get("jst", 4)),
            resolv=sample.get("resolv"),  # type: ignore[arg-type]
            nslm=nslm,
        )
        for sample in samples
    )


def load_first_alt_residual_slice(
    trace_dir: str | Path,
    *,
    jst: int = 4,
    resolv: bool | None = None,
    nslm: int = 11,
) -> HydrolAltResidualTraceSlice:
    """Stream the first alternate residual-boundary profile matching filters.

    Fortran provenance: combines `alt_first_solve` after line 7005 with
    `alt_residual` after line 7042. The active/inactive `resolv` value comes
    from the trigger at `hydrol_soil` lines 7011-7018.
    """

    trace_dir = Path(trace_dir)
    path = trace_dir / "orchjax_hydrol_alt_residual_trace.txt"
    selected: list[HydrolListDirectedRecord] = []
    selected_key: tuple[int, int, int] | None = None

    for record in iter_list_directed_records(path, labels=("alt_residual",)):
        if record.values["jst"] != jst:
            continue
        if resolv is not None and record.values["resolv"] is not resolv:
            continue
        if record.values["jsl"] != 1:
            continue

        key = (int(record.values["kjit"]), int(record.values["ji"]), int(record.values["jst"]))
        selected_key = key
        selected = [record]
        break

    if selected_key is None:
        raise ValueError(f"no alt_residual profile found for jst={jst}, resolv={resolv}")

    if nslm > 1:
        for record in iter_list_directed_records(path, labels=("alt_residual",)):
            key = (int(record.values["kjit"]), int(record.values["ji"]), int(record.values["jst"]))
            if key == selected_key and int(record.values["jsl"]) > 1:
                selected.append(record)
                if len(selected) == nslm:
                    break

    if len(selected) != nslm:
        raise ValueError(f"found {len(selected)} alt_residual layers for {selected_key}, expected {nslm}")
    if tuple(record.values["jsl"] for record in selected) != tuple(range(1, nslm + 1)):
        raise ValueError(f"alt_residual records are not the first contiguous {nslm} HYDROL layers")

    kjit, ji, jst = selected_key
    alt_first = first_matching_records(
        trace_dir / "orchjax_hydrol_alt_trace.txt",
        "alt_first_solve",
        match={"kjit": kjit, "ji": ji, "jst": jst},
        count=1,
    )[0]
    return HydrolAltResidualTraceSlice(
        alt_first_solve=alt_first,
        alt_residual_layers=tuple(selected),
        requires_trace=ALT_RESIDUAL_REQUIRES_TRACE,
    )
