"""Paper-case single-landpoint mosaic discovery helpers.

The paper-scale spatial product is assembled from independent one-landpoint
case directories such as ``arg2_1.0/001.0-071.0/I10/S...``. These helpers only
describe that run protocol; they do not claim true ``nbp_glo > 1`` routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from jax_orchidee.driver.init import parse_run_def
from jax_orchidee.driver.run_def_materialization import materialize_case_run_def_values, write_materialized_run_def


_CASE_LABEL_RE = re.compile(r"^(?P<first>\d+(?:\.\d+)?)-(?P<second>\d+(?:\.\d+)?)$")
_RUNTIME_REQUIRED_RUN_DEF_KEYS = (
    "DT_SECHIBA",
    "DT_STOMATE",
    "STOMATE_OK_STOMATE",
    "STOMATE_OK_DGVM",
    "NVM",
    "NSTM",
    "LIMIT_WEST",
    "LIMIT_EAST",
    "LIMIT_SOUTH",
    "LIMIT_NORTH",
)


@dataclass(frozen=True)
class PaperMosaicCase:
    """One independent single-landpoint case in the paper mosaic.

    Run-protocol provenance: paper-era output directories under
    ``orc_calibrate_250919_sen/arg2_*/<case>/I*/S*/`` and
    ``fortran_run_scripts/paper_250919/Job0_bio`` lines 55-87, 141-162, and
    325-335, which materialize per-case ``run.def`` values before launching the
    model. Each case is treated as an independent one-landpoint run; spatial
    mosaicking happens outside ORCHIDEE routing.
    """

    root: Path
    arg2_label: str
    case_label: str
    iteration_label: str
    sample_label: str
    run_def_path: Path

    @property
    def case_dir(self) -> Path:
        return self.run_def_path.parent

    @property
    def key(self) -> str:
        return f"{self.arg2_label}/{self.case_label}/{self.iteration_label}/{self.sample_label}"

    @property
    def case_tokens(self) -> tuple[float, float]:
        match = _CASE_LABEL_RE.match(self.case_label)
        if match is None:
            raise ValueError(f"paper mosaic case label is not '<number>-<number>': {self.case_label}")
        return float(match.group("first")), float(match.group("second"))

    @property
    def independent_single_landpoint(self) -> bool:
        return True

    @property
    def routing_scope_note(self) -> str:
        return (
            "paper mosaic case is an independent single-landpoint run; true "
            "nbp_glo>1 routing remains a separate workflow"
        )

    def run_def_values(self) -> dict[str, str]:
        return parse_run_def(self.run_def_path)

    def history_path(self, year: int) -> Path:
        return self.case_dir / f"stomate_history_{int(year)}.nc"

    def start_file_path(self, filename: str) -> Path:
        return self.case_dir / filename

    @property
    def used_run_def_path(self) -> Path | None:
        path = self.case_dir / "z1" / "used_run.def"
        return path if path.exists() else None

    def missing_history_years(self, years: Iterable[int]) -> tuple[int, ...]:
        return tuple(int(year) for year in years if not self.history_path(int(year)).exists())


@dataclass(frozen=True)
class PaperMosaicCaseManifest:
    """Lightweight execution-planning row for one paper mosaic case.

    Run-protocol provenance: ``Job0_bio`` lines 55-58 writes the zoom limits,
    lines 325-335 writes per-year forcing and CO2 values into the run
    directory, and lines 344-365 archive yearly ``stomate_history`` files
    below each one-cell case directory. The final archived ``run.def`` can
    contain the last materialized year, so those scalar values are recorded as
    archived run-definition metadata, not as a year-specific manifest.
    """

    case: PaperMosaicCase
    limit_west: float | None
    limit_east: float | None
    limit_south: float | None
    limit_north: float | None
    archived_run_def_forcing_file: str | None
    archived_run_def_atm_co2: float | None
    missing_history_years: tuple[int, ...]
    first_history_lat_lon_shape: tuple[int, int] | None = None

    @property
    def key(self) -> str:
        return self.case.key

    @property
    def domain_limits(self) -> dict[str, float | None]:
        return {
            "west": self.limit_west,
            "east": self.limit_east,
            "south": self.limit_south,
            "north": self.limit_north,
        }

    @property
    def has_single_landpoint_history(self) -> bool | None:
        if self.first_history_lat_lon_shape is None:
            return None
        return self.first_history_lat_lon_shape == (1, 1)

    def to_json_row(self) -> dict[str, object]:
        return {
            "key": self.key,
            "run_def": str(self.case.run_def_path),
            "independent_single_landpoint": self.case.independent_single_landpoint,
            "domain_limits": self.domain_limits,
            "case_tokens": list(self.case.case_tokens),
            "archived_run_def_forcing_file": self.archived_run_def_forcing_file,
            "archived_run_def_atm_co2": self.archived_run_def_atm_co2,
            "missing_history_years": list(self.missing_history_years),
            "first_history_lat_lon_shape": (
                None if self.first_history_lat_lon_shape is None else list(self.first_history_lat_lon_shape)
            ),
            "has_single_landpoint_history": self.has_single_landpoint_history,
        }


@dataclass(frozen=True)
class PaperMosaicYearTask:
    """One year of one independent paper mosaic case.

    This is a scheduling record only. It deliberately keeps each paper case as
    an independent ``nbp_glo=1`` run task and points at the archived reference
    history file that can validate the corresponding JAX output.
    """

    manifest: PaperMosaicCaseManifest
    year: int

    @property
    def case_key(self) -> str:
        return self.manifest.key

    @property
    def task_key(self) -> str:
        return f"{self.case_key}/{int(self.year)}"

    @property
    def reference_history_path(self) -> Path:
        return self.manifest.case.history_path(int(self.year))

    @property
    def reference_history_exists(self) -> bool:
        return self.reference_history_path.exists()

    @property
    def driver_start_path(self) -> Path:
        return self.manifest.case.start_file_path("driver_start.nc")

    @property
    def sechiba_start_path(self) -> Path:
        return self.manifest.case.start_file_path("sechiba_start.nc")

    @property
    def stomate_start_path(self) -> Path:
        return self.manifest.case.start_file_path("stomate_start.nc")

    def to_json_row(self) -> dict[str, object]:
        return {
            "task_key": self.task_key,
            "case_key": self.case_key,
            "year": int(self.year),
            "run_def": str(self.manifest.case.run_def_path),
            "reference_history": str(self.reference_history_path),
            "reference_history_exists": self.reference_history_exists,
            "driver_start": str(self.driver_start_path),
            "sechiba_start": str(self.sechiba_start_path),
            "stomate_start": str(self.stomate_start_path),
            "independent_single_landpoint": self.manifest.case.independent_single_landpoint,
            "domain_limits": self.manifest.domain_limits,
        }


@dataclass(frozen=True)
class PaperMosaicRuntimeTask:
    """Runnable planning record for one independent case-year.

    This still does not run the model. It binds a case-year validation task to
    a complete materialized ``used_run.def`` and reports missing assets before
    a runner is allowed to call the driver.
    """

    year_task: PaperMosaicYearTask
    materialized_run_def_path: Path

    @property
    def task_key(self) -> str:
        return self.year_task.task_key

    @property
    def case_key(self) -> str:
        return self.year_task.case_key

    @property
    def reference_history_exists(self) -> bool:
        return self.year_task.reference_history_exists

    @property
    def missing_start_files(self) -> tuple[str, ...]:
        files = {
            "driver_start.nc": self.year_task.driver_start_path,
            "sechiba_start.nc": self.year_task.sechiba_start_path,
            "stomate_start.nc": self.year_task.stomate_start_path,
        }
        return tuple(name for name, path in files.items() if not path.exists())

    @property
    def materialized_run_def_exists(self) -> bool:
        return self.materialized_run_def_path.exists()

    def missing_run_def_keys(self, required_keys: Iterable[str] = _RUNTIME_REQUIRED_RUN_DEF_KEYS) -> tuple[str, ...]:
        if not self.materialized_run_def_exists:
            return tuple(str(key) for key in required_keys)
        values = parse_run_def(self.materialized_run_def_path)
        return tuple(str(key) for key in required_keys if str(key) not in values)

    def readiness_gaps(self) -> tuple[str, ...]:
        gaps: list[str] = []
        if not self.reference_history_exists:
            gaps.append("missing_reference_history")
        missing_start = self.missing_start_files
        if missing_start:
            gaps.append("missing_start_files:" + ",".join(missing_start))
        if not self.materialized_run_def_exists:
            gaps.append("missing_materialized_run_def")
        missing_keys = self.missing_run_def_keys()
        if missing_keys:
            gaps.append("missing_materialized_run_def_keys:" + ",".join(missing_keys))
        return tuple(gaps)

    @property
    def ready_for_runtime(self) -> bool:
        return not self.readiness_gaps()

    def to_json_row(self) -> dict[str, object]:
        gaps = self.readiness_gaps()
        return {
            **self.year_task.to_json_row(),
            "missing_start_files": list(self.missing_start_files),
            "materialized_run_def": str(self.materialized_run_def_path),
            "materialized_run_def_exists": self.materialized_run_def_exists,
            "ready_for_runtime": not gaps,
            "readiness_gaps": list(gaps),
        }


def _optional_float(values: dict[str, str], name: str) -> float | None:
    if name not in values:
        return None
    return float(values[name])


def _first_available_history_shape(case: PaperMosaicCase, years: tuple[int, ...]) -> tuple[int, int] | None:
    for year in years:
        path = case.history_path(year)
        if path.exists():
            return history_lat_lon_shape(path)
    return None


def paper_mosaic_case_manifest(
    case: PaperMosaicCase,
    *,
    years: Iterable[int] = (),
    check_history_shape: bool = False,
) -> PaperMosaicCaseManifest:
    """Build one manifest row from a materialized case ``run.def``."""

    values = case.run_def_values()
    years_tuple = tuple(int(year) for year in years)
    return PaperMosaicCaseManifest(
        case=case,
        limit_west=_optional_float(values, "LIMIT_WEST"),
        limit_east=_optional_float(values, "LIMIT_EAST"),
        limit_south=_optional_float(values, "LIMIT_SOUTH"),
        limit_north=_optional_float(values, "LIMIT_NORTH"),
        archived_run_def_forcing_file=values.get("FORCING_FILE"),
        archived_run_def_atm_co2=_optional_float(values, "ATM_CO2"),
        missing_history_years=case.missing_history_years(years_tuple),
        first_history_lat_lon_shape=(
            _first_available_history_shape(case, years_tuple) if check_history_shape else None
        ),
    )


def build_paper_mosaic_manifest(
    root: str | Path,
    *,
    years: Iterable[int] = (),
    check_history_shape: bool = False,
    max_cases: int | None = None,
) -> tuple[PaperMosaicCaseManifest, ...]:
    """Discover cases and return deterministic manifest rows."""

    cases = discover_paper_mosaic_cases(root)
    selected = cases if max_cases is None or int(max_cases) < 0 else cases[: int(max_cases)]
    return tuple(
        paper_mosaic_case_manifest(case, years=years, check_history_shape=check_history_shape)
        for case in selected
    )


def build_paper_mosaic_year_tasks(
    root: str | Path,
    *,
    years: Iterable[int],
    max_cases: int | None = None,
) -> tuple[PaperMosaicYearTask, ...]:
    """Return deterministic case-year tasks for the independent mosaic."""

    years_tuple = tuple(int(year) for year in years)
    manifests = build_paper_mosaic_manifest(root, years=years_tuple, max_cases=max_cases)
    return tuple(
        PaperMosaicYearTask(manifest=row, year=year)
        for row in manifests
        for year in years_tuple
    )


def build_paper_mosaic_runtime_tasks(
    root: str | Path,
    *,
    years: Iterable[int],
    materialized_run_def_root: str | Path,
    max_cases: int | None = None,
) -> tuple[PaperMosaicRuntimeTask, ...]:
    """Return case-year tasks with their materialized runtime run.def paths."""

    year_tasks = build_paper_mosaic_year_tasks(root, years=years, max_cases=max_cases)
    return tuple(
        PaperMosaicRuntimeTask(
            year_task=task,
            materialized_run_def_path=materialized_case_run_def_path(
                materialized_run_def_root,
                task.manifest.case,
            ),
        )
        for task in year_tasks
    )


def materialized_case_run_def_path(output_root: str | Path, case: PaperMosaicCase) -> Path:
    """Return the deterministic materialized ``used_run.def`` path for a case."""

    return (
        Path(output_root)
        / case.arg2_label
        / case.case_label
        / case.iteration_label
        / case.sample_label
        / "used_run.def"
    )


def materialize_paper_mosaic_case_run_defs(
    root: str | Path,
    *,
    base_used_run_def: str | Path,
    output_root: str | Path,
    max_cases: int | None = None,
    dry_run: bool = False,
) -> tuple[dict[str, object], ...]:
    """Materialize complete runtime run.def files for discovered mosaic cases."""

    cases = discover_paper_mosaic_cases(root)
    selected = cases if max_cases is None or int(max_cases) < 0 else cases[: int(max_cases)]
    rows: list[dict[str, object]] = []
    for case in selected:
        output = materialized_case_run_def_path(output_root, case)
        case_used_truth = case.used_run_def_path
        materialization_base = case_used_truth or Path(base_used_run_def)
        values = materialize_case_run_def_values(
            base_used_run_def=materialization_base,
            case_run_def=case.run_def_path,
            base_is_fortran_used_truth=case_used_truth is not None,
        )
        if not dry_run:
            write_materialized_run_def(
                values,
                output,
                header_lines=(
                    "# Materialized paper mosaic case run.def for ORCHIDEE-MAN-JAX.",
                    f"# Base materialized truth/defaults: {materialization_base}",
                    f"# Archived case submission run.def: {case.run_def_path}",
                    "# Year-dynamic forcing, CO2, and restart keys are intentionally excluded from case overrides.",
                ),
            )
        rows.append(
            {
                "key": case.key,
                "source_run_def": str(case.run_def_path),
                "materialized_run_def": str(output),
                "written": not dry_run,
                "materialized_key_count": len(values),
            }
        )
    return tuple(rows)


def summarize_paper_mosaic_manifest(
    rows: Iterable[PaperMosaicCaseManifest],
    *,
    discovered_cases: int,
    expected_landpoint_cases: int | None = None,
) -> dict[str, object]:
    """Summarize manifest rows without treating subset checks as full closure."""

    row_tuple = tuple(rows)
    missing_history = sum(1 for row in row_tuple if row.missing_history_years)
    non_single = [
        row.key
        for row in row_tuple
        if row.has_single_landpoint_history is False
    ]
    return {
        "expected_landpoint_cases": expected_landpoint_cases,
        "discovered_cases": int(discovered_cases),
        "manifest_rows": len(row_tuple),
        "missing_history_case_count_in_rows": missing_history,
        "non_single_history_cases_in_rows": non_single,
        "paper_target_scope": "669 independent single-landpoint cases",
        "branch_coverage_scope": (
            "changed forcing, landpoint state, or continuous parameters still "
            "require source-driven PFT14 branch audit"
        ),
    }


def summarize_paper_mosaic_year_tasks(tasks: Iterable[PaperMosaicYearTask]) -> dict[str, object]:
    """Summarize scheduled paper mosaic case-year validation tasks."""

    task_tuple = tuple(tasks)
    years = sorted({int(task.year) for task in task_tuple})
    cases = sorted({task.case_key for task in task_tuple})
    missing = [task.task_key for task in task_tuple if not task.reference_history_exists]
    return {
        "task_count": len(task_tuple),
        "case_count": len(cases),
        "years": years,
        "missing_reference_task_count": len(missing),
        "missing_reference_tasks": missing,
        "execution_unit": "independent_single_landpoint_year",
    }


def summarize_paper_mosaic_runtime_tasks(tasks: Iterable[PaperMosaicRuntimeTask]) -> dict[str, object]:
    """Summarize ready/missing assets for runtime case-year tasks."""

    task_tuple = tuple(tasks)
    missing_reference = [task.task_key for task in task_tuple if not task.reference_history_exists]
    missing_start_files = {
        task.task_key: list(task.missing_start_files)
        for task in task_tuple
        if task.missing_start_files
    }
    missing_run_def = [task.task_key for task in task_tuple if not task.materialized_run_def_exists]
    missing_keys = {
        task.task_key: list(task.missing_run_def_keys())
        for task in task_tuple
        if task.materialized_run_def_exists and task.missing_run_def_keys()
    }
    ready = [task.task_key for task in task_tuple if task.ready_for_runtime]
    return {
        "task_count": len(task_tuple),
        "ready_task_count": len(ready),
        "missing_reference_task_count": len(missing_reference),
        "missing_start_file_task_count": len(missing_start_files),
        "missing_materialized_run_def_task_count": len(missing_run_def),
        "missing_required_run_def_key_task_count": len(missing_keys),
        "missing_reference_tasks": missing_reference,
        "missing_start_files": missing_start_files,
        "missing_materialized_run_def_tasks": missing_run_def,
        "missing_required_run_def_keys": missing_keys,
        "execution_unit": "independent_single_landpoint_year_with_materialized_run_def",
    }


def discover_paper_mosaic_cases(root: str | Path) -> tuple[PaperMosaicCase, ...]:
    """Discover paper mosaic cases below an ``orc_calibrate_*`` output root.

    The discovery pattern is intentionally narrow:
    ``arg2_* / <case-label> / I* / S* / run.def``. It avoids recursing into
    per-year restart directories such as ``z1`` and refuses to infer missing
    run definitions.
    """

    root_path = Path(root)
    cases: list[PaperMosaicCase] = []
    for run_def in root_path.glob("arg2_*/*/I*/S*/run.def"):
        sample_dir = run_def.parent
        iteration_dir = sample_dir.parent
        case_dir = iteration_dir.parent
        arg2_dir = case_dir.parent
        if arg2_dir.parent != root_path:
            continue
        cases.append(
            PaperMosaicCase(
                root=root_path,
                arg2_label=arg2_dir.name,
                case_label=case_dir.name,
                iteration_label=iteration_dir.name,
                sample_label=sample_dir.name,
                run_def_path=run_def,
            )
        )
    return tuple(sorted(cases, key=lambda case: case.key))


def history_lat_lon_shape(path: str | Path) -> tuple[int, int]:
    """Return ``(lat, lon)`` sizes for a STOMATE history file."""

    import xarray as xr

    history = xr.open_dataset(path, decode_times=False)
    try:
        sizes = history.sizes
        if "lat" not in sizes or "lon" not in sizes:
            raise ValueError(f"{Path(path).name} is missing lat/lon dimensions")
        return int(sizes["lat"]), int(sizes["lon"])
    finally:
        history.close()


def history_is_single_landpoint(path: str | Path) -> bool:
    return history_lat_lon_shape(path) == (1, 1)
