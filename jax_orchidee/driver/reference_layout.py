"""Reference truth package discovery for paper-case landpoints.

This module is a file-layout resolver only. It does not interpret model
process state. The complete raw server copy under ``reference/OUT`` and
``reference/script`` is canonical. The older normalized and legacy layouts
remain read-only compatibility fallbacks.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile


PAPER_EXPERIMENT = "orc_calibrate_250919_sen"
PAPER_ARG = "arg2_1.0"
LEGACY_LANDPOINT_ID = "001.0-071.0"


@dataclass(frozen=True)
class ZipMember:
    """A reference asset that exists inside a local zip archive."""

    archive: Path
    member: str
    size: int


@dataclass(frozen=True)
class PaperLandpointReference:
    """Resolved local truth assets for one independent paper landpoint."""

    landpoint_id: str
    layout: str
    output_dir: Path | None
    iteration_id: str | None
    param_set: str | None
    histories: tuple[Path, ...]
    driver_start: Path | None
    sechiba_start: Path | None
    stomate_start: Path | None
    driver_restart: Path | None
    sechiba_restart: Path | None
    stomate_restart: Path | None
    run_def: Path | None
    used_run_def: Path | None
    modelout_csv: Path | None
    job_script: Path | None
    generated_run_def: Path | None
    modelout_csv_zip: ZipMember | None = None
    job_script_zip: ZipMember | None = None
    generated_run_def_zip: ZipMember | None = None

    @property
    def has_minimum_annual_truth(self) -> bool:
        """True when NetCDF history/restart and modelout CSV truth are present."""

        return (
            self.output_dir is not None
            and self.stomate_start is not None
            and self.stomate_restart is not None
            and bool(self.histories)
            and (self.modelout_csv is not None or self.modelout_csv_zip is not None)
        )

    @property
    def years_available(self) -> tuple[int, ...]:
        years: list[int] = []
        for path in self.histories:
            stem = path.stem
            try:
                years.append(int(stem.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return tuple(sorted(years))


def _first_existing(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _single_child(directory: Path, pattern: str = "*") -> Path | None:
    if not directory.exists():
        return None
    children = sorted(path for path in directory.glob(pattern) if path.is_dir())
    if len(children) == 1:
        return children[0]
    return None


@lru_cache(maxsize=8)
def _zip_members(archive: Path) -> dict[str, ZipMember]:
    if not archive.exists():
        return {}
    with ZipFile(archive) as zip_file:
        return {
            info.filename.replace("\\", "/"): ZipMember(
                archive=archive,
                member=info.filename,
                size=int(info.file_size),
            )
            for info in zip_file.infolist()
        }


def _zip_member(archive: Path, member_name: str) -> ZipMember | None:
    return _zip_members(archive).get(member_name.replace("\\", "/"))


def _raw_output_dir(reference_root: Path, landpoint_id: str) -> tuple[Path | None, str | None, str | None]:
    landpoint_root = (
        reference_root
        / "OUT"
        / PAPER_EXPERIMENT
        / PAPER_ARG
        / landpoint_id
    )
    iteration_dir = _single_child(landpoint_root, "I*")
    if iteration_dir is None:
        return None, None, None
    param_dir = _single_child(iteration_dir, "S*")
    if param_dir is None:
        return None, iteration_dir.name, None
    return param_dir, iteration_dir.name, param_dir.name


def _legacy_output_dir(reference_root: Path, landpoint_id: str) -> tuple[Path | None, str | None, str | None]:
    if landpoint_id != LEGACY_LANDPOINT_ID:
        return None, None, None
    landpoint_root = (
        reference_root
        / "case_001_071"
        / "OUT"
        / PAPER_EXPERIMENT
        / PAPER_ARG
        / landpoint_id
    )
    iteration_dir = _single_child(landpoint_root, "I*")
    if iteration_dir is None:
        return None, None, None
    param_dir = _single_child(iteration_dir, "S*")
    if param_dir is None:
        return None, iteration_dir.name, None
    return param_dir, iteration_dir.name, param_dir.name


def _normalized_output_dir(reference_root: Path, landpoint_id: str) -> tuple[Path | None, str | None, str | None]:
    landpoint_root = reference_root / "paper_250919" / "landpoints" / landpoint_id
    output_root = landpoint_root / "output"
    param_dir = _single_child(output_root, "S*")
    if param_dir is None:
        return None, None, None
    iteration_id = None
    metadata_path = landpoint_root / "metadata.csv"
    if metadata_path.exists():
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            metadata = {row["key"]: row["value"] for row in csv.DictReader(handle) if row.get("key")}
        iteration_id = metadata.get("iteration_id") or None
    return param_dir, iteration_id, param_dir.name


def _modelout_csv(reference_root: Path, landpoint_id: str) -> tuple[Path | None, ZipMember | None]:
    name = f"modelout_{landpoint_id}.csv"
    extracted = _first_existing(
        (
            reference_root / "script" / "orc_cali_250919" / "modelout_sen" / PAPER_ARG / name,
            reference_root / "paper_250919" / "landpoints" / landpoint_id / "modelout" / name,
            reference_root / "case_001_071" / "script" / "orc_cali_250919" / "modelout_sen" / PAPER_ARG / name,
        )
    )
    if extracted is not None:
        return extracted, None
    zipped = _zip_member(
        reference_root / "script" / "orc_cali_250919" / "modelout_sen" / f"{PAPER_ARG}.zip",
        f"{PAPER_ARG}/{name}",
    )
    return None, zipped


def _script_assets(
    reference_root: Path,
    landpoint_id: str,
    iteration_id: str | None,
    param_set: str | None,
) -> tuple[Path | None, Path | None, ZipMember | None, ZipMember | None]:
    if param_set is None:
        return None, None, None, None
    generated_run_name = "run.def_" + "_".join(param_set.split("_")[1:])
    normalized_job_script = reference_root / "paper_250919" / "landpoints" / landpoint_id / "run" / f"Job_{landpoint_id}.sh"
    normalized_run_def = reference_root / "paper_250919" / "landpoints" / landpoint_id / "run" / generated_run_name
    if iteration_id is None:
        return (
            normalized_job_script if normalized_job_script.exists() else None,
            normalized_run_def if normalized_run_def.exists() else None,
            None,
            None,
        )
    extracted_root = (
        reference_root
        / "script"
        / "orc_cali_250919"
        / "sen"
        / PAPER_ARG
        / "Job"
        / landpoint_id
        / iteration_id
        / param_set
    )
    job_script = _first_existing(
        (
            extracted_root / f"Job_{landpoint_id}.sh",
            normalized_job_script,
            reference_root
            / "case_001_071"
            / "script"
            / "orc_cali_250919"
            / "sen"
            / PAPER_ARG
            / "Job"
            / landpoint_id
            / iteration_id
            / param_set
            / f"Job_{landpoint_id}.sh",
        )
    )
    generated_run_def = _first_existing(
        (
            extracted_root / generated_run_name,
            normalized_run_def,
            reference_root
            / "case_001_071"
            / "script"
            / "orc_cali_250919"
            / "sen"
            / PAPER_ARG
            / "Job"
            / landpoint_id
            / iteration_id
            / param_set
            / generated_run_name,
        )
    )
    if job_script is not None and generated_run_def is not None:
        return job_script, generated_run_def, None, None

    archive = reference_root / "script" / "orc_cali_250919" / "sen" / f"{PAPER_ARG} (1).zip"
    zip_root = f"{PAPER_ARG}/Job/{landpoint_id}/{iteration_id}/{param_set}"
    return (
        job_script,
        generated_run_def,
        None if job_script is not None else _zip_member(archive, f"{zip_root}/Job_{landpoint_id}.sh"),
        None if generated_run_def is not None else _zip_member(archive, f"{zip_root}/{generated_run_name}"),
    )


def resolve_paper_landpoint_reference(
    root: str | Path,
    landpoint_id: str = LEGACY_LANDPOINT_ID,
) -> PaperLandpointReference:
    """Resolve available local reference assets for one paper landpoint."""

    root = Path(root)
    reference_root = root / "reference" if (root / "reference").exists() else root

    output_dir, iteration_id, param_set = _raw_output_dir(reference_root, landpoint_id)
    layout = "raw_server_copy"
    if output_dir is None:
        output_dir, iteration_id, param_set = _normalized_output_dir(reference_root, landpoint_id)
        layout = "normalized_compat"
    if output_dir is None:
        output_dir, iteration_id, param_set = _legacy_output_dir(reference_root, landpoint_id)
        layout = "legacy_case_001_071"

    histories: tuple[Path, ...] = ()
    if output_dir is not None:
        histories = tuple(sorted(output_dir.glob("stomate_history_*.nc")))
    modelout_path, modelout_zip = _modelout_csv(reference_root, landpoint_id)
    job_script, generated_run_def, job_script_zip, generated_run_def_zip = _script_assets(
        reference_root,
        landpoint_id,
        iteration_id,
        param_set,
    )

    def output_file(name: str) -> Path | None:
        if output_dir is None:
            return None
        path = output_dir / name
        return path if path.exists() else None

    if output_dir is None:
        layout = "modelout_only" if modelout_path is not None or modelout_zip is not None else "unresolved"

    return PaperLandpointReference(
        landpoint_id=landpoint_id,
        layout=layout,
        output_dir=output_dir,
        iteration_id=iteration_id,
        param_set=param_set,
        histories=histories,
        driver_start=output_file("driver_start.nc"),
        sechiba_start=output_file("sechiba_start.nc"),
        stomate_start=output_file("stomate_start.nc"),
        driver_restart=output_file("driver_restart.nc"),
        sechiba_restart=output_file("sechiba_restart.nc"),
        stomate_restart=output_file("stomate_restart.nc"),
        run_def=output_file("run.def"),
        used_run_def=output_file("z1/used_run.def"),
        modelout_csv=modelout_path,
        job_script=job_script,
        generated_run_def=generated_run_def,
        modelout_csv_zip=modelout_zip,
        job_script_zip=job_script_zip,
        generated_run_def_zip=generated_run_def_zip,
    )


def discover_paper_landpoint_ids(root: str | Path) -> tuple[str, ...]:
    """Return landpoint IDs present in local output/script assets."""

    root = Path(root)
    reference_root = root / "reference" if (root / "reference").exists() else root
    ids: set[str] = set()
    raw_root = reference_root / "OUT" / PAPER_EXPERIMENT / PAPER_ARG
    if raw_root.exists():
        ids.update(path.name for path in raw_root.iterdir() if path.is_dir())
    normalized_root = reference_root / "paper_250919" / "landpoints"
    if normalized_root.exists():
        ids.update(path.name for path in normalized_root.iterdir() if path.is_dir())
    legacy_root = reference_root / "case_001_071"
    if legacy_root.exists():
        ids.add(LEGACY_LANDPOINT_ID)

    extracted_modelout_root = reference_root / "script" / "orc_cali_250919" / "modelout_sen" / PAPER_ARG
    if extracted_modelout_root.exists():
        for path in extracted_modelout_root.glob("modelout_*.csv"):
            ids.add(path.name[len("modelout_") : -len(".csv")])

    modelout_zip = reference_root / "script" / "orc_cali_250919" / "modelout_sen" / f"{PAPER_ARG}.zip"
    if modelout_zip.exists():
        with ZipFile(modelout_zip) as zip_file:
            for info in zip_file.infolist():
                name = Path(info.filename.replace("\\", "/")).name
                if name.startswith("modelout_") and name.endswith(".csv"):
                    ids.add(name[len("modelout_") : -len(".csv")])
    return tuple(sorted(ids))


def inventory_paper_references(root: str | Path) -> tuple[PaperLandpointReference, ...]:
    """Resolve all landpoints discoverable from local reference assets."""

    return tuple(resolve_paper_landpoint_reference(root, landpoint_id) for landpoint_id in discover_paper_landpoint_ids(root))


def write_reference_manifest(root: str | Path, path: str | Path) -> Path:
    """Write a lightweight manifest for currently discoverable local assets."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = inventory_paper_references(root)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "landpoint_id",
                "layout",
                "iteration_id",
                "param_set",
                "output_dir",
                "history_count",
                "years_available",
                "used_run_def",
                "modelout_csv",
                "modelout_csv_zip",
                "job_script",
                "job_script_zip",
                "generated_run_def",
                "generated_run_def_zip",
                "has_minimum_annual_truth",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "landpoint_id": row.landpoint_id,
                    "layout": row.layout,
                    "iteration_id": row.iteration_id or "",
                    "param_set": row.param_set or "",
                    "output_dir": "" if row.output_dir is None else str(row.output_dir),
                    "history_count": len(row.histories),
                    "years_available": ",".join(str(year) for year in row.years_available),
                    "used_run_def": "" if row.used_run_def is None else str(row.used_run_def),
                    "modelout_csv": "" if row.modelout_csv is None else str(row.modelout_csv),
                    "modelout_csv_zip": "" if row.modelout_csv_zip is None else f"{row.modelout_csv_zip.archive}!{row.modelout_csv_zip.member}",
                    "job_script": "" if row.job_script is None else str(row.job_script),
                    "job_script_zip": "" if row.job_script_zip is None else f"{row.job_script_zip.archive}!{row.job_script_zip.member}",
                    "generated_run_def": "" if row.generated_run_def is None else str(row.generated_run_def),
                    "generated_run_def_zip": ""
                    if row.generated_run_def_zip is None
                    else f"{row.generated_run_def_zip.archive}!{row.generated_run_def_zip.member}",
                    "has_minimum_annual_truth": row.has_minimum_annual_truth,
                }
            )
    return output_path
