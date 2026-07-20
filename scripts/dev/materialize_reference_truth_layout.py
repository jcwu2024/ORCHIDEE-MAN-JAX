from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.reference_layout import (  # noqa: E402
    inventory_paper_references,
    resolve_paper_landpoint_reference,
    write_reference_manifest,
)


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _extract_member(archive: Path, member: str, dst: Path, *, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zip_file:
        with zip_file.open(member) as source, dst.open("wb") as target:
            shutil.copyfileobj(source, target)
    return True


def _materialize_landpoint(root: Path, landpoint_id: str, *, overwrite: bool) -> dict[str, object]:
    reference = resolve_paper_landpoint_reference(root, landpoint_id)
    row: dict[str, object] = {
        "landpoint_id": landpoint_id,
        "source_layout": reference.layout,
        "param_set": reference.param_set,
        "copied_files": 0,
        "extracted_files": 0,
        "skipped": False,
        "notes": "",
    }
    if reference.output_dir is None or reference.param_set is None:
        row["skipped"] = True
        row["notes"] = "missing raw or normalized output directory"
        return row

    target_root = root / "reference" / "paper_250919" / "landpoints" / landpoint_id
    target_output = target_root / "output" / reference.param_set
    for source in sorted(reference.output_dir.iterdir()):
        if source.is_file():
            if _copy_file(source, target_output / source.name, overwrite=overwrite):
                row["copied_files"] = int(row["copied_files"]) + 1

    modelout_target = target_root / "modelout" / f"modelout_{landpoint_id}.csv"
    if reference.modelout_csv is not None:
        if _copy_file(reference.modelout_csv, modelout_target, overwrite=overwrite):
            row["copied_files"] = int(row["copied_files"]) + 1
    elif reference.modelout_csv_zip is not None:
        if _extract_member(
            reference.modelout_csv_zip.archive,
            reference.modelout_csv_zip.member,
            modelout_target,
            overwrite=overwrite,
        ):
            row["extracted_files"] = int(row["extracted_files"]) + 1

    job_target = target_root / "run" / f"Job_{landpoint_id}.sh"
    if reference.job_script is not None:
        if _copy_file(reference.job_script, job_target, overwrite=overwrite):
            row["copied_files"] = int(row["copied_files"]) + 1
    elif reference.job_script_zip is not None:
        if _extract_member(reference.job_script_zip.archive, reference.job_script_zip.member, job_target, overwrite=overwrite):
            row["extracted_files"] = int(row["extracted_files"]) + 1

    generated_run_name = None
    if reference.generated_run_def is not None:
        generated_run_name = reference.generated_run_def.name
    elif reference.generated_run_def_zip is not None:
        generated_run_name = Path(reference.generated_run_def_zip.member).name
    if generated_run_name is not None:
        run_def_target = target_root / "run" / generated_run_name
        if reference.generated_run_def is not None:
            if _copy_file(reference.generated_run_def, run_def_target, overwrite=overwrite):
                row["copied_files"] = int(row["copied_files"]) + 1
        elif reference.generated_run_def_zip is not None:
            if _extract_member(
                reference.generated_run_def_zip.archive,
                reference.generated_run_def_zip.member,
                run_def_target,
                overwrite=overwrite,
            ):
                row["extracted_files"] = int(row["extracted_files"]) + 1

    metadata = target_root / "metadata.csv"
    if overwrite or not metadata.exists():
        metadata.parent.mkdir(parents=True, exist_ok=True)
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("key", "value"))
            writer.writeheader()
            writer.writerow({"key": "landpoint_id", "value": landpoint_id})
            writer.writerow({"key": "param_set", "value": reference.param_set})
            writer.writerow({"key": "iteration_id", "value": reference.iteration_id or ""})
            writer.writerow({"key": "source_output_dir", "value": str(reference.output_dir)})
        row["copied_files"] = int(row["copied_files"]) + 1
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize local paper reference truth into reference/paper_250919.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--landpoint-id", action="append", default=None)
    parser.add_argument("--complete-only", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "reference" / "paper_250919" / "manifest.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "reference_materialize_report.csv")
    args = parser.parse_args()

    if args.landpoint_id:
        landpoint_ids = tuple(args.landpoint_id)
    else:
        references = inventory_paper_references(args.root)
        if args.complete_only:
            references = tuple(reference for reference in references if reference.has_minimum_annual_truth)
        landpoint_ids = tuple(reference.landpoint_id for reference in references)

    rows = [_materialize_landpoint(args.root, landpoint_id, overwrite=args.overwrite) for landpoint_id in landpoint_ids]
    write_reference_manifest(args.root, args.manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("landpoint_id", "source_layout", "param_set", "copied_files", "extracted_files", "skipped", "notes"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"materialized={sum(1 for row in rows if not row['skipped'])} skipped={sum(1 for row in rows if row['skipped'])}")
    print(f"manifest={args.manifest}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
