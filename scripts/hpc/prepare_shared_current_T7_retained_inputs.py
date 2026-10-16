"""CPU-only admission of external inputs consumed by the retained T7 tail."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

FORCING_SHA256 = "bf2e53a34b77e55badf4f4527c36a7c1c9da61636d9317b43288080ad28e790d"
CO2_SHA256 = "ad5e96976dd89d8e3ddf9cbc57293acdab8d07a33a8e016f3285055a3df02c31"
CONFIG_SHA256 = "3e3ec5c4518ebf4d470c60eda3e9bd6ddfd22e5c6dee473694f2409f17e619af"
CASE_SHA256 = "8c0ed07cd295a28a2f0ca0a9ae46b1904fc59328863aa4c89af9e6f83e820448"
STATE_SHA256 = "54d01d0f376b0ceed8a843740328dba9d61d2a8989507436bda1a0d16c624f5d"
RUNTIME_YEAR = 1991
MODEL_TSTEP = 48
FORCING_INDICES = ((145, 46),)
FORCING_LON_LAT = ((111.0, -3.0),)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_regular_file(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if not path.exists():
        raise FileNotFoundError(f"{label} is missing: {path}")
    resolved = path.resolve(strict=True)
    details = resolved.stat()
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    actual = sha256(resolved)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs: {resolved}")
    return {"path": str(resolved), "sha256": actual, "size_bytes": details.st_size}


def validate_data_root(data_root: str | Path) -> tuple[Path, dict[str, dict[str, object]]]:
    root = Path(data_root)
    if root.is_symlink():
        raise ValueError(f"ORCHIDEE_DATA_ROOT must not be a symlink: {root}")
    if not root.exists():
        raise FileNotFoundError(f"ORCHIDEE_DATA_ROOT is missing: {root}")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"ORCHIDEE_DATA_ROOT must be a directory: {root}")
    inputs = {
        "forcing_1991": _verified_regular_file(
            root / "forcing" / "cruncep_twodeg_1991.nc", FORCING_SHA256, "1991 forcing"
        ),
        "co2_annual": _verified_regular_file(
            root / "INPUTDIR_ZZ" / "global_co2_ann_1700_2017.txt", CO2_SHA256, "annual CO2"
        ),
    }
    return root, inputs


def _require_retained_identity(assets: dict[str, object]):
    manifest = assets["manifest_metadata"]
    math = manifest["saved_B1_mathematical_identity"]
    selection = manifest["sampling"]
    if (selection.get("artifact_id") != 299 or selection.get("window_index") != 0
            or selection.get("horizon") != 7 or selection.get("spatial_split") != "train"
            or selection.get("temporal_split") != "train"):
        raise ValueError("Exact saved artifact299/window0/T7 TRAIN sampling required")
    case_identity = assets["manifest_metadata"]["case_identity"]
    if (case_identity.get("case_sha256") != CASE_SHA256
            or case_identity.get("initial_state_sha256") != STATE_SHA256):
        raise ValueError("Saved artifact299 public case/state identity differs")
    saved_context = assets["context"]
    adapter = saved_context.get("template")
    owner = getattr(adapter, "static_owner", None)
    context = getattr(owner, "context", None)
    if owner is None or context is None:
        raise ValueError("Saved B1 context lacks the canonical retained-tail static owner")
    if (int(owner.runtime_year) != RUNTIME_YEAR
            or int(owner.lifecycle_start_tstep) != MODEL_TSTEP
            or int(owner.result_day_index) != 2):
        raise ValueError("Saved retained-tail year/day identity differs")
    domain = context.first_step_bundle.domain
    indices = tuple(tuple(map(int, row)) for row in np.asarray(domain.forcing_indices_zero_based))
    if indices != FORCING_INDICES or int(domain.nbindex) != 1:
        raise ValueError("Saved retained-tail forcing landpoint differs")
    lon = np.asarray(domain.lon, dtype=np.float64).reshape(-1)
    lat = np.asarray(domain.lat, dtype=np.float64).reshape(-1)
    actual_lon_lat = tuple((float(x), float(y)) for x, y in zip(lon, lat))
    if actual_lon_lat != FORCING_LON_LAT:
        raise ValueError("Saved retained-tail forcing coordinates differ")
    config_path = Path(context.config_path).resolve(strict=True)
    if sha256(config_path) != CONFIG_SHA256:
        raise ValueError(f"Saved retained-tail source config SHA-256 differs: {config_path}")
    return math, selection, owner, context, domain, config_path


def _field_summary(value) -> dict[str, object]:
    array = np.asarray(value)
    result: dict[str, object] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.dtype.kind in "fiu":
        result["finite"] = bool(np.all(np.isfinite(array)))
        result["sha256"] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return result


def _validated_numeric_summary(name: str, value, *, expected_shape=None, npts: int | None = None):
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"Retained input {name} must be numeric, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Retained input {name} contains nonfinite values")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"Retained input {name} shape {array.shape} differs from {expected_shape}")
    if npts is not None and (array.ndim == 0 or array.shape[0] != npts):
        raise ValueError(f"Retained input {name} first dimension must be {npts}, got {array.shape}")
    return _field_summary(array)


def consume_retained_inputs(assets: dict[str, object], data_root: str | Path) -> dict[str, object]:
    """Exercise the exact year-1991/tstep-48 forcing and retained static payload boundary."""
    math, selection, owner, context, domain_grid, config_path = _require_retained_identity(assets)
    root, input_files = validate_data_root(data_root)

    from jax_orchidee.driver import domain, orchestration as teacher

    previous_root = os.environ.get("ORCHIDEE_DATA_ROOT")
    os.environ["ORCHIDEE_DATA_ROOT"] = str(root)
    domain._read_forcing_land_cache.cache_clear()
    domain._read_annual_co2_cached.cache_clear()
    try:
        config = domain.load_case_config(config_path)
        resolved_forcing = domain.forcing_file_for_year(config, RUNTIME_YEAR).resolve(strict=True)
        resolved_co2 = Path(config["drivers"]["co2"]["file"]).resolve(strict=True)
        expected_forcing = Path(input_files["forcing_1991"]["path"])
        expected_co2 = Path(input_files["co2_annual"]["path"])
        if resolved_forcing != expected_forcing or resolved_co2 != expected_co2:
            raise ValueError("ORCHIDEE_DATA_ROOT does not resolve the retained inputs to verified files")

        bundle = teacher._paper_1961_step_bundle_from_context(
            context, year=RUNTIME_YEAR, tstep=MODEL_TSTEP
        )
        if (int(bundle.year) != RUNTIME_YEAR or int(bundle.tstep) != MODEL_TSTEP
                or not np.isfinite(float(bundle.co2_ppm))):
            raise ValueError("Retained forcing step/annual CO2 did not produce finite expected inputs")
        npts = int(bundle.domain.nbindex)
        if npts != 1:
            raise ValueError(f"Retained payload expects the single admitted land point, got {npts}")
        domain_fields = {
            "area": _validated_numeric_summary("domain.area", bundle.domain.area),
            "contfrac_land": _validated_numeric_summary(
                "domain.contfrac_land", bundle.domain.contfrac_land,
                expected_shape=(npts,),
            ),
        }
        if np.asarray(bundle.domain.area).size != npts:
            raise ValueError("Retained domain area does not match the admitted land-point count")
        forcing_fields = {
            name: _validated_numeric_summary(f"forcing.{name}", getattr(bundle.forcing, name))
            for name in (
                "Tair", "PSurf", "Qair", "Wind_E", "Wind_N", "Rainf", "Snowf",
                "SWdown", "LWdown", "Areas", "contfrac",
            )
        }
        if any(np.asarray(getattr(bundle.forcing, name)).size != npts for name in forcing_fields):
            raise ValueError("Retained forcing fields do not match the admitted land-point count")
        height_fields = {
            "Height_Lev1": float(bundle.forcing.Height_Lev1),
            "Height_Levuv": float(bundle.forcing.Height_Levuv),
        }
        if not all(np.isfinite(value) for value in height_fields.values()):
            raise ValueError("Retained forcing wind heights contain nonfinite values")
        canopy_summary = _validated_numeric_summary(
            "bundle.ccanopy", bundle.ccanopy, expected_shape=(npts,)
        )
        payload = teacher.build_intersurf_first_step_payload(
            bundle, driver_z0_for_wind=0.1, dt_sechiba=context.dt_sechiba
        )
        from jax_orchidee.driver.sechiba_boundary import IntersurfFirstStepPayload

        required_payload_fields = (
            "kindex", "lalo", "lon", "lat", "contfrac", "zlev", "zlevuv", "u", "v",
            "qair", "temp_air", "pb", "precip_rain", "precip_snow", "lwdown", "swdown",
            "ccanopy", "veget_max", "soiltile",
        )
        payload_domain = bundle.domain
        source_shapes = {
            "kindex": np.shape(payload_domain.kindex),
            "lalo": np.shape(payload_domain.lalo),
            "lon": np.shape(payload_domain.lon),
            "lat": np.shape(payload_domain.lat),
            "contfrac": np.shape(payload_domain.contfrac_land),
            "resolution": None if payload_domain.resolution is None else np.shape(payload_domain.resolution),
            "neighbours": None if payload_domain.neighbours is None else np.shape(payload_domain.neighbours),
            "corners": None if payload_domain.corners is None else np.shape(payload_domain.corners),
            "seglength": None if payload_domain.seglength is None else np.shape(payload_domain.seglength),
        }
        vector_fields = {
            "zlev", "zlevuv", "u", "v", "qair", "temp_air", "pb", "precip_rain",
            "precip_snow", "lwdown", "swdown", "ccanopy",
        }
        payload_fields = {}
        for field in fields(IntersurfFirstStepPayload):
            name = field.name
            if name == "missing_fields":
                continue
            value = getattr(payload, name)
            if name in {"kjpindex", "nbindex"}:
                payload_fields[name] = _validated_numeric_summary(
                    f"payload.{name}", value, expected_shape=(),
                )
                continue
            if name in required_payload_fields and value is None:
                raise ValueError(f"Required retained-tail payload field is missing: {name}")
            expected_shape = source_shapes.get(name)
            if name in source_shapes and expected_shape is None and value is not None:
                raise ValueError(f"Retained payload field {name} is present but its builder source is absent")
            if name in vector_fields:
                expected_shape = (npts,)
            payload_fields[name] = _validated_numeric_summary(
                f"payload.{name}", value, expected_shape=expected_shape,
                npts=None if name in source_shapes or name in vector_fields else npts,
            )
        if int(payload.kjpindex) != npts or int(payload.nbindex) != npts:
            raise ValueError("Retained static payload point counts differ from the admitted land point")
        payload_missing_fields = list(payload.missing_fields)
        return {
            "status": "retained_inputs_consumed",
            "scope": "CPU forcing read, year-specific CO2 read, and retained-tail static payload construction; no neural or physical transition",
            "sampling_identity": {
                "artifact_id": selection["artifact_id"],
                "window_index": selection["window_index"],
                "horizon": selection["horizon"],
                "landpoint_id": selection["landpoint_id"],
                "year": selection["year"],
                "day_indices": selection["day_indices"],
                "model_date_labels": selection["model_date_labels"],
                "case_sha256": assets["manifest_metadata"]["case_identity"]["case_sha256"],
                "initial_state_sha256": assets["manifest_metadata"]["case_identity"]["initial_state_sha256"],
            },
            "retained_template_identity": {
                "config_path": str(config_path),
                "config_sha256": CONFIG_SHA256,
                "runtime_year": RUNTIME_YEAR,
                "model_tstep": MODEL_TSTEP,
                "result_day_index": int(owner.result_day_index),
                "serialized_first_step_bundle_year": int(context.first_step_bundle.year),
                "forcing_indices_zero_based": [list(row) for row in FORCING_INDICES],
                "longitude_latitude": [list(row) for row in FORCING_LON_LAT],
                "domain_nbindex": int(domain_grid.nbindex),
            },
            "data_root": str(root),
            "inputs": input_files,
            "resolved_annual_co2_ppm": float(bundle.co2_ppm),
            "annual_co2_ppm_finite": True,
            "ccanopy": canopy_summary,
            "forcing_step_fields": forcing_fields,
            "forcing_heights_m": height_fields,
            "retained_domain_fields": domain_fields,
            "static_intersurf_payload_fields": payload_fields,
            "static_intersurf_payload_missing_fields": payload_missing_fields,
            "original_prepared_case_unchanged": True,
            "scientific_template_target_appropriateness_assessed": False,
            "historical_forcing_sha256_pin": None,
        }
    finally:
        domain._read_forcing_land_cache.cache_clear()
        domain._read_annual_co2_cached.cache_clear()
        if previous_root is None:
            os.environ.pop("ORCHIDEE_DATA_ROOT", None)
        else:
            os.environ["ORCHIDEE_DATA_ROOT"] = previous_root


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--parameters", required=True, type=Path)
    result.add_argument("--prepared-context", required=True, type=Path)
    result.add_argument("--training-prepared-manifest", required=True, type=Path)
    result.add_argument("--dataset-manifest", required=True, type=Path)
    result.add_argument("--saved-identity-receipt", required=True, type=Path)
    result.add_argument("--data-root", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError("preserve existing retained-input receipt")
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax

    jax.config.update("jax_enable_x64", True)
    from scripts.hpc.shared_current_T7_precision_assets import load_assets

    assets = load_assets(
        jax,
        parameters=args.parameters,
        prepared_context=args.prepared_context,
        training_prepared_manifest=args.training_prepared_manifest,
        dataset_manifest=args.dataset_manifest,
        saved_identity_receipt=args.saved_identity_receipt,
    )
    report = consume_retained_inputs(assets, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
