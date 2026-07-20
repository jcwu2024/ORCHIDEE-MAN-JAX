"""Stage 4 Batch D closure for the three thermal initialization owners."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.interpolation_aggregate import aggregate_2d  # noqa: E402
from jax_orchidee.driver.interpolation_core12 import (  # noqa: E402
    AggregatePacket,
    InterpolationTarget,
)
from jax_orchidee.sechiba.hydrol_thermosoil_completion import (  # noqa: E402
    read_refsoc_netcdf,
    read_refsocfile,
)
from jax_orchidee.sechiba.thermosoil import (  # noqa: E402
    QZ_USDA,
    SMCMAX_USDA,
    SO_CAPA_DRY_NS_USDA,
    thermosoil_cold_start_coef_closure,
    thermosoil_getdiff_thinsnow,
    thermosoil_initialize,
)
from scripts.dev.extract_fortran_micro_oracle import (  # noqa: E402
    ExtractedProcedure,
    extract_procedure_bytes,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    COMPILE_FLAGS,
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    float_comparison,
    write_point_comparisons,
    write_result,
)
from scripts.dev.fortran_gcov import (  # noqa: E402
    parse_gcov,
    parse_gcov_line_counts,
    select_arm_branch,
)

FAMILY = "stage4_batch_d_thermal"
THERMOSOIL_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
AGGREGATE_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/interpol_help.f90"
SOIL_CONSTANTS_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_parameters/constantes_soil_var.f90"
READER_TEMPLATE = ROOT / "scripts/dev/oracle_stage4_batch_d_thermal_reader.f90.template"
INITIALIZE_TEMPLATE = ROOT / "scripts/dev/oracle_stage4_batch_d_thermal_initialize.f90.template"
NETCDF_HELPER = ROOT / "scripts/dev/oracle_stage4_batch_d_thermal_netcdf.c"
OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
CONTRACT_CLASSES = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"
GCOV = Path(r"C:\msys64\ucrt64\bin\gcov.exe")
GCC = Path(r"C:\msys64\ucrt64\bin\gcc.exe")
GENDEF = Path(r"C:\msys64\ucrt64\bin\gendef.exe")
DLLTOOL = Path(r"C:\msys64\ucrt64\bin\dlltool.exe")
NETCDF_ROOT = Path(sys.prefix) / "Library"
NETCDF_DLL = NETCDF_ROOT / "bin/netcdf.dll"
NETCDF_INCLUDE = NETCDF_ROOT / "include"

OWNER_IDS = {
    "read_refsocfile": "pft14-owner-contract-da695e536e76",
    "thermosoil_var_init": "pft14-owner-contract-7cedec1972ee",
    "thermosoil_initialize": "pft14-owner-contract-d63791b5a816",
}

NPTS = 3
NLON = 7
NLAT = 7
NGRND = 4
NBNEIGHB = 8
READER_WIDTH = 32
READER_FLOAT_LAYOUT = (
    ("reader.refSOC", (NPTS, NGRND)),
    ("reader.longitude", (NLON,)),
    ("reader.latitude", (NLAT,)),
    ("reader.mask_lu", (NLON, NLAT)),
    ("reader.refSOC_file", (NLON, NLAT, NGRND)),
    ("reader.lon_rel", (NLON, NLAT)),
    ("reader.lat_rel", (NLON, NLAT)),
    ("reader.sub_area", (NPTS, READER_WIDTH)),
)
READER_DISCRETE_LAYOUT = (
    ("reader.mask", (NLON, NLAT)),
    ("reader.sub_index", (NPTS, READER_WIDTH, 2)),
    ("reader.nbvmax", (1,)),
    ("reader.attempt_count", (1,)),
    ("reader.attempts", (8,)),
)
NVM = 14
NSLM = 3
NSNOW = 3
NDEEP = 4
NSCM = 12
INITIALIZE_FLOAT_LAYOUT = (
    ("reftemp", (NPTS, NGRND)),
    ("refSOC", (NPTS, NGRND)),
    ("ptn", (NPTS, NGRND, NVM)),
    ("ptn_pftmean", (NPTS, NGRND)),
    ("dz1", (NGRND,)),
    ("z1", (NPTS,)),
    ("cgrnd", (NPTS, NGRND - 1, NVM)),
    ("dgrnd", (NPTS, NGRND - 1, NVM)),
    ("pcapa", (NPTS, NGRND, NVM)),
    ("pkappa", (NPTS, NGRND, NVM)),
    ("pcapa_en", (NPTS, NGRND, NVM)),
    ("pcapa_snow", (NPTS, NSNOW)),
    ("pkappa_snow", (NPTS, NSNOW)),
    ("ptn_beg", (NPTS, NGRND, NVM)),
    ("temp_sol_beg", (NPTS,)),
    ("shum_ngrnd_perma", (NPTS, NGRND, NVM)),
    ("shum_ngrnd_permalong", (NPTS, NGRND, NVM)),
    ("profil_froz", (NPTS, NGRND, NVM)),
    ("pcappa_supp", (NPTS, NGRND, NVM)),
    ("e_soil_lat", (NPTS, NVM)),
    ("dz5", (NGRND,)),
    ("mcs", (NSCM,)),
    ("SMCMAX", (NSCM,)),
    ("QZ", (NSCM,)),
    ("so_capa_dry_ns", (NSCM,)),
    ("mc_layt", (NPTS, NGRND)),
    ("mcl_layt", (NPTS, NGRND)),
    ("tmc_layt", (NPTS, NGRND)),
    ("mc_layt_pft", (NPTS, NGRND, NVM)),
    ("mcl_layt_pft", (NPTS, NGRND, NVM)),
    ("tmc_layt_pft", (NPTS, NGRND, NVM)),
    ("stempdiag", (NPTS, NSLM)),
    ("soilcap", (NPTS,)),
    ("soilcap_pft", (NPTS, NVM)),
    ("soilflx", (NPTS,)),
    ("soilflx_pft", (NPTS, NVM)),
    ("gtemp", (NPTS,)),
    ("lambda_snow", (NPTS,)),
    ("cgrnd_snow", (NPTS, NSNOW)),
    ("dgrnd_snow", (NPTS, NSNOW)),
    ("veget_max_bg", (NPTS, NVM)),
    ("veget_mask_real", (NPTS, NVM)),
    ("organic_layer_thick", (NPTS,)),
    ("lambda", (1,)),
)
INITIALIZE_DISCRETE_LAYOUT = (
    ("veget_mask_2d", (NPTS, NVM)),
    ("controls", (8,)),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _span_record(
    span: ExtractedProcedure, source: Path, extracted_file: Path
) -> dict[str, Any]:
    return {
        "source_file": source.relative_to(ROOT).as_posix(),
        "source_sha256": span.source_sha256,
        "start_line": span.start_line,
        "end_line": span.end_line,
        "span_sha256": span.span_sha256,
        "extracted_file": extracted_file.relative_to(ROOT).as_posix(),
        "byte_count": len(span.span_bytes),
    }


def _instrument_reader(span: ExtractedProcedure) -> bytes:
    marker = b"  DEALLOCATE (lat_lu)"
    if span.span_bytes.count(marker) != 1:
        raise ValueError("read_refSOCfile deallocation marker drifted")
    capture = b"""  ! Oracle capture: observation only; calculation above is unchanged.\n  oracle_nbvmax = nbvmax\n  allocate(oracle_lon_lu(iml),oracle_lat_lu(jml))\n  allocate(oracle_mask_lu(iml,jml),oracle_refSOC_file(iml,jml,lml))\n  allocate(oracle_lon_rel(iml,jml),oracle_lat_rel(iml,jml))\n  allocate(oracle_mask(iml,jml),oracle_sub_area(nbpt,nbvmax))\n  allocate(oracle_sub_index(nbpt,nbvmax,2))\n  oracle_lon_lu=lon_lu; oracle_lat_lu=lat_lu\n  oracle_mask_lu=mask_lu; oracle_refSOC_file=refSOC_file\n  oracle_lon_rel=lon_rel; oracle_lat_rel=lat_rel; oracle_mask=mask\n  oracle_sub_area=sub_area; oracle_sub_index=sub_index\n\n"""
    return span.span_bytes.replace(marker, capture + marker)


def compose_reader(source_path: Path, extracted_dir: Path) -> dict[str, Any]:
    reader = extract_procedure_bytes(THERMOSOIL_SOURCE, "read_refSOCfile")
    aggregate = extract_procedure_bytes(AGGREGATE_SOURCE, "aggregate_2d")
    extracted_dir.mkdir(parents=True, exist_ok=True)
    reader_file = extracted_dir / "read_refSOCfile.f90"
    aggregate_file = extracted_dir / "aggregate_2d.f90"
    reader_file.write_bytes(reader.span_bytes)
    aggregate_file.write_bytes(aggregate.span_bytes)
    generated = READER_TEMPLATE.read_bytes()
    generated = generated.replace(b"! <AGGREGATE_2D>", aggregate.span_bytes)
    generated = generated.replace(b"! <READ_REFSOCFILE>", _instrument_reader(reader))
    if b"! <" in generated:
        raise ValueError("unreplaced reader template marker")
    source_path.write_bytes(generated)
    return {
        "read_refSOCfile": _span_record(reader, THERMOSOIL_SOURCE, reader_file),
        "aggregate_2d": _span_record(aggregate, AGGREGATE_SOURCE, aggregate_file),
        "instrumented_read_refSOCfile_sha256": hashlib.sha256(
            _instrument_reader(reader)
        ).hexdigest(),
    }


def _source_fragment(source: Path, start: int, end: int) -> bytes:
    lines = source.read_bytes().splitlines(keepends=True)
    return b"".join(lines[start - 1 : end])


def _instrument_initialize(span: ExtractedProcedure) -> bytes:
    marker = b"  END SUBROUTINE thermosoil_initialize"
    if span.span_bytes.count(marker) != 1:
        raise ValueError("thermosoil_initialize terminator drifted")
    capture = b"""    ! Oracle capture: observation only; calculation above is unchanged.\n    oracle_calculate_coef=calculate_coef\n    oracle_ok_zimov=ok_zimov\n    oracle_veget_max_bg=veget_max_bg\n    oracle_veget_mask_real=veget_mask_real\n\n"""
    return span.span_bytes.replace(marker, capture + marker)


def compose_initialize(source_path: Path, extracted_dir: Path) -> dict[str, Any]:
    procedures = (
        "thermosoil_initialize",
        "thermosoil_var_init",
        "thermosoil_humlev",
        "thermosoil_cond_pft",
        "thermosoil_cond_nopft",
        "thermosoil_cond",
        "thermosoil_getdiff",
        "thermosoil_getdiff_thinsnow",
        "thermosoil_diaglev",
        "thermosoil_coef",
    )
    spans = {
        procedure: extract_procedure_bytes(THERMOSOIL_SOURCE, procedure)
        for procedure in procedures
    }
    table_spans = {
        "fao_mcs": (200, 201),
        "fao_thermal_tables": (231, 238),
        "usda_mcs": (265, 268),
        "usda_thermal_tables": (330, 343),
    }
    extracted_dir.mkdir(parents=True, exist_ok=True)
    generated = INITIALIZE_TEMPLATE.read_bytes()
    records: dict[str, Any] = {}
    for procedure, span in spans.items():
        extracted_file = extracted_dir / f"{procedure}.f90"
        extracted_file.write_bytes(span.span_bytes)
        payload = (
            _instrument_initialize(span)
            if procedure == "thermosoil_initialize"
            else span.span_bytes
        )
        marker = f"! <{procedure.upper()}>".encode("ascii")
        if generated.count(marker) != 1:
            raise ValueError(f"expected one initialize marker for {procedure}")
        generated = generated.replace(marker, payload)
        records[procedure] = _span_record(
            span, THERMOSOIL_SOURCE, extracted_file
        )
    for name, (start, end) in table_spans.items():
        payload = _source_fragment(SOIL_CONSTANTS_SOURCE, start, end)
        extracted_file = extracted_dir / f"{name}.f90"
        extracted_file.write_bytes(payload)
        marker = f"! <{name.upper()}>".encode("ascii")
        if generated.count(marker) != 1:
            raise ValueError(f"expected one table marker for {name}")
        generated = generated.replace(marker, payload)
        records[name] = {
            "source_file": SOIL_CONSTANTS_SOURCE.relative_to(ROOT).as_posix(),
            "source_sha256": _sha256(SOIL_CONSTANTS_SOURCE),
            "start_line": start,
            "end_line": end,
            "span_sha256": hashlib.sha256(payload).hexdigest(),
            "extracted_file": extracted_file.relative_to(ROOT).as_posix(),
            "byte_count": len(payload),
        }
    if b"! <" in generated:
        raise ValueError("unreplaced initialize template marker")
    source_path.write_bytes(generated)
    records["instrumented_thermosoil_initialize_sha256"] = hashlib.sha256(
        _instrument_initialize(spans["thermosoil_initialize"])
    ).hexdigest()
    return records


def _netcdf_environment(compiler: Path) -> dict[str, str]:
    environment = compiler_environment(compiler)
    environment["PATH"] = (
        str(compiler.parent)
        + os.pathsep
        + str(NETCDF_DLL.parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    return environment


def compile_reader(
    source: Path, executable: Path, compiler: Path, *, coverage: bool
) -> dict[str, Any]:
    required = (GCC, GENDEF, DLLTOOL, NETCDF_DLL, NETCDF_HELPER)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"reader NetCDF toolchain is incomplete: {missing}")
    subprocess.run(
        [str(GENDEF), str(NETCDF_DLL)],
        cwd=source.parent,
        env=_netcdf_environment(compiler),
        check=True,
        capture_output=True,
        text=True,
    )
    definition = source.parent / "netcdf.def"
    import_library = source.parent / "libnetcdf.dll.a"
    subprocess.run(
        [str(DLLTOOL), "-d", str(definition), "-l", str(import_library)],
        cwd=source.parent,
        env=_netcdf_environment(compiler),
        check=True,
        capture_output=True,
        text=True,
    )
    helper_object = source.parent / "thermal_netcdf.o"
    subprocess.run(
        [
            str(GCC),
            "-std=c11",
            "-O0",
            "-Wall",
            "-Wextra",
            f"-I{NETCDF_INCLUDE}",
            "-c",
            str(NETCDF_HELPER),
            "-o",
            str(helper_object),
        ],
        cwd=source.parent,
        env=_netcdf_environment(compiler),
        check=True,
        capture_output=True,
        text=True,
    )
    base_flags = tuple(
        flag
        for flag in COMPILE_FLAGS
        if not flag.startswith("-ffpe-trap")
        and not (coverage and flag == "-fcheck=all")
    )
    metadata = compile_fortran(
        source,
        executable,
        compiler,
        extra_flags=(
            "-cpp",
            *(("--coverage",) if coverage else ()),
            str(helper_object),
            f"-L{source.parent}",
            "-lnetcdf",
        ),
        base_flags=base_flags,
    )
    return {
        **metadata,
        "netcdf_library": str(NETCDF_DLL),
        "netcdf_library_sha256": _sha256(NETCDF_DLL),
        "netcdf_helper_sha256": _sha256(NETCDF_HELPER),
    }


def compile_initialize(
    source: Path, executable: Path, compiler: Path, *, coverage: bool
) -> dict[str, Any]:
    base_flags = tuple(
        "-std=gnu" if flag == "-std=f2008" else flag
        for flag in COMPILE_FLAGS
        if not (coverage and flag == "-fcheck=all")
    )
    return compile_fortran(
        source,
        executable,
        compiler,
        extra_flags=(("--coverage",) if coverage else ()),
        base_flags=base_flags,
    )


def _reader_fixture(path: Path) -> dict[str, np.ndarray]:
    longitude = np.linspace(-3.0, 3.0, NLON, dtype=np.float64)
    latitude = np.linspace(-3.0, 3.0, NLAT, dtype=np.float64)
    level = np.asarray([0.05, 0.2, 0.8, 2.0], dtype=np.float64)
    mask = np.full((NLON, NLAT), 2.0, dtype=np.float64)
    mask[0, 0] = 0.0
    mask[0, -1] = -1.0
    mask[-1, 0] = 0.0
    mask[-1, -1] = -2.0
    i, j, k = np.indices((NLON, NLAT, NGRND))
    refsoc = (100.0 * (k + 1) + 10.0 * (j + 1) + (i + 1)).astype(np.float64)
    dataset = xr.Dataset(
        data_vars={
            "mask": (("longitude", "latitude"), mask),
            "soil_organic_carbon": (
                ("longitude", "latitude", "level"),
                refsoc,
            ),
        },
        coords={
            "longitude": longitude,
            "latitude": latitude,
            "level": level,
        },
    )
    dataset.to_netcdf(path, engine="scipy", format="NETCDF3_64BIT")
    return {
        "longitude": longitude,
        "latitude": latitude,
        "mask": mask,
        "soil_organic_carbon": refsoc,
    }


def _reader_geometry() -> dict[str, np.ndarray]:
    return {
        "lalo": np.asarray([[0.0, 0.0], [50.0, 50.0], [0.0, 0.0]], dtype=np.float64),
        "resolution": np.asarray(
            [[1000000.0, 1000000.0], [10000.0, 10000.0], [100000.0, 100000.0]],
            dtype=np.float64,
        ),
        "neighbours": np.zeros((NPTS, NBNEIGHB), dtype=np.int32),
        "contfrac": np.asarray([1.0, 0.5, 0.75], dtype=np.float64),
    }


def _write_stream(path: Path, fields: tuple[np.ndarray, ...]) -> None:
    with path.open("wb") as handle:
        for field in fields:
            np.asfortranarray(field).ravel(order="F").tofile(handle)


def _read_stream(
    path: Path, layout: tuple[tuple[str, tuple[int, ...]], ...], dtype: np.dtype
) -> dict[str, np.ndarray]:
    raw = np.fromfile(path, dtype=dtype)
    values: dict[str, np.ndarray] = {}
    offset = 0
    for name, shape in layout:
        count = int(np.prod(shape))
        values[name] = raw[offset : offset + count].reshape(shape, order="F")
        offset += count
    if offset != raw.size:
        raise ValueError(f"{path.name}: consumed {offset} values, found {raw.size}")
    return values


def _reader_jax_expected(
    netcdf_path: Path, geometry: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    target = InterpolationTarget(
        geometry["lalo"],
        geometry["resolution"],
        geometry["neighbours"],
        geometry["contfrac"],
    )
    packets: list[tuple[Any, AggregatePacket]] = []

    def aggregate(request):
        result = aggregate_2d(
            request.lalo.shape[0],
            request.lalo,
            request.neighbours,
            request.resolution,
            request.contfrac,
            request.longitude.shape[0],
            request.longitude.shape[1],
            request.longitude,
            request.latitude,
            request.mask,
            request.callsign,
            request.nbvmax,
        )
        packet = AggregatePacket(result.indinc, result.areaoverlap, result.ok)
        packets.append((request, packet))
        return packet

    result = read_refsocfile(
        target=target,
        aggregate=aggregate,
        path=netcdf_path,
        reader=read_refsoc_netcdf,
    )
    if result.nbvmax_attempts != (16, 32):
        raise RuntimeError(
            f"deterministic reader fixture did not exercise retry: {result.nbvmax_attempts}"
        )
    request, packet = packets[-1]
    float_values = {
        "reader.refSOC": np.asarray(result.refsoc),
        "reader.longitude": np.asarray(result.source.longitude),
        "reader.latitude": np.asarray(result.source.latitude),
        "reader.mask_lu": np.asarray(result.source.mask),
        "reader.refSOC_file": np.asarray(result.source.soil_organic_carbon),
        "reader.lon_rel": np.asarray(request.longitude),
        "reader.lat_rel": np.asarray(request.latitude),
        "reader.sub_area": np.asarray(packet.sub_area),
    }
    attempts = np.zeros(8, dtype=np.int32)
    attempts[: len(result.nbvmax_attempts)] = result.nbvmax_attempts
    discrete_values = {
        "reader.mask": np.asarray(request.mask, dtype=np.int32),
        "reader.sub_index": np.asarray(packet.sub_index, dtype=np.int32),
        "reader.nbvmax": np.asarray([request.nbvmax], dtype=np.int32),
        "reader.attempt_count": np.asarray([len(result.nbvmax_attempts)], dtype=np.int32),
        "reader.attempts": attempts,
    }
    return float_values, discrete_values


def run_reader_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    netcdf_path = output_dir / "minimal_refsoc.nc"
    fixture = _reader_fixture(netcdf_path)
    geometry = _reader_geometry()
    np.savez(
        output_dir / "reader_inputs.npz",
        **fixture,
        **{f"target_{name}": value for name, value in geometry.items()},
    )
    expected_float, expected_discrete = _reader_jax_expected(netcdf_path, geometry)
    with tempfile.TemporaryDirectory(prefix="orchidee_stage4_thermal_reader_") as td:
        build = Path(td)
        source = build / "reader_oracle.f90"
        spans = compose_reader(source, output_dir / "extracted_original_bytes")
        executable = build / "reader_oracle.exe"
        compiler_metadata = compile_reader(source, executable, compiler, coverage=False)
        runtime_netcdf = build / "r.nc"
        shutil.copy2(netcdf_path, runtime_netcdf)
        geometry_path = build / "reader_geometry.bin"
        float_path = build / "reader_float.bin"
        discrete_path = build / "reader_discrete.bin"
        _write_stream(
            geometry_path,
            (
                geometry["lalo"],
                geometry["resolution"],
                geometry["neighbours"],
                geometry["contfrac"],
            ),
        )
        subprocess.run(
            [
                str(executable),
                runtime_netcdf.name,
                str(geometry_path),
                str(float_path),
                str(discrete_path),
            ],
            cwd=build,
            env=_netcdf_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        actual_float = _read_stream(float_path, READER_FLOAT_LAYOUT, np.float64)
        actual_discrete = _read_stream(
            discrete_path, READER_DISCRETE_LAYOUT, np.int32
        )
    write_point_comparisons(
        output_dir / "reader_point_comparisons.csv",
        actual_float,
        expected_float,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    comparisons = [
        float_comparison(
            name,
            actual_float[name],
            expected_float[name],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        for name, _ in READER_FLOAT_LAYOUT
    ]
    comparisons.extend(
        exact_comparison(name, actual_discrete[name], expected_discrete[name])
        for name, _ in READER_DISCRETE_LAYOUT
    )
    return {
        "comparisons": comparisons,
        "metadata": {
            "reader_source_spans": spans,
            "reader_compiler": compiler_metadata,
            "minimal_real_format_input": {
                "asset": netcdf_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(netcdf_path),
                "format": "NetCDF3 64-bit offset",
                "dimensions": {"longitude": NLON, "latitude": NLAT, "level": NGRND},
            },
        },
    }


def _initialize_inputs() -> dict[str, np.ndarray]:
    znt = np.asarray([0.05, 0.20, 0.80, 2.00], dtype=np.float64)
    zlt = np.asarray([0.10, 0.40, 1.20, 3.00], dtype=np.float64)
    dlt = np.asarray([0.10, 0.30, 0.80, 1.80], dtype=np.float64)
    external_reftemp = np.asarray(
        [[280.0, 278.0, 276.0, 274.0], [273.0, 272.0, 271.0, 270.0], [268.0, 269.0, 270.0, 271.0]],
        dtype=np.float64,
    )
    external_refsoc = np.asarray(
        [[20.0, 40.0, 60.0, 80.0], [100.0, 200.0, 400.0, 800.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    veget_max = np.zeros((NPTS, NVM), dtype=np.float64)
    veget_max[0, 13] = 1.0
    veget_max[1, 0] = 0.2
    veget_max[1, 13] = 0.8
    veget_max[2, 0] = 1.0
    shumdiag_perma = np.asarray(
        [[0.20, 0.30, 0.40], [0.55, 0.70, 0.85], [0.10, 0.50, 0.90]],
        dtype=np.float64,
    )
    mc_layh = np.asarray(
        [[0.12, 0.18, 0.24], [0.20, 0.28, 0.34], [0.15, 0.22, 0.30]],
        dtype=np.float64,
    )
    mcl_layh = mc_layh * np.asarray([[1.0], [0.7], [0.4]])
    tmc_layh = np.asarray(
        [[12.0, 18.0, 24.0], [20.0, 28.0, 34.0], [15.0, 22.0, 30.0]],
        dtype=np.float64,
    )
    pft_scale = 1.0 + np.arange(NVM, dtype=np.float64) * 0.001
    mc_layh_pft = mc_layh[:, :, None] * pft_scale[None, None, :]
    mcl_layh_pft = mcl_layh[:, :, None] * pft_scale[None, None, :]
    tmc_layh_pft = tmc_layh[:, :, None] * pft_scale[None, None, :]
    snowdz = np.asarray(
        [[0.0, 0.0, 0.0], [0.005, 0.0, 0.0], [0.05, 0.05, 0.05]],
        dtype=np.float64,
    )
    snowrho = np.asarray(
        [[50.0, 50.0, 50.0], [180.0, 80.0, 50.0], [250.0, 280.0, 320.0]],
        dtype=np.float64,
    )
    snowtemp = np.asarray(
        [[273.15, 273.15, 273.15], [271.0, 272.0, 273.0], [265.0, 267.0, 269.0]],
        dtype=np.float64,
    )
    ptn_restart = np.broadcast_to(
        external_reftemp[:, :, None] + 1.25,
        (NPTS, NGRND, NVM),
    ).copy()
    layer = np.arange(NGRND, dtype=np.float64)[None, :, None]
    pft = np.arange(NVM, dtype=np.float64)[None, None, :]
    point = np.arange(NPTS, dtype=np.float64)[:, None, None]
    rst_shum_long = 0.25 + 0.03 * point + 0.02 * layer + 0.001 * pft
    rst_shum_perma = 0.35 + 0.02 * point + 0.01 * layer + 0.001 * pft
    return {
        "znt": znt,
        "zlt": zlt,
        "dlt": dlt,
        "external_reftemp": external_reftemp,
        "external_refsoc": external_refsoc,
        "lalo": np.asarray([[0.0, 0.0], [45.0, 90.0], [-30.0, 150.0]], dtype=np.float64),
        "neighbours": np.zeros((NPTS, NBNEIGHB), dtype=np.int32),
        "resolution": np.full((NPTS, 2), 100000.0, dtype=np.float64),
        "contfrac": np.asarray([1.0, 0.8, 0.6], dtype=np.float64),
        "veget_max": veget_max,
        "shumdiag_perma": shumdiag_perma,
        "snow": np.asarray([0.0, 5.0, 150.0], dtype=np.float64),
        "thawed_humidity": np.asarray([0.4, 0.6, 0.8], dtype=np.float64),
        "soilc_total": np.zeros((NPTS, NDEEP, NVM), dtype=np.float64),
        "temp_sol_new": np.asarray([281.0, 272.0, 268.0], dtype=np.float64),
        "temp_sol_new_pft": np.broadcast_to(
            np.asarray([281.0, 272.0, 268.0])[:, None], (NPTS, NVM)
        ).copy(),
        "organic_layer_thick": np.asarray([0.05, 0.50, 2.50], dtype=np.float64),
        "mc_layh": mc_layh,
        "mcl_layh": mcl_layh,
        "tmc_layh": tmc_layh,
        "mc_layh_pft": mc_layh_pft,
        "mcl_layh_pft": mcl_layh_pft,
        "tmc_layh_pft": tmc_layh_pft,
        "njsc": np.asarray([1, 6, 12], dtype=np.int32),
        "frac_snow_veg": np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        "frac_snow_nobio": np.asarray([[0.0], [0.25], [0.75]], dtype=np.float64),
        "totfrac_nobio": np.asarray([0.0, 0.1, 0.2], dtype=np.float64),
        "snowdz": snowdz,
        "snowrho": snowrho,
        "snowtemp": snowtemp,
        "pb": np.asarray([101300.0, 90000.0, 78000.0], dtype=np.float64),
        "rst_ptn": ptn_restart,
        "rst_refsoc": external_refsoc + 50.0,
        "rst_shum_long": rst_shum_long,
        "rst_shum_perma": rst_shum_perma,
        "rst_e_soil_lat": 10.0 + np.arange(NPTS * NVM, dtype=np.float64).reshape((NPTS, NVM)),
        "rst_gtemp": np.asarray([271.0, 272.0, 273.0], dtype=np.float64),
        "rst_soilcap": np.asarray([1101.0, 1102.0, 1103.0], dtype=np.float64),
        "rst_soilcap_pft": 1200.0 + np.arange(NPTS * NVM, dtype=np.float64).reshape((NPTS, NVM)),
        "rst_soilflx": np.asarray([1301.0, 1302.0, 1303.0], dtype=np.float64),
        "rst_soilflx_pft": 1400.0 + np.arange(NPTS * NVM, dtype=np.float64).reshape((NPTS, NVM)),
        "rst_cgrnd": 1500.0 + np.arange(NPTS * (NGRND - 1) * NVM, dtype=np.float64).reshape((NPTS, NGRND - 1, NVM)),
        "rst_dgrnd": 1700.0 + np.arange(NPTS * (NGRND - 1) * NVM, dtype=np.float64).reshape((NPTS, NGRND - 1, NVM)),
        "rst_cgrnd_snow": 1900.0 + np.arange(NPTS * NSNOW, dtype=np.float64).reshape((NPTS, NSNOW)),
        "rst_dgrnd_snow": 2000.0 + np.arange(NPTS * NSNOW, dtype=np.float64).reshape((NPTS, NSNOW)),
        "rst_lambda_snow": np.asarray([2101.0, 2102.0, 2103.0], dtype=np.float64),
    }


def _initialize_input_stream(
    values: dict[str, np.ndarray], *, restart: bool
) -> tuple[np.ndarray, ...]:
    names = (
        "znt", "zlt", "dlt", "external_reftemp", "external_refsoc",
        "lalo", "neighbours", "resolution", "contfrac", "veget_max",
        "shumdiag_perma", "snow", "thawed_humidity", "soilc_total",
        "temp_sol_new", "temp_sol_new_pft", "organic_layer_thick",
        "mc_layh", "mcl_layh", "tmc_layh", "mc_layh_pft",
        "mcl_layh_pft", "tmc_layh_pft", "njsc", "frac_snow_veg",
        "frac_snow_nobio", "totfrac_nobio", "snowdz", "snowrho",
        "snowtemp", "pb", "rst_ptn", "rst_refsoc", "rst_shum_long",
        "rst_shum_perma", "rst_e_soil_lat", "rst_gtemp", "rst_soilcap",
        "rst_soilcap_pft", "rst_soilflx", "rst_soilflx_pft", "rst_cgrnd",
        "rst_dgrnd", "rst_cgrnd_snow", "rst_dgrnd_snow", "rst_lambda_snow",
    )
    return (np.asarray([int(restart)], dtype=np.int32), *(values[name] for name in names))


def _restart_fields(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "ptn": values["rst_ptn"],
        "refSOC": values["rst_refsoc"],
        "shum_ngrnd_prmlng": values["rst_shum_long"],
        "shum_ngrnd_perma": values["rst_shum_perma"],
        "e_soil_lat": values["rst_e_soil_lat"],
        "gtemp": values["rst_gtemp"],
        "soilcap": values["rst_soilcap"],
        "soilcap_pft": values["rst_soilcap_pft"],
        "soilflx": values["rst_soilflx"],
        "soilflx_pft": values["rst_soilflx_pft"],
        "cgrnd": values["rst_cgrnd"],
        "dgrnd": values["rst_dgrnd"],
        "cgrnd_snow": values["rst_cgrnd_snow"],
        "dgrnd_snow": values["rst_dgrnd_snow"],
        "lambda_snow": values["rst_lambda_snow"],
    }


def _coefficient_inputs(values: dict[str, np.ndarray]) -> dict[str, Any]:
    dz1 = 1.0 / np.diff(values["znt"])
    dz5 = np.zeros(NGRND, dtype=np.float64)
    dz5[:-1] = (values["zlt"][:-1] - values["znt"][:-1]) * dz1
    moisture = SimpleNamespace(
        shumdiag_perma=values["shumdiag_perma"],
        mc_layh=values["mc_layh"],
        mcl_layh=values["mcl_layh"],
        tmc_layh=values["tmc_layh"],
        mc_layh_pft=values["mc_layh_pft"],
        mcl_layh_pft=values["mcl_layh_pft"],
        tmc_layh_pft=values["tmc_layh_pft"],
    )
    return {
        "moisture": moisture,
        "temp_sol_new_pft": values["temp_sol_new_pft"],
        "snowdz": values["snowdz"],
        "snowrho": values["snowrho"],
        "snowtemp": values["snowtemp"],
        "njsc": values["njsc"],
        "pb": values["pb"],
        "dlt": values["dlt"],
        "dz1": dz1,
        "zlt": values["zlt"],
        "znt": values["znt"],
        "dz5": dz5,
        "dt_sechiba": 1800.0,
        "frac_snow_veg": values["frac_snow_veg"],
        "frac_snow_nobio": values["frac_snow_nobio"],
        "totfrac_nobio": values["totfrac_nobio"],
        "ok_laidev": np.zeros(NVM, dtype=bool),
    }


def _initialize_jax_expected(
    values: dict[str, np.ndarray], *, restart: bool
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    coefficient_inputs = _coefficient_inputs(values)
    restart_fields = _restart_fields(values) if restart else {}
    external_ptn = None if restart else np.broadcast_to(
        values["external_reftemp"][:, :, None], (NPTS, NGRND, NVM)
    ).copy()
    result = thermosoil_initialize(
        ngrnd=NGRND,
        nscm=NSCM,
        temp_sol_new=values["temp_sol_new"],
        veget_max=values["veget_max"],
        coefficient_inputs=coefficient_inputs,
        restart_fields=restart_fields,
        external_ptn=external_ptn,
        external_refsoc=None if restart else values["external_refsoc"],
        read_reftemp=True,
        use_refSOC=True,
        use_soilc_tempdiff=True,
        use_toporganiclayer_tempdiff=False,
        ok_freeze_thermix=True,
        ok_pc=False,
        ok_leak=True,
        ok_Ecorr=True,
        ok_wetdiaglong=False,
        ok_zimov=False,
        satsoil=False,
        bedrock_flag=0,
        val_exp=999999.0,
    )
    closure = thermosoil_cold_start_coef_closure(
        ptn=result.ptn,
        temp_sol_new=values["temp_sol_new"],
        veget_max=values["veget_max"],
        refsoc=result.refsoc,
        use_refSOC=True,
        use_soilc_tempdiff=True,
        shum_ngrnd_permalong=result.shum_ngrnd_permalong,
        ok_freeze_thermix=True,
        brk_flag=0,
        **coefficient_inputs,
    )
    if not closure.ok or closure.humlev is None or closure.getdiff is None:
        raise RuntimeError(f"JAX initialization closure is incomplete: {closure.missing_inputs}")
    humlev = closure.humlev
    getdiff = closure.getdiff
    if restart:
        thin_snow = thermosoil_getdiff_thinsnow(
            ptn=result.ptn,
            shum_ngrnd_permalong=result.shum_ngrnd_permalong,
            snowdz=values["snowdz"],
            pcapa=getdiff.pcapa,
            pcapa_en=getdiff.pcapa_en,
            pkappa=getdiff.pkappa,
            profil_froz=getdiff.profil_froz,
            veget_mask_2d=np.ones((NPTS, NVM), dtype=bool),
            zlt=values["zlt"],
        )
        getdiff = getdiff._replace(
            pcapa=thin_snow.pcapa,
            pcapa_en=thin_snow.pcapa_en,
            pkappa=thin_snow.pkappa,
            profil_froz=thin_snow.profil_froz,
        )
    dz1 = coefficient_inputs["dz1"]
    dz5 = coefficient_inputs["dz5"]
    float_values = {
        "reftemp": np.zeros((NPTS, NGRND)) if restart else values["external_reftemp"],
        "refSOC": np.asarray(result.refsoc),
        "ptn": np.asarray(result.ptn),
        "ptn_pftmean": np.asarray(result.ptn_pftmean),
        "dz1": np.concatenate((np.asarray(dz1), np.zeros(1))),
        "z1": np.zeros(NPTS),
        "cgrnd": np.asarray(result.cgrnd),
        "dgrnd": np.asarray(result.dgrnd),
        "pcapa": np.asarray(getdiff.pcapa),
        "pkappa": np.asarray(getdiff.pkappa),
        "pcapa_en": np.asarray(getdiff.pcapa_en),
        "pcapa_snow": np.asarray(getdiff.pcapa_snow),
        "pkappa_snow": np.asarray(getdiff.pkappa_snow),
        "ptn_beg": np.asarray(result.ptn_beg),
        "temp_sol_beg": np.asarray(result.temp_sol_beg),
        "shum_ngrnd_perma": np.asarray(result.shum_ngrnd_perma),
        "shum_ngrnd_permalong": np.asarray(result.shum_ngrnd_permalong),
        "profil_froz": np.asarray(result.profil_froz),
        "pcappa_supp": np.asarray(result.pcappa_supp),
        "e_soil_lat": np.asarray(result.e_soil_lat),
        "dz5": np.asarray(dz5),
        "mcs": np.asarray(SMCMAX_USDA),
        "SMCMAX": np.asarray(SMCMAX_USDA),
        "QZ": np.asarray(QZ_USDA),
        "so_capa_dry_ns": np.asarray(SO_CAPA_DRY_NS_USDA),
        "mc_layt": np.asarray(humlev.mc_layt),
        "mcl_layt": np.asarray(humlev.mcl_layt),
        "tmc_layt": np.asarray(humlev.tmc_layt),
        "mc_layt_pft": np.asarray(humlev.mc_layt_pft),
        "mcl_layt_pft": np.asarray(humlev.mcl_layt_pft),
        "tmc_layt_pft": np.asarray(humlev.tmc_layt_pft),
        "stempdiag": np.asarray(result.stempdiag),
        "soilcap": np.asarray(result.soilcap),
        "soilcap_pft": np.asarray(result.soilcap_pft),
        "soilflx": np.asarray(result.soilflx),
        "soilflx_pft": np.asarray(result.soilflx_pft),
        "gtemp": np.asarray(result.gtemp),
        "lambda_snow": np.asarray(result.lambda_snow),
        "cgrnd_snow": np.asarray(result.cgrnd_snow),
        "dgrnd_snow": np.asarray(result.dgrnd_snow),
        "veget_max_bg": np.asarray(result.veget_max_bg),
        "veget_mask_real": np.asarray(result.veget_mask_real),
        "organic_layer_thick": values["organic_layer_thick"],
        "lambda": np.asarray([values["znt"][0] * dz1[0]]),
    }
    discrete_values = {
        "veget_mask_2d": np.asarray(result.veget_mask_2d, dtype=np.int32),
        "controls": np.asarray(
            [
                int(result.calculate_coef),
                int(result.ok_zimov),
                int(result.ok_shum_ngrnd_permalong),
                result.brk_flag,
                int(result.satsoil),
                0,
                1,
                1,
            ],
            dtype=np.int32,
        ),
    }
    return float_values, discrete_values


def run_initialize_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, Any]:
    values = _initialize_inputs()
    np.savez(output_dir / "initialize_inputs.npz", **values)
    expected = {
        case: _initialize_jax_expected(values, restart=case == "restart")
        for case in ("cold", "restart")
    }
    comparisons: list[dict[str, Any]] = []
    compiler_metadata: dict[str, Any]
    spans: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="orchidee_stage4_thermal_initialize_") as td:
        build = Path(td)
        source = build / "initialize_oracle.f90"
        spans = compose_initialize(source, output_dir / "extracted_original_bytes")
        executable = build / "initialize_oracle.exe"
        compiler_metadata = compile_initialize(
            source, executable, compiler, coverage=False
        )
        for case in ("cold", "restart"):
            input_path = build / f"{case}_input.bin"
            float_path = build / f"{case}_float.bin"
            discrete_path = build / f"{case}_discrete.bin"
            _write_stream(
                input_path,
                _initialize_input_stream(values, restart=case == "restart"),
            )
            subprocess.run(
                [str(executable), str(input_path), str(float_path), str(discrete_path)],
                cwd=build,
                env=compiler_environment(compiler),
                check=True,
                capture_output=True,
                text=True,
            )
            actual_float = _read_stream(float_path, INITIALIZE_FLOAT_LAYOUT, np.float64)
            actual_discrete = _read_stream(
                discrete_path, INITIALIZE_DISCRETE_LAYOUT, np.int32
            )
            expected_float, expected_discrete = expected[case]
            write_point_comparisons(
                output_dir / f"initialize_{case}_point_comparisons.csv",
                actual_float,
                expected_float,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            comparisons.extend(
                float_comparison(
                    f"{case}.{name}",
                    actual_float[name],
                    expected_float[name],
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
                for name, _ in INITIALIZE_FLOAT_LAYOUT
            )
            comparisons.extend(
                exact_comparison(
                    f"{case}.{name}",
                    actual_discrete[name],
                    expected_discrete[name],
                )
                for name, _ in INITIALIZE_DISCRETE_LAYOUT
            )
    return {
        "comparisons": comparisons,
        "metadata": {
            "initialize_source_spans": spans,
            "initialize_compiler": compiler_metadata,
            "initialize_cases": {
                "cold": "real reference profiles; no/thin/deep snow; shallow/deep moisture; absent-PFT14 point",
                "restart": "complete deterministic restart group with distinct values for every carried coefficient",
            },
            "undefined_allocations_excluded": {
                "fields": ["surfheat_incr", "coldcont_incr"],
                "reason": "thermosoil_initialize lines 429-433 allocate but do not assign these arrays",
            },
        },
    }


def _generated_procedure_start(
    generated: bytes, span: ExtractedProcedure
) -> int:
    declaration = span.span_bytes.splitlines()[0].strip().lower()
    matches = [
        index
        for index, line in enumerate(generated.splitlines(), start=1)
        if line.strip().lower() == declaration
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one generated declaration {declaration!r}, found {matches}"
        )
    return matches[0]


def _gcov_listing(
    build: Path, source: Path, gcov: Path, compiler: Path
) -> Path:
    notes = list(build.glob("*.gcno"))
    if len(notes) != 1:
        raise RuntimeError(f"expected one Fortran gcno in {build}, found {notes}")
    subprocess.run(
        [str(gcov), "-b", "-c", notes[0].name],
        cwd=build,
        env=compiler_environment(compiler),
        check=True,
        capture_output=True,
        text=True,
    )
    listings = list(build.glob("*.gcov"))
    preferred = [path for path in listings if path.name == f"{source.name}.gcov"]
    if len(preferred) == 1:
        return preferred[0]
    if len(listings) != 1:
        raise RuntimeError(f"expected one Fortran gcov listing in {build}, found {listings}")
    return listings[0]


def _coverage_record(
    arm: dict[str, Any],
    branches: dict[int, list[dict[str, int]]],
    line_counts: dict[int, int],
    generated_line: int,
    span: ExtractedProcedure,
    listing_asset: str,
) -> dict[str, Any]:
    rows = branches.get(generated_line, [])
    selected = select_arm_branch(str(arm["arm_id"]), rows)
    kind = str(arm["arm_id"]).rsplit(":", 2)[-2]
    witness_line = generated_line + 1 if kind == "case" and selected is None else None
    witness_count = 0 if witness_line is None else line_counts.get(witness_line, 0)
    return {
        "arm_id": str(arm["arm_id"]),
        "base_region_id": str(arm["base_region_id"]),
        "procedure": str(arm["fortran_procedure"]),
        "original_line": int(arm["line"]),
        "generated_line": generated_line,
        "evidence_route": "gcov",
        "gcov_branch_index": -1 if selected is None else selected["branch_index"],
        "gcov_taken": witness_count if selected is None else selected["taken"],
        "gcov_case_witness_line": witness_line,
        "gcov_case_witness_count": witness_count,
        "procedure_span_sha256": span.span_sha256,
        "gcov_listing_asset": listing_asset,
        "raw_gcov_branches": rows,
        "passed": bool(
            (selected is not None and selected["taken"] > 0)
            or (kind == "case" and witness_count > 0)
        ),
    }


def _reader_gcov_records(
    arms: list[dict[str, Any]],
    output_dir: Path,
    compiler: Path,
    gcov: Path,
) -> list[dict[str, Any]]:
    geometry = _reader_geometry()
    canonical_netcdf = output_dir / "minimal_refsoc.nc"
    saved = output_dir / "gcov/reader_oracle.f90.gcov"
    saved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_stage4_thermal_reader_gcov_") as td:
        build = Path(td)
        source = build / "reader_oracle.f90"
        compose_reader(source, output_dir / "extracted_original_bytes")
        executable = build / "reader_oracle.exe"
        compile_reader(source, executable, compiler, coverage=True)
        runtime_netcdf = build / "r.nc"
        shutil.copy2(canonical_netcdf, runtime_netcdf)
        geometry_path = build / "reader_geometry.bin"
        _write_stream(
            geometry_path,
            (
                geometry["lalo"], geometry["resolution"], geometry["neighbours"], geometry["contfrac"]
            ),
        )
        subprocess.run(
            [
                str(executable), runtime_netcdf.name, str(geometry_path),
                str(build / "float.bin"), str(build / "discrete.bin"),
            ],
            cwd=build,
            env=_netcdf_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        listing = _gcov_listing(build, source, gcov, compiler)
        shutil.copy2(listing, saved)
        branches = parse_gcov(listing)
        line_counts = parse_gcov_line_counts(listing)
        span = extract_procedure_bytes(THERMOSOIL_SOURCE, "read_refSOCfile")
        generated_start = _generated_procedure_start(source.read_bytes(), span)
        return [
            _coverage_record(
                arm,
                branches,
                line_counts,
                generated_start + int(arm["line"]) - span.start_line,
                span,
                saved.relative_to(ROOT).as_posix(),
            )
            for arm in arms
        ]


def _initialize_gcov_records(
    arms: list[dict[str, Any]],
    output_dir: Path,
    compiler: Path,
    gcov: Path,
) -> list[dict[str, Any]]:
    values = _initialize_inputs()
    saved = output_dir / "gcov/initialize_oracle.f90.gcov"
    saved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_stage4_thermal_initialize_gcov_") as td:
        build = Path(td)
        source = build / "initialize_oracle.f90"
        compose_initialize(source, output_dir / "extracted_original_bytes")
        executable = build / "initialize_oracle.exe"
        compile_initialize(source, executable, compiler, coverage=True)
        for case in ("cold", "restart"):
            input_path = build / f"{case}_input.bin"
            _write_stream(
                input_path,
                _initialize_input_stream(values, restart=case == "restart"),
            )
            subprocess.run(
                [
                    str(executable), str(input_path),
                    str(build / f"{case}_float.bin"),
                    str(build / f"{case}_discrete.bin"),
                ],
                cwd=build,
                env=compiler_environment(compiler),
                check=True,
                capture_output=True,
                text=True,
            )
        listing = _gcov_listing(build, source, gcov, compiler)
        shutil.copy2(listing, saved)
        branches = parse_gcov(listing)
        line_counts = parse_gcov_line_counts(listing)
        spans = {
            procedure: extract_procedure_bytes(THERMOSOIL_SOURCE, procedure)
            for procedure in ("thermosoil_initialize", "thermosoil_var_init")
        }
        starts = {
            procedure: _generated_procedure_start(source.read_bytes(), span)
            for procedure, span in spans.items()
        }
        listing_asset = saved.relative_to(ROOT).as_posix()
        records = []
        for arm in arms:
            procedure = str(arm["fortran_procedure"])
            span = spans[procedure]
            records.append(
                _coverage_record(
                    arm,
                    branches,
                    line_counts,
                    starts[procedure] + int(arm["line"]) - span.start_line,
                    span,
                    listing_asset,
                )
            )
        return records


def _target_contract() -> tuple[
    dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]
]:
    contract = json.loads(CONTRACT_CLASSES.read_text(encoding="utf-8"))
    proofs = json.loads(SOURCE_PROOFS.read_text(encoding="utf-8"))
    target_ids = set(OWNER_IDS.values())
    owners = {
        str(record["owner_region_id"]): record
        for record in contract["owner_regions"]
        if str(record.get("owner_region_id")) in target_ids
    }
    if set(owners) != target_ids:
        raise RuntimeError(f"thermal owner contract drift: {set(owners)} != {target_ids}")
    base_ids = {str(record["base_region_id"]) for record in owners.values()}
    arms = [
        record
        for record in contract["arm_classifications"]
        if str(record.get("base_region_id")) in base_ids
    ]
    proof_by_arm = {
        str(record["arm_id"]): record
        for record in proofs.get("records", [])
        if record.get("passed") is True
    }
    return owners, arms, proof_by_arm


def run_coverage(
    output_dir: Path = OUTPUT_DIR,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
) -> dict[str, Any]:
    comparison = run_oracle(output_dir, compiler)
    if comparison.get("status") != "passed":
        raise RuntimeError("thermal comparison must pass before owner evidence is emitted")
    owners, arms, proof_by_arm = _target_contract()
    source_records: list[dict[str, Any]] = []
    gcov_targets: list[dict[str, Any]] = []
    for arm in arms:
        arm_id = str(arm["arm_id"])
        proof = proof_by_arm.get(arm_id)
        if proof is None:
            gcov_targets.append(arm)
            continue
        edge = str(proof.get("edge_class", ""))
        policy = str(proof.get("policy_id", ""))
        if edge == "paper_static_dispatch":
            route = "pinned_proof"
        elif "fatal" in edge.lower() or "fatal" in policy.lower():
            route = "fatal_proof"
        else:
            raise RuntimeError(
                f"canonical source proof for {arm_id} is neither pinned nor fatal"
            )
        source_records.append(
            {
                "arm_id": arm_id,
                "base_region_id": str(arm["base_region_id"]),
                "procedure": str(arm["fortran_procedure"]),
                "original_line": int(arm["line"]),
                "evidence_route": route,
                "source_proof_asset": SOURCE_PROOFS.relative_to(ROOT).as_posix(),
                "source_line_sha256": proof.get("fortran_physical_line_sha256"),
                "policy_id": policy,
                "checks": proof.get("checks", []),
                "passed": True,
            }
        )
    reader_targets = [
        arm for arm in gcov_targets if arm["fortran_procedure"] == "read_refsocfile"
    ]
    initialize_targets = [
        arm
        for arm in gcov_targets
        if arm["fortran_procedure"] in {"thermosoil_initialize", "thermosoil_var_init"}
    ]
    if len(reader_targets) + len(initialize_targets) != len(gcov_targets):
        raise RuntimeError("thermal GCOV target routed outside the owned procedures")
    gcov_records = [
        *_reader_gcov_records(reader_targets, output_dir, compiler, gcov),
        *_initialize_gcov_records(initialize_targets, output_dir, compiler, gcov),
    ]
    records = sorted(
        (*gcov_records, *source_records), key=lambda record: record["arm_id"]
    )
    required_ids = {str(arm["arm_id"]) for arm in arms}
    covered_ids = {record["arm_id"] for record in records if record["passed"]}
    route_counts = {
        route: sum(record["evidence_route"] == route for record in records)
        for route in ("gcov", "fatal_proof", "pinned_proof")
    }
    branch_coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": required_ids == covered_ids,
        "required_arm_count": len(required_ids),
        "covered_arm_count": len(required_ids & covered_ids),
        "route_counts": route_counts,
        "gcov_arm_ids": sorted(
            record["arm_id"] for record in records if record["evidence_route"] == "gcov" and record["passed"]
        ),
        "fatal_proof_arm_ids": sorted(
            record["arm_id"] for record in records if record["evidence_route"] == "fatal_proof" and record["passed"]
        ),
        "pinned_proof_arm_ids": sorted(
            record["arm_id"] for record in records if record["evidence_route"] == "pinned_proof" and record["passed"]
        ),
        "missing_arm_ids": sorted(required_ids - covered_ids),
        "arms": records,
        "compiler": str(compiler),
        "gcov": str(gcov),
        "fatal_proof_note": "No target thermal arm is classified as a fatal path by the canonical contract; the fatal route is therefore empty.",
    }
    (output_dir / "branch_coverage.json").write_text(
        json.dumps(branch_coverage, indent=2) + "\n", encoding="ascii"
    )
    records_by_base: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_base.setdefault(record["base_region_id"], []).append(record)
    owner_records = []
    for owner_id, owner in sorted(owners.items()):
        base_id = str(owner["base_region_id"])
        owner_arm_records = records_by_base.get(base_id, [])
        required = {str(value) for value in owner["arm_ids"]}
        gcov_ids = {
            record["arm_id"]
            for record in owner_arm_records
            if record["evidence_route"] == "gcov" and record["passed"]
        }
        fatal_ids = {
            record["arm_id"]
            for record in owner_arm_records
            if record["evidence_route"] == "fatal_proof" and record["passed"]
        }
        pinned_ids = {
            record["arm_id"]
            for record in owner_arm_records
            if record["evidence_route"] == "pinned_proof" and record["passed"]
        }
        covered = gcov_ids | fatal_ids | pinned_ids
        owner_records.append(
            {
                "owner_region_id": owner_id,
                "base_region_id": base_id,
                "fortran_procedure": owner["fortran_procedure"],
                "jax_owners": owner["jax_owners"],
                "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json",
                "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
                "required_arm_ids": sorted(required),
                "gcov_arm_ids": sorted(gcov_ids),
                "fatal_proof_arm_ids": sorted(fatal_ids),
                "pinned_proof_arm_ids": sorted(pinned_ids),
                "source_proof_arm_ids": sorted(fatal_ids | pinned_ids),
                "covered_arm_ids": sorted(covered),
                "missing_arm_ids": sorted(required - covered),
                "passed": required == covered,
            }
        )
    owner_evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": all(record["passed"] for record in owner_records),
        "records": owner_records,
    }
    (output_dir / "owner_region_evidence.json").write_text(
        json.dumps(owner_evidence, indent=2) + "\n", encoding="ascii"
    )
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "owner_ids": sorted(OWNER_IDS.values()),
                "cases": [
                    "real_netcdf_masked_retry_and_no_overlap",
                    "cold_no_thin_deep_snow_shallow_deep_profile",
                    "complete_restart_no_recompute",
                ],
                "float_policy": {"rtol": 1.0e-12, "atol": 1.0e-12},
                "discrete_policy": "exact",
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    if not branch_coverage["branch_complete"] or not owner_evidence["complete"]:
        raise RuntimeError(
            f"thermal owner coverage incomplete: {branch_coverage['missing_arm_ids']}"
        )
    return {
        "comparison": comparison,
        "branch_coverage": branch_coverage,
        "owner_evidence": owner_evidence,
    }


def run_oracle(
    output_dir: Path = OUTPUT_DIR, compiler: Path = DEFAULT_COMPILER
) -> dict[str, Any]:
    reader = run_reader_oracle(output_dir, compiler)
    initialize = run_initialize_oracle(output_dir, compiler)
    return write_result(
        output_dir,
        FAMILY,
        (*reader["comparisons"], *initialize["comparisons"]),
        {
            **reader["metadata"],
            **initialize["metadata"],
            "scope": "Stage 4 Batch D thermal read_refSOCfile, thermosoil_var_init, and thermosoil_initialize",
        },
    )


if __name__ == "__main__":
    try:
        result = run_coverage()
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or "", file=sys.stderr)
        print(exc.stderr or "", file=sys.stderr)
        raise
    print(
        json.dumps(
            {
                "status": "passed",
                "owners": len(result["owner_evidence"]["records"]),
                "covered_arms": result["branch_coverage"]["covered_arm_count"],
                "route_counts": result["branch_coverage"]["route_counts"],
            },
            indent=2,
        )
    )
