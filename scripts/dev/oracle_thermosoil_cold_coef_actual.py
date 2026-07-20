from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import (  # noqa: E402
    paper_1961_driver_cold_start_first_step_coverage,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.driver.init import parse_run_def_bool  # noqa: E402
from jax_orchidee.sechiba.thermosoil import (  # noqa: E402
    PHIGEOTH,
    PSNOWDZMIN,
    QZ_USDA,
    SMCMAX_USDA,
    SO_CAPA_DRY_NS_USDA,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)


FAMILY = "thermosoil_cold_coef_actual"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_thermosoil_cold_coef_actual.f90.template"
CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
RUN_DEF = (
    ROOT
    / "outputs/paper_250919_materialized_run_defs/arg2_1.0/319.0-057.0"
    / "I5/S26_69.834_0.0019_0.3980_97.918/used_run.def"
)
LANDPOINT_ID = "319.0-057.0"
EXPECTED_DIMS = {"npts": 1, "ngrnd": 32, "nslm": 11, "nvm": 14, "nsnow": 3, "ndeep": 32, "nnobio": 1}


def _span(start: int, end: int) -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[start - 1 : end])


def _instrument(span: bytes, terminator: bytes, statements: tuple[bytes, ...]) -> bytes:
    if span.count(terminator) != 1:
        raise ValueError(f"expected one instrumentation terminator: {terminator!r}")
    payload = b"\n".join((b"  ! Oracle capture: observation only; calculation above is unchanged.", *statements, terminator))
    return span.replace(terminator, payload)


def _fortran_bytes(value: object, dtype: np.dtype | type = np.float64) -> np.ndarray:
    return np.asfortranarray(np.asarray(value, dtype=dtype))


def _write_stream(path: Path, ordered: tuple[tuple[str, np.ndarray], ...]) -> None:
    with path.open("wb") as handle:
        for _, values in ordered:
            values.ravel(order="F").tofile(handle)


def _numeric_fields(obj: object) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for name in obj._fields:
        value = getattr(obj, name)
        if hasattr(value, "shape"):
            fields[name] = np.asarray(value, dtype=np.float64)
    return fields


def _production_inputs() -> tuple[
    tuple[tuple[str, np.ndarray], ...],
    dict[str, np.ndarray],
    dict[str, object],
]:
    context = prepare_paper_1961_driver_context(CONFIG, used_run_def_path=RUN_DEF)
    coverage = paper_1961_driver_cold_start_first_step_coverage(CONFIG, prepared_context=context)
    cold = coverage.initialized_payloads["thermosoil_cold_start_coef"]
    if not cold.ok:
        raise RuntimeError(f"production cold coefficient coverage is not closed: {cold.missing_inputs}")

    payload = coverage.initialized_payloads["first_step_payload"]
    vegetation = coverage.initialized_payloads["slowproc_cold_start_vegetation"]
    hydrol = coverage.initialized_payloads["hydrol_cold_start_state"]
    enerbil = coverage.initialized_payloads["enerbil_surface_state"]
    stomate = coverage.initialized_payloads["stomate_cold_start_entry_state"]
    ptn = np.asarray(coverage.initialized_payloads["thermosoil_ptn_constant"], dtype=np.float64)
    refsoc = np.asarray(coverage.initialized_payloads["thermosoil_refSOC"], dtype=np.float64)
    humlev = cold.humlev
    grid = context.cwrr_grid
    run_def = context.run_def_values

    dims = {
        "npts": ptn.shape[0],
        "ngrnd": ptn.shape[1],
        "nvm": ptn.shape[2],
        "nslm": np.asarray(hydrol.mc).shape[1],
        "nsnow": np.asarray(hydrol.explicit_snow.snowdz).shape[1],
        "ndeep": np.asarray(stomate.soilc_total).shape[1],
        "nnobio": np.asarray(vegetation.vegetation.frac_nobio).shape[1],
    }
    if dims != EXPECTED_DIMS:
        raise ValueError(f"unexpected production dimensions: {dims}")

    ok_laidev = np.asarray(
        [parse_run_def_bool(run_def.get(f"OK_LAIDEV__{index:05d}", "FALSE")) for index in range(1, dims["nvm"] + 1)],
        dtype=np.int32,
    )
    controls = np.asarray(
        [
            int(parse_run_def_bool(run_def.get("USE_TOPORGANICLAYER_TEMPDIFF", "FALSE"))),
            int(parse_run_def_bool(run_def.get("USE_SOILC_TEMPDIFF", "FALSE"))),
            int(parse_run_def_bool(run_def.get("use_refSOC", "FALSE"))),
            int(parse_run_def_bool(run_def.get("OK_FREEZE_THERMIX", "TRUE"))),
            int(parse_run_def_bool(run_def.get("OK_EXPLICITSNOW", "TRUE"))),
            0,
        ],
        dtype=np.int32,
    )
    npts, _, nvm = ptn.shape
    frac_nobio = np.asarray(vegetation.vegetation.frac_nobio, dtype=np.float64)
    initial_snow_coef = np.zeros((npts, dims["nsnow"]), dtype=np.float64)
    scalars = np.asarray(
        [
            float(context.dt_sechiba),
            float(np.asarray(grid.znt)[0] * np.asarray(grid.dz1)[0]),
            float(PHIGEOTH),
            float(PSNOWDZMIN),
        ],
        dtype=np.float64,
    )

    ordered = (
        ("QZ", _fortran_bytes(QZ_USDA)),
        ("SMCMAX", _fortran_bytes(SMCMAX_USDA)),
        ("mcs", _fortran_bytes(SMCMAX_USDA)),
        ("so_capa_dry_ns", _fortran_bytes(SO_CAPA_DRY_NS_USDA)),
        ("zlt", _fortran_bytes(grid.zlt)),
        ("dlt", _fortran_bytes(grid.dlt)),
        ("dz1", _fortran_bytes(grid.dz1)),
        ("refSOC", _fortran_bytes(refsoc)),
        ("mc_layt", _fortran_bytes(humlev.mc_layt)),
        ("mcl_layt", _fortran_bytes(humlev.mcl_layt)),
        ("tmc_layt", _fortran_bytes(humlev.tmc_layt)),
        ("mc_layt_pft", _fortran_bytes(humlev.mc_layt_pft)),
        ("mcl_layt_pft", _fortran_bytes(humlev.mcl_layt_pft)),
        ("tmc_layt_pft", _fortran_bytes(humlev.tmc_layt_pft)),
        ("ptn", _fortran_bytes(ptn)),
        ("ptn_pftmean", _fortran_bytes(cold.ptn_pftmean)),
        # thermosoil_initialize defaults missing shum_ngrnd_prmlng restart to one.
        ("shum_ngrnd_permalong", _fortran_bytes(np.ones_like(ptn))),
        ("njsc", _fortran_bytes(payload.njsc, np.int32)),
        ("temp_sol_new", _fortran_bytes(enerbil.temp_sol_new)),
        ("temp_sol_new_pft", _fortran_bytes(enerbil.temp_sol_new_pft)),
        ("snow", _fortran_bytes(hydrol.snow)),
        ("organic_layer_thick", _fortran_bytes(stomate.depth_organic_soil)),
        ("soilc_total", _fortran_bytes(stomate.soilc_total)),
        ("veget_max", _fortran_bytes(vegetation.vegetation.veget_max)),
        ("snowdz", _fortran_bytes(hydrol.explicit_snow.snowdz)),
        ("snowrho", _fortran_bytes(hydrol.explicit_snow.snowrho)),
        ("snowtemp", _fortran_bytes(hydrol.explicit_snow.snowtemp)),
        ("pb", _fortran_bytes(payload.pb)),
        ("frac_snow_veg", _fortran_bytes(np.zeros((npts,), dtype=np.float64))),
        ("frac_snow_nobio", _fortran_bytes(np.zeros_like(frac_nobio))),
        ("totfrac_nobio", _fortran_bytes(vegetation.vegetation.totfrac_nobio)),
        ("lambda_snow_initial", _fortran_bytes(np.zeros((npts,), dtype=np.float64))),
        ("cgrnd_snow_initial", _fortran_bytes(initial_snow_coef)),
        ("dgrnd_snow_initial", _fortran_bytes(initial_snow_coef)),
        ("ok_laidev", _fortran_bytes(ok_laidev, np.int32)),
        ("veget_mask_2d", _fortran_bytes(np.ones((npts, nvm), dtype=np.int32), np.int32)),
        ("controls", _fortran_bytes(controls, np.int32)),
        ("scalars", _fortran_bytes(scalars)),
    )

    jax: dict[str, np.ndarray] = {
        **{f"getdiff.{name}": value.ravel(order="F") for name, value in _numeric_fields(cold.getdiff).items()},
        **{f"coef.soil.{name}": value.ravel(order="F") for name, value in _numeric_fields(cold.coef.soil).items()},
        **{
            f"coef.{name}": value.ravel(order="F")
            for name, value in _numeric_fields(cold.coef).items()
            if name != "soil"
        },
    }
    metadata = {
        "landpoint_id": LANDPOINT_ID,
        "dimensions": dims,
        "run_def": str(RUN_DEF),
        "coverage_mode": coverage.mode,
        "coverage_ready": coverage.ready_for_cold_start_first_step,
        "input_fields": [name for name, _ in ordered],
    }
    return ordered, jax, metadata


OUTPUT_LAYOUT = (
    ("getdiff.pcapa", (1, 32, 14)),
    ("getdiff.pcapa_en", (1, 32, 14)),
    ("getdiff.pkappa", (1, 32, 14)),
    ("getdiff.profil_froz", (1, 32, 14)),
    ("getdiff.pcappa_supp", (1, 32, 14)),
    ("getdiff.pcapa_snow", (1, 3)),
    ("getdiff.pkappa_snow", (1, 3)),
    ("getdiff.poros_net", (1, 32, 14)),
    ("getdiff.zx1", (1, 32, 14)),
    ("getdiff.zx2", (1, 32, 14)),
    ("coef.soil.cgrnd", (1, 31, 14)),
    ("coef.soil.dgrnd", (1, 31, 14)),
    ("coef.soil.soilcap", (1,)),
    ("coef.soil.soilcap_pft", (1, 14)),
    ("coef.soil.soilflx", (1,)),
    ("coef.soil.soilflx_pft", (1, 14)),
    ("coef.soil.soilcap_pft_nosnow", (1, 14)),
    ("coef.soil.soilflx_pft_nosnow", (1, 14)),
    ("coef.soil.zdz1", (1, 31, 14)),
    ("coef.soil.zdz2", (1, 32, 14)),
    ("coef.soil.cgrnd_soil", (1,)),
    ("coef.soil.dgrnd_soil", (1,)),
    ("coef.soil.zdz1_soil", (1,)),
    ("coef.soil.zdz2_soil", (1,)),
    ("coef.lambda_snow", (1,)),
    ("coef.cgrnd_snow", (1, 3)),
    ("coef.dgrnd_snow", (1, 3)),
    ("coef.snowcap", (1,)),
    ("coef.snowflx", (1,)),
    ("coef.dz1_snow", (1, 3)),
    ("coef.dz2_snow", (1, 3)),
    ("coef.zdz1_snow", (1, 3)),
    ("coef.zdz2_snow", (1, 3)),
    ("coef.soilcap", (1,)),
    ("coef.soilflx", (1,)),
)


def _read_outputs(path: Path) -> dict[str, np.ndarray]:
    raw = np.fromfile(path, dtype=np.float64)
    offset = 0
    values: dict[str, np.ndarray] = {}
    for name, shape in OUTPUT_LAYOUT:
        count = int(np.prod(shape))
        values[name] = raw[offset : offset + count]
        offset += count
    if offset != raw.size:
        raise ValueError(f"Fortran output size mismatch: consumed {offset}, found {raw.size}")
    return values


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_inputs, jax, input_metadata = _production_inputs()
    np.savez(output_dir / "production_inputs.npz", **{name: value for name, value in ordered_inputs})
    (output_dir / "inputs.json").write_text(json.dumps({"schema_version": 1, **input_metadata}, indent=2) + "\n", encoding="ascii")

    raw_spans = {
        "thermosoil_cond_pft": _span(2007, 2127),
        "thermosoil_cond_nopft": _span(2129, 2231),
        "thermosoil_cond": _span(1883, 1972),
        "thermosoil_getdiff": _span(2566, 2863),
        "thermosoil_coef": _span(1386, 1729),
    }
    spans = {
        **{name: raw_spans[name] for name in ("thermosoil_cond_pft", "thermosoil_cond_nopft", "thermosoil_cond")},
        "thermosoil_getdiff": _instrument(
            raw_spans["thermosoil_getdiff"],
            b"  END SUBROUTINE thermosoil_getdiff",
            (b"  oracle_porosity = poros_net", b"  oracle_zx1 = zx1", b"  oracle_zx2 = zx2"),
        ),
        "thermosoil_coef": _instrument(
            raw_spans["thermosoil_coef"],
            b"  END SUBROUTINE thermosoil_coef",
            (
                b"  oracle_soilcap_nosnow = soilcap_nosnow",
                b"  oracle_soilflx_nosnow = soilflx_nosnow",
                b"  oracle_soilcap_pft_nosnow = soilcap_pft_nosnow",
                b"  oracle_soilflx_pft_nosnow = soilflx_pft_nosnow",
                b"  oracle_zdz1 = zdz1",
                b"  oracle_zdz2 = zdz2",
                b"  oracle_cgrnd_soil = cgrnd_soil",
                b"  oracle_dgrnd_soil = dgrnd_soil",
                b"  oracle_zdz1_soil = zdz1_soil",
                b"  oracle_zdz2_soil = zdz2_soil",
                b"  oracle_snowcap = snowcap",
                b"  oracle_snowflx = snowflx",
                b"  oracle_dz1_snow = dz1_snow",
                b"  oracle_dz2_snow = dz2_snow",
                b"  oracle_zdz1_snow = zdz1_snow",
                b"  oracle_zdz2_snow = zdz2_snow",
            ),
        ),
    }

    with tempfile.TemporaryDirectory(prefix="orchidee_thermosoil_cold_coef_actual_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        generated = TEMPLATE.read_bytes()
        for marker, span in spans.items():
            generated = generated.replace(f"! <{marker.upper()}>".encode(), span)
        source.write_bytes(generated)
        executable = build / "oracle.exe"
        compiler_meta = compile_fortran(source, executable, compiler)
        input_path = build / "production_inputs.bin"
        output_path = build / "fortran_outputs.bin"
        _write_stream(input_path, ordered_inputs)
        subprocess.run(
            [str(executable), str(input_path), str(output_path)],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        fortran = _read_outputs(output_path)

    missing_fortran = sorted(set(jax) - set(fortran))
    missing_jax = sorted(set(fortran) - set(jax))
    if missing_fortran or missing_jax:
        raise ValueError(f"comparison field mismatch: missing_fortran={missing_fortran}, missing_jax={missing_jax}")
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran, jax, rtol=1.0e-12, atol=1.0e-12)
    comparisons = [
        float_comparison(name, fortran[name], jax[name], rtol=1.0e-12, atol=1.0e-12)
        for name, _ in OUTPUT_LAYOUT
    ]
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "procedure_span_sha256": {name: hashlib.sha256(span).hexdigest() for name, span in raw_spans.items()},
            "instrumented_span_sha256": {name: hashlib.sha256(span).hexdigest() for name, span in spans.items()},
            "compiler": compiler_meta,
            "scope": "point 319 production cold coverage; complete public getdiff/coef numeric outputs",
            "instrumentation": "local arrays are copied after the original calculations in the temporary compile unit only",
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    target = ROOT / "outputs/reference_mode/micro_oracles/thermosoil_cold_coef_actual"
    try:
        result = run_oracle(target)
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or "", file=sys.stderr)
        print(exc.stderr or "", file=sys.stderr)
        raise
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
