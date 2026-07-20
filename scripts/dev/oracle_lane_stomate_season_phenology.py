from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    ROOT,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)
from stomate_oracle_arm_coverage import audit_document  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHENOLOGY_FAMILY = "stomate_phenology"
PHENOLOGY_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_phenology.f90"
PHENOLOGY_TEMPLATE = ROOT / "scripts/dev/oracle_lane_stomate_season_phenology.f90.template"
SEASON_FAMILY = "stomate_season_memory"
SEASON_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90"
SOLAR_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/solar.f90"
SEASON_TEMPLATE = ROOT / "scripts/dev/oracle_lane_stomate_season_memory.f90.template"
FRAGMENT = ROOT / "docs/source_audits/oracle_families/stomate_season_phenology.yaml"


def validate_fragment() -> dict:
    document = yaml.safe_load(FRAGMENT.read_text(encoding="utf-8"))
    audit_document(document, ROOT)
    return document


def _compile_legacy(source: Path, executable: Path, compiler: Path) -> dict[str, object]:
    flags = [
        "-std=legacy",
        "-fdec",
        "-fdefault-real-8",
        "-ffree-line-length-none",
        "-O0",
        "-fcheck=all",
        "-ffpe-trap=invalid,zero,overflow",
        "-Wall",
        "-Wextra",
    ]
    completed = subprocess.run(
        [str(compiler), *flags, str(source), "-o", str(executable)],
        cwd=source.parent,
        env=compiler_environment(compiler),
        text=True,
        capture_output=True,
        check=True,
    )
    version = subprocess.run(
        [str(compiler), "--version"],
        env=compiler_environment(compiler),
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()[0]
    return {
        "compiler": str(compiler),
        "compiler_version": version,
        "compile_flags": flags,
        "compile_stdout": completed.stdout,
        "compile_stderr": completed.stderr,
        "compile_unit_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _jax_phenology() -> dict[str, np.ndarray]:
    from jax_orchidee.stomate.carbon_kernels import phenology_step

    npts, nvm, nparts, nelements, nleafages = 2, 14, 12, 2, 4
    present = np.zeros((npts, nvm), bool)
    present[:, 13] = True
    biomass = np.zeros((npts, nvm, nparts, nelements))
    biomass[:, 13, 7, 0] = 2.0
    leaf_frac = np.zeros((npts, nvm, nleafages))
    leaf_frac[:, :, 0] = 1.0
    leaf_age = np.full((npts, nvm, nleafages), 4.0)
    wg = np.zeros((npts, nvm))
    wg[:, 13] = (299.0, 301.0)
    first = phenology_step(
        dt_days=1.0,
        pft_present=present,
        biomass=biomass,
        leaf_frac=leaf_frac,
        leaf_age=leaf_age,
        when_growthinit=wg,
        co2_to_bm=np.zeros((npts, nvm)),
        pheno_model=("none",) * nvm,
        active_pft_mask=(False,) * 13 + (True,),
    )
    result = phenology_step(
        dt_days=1.0,
        pft_present=present,
        biomass=np.asarray(first.biomass),
        leaf_frac=np.asarray(first.leaf_frac),
        leaf_age=np.asarray(first.leaf_age),
        when_growthinit=np.asarray(first.when_growthinit),
        co2_to_bm=np.asarray(first.co2_to_bm),
        pheno_model=("none",) * nvm,
        active_pft_mask=(False,) * 13 + (True,),
    )
    return {
        "when_growthinit": np.asarray(result.when_growthinit).ravel(order="F"),
        "co2_to_bm": np.asarray(result.co2_to_bm).ravel(order="F"),
        "begin_leaves": np.asarray(result.begin_leaves, dtype=float).ravel(order="F"),
        "gdd_midwinter": np.full((npts, nvm), 20.0).ravel(order="F"),
        "leaf_frac": np.asarray(result.leaf_frac).transpose(1, 0, 2).ravel(),
        "leaf_age": np.asarray(result.leaf_age).transpose(1, 0, 2).ravel(),
        "biomass_pft14": np.asarray(result.biomass)[:, 13, :, :]
        .transpose(2, 1, 0)
        .ravel(),
    }


def run_phenology_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    validate_fragment()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = PHENOLOGY_SOURCE.read_bytes()
    unit = PHENOLOGY_TEMPLATE.read_bytes().replace(b"! <PHENOLOGY_MODULE>", raw)
    fortran_csv = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_phenology_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        source.write_bytes(unit)
        executable = build / "oracle.exe"
        try:
            metadata = _compile_legacy(source, executable, compiler)
        except subprocess.CalledProcessError as error:
            command = [
                str(compiler),
                "-std=f2008",
                "-fdec",
                "-fdefault-real-8",
                "-ffree-line-length-none",
                "-O0",
                "-fcheck=all",
                str(source),
                "-o",
                str(executable),
            ]
            diagnostic = subprocess.run(
                command,
                cwd=build,
                env=compiler_environment(compiler),
                text=True,
                capture_output=True,
                check=False,
            )
            raise RuntimeError(diagnostic.stderr) from error
        executed = subprocess.run(
            [str(executable), str(fortran_csv.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            text=True,
            capture_output=True,
            check=False,
        )
        if executed.returncode:
            raise RuntimeError(executed.stderr)
    fortran = _read_csv(fortran_csv)
    jax = _jax_phenology()
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(name, values, jax[name], rtol=1e-12, atol=1e-14)
        for name, values in fortran.items()
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "PFT1 inactive and PFT14 present",
                    "when_growthinit below and above min_growthinit_time",
                    "pheno_model none later-call state writeback",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    return write_result(
        output_dir,
        PHENOLOGY_FAMILY,
        comparisons,
        {
            **metadata,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_file": str(PHENOLOGY_SOURCE.relative_to(ROOT)).replace("\\", "/"),
        },
    )


def _jax_season(*, cold_start: bool) -> dict[str, np.ndarray]:
    from jax_orchidee.stomate.season import (
        SeasonAnnualState,
        SeasonBiometeorologyState,
        SeasonMemoryState,
        season_annual_step,
        season_biometeorology_step,
        season_memory_step,
        season_update_gdd_init_date,
    )

    npts, nvm, nslm = 2, 14, 2
    t2m_daily = np.array([0.0, 290.0])
    moisture_daily = np.full((npts, nvm), 0.5)
    moisture_daily[0, 13] = 0.0
    tsoil_daily = np.full((npts, nslm), 285.0)
    tsoil_daily[0, :] = 0.0
    soilhum_daily = np.full((npts, nslm), 0.4)
    soilhum_daily[0, :] = 0.0
    state = SeasonMemoryState(
        moiavail_month=np.zeros((npts, nvm)) if cold_start else np.full((npts, nvm), 0.45),
        moiavail_week=np.zeros((npts, nvm)) if cold_start else np.full((npts, nvm), 0.4),
        t2m_longterm=np.zeros(npts) if cold_start else np.array([259.0, 289.0]),
        tau_longterm=100.0,
        t2m_month=np.zeros(npts) if cold_start else np.array([259.0, 289.0]),
        t2m_week=np.zeros(npts) if cold_start else np.array([258.0, 288.0]),
        tseason=np.zeros(npts),
        tseason_length=np.zeros(npts),
        tseason_tmp=np.zeros(npts),
        tmin_spring_time=np.zeros((npts, nvm)),
        onset_date=np.zeros((npts, nvm)),
        tsoil_month=np.zeros((npts, nslm)) if cold_start else np.full((npts, nslm), 284.0),
        soilhum_month=np.zeros((npts, nslm)) if cold_start else np.full((npts, nslm), 0.35),
        gdd_from_growthinit=np.zeros((npts, nvm)),
        tsurf_year=np.zeros(npts) if cold_start else np.array([0.0, 291.0]),
        t2m_14=np.zeros(npts) if cold_start else t2m_daily,
    )
    begin = np.zeros((npts, nvm), bool)
    begin[0, 13] = True
    first = season_memory_step(
        state,
        dt_days=1.0,
        end_of_year=False,
        moiavail_daily=moisture_daily,
        t2m_daily=t2m_daily,
        tsurf_daily=np.array([0.0, 291.0]),
        tsoil_daily=tsoil_daily,
        soilhum_daily=soilhum_daily,
        begin_leaves=begin,
        julian_diff=100.0,
        when_growthinit=np.full((npts, nvm), 10.0),
        natural=np.array([False] * 13 + [True]),
        leaf_tab=np.zeros(nvm, int),
        pheno_type=np.zeros(nvm, int),
        pft_to_mtc=np.zeros(nvm, int),
        firstcall=True,
        undef=1.0e20,
    ).state
    result = season_memory_step(
        first,
        dt_days=1.0,
        end_of_year=True,
        moiavail_daily=moisture_daily,
        t2m_daily=t2m_daily,
        tsurf_daily=np.array([0.0, 291.0]),
        tsoil_daily=tsoil_daily,
        soilhum_daily=soilhum_daily,
        begin_leaves=begin,
        julian_diff=100.0,
        when_growthinit=np.full((npts, nvm), 10.0),
        natural=np.array([False] * 13 + [True]),
        leaf_tab=np.zeros(nvm, int),
        pheno_type=np.zeros(nvm, int),
        pft_to_mtc=np.zeros(nvm, int),
        firstcall=False,
        undef=1.0e20,
    ).state
    annual = SeasonAnnualState(
        npp_longterm=np.zeros((npts, nvm)) if cold_start else np.full((npts, nvm), 0.05),
        turnover_longterm=np.zeros((npts, nvm, 12, 2)),
        gpp_week=np.zeros((npts, nvm)) if cold_start else np.full((npts, nvm), 0.1),
        maxmoiavail_lastyear=np.full((npts, nvm), 0.7),
        maxmoiavail_thisyear=np.zeros((npts, nvm)) if cold_start else np.full((npts, nvm), 0.6),
        minmoiavail_lastyear=np.full((npts, nvm), 0.2),
        minmoiavail_thisyear=np.zeros((npts, nvm)) if cold_start else np.full((npts, nvm), 0.3),
        maxgppweek_lastyear=np.full((npts, nvm), 0.4),
        maxgppweek_thisyear=np.full((npts, nvm), 0.3),
        gdd0_lastyear=np.full(npts, 1.0),
        gdd0_thisyear=np.full(npts, 2.0),
        precip_lastyear=np.full(npts, 3.0),
        precip_thisyear=np.full(npts, 4.0),
        lm_lastyearmax=np.zeros((npts, nvm)),
        lm_thisyearmax=np.zeros((npts, nvm)),
        maxfpc_lastyear=np.zeros((npts, nvm)),
        maxfpc_thisyear=np.zeros((npts, nvm)),
    )
    natural = np.array([False] * 13 + [True])
    veget = np.zeros((npts, nvm))
    veget[0, 13] = 0.5
    veget_max = np.zeros((npts, nvm))
    veget_max[0, 13] = 1.0
    biomass = np.zeros((npts, nvm, 12, 2))
    biomass[0, 13, 0, 0] = 1.0
    turnover = np.zeros_like(biomass)
    turnover[1, 13, 0, 0] = 0.1
    npp_daily = np.full((npts, nvm), 0.1)
    npp_daily[0, 13] = 0.0
    gpp_daily = np.full((npts, nvm), 0.2)
    gpp_daily[0, 13] = 0.0
    for end_of_year, memory_state in zip((False, True), (first, result), strict=True):
        annual = season_annual_step(
            annual,
            dt_days=1.0,
            tau_longterm=float(memory_state.tau_longterm),
            end_of_year=end_of_year,
            veget=veget,
            veget_max=veget_max,
            moiavail_daily=moisture_daily,
            t2m_daily=t2m_daily,
            precip_daily=np.full(npts, 2.0),
            biomass=biomass,
            npp_daily=npp_daily,
            turnover_daily=turnover,
            gpp_daily=gpp_daily,
            natural=natural,
            pasture=np.zeros(nvm, bool),
            leaflife_tab=np.full(nvm, 365.0),
            pheno_model=("none",) * nvm,
            hvc1=0.3,
            hvc2=0.6,
            firstcall=not end_of_year,
        ).state
    bio_state = SeasonBiometeorologyState(
        gdd_m5_dormance=np.zeros((npts, nvm)) if cold_start else np.ones((npts, nvm)),
        gdd_midwinter=np.zeros((npts, nvm)) if cold_start else np.ones((npts, nvm)),
        ncd_dormance=np.zeros((npts, nvm)) if cold_start else np.ones((npts, nvm)),
        ngd_minus5=np.zeros((npts, nvm)),
        time_hum_min=np.zeros((npts, nvm)),
        hum_min_dormance=np.full((npts, nvm), 0.5),
    )
    gdd_init = np.zeros((npts, 2))
    if not cold_start:
        gdd_init[0, 1] = 2000.0
    memory_states = (first, result)
    for call_index, memory in enumerate(memory_states):
        gdd_init = np.asarray(
            season_update_gdd_init_date(
                gdd_init, latitude=np.full(npts, 10.0), julian_diff=100.0, calendar_str="noleap"
            )
        )
        bio_state = season_biometeorology_step(
            bio_state,
            dt_days=1.0,
            julian_diff=100.0,
            t2m_daily=t2m_daily,
            t2m_month=np.asarray(memory.t2m_month),
            t2m_week=np.asarray(memory.t2m_week),
            t2m_longterm=np.asarray(memory.t2m_longterm),
            moiavail_month=np.asarray(memory.moiavail_month),
            when_growthinit=np.full((npts, nvm), 10.0),
            gdd_init_date=gdd_init,
            pheno_gdd_crit=np.full((nvm, 4), 1.0e20),
            ncdgdd_temp=np.full(nvm, 1.0e20),
            hum_min_time=np.full(nvm, 1.0e20),
            firstcall=call_index == 0,
            undef=1.0e20,
        ).state
    return {
        "t2m_longterm": np.asarray(result.t2m_longterm),
        "t2m_month": np.asarray(result.t2m_month),
        "t2m_week": np.asarray(result.t2m_week),
        "tseason_tmp": np.asarray(result.tseason_tmp),
        "tseason_length": np.asarray(result.tseason_length),
        "tsurf_year": np.asarray(result.tsurf_year),
        "t2m_14": np.asarray(result.t2m_14),
        "moiavail_month": np.asarray(result.moiavail_month).ravel(order="F"),
        "moiavail_week": np.asarray(result.moiavail_week).ravel(order="F"),
        "tmin_spring_time": np.asarray(result.tmin_spring_time).ravel(order="F"),
        "onset_date": np.asarray(result.onset_date).ravel(order="F"),
        "gdd_from_growthinit": np.asarray(result.gdd_from_growthinit).ravel(order="F"),
        "tsoil_month": np.asarray(result.tsoil_month).ravel(order="F"),
        "soilhum_month": np.asarray(result.soilhum_month).ravel(order="F"),
        "npp_longterm": np.asarray(annual.npp_longterm).ravel(order="F"),
        "turnover_longterm": np.asarray(annual.turnover_longterm).transpose(1, 0, 2, 3).ravel(),
        "gpp_week": np.asarray(annual.gpp_week).ravel(order="F"),
        "maxmoiavail_lastyear": np.asarray(annual.maxmoiavail_lastyear).ravel(order="F"),
        "maxmoiavail_thisyear": np.asarray(annual.maxmoiavail_thisyear).ravel(order="F"),
        "minmoiavail_lastyear": np.asarray(annual.minmoiavail_lastyear).ravel(order="F"),
        "minmoiavail_thisyear": np.asarray(annual.minmoiavail_thisyear).ravel(order="F"),
        "maxgppweek_lastyear": np.asarray(annual.maxgppweek_lastyear).ravel(order="F"),
        "maxgppweek_thisyear": np.asarray(annual.maxgppweek_thisyear).ravel(order="F"),
        "gdd0_lastyear": np.asarray(annual.gdd0_lastyear),
        "gdd0_thisyear": np.asarray(annual.gdd0_thisyear),
        "precip_lastyear": np.asarray(annual.precip_lastyear),
        "precip_thisyear": np.asarray(annual.precip_thisyear),
        "lm_lastyearmax": np.asarray(annual.lm_lastyearmax).ravel(order="F"),
        "lm_thisyearmax": np.asarray(annual.lm_thisyearmax).ravel(order="F"),
        "maxfpc_lastyear": np.asarray(annual.maxfpc_lastyear).ravel(order="F"),
        "maxfpc_thisyear": np.asarray(annual.maxfpc_thisyear).ravel(order="F"),
        "gdd_m5_dormance": np.asarray(bio_state.gdd_m5_dormance).ravel(order="F"),
        "gdd_midwinter": np.asarray(bio_state.gdd_midwinter).ravel(order="F"),
        "ncd_dormance": np.asarray(bio_state.ncd_dormance).ravel(order="F"),
        "ngd_minus5": np.asarray(bio_state.ngd_minus5).ravel(order="F"),
        "time_hum_min": np.asarray(bio_state.time_hum_min).ravel(order="F"),
        "hum_min_dormance": np.asarray(bio_state.hum_min_dormance).ravel(order="F"),
        "gdd_init_date": np.asarray(gdd_init).ravel(order="F"),
    }


def run_season_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    validate_fragment()
    output_dir.mkdir(parents=True, exist_ok=True)
    season_raw = SEASON_SOURCE.read_bytes()
    solar_lines = SOLAR_SOURCE.read_bytes().splitlines(keepends=True)
    solar_span = b"".join(solar_lines[289:597])
    unit = SEASON_TEMPLATE.read_bytes()
    unit = unit.replace(b"! <DOWNWARD_SOLAR_FLUX>", solar_span)
    unit = unit.replace(b"! <SEASON_MODULE>", season_raw)
    fortran_csv = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_season_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        source.write_bytes(unit)
        executable = build / "oracle.exe"
        try:
            metadata = _compile_legacy(source, executable, compiler)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(error.stderr) from error
        case_paths = []
        for mode in (0, 1):
            case_path = output_dir / f"fortran_outputs_case_{mode}.csv"
            executed = subprocess.run(
                [str(executable), str(case_path.resolve()), str(mode)],
                cwd=build,
                env=compiler_environment(compiler),
                text=True,
                capture_output=True,
                check=False,
            )
            if executed.returncode:
                raise RuntimeError(executed.stderr)
            case_paths.append(case_path)
    case_values = [_read_csv(path) for path in case_paths]
    fortran = {
        name: np.concatenate([case[name] for case in case_values])
        for name in case_values[0]
    }
    with fortran_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(("field", "value"))
        for name, values in fortran.items():
            writer.writerows((name, f"{value:.17e}") for value in values)
    jax_cases = [_jax_season(cold_start=False), _jax_season(cold_start=True)]
    jax = {
        name: np.concatenate([case[name] for case in jax_cases])
        for name in jax_cases[0]
    }
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [float_comparison(k, v, jax[k], rtol=1e-12, atol=1e-14) for k, v in fortran.items()]
    (output_dir / "inputs.json").write_text(
        json.dumps({"schema_version": 1, "cases": ["nonzero restart first/later and year end", "zero restart firstcall initialization", "cold/warm and active/inactive PFT14 masks"]}, indent=2) + "\n",
        encoding="ascii",
    )
    return write_result(output_dir, SEASON_FAMILY, comparisons, {**metadata, "source_sha256": hashlib.sha256(season_raw).hexdigest(), "solar_span_sha256": hashlib.sha256(solar_span).hexdigest()})


if __name__ == "__main__":
    results = [
        run_phenology_oracle(ROOT / "outputs/reference_mode/micro_oracles" / PHENOLOGY_FAMILY),
        run_season_oracle(ROOT / "outputs/reference_mode/micro_oracles" / SEASON_FAMILY),
    ]
    print(json.dumps(results, indent=2))
    raise SystemExit(0 if all(result["status"] == "passed" for result in results) else 1)
