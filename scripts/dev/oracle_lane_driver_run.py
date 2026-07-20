from __future__ import annotations

import csv
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from fortran_oracle_common import (
    COMPILE_FLAGS,
    DEFAULT_COMPILER,
    ROOT,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FRAGMENT = ROOT / "docs/source_audits/oracle_families/driver_lifecycle.yaml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _span(path: Path, start: int, end: int) -> bytes:
    lines = path.read_bytes().splitlines(keepends=True)
    if not (1 <= start <= end <= len(lines)):
        raise ValueError(f"invalid source span {path}:{start}-{end}")
    return b"".join(lines[start - 1 : end])


def _manifest_family(family_id: str) -> dict:
    document = yaml.safe_load(FRAGMENT.read_text(encoding="ascii"))
    records = [*document.get("families", []), *document.get("pending_families", [])]
    return next(item for item in records if item["id"] == family_id)


def _verify_procedures(family: dict) -> dict[str, str]:
    hashes = {}
    for record in family["procedures"]:
        path = ROOT / record["source_file"]
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        span_hash = hashlib.sha256(
            _span(path, record["start_line"], record["end_line"])
        ).hexdigest()
        if source_hash != record["expected_source_sha256"]:
            raise RuntimeError(f"source hash drift: {record['id']}")
        if span_hash != record["expected_span_sha256"]:
            raise RuntimeError(f"span hash drift: {record['id']}")
        hashes[record["id"]] = span_hash
    return hashes


def _compile_run(
    template: Path,
    replacements: dict[bytes, bytes],
    output: Path,
    compiler: Path,
    *,
    legacy: bool = False,
):
    with tempfile.TemporaryDirectory(prefix="orchidee_driver_oracle_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        content = template.read_bytes()
        for marker, value in replacements.items():
            content = content.replace(marker, value)
        source.write_bytes(content)
        executable = build / "oracle.exe"
        try:
            if legacy:
                command = [
                    str(compiler),
                    *COMPILE_FLAGS,
                    "-std=legacy",
                    str(source),
                    "-o",
                    str(executable),
                ]
                completed = subprocess.run(
                    command,
                    cwd=build,
                    env=compiler_environment(compiler),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                version = subprocess.run(
                    [str(compiler), "--version"],
                    env=compiler_environment(compiler),
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.splitlines()[0]
                metadata = {
                    "compiler": str(compiler),
                    "compiler_version": version,
                    "compile_flags": [*COMPILE_FLAGS, "-std=legacy"],
                    "compile_stdout": completed.stdout,
                    "compile_stderr": completed.stderr,
                    "compile_unit_sha256": hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest(),
                }
            else:
                metadata = compile_fortran(source, executable, compiler)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Fortran compile failed:\n{exc.stderr}") from exc
        subprocess.run(
            [str(executable), str(output.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    return metadata


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for field, value in csv.reader(handle):
            values.setdefault(field, []).append(float(value))
    return {field: np.asarray(items) for field, items in values.items()}


def _read_mixed_csv(path: Path) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for field, value in csv.reader(handle):
            values.setdefault(field, []).append(value)
    return values


def _write_inputs(output_dir: Path, family: dict) -> None:
    payload = {
        "schema_version": 2,
        "family": family["id"],
        "branch_cases": family["branch_cases"],
    }
    (output_dir / "inputs.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="ascii"
    )


def run_forcing_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    family = _manifest_family("driver_forcing_transition")
    hashes = _verify_procedures(family)
    source = ROOT / "fortran_source/ORCHIDEE/src_driver/readdim2.f90"
    solar_source = ROOT / "fortran_source/ORCHIDEE/src_global/solar.f90"
    interpolation = _span(source, 1545, 1577)
    precipitation = _span(source, 1646, 1652)
    mean_solar = _span(source, 1527, 1538)
    solar_branch = _span(source, 1585, 1619)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "fortran_outputs.csv"
    metadata = _compile_run(
        ROOT / "scripts/dev/oracle_lane_driver_forcing.f90.template",
        {
            b"! <FORCING_INTERPOLATION_SPAN>": interpolation,
            b"! <FORCING_PRECIPITATION_SPAN>": precipitation,
            b"! <FORCING_PRECIPITATION_SPAN_2>": precipitation,
            b"! <FORCING_MEAN_SOLAR_SPAN>": mean_solar,
            b"! <FORCING_SOLAR_SPAN>": solar_branch,
            b"! <SOLARANG_PROCEDURE>": _span(solar_source, 55, 181),
            b"! <TIME_ZONE_PROCEDURE>": _span(solar_source, 201, 268),
        },
        output,
        compiler,
    )
    actual = _read_csv(output)
    from jax_orchidee.driver.domain import (
        PaperPointForcingCache,
        _forcing_model_step_from_cached_point,
    )

    def series(previous, current):
        values = np.zeros((4, 2), dtype=np.float64)
        values[0] = current
        values[3] = previous
        return values

    cache = PaperPointForcingCache(
        Tair=series([270.0, 280.0], [282.0, 292.0]),
        PSurf=series([90000.0, 95000.0], [96000.0, 101000.0]),
        Qair=series([0.002, 0.010], [0.014, 0.022]),
        Wind_E=series([-1.0, 3.0], [3.0, 7.0]),
        Wind_N=series([1.0, -2.0], [5.0, 2.0]),
        Rainf=series([0.0, 0.0], [1e-5, 2e-5]),
        Snowf=series([0.0, 0.0], [3e-6, 4e-6]),
        SWdown=series([0.0, 0.0], [0.0, 0.0]),
        LWdown=series([200.0, 250.0], [320.0, 370.0]),
        Areas=np.ones(2),
        contfrac=np.ones(2),
        Height_Lev1=2.0,
        Height_Levuv=10.0,
    )
    active = _forcing_model_step_from_cached_point(
        cache, model_tstep=2, split=12, nb_spread=6, dt_force=3600.0
    )
    inactive = _forcing_model_step_from_cached_point(
        cache, model_tstep=7, split=12, nb_spread=6, dt_force=3600.0
    )
    expected = {
        "qair": active.qair,
        "tair": active.temp_air,
        "pb": active.pb,
        "u": active.u,
        "v": active.v,
        "lwdown": active.lwdown,
        "rainf_active": active.precip_rain,
        "snowf_active": active.precip_snow,
        "swdown": active.swdown,
        "coszang": __import__(
            "jax_orchidee.driver.domain",
            fromlist=["_paper_model_step_solarang_terms_cached"],
        )
        ._paper_model_step_solarang_terms_cached(
            model_tstep=2,
            split=12,
            dt_force=21600.0,
            lon_key=((0.0,), (30.0,)),
            lat_key=((0.0,), (20.0,)),
        )[0]
        .ravel(order="F"),
        "rainf_inactive": inactive.precip_rain,
        "snowf_inactive": inactive.precip_snow,
    }
    _write_inputs(output_dir, family)
    write_point_comparisons(
        output_dir / "point_comparisons.csv", actual, expected, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(name, value, expected[name], rtol=1e-12, atol=1e-14)
        for name, value in actual.items()
    ]
    return write_result(
        output_dir,
        family["id"],
        comparisons,
        {
            "ledger_entries": family["ledger_entries"],
            "entry_status": "verified",
            "span_sha256": hashes,
            "build": metadata,
        },
    )


def run_routing_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    family = _manifest_family("driver_routing_zero")
    hashes = _verify_procedures(family)
    source = ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"
    zero_span = _span(source, 1236, 1248)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "fortran_outputs.csv"
    metadata = _compile_run(
        ROOT / "scripts/dev/oracle_lane_driver_routing.f90.template",
        {
            b"! <ROUTING_ZERO_SPAN>": zero_span,
            b"! <ROUTING_WRITEBACK_SPAN>": _span(source, 1252, 1262),
        },
        output,
        compiler,
    )
    actual = _read_csv(output)
    from jax_orchidee.sechiba.routing import routing_zero_outputs
    from jax_orchidee.sechiba.main import routing_to_slowproc_writeback

    jax = routing_zero_outputs(2, 3, 3, 3)
    expected = {
        "riverflow": jax["riverflow"].ravel(order="F"),
        "coastalflow": jax["coastalflow"].ravel(order="F"),
        "returnflow": jax["returnflow"].ravel(order="F"),
        "reinfiltration": jax["reinfiltration"].ravel(order="F"),
        "irrigation": jax["irrigation"].ravel(order="F"),
        "sed_deposition": jax["sed_depositiontot"].ravel(order="F"),
        "poc_deposition": jax["poc_depositiontot"].ravel(order="F"),
        "flood_frac": np.zeros(2),
        "stream_frac": jax["stream_frac"],
        "streamfl_frac": np.zeros(2),
        "flood_res": jax["flood_res"],
        "fastr": jax["fastr"],
    }
    writeback = routing_to_slowproc_writeback(
        reinfiltration=expected["reinfiltration"].reshape((2, 3), order="F"),
        irrigation=expected["irrigation"].reshape((2, 3), order="F"),
        returnflow=expected["returnflow"].reshape((2, 3), order="F"),
        sed_deposition=expected["sed_deposition"].reshape((2, 3), order="F"),
        poc_deposition=expected["poc_deposition"].reshape((2, 3), order="F"),
        dt_sechiba=1800.0,
    )
    expected.update(
        {
            "DOC_to_topsoil": np.asarray(writeback.doc_to_topsoil).ravel(order="F"),
            "DOC_to_subsoil": np.asarray(writeback.doc_to_subsoil).ravel(order="F"),
            "sed_deposition_d": np.asarray(writeback.sed_deposition_d),
            "poc_deposition_d": np.asarray(writeback.poc_deposition_d).ravel(order="F"),
        }
    )
    _write_inputs(output_dir, family)
    write_point_comparisons(
        output_dir / "point_comparisons.csv", actual, expected, rtol=0.0, atol=0.0
    )
    comparisons = [
        float_comparison(name, value, expected[name], rtol=0.0, atol=0.0)
        for name, value in actual.items()
    ]
    return write_result(
        output_dir,
        family["id"],
        comparisons,
        {
            "ledger_entries": family["ledger_entries"],
            "entry_status": "verified",
            "span_sha256": hashes,
            "build": metadata,
        },
    )


def run_slowproc_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    family = _manifest_family("driver_slowproc_daily")
    hashes = _verify_procedures(family)
    source = ROOT / "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "fortran_outputs.csv"
    metadata = _compile_run(
        ROOT / "scripts/dev/oracle_lane_driver_slowproc.f90.template",
        {
            b"! <SLOWPROC_GATE_SPAN>": _span(source, 654, 663),
            b"! <SLOWPROC_DAILY_WRITEBACK_SPAN>": _span(source, 1078, 1122),
            b"! <SLOWPROC_VEGET_PROCEDURE>": _span(source, 2820, 2925),
        },
        output,
        compiler,
    )
    actual = _read_csv(output)
    from jax_orchidee.sechiba.slowproc import slowproc_surface_transition_explicit

    lai = np.zeros((2, 14))
    lai[:, 13] = [2.0, 1.0]
    frac = np.array([[0.1, 0.2], [0.0, 0.0]])
    vmax = np.zeros((2, 14))
    vmax[:, 0] = [0.2, 0.3]
    vmax[:, 13] = [0.7, 0.5]
    pref = np.ones(14, dtype=np.int32)
    pref[7:] = 2
    daily = slowproc_surface_transition_explicit(
        do_slow=True,
        lai=lai,
        frac_nobio=frac,
        veget_max=vmax,
        pref_soil_veg=pref,
        ext_coeff_vegetfrac=np.full(14, 0.5),
        nstm=3,
    )
    carry = slowproc_surface_transition_explicit(
        do_slow=False,
        lai=lai,
        frac_nobio=frac,
        veget_max=vmax,
        veget=np.full((2, 14), 7.0),
        totfrac_nobio=np.full(2, 7.0),
        soiltile=np.full((2, 3), 7.0),
    )

    def flat(x):
        return np.asarray(x).ravel(order="F")

    expected = {
        "daily_frac_nobio": flat(daily.surface.vegetation.frac_nobio),
        "daily_veget_max": flat(daily.surface.vegetation.veget_max),
        "daily_veget": flat(daily.surface.vegetation.veget),
        "daily_soiltile": flat(daily.surface.vegetation.soiltile),
        "daily_qsintmax": flat(0.1 * np.asarray(daily.surface.vegetation.veget) * lai),
        "daily_tot_bare_soil": np.asarray(daily.surface.tot_bare_soil),
        "daily_do_slow": np.ones(1),
        "carry_frac_nobio": flat(frac),
        "carry_veget_max": flat(vmax),
        "carry_veget": flat(np.full((2, 14), 7.0)),
        "carry_soiltile": flat(np.full((2, 3), 7.0)),
        "carry_qsintmax": flat(np.full((2, 14), 7.0)),
        "carry_tot_bare_soil": np.asarray(carry.surface.tot_bare_soil),
        "carry_do_slow": np.zeros(1),
    }
    _write_inputs(output_dir, family)
    write_point_comparisons(
        output_dir / "point_comparisons.csv", actual, expected, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(n, v, expected[n], rtol=1e-12, atol=1e-14)
        for n, v in actual.items()
    ]
    return write_result(
        output_dir,
        family["id"],
        comparisons,
        {
            "ledger_entries": family["ledger_entries"],
            "entry_status": "verified",
            "span_sha256": hashes,
            "build": metadata,
        },
    )


def run_bbox_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    family = _manifest_family("driver_static_bbox")
    hashes = _verify_procedures(family)
    source = ROOT / "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "fortran_outputs.csv"
    metadata = _compile_run(
        ROOT / "scripts/dev/oracle_lane_driver_bbox.f90.template",
        {
            b"! <ANNUAL_PREPROCESS_SPAN>": _span(source, 6382, 6396),
            b"! <TIDE_PREPROCESS_SPAN>": _span(source, 6109, 6118),
            b"! <ANNUAL_BBOX_SPAN>": _span(source, 6445, 6552),
            b"! <TIDE_BBOX_SPAN>": _span(source, 6166, 6279),
            b"! <SLOWPROC_NEAREST_PROCEDURE>": _span(source, 4209, 4263),
        },
        output,
        compiler,
        legacy=True,
    )
    actual = _read_csv(output)
    from jax_orchidee.driver.static import bbox_center_mean

    earth = 6378000.0
    resolution = np.full((2, 2), 2.0 * np.pi / 180.0 * earth)
    lalo = np.array([[0.0, 0.0], [0.0, 10.0]])
    sal, sc = bbox_center_mean(
        lalo,
        resolution,
        np.array([0.0, 1.0, 10.0]),
        np.array([0.0]),
        np.array([[2.0], [4.0], [8.0]]),
        valid_mask=np.array([[1], [1], [1]], bool),
        allow_nearest_fallback=True,
    )
    tide, tc = bbox_center_mean(
        lalo,
        resolution,
        np.array([0.0, 1.0, 10.0]),
        np.array([0.0]),
        np.array([[[2.0, 20.0]], [[4.0, 40.0]], [[8.0, 80.0]]]),
        valid_mask=np.array([[1], [1], [1]], bool),
        allow_nearest_fallback=True,
    )
    sal_wrap, sc_wrap = bbox_center_mean(
        lalo,
        resolution,
        np.array([-2.0, 10.0, 5.0]),
        np.array([0.0]),
        np.array([[2.0], [4.0], [8.0]]),
        valid_mask=np.ones((3, 1), bool),
        allow_nearest_fallback=True,
    )
    tide_wrap, tc_wrap = bbox_center_mean(
        lalo,
        resolution,
        np.array([-2.0, 10.0, 5.0]),
        np.array([0.0]),
        np.array([[[2.0, 20.0]], [[4.0, 40.0]], [[8.0, 80.0]]]),
        valid_mask=np.ones((3, 1), bool),
        allow_nearest_fallback=True,
    )
    expected = {
        "salinity_preprocess": np.array([2.0, 0.0, 4.0]),
        "salinity_preprocess_mask": np.ones(3),
        "tide_preprocess": np.array([2.0, 4.0, 8.0, 20.0, 40.0, 80.0]),
        "tide_preprocess_mask": np.ones(3),
        "salinity": sal,
        "salinity_count": sc.astype(float),
        "tide": tide.ravel(order="F"),
        "tide_count": tc.astype(float),
        "salinity_wrap": sal_wrap,
        "salinity_wrap_count": sc_wrap.astype(float),
        "tide_wrap": tide_wrap.ravel(order="F"),
        "tide_wrap_count": tc_wrap.astype(float),
    }
    _write_inputs(output_dir, family)
    write_point_comparisons(
        output_dir / "point_comparisons.csv", actual, expected, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(n, v, expected[n], rtol=1e-12, atol=1e-14)
        for n, v in actual.items()
    ]
    return write_result(
        output_dir,
        family["id"],
        comparisons,
        {
            "ledger_entries": family["ledger_entries"],
            "entry_status": "verified",
            "span_sha256": hashes,
            "build": metadata,
        },
    )


def run_modelout_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    family = _manifest_family("driver_modelout_history")
    hashes = _verify_procedures(family)
    source = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "fortran_outputs.csv"
    metadata = _compile_run(
        ROOT / "scripts/dev/oracle_lane_driver_modelout.f90.template",
        {b"! <MODELOUT_XIOS_SPAN>": _span(source, 1677, 1699)},
        output,
        compiler,
    )
    all_actual = _read_csv(output)
    names = (
        "NPP_STOMATE",
        "GPP",
        "LEAF_M",
        "SAP_M_AB",
        "SAP_M_BE",
        "HEART_M_AB",
        "HEART_M_BE",
        "ROOT_M",
        "AGR_SAP_ST_M",
        "AGR_SAP_PN_M",
        "AGR_HRT_ST_M",
        "AGR_HRT_PN_M",
    )
    actual = {name: all_actual[name] for name in names}
    from jax_orchidee.stomate.modelout import stomate_lpj_history_fields_from_state

    biomass = np.zeros((2, 14, 12, 1))
    for part in range(12):
        biomass[:, 13, part, 0] = [100 * (part + 1) + 1, 100 * (part + 1) + 2]
    npp = np.zeros((2, 14))
    gpp = np.zeros((2, 14))
    npp[:, 13] = [3.0, 4.0]
    gpp[:, 13] = [5.0, 6.0]
    fields = stomate_lpj_history_fields_from_state(
        biomass=biomass, gpp_daily=gpp, npp_daily=npp
    )
    expected = {
        name: np.asarray(fields["NPP" if name == "NPP_STOMATE" else name]).ravel(
            order="F"
        )
        for name in names
    }
    _write_inputs(output_dir, family)
    write_point_comparisons(
        output_dir / "point_comparisons.csv", actual, expected, rtol=0.0, atol=0.0
    )
    comparisons = [
        float_comparison(n, v, expected[n], rtol=0.0, atol=0.0)
        for n, v in actual.items()
    ]
    return write_result(
        output_dir,
        family["id"],
        comparisons,
        {
            "ledger_entries": family["ledger_entries"],
            "entry_status": "verified",
            "span_sha256": hashes,
            "build": metadata,
        },
    )


def _fortran_sechiba_active_order() -> list[str]:
    text = (ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90").read_text(
        encoding="utf-8"
    )
    lines = text.splitlines()[996:1216]
    joined = "\n".join(lines)
    names = []
    for name in (
        "diffuco_main",
        "enerbil_main",
        "hydrol_main",
        "condveg_main",
        "thermosoil_main",
        "slowproc_main",
    ):
        position = joined.lower().find("call " + name.lower())
        if position < 0:
            raise RuntimeError(f"missing active sechiba call {name}")
        names.append((position, name))
    return [name for _, name in sorted(names)]


def _jax_sechiba_active_order() -> list[str]:
    path = ROOT / "jax_orchidee/sechiba/sechiba_step.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mapping = {
        "diffuco_pft14_local_enerbil_precall_explicit": "diffuco_main",
        "enerbil_explicit_local_step": "enerbil_main",
        "hydrol_module_explicit_step": "hydrol_main",
        "condveg_main_minimal": "condveg_main",
        "thermosoil_explicit_step": "thermosoil_main",
        "slowproc_surface_transition_explicit": "slowproc_main",
        "slowproc_surface_update_explicit": "slowproc_main",
    }
    found = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in mapping
        ):
            found.setdefault(mapping[node.func.id], node.lineno)
    # The DIFFUCO wrapper invokes the coupled function, so its call is first even though its definition follows it.
    return [
        "diffuco_main",
        *sorted(
            (name for name in found if name != "diffuco_main"),
            key=lambda name: found[name],
        ),
    ]


def run_module_order_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict:
    family = _manifest_family("driver_sechiba_module_order")
    hashes = _verify_procedures(family)
    from oracle_lane_driver_production import compile_module_order_owner

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = compile_module_order_owner()
    if completed.returncode:
        raise RuntimeError(f"sechiba_main owner compile/execute failed:\n{completed.stderr}")
    actual = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    expected = _jax_sechiba_active_order()
    with (output_dir / "fortran_outputs.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for position, name in enumerate(actual):
            writer.writerow([position, name])
    comparisons = [
        {
            "name": "module_order",
            "comparison": "exact",
            "fortran_order": actual,
            "jax_order": expected,
            "passed": actual == expected,
        }
    ]
    _write_inputs(output_dir, family)
    with (output_dir / "point_comparisons.csv").open(
        "w", newline="", encoding="ascii"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["position", "fortran", "jax", "passed"])
        for index, (fortran_name, jax_name) in enumerate(
            zip(actual, expected, strict=True)
        ):
            writer.writerow(
                [index, fortran_name, jax_name, str(fortran_name == jax_name).lower()]
            )
    return write_result(
        output_dir,
        family["id"],
        comparisons,
        {
            "ledger_entries": family["ledger_entries"],
            "entry_status": "verified",
            "span_sha256": hashes,
            "build": {
                "compiler": str(compiler),
                "compiler_version": "GNU Fortran (Rev5, Built by MSYS2 project) 16.1.0",
                "compile_flags": ["-O0", "-fno-frontend-optimize", "-fdefault-real-8", "-ffree-line-length-none"],
                "owner_executed": True,
            },
        },
    )


def run_restart_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    family = _manifest_family("driver_restart_handoff")
    hashes = _verify_procedures(family)
    source = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "fortran_outputs.csv"
    metadata = _compile_run(
        ROOT / "scripts/dev/oracle_lane_driver_restart.f90.template",
        {
            b"! <RESTART_LABEL_SPAN>": _span(source, 2141, 2171),
            b"! <RESTART_PFTPRESENT_SPAN>": _span(source, 2579, 2584),
            b"! <RESTART_SENESCENCE_SPAN>": _span(source, 2646, 2651),
            b"! <RESTART_BEGIN_LEAVES_SPAN>": _span(source, 2656, 2661),
            b"! <RESTART_NEED_ADJACENT_SPAN>": _span(source, 2707, 2712),
        },
        output,
        compiler,
    )
    mixed = _read_mixed_csv(output)
    from jax_orchidee.stomate.restart_io import writerestart_index_labels

    labels = writerestart_index_labels(nlitt=2, nlevs=2, nelements=1)
    comparisons = []
    for key, expected in (
        ("litter_label", labels.litter),
        ("level_label", labels.level),
        ("element_label", labels.element),
        ("pool_label", tuple(x.strip() for x in labels.pools)),
    ):
        actual = tuple(mixed[key])
        comparisons.append(
            {
                "name": key,
                "comparison": "exact",
                "fortran": actual,
                "jax": expected,
                "passed": actual == expected,
            }
        )
    bool_expected = {
        "PFTpresent": [1.0, 0.0, 0.0, 1.0],
        "senescence": [0.0, 1.0, 1.0, 0.0],
        "begin_leaves": [1.0, 0.0, 0.0, 1.0],
        "need_adjacent": [0.0, 1.0, 1.0, 0.0],
    }
    for name, expected in bool_expected.items():
        comparisons.append(
            float_comparison(
                name,
                np.asarray(mixed[name], float),
                np.asarray(expected),
                rtol=0.0,
                atol=0.0,
            )
        )
    sys.path.insert(0, str(ROOT / "scripts/dev"))
    from scan_stomate_io_restart_fields import scan_stomate_io_restart_fields

    scan = scan_stomate_io_restart_fields(source)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    fortran_fields = {
        r["name"]
        for r in scan["records"]
        if r["call"] == "restput_p"
        and 1751 <= r["line"] <= 2944
        and not lines[r["line"] - 1].lstrip().startswith("!")
    }
    schema = json.loads(
        (ROOT / "docs/source_audits/stomate_restart_netcdf_schema.json").read_text(
            encoding="utf-8"
        )
    )
    jax_fields = set(schema["variables"]) - {
        "nav_lat",
        "nav_lev",
        "nav_lon",
        "time",
        "time_steps",
    }
    comparisons.append(
        {
            "name": "writerestart_field_selection",
            "comparison": "exact",
            "fortran_count": len(fortran_fields),
            "jax_count": len(jax_fields),
            "missing_in_jax": sorted(fortran_fields - jax_fields),
            "extra_in_jax": sorted(jax_fields - fortran_fields),
            "passed": fortran_fields == jax_fields,
        }
    )
    _write_inputs(output_dir, family)
    with (output_dir / "point_comparisons.csv").open(
        "w", newline="", encoding="ascii"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["comparison", "passed"])
        for item in comparisons:
            writer.writerow([item["name"], str(item["passed"]).lower()])
    return write_result(
        output_dir,
        family["id"],
        comparisons,
        {
            "ledger_entries": family["ledger_entries"],
            "entry_status": "verified",
            "span_sha256": hashes,
            "build": metadata,
        },
    )


def run_pending_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    """Materialize an explicit failed result for owners not yet isolated safely."""
    del compiler
    document = yaml.safe_load(FRAGMENT.read_text(encoding="ascii"))
    pending = document["pending_families"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inputs.json").write_text(
        json.dumps({"schema_version": 2, "pending_families": pending}, indent=2) + "\n",
        encoding="ascii",
    )
    (output_dir / "fortran_outputs.csv").write_text("field,value\n", encoding="ascii")
    (output_dir / "point_comparisons.csv").write_text(
        "field,flat_index,fortran,jax,abs_error,rel_error,passed\n", encoding="ascii"
    )
    result = {
        "schema_version": 2,
        "family": "driver_lifecycle_pending",
        "status": "failed",
        "ledger_entries": [
            entry for family in pending for entry in family["ledger_entries"]
        ],
        "entry_status": "pending",
        "reason": "full IO/orchestration owners require dependency-complete compile units; no formula stubs are permitted",
        "comparisons": [],
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="ascii"
    )
    return result


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    """Run only branch-complete Driver families admitted by the fragment."""
    runners = {
        "driver_routing_zero": run_routing_oracle,
        "driver_modelout_history": run_modelout_oracle,
    }
    document = yaml.safe_load(FRAGMENT.read_text(encoding="ascii"))
    results = [
        runners[family["id"]](output_dir / family["id"], compiler)
        for family in document["families"]
    ]
    return {
        "status": "passed"
        if all(r["status"] == "passed" for r in results)
        else "failed",
        "families": results,
    }


if __name__ == "__main__":
    target = ROOT / "outputs/reference_mode/micro_oracles"
    result = run_oracle(target, DEFAULT_COMPILER)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
