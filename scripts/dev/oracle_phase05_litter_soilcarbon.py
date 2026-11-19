"""Finite phase05 original-Fortran litter -> TF-DOC -> soilcarbon oracle.

No model timestep/day/year is executed.  The two original procedures are
extracted byte-for-byte by existing audited builders; only their standalone
program wrappers gain explicit binary inputs and one-call execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# Standalone entry points resolve sibling builders after the path setup above.
from fortran_oracle_common import compiler_environment  # noqa: E402
from oracle_lane_soilcarbon_owner_probe import COMPILER  # noqa: E402
from oracle_lane_soilcarbon_owner_probe import build_source as soil_source  # noqa: E402
from oracle_lane_soilcarbon_owner_run import (  # noqa: E402
    FLOAT_FIELDS,
    FLOAT_SAVE_2,
    FLOAT_SAVE_3,
    INT_FIELDS,
    INT_SAVE_2,
    INT_SAVE_3,
    LOGICAL_FIELDS,
)
from oracle_lane_stomate_soilcarbon_litter import SHAPES as LITTER_FIELDS  # noqa: E402
from oracle_lane_stomate_soilcarbon_litter import build_source as litter_source  # noqa: E402

OUT = (
    ROOT
    / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/litter-soilcarbon"
)
FIXTURE = ROOT / "tests/fixtures/phase05_litter_soilcarbon_reference.json"
RTOL, ATOL = 1e-12, 1e-14
SPECS = [
    dict(id="warm_frozen_nonpeat", temperature=[285.0, 268.0], moisture=[0.45, 0.18]),
    dict(id="peat_runon", temperature=[285.0, 278.0], moisture=[0.45, 0.7], peat=True, runon=0.15),
    dict(id="dry_zero_water", temperature=[285.0, 278.0], moisture=[0.0, 0.0], water=0.0),
    dict(id="zero_pools_fluxes", temperature=[285.0, 278.0], moisture=[0.45, 0.4], zero=True),
    dict(id="lignin_fuel_floor", temperature=[285.0, 278.0], moisture=[0.45, 0.4], floor=True),
    dict(id="lignin_floor_below", temperature=[285.0, 278.0], moisture=[0.45, 0.4], floor=True, floor_value=5e-9),
    dict(id="lignin_floor_equal", temperature=[285.0, 278.0], moisture=[0.45, 0.4], floor=True, floor_value=1e-8),
    dict(id="lignin_floor_above", temperature=[285.0, 278.0], moisture=[0.45, 0.4], floor=True, floor_value=2e-8),
    dict(id="tf_storage_limited", temperature=[285.0, 278.0], moisture=[0.45, 0.4], storage=[0.0001, 0.0002]),
    dict(id="tf_absent_vegetation", temperature=[285.0, 278.0], moisture=[0.45, 0.4], veget=[0.0, 0.6]),
    dict(id="mixed_water_directions", temperature=[285.0, 278.0], moisture=[0.45, 0.4], mixed_water=True),
    dict(id="diffusion_temperature_boundary", temperature=[272.15, 272.1500001], moisture=[0.45, 0.4]),
    dict(id="high_doc_export_cap", temperature=[285.0, 278.0], moisture=[0.45, 0.4], doc_scale=2000.0),
    dict(id="active_layer_restart_depth", temperature=[285.0, 268.0], moisture=[0.45, 0.4], restart_altmax=0.3),
    dict(
        id="active_layer_restart_newalt_spinup",
        temperature=[285.0, 268.0],
        moisture=[0.45, 0.4],
        restart_altmax=2.5,
        newaltcalc=True,
        spinup=True,
    ),
]


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def case_inputs(spec):
    """All actual component inputs, declared independently of endpoint states."""
    n, v, d, h, s = 2, 14, 32, 11, 3
    z = np.array([0.0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2.0])
    zz = np.linspace(2.0 / d, 2.0, d)
    veg = np.zeros((n, v))
    veg[:, 13] = spec.get("veget", [0.7, 0.6])
    labove = np.zeros((n, 2, v, 1))
    labove[:, :, 13, 0] = [[4.0, 8.0], [3.0, 6.0]]
    lbelow = np.zeros((n, 2, v, d, 1))
    lbelow[:, 0, 13, :, 0] = np.array([0.08, 0.06])[:, None]
    lbelow[:, 1, 13, :, 0] = np.array([0.12, 0.09])[:, None]
    lignina = np.zeros((n, v))
    lignina[:, 13] = 0.2
    ligninb = np.zeros((n, v, d))
    ligninb[:, 13] = 0.2
    part = np.zeros((n, v, 2))
    part[:, 13] = 1.0
    bm = np.zeros((n, v, 12, 1))
    bm[:, 13, 0, 0] = [0.03, 0.02]
    bm[:, 13, 5, 0] = [0.02, 0.015]
    turn = bm * 0.5
    carbon = np.zeros((n, 3, v, d))
    carbon[:, 0, 13] = 2.0
    carbon[:, 1, 13] = 1.0
    carbon[:, 2, 13] = 0.5
    doc = np.zeros((n, v, d, 2, 7, 1))
    for pool in range(7):
        doc[:, 13, :, 0, pool, 0] = (0.01 + 0.002 * (pool + 1)) * spec.get("doc_scale", 1.0)
        doc[:, 13, :, 1, pool, 0] = 0.003 * (pool + 1)
    mc = np.broadcast_to(np.asarray(spec["moisture"])[:, None, None], (n, h, s)).copy()
    mc32 = np.broadcast_to(np.asarray(spec["moisture"])[:, None, None], (n, d, s)).copy()
    fb = np.full((n, d, v), 0.08)
    fbd = np.full_like(fb, 0.06)
    tp = np.broadcast_to(np.asarray(spec["temperature"])[:, None, None], (n, d, v)).copy()
    water = np.full((n, h, s), spec.get("water", 0.01))
    if spec.get("mixed_water"):
        water[:, :, :] = np.array([0.02, -0.02, 0.0, 0.02, -0.02, 0.0, 0.02, -0.02, 0.02, -0.02, 0.02])[None, :, None]
    pg = np.ones((n, v))
    pc = np.zeros((n, v))
    pc[:, 13] = 0.2
    cg = np.zeros((n, v))
    cg[:, 13] = 0.05
    store = np.zeros((n, v, 1))
    store[:, 13, 0] = spec.get("storage", [0.01, 0.01])
    biomass = np.zeros((n, v, 12, 1))
    biomass[:, 13, 0, 0] = 5.0
    dtop = np.zeros((n, 3))
    dsub = np.zeros_like(dtop)
    dtop[:, 1] = 0.001
    dsub[:, 2] = 0.0005
    natural = np.zeros(v, bool)
    natural[13] = True
    peat = np.zeros(v, bool)
    peat[13] = spec.get("peat", False)
    if spec.get("floor"):
        labove[:, 1, 13] = spec.get("floor_value", 1e-13)
        lbelow[:, 1, 13] = spec.get("floor_value", 1e-13)
        bm[:] = 0.0
        turn[:] = 0.0
    if spec.get("zero"):
        for a in [labove, lbelow, bm, turn, carbon, doc, pg, pc, cg, store, biomass, dtop, dsub]:
            a[:] = 0.0
    fuel = [labove.transpose(0, 2, 1, 3).copy() / 4.0 for _ in range(4)]
    x = dict(
        litter_above=labove,
        litter_below=lbelow,
        lignin_struc_above=lignina,
        lignin_struc_below=ligninb,
        litterpart=part,
        dead_leaves=np.zeros((n, v, 2)),
        bm_to_litter=bm,
        turnover=turn,
        rprof=np.full((n, v), 0.5),
        veget_max=veg,
        sla_calc=np.full((n, v), 0.02),
        fbact_litter=fb,
        poor_soils=np.array([0.0, 0.3]),
        flood_frac=np.array([0.0, 0.4]),
        carbon_32l=carbon,
        doc=doc,
        interception_storage=store,
        canopy2ground=cg,
        doc_to_topsoil=dtop,
        doc_to_subsoil=dsub,
        fbact_doc=fbd,
        fbact_soilcarbon=fb,
        soil_mc=mc,
        soil_mc_32l=mc32,
        wat_flux=water,
        soilwater_31mm=np.einsum("l,pls->ps", np.diff(z)[:5], mc[:, :5]),
        runoff_per_soil=np.full((n, s), 0.2),
        drainage_per_soil=np.full((n, s), 0.1),
        runoff2peat=np.full((n, s), spec.get("runon", 0.0)),
        fastr=np.array([25.0, 9.0]),
        pref_soil_veg=np.array([0] * 13 + [2]),
        tprof=tp,
        clay=np.array([0.25, 0.4]),
        bulk_dens=np.array([1200.0, 1350.0]),
        z_soil=z,
        zf_soil_b=np.r_[0.0, zz],
        flux_red=0.5 * (1 - np.array([0.0, 0.3])) + np.array([0.0, 0.3]),
        natural=natural,
        is_peat=peat,
        is_c4=np.zeros(v, bool),
        resp_maint_part_radia=np.zeros((n, v, 12)),
        flood_root_radia=np.zeros((n, v)),
        dt_days=1 / 48,
        nslm=h,
        ndeep=d,
        sro_bottom=5,
        min_sechiba=1e-8,
        capture_transfers=True,
    )
    for name, a in zip(("fuel_1hr", "fuel_10hr", "fuel_100hr", "fuel_1000hr"), fuel):
        x[name] = a
    # Independent source formula, soilcarbon1212-1237. The original output is
    # subsequently compared with these exact actual-caller boundary operands.
    x["doc_precip2ground"] = pg[:, :, None] * 3.02e-3
    x["doc_precip2ground"][:, 0] = 0.0
    x["doc_precip2canopy"] = pc[:, :, None] * 3.02e-3
    x["doc_precip2canopy"][:, 0] = 0.0
    dry = np.zeros((n, v, 1))
    dry[:, 13, 0] = 0.00092 / 48 * biomass[:, 13, 0, 0] * veg[:, 13]
    x["dry_dep_canopy"] = dry
    from jax_orchidee.stomate.carbon_kernels import littercalc_aboveground_controls

    c = littercalc_aboveground_controls(
        np.array(spec["temperature"]), mc, z, x["pref_soil_veg"], frozen_respiration_func=1
    )
    x.update(
        control_temp_above=np.asarray(c.control_temp_above),
        control_moist_above=np.asarray(c.control_moist_above),
        soil_mc_top_by_pft=np.asarray(c.soil_mc_top_by_pft),
    )
    raw = dict(
        tsurf=np.asarray(spec["temperature"]),
        pg=pg,
        pc=pc,
        biomass=biomass,
        zz=zz,
        altmax=np.full((n, v), spec.get("restart_altmax", 0.0)),
        fixed=np.zeros((n, v)),
        newaltcalc=spec.get("newaltcalc", False),
        spinup=spec.get("spinup", False),
    )
    return x, raw


LITTER_INPUTS = [
    ("turnover", "turnover"),
    ("bm", "bm_to_litter"),
    ("veget", "veget_max"),
    ("tsurf", "@tsurf"),
    ("soilmc", "soil_mc"),
    ("rprof", "rprof"),
    ("litterpart", "litterpart"),
    ("labove", "litter_above"),
    ("lbelow", "litter_below"),
    ("dead", "dead_leaves"),
    ("lignina", "lignin_struc_above"),
    ("ligninb", "lignin_struc_below"),
    ("fuel1", "fuel_1hr"),
    ("fuel10", "fuel_10hr"),
    ("fuel100", "fuel_100hr"),
    ("fuel1000", "fuel_1000hr"),
    ("bulk", "bulk_dens"),
    ("poor", "poor_soils"),
    ("fbact", "fbact_litter"),
    ("flood", "flood_frac"),
    ("carbon", "carbon_32l"),
    ("tprof", "tprof"),
    ("clay", "clay"),
    ("sla", "sla_calc"),
]
SOIL_INPUTS = [
    ("scin", "soilcarbon_input_doc"),
    ("fcin", "floodcarbon_input"),
    ("carbon32", "carbon_32l"),
    ("labove", "litter_above"),
    ("lbelow", "litter_below"),
    ("doc", "doc"),
    ("ligna", "lignin_struc_above"),
    ("lignb", "lignin_struc_below"),
    ("runoff", "runoff_per_soil"),
    ("runpeat", "runoff2peat"),
    ("drain", "drainage_per_soil"),
    ("wat", "wat_flux"),
    ("bulk", "bulk_dens"),
    ("poor", "poor_soils"),
    ("veget", "veget_max"),
    ("fb", "fbact_soilcarbon"),
    ("fbd", "fbact_doc"),
    ("tp", "tprof"),
    ("mc32", "soil_mc_32l"),
    ("mc", "soil_mc"),
    ("dtop", "doc_to_topsoil"),
    ("dsub", "doc_to_subsoil"),
    ("flood", "flood_frac"),
    ("pg", "@pg"),
    ("pc", "@pc"),
    ("cg", "canopy2ground"),
    ("storage", "interception_storage"),
    ("biomass", "@biomass"),
    ("fastr", "fastr"),
    ("clay", "clay"),
    ("is_peat", "is_peat"),
    ("altmax0", "@altmax"),
    ("fixed", "@fixed"),
]


def _wrapper(kind):
    source, hashes = litter_source() if kind == "litter" else soil_source()
    # Original comments contain legacy non-UTF8 bytes. A latin1 roundtrip
    # preserves every procedure byte instead of normalizing or dropping them.
    text = source.decode("latin1")
    marker = "program run_litter_oracle" if kind == "litter" else "program run_soilcarbon_owner_oracle"
    head, program = text.split(marker, 1)
    # The old family wrapper used a 1e-12 micro-domain epsilon. This new
    # wrapper binds the actual source constant (constantes_var.f90:177),
    # leaving every original procedure byte and every old fixture untouched.
    head = head.replace("min_stomate=1.d-12", "min_stomate=1.d-8")
    program = marker + program
    program = (
        program.replace("character(1024) :: outfile", "character(1024) :: outfile, infile")
        if kind == "litter"
        else program.replace("character(1024)::outfile", "character(1024)::outfile,infile")
    )
    start = program.index("  call get_command_argument(1,outfile)")
    end = program.index("  close(u)", start) + len("  close(u)")
    fields = LITTER_INPUTS if kind == "litter" else SOIL_INPUTS
    block = "  call get_command_argument(2,infile)\n  open(newunit=u,file=trim(infile),access='stream',form='unformatted',status='old')\n"
    block += "  read(u) " + ",".join(a for a, b in fields) + "\n  close(u)\n"
    if kind == "soil":
        block += "  carbon=sum(carbon32,dim=4)\n"
    block += "  call get_command_argument(1,outfile)\n  open(newunit=u,file=trim(outfile),access='stream',form='unformatted',status='replace')\n  call execute_case(u)\n  close(u)"
    return (head + program[:start] + block + program[end:]).encode("latin1"), hashes


def _write_inputs(path, fields, values, raw):
    with path.open("wb") as f:
        for name, key in fields:
            a = raw[key[1:]] if key.startswith("@") else values[key]
            np.asarray(a, dtype="<i4" if np.asarray(a).dtype.kind == "b" else "<f8").ravel(order="F").tofile(f)


def _read_fields(path, fields):
    data = path.read_bytes()
    offset = 0
    result = {}
    for name, shape in fields:
        dtype = (
            np.dtype("<i4")
            if name.startswith("save_") and name not in {"save_alt", "save_altmax_lastyear", "save_z_root"}
            else np.dtype("<f8")
        )
        size = int(np.prod(shape))
        result[name] = np.frombuffer(data, dtype=dtype, count=size, offset=offset).reshape(shape, order="F").copy()
        offset += size * dtype.itemsize
    if offset != len(data):
        raise AssertionError((offset, len(data)))
    return result


def actual_outputs(inputs):
    """Real maintained adapter and real pure half-hour writer, no mocks."""
    from jax_orchidee.driver.orchestration import _paper_half_hour_ok_leak_state_updates
    from jax_orchidee.stomate.integration import stomate_ok_leak_explicit

    result = stomate_ok_leak_explicit(**inputs)
    litter = result.littercalc
    soil = result.soilcarbon
    flat = {
        k: np.asarray(getattr(litter, k))
        for k in [
            "litter_above",
            "litter_below",
            "lignin_struc_above",
            "lignin_struc_below",
            "litterpart",
            "dead_leaves",
            "deadleaf_cover",
            "resp_hetero_litter",
            "resp_hetero_flood",
            "soilcarbon_input_doc",
            "floodcarbon_input",
        ]
    }
    flat.update(
        {k: np.asarray(getattr(litter.fuel, k)) for k in ["fuel_1hr", "fuel_10hr", "fuel_100hr", "fuel_1000hr"]}
    )
    flat.update(
        {
            k: np.asarray(getattr(soil, k))
            for k in ["carbon_32l", "doc", "doc_exp", "resp_hetero_soil", "resp_flood_soil"]
        }
    )
    flat["interception_storage"] = np.asarray(result.interception_storage)
    writer = _paper_half_hour_ok_leak_state_updates(result)
    for name, a in writer.items():
        np.testing.assert_array_equal(a, flat["doc" if name == "DOC" else name])
    # A second standalone component consumes the genuine writer operands. This
    # proves the finite next-consumer connection, not a natural trajectory.
    next_inputs = dict(inputs)
    for name, a in writer.items():
        next_inputs["doc" if name == "DOC" else name] = a
    next_inputs["turnover"] = np.zeros_like(inputs["turnover"])
    next_inputs["bm_to_litter"] = np.zeros_like(inputs["bm_to_litter"])
    return flat, result, writer, next_inputs


def _compare(expected, actual):
    rows = []
    for name, f in expected.items():
        j = np.asarray(actual[name])
        delta = np.abs(f - j)
        scale = np.maximum(np.abs(f), np.abs(j))
        mask = np.isfinite(f) & np.isfinite(j) & (delta <= ATOL + RTOL * scale)
        rows.append(
            dict(
                field=name,
                passed=bool(np.all(mask)),
                count=int(f.size),
                failed=int(np.count_nonzero(~mask)),
                max_abs=float(delta.max(initial=0.0)),
            )
        )
    return rows


def _reference(builds, x, raw, temp, output, ident):
    inp = temp / "input.bin"
    out = temp / "output.bin"
    _write_inputs(inp, LITTER_INPUTS, x, raw)
    r = subprocess.run(
        [str(builds["litter"]), str(out), str(inp)],
        cwd=temp,
        env=compiler_environment(COMPILER),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    (output / f"{ident}.litter.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    lit = _read_fields(out, LITTER_FIELDS)
    sx = dict(x)
    sx.update(
        {
            k: lit[k]
            for k in [
                "litter_above",
                "litter_below",
                "lignin_struc_above",
                "lignin_struc_below",
                "soilcarbon_input_doc",
                "floodcarbon_input",
                "bulk_dens",
            ]
        }
    )
    _write_inputs(inp, SOIL_INPUTS, sx, raw)
    env = compiler_environment(COMPILER)
    env["ORACLE_NEWALTCALC"] = "1" if raw["newaltcalc"] else "0"
    env["ORACLE_SPINUP"] = "1" if raw["spinup"] else "0"
    r = subprocess.run(
        [str(builds["soil"]), str(out), str(inp)],
        cwd=temp,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    (output / f"{ident}.soil.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    soil = _read_fields(
        out, FLOAT_FIELDS + INT_FIELDS + FLOAT_SAVE_2 + INT_SAVE_2 + FLOAT_SAVE_3 + INT_SAVE_3 + LOGICAL_FIELDS
    )
    expected = {
        k: lit[k]
        for k in [
            "litter_above",
            "litter_below",
            "lignin_struc_above",
            "lignin_struc_below",
            "litterpart",
            "dead_leaves",
            "deadleaf_cover",
            "resp_hetero_litter",
            "resp_hetero_flood",
            "soilcarbon_input_doc",
            "floodcarbon_input",
            "fuel_1hr",
            "fuel_10hr",
            "fuel_100hr",
            "fuel_1000hr",
        ]
    }
    expected.update(
        {
            k: soil[k]
            for k in ["carbon_32l", "doc", "doc_exp", "resp_hetero_soil", "resp_flood_soil", "interception_storage"]
        }
    )
    return expected, lit, soil


def run(output=OUT, write_fixture=False):
    import jax

    jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=True)
    builds = {}
    receipts = []
    saved = {}
    fixture = []
    with tempfile.TemporaryDirectory(prefix="orcjax_phase05_litter_") as td:
        temp = Path(td)
        for kind in ["litter", "soil"]:
            source, hashes = _wrapper(kind)
            src = temp / f"{kind}.f90"
            src.write_bytes(source)
            (output / f"{kind}-oracle.f90").write_bytes(source)
            exe = temp / f"{kind}.exe"
            cmd = [
                str(COMPILER),
                "-std=legacy",
                "-cpp",
                "-fdefault-real-8",
                "-ffree-line-length-none",
                "-O0",
                "-fcheck=all",
                str(src),
                "-o",
                str(exe),
            ]
            r = subprocess.run(
                cmd,
                cwd=temp,
                env=compiler_environment(COMPILER),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if r.returncode:
                raise RuntimeError(r.stderr)
            receipts.append(
                dict(
                    kind=kind,
                    command=cmd,
                    returncode=r.returncode,
                    stdout=r.stdout,
                    stderr=r.stderr,
                    span_sha256=hashes,
                    compiled_source_sha256=hashlib.sha256(source).hexdigest(),
                )
            )
            builds[kind] = exe
        for spec in SPECS:
            x, raw = case_inputs(spec)
            ident = spec["id"]
            saved.update({ident + ".input." + k: np.asarray(v) for k, v in x.items()})
            saved.update({ident + ".raw_boundary." + k: np.asarray(v) for k, v in raw.items()})
            expected, lit, soil = _reference(builds, x, raw, temp, output, ident)
            actual, result, writer, next_inputs = actual_outputs(x)
            rows = _compare(expected, actual)
            from jax_orchidee.coupled import stomate_active_layer_ok_leak_inputs

            active = stomate_active_layer_ok_leak_inputs(
                tprof=x["tprof"],
                zi_soil=raw["zz"],
                restart_altmax=raw["altmax"],
                restart_fixed_cryoturbation_depth=raw["fixed"],
                dayno=1,
                firstcall_soilcarbon=True,
                newaltcalc=raw["newaltcalc"],
                soilc_isspinup=raw["spinup"],
            )
            active_expected = {
                k: soil["save_" + k] for k in ["alt", "alt_ind", "altmax_lastyear", "altmax_ind", "altmax_ind_lastyear"]
            }
            active_expected["altmax"] = soil["altmax"]
            rows.extend(_compare(active_expected, active.state_updates))
            for key, a in active.state_updates.items():
                saved[ident + ".jax.active_layer." + key] = np.asarray(a)
            for name in ["doc_precip2ground", "doc_precip2canopy", "dry_dep_canopy"]:
                rows.extend(_compare({name: soil[name]}, {name: x[name]}))
            for name, a in lit.items():
                saved[ident + ".fortran.litter." + name] = a
            for name, a in soil.items():
                saved[ident + ".fortran.soil." + name] = a
            for name, a in actual.items():
                saved[ident + ".jax." + name] = a
            for name, a in writer.items():
                saved[ident + ".writer." + name] = np.asarray(a)
            # Actual internal flux diagnostics are preserved as observations,
            # without claiming an original Fortran oracle for unreturned locals.
            for name in result.transfers._fields:
                saved[ident + ".jax.transfer." + name] = np.asarray(getattr(result.transfers, name))
            fixture.append(
                dict(
                    spec=spec,
                    expected={k: dict(shape=list(a.shape), values=a.ravel().tolist()) for k, a in expected.items()},
                    active_layer_expected={
                        k: dict(shape=list(a.shape), values=a.ravel().tolist()) for k, a in active_expected.items()
                    },
                )
            )
            receipts.append(
                dict(
                    case=ident,
                    passed=all(r["passed"] for r in rows),
                    comparisons=rows,
                    writer_fields=list(writer),
                    next_consumer="stomate_ok_leak_explicit(**next_component_inputs(...))",
                    next_consumer_executed=False,
                )
            )
            if ident in {"warm_frozen_nonpeat", "peat_runon"}:
                # The reference carry comes only from original Fortran return
                # values; the actual carry comes only from the actual writer.
                reference_next = dict(x)
                for key in writer:
                    name = "doc" if key == "DOC" else key
                    reference_next[name] = expected[name]
                reference_next["turnover"] = np.zeros_like(x["turnover"])
                reference_next["bm_to_litter"] = np.zeros_like(x["bm_to_litter"])
                second, _, _ = _reference(builds, reference_next, raw, temp, output, ident + ".next")
                next_actual, _, next_writer, _ = actual_outputs(next_inputs)
                next_rows = _compare(second, next_actual)
                fixture[-1]["next_expected"] = {
                    k: dict(shape=list(a.shape), values=a.ravel().tolist()) for k, a in second.items()
                }
                receipts[-1].update(
                    next_consumer_executed=True,
                    next_comparisons=next_rows,
                    next_passed=all(r["passed"] for r in next_rows),
                )
                for key, a in second.items():
                    saved[ident + ".fortran.next." + key] = a
                for key, a in next_actual.items():
                    saved[ident + ".jax.next." + key] = a
    np.savez_compressed(output / "raw-inputs-outputs.npz", **saved)
    report = dict(
        schema="phase05-litter-soilcarbon-v1",
        scope="finite standalone components; PFT14 only; no whole-model integration",
        rtol=RTOL,
        atol=ATOL,
        jax_version=jax.__version__,
        backend=jax.default_backend(),
        x64=jax.config.x64_enabled,
        source_min_stomate=dict(file="fortran_source/ORCHIDEE/src_parameters/constantes_var.f90", line=177, value=1e-8),
        receipts=receipts,
        passed=all(r.get("passed", True) and r.get("next_passed", True) for r in receipts),
        production_changes="carbon core repair owned by root; this runner does not edit production",
    )
    _json(output / "report.json", report)
    if write_fixture:
        _json(
            FIXTURE,
            dict(
                schema="phase05-litter-soilcarbon-reference-v1",
                rtol=RTOL,
                atol=ATOL,
                source="byte-exact original Fortran finite two-procedure chain",
                cases=fixture,
            ),
        )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--write-fixture", action="store_true")
    a = p.parse_args()
    r = run(a.output, a.write_fixture)
    print(json.dumps({"passed": r["passed"], "cases": len(SPECS), "output": str(a.output)}))
    raise SystemExit(0 if r["passed"] else 1)
