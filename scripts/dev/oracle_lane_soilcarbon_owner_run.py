from __future__ import annotations

import csv
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
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from fortran_oracle_common import compiler_environment  # noqa: E402
from oracle_lane_soilcarbon_owner_probe import COMPILER, SOURCE, build_source  # noqa: E402

NPTS, NVM, NDEEP, NSLM, NSTM = 2, 14, 32, 11, 3
NPOOL, NDOC, NEXP, NELEMENTS, NPARTS, NCARB = 7, 2, 3, 1, 12, 3
FLOAT_FIELDS = (
    ("resp_hetero_soil", (NPTS, NVM)),
    ("resp_flood_soil", (NPTS, NVM)),
    ("doc_exp", (NPTS, NVM, NEXP, NPOOL, NELEMENTS)),
    ("dry_dep_canopy", (NPTS, NVM, NELEMENTS)),
    ("doc_precip2ground", (NPTS, NVM, NELEMENTS)),
    ("doc_precip2canopy", (NPTS, NVM, NELEMENTS)),
    ("doc_canopy2ground", (NPTS, NVM, NELEMENTS)),
    ("soilcarbon_input_doc", (NPTS, NVM, NDEEP, NPOOL, NELEMENTS)),
    ("floodcarbon_input", (NPTS, NVM, NPOOL, NELEMENTS)),
    ("carbon", (NPTS, NCARB, NVM)),
    ("carbon_32l", (NPTS, NCARB, NVM, NDEEP)),
    ("altmax", (NPTS, NVM)),
    ("fixed_cryoturbation_depth", (NPTS, NVM)),
    ("litter_below", (NPTS, 2, NVM, NDEEP, NELEMENTS)),
    ("doc", (NPTS, NVM, NDEEP, NDOC, NPOOL, NELEMENTS)),
    ("interception_storage", (NPTS, NVM, NELEMENTS)),
    ("biomass", (NPTS, NVM, NPARTS, NELEMENTS)),
    ("bulk_dens", (NPTS,)),
    ("deepc_peat", (NPTS, NDEEP, NVM)),
    ("save_alt", (NPTS, NVM)),
)
INT_FIELDS = (("save_alt_ind", (NPTS, NVM)),)
FLOAT_SAVE_2 = (("save_altmax_lastyear", (NPTS, NVM)),)
INT_SAVE_2 = (
    ("save_altmax_ind", (NPTS, NVM)),
    ("save_altmax_ind_lastyear", (NPTS, NVM)),
)
FLOAT_SAVE_3 = (("save_z_root", (NPTS, NVM)),)
INT_SAVE_3 = (("save_rootlev", (NPTS, NVM)),)
LOGICAL_FIELDS = (("save_veget_mask", (NPTS, NVM)),)


def _take(
    data: bytes, offset: int, dtype: np.dtype, shape: tuple[int, ...]
) -> tuple[np.ndarray, int]:
    count = int(np.prod(shape))
    item = (
        np.frombuffer(data, dtype=dtype, count=count, offset=offset)
        .copy()
        .reshape(shape, order="F")
    )
    return item, offset + count * dtype.itemsize


def read_fortran(path: Path) -> list[dict[str, np.ndarray]]:
    data = path.read_bytes()
    offset = 0
    calls = []
    f8, i4 = np.dtype("<f8"), np.dtype("<i4")
    for _ in range(4):
        call: dict[str, np.ndarray] = {}
        for name, shape in FLOAT_FIELDS:
            call[name], offset = _take(data, offset, f8, shape)
        for name, shape in INT_FIELDS:
            call[name], offset = _take(data, offset, i4, shape)
        for name, shape in FLOAT_SAVE_2:
            call[name], offset = _take(data, offset, f8, shape)
        for name, shape in INT_SAVE_2:
            call[name], offset = _take(data, offset, i4, shape)
        for name, shape in FLOAT_SAVE_3:
            call[name], offset = _take(data, offset, f8, shape)
        for name, shape in INT_SAVE_3 + LOGICAL_FIELDS:
            call[name], offset = _take(data, offset, i4, shape)
        calls.append(call)
    if offset != len(data):
        raise ValueError(f"binary contract consumed {offset} of {len(data)} bytes")
    return calls


def _inputs() -> dict[str, np.ndarray]:
    zz = np.linspace(2.0 / NDEEP, 2.0, NDEEP)
    diag = np.array((0.01, 0.03, 0.06, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2.0))
    veget = np.zeros((NPTS, NVM))
    veget[:, 13] = (0.7, 0.6)
    biomass = np.zeros((NPTS, NVM, NPARTS, 1))
    biomass[:, 13, 0, 0] = 5
    carbon32 = np.zeros((NPTS, NCARB, NVM, NDEEP))
    carbon32[:, 0, 13, :] = 2
    doc = np.zeros((NPTS, NVM, NDEEP, NDOC, NPOOL, 1))
    for pool in range(NPOOL):
        doc[:, 13, :, 0, pool, 0] = 0.01 + 0.002 * (pool + 1)
    scin = np.zeros((NPTS, NVM, NDEEP, NPOOL, 1))
    scin[:, 13, :, 4, 0] = 0.002
    labove = np.zeros((NPTS, 2, NVM, 1))
    labove[:, 0, 13, 0] = 2
    labove[:, 1, 13, 0] = 4
    lbelow = np.zeros((NPTS, 2, NVM, NDEEP, 1))
    lbelow[:, 0, 13, :, 0] = 0.05
    lbelow[:, 1, 13, :, 0] = 0.1
    ligna = np.zeros((NPTS, NVM))
    ligna[:, 13] = 0.2
    lignb = np.zeros((NPTS, NVM, NDEEP))
    lignb[:, 13, :] = 0.2
    mc = np.full((NPTS, NSLM, NSTM), 0.4)
    mc32 = np.full((NPTS, NDEEP, NSTM), 0.4)
    return locals()


def jax_calls(
    *,
    newaltcalc: bool = False,
    soilc_isspinup: bool = False,
    initial_altmax: float = 0.0,
) -> list[dict[str, np.ndarray]]:
    from jax_orchidee.coupled import stomate_soilwater_31mm_ok_leak_inputs
    from jax_orchidee.stomate.soilcarbon_kernels import (
        altcalc_doc,
        soilcarbon_leak_core_step,
        soilcarbon_leak_tf_doc_ground_fluxes,
        soilcarbon_leak_tf_doc_inputs,
    )

    x = _inputs()
    dt = 1800 / 86400
    pref = np.zeros(NVM, dtype=np.int32)
    pref[13] = 2
    natural = np.zeros(NVM, dtype=bool)
    natural[13] = True
    false = np.zeros(NVM, dtype=bool)
    is_tree = np.zeros(NVM, dtype=bool)
    is_tree[13] = True
    flood = np.array((0.0, 0.3))
    poor = np.array((0.0, 0.3))
    runoff = np.full((NPTS, NSTM), 0.2)
    runpeat = np.zeros_like(runoff)
    drain = np.full_like(runoff, 0.1)
    wat = np.full((NPTS, NSLM, NSTM), 0.01)
    fb = np.full((NPTS, NDEEP, NVM), 0.08)
    fbd = np.full_like(fb, 0.06)
    tp = np.full_like(fb, 278.0)
    tp[1] = 268
    dtop = np.zeros((NPTS, 3))
    dsub = np.zeros_like(dtop)
    dtop[:, 1] = 0.001
    dsub[:, 2] = 0.0005
    pg = np.ones((NPTS, NVM))
    pc = np.zeros_like(pg)
    pc[:, 13] = 0.2
    cg = np.zeros_like(pg)
    cg[:, 13] = 0.05
    storage = np.zeros((NPTS, NVM, 1))
    storage[:, 13, 0] = 0.01
    bulk = np.array((1200.0, 1350.0))
    clay = np.array((0.25, 0.4))
    fastr = np.array((25.0, 9.0))
    zsoil = np.r_[0.0, x["diag"]]
    altmax = np.full((NPTS, NVM), initial_altmax)
    altmax_ind = np.zeros((NPTS, NVM), dtype=np.int32)
    last = altmax.copy()
    lasti = altmax_ind.copy()
    results = []
    carbon32 = x["carbon32"].copy()
    doc = x["doc"].copy()
    lbelow = x["lbelow"].copy()
    scin = x["scin"].copy()
    for call_index in range(4):
        if call_index == 2:
            x["veget"][1, 13] = 0.0
            x["mc32"][0, :, :] = 0.0
            x["mc"][0, :, :] = 0.0
            wat[1, :, :] = -0.01
            doc = doc.copy()
            for layer in range(NDEEP):
                doc[0, 13, layer, 0, :, 0] = 1.0 if layer % 2 else 1.0e-9
        if call_index == 3:
            x["veget"][:, 13] = (0.7, 0.6)
            x["mc32"][:] = 0.4
            x["mc"][:] = 0.4
            wat[:] = 0.01
            carbon32 = carbon32.copy()
            carbon32[0, 0, 13, 0] = 1e5
        carbon_before_decomposition = np.sum(carbon32, axis=3)
        soilwater = stomate_soilwater_31mm_ok_leak_inputs(
            soil_mc=x["mc"], z_soil=zsoil, sro_bottom=5
        ).ok_leak_inputs["soilwater_31mm"]
        dayno = 2 if call_index >= 2 else 1
        alt = altcalc_doc(
            tp,
            x["zz"],
            altmax,
            altmax_ind,
            last,
            lasti,
            np.ones((NPTS, NVM), bool),
            firstcall=call_index == 0,
            newaltcalc=newaltcalc,
            dayno=dayno,
            soilc_isspinup=soilc_isspinup,
        )
        alt = altcalc_doc(
            tp,
            x["zz"],
            alt.altmax,
            alt.altmax_ind,
            alt.altmax_lastyear,
            alt.altmax_ind_lastyear,
            np.ones((NPTS, NVM), bool),
            firstcall=False,
            newaltcalc=newaltcalc,
            dayno=dayno,
            soilc_isspinup=soilc_isspinup,
        )
        z_root_before = np.where(
            np.asarray(alt.altmax_lastyear) < 2.0, np.asarray(alt.altmax_lastyear), 2.0
        )
        rootlev_before = np.asarray(alt.altmax_ind_lastyear)
        tf = soilcarbon_leak_tf_doc_inputs(
            pg,
            pc,
            x["veget"],
            x["biomass"],
            is_tree,
            ok_tf_doc=True,
            dt_days=dt,
            conc_doc_rain=3.02,
            doc_incr_per_leaf_m2=0.00092,
        )
        ground = soilcarbon_leak_tf_doc_ground_fluxes(
            tf.doc_precip2ground,
            tf.doc_precip2canopy,
            tf.dry_dep_canopy,
            storage,
            cg,
            x["veget"],
            flood,
            conc_doc_max=100.0,
        )
        peat_flags = false.copy()
        peat_flags[13] = call_index == 3
        thickness = np.diff(np.r_[0.0, x["zz"]])
        peat_soc = 1.0 / ((0.4 * 0.12 + 0.13) ** 2.19) * 0.01
        cmax = 0.12 * 1e6 * peat_soc * thickness
        core = soilcarbon_leak_core_step(
            carbon32,
            doc,
            scin,
            dtop,
            dsub,
            ground.wet_dep_ground,
            ground.wet_dep_flood,
            x["labove"],
            lbelow,
            x["ligna"],
            x["lignb"],
            fbd,
            fb,
            x["mc"],
            x["mc32"],
            wat,
            soilwater,
            runoff,
            drain,
            runpeat,
            fastr,
            pref,
            x["veget"],
            flood,
            np.zeros((NPTS, NVM, NPOOL, 1)),
            tp,
            clay,
            bulk,
            zsoil,
            np.r_[0.0, x["zz"]],
            0.5 * (1 - poor) + poor,
            natural,
            peat_flags,
            false,
            dt_days=dt,
            dif_doc=np.full(NPTS, 1e-5 * dt),
            cue=0.3,
            nslm=NSLM,
            sro_bottom=5,
            priming=call_index != 2,
            perma_peat=call_index == 3,
            cmax_peat=cmax,
            perma_peat_veget_mask=np.ones((NPTS, NVM), dtype=bool),
        )
        carbon_output = (
            np.sum(np.asarray(core.perma_peat.carbon_32l), axis=3)
            if core.perma_peat is not None
            else carbon_before_decomposition
        )
        deepc_output = (
            np.asarray(core.perma_peat.deepc_peat)
            if core.perma_peat is not None
            else np.zeros((NPTS, NDEEP, NVM))
        )
        current = {
            "resp_hetero_soil": np.asarray(core.resp_hetero_soil),
            "resp_flood_soil": np.asarray(core.resp_flood_soil),
            "doc_exp": np.asarray(core.doc_exp),
            "dry_dep_canopy": np.asarray(tf.dry_dep_canopy),
            "doc_precip2ground": np.asarray(tf.doc_precip2ground),
            "doc_precip2canopy": np.asarray(tf.doc_precip2canopy),
            "doc_canopy2ground": np.asarray(ground.doc_canopy2ground),
            "soilcarbon_input_doc": scin,
            "floodcarbon_input": np.zeros((NPTS, NVM, NPOOL, 1)),
            "carbon": carbon_output,
            "carbon_32l": np.asarray(core.carbon_32l),
            "altmax": np.asarray(alt.altmax),
            "fixed_cryoturbation_depth": np.zeros((NPTS, NVM)),
            "litter_below": lbelow,
            "doc": np.asarray(core.doc),
            "interception_storage": np.asarray(ground.interception_storage),
            "biomass": x["biomass"],
            "bulk_dens": bulk,
            "deepc_peat": deepc_output,
            "save_alt": np.asarray(alt.alt),
            "save_alt_ind": np.asarray(alt.alt_ind),
            "save_altmax_lastyear": np.asarray(alt.altmax_lastyear),
            "save_altmax_ind": np.asarray(alt.altmax_ind),
            "save_altmax_ind_lastyear": np.asarray(alt.altmax_ind_lastyear),
            "save_z_root": z_root_before,
            "save_rootlev": rootlev_before,
            "save_veget_mask": np.ones((NPTS, NVM), dtype=np.int32),
        }
        results.append(current)
        carbon32 = current["carbon_32l"]
        doc = current["doc"]
        storage = current["interception_storage"]
        altmax = current["altmax"]
        altmax_ind = current["save_altmax_ind"]
        last = current["save_altmax_lastyear"]
        lasti = current["save_altmax_ind_lastyear"]
        scin = np.zeros_like(scin)
        dtop[:] = 0
        dsub[:] = 0
        pc[:] = 0
    return results


def run_oracle(output_dir: Path, compiler: Path = COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source, hashes = build_source()
    with tempfile.TemporaryDirectory(prefix="orcjax_soilcarbon_owner_") as tmp:
        src = Path(tmp) / "oracle.f90"
        exe = Path(tmp) / "oracle.exe"
        src.write_bytes(source)
        subprocess.run(
            [
                str(compiler),
                "-std=legacy",
                "-cpp",
                "-fdefault-real-8",
                "-ffree-line-length-none",
                "-O0",
                "-fcheck=all",
                str(src),
                "-o",
                str(exe),
            ],
            cwd=tmp,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
        )
        actual_by_scenario = []
        for scenario, newaltcalc, spinup in (
            ("normal", False, False),
            ("newaltcalc_spinup", True, True),
        ):
            scenario_raw = Path(tmp) / f"outputs_{scenario}.bin"
            env = compiler_environment(compiler)
            env["ORACLE_NEWALTCALC"] = "1" if newaltcalc else "0"
            env["ORACLE_SPINUP"] = "1" if spinup else "0"
            subprocess.run(
                [str(exe), str(scenario_raw)],
                cwd=tmp,
                env=env,
                check=True,
                capture_output=True,
            )
            actual_by_scenario.append((scenario, read_fortran(scenario_raw)))
    expected_by_scenario = (
        ("normal", jax_calls()),
        (
            "newaltcalc_spinup",
            jax_calls(
                newaltcalc=True,
                soilc_isspinup=True,
                initial_altmax=1.0,
            ),
        ),
    )
    rows = []
    passed = True
    discrete = {n for n, _ in INT_FIELDS + INT_SAVE_2 + INT_SAVE_3 + LOGICAL_FIELDS}
    for (scenario, actual), (expected_scenario, expected) in zip(
        actual_by_scenario, expected_by_scenario, strict=True
    ):
        if scenario != expected_scenario:
            raise AssertionError(
                f"scenario mismatch: {scenario} != {expected_scenario}"
            )
        for ci, (f, j) in enumerate(zip(actual, expected, strict=True), 1):
            for name in f:
                diff = np.abs(f[name].astype(float) - j[name].astype(float))
                scale = np.maximum(np.abs(f[name]), np.abs(j[name]))
                ok = (
                    bool(np.array_equal(f[name], j[name]))
                    if name in discrete
                    else bool(np.all(diff <= 1e-14 + 1e-12 * scale))
                )
                passed &= ok
                rel = np.zeros_like(diff)
                np.divide(diff, scale, out=rel, where=scale > 0)
                where = (
                    np.unravel_index(int(np.argmax(diff)), diff.shape)
                    if diff.size
                    else ()
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "call": ci,
                        "field": name,
                        "passed": ok,
                        "max_abs_error": float(diff.max(initial=0)),
                        "max_rel_error": float(rel.max(initial=0)),
                        "max_index": str(where),
                        "fortran_at_max": float(f[name][where]) if where else 0.0,
                        "jax_at_max": float(j[name][where]) if where else 0.0,
                    }
                )
    result = {
        "status": "passed" if passed else "failed",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "span_sha256": hashes,
        "comparisons": rows,
    }
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "npts": 2,
                "nvm": 14,
                "scenarios": [
                    {"id": "normal", "newaltcalc": False, "soilc_isspinup": False},
                    {
                        "id": "newaltcalc_spinup",
                        "newaltcalc": True,
                        "soilc_isspinup": True,
                    },
                ],
                "calls": [
                    "firstcall-warm-cold-flood",
                    "same-day-later-save-reuse",
                    "day2-dry-negative-flux",
                    "perma-peat-redistribution",
                ],
                "arms": [
                    "decomposition",
                    "adsorption",
                    "transport",
                    "diffusion",
                    "export",
                    "tf-doc",
                    "active-layer",
                    "perma-peat",
                ],
                "switches": {
                    "priming": [True, True, False, True],
                    "dayno": [1, 1, 2, 2],
                    "perma_peat": [False, False, False, True],
                },
                "day2_doc_profile": "warm-point alternating 1e-9/1.0 by layer; frozen point retains prior state",
            },
            indent=2,
        )
        + "\n"
    )
    with (output_dir / "point_comparisons.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    if not passed:
        raise AssertionError([r for r in rows if not r["passed"]])
    return result


if __name__ == "__main__":
    print(
        run_oracle(
            ROOT / "outputs/reference_mode/micro_oracles/stomate_soilcarbon_owner"
        )["status"]
    )
