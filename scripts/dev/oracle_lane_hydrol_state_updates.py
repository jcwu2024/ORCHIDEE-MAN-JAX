from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import hydrol_tmc_update, hydrol_vegupd_static_state  # noqa: E402
from jax_orchidee.sechiba.hydrol_thermosoil_completion import hydrol_rotation_update  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)
from scripts.dev.oracle_lane_hydrol_soil_owner import (  # noqa: E402
    _allocation_and_reset_source,
    _module_state,
)

SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_hydrol_state_updates.f90.template"
PROCEDURES = ("hydrol_tmc_update", "hydrol_rotation_update", "hydrol_vegupd")
PREF = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1, 2], dtype=np.int32)
DZ = np.array([2, 4, 8, 12, 18, 24, 30.0], dtype=float)


def compose(path: Path) -> dict[str, str]:
    spans = {n: extract_procedure_bytes(SOURCE, n) for n in PROCEDURES}
    data = TEMPLATE.read_bytes()
    data = data.replace(b"! <MODULE_STATE>", _module_state()).replace(
        b"! <STATE_SETUP>", _allocation_and_reset_source()
    )
    data = data.replace(
        b"! <HYDROL_PROCEDURES>", b"\n\n".join(spans[n].span_bytes for n in PROCEDURES)
    )
    path.write_bytes(data)
    return {n: s.span_sha256 for n, s in spans.items()}


def _seed(n=6):
    mc = np.empty((n, 7, 6))
    for k in range(6):
        for j in range(7):
            for i in range(n):
                mc[i, j, k] = 0.1 + 0.01 * (i + 1) + 0.02 * (j + 1) + 0.03 * (k + 1)
    water = np.arange(1, n * 6 + 1, dtype=float).reshape((n, 6), order="F") / 10
    soil = np.full((n, 6), 1 / 6)
    qs = np.full((n, 14), 0.01)
    # The Fortran seed leaves tmc zero; that is intentional state input.
    return dict(
        mc=mc,
        water2infilt=water,
        tmc=np.zeros((n, 6)),
        resdist=soil.copy(),
        qsintveg=qs,
        vegtot_old=np.full(n, 0.8),
        vegtot=np.full(n, 0.8),
    )


def _state(result):
    return {
        name: np.asarray(getattr(result, name)).ravel(order="F")
        for name in (
            "mc",
            "water2infilt",
            "tmc",
            "resdist",
            "qsintveg",
            "drain_upd",
            "runoff_upd",
            "humtot",
        )
    }


def _jax_cases():
    n = 6
    base = _seed(n)
    vmax = np.zeros((n, 14))
    vmax[:, 0] = 0.2
    vmax[:, 13] = 0.6
    veget = vmax.copy()
    soil = np.full((n, 6), 1 / 6)
    base["vegtot_old"] = np.array([0.5, 0.8, 0.8, 0.8, 0.8, 0])
    base["vegtot"] = np.array([0.8, 0.5, 0, 0.8, 0.8, 0.8])
    vmax[2, 0] = 0
    veget[2, 0] = 0
    vmax[3, 13] = 0
    base["qsintveg"][3, 13] = 0.4
    soil[0] = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
    soil[1] = [0.05, 0.25, 0.2, 0.2, 0.2, 0.1]
    soil[2] = [0, 0.2, 0.2, 0.2, 0.2, 0.2]
    one = hydrol_tmc_update(
        veget_max=vmax, soiltile=soil, pref_soil_veg=PREF, dz_mm=DZ, **base
    )
    cases = {1: _state(one)}
    two = hydrol_tmc_update(
        veget_max=vmax,
        soiltile=soil,
        pref_soil_veg=PREF,
        dz_mm=DZ,
        mc=one.mc,
        water2infilt=one.water2infilt,
        tmc=one.tmc,
        resdist=one.resdist,
        qsintveg=one.qsintveg,
        vegtot_old=base["vegtot_old"],
        vegtot=base["vegtot"],
    )
    static = hydrol_vegupd_static_state(
        veget=veget,
        veget_max=vmax,
        soiltile=soil,
        vegtot=base["vegtot"],
        pref_soil_veg=PREF,
    )
    cases[2] = {
        **_state(two),
        **{
            name: np.asarray(getattr(static, name)).ravel(order="F")
            for name in (
                "frac_bare",
                "mask_veget",
                "mask_soiltile",
                "frac_bare_ns",
                "vegetmax_soil",
            )
        },
    }
    base3 = _seed(n)
    v3 = np.zeros((n, 14))
    v3[:, 0] = 0.2
    v3[:, 13] = 0.6
    first3 = hydrol_tmc_update(
        veget_max=v3,
        soiltile=np.full((n, 6), 1 / 6),
        pref_soil_veg=PREF,
        dz_mm=DZ,
        **base3,
    )
    three = hydrol_tmc_update(
        veget_max=v3,
        soiltile=np.full((n, 6), 1 / 6),
        pref_soil_veg=PREF,
        dz_mm=DZ,
        mc=first3.mc,
        water2infilt=first3.water2infilt,
        tmc=first3.tmc,
        resdist=first3.resdist,
        qsintveg=first3.qsintveg,
        vegtot_old=base3["vegtot_old"],
        vegtot=base3["vegtot"],
        check_cwrr=True,
    )
    cases[3] = _state(three)
    b4 = _seed(n)
    b4["qsintveg"][:] = 0
    b4["qsintveg"][:, [1, 13]] = 0.01
    old = np.zeros(14)
    old[1] = 0.2
    old[13] = 0.6
    v4 = np.zeros((n, 14))
    v4[:, 1] = 0.35
    v4[:, 13] = 0.45
    rot = np.zeros((14, 14))
    rot[13, 1] = 0.25
    four = hydrol_rotation_update(
        ip_fortran=1,
        rot_matrix=rot,
        old_veget_max=old,
        veget_max=v4,
        soiltile=np.full((n, 6), 1 / 6),
        qsintveg=b4["qsintveg"],
        pref_soil_veg=PREF,
        mc=b4["mc"],
        water2infilt=b4["water2infilt"],
        tmc=b4["tmc"],
        humtot=np.zeros(n),
        resdist=b4["resdist"],
        dz=DZ,
    )
    cases[4] = {
        name: np.asarray(getattr(four, name)).ravel(order="F")
        for name in ("mc", "water2infilt", "tmc", "resdist", "qsintveg", "humtot")
    }
    cases[4].update({"drain_upd": np.zeros(n), "runoff_upd": np.zeros(n)})
    return cases


def _read(path):
    out = {}
    with path.open(newline="", encoding="ascii") as h:
        for r in csv.DictReader(h):
            out.setdefault(int(r["case_id"]), {}).setdefault(r["field"], []).append(
                float(r["value"])
            )
    return {c: {f: np.asarray(v) for f, v in x.items()} for c, x in out.items()}


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER):
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_hydrol_updates_") as td:
        src = Path(td) / "oracle.f90"
        exe = Path(td) / "oracle.exe"
        hashes = compose(src)
        meta = compile_fortran(src, exe, compiler)
        subprocess.run(
            [str(exe)],
            cwd=td,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        shutil.copyfile(
            Path(td) / "fortran_outputs.csv", output_dir / "fortran_outputs.csv"
        )
    f = _read(output_dir / "fortran_outputs.csv")
    j = _jax_cases()
    comps = []
    ff = {}
    jj = {}
    for c, fields in f.items():
        for field, a in fields.items():
            name = f"case{c}.{field}"
            ff[name] = a
            jj[name] = j[c][field]
            comps.append(float_comparison(name, a, jj[name], rtol=1e-12, atol=1e-14))
    write_point_comparisons(
        output_dir / "point_comparisons.csv", ff, jj, rtol=1e-12, atol=1e-14
    )
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": ["mixed_updates", "vegupd_writeback", "no_update", "rotation"],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    return write_result(
        output_dir,
        "hydrol_state_updates",
        comps,
        {
            **meta,
            "source_spans": hashes,
            "tolerance_policy": {"rtol": 1e-12, "atol": 1e-14},
        },
    )
