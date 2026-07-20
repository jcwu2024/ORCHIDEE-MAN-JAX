from __future__ import annotations
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from fortran_oracle_common import (
    DEFAULT_COMPILER,
    ROOT,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FAMILY = "surface_diffuco_co2_fvcb"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_surface_fvcb.f90.template"


def _segment():
    lines = SOURCE.read_bytes().splitlines(keepends=True)
    data = b"".join(lines[2704:2786])
    return data, hashlib.sha256(data).hexdigest()


def _source_segment(start: int, end: int):
    lines = SOURCE.read_bytes().splitlines(keepends=True)
    data = b"".join(lines[start - 1 : end])
    return data, hashlib.sha256(data).hexdigest()


def _activity_segment():
    lines = SOURCE.read_bytes().splitlines(keepends=True)
    data = b"".join(lines[2337:2348]) + b"".join(lines[2357:2400])
    return data, hashlib.sha256(data).hexdigest()


def _jax():
    from jax_orchidee.sechiba.diffuco import (
        diffuco_apply_pft14_assimtot_controls,
        diffuco_c3_assimilation_yin_layer,
        diffuco_trans_co2_outputs_from_fvcb,
        diffuco_trans_co2_activity,
        diffuco_co2_downregulated_vcmax,
    )

    down_assim = np.full((3, 14), 50.0)
    down_coeff = np.full(14, 0.1)
    down_coeff[13] = 0.2
    down_ca = np.array([300.0, 400.0, 300.0])
    activity = diffuco_trans_co2_activity(
        swdown=np.array([100.0, 0.0, 100.0]),
        humrel=np.pad(np.array([[0.5], [0.5], [0.0]]), ((0, 0), (13, 0))),
        veget=np.pad(np.full((3, 1), 0.7), ((0, 0), (13, 0))),
        veget_max=np.pad(np.full((3, 1), 0.8), ((0, 0), (13, 0))),
        lai=np.pad(np.array([[2.0], [0.0], [2.0]]), ((0, 0), (13, 0))),
        qsintveg=np.pad(np.array([[0.1], [0.0], [0.0]]), ((0, 0), (13, 0))),
        qsintmax=np.pad(np.array([[0.2], [0.0], [0.0]]), ((0, 0), (13, 0))),
        temp_growth=np.full(3, 25.0),
        tphoto_min=np.zeros(14),
        tphoto_max=np.full(14, 40.0),
        ok_laidev=np.array([False] * 13 + [True]),
    )
    args = {
        "vc2": np.array([30.0, 60.0, 33.007402396106684]),
        "jj": np.array([60.0, 120.0, 6221.1054654979025]),
        "rd": np.array([0.5, 1.0, 5.077229369829301e-5]),
        "ca": np.array([300.0, 400.0, 454.544212940868]),
        "gamma_star": np.array([35.0, 42.0, 1676.3152557650226]),
        "kmc": np.array([350.0, 400.0, 33907.936022990296]),
        "kmo": np.array([250.0, 280.0, 848062.7445970513]),
        "sco": np.array([2500.0, 2600.0, 2414.3708895507157]),
        "gm": np.array([0.15, 0.2, 1.3697344303394207e-8]),
        "gb_co2": np.array([0.8, 1.0, 1.3288257015011667e-4]),
        "g0var": np.array([0.005, 0.01, 2.573931168052183]),
        "fvpd": np.array([3.0, 5.0, 175.70077906202914]),
    }
    r = diffuco_c3_assimilation_yin_layer(**args)
    result = {
        "vcmax_no_downreg": np.asarray(diffuco_co2_downregulated_vcmax(down_assim, down_ca, downregulation_co2=False, downregulation_co2_coeff=down_coeff, downregulation_co2_baselevel=350.0))[:, 13],
        "vcmax_downreg": np.asarray(diffuco_co2_downregulated_vcmax(down_assim, down_ca, downregulation_co2=True, downregulation_co2_coeff=down_coeff, downregulation_co2_baselevel=350.0))[:, 13],
        "activity_assimilate": np.asarray(activity.assimilate[:, 13], dtype=np.float64),
        "activity_zqsvegrap": np.asarray(activity.zqsvegrap)[:, 13],
        "activity_water_lim": np.asarray(activity.water_lim)[:, 13],
        "activity_nia": np.asarray([np.count_nonzero(activity.assimilate[:, 13])], dtype=np.float64),
        "activity_nina": np.asarray([3 - np.count_nonzero(activity.assimilate[:, 13])], dtype=np.float64),
        "assimi": np.asarray(r.assimi),
        "cc": np.asarray(r.cc),
        "leaf_ci": np.asarray(r.leaf_ci),
        "gs": np.asarray(r.gs),
        "info": np.asarray(r.info_limitphoto),
        "empty_assimi": np.array([0.0]),
        "empty_cc": np.array([0.0]),
        "empty_leaf_ci": np.array([300.0]),
        "empty_gs": np.array([0.0]),
        "empty_info": np.array([0.0]),
        "mask_layer1": np.array([1.0, 1.0, 0.0]),
        "mask_layer1_nic": np.array([2.0]),
        "mask_layer3": np.array([0.0, 0.0, 0.0]),
        "mask_layer3_nic": np.array([0.0]),
        "mask_empty_nic": np.array([0.0]),
        "integrate_assimtot": np.array([4.5, 6.0, 0.0]),
        "integrate_rdtot": np.array([0.45, 0.6, 0.0]),
        "integrate_gstot": np.array([0.045, 0.06, 0.0]),
        "integrate_leaf_gs_top": np.array([0.01, 0.02, 0.0]),
        "integrate_ilai": np.array([2.0, 2.0, 0.0]),
        "cim": np.array([300.0, (310.0 * 0.5 + 311.0) / 1.5, (320.0 * 0.5 + 321.0 + 322.0 * 1.5) / 3.0]),
        "cim_laisum": np.array([0.5, 1.5, 3.0]),
    }
    at = diffuco_apply_pft14_assimtot_controls(
        np.array([8.0, 12.0, 5.0]),
        control_salinity=np.array([1.0, 0.8, 0.5]),
        control_inudate=np.array([1.0, 0.6, 0.2]),
    )
    n, nvm = 3, 14
    shape = (n, nvm)

    def pft14(values):
        out = np.zeros(shape)
        out[:, 13] = values
        return out

    out = diffuco_trans_co2_outputs_from_fvcb(
        assimilate=pft14([True, True, True]).astype(bool),
        assimtot=pft14(np.asarray(at)), rdtot=pft14([0.4, 0.7, 0.2]),
        gstot=pft14([0.15, 0.22, 0.08]), leaf_gs_top=pft14([0.06, 0.08, 0.04]),
        gamma_star=pft14([40.0, 42.0, 45.0]), fvpd=pft14([3.0, 5.0, 4.0]),
        g0var=pft14([0.005, 0.01, 0.008]), laisum=pft14([1.0, 2.0, 0.5]),
        ilai=np.where(pft14([1, 2, 1]) > 0, pft14([1, 2, 1]), 1).astype(np.int32),
        laitab=np.array([0.0, 0.5, 1.5, 3.0]),
        veget_max=pft14([0.7, 0.8, 0.4]), humrel=pft14([0.5, 0.0, 0.7]),
        zqsvegrap=pft14([0.1, 0.2, 0.05]), vbeta23=pft14([0.02, 0.03, 0.01]),
        t2m=np.array([298.0, 285.0, 275.0]), pb=np.array([1000.0, 950.0, 1020.0]),
        wind=np.array([0.05, 2.0, 5.0]), q_cdrag=np.array([0.01, 0.02, 0.015]),
        q_cdrag_pft=pft14([0.012, 0.025, 0.018]), rveg_pft=np.ones(nvm),
        ca=np.array([300.0, 400.0, 420.0]), rstruct_const=np.full(nvm, 1e-8),
        ok_laidev=np.array([False] * 13 + [True]), dt_sechiba=1800.0,
    )
    result["assimtot_controlled"] = np.asarray(at)
    for name in ("gsmean", "cimean", "gpp", "rveget", "rstruct", "vbeta3", "vbeta3pot"):
        result[name] = np.asarray(getattr(out, name))[:, 13]
    baseline = diffuco_trans_co2_outputs_from_fvcb(
        assimilate=pft14([True, True, True]).astype(bool),
        assimtot=pft14([8.0, 12.0, 5.0]), rdtot=pft14([0.4, 0.7, 0.2]),
        gstot=pft14(np.array([0.005, 0.01, 0.008]) * np.array([1.0, 2.0, 0.5])),
        leaf_gs_top=pft14([0.06, 0.08, 0.04]), gamma_star=pft14([40.0, 42.0, 45.0]),
        fvpd=pft14([3.0, 5.0, 4.0]), g0var=pft14([0.005, 0.01, 0.008]),
        laisum=pft14([1.0, 2.0, 0.5]),
        ilai=np.where(pft14([1, 2, 1]) > 0, pft14([1, 2, 1]), 1).astype(np.int32),
        laitab=np.array([0.0, 0.5, 1.5, 3.0]), veget_max=pft14([0.7, 0.8, 0.4]),
        humrel=pft14([0.5, 0.0, 0.7]), zqsvegrap=pft14([0.1, 0.2, 0.05]),
        vbeta23=pft14([0.02, 0.03, 0.01]), t2m=np.array([298.0, 285.0, 275.0]),
        pb=np.array([1000.0, 950.0, 1020.0]), wind=np.array([0.05, 2.0, 5.0]),
        q_cdrag=np.array([0.01, 0.02, 0.015]), q_cdrag_pft=pft14([0.012, 0.025, 0.018]),
        rveg_pft=np.ones(nvm), ca=np.array([300.0, 400.0, 420.0]),
        rstruct_const=np.full(nvm, 1e-8), ok_laidev=np.array([False] * 13 + [True]),
        dt_sechiba=1800.0,
    )
    result["cimean_baseline"] = np.asarray(baseline.cimean)[:, 13]
    inactive = diffuco_trans_co2_outputs_from_fvcb(
        assimilate=pft14([False, False, False]).astype(bool),
        assimtot=pft14([8.0, 12.0, 5.0]), rdtot=pft14([0.4, 0.7, 0.2]),
        gstot=pft14([0.15, 0.22, 0.08]), leaf_gs_top=pft14([0.06, 0.08, 0.04]),
        gamma_star=pft14([40.0, 42.0, 45.0]), fvpd=pft14([3.0, 5.0, 4.0]),
        g0var=pft14([0.005, 0.01, 0.008]), laisum=pft14([1.0, 2.0, 0.5]),
        ilai=np.where(pft14([1, 2, 1]) > 0, pft14([1, 2, 1]), 1).astype(np.int32),
        laitab=np.array([0.0, 0.5, 1.5, 3.0]),
        veget_max=pft14([0.7, 0.8, 0.4]), humrel=pft14([0.5, 0.0, 0.7]),
        zqsvegrap=pft14([0.1, 0.2, 0.05]), vbeta23=pft14([0.02, 0.03, 0.01]),
        t2m=np.array([298.0, 285.0, 275.0]), pb=np.array([1000.0, 950.0, 1020.0]),
        wind=np.array([0.05, 2.0, 5.0]), q_cdrag=np.array([0.01, 0.02, 0.015]),
        q_cdrag_pft=pft14([0.012, 0.025, 0.018]), rveg_pft=np.ones(nvm),
        ca=np.array([300.0, 400.0, 420.0]), rstruct_const=np.full(nvm, 1e-8),
        ok_laidev=np.array([False] * 13 + [True]), dt_sechiba=1800.0,
    )
    for name in ("gsmean", "cimean", "gpp", "rveget", "rstruct", "vbeta3", "vbeta3pot"):
        result[f"inactive_{name}"] = np.asarray(getattr(inactive, name))[:, 13]
    return result


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fortran_outputs.csv"
    segment, sha = _segment()
    downreg, downreg_sha = _source_segment(2240, 2246)
    activity, activity_sha = _activity_segment()
    control, control_sha = _source_segment(2834, 2844)
    outputs, outputs_sha = _source_segment(2878, 2974)
    mask, mask_sha = _source_segment(2519, 2530)
    integrate, integrate_sha = _source_segment(2789, 2831)
    cim, cim_sha = _source_segment(2861, 2872)
    with tempfile.TemporaryDirectory(prefix="orchidee_fvcb_") as td:
        source = Path(td) / "oracle.f90"
        source.write_bytes(
            TEMPLATE.read_bytes()
            .replace(b"! <DOWNREG_SEGMENT>", downreg)
            .replace(b"! <ACTIVITY_SEGMENT>", activity)
            .replace(b"! <C3_SEGMENT>", segment)
            .replace(b"! <C3_EMPTY_SEGMENT>", segment)
            .replace(b"! <MASK_SEGMENT>", mask)
            .replace(b"! <INTEGRATE_SEGMENT>", integrate)
            .replace(b"! <CIM_SEGMENT>", cim)
            .replace(b"! <CONTROL_SEGMENT>", control)
            .replace(b"! <OUTPUT_SEGMENT>", outputs)
        )
        exe = Path(td) / "oracle.exe"
        meta = compile_fortran(source, exe, compiler)
        subprocess.run(
            [str(exe), str(csv_path.resolve())],
            cwd=td,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    values = {}
    with csv_path.open(newline="", encoding="ascii") as h:
        for row in csv.DictReader(h):
            values.setdefault(row["field"], []).append(float(row["value"]))
    f = {k: np.asarray(v) for k, v in values.items()}
    j = _jax()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "three C3 Yin/FvCB layer states",
                    "PFT14 controlled output writeback",
                ],
                "downregulation_ca": [300.0, 400.0, 300.0],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_point_comparisons(
        output_dir / "point_comparisons.csv", f, j, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(k, v, j[k], rtol=1e-12, atol=1e-14) for k, v in f.items()
    ]
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "ledger_entries": ["diffuco.active.co2_fvcb"],
            "source_segment": {
                "start_line": 2705,
                "end_line": 2786,
                "span_sha256": sha,
            },
            "additional_source_segments": [
                {"start_line": 2240, "end_line": 2246, "span_sha256": downreg_sha},
                {"start_line": 2338, "end_line": 2400, "preprocessed_exclusion": "2350-2355 STRICT_CHECK", "span_sha256": activity_sha},
                {"start_line": 2519, "end_line": 2530, "span_sha256": mask_sha},
                {"start_line": 2789, "end_line": 2831, "span_sha256": integrate_sha},
                {"start_line": 2834, "end_line": 2844, "span_sha256": control_sha},
                {"start_line": 2861, "end_line": 2872, "span_sha256": cim_sha},
                {"start_line": 2878, "end_line": 2974, "span_sha256": outputs_sha},
            ],
            "build": meta,
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "passed")
