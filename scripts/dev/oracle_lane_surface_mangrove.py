from __future__ import annotations
import csv
import json
import hashlib
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
FAMILY = "surface_diffuco_mangrove_controls"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_surface_mangrove.f90.template"


def _segment():
    lines = SOURCE.read_bytes().splitlines(keepends=True)
    # Line 623 is the syntactic ENDIF closing the scientific ledger span 445-622.
    data = b"".join(lines[444:623])
    return data, hashlib.sha256(data).hexdigest()


def _jax():
    from jax_orchidee.sechiba.diffuco import (
        ControlInundateInputs,
        mangrove_control_inundation,
        mangrove_control_salinity,
        z_soil_from_diaglev,
    )

    n, nvm = 2, 14
    sal = np.array([5.0, 35.0])
    rp = np.full((n, nvm), 0.5)
    rp[:, 13] = [0.35, 0.8]
    tide = np.empty((n, 1006))
    tide[:, :6] = np.array(
        [[-1.2, -0.5, -0.1, 0.0, 0.2, 0.8], [-2.0, -0.8, -0.2, 0.1, 0.5, 1.5]]
    )
    tide[:, 6:] = tide[:, 5:6]
    bio = np.ones((n, nvm, 12, 1))
    bio[:, 13, 8, 0] = [20.0, 150.0]
    bio[:, 13, 10, 0] = [10.0, 100.0]
    bio[:, 13, 9, 0] = [5.0, 30.0]
    bio[:, 13, 11, 0] = [3.0, 20.0]
    bio[:, 1, 9, 0] = 0.0
    bio[:, 1, 11, 0] = 0.0
    result = mangrove_control_inundation(
        ControlInundateInputs(
            z_soil=z_soil_from_diaglev(np.array([0.1, 0.4, 1.0])),
            rprof=rp,
            tide_height=tide,
            biomass=bio,
        ),
        control_inudate_min=0.5,
        agb_agr_ven_all_st=200.0,
        agb_agr_ven_all_pn=40.0,
        h_agr_max_st=2.0,
        h_agr_max_pn=0.4,
    )
    outputs = {
        "control_salinity": np.asarray(
            mangrove_control_salinity(sal, control_salinity_min=0.5)
        ),
        "control_inudate": np.asarray(result.control_inudate),
        "agb_st": np.asarray(result.agr.agb_agr_st),
        "agb_pn": np.asarray(result.agr.agb_agr_pn),
        "h_st": np.asarray(result.agr.h_agr_st),
        "h_pn": np.asarray(result.agr.h_agr_pn),
        "frac_root_soil": np.asarray(result.frac_root_soil).ravel(order="F"),
        "n_soil_inudate": np.asarray(result.n_soil_inudate).ravel(order="F"),
        "frac_root_inudate": np.asarray(result.frac_root_inudate).ravel(order="F"),
        "frac_root_ventilate": np.asarray(result.ventilation.frac_root_ventilate).ravel(
            order="F"
        ),
        "frac_root_anoxia": np.asarray(result.frac_root_anoxia).ravel(order="F"),
    }
    outputs["later_control_salinity"] = outputs["control_salinity"]
    outputs["later_control_inudate"] = outputs["control_inudate"]
    outputs["later_firstcall_flag"] = np.asarray([0.0])
    return outputs


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fortran_outputs.csv"
    segment, segment_hash = _segment()
    with tempfile.TemporaryDirectory(prefix="orchidee_mangrove_") as td:
        source = Path(td) / "oracle.f90"
        source.write_bytes(
            TEMPLATE.read_bytes().replace(b"! <MANGROVE_SEGMENT>", segment)
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
                    "salinity clipped/unclipped",
                    "negative/positive tide",
                    "low/high AGR biomass",
                ],
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
            "ledger_entries": ["diffuco.active.mangrove_controls"],
            "source_segment": {
                "file": str(SOURCE.relative_to(ROOT)),
                "start_line": 445,
            "end_line": 623,
                "span_sha256": segment_hash,
            },
            "build": meta,
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "passed")
