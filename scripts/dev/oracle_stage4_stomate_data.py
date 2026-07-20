"""Emit the local, protocol-faithful PFT14 ``stomate_data::data`` harness.

The package is intentionally not an oracle result or a closure claim.  It
replays the relevant literal ``Job0_bio::remplace`` writes over ``run.def.vn``
and asks a linked original executable to capture its configured PFT14 inputs.
The generated JAX adapter derives expectations from that capture, never from a
second, synthetic PFT fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FAMILY = "stage4_stomate_data"
JOB = ROOT / "fortran_run_scripts/paper_250919/Job0_bio"
RUN_DEF_VN = ROOT / "fortran_run_scripts/paper_250919/run.def.vn"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_data.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_stage4_stomate_data.f90.template"
COMPARATOR = ROOT / "scripts/dev/compare_stage4_stomate_data_capture.py"
DEFAULT_OUTPUT = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
PFT = 14

# Literal active-path writes in Job0_bio.  These are deliberately a small,
# isolated transform: path/year-dependent shell substitutions are not a
# source-config input to pft_parameters_main and are not guessed here.
# Provenance is the Job0_bio remplace body, lines 62-77, 159-163, and 182.
JOB0_BIO_PFT14_REPLACES = (
    ("IMPOSE_VEG", "y"), ("LAND_COVER_CHANGE", "n"),
    *((f"SECHIBA_VEGMAX__{i:05d}", "0.0") for i in range(1, 14)),
    ("SECHIBA_VEGMAX__00014", "1.0"),
    ("STOMATE_OK_DGVM", "n"), ("GLUC_USE_AGE_CLASS", "n"),
    ("NSTM", "6"), ("NVM", "14"), ("PREF_SOIL_VEG__00014", "4"),
    ("PFT_TO_MTC__00014", "2"), ("PEAT_NODR", "y"), ("OK_RU2PEAT", "y"),
    ("OK_WT_AB", "y"), ("max_wt_ab", "100."), ("IS_PEAT__00014", "y"),
    ("tides", "y"),
)

# This Oracle has no XIOS server because it does not create model output.
# The override replaces external transport only; PFT14 science configuration
# above remains the exact active Job0_bio transform.
ORACLE_TRANSPORT_REPLACES = (("XIOS_ORCHIDEE_OK", "n"),)


def remplace(lines: list[str], key: str, value: str) -> list[str]:
    """Reproduce Job0_bio's ``sed "/$1/D"`` then append behavior exactly."""
    return [line for line in lines if key not in line] + [f"{key}={value} \n"]


def transform_paper_run_def(run_def_vn: Path = RUN_DEF_VN) -> str:
    """Return the isolated, source-ordered PFT14 Job0_bio run.def transform."""
    lines = run_def_vn.read_text(encoding="ascii").splitlines(keepends=True)
    for key, value in JOB0_BIO_PFT14_REPLACES:
        lines = remplace(lines, key, value)
    for key, value in ORACLE_TRANSPORT_REPLACES:
        lines = remplace(lines, key, value)
    return "".join(lines)


def build(output: Path) -> None:
    if output.name != FAMILY:
        raise ValueError(f"output must be the isolated {FAMILY!r} family")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.def").write_text(transform_paper_run_def(), encoding="ascii")
    (output / "stage4_stomate_data_pft14.f90").write_text(
        TEMPLATE.read_text(encoding="ascii"), encoding="ascii"
    )
    (output / "compare_capture.py").write_text(COMPARATOR.read_text(encoding="ascii"), encoding="ascii")
    legacy_adapter = output / "derive_jax_expected.py"
    if legacy_adapter.exists():
        legacy_adapter.unlink()
    provenance = {
        "schema_version": 3,
        "case": "Job0_bio_active_tides_pft14",
        "fortran_pft_index": PFT,
        "jax_axis_index": PFT - 1,
        "compiled_nvm": 14,
        "run_def_source": str(RUN_DEF_VN.relative_to(ROOT)).replace("\\", "/"),
        "run_def_source_sha256": hashlib.sha256(RUN_DEF_VN.read_bytes()).hexdigest(),
        "job_source": str(JOB.relative_to(ROOT)).replace("\\", "/"),
        "job_source_sha256": hashlib.sha256(JOB.read_bytes()).hexdigest(),
        "data_source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "data_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "remplace_provenance": "Job0_bio::remplace lines 31-37; active PFT14 writes lines 62-77, 159-163, 182",
        "literal_replaces": list(JOB0_BIO_PFT14_REPLACES),
        "external_transport_overrides": list(ORACLE_TRANSPORT_REPLACES),
        "owner_input_provenance": {
            "destination_arrays_and_pft14_vectors": "stomate_data.f90::data lines 261-588; pft_parameters_main post-configuration state",
            "phenology_and_senescence_coefficients": "stomate_data.f90::data lines 159-180 and 443-538; pft_parameters_var module state",
            "flags_and_leaf_age_dimension": "stomate_data.f90::data lines 303-348 and 574-588; pft_parameters/constantes module state",
            "global_constants": "src_parameters/constantes_var.f90 lines 1087-1180, initialized by src_parameters/constantes.f90 lines 1246-1566",
            "pi": "src_parameters/constantes_var.f90; consumed by stomate_data.f90::data lines 272 and 284",
        },
        "capture_schema": "STAGE4_SOURCE_CONFIG schema 3 is the complete PFT14 stomate_data_owner input boundary; STAGE4_DATA is the sole expected-output source",
        "comparator": "compare_capture.py parses the supplied capture and fails closed on absent source records; no closure claim",
    }
    (output / "source_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "Local transfer package only. run.def is the isolated Job0_bio PFT14 transform of run.def.vn. "
        "Link the driver to original objects, capture its schema-3 post-pft_parameters_main PFT14 source inputs and STAGE4_DATA records, then run compare_capture.py <capture.log>. "
        "The comparator rejects older/incomplete logs, uses no numeric fixture, and makes no closure claim.\n", encoding="ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())
    print(f"WROTE {args.output} pft={PFT} nvm=14")


if __name__ == "__main__":
    main()
