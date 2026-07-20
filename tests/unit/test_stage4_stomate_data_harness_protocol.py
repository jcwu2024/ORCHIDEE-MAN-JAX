"""Source-provenance checks for the local Stage4 stomate_data package."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/dev/oracle_stage4_stomate_data.py"


def _module():
    spec = importlib.util.spec_from_file_location("stage4_stomate_data_generator", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_job0_bio_pft14_transform_replays_literal_remplace_and_records_source(tmp_path):
    generator = _module()
    output = tmp_path / "stage4_stomate_data"
    generator.build(output)

    transformed = (output / "run.def").read_text(encoding="ascii")
    assert "NVM=14 \n" in transformed
    assert "SECHIBA_VEGMAX__00014=1.0 \n" in transformed
    assert "PFT_TO_MTC__00014=2 \n" in transformed
    assert "IS_PEAT__00014=y \n" in transformed
    assert "tides=y \n" in transformed
    assert "#NVM = 13" not in transformed  # Job0_bio sed removes substring matches.

    provenance = json.loads((output / "source_provenance.json").read_text(encoding="utf-8"))
    assert provenance["compiled_nvm"] == 14
    assert provenance["fortran_pft_index"] == 14
    assert provenance["jax_axis_index"] == 13
    assert provenance["run_def_source"] == "fortran_run_scripts/paper_250919/run.def.vn"
    assert "Job0_bio::remplace" in provenance["remplace_provenance"]
    assert provenance["run_def_source_sha256"] == hashlib.sha256(
        (ROOT / "fortran_run_scripts/paper_250919/run.def.vn").read_bytes()
    ).hexdigest()
    assert provenance["job_source_sha256"] == hashlib.sha256(
        (ROOT / "fortran_run_scripts/paper_250919/Job0_bio").read_bytes()
    ).hexdigest()


def test_pft14_driver_and_comparator_do_not_reintroduce_synthetic_pft2_fixture(tmp_path):
    generator = _module()
    output = tmp_path / "stage4_stomate_data"
    generator.build(output)

    driver = (output / "stage4_stomate_data_pft14.f90").read_text(encoding="ascii")
    assert "nvm /= 14_i_std" in driver
    assert "migrate_pft14" in driver
    assert "migrate_pft2" not in driver
    assert "pft = 14_i_std" in driver
    assert "schema_version', 3" in driver
    for required in (
        "bm_sapl_initial_pft14", "is_grassland_manag_pft14",
        "pheno_gdd_crit_pft14", "senescence_temp_pft14",
        "use_age_class", "bm_sapl_leaf", "maxdia_coeff", "zero_celsius",
    ):
        assert required in driver
    comparator = (output / "compare_capture.py").read_text(encoding="ascii")
    assert "strict comparison requires schema" in comparator
    assert "numeric fixture" in comparator


def test_current_successful_server_capture_is_rejected_until_it_has_complete_schema3_inputs():
    comparator_path = ROOT / "scripts/dev/compare_stage4_stomate_data_capture.py"
    spec = importlib.util.spec_from_file_location("stage4_stomate_data_comparator", comparator_path)
    assert spec and spec.loader
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)

    capture = ROOT / "outputs/reference_mode/micro_oracles/stage4_stomate_data/fortran_server_pft14_run.log"
    try:
        comparator.parse_capture(capture)
    except ValueError as error:
        assert "schema 3" in str(error)
    else:
        raise AssertionError("incomplete pre-schema-3 capture must fail closed")
