from __future__ import annotations

from pathlib import Path

import pytest

from jax_orchidee.driver.orchestration import prepare_paper_1961_driver_context
from jax_orchidee.driver.reference_layout import (
    discover_paper_landpoint_ids,
    inventory_paper_references,
    resolve_paper_landpoint_reference,
)
from jax_orchidee.driver.run_def_materialization import materialize_case_run_def_values, write_materialized_run_def
from jax_orchidee.stomate.reference import default_paper_modelout_csv_path, find_stomate_reference_files, read_paper_modelout_csv


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
BASE_USED_RUN_DEF = ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"


def test_reference_layout_resolves_complete_raw_output_as_canonical():
    references = {
        reference.landpoint_id: reference
        for reference in inventory_paper_references(ROOT)
        if reference.landpoint_id in {"001.0-071.0", "001.0-073.0", "003.0-077.0", "005.0-069.0"}
    }

    assert set(references) == {"001.0-071.0", "001.0-073.0", "003.0-077.0", "005.0-069.0"}
    for reference in references.values():
        assert reference.layout == "raw_server_copy"
        assert reference.output_dir is not None
        assert "reference\\OUT\\orc_calibrate_250919_sen\\arg2_1.0" in str(reference.output_dir)
        assert reference.output_dir.exists()
        assert len(reference.histories) == 50
        assert reference.years_available == tuple(range(1961, 2011))
        assert reference.stomate_start is not None
        assert reference.stomate_restart is not None
        assert reference.run_def is not None
        assert reference.used_run_def is not None
        assert reference.used_run_def.name == "used_run.def"
        assert reference.modelout_csv is not None or reference.modelout_csv_zip is not None
        assert reference.has_minimum_annual_truth


def test_reference_layout_discovers_all_complete_raw_landpoints():
    ids = discover_paper_landpoint_ids(ROOT)

    assert "001.0-071.0" in ids
    assert "001.0-073.0" in ids
    assert len(ids) == 669


def test_paper_modelout_reader_can_read_downloaded_zip_csv_for_nonlegacy_landpoint():
    rows = read_paper_modelout_csv(root=ROOT, landpoint_id="001.0-073.0")

    assert len(rows) == 1
    assert rows[0].i_grid == "001.0-073.0"
    assert rows[0].age == 49
    assert rows[0].target_year == 2009


def test_reference_layout_resolves_single_landpoint_directly():
    reference = resolve_paper_landpoint_reference(ROOT, "003.0-077.0")

    assert reference.iteration_id == "I16"
    assert reference.param_set == "S5_69.110_0.0906_0.2045_97.623"
    assert reference.job_script is not None or reference.job_script_zip is not None
    assert reference.generated_run_def is not None or reference.generated_run_def_zip is not None


def test_stomate_reference_files_support_nonlegacy_landpoint():
    files = find_stomate_reference_files(ROOT, landpoint_id="005.0-069.0")

    assert files.run_dir.name == "S30_62.086_0.0834_0.1434_31.299"
    assert files.start.name == "stomate_start.nc"
    assert files.restart.name == "stomate_restart.nc"
    assert len(files.histories) == 50


def test_default_modelout_csv_prefers_complete_extracted_script_tree():
    path = default_paper_modelout_csv_path(ROOT, landpoint_id="003.0-077.0")

    assert path == ROOT / "reference" / "script" / "orc_cali_250919" / "modelout_sen" / "arg2_1.0" / "modelout_003.0-077.0.csv"


@pytest.mark.parametrize("landpoint_id", ("001.0-071.0", "001.0-073.0", "003.0-077.0", "005.0-069.0"))
def test_prepared_driver_context_binds_selected_landpoint_restart_package(tmp_path, landpoint_id):
    reference = resolve_paper_landpoint_reference(ROOT, landpoint_id)

    assert reference.output_dir is not None
    assert reference.run_def is not None
    assert reference.driver_start is not None
    assert reference.sechiba_start is not None
    assert reference.stomate_start is not None

    runtime_run_def = write_materialized_run_def(
        materialize_case_run_def_values(base_used_run_def=BASE_USED_RUN_DEF, case_run_def=reference.run_def),
        tmp_path / landpoint_id / "used_run.def",
    )

    context = prepare_paper_1961_driver_context(
        CONFIG,
        used_run_def_path=runtime_run_def,
        reference_run_dir=reference.output_dir,
    )

    assert context.reference_run_dir is not None
    assert context.reference_run_dir.resolve() == reference.output_dir.resolve()
    assert context.run_def_path.resolve() == runtime_run_def.resolve()
    assert context.first_step_bundle.domain.nbindex == 1
    assert context.first_step_restart_state.run_dir.resolve() == reference.output_dir.resolve()
    assert context.first_step_restart_state.driver_start.resolve() == reference.driver_start.resolve()
    assert context.first_step_restart_state.sechiba_start.resolve() == reference.sechiba_start.resolve()
    assert context.first_step_restart_state.stomate_input.resolve() == reference.stomate_start.resolve()
    assert context.first_step_bundle.restart_anchors is not None
    assert context.first_step_bundle.restart_anchors.path.resolve() == reference.sechiba_start.resolve()
    # pft_parameters.f90:3629-3630 reads uppercase TIDES. The paper script
    # adds lowercase tides=y, while the final Fortran used_run.def records
    # uppercase TIDES=FALSE.
    assert context.hydrol_tides is False


def test_prepared_driver_context_uses_selected_landpoint_used_run_def_defaults(tmp_path):
    reference = resolve_paper_landpoint_reference(ROOT, "069.0-119.0")

    assert reference.output_dir is not None
    assert reference.run_def is not None
    assert reference.used_run_def is not None
    runtime_run_def = write_materialized_run_def(
        materialize_case_run_def_values(
            base_used_run_def=reference.used_run_def,
            case_run_def=reference.run_def,
        ),
        tmp_path / reference.landpoint_id / "used_run.def",
    )

    context = prepare_paper_1961_driver_context(
        CONFIG,
        used_run_def_path=runtime_run_def,
        reference_run_dir=reference.output_dir,
    )

    assert context.reference_run_dir.resolve() == reference.output_dir.resolve()
    assert context.hydrol_reinf_slope == (0.0,)
