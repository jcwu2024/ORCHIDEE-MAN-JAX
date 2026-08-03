from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from jax_orchidee.driver.orchestration import (
    paper_1961_driver_cold_start_multiday_modelout_lite_run,
    paper_1961_driver_multiday_modelout_lite_run,
)
from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference
from jax_orchidee.driver.restart_bundle import (
    PaperRestartBundlePhysicalState,
    paper_restart_bundle_state_from_day_end_packet,
    read_restart_physical_state,
    write_paper_restart_start_bundle,
)
from jax_orchidee.driver.restart_state import reference_case_first_step_restart_state
from jax_orchidee.driver.run_def_materialization import (
    read_run_def_values,
    write_materialized_run_def,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
CATALOG = ROOT / "configs" / "pft_catalogs" / "orchidee_man_paper_250919.json"
LANDPOINT_ID = "001.0-071.0"


def _compact_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    reference = resolve_paper_landpoint_reference(ROOT, LANDPOINT_ID)
    if reference.output_dir is None or reference.used_run_def is None:
        pytest.skip("paper reference package is unavailable")

    catalog_payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    catalog_payload["layouts"]["paper_compact"] = ["bare_soil", "mangrove_pft14"]
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")

    config_payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config_payload["pft_catalog"] = {"path": str(catalog_path), "layout_id": "paper_compact"}
    config_payload["structural_overrides"]["NVM"] = 2
    config_path = tmp_path / "compact.yaml"
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    run_def_values = read_run_def_values(reference.used_run_def)
    run_def_values["NVM"] = "2"
    run_def_path = write_materialized_run_def(run_def_values, tmp_path / "compact_used_run.def")
    monkeypatch.setenv("ORCHIDEE_REPO_ROOT", str(ROOT))
    monkeypatch.setenv("ORCHIDEE_DATA_ROOT", str(ROOT / "data"))
    monkeypatch.setenv("ORCHIDEE_REFERENCE_ROOT", str(ROOT / "reference"))
    monkeypatch.setenv("ORCHIDEE_OUTPUT_ROOT", str(ROOT / "outputs"))
    return config_path, run_def_path, reference.output_dir


def test_compact_cold_day_writes_restart_and_resumes_by_stable_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, run_def, reference_run = _compact_case(tmp_path, monkeypatch)
    run = paper_1961_driver_cold_start_multiday_modelout_lite_run(
        config,
        year=1961,
        ndays=1,
        used_run_def_path=run_def,
        reference_run_dir=reference_run,
        single_pass_daily_fold=True,
    )

    assert run.ready_for_requested_days
    assert len(run.daily_modelout) == 1
    assert run.last_day_end_state is not None
    slowproc = run.last_day_end_state.fields_by_component[
        "slowproc_stomate_previous_step_state"
    ]
    assert np.asarray(slowproc["biomass"]).shape[1] == 2
    assert np.asarray(slowproc["veget_max"]).shape[1] == 2

    base = reference_case_first_step_restart_state(
        config,
        root=ROOT,
        run_dir=reference_run,
        run_def_path=run_def,
        stomate_filename="stomate_start.nc",
    )
    state = paper_restart_bundle_state_from_day_end_packet(
        run.last_day_end_state,
        base_stomate=base.stomate_readstart,
        kjit=48,
    )
    physical = PaperRestartBundlePhysicalState(
        driver=read_restart_physical_state(reference_run / "driver_start.nc"),
        sechiba=read_restart_physical_state(reference_run / "sechiba_start.nc"),
        stomate=read_restart_physical_state(reference_run / "stomate_start.nc"),
    )
    report = write_paper_restart_start_bundle(
        tmp_path / "cold_day_restart",
        state=state,
        physical_state=physical,
        pft_layout=base.pft_layout,
    )
    restored = reference_case_first_step_restart_state(
        config,
        root=ROOT,
        run_dir=report.output_directory,
        run_def_path=run_def,
        stomate_filename="stomate_start.nc",
    )

    assert restored.pft_layout.pft_ids == ("bare_soil", "mangrove_pft14")
    for name, expected in state.sechiba.fields.items():
        np.testing.assert_array_equal(restored.sechiba_restart_state.fields[name], expected)
    np.testing.assert_array_equal(restored.stomate.biomass, state.stomate.entry_state.biomass)
    np.testing.assert_array_equal(restored.stomate.carbon_32l, state.stomate.entry_state.carbon_32l)

    resumed = paper_1961_driver_multiday_modelout_lite_run(
        config,
        year=1961,
        ndays=1,
        used_run_def_path=run_def,
        reference_run_dir=report.output_directory,
        root=ROOT,
        single_pass_daily_fold=True,
    )
    assert resumed.ready_for_requested_days
    assert resumed.last_day_end_state is not None
    resumed_slowproc = resumed.last_day_end_state.fields_by_component[
        "slowproc_stomate_previous_step_state"
    ]
    assert np.asarray(resumed_slowproc["biomass"]).shape[1] == 2
