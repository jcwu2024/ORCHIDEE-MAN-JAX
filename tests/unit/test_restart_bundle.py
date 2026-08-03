from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.restart_bundle import (  # noqa: E402
    PaperRestartBundlePhysicalState,
    PaperRestartBundleState,
    read_restart_physical_state,
    paper_restart_bundle_state_from_day_end_packet,
    write_paper_restart_start_bundle,
)
from jax_orchidee.driver.restart_file_io import read_dim2_driver_restart  # noqa: E402
from jax_orchidee.driver.restart_state import reference_case_first_step_restart_state  # noqa: E402
from jax_orchidee.sechiba.restart_io import (  # noqa: E402
    read_sechiba_restart_state,
    sechiba_finalize_source_state_from_restart,
)


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"


def _reference_run() -> Path:
    root = (
        ROOT
        / "reference"
        / "OUT"
        / "orc_calibrate_250919_sen"
        / "arg2_1.0"
        / "001.0-071.0"
    )
    candidates = sorted(path.parent for path in root.rglob("driver_start.nc"))
    if not candidates:
        pytest.skip("paper restart triplet is unavailable")
    return candidates[0]


def test_restart_name_normalization_closes_exact_93_field_denominator() -> None:
    run = _reference_run()
    state = read_sechiba_restart_state(run / "sechiba_start.nc")
    source = sechiba_finalize_source_state_from_restart(state)

    assert len(source) == 93
    assert "q_cdrag_pft" in source and "cdrag_pft" not in source
    assert "tsol_rad" in source and "tsolrad" not in source
    assert "shum_ngrnd_permalong" in source and "shum_ngrnd_prmlng" not in source


def test_three_file_bundle_is_directly_readable_as_next_year_start(
    tmp_path: Path,
) -> None:
    run = _reference_run()
    original = reference_case_first_step_restart_state(
        CONFIG, root=ROOT, run_dir=run, stomate_filename="stomate_start.nc"
    )
    bundle_state = PaperRestartBundleState(
        driver=read_dim2_driver_restart(run / "driver_start.nc"),
        sechiba=original.sechiba_restart_state,
        stomate=original.stomate_readstart,
    )
    physical = PaperRestartBundlePhysicalState(
        driver=read_restart_physical_state(run / "driver_start.nc"),
        sechiba=read_restart_physical_state(run / "sechiba_start.nc"),
        stomate=read_restart_physical_state(run / "stomate_start.nc"),
    )

    report = write_paper_restart_start_bundle(
        tmp_path / "next_year",
        state=bundle_state,
        physical_state=physical,
        pft_layout=original.pft_layout,
    )
    restored = reference_case_first_step_restart_state(
        CONFIG,
        root=ROOT,
        run_dir=report.output_directory,
        stomate_filename="stomate_start.nc",
    )

    assert all(path.exists() for path in report.output_paths)
    assert len(report.driver.written_fields) == 13
    assert len(report.sechiba.written_fields) == 93
    assert len(report.stomate.written_fields) == 161
    assert restored.sechiba_source_pft_layout.pft_ids == original.pft_layout.pft_ids
    assert restored.stomate_source_pft_layout.pft_ids == original.pft_layout.pft_ids
    for name, expected in original.sechiba_restart_state.fields.items():
        np.testing.assert_array_equal(restored.sechiba_restart_state.fields[name], expected)
    np.testing.assert_array_equal(restored.stomate.age, original.stomate.age)
    np.testing.assert_array_equal(
        restored.stomate_readstart.remainder_state.MatrixV,
        original.stomate_readstart.remainder_state.MatrixV,
    )

    with pytest.raises(FileExistsError):
        write_paper_restart_start_bundle(
            report.output_directory,
            state=bundle_state,
            physical_state=physical,
            pft_layout=original.pft_layout,
        )


def test_complete_day_end_packet_constructs_all_three_scientific_states() -> None:
    run = _reference_run()
    original = reference_case_first_step_restart_state(
        CONFIG, root=ROOT, run_dir=run, stomate_filename="stomate_start.nc"
    )
    driver = read_dim2_driver_restart(run / "driver_start.nc")
    slowproc = {
        **{k: v for k, v in original.stomate_readstart.entry_state._asdict().items() if k != "provenance"},
        **{k: v for k, v in original.stomate_readstart.season_state._asdict().items() if k != "provenance"},
        "daily_accumulators": {
            k: v
            for k, v in original.stomate_readstart.daily_state._asdict().items()
            if k != "provenance"
        },
    }
    packet = SimpleNamespace(
        fields_by_component={
            "driver_previous_step_state": {
                name: getattr(driver, name) for name in driver.__dataclass_fields__
            },
            "sechiba_finalize_state": sechiba_finalize_source_state_from_restart(
                original.sechiba_restart_state
            ),
            "slowproc_stomate_previous_step_state": slowproc,
        }
    )

    state = paper_restart_bundle_state_from_day_end_packet(
        packet, base_stomate=original.stomate_readstart, kjit=17520
    )

    assert all(getattr(state.driver, name) is not None for name in driver.__dataclass_fields__)
    assert len(state.sechiba.fields) == 93
    np.testing.assert_array_equal(state.stomate.entry_state.age, original.stomate.age)
