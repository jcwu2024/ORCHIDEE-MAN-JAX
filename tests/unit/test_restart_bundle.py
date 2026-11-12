from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import yaml
from netCDF4 import Dataset


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
from jax_orchidee.driver.restart_export import (  # noqa: E402
    stomate_restart_states_from_day_end_packet,
    write_stomate_restart_from_day_end_packet,
)
from jax_orchidee.driver.restart_state import reference_case_first_step_restart_state  # noqa: E402
from jax_orchidee.sechiba.restart_io import (  # noqa: E402
    read_sechiba_restart_state,
    sechiba_finalize_source_state_from_restart,
)


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
CATALOG = ROOT / "configs" / "pft_catalogs" / "orchidee_man_paper_250919.json"


def _compact_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    catalog_payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    catalog_payload["layouts"]["paper_compact"] = ["bare_soil", "mangrove_pft14"]
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")

    config_payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config_payload["pft_catalog"] = {"path": str(catalog_path), "layout_id": "paper_compact"}
    config_payload["structural_overrides"]["NVM"] = 2
    config_path = tmp_path / "compact.yaml"
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    monkeypatch.setenv("ORCHIDEE_REPO_ROOT", str(ROOT))
    return config_path


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


def test_compact_layout_restart_bundle_writes_reads_and_preserves_stable_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _reference_run()
    config = _compact_config(tmp_path, monkeypatch)
    original = reference_case_first_step_restart_state(
        config,
        root=ROOT,
        run_dir=run,
        stomate_filename="stomate_start.nc",
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
        tmp_path / "compact_restart",
        state=bundle_state,
        physical_state=physical,
        pft_layout=original.pft_layout,
    )
    restored = reference_case_first_step_restart_state(
        config,
        root=ROOT,
        run_dir=report.output_directory,
        stomate_filename="stomate_start.nc",
    )

    assert original.pft_layout.pft_ids == ("bare_soil", "mangrove_pft14")
    assert restored.pft_layout == original.pft_layout
    assert restored.sechiba_source_pft_layout.pft_ids == original.pft_layout.pft_ids
    assert restored.stomate_source_pft_layout.pft_ids == original.pft_layout.pft_ids
    for path in (report.sechiba.output_path, report.stomate.output_path):
        with Dataset(path) as dataset:
            assert len(dataset.dimensions["z_a"]) == 2
            assert len(dataset.dimensions["l_d"]) == 2
            assert json.loads(dataset.orchidee_jax_pft_ids_json) == [
                "bare_soil",
                "mangrove_pft14",
            ]
    for name, expected in original.sechiba_restart_state.fields.items():
        np.testing.assert_array_equal(restored.sechiba_restart_state.fields[name], expected)
    np.testing.assert_array_equal(restored.stomate.biomass, original.stomate.biomass)
    np.testing.assert_array_equal(restored.stomate.carbon_32l, original.stomate.carbon_32l)
    np.testing.assert_array_equal(
        restored.stomate_readstart.remainder_state.MatrixV,
        original.stomate_readstart.remainder_state.MatrixV,
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


@pytest.mark.parametrize("route", ("bundle", "standalone"))
@pytest.mark.parametrize("include_gas", (False, True))
def test_day_end_bundle_serializes_packet_owned_remainder_fields(
    tmp_path: Path, route: str, include_gas: bool,
) -> None:
    """Remainder fields are persisted memory, even outside entry/season/daily."""
    run = _reference_run()
    original = reference_case_first_step_restart_state(CONFIG, root=ROOT, run_dir=run)
    base = original.stomate_readstart
    names = ("resp_hetero", "tsurf_year", "deepC_peat", "depth_deepsoil", "t2m_14")
    changed = {name: np.asarray(getattr(base.remainder_state, name)) + 2.25 for name in names}
    changed["prod10"] = np.asarray(base.entry_state.prod10) + 0.125
    changed["prod100"] = np.asarray(base.entry_state.prod100) + 0.25
    for name in ("deepC_a", "deepC_s", "deepC_p"):
        changed[name] = np.asarray(getattr(base.entry_state, name)) + 0.375
    if include_gas:
        changed.update({
            name: np.asarray(getattr(base.gas_state, name)) + 1.75
            for name in base.gas_state._fields if name != "provenance"
        })
    changed["daily_accumulators"] = base.daily_state._asdict()
    driver = read_dim2_driver_restart(run / "driver_start.nc")
    packet = SimpleNamespace(fields_by_component={
        "driver_previous_step_state": {name: getattr(driver, name) for name in driver.__dataclass_fields__},
        "sechiba_finalize_state": sechiba_finalize_source_state_from_restart(original.sechiba_restart_state),
        "slowproc_stomate_previous_step_state": changed,
    })
    state = paper_restart_bundle_state_from_day_end_packet(packet, base_stomate=base, kjit=17520)
    physical = PaperRestartBundlePhysicalState(**{
        name: read_restart_physical_state(run / f"{name}_start.nc")
        for name in ("driver", "sechiba", "stomate")
    })
    if route == "bundle":
        report = write_paper_restart_start_bundle(tmp_path / "handoff", state=state, physical_state=physical, pft_layout=original.pft_layout)
        restored = reference_case_first_step_restart_state(CONFIG, root=ROOT, run_dir=report.output_directory)
    else:
        output = tmp_path / "standalone.nc"
        write_stomate_restart_from_day_end_packet(
            output, changed, base_states=base, physical_state=physical.stomate,
            pft_layout=original.pft_layout,
        )
        restored = reference_case_first_step_restart_state(
            CONFIG, root=ROOT, run_dir=run, stomate_filename=output,
        )
    for name in names:
        np.testing.assert_array_equal(getattr(restored.stomate_readstart.remainder_state, name), changed[name])
    for name, pool in (("prod10_total", "prod10"), ("prod100_total", "prod100")):
        np.testing.assert_array_equal(getattr(restored.stomate_readstart.entry_state, name), np.sum(changed[pool], axis=(1, 2)))
    np.testing.assert_array_equal(
        restored.stomate_readstart.entry_state.soilc_total,
        changed["deepC_a"] + changed["deepC_s"] + changed["deepC_p"],
    )
    for name in base.gas_state._fields:
        if name != "provenance":
            np.testing.assert_array_equal(
                getattr(restored.stomate_readstart.gas_state, name),
                changed.get(name, getattr(base.gas_state, name)),
            )
    # Missing packet coordinates keep the complete declared reader defaults.
    for name in base.remainder_state._fields:
        if name != "provenance" and name not in changed:
            np.testing.assert_array_equal(
                getattr(state.stomate.remainder_state, name),
                getattr(base.remainder_state, name),
            )


@pytest.mark.parametrize("name", ("prod10_total", "prod100_total", "soilc_total"))
def test_day_end_export_preserves_explicit_derived_values_for_writer_validation(
    tmp_path: Path, name: str,
) -> None:
    run = _reference_run()
    original = reference_case_first_step_restart_state(CONFIG, root=ROOT, run_dir=run)
    base = original.stomate_readstart
    expected = np.asarray(getattr(base.entry_state, name))
    for valid in (True, False):
        explicit = expected if valid else expected + 1.0
        packet = {name: explicit, "daily_accumulators": base.daily_state._asdict()}
        entry, _, _, merge = stomate_restart_states_from_day_end_packet(
            packet, base_entry=base.entry_state, base_season=base.season_state,
            base_daily=base.daily_state,
        )
        np.testing.assert_array_equal(getattr(entry, name), explicit)
        assert name not in merge.derived_fields
        assert name not in merge.retained_base_fields
        assert set(merge.derived_fields) == {"prod10_total", "prod100_total", "soilc_total"} - {name}
        output = tmp_path / f"{name}-{valid}.nc"
        kwargs = dict(
            base_states=base, physical_state=read_restart_physical_state(run / "stomate_start.nc"),
            pft_layout=original.pft_layout,
        )
        if valid:
            write_stomate_restart_from_day_end_packet(output, packet, **kwargs)
        else:
            with pytest.raises(ValueError, match=f"derived restart field {name}.*inconsistent"):
                write_stomate_restart_from_day_end_packet(output, packet, **kwargs)
            assert not output.exists()
