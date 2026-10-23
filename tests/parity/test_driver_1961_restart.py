from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.init import initialize_imposed_vegetation_state, read_run_scalars
from jax_orchidee.driver import orchestration
from jax_orchidee.driver.restart import (
    SECHIBA_STATIC_FIELDS,
    inventory_restart_fields,
    read_restart_fields,
    read_sechiba_static_restart,
    reference_restart_path,
)
from jax_orchidee.driver.orchestration import read_driver_runtime_scalars
from jax_orchidee.driver.restart_state import reference_case_first_step_restart_state


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"


def test_reference_restart_inventory_exposes_exact_static_truth_fields():
    sechiba_start = reference_restart_path(CONFIG, "sechiba_start.nc")
    driver_start = reference_restart_path(CONFIG, "driver_start.nc")

    sechiba_inventory = inventory_restart_fields(
        sechiba_start,
        (*SECHIBA_STATIC_FIELDS, "soiltile", "soilclass", "salinity", "tide_height"),
    )
    assert set(SECHIBA_STATIC_FIELDS).issubset(sechiba_inventory.present)
    assert "soiltile" in sechiba_inventory.missing
    assert "soilclass" in sechiba_inventory.missing
    assert "salinity" in sechiba_inventory.missing
    assert "tide_height" in sechiba_inventory.missing

    driver_inventory = inventory_restart_fields(driver_start, SECHIBA_STATIC_FIELDS)
    assert driver_inventory.present == ()
    assert set(driver_inventory.missing) == set(SECHIBA_STATIC_FIELDS)


def test_read_sechiba_static_restart_normalizes_shapes_and_anchor_values():
    restart = read_sechiba_static_restart(reference_restart_path(CONFIG, "sechiba_start.nc"))

    assert restart.veget.shape == (1, 14)
    assert restart.veget_max.shape == (1, 14)
    assert restart.frac_nobio.shape == (1, 1)
    assert restart.njsc.shape == (1,)
    assert restart.clay_frac.shape == (1,)
    assert restart.sand_frac.shape == (1,)
    assert restart.bulk_dens.shape == (1,)
    assert restart.soil_ph.shape == (1,)
    assert restart.poor_soils.shape == (1,)
    assert restart.wtp.shape == (1,)
    assert restart.wt_ab_tide.shape == (1,)

    expected_veget_max = np.zeros((1, 14), dtype=np.float64)
    expected_veget_max[0, 13] = 1.0
    assert np.allclose(restart.veget_max, expected_veget_max)
    assert np.allclose(restart.veget[0, 13], 0.973181842386765)
    assert np.allclose(restart.frac_nobio, [[0.0]])
    assert np.array_equal(restart.njsc, np.asarray([2], dtype=np.int32))
    assert np.allclose(restart.clay_frac, [0.2])
    assert np.allclose(restart.sand_frac, [0.4])
    assert np.allclose(restart.bulk_dens, [1650.0])
    assert np.allclose(restart.soil_ph, [7.0])
    assert np.allclose(restart.poor_soils, [0.0])
    assert np.allclose(restart.wtp, [0.0])
    assert np.allclose(restart.wt_ab_tide, [0.0])


def test_imposed_phase1c_fields_match_restart_truth_where_formula_is_complete():
    scalars = read_run_scalars(CONFIG)
    state = initialize_imposed_vegetation_state(scalars, npts=1)
    restart = read_sechiba_static_restart(reference_restart_path(CONFIG, "sechiba_start.nc"))

    assert np.allclose(state.veget_max, restart.veget_max)
    assert np.allclose(state.frac_nobio, restart.frac_nobio)
    assert np.allclose(state.veget[0, :13], 0.0)
    assert np.allclose(state.veget[0, 13], 1.0)
    assert not np.allclose(state.veget, restart.veget)


def test_missing_restart_field_selection_raises_instead_of_fabricating_truth():
    sechiba_start = reference_restart_path(CONFIG, "sechiba_start.nc")

    try:
        read_restart_fields(sechiba_start, ["veget_max", "soiltile"])
    except KeyError as exc:
        assert "soiltile" in str(exc)
    else:
        raise AssertionError("missing restart field did not raise")


def test_reference_case_first_step_restart_state_aggregates_exact_restart_readers():
    state = reference_case_first_step_restart_state(CONFIG, root=ROOT)

    assert state.driver_start.name == "driver_start.nc"
    assert state.sechiba_start.name == "sechiba_start.nc"
    assert state.stomate_input.name == "stomate_start.nc"
    assert state.driver_albedo.albedo.shape == (1, 2)
    assert state.enerbil_thermal.soilcap.shape == (1,)
    assert state.enerbil_thermal.soilcap_pft.shape == (1, 14)
    assert state.thermosoil.ptn.shape == (1, 32, 14)
    assert state.thermosoil.cgrnd.shape == (1, 31, 14)
    assert state.thermosoil.cgrnd_snow.shape == (1, 3)
    assert state.hydrol.moistc.shape == (1, 6, 11)
    assert state.hydrol.evap_bare_lim_ns.shape == (1, 6)
    assert state.hydrol.us.shape == (1, 14, 6, 11)
    assert state.hydrol.fwet_new.shape == (1,)
    assert state.slowproc.lai.shape == (1, 14)
    assert state.slowproc.frac_age.shape == (1, 14, 4)
    assert state.stomate.biomass.shape[:2] == (1, 14)
    assert state.stomate.carbon_32l.shape[:3] == (1, 3, 14)
    assert state.stomate_readstart.gas_state.O2_soil.shape[:2] == (1, 32)
    assert state.stomate_readstart.remainder_state.MatrixV.shape[:2] == (1, 14)
    assert state.stomate_readstart.report.implemented
    assert np.allclose(state.slowproc.veget_max[0, 13], 1.0)
    assert np.allclose(state.driver_albedo.albedo[0], [0.04526854, 0.22753724])
    assert np.allclose(state.hydrol.fwet_new, [0.0])
    assert any("Job0_bio lines 277-322" in item for item in state.provenance)


def _reference_restart_run() -> Path:
    driver_start = reference_restart_path(CONFIG, "driver_start.nc")
    if not driver_start.exists():
        pytest.skip("paper restart triplet is unavailable")
    return driver_start.parent


def _copy_restart_triplet(source: Path, target: Path) -> Path:
    target.mkdir()
    for name in ("driver_start.nc", "sechiba_start.nc", "stomate_start.nc"):
        shutil.copyfile(source / name, target / name)
    return target


def _assert_readstart_states_equal(left, right) -> None:
    for group_name in (
        "entry_state",
        "season_state",
        "daily_state",
        "gas_state",
        "remainder_state",
    ):
        left_group = getattr(left, group_name)
        right_group = getattr(right, group_name)
        for field_name in left_group._fields:
            np.testing.assert_array_equal(
                getattr(left_group, field_name),
                getattr(right_group, field_name),
                err_msg=f"{group_name}.{field_name} changed",
            )
    assert left.report == right.report


def test_first_step_tau_longterm_uses_configured_cap_at_restart_and_selector_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORCHIDEE_REPO_ROOT", str(ROOT))
    source_run = _reference_restart_run()
    normal_run = _copy_restart_triplet(source_run, tmp_path / "normal")
    sentinel_run = _copy_restart_triplet(source_run, tmp_path / "sentinel")
    missing_run = _copy_restart_triplet(source_run, tmp_path / "missing")

    used_run_def = ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"
    default_used_run_def = tmp_path / "default" / "used_run.def"
    custom_used_run_def = tmp_path / "custom" / "used_run.def"
    default_used_run_def.parent.mkdir()
    custom_used_run_def.parent.mkdir()
    run_def_text = used_run_def.read_text(encoding="utf-8")
    default_used_run_def.write_text(run_def_text, encoding="utf-8")
    custom_text, replacement_count = re.subn(
        r"(?m)^(\s*COEFF_TAU_LONGTERM\s*=\s*)[^\r\n]+",
        r"\g<1>2.5",
        run_def_text,
        count=1,
    )
    assert replacement_count == 1
    custom_used_run_def.write_text(custom_text, encoding="utf-8")

    with Dataset(normal_run / "stomate_start.nc") as restart:
        assert "t2m_longterm" in restart.variables
        assert "tau_longterm" in restart.variables
        valid_tau = float(restart.variables["tau_longterm"][:].reshape(-1)[0])
        assert valid_tau == 1095.0
    with Dataset(sentinel_run / "stomate_start.nc", "r+") as restart:
        restart.variables["tau_longterm"][:] = 999999.0
    with Dataset(missing_run / "stomate_start.nc", "r+") as restart:
        restart.renameVariable("tau_longterm", "tau_longterm_missing")

    normal_default = reference_case_first_step_restart_state(
        CONFIG,
        root=ROOT,
        run_dir=normal_run,
        run_def_path=default_used_run_def,
    )
    normal_custom = reference_case_first_step_restart_state(
        CONFIG,
        root=ROOT,
        run_dir=normal_run,
        run_def_path=custom_used_run_def,
    )
    assert normal_default.stomate_readstart.season_state.tau_longterm == valid_tau
    assert normal_custom.stomate_readstart.season_state.tau_longterm == valid_tau
    _assert_readstart_states_equal(normal_default.stomate_readstart, normal_custom.stomate_readstart)

    sentinel_default = reference_case_first_step_restart_state(
        CONFIG,
        root=ROOT,
        run_dir=sentinel_run,
        run_def_path=default_used_run_def,
    )
    sentinel_custom = reference_case_first_step_restart_state(
        CONFIG,
        root=ROOT,
        run_dir=sentinel_run,
        run_def_path=custom_used_run_def,
    )
    assert sentinel_default.stomate_readstart.season_state.tau_longterm == 1095.0
    assert sentinel_custom.stomate_readstart.season_state.tau_longterm == 912.5
    assert "tau_longterm" in sentinel_custom.stomate_readstart.report.defaulted_fields
    np.testing.assert_array_equal(
        sentinel_custom.stomate_readstart.season_state.t2m_longterm,
        normal_default.stomate_readstart.season_state.t2m_longterm,
    )

    with pytest.raises(KeyError, match="tau_longterm"):
        reference_case_first_step_restart_state(
            CONFIG,
            root=ROOT,
            run_dir=missing_run,
            run_def_path=custom_used_run_def,
        )

    selected = {}

    class SelectionCaptured(Exception):
        pass

    def capture_source_kwargs(**kwargs):
        selected.update(kwargs)
        raise SelectionCaptured

    monkeypatch.setattr(orchestration, "paper_case_stomate_bundle_source_kwargs", capture_source_kwargs)
    runtime = read_driver_runtime_scalars(custom_used_run_def)
    with pytest.raises(SelectionCaptured):
        orchestration._paper_first_step_stomate_restart_input_bundles(
            config_path=CONFIG,
            run_def_path=custom_used_run_def,
            restart_state=sentinel_custom,
            runtime=runtime,
            tstep=int(round(runtime.dt_stomate / runtime.dt_sechiba)) - 1,
        )

    assert selected["season"] is sentinel_custom.stomate_readstart.season_state
    assert selected["season"].tau_longterm == 912.5
    np.testing.assert_array_equal(
        selected["season"].t2m_longterm,
        sentinel_custom.stomate_readstart.season_state.t2m_longterm,
    )
