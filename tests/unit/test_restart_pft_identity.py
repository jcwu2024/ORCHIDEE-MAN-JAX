from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from jax_orchidee.driver.restart import SECHIBA_RESTART_PFT_AXES
from jax_orchidee.parameters.pft_catalog import (
    build_pft_run_layout,
    build_selected_pft_run_layout,
    load_pft_catalog,
)
from jax_orchidee.sechiba.restart_io import (
    SECHIBA_NORMALIZED_RESTART_PFT_AXES,
    read_sechiba_restart_state,
)
from jax_orchidee.stomate.reference import (
    STOMATE_DAILY_PFT_AXES,
    STOMATE_ENTRY_PFT_AXES,
    STOMATE_GAS_PFT_AXES,
    STOMATE_SEASON_PFT_AXES,
    read_stomate_daily_accumulator_state,
    read_stomate_ok_pc_restart_gas_state,
    read_stomate_restart_entry_state,
    read_stomate_restart_season_state,
)
from jax_orchidee.stomate.restart_io import (
    STOMATE_REMAINDER_PFT_AXES,
    read_stomate_readstart_states_from_template,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "configs" / "pft_catalogs" / "orchidee_man_paper_250919.json"


def _reference_file(name: str) -> Path:
    root = (
        ROOT
        / "reference"
        / "OUT"
        / "orc_calibrate_250919_sen"
        / "arg2_1.0"
        / "001.0-071.0"
    )
    paths = sorted(root.rglob(name))
    if not paths:
        pytest.skip(f"paper {name} is unavailable")
    return paths[0]


def _layouts():
    catalog = load_pft_catalog(CATALOG)
    source = build_pft_run_layout(
        catalog,
        layout_id="paper_250919_legacy14",
        fractions=[0.0] * 13 + [1.0],
    )
    target = build_selected_pft_run_layout(
        catalog,
        layout_id="compact_bare_mangrove",
        pft_ids=("bare_soil", "mangrove_pft14"),
        fractions=(0.0, 1.0),
    )
    return source, target


def _assert_registry_remapped(full, compact, registry: dict[str, int]) -> None:
    for field, axis in registry.items():
        expected_value = getattr(full, field)
        actual_value = getattr(compact, field)
        if expected_value is None:
            assert actual_value is None
            continue
        expected = np.take(np.asarray(expected_value), [0, 13], axis=axis)
        np.testing.assert_array_equal(actual_value, expected, err_msg=field)


def test_stomate_all_declared_pft_axes_remap_legacy14_to_compact_layout() -> None:
    path = _reference_file("stomate_start.nc")
    source, target = _layouts()
    full_entry = read_stomate_restart_entry_state(path)
    full_season = read_stomate_restart_season_state(path)
    full_daily = read_stomate_daily_accumulator_state(path)
    full_gas = read_stomate_ok_pc_restart_gas_state(path)
    remap_kwargs = {
        "source_pft_layout": source,
        "target_pft_layout": target,
    }

    compact_entry = read_stomate_restart_entry_state(path, **remap_kwargs)
    compact_season = read_stomate_restart_season_state(path, **remap_kwargs)
    compact_daily = read_stomate_daily_accumulator_state(path, **remap_kwargs)
    compact_gas = read_stomate_ok_pc_restart_gas_state(path, **remap_kwargs)

    _assert_registry_remapped(full_entry, compact_entry, STOMATE_ENTRY_PFT_AXES)
    _assert_registry_remapped(full_season, compact_season, STOMATE_SEASON_PFT_AXES)
    _assert_registry_remapped(full_daily, compact_daily, STOMATE_DAILY_PFT_AXES)
    _assert_registry_remapped(full_gas, compact_gas, STOMATE_GAS_PFT_AXES)

    with Dataset(path) as dataset:
        full_readstart = read_stomate_readstart_states_from_template(
            path,
            t2m=np.asarray(full_daily.t2m_daily),
            nvm=source.n_pft,
            nslm=full_season.tsoil_month.shape[1],
            ndeep=full_entry.carbon_32l.shape[3],
            nsnow=int(dataset.variables["O2_snow"].shape[2]),
            nvert=int(dataset.variables["uo_0"].shape[1]),
            months_num=int(dataset.variables["fwet_series"].shape[1]),
            ncarb=full_entry.carbon.shape[1],
            nlitt=full_entry.litter.shape[1],
            nbpools=int(dataset.variables["MatrixV"].shape[1]),
        )
        compact_readstart = read_stomate_readstart_states_from_template(
            path,
            t2m=np.asarray(full_daily.t2m_daily),
            nvm=source.n_pft,
            nslm=full_season.tsoil_month.shape[1],
            ndeep=full_entry.carbon_32l.shape[3],
            nsnow=int(dataset.variables["O2_snow"].shape[2]),
            nvert=int(dataset.variables["uo_0"].shape[1]),
            months_num=int(dataset.variables["fwet_series"].shape[1]),
            ncarb=full_entry.carbon.shape[1],
            nlitt=full_entry.litter.shape[1],
            nbpools=int(dataset.variables["MatrixV"].shape[1]),
            **remap_kwargs,
        )
    _assert_registry_remapped(
        full_readstart.remainder_state,
        compact_readstart.remainder_state,
        STOMATE_REMAINDER_PFT_AXES,
    )


def test_sechiba_all_declared_pft_axes_remap_legacy14_to_compact_layout() -> None:
    path = _reference_file("sechiba_start.nc")
    source, target = _layouts()
    full = read_sechiba_restart_state(path)
    compact = read_sechiba_restart_state(
        path,
        source_pft_layout=source,
        target_pft_layout=target,
    )

    assert set(SECHIBA_RESTART_PFT_AXES) <= set(full.fields)
    for field, axis in SECHIBA_NORMALIZED_RESTART_PFT_AXES.items():
        expected = np.take(np.asarray(full.fields[field]), [0, 13], axis=axis)
        np.testing.assert_array_equal(compact.fields[field], expected, err_msg=field)
