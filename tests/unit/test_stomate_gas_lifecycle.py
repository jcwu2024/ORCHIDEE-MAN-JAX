"""Persisted inactive gas: true NetCDF and caller/carry seams, no model run."""
from dataclasses import replace
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace

from netCDF4 import Dataset
import numpy as np
import pytest
import yaml

from jax_orchidee.driver import orchestration as orch
from jax_orchidee.driver.fast_state import fast_state_from_previous_packet
from jax_orchidee.driver.restart_bundle import (
    PaperRestartBundlePhysicalState, paper_restart_bundle_state_from_day_end_packet,
    read_restart_physical_state, write_paper_restart_start_bundle,
)
from jax_orchidee.driver.restart_file_io import read_dim2_driver_restart
from jax_orchidee.driver.restart_state import reference_case_first_step_restart_state
from jax_orchidee.sechiba.restart_io import sechiba_finalize_source_state_from_restart
from jax_orchidee.stomate.restart_io import (
    STOMATE_GAS_FIELDS, read_stomate_readstart_states_from_template,
    stomate_cold_start_gas_state,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/orchidee_man_250919.yaml"


def _small_read(path: Path, **kwargs):
    return read_stomate_readstart_states_from_template(
        path, t2m=np.array([281.5]), nvm=2, nslm=2, ndeep=3,
        nsnow=2, nvert=6, ns=4, months_num=3, nbpools=3, **kwargs,
    )


@pytest.mark.parametrize("case", ("missing", "all_sentinel", "partial", "override", "stored_override", "partial_override"))
def test_gas_reader_obeys_source_defaults_and_stored_value_priority(tmp_path, case):
    path = tmp_path / "gas.nc"
    stored = {}
    with Dataset(path, "w") as nc:
        for name, count in (("time", 1), ("pft", 2), ("deep", 3), ("snow", 2), ("y", 1), ("x", 1)):
            nc.createDimension(name, count)
        if case in ("all_sentinel", "partial", "stored_override", "partial_override"):
            for name in STOMATE_GAS_FIELDS:
                levels = 2 if name.endswith("snow") else 3
                value = np.full((1, levels, 2), 999999.0)
                if case == "stored_override":
                    value[:] = 23.75
                if case in ("partial", "partial_override"):
                    value[0, 0, 0] = 17.25
                stored[name] = value
                var = nc.createVariable(name, "f8", ("time", "pft", "snow" if name.endswith("snow") else "deep", "y", "x"))
                var[:] = value.transpose(0, 2, 1).reshape(1, 2, levels, 1, 1)
    kwargs = {"o2_init_conc": 31.5, "ch4_init_conc": 0.007} if case.endswith("override") else {}
    state = _small_read(path, **kwargs)
    for name in STOMATE_GAS_FIELDS:
        value = getattr(state.gas_state, name)
        constant = (31.5 if name.startswith("O2") else 0.007) if case == "override" else (298.3 if name.startswith("O2") else 0.001267)
        expected = stored[name] if case in ("partial", "stored_override", "partial_override") else np.full(value.shape, constant)
        np.testing.assert_array_equal(value, expected)
        assert value.dtype == np.float64
        if case != "stored_override":
            assert name in (state.report.mixed_sentinel_fields if case in ("partial", "partial_override") else state.report.defaulted_fields)


def test_cold_gas_initializer_uses_source_axes_and_explicit_overrides():
    state = stomate_cold_start_gas_state(npts=2, nvm=4, ndeep=3, nsnow=2)
    explicit = stomate_cold_start_gas_state(npts=2, nvm=4, ndeep=3, nsnow=2, o2_init_conc=27., ch4_init_conc=.09)
    for name in STOMATE_GAS_FIELDS:
        shape = (2, 2 if name.endswith("snow") else 3, 4)
        np.testing.assert_array_equal(getattr(state, name), np.full(shape, 298.3 if name.startswith("O2") else .001267))
        np.testing.assert_array_equal(getattr(explicit, name), np.full(shape, 27. if name.startswith("O2") else .09))


def _reference_run():
    case = ROOT / "reference/OUT/orc_calibrate_250919_sen/arg2_1.0/001.0-071.0"
    paths = sorted(case.rglob("driver_start.nc"))
    if not paths:
        pytest.skip("real paper restart is unavailable")
    return paths[0].parent


def _assert_gas(packet, expected):
    fields = packet.fields_by_component["slowproc_stomate_previous_step_state"]
    for name in STOMATE_GAS_FIELDS:
        assert name in fields
        np.testing.assert_array_equal(fields[name], getattr(expected, name))


@pytest.mark.parametrize("origin", ("cold", "restart"))
def test_nondefault_gas_survives_actual_extract_carry_day_end_and_real_codec(tmp_path, monkeypatch, origin):
    run = _reference_run()
    saved = tmp_path / "stored-gas.nc"
    copyfile(run / "stomate_start.nc", saved)
    with Dataset(saved, "r+") as nc:
        for index, name in enumerate(STOMATE_GAS_FIELDS):
            values = np.arange(nc[name].size, dtype=np.float64).reshape(nc[name].shape) / 16 + 17 * (index + 1)
            nc[name][:] = values
    original = reference_case_first_step_restart_state(CONFIG, root=ROOT, run_dir=run, stomate_filename=saved)
    base = original.stomate_readstart
    components = dict(enerbil_first_step_local=None, diffuco_first_step_local_enerbil_precall=None,
                      diffuco_control_salinity=None, diffuco_control_assembly=None,
                      thermosoil_first_step_module=None, condveg_first_step_module=None,
                      hydrol_first_step_module=None)
    finalize = sechiba_finalize_source_state_from_restart(original.sechiba_restart_state)
    # Orthogonal SECHIBA recomputation is outside this gas-seam regression.
    # Use its real saved 93-field mapping for both extraction routes.
    monkeypatch.setattr(orch, "_sechiba_finalize_state_from_components", lambda **kw: dict(finalize))
    if origin == "restart":
        scaffold = SimpleNamespace(
            step=SimpleNamespace(tstep=0), first_step_restart_state=original,
            diffuco_first_step_precall=None, stomate_first_step_daily_accumulation=None, **components,
        )
        first = orch.driver_previous_step_state_from_timestep_scaffold(scaffold)
    else:
        # Only orthogonal SECHIBA cold owners are stand-ins. Gas is read from
        # real NC, the full production cold extractor and every carry run.
        class OtherColdOwner:
            def __getattr__(self, name):
                return finalize.get(name, np.zeros((1, 14)))
        unrelated = OtherColdOwner()
        slow = SimpleNamespace(**{name: getattr(unrelated, name) for name in (
            "veget", "veget_max", "lai", "frac_nobio", "frac_age", "njsc", "reinf_slope",
            "clayfraction", "sandfraction", "height", "soil_ph", "poor_soils", "bulk_density",
        )}, peat=unrelated)
        payloads = {**{name: None for name in components},
                    "diffuco_first_step_precall": None, "slowproc_init_pft14": slow,
                    "hydrol_cold_start_state": unrelated, "enerbil_surface_state": unrelated,
                    "condveg_soilalbedo_bg": np.zeros((1, 3)), "thermosoil_refSOC": np.zeros((1, 32, 14)),
                    "stomate_cold_start_gas_state": base.gas_state}
        first = orch._previous_step_state_from_cold_start_payloads(payloads)
    _assert_gas(first, base.gas_state)
    # First-day cold AND restart runners use the scaffold extractor before
    # the compact runtime path. Exercise its separate actual carry list.
    first = orch.driver_previous_step_state_from_next_step_hydrol_scaffold(
        SimpleNamespace(
            ready_for_next_state=True, tstep=1,
            enerbil=SimpleNamespace(previous_state=first, enerbil_local=None,
                                   diffuco_local_enerbil_precall=None),
            thermosoil_module=None, condveg_module=None, hydrol_module=None,
        ),
    )
    _assert_gas(first, base.gas_state)
    step_kwargs = dict(enerbil_local=None, diffuco_local=None, thermosoil_module=None,
                       condveg_module=None, hydrol_module=None)
    ordinary = orch._runtime_next_state_from_components(tstep=1, previous_state=first, **step_kwargs)
    _assert_gas(ordinary, base.gas_state)
    fast = orch._runtime_next_state_from_components(
        tstep=2, previous_state=fast_state_from_previous_packet(ordinary), **step_kwargs,
    )
    _assert_gas(fast, base.gas_state)
    day_end = orch._paper_day_end_state_packet(
        previous_state=fast, stomate_state=orch.DriverFirstDayStomateStatePacket(
            day_index=1, fields={"daily_accumulators": base.daily_state._asdict()}, provenance_by_field={},
        ),
    )
    _assert_gas(day_end, base.gas_state)
    # Fill only orthogonal file owners. Deliberately poison the writer base
    # gas so a lost packet field cannot be hidden by a fallback/default.
    driver = read_dim2_driver_restart(run / "driver_start.nc")
    day_end.fields_by_component["driver_previous_step_state"] = vars(driver)
    day_end.fields_by_component["sechiba_finalize_state"] = finalize
    poison = base.gas_state._replace(**{name: np.zeros_like(getattr(base.gas_state, name)) for name in STOMATE_GAS_FIELDS})
    state = paper_restart_bundle_state_from_day_end_packet(day_end, base_stomate=replace(base, gas_state=poison), kjit=48)
    physical = PaperRestartBundlePhysicalState(**{
        name: read_restart_physical_state(run / f"{name}_start.nc") for name in ("driver", "sechiba", "stomate")
    })
    report = write_paper_restart_start_bundle(tmp_path / origin, state=state, physical_state=physical, pft_layout=original.pft_layout)
    restored = reference_case_first_step_restart_state(CONFIG, root=ROOT, run_dir=report.output_directory)
    for name in STOMATE_GAS_FIELDS:
        np.testing.assert_array_equal(getattr(restored.stomate_readstart.gas_state, name), getattr(base.gas_state, name))


def test_gas_contract_declares_both_initialization_routes_and_lifecycle():
    contract = yaml.safe_load((ROOT / "docs/source_audits/restart_state_lifecycle.yaml").read_text(encoding="utf-8"))
    for name in STOMATE_GAS_FIELDS:
        item = contract["restart_field_contracts"][name]
        assert item["initialization_constants"] == {"O2_init_conc": 298.3, "CH4_init_conc": .001267}
        assert "cold_owner" in item and "restart_extraction_owner" in item
        assert "half_hour_owner" in item and "day_end_owner" in item and "compiled_carry" in item
        assert "first_day_scaffold_half_hour_owner" in item
        assert contract["fixed_path_carry_fields"][name]["guard"] == "OK_PC=false"
