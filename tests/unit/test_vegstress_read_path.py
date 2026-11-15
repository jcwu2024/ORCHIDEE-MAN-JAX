"""Source-derived restart-to-initialization regression; no physical delegates run.

HYDROL hydrol_init3045-3058 and identified IOIPSL restcom1715-1736,
stringop80-113, mathelp1093-1123 define these expectations. They are source
contracts, not observations of a native Fortran executable or trajectory.
"""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset, default_fillvals

from jax_orchidee.sechiba import hydrol
from jax_orchidee.sechiba.restart_export import paper_sechiba_restart_state_from_finalize_state
from jax_orchidee.sechiba.restart_io import (
    DEFAULT_SECHIBA_RESTART_SCHEMA,
    PHYSICAL_RESTART_VARIABLES,
    SechibaRestartState,
    read_hydrol_init_restart_state,
    read_sechiba_restart_state,
    sechiba_finalize_source_state_from_restart,
    write_sechiba_restart_state,
)
from jax_orchidee.stomate.restart_io import StomateRestartPhysicalState


@pytest.fixture
def restart_fixture(tmp_path):
    schema = json.loads(DEFAULT_SECHIBA_RESTART_SCHEMA.read_text(encoding="utf8"))
    schema["dimensions"]["x"]["size"] = 2
    sizes = {name: row["size"] for name, row in schema["dimensions"].items()}
    fields = {}
    for name, row in schema["variables"].items():
        if name in PHYSICAL_RESTART_VARIABLES:
            continue
        axes = [axis for axis in row["dimensions"] if axis not in {"time", "x", "y"}]
        fields[name] = np.zeros((2, *(sizes[axis] for axis in reversed(axes))), dtype=row["dtype"])
    assert len(fields) == 93 and fields["us"].shape == (2, 14, 6, 11)
    fields["njsc"][:] = 6
    fields["veget_max"][:, 13] = 1.0
    fields["veget"][:, 13] = [0.8, 0.6]
    fields["moistc"][:] = fields["moistcl"][:] = 0.3
    fields["free_drain_coef"][:] = 1.0
    fields["zwt_force"][:] = 1.0e20
    fields["resdist"][:, 3] = 1.0
    fields["vegstress"][:] = 0.17
    pref = np.asarray([1] + [3] * 12 + [4], dtype=np.int32)
    selected = np.stack((np.linspace(0.08, 0.6, 14), np.linspace(0.11, 0.9, 14)))
    fields["us"][:] = 0.91  # All nonselected soil tiles are material decoys.
    for point in range(2):
        for pft, tile in enumerate(pref):
            fields["us"][point, pft, tile - 1, :] = selected[point, pft] / 11
    schema_path = tmp_path / "tiny-schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf8")
    physical = StomateRestartPhysicalState(
        nav_lon=np.zeros((1, 2)), nav_lat=np.zeros((1, 2)), nav_lev=np.asarray([1.0]),
        time=np.asarray([0.0]), time_steps=np.asarray([1]),
    )
    original = tmp_path / "complete.nc"
    write_sechiba_restart_state(original, state=SechibaRestartState(fields, {}),
        physical_state=physical, schema_path=schema_path)
    return SimpleNamespace(fields=fields, pref=pref, selected=selected, schema=schema_path,
        physical=physical, original=original)


def _initialize(monkeypatch, fixture, restart, *, points=slice(None)):
    values = fixture.fields
    veget = values["veget"][points]
    npts, nvm = veget.shape
    aliases = {"mc": "moistc", "mcl": "moistcl"}
    cold_values = {name: None for name in hydrol.HydrolColdStartState._fields}
    for name in cold_values:
        field = aliases.get(name, name)
        if field in values:
            cold_values[name] = values[field][points].copy()
    cold_values.update(us=np.zeros((npts, nvm, 6, 11)), zforce=False,
        vegtot=np.sum(veget, axis=1), explicit_snow=None, provenance=(), notes=())
    cold = hydrol.HydrolColdStartState(**cold_values)
    calls = []

    def cold_delegate(**kwargs):
        calls.append("cold")
        return cold

    def var_delegate(**kwargs):
        calls.append("var_init")
        assert "vegstress" not in kwargs
        return SimpleNamespace(_asdict=lambda: {})

    def vegetation_delegate(**kwargs):
        calls.append("vegetation")
        return SimpleNamespace(_asdict=lambda: {})

    before = {} if restart is None else {name: np.asarray(value).copy() for name, value in restart.items()}
    with monkeypatch.context() as patch:
        patch.setattr(hydrol, "_hydrol_init_usda_parameters", lambda values: {})
        patch.setattr(hydrol, "hydrol_cold_start_state", cold_delegate)
        patch.setattr(hydrol, "hydrol_var_init", var_delegate)
        patch.setattr(hydrol, "hydrol_vegupd_static_state", vegetation_delegate)
        result = hydrol.hydrol_init_pft14_owner(
            veget=veget, veget_max=values["veget_max"][points],
            soiltile=values["resdist"][points], pref_soil_veg=fixture.pref,
            njsc=values["njsc"][points], znh_m=np.linspace(0.01, 2.0, 11),
            dnh_m=np.ones(11), dlh_m=np.ones(11), humcste=np.ones(14),
            altmax=np.zeros((npts, nvm)), restart_state=restart,
        )
    for name, value in before.items():
        np.testing.assert_array_equal(restart[name], value)
    assert calls == ["cold", "var_init", "vegetation"]
    assert np.asarray(result.state["vegstress"]).shape == (npts, nvm)
    np.testing.assert_array_equal(result.state["vegstress"], result.span_state["vegstress"])
    return result


def _file_variant(original, output, mode):
    """Keep the otherwise-complete real NetCDF while varying one variable."""
    with Dataset(original) as source, Dataset(output, "w") as target:
        for name, dimension in source.dimensions.items():
            target.createDimension(name, None if dimension.isunlimited() else len(dimension))
        target.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
        for name, variable in source.variables.items():
            if name == "vegstress" and mode == "absent":
                continue
            attrs = {name: variable.getncattr(name) for name in variable.ncattrs()}
            fill = attrs.pop("_FillValue", None)
            if name == "vegstress" and mode in {"fill999999", "fill1e20", "default_fill"}:
                fill = {"fill999999": 999999.0, "fill1e20": 1.0e20, "default_fill": None}[mode]
                attrs.pop("missing_value", None)
            options = {} if fill is None else {"fill_value": fill}
            new = target.createVariable(name, variable.dtype, variable.dimensions, **options)
            new.setncatts(attrs)
            variable.set_auto_maskandscale(False)
            data = np.asarray(variable[:])
            if name == "vegstress":
                if mode in {"fill999999", "fill1e20", "default_fill"}:
                    continue  # Exercise actual unwritten NetCDF fill backing numbers.
                if mode in {"all999999", "missing_value999999"}:
                    data[:] = 999999.0
                elif mode == "partial_coordinate":
                    data[0, 13, 0, 0] = 999999.0
                elif mode == "partial_land":
                    data[0, :, 0, 0] = 999999.0
                elif mode == "near_marker":
                    data[:] = np.nextafter(999999.0, np.inf)
                if mode == "missing_value999999":
                    new.setncattr("missing_value", 999999.0)
            new[:] = data


@pytest.mark.parametrize("mode", [
    "ordinary", "all999999", "partial_coordinate", "partial_land", "near_marker",
    "fill999999", "missing_value999999", "fill1e20", "default_fill", "absent",
])
def test_real_file_init_owner_and_finalize_wire_follow_source(monkeypatch, tmp_path, restart_fixture, mode):
    fixture = restart_fixture
    path = tmp_path / "input.nc"
    _file_variant(fixture.original, path, mode)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = read_sechiba_restart_state(path)
    assert len(raw.fields) == (92 if mode == "absent" else 93)
    # This production adapter owns field selection, renaming and known absence.
    restart = read_hydrol_init_restart_state(path, npts=2, nvm=14)
    np.testing.assert_array_equal(restart["mc"], fixture.fields["moistc"])
    np.testing.assert_array_equal(restart["us"], fixture.fields["us"])
    supplied = np.asarray(restart["vegstress"]).copy()
    expected = supplied.copy()
    if mode in {"all999999", "fill999999", "missing_value999999", "absent"}:
        # Literal source3055: the existing val_exp is part of the sum.
        expected = 999999.0 + fixture.selected
    if mode == "absent":
        assert "vegstress" not in raw.fields
        np.testing.assert_array_equal(supplied, np.full((2, 14), 999999.0))
        with pytest.raises(ValueError, match="missing=\\['vegstress'\\]"):
            sechiba_finalize_source_state_from_restart(raw)
        with pytest.raises(ValueError, match="missing=\\['vegstress'\\]"):
            write_sechiba_restart_state(tmp_path / "incomplete-output.nc", state=raw,
                physical_state=fixture.physical, schema_path=fixture.schema)
        assert not (tmp_path / "incomplete-output.nc").exists()
    else:
        np.testing.assert_array_equal(supplied, raw.fields["vegstress"])
    if mode == "fill1e20":
        np.testing.assert_array_equal(supplied, np.full((2, 14), 1.0e20))
    if mode == "default_fill":
        np.testing.assert_array_equal(supplied, np.full((2, 14), default_fillvals["f8"]))
    result = _initialize(monkeypatch, fixture, restart)
    np.testing.assert_array_equal(result.state["vegstress"], expected)
    np.testing.assert_allclose(result.span_state["humrelv"][:, :, fixture.pref[13] - 1][..., 13],
        fixture.selected[:, 13], rtol=0, atol=1e-15)
    # Other components are fixture-supplied. The HYDROL result completes the
    # missing variable before the unchanged complete-state finalizer/codec.
    initialized = SechibaRestartState({**raw.fields, "vegstress": np.asarray(result.state["vegstress"])},
        raw.provenance_by_field)
    source = sechiba_finalize_source_state_from_restart(initialized)
    exported = paper_sechiba_restart_state_from_finalize_state({**source, **result.state}, kjit=0)
    assert len(exported.fields) == 93
    np.testing.assert_array_equal(exported.fields["vegstress"], expected)
    output = tmp_path / "finalized.nc"
    write_sechiba_restart_state(output, state=exported, physical_state=fixture.physical, schema_path=fixture.schema)
    np.testing.assert_array_equal(read_sechiba_restart_state(output).fields["vegstress"], expected)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    print("READ_PATH " + json.dumps({"mode": mode, "input_fields": len(raw.fields), "output_fields": 93,
        "selected_soil_PFT14": 4, "source_expected_PFT14": expected[:, 13].tolist(),
        "owner_span_finalize_written_PFT14": np.asarray(result.state["vegstress"])[:, 13].tolist(),
        "input_file_unchanged": True, "physical_delegates_replaced": 4}, sort_keys=True))


@pytest.mark.parametrize("mode", ["cold_api", "omitted_mapping_key"])
def test_file_absence_is_distinct_from_cold_and_omitted_mapping(monkeypatch, restart_fixture, mode):
    fixture = restart_fixture
    restart = None if mode == "cold_api" else {"us": fixture.fields["us"].copy()}
    result = _initialize(monkeypatch, fixture, restart)
    expected = np.zeros((2, 14)) if mode == "cold_api" else fixture.selected
    np.testing.assert_allclose(result.state["vegstress"], expected, rtol=0, atol=1e-15)
    assert ("vegstress" in result.restart_inputs) is False


def test_all_is_evaluated_on_the_materialized_local_partition(monkeypatch, restart_fixture):
    fixture = restart_fixture
    stress = np.full((2, 14), 999999.0)
    stress[1, 0] = 0.17
    whole = _initialize(monkeypatch, fixture, {"vegstress": stress, "us": fixture.fields["us"]})
    np.testing.assert_array_equal(whole.state["vegstress"], stress)
    local = _initialize(monkeypatch, fixture, {"vegstress": stress[:1], "us": fixture.fields["us"][:1]},
        points=slice(0, 1))
    np.testing.assert_array_equal(local.state["vegstress"], 999999.0 + fixture.selected[:1])


@pytest.mark.parametrize("shape", [(1, 14), (2, 13), (28,)])
def test_owner_rejects_stress_with_inconsistent_shape(monkeypatch, restart_fixture, shape):
    with pytest.raises(ValueError, match="vegstress"):
        _initialize(monkeypatch, restart_fixture, {"vegstress": np.zeros(shape)})


def test_file_adapter_binds_the_declared_owner_dimensions(restart_fixture):
    with pytest.raises(ValueError, match="us must have leading dimensions"):
        read_hydrol_init_restart_state(restart_fixture.original, npts=3, nvm=14)


def _source_stress_expectation(fields, pref, stress):
    """Independent NumPy indexing of hydrol_init3039/3052-3058."""
    selected = np.stack([
        np.sum(fields["us"][:, pft, tile - 1, :], axis=-1)
        for pft, tile in enumerate(pref)
    ], axis=1)
    if stress is None:
        return selected
    return stress + selected if np.all(stress == 999999.0) else stress.copy()


def _paper_entry_files(tmp_path, fixture, mode):
    path = tmp_path / "sechiba_start.nc"
    _file_variant(fixture.original, path, mode)
    with Dataset(tmp_path / "driver_start.nc", "w"):
        pass
    # The actual aggregate reader inspects these four metadata shapes. All
    # unrelated STOMATE scientific readers are replaced below.
    with Dataset(tmp_path / "stomate_start.nc", "w") as nc:
        for name, size in (("x", 2), ("one", 1), ("veget", 14), ("snow", 3)):
            nc.createDimension(name, size)
        for name in ("uo_0", "fwet_series", "MatrixV"):
            nc.createVariable(name, "f8", ("x", "one"))[:] = 0.0
        nc.createVariable("O2_snow", "f8", ("x", "veget", "snow"))[:] = 0.0
    return path


@pytest.mark.parametrize("mode", ["absent", "all999999", "ordinary", "partial_land"])
def test_production_paper_file_entry_initializes_stress_once_and_writes_back(
    monkeypatch, tmp_path, restart_fixture, mode,
):
    from pathlib import Path
    import jax_orchidee.coupled as coupled
    import jax_orchidee.driver.restart_state as entry

    fixture = restart_fixture
    path = _paper_entry_files(tmp_path, fixture, mode)
    input_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = read_sechiba_restart_state(path)
    root = Path(__file__).resolve().parents[2]
    config = root / "configs/orchidee_man_250919.yaml"
    # Keep the actual catalog, stable-PFT layout, run-scalars selection, raw
    # SECHIBA reader, HYDROL diagnostic anchors and initialization adapter.
    scalars = entry.read_run_scalars(config)
    monkeypatch.setattr(entry, "read_stomate_restart_entry_state", lambda *a, **kw: SimpleNamespace(
        biomass=np.zeros((2, 14, 2)), carbon_32l=np.zeros((2, 3, 14, 1)),
        carbon=np.zeros((2, 3, 14)), litter=np.zeros((2, 2, 14)),
    ))
    monkeypatch.setattr(entry, "read_stomate_daily_accumulator_state",
                        lambda *a, **kw: SimpleNamespace(t2m_daily=np.ones(2)))
    monkeypatch.setattr(entry, "read_stomate_restart_season_state",
                        lambda *a, **kw: SimpleNamespace(tsoil_month=np.ones((2, 11))))
    monkeypatch.setattr(coupled, "paper_case_season_time_scales",
                        lambda *a, **kw: SimpleNamespace(tau_longterm_max=1095.0))
    for name in (
        "read_stomate_readstart_states_from_template", "read_driver_albedo_restart",
        "read_sechiba_static_restart", "read_diffuco_first_step_restart_state",
        "read_enerbil_soil_thermal_state_restart", "read_thermosoil_restart_state",
        "read_slowproc_restart_entry_state",
    ):
        monkeypatch.setattr(entry, name, lambda *a, **kw: SimpleNamespace())
    for name in ("_hydrol_init_usda_parameters", "hydrol_cold_start_state",
                 "hydrol_var_init", "hydrol_vegupd_static_state"):
        monkeypatch.setattr(hydrol, name, lambda *a, **kw: pytest.fail("file entry ran physical initialization"))
    actual_transition = hydrol.hydrol_init_vegstress
    transitions = []

    def transition(**kwargs):
        transitions.append(kwargs)
        return actual_transition(**kwargs)

    monkeypatch.setattr(hydrol, "hydrol_init_vegstress", transition)
    result = entry.reference_case_first_step_restart_state(config, root=root, run_dir=tmp_path)
    expected = _source_stress_expectation(
        fixture.fields, scalars.pref_soil_veg,
        raw.fields.get("vegstress", np.full((2, 14), 999999.0)),
    )
    assert len(transitions) == 1
    np.testing.assert_array_equal(result.sechiba_restart_state.fields["vegstress"], expected)
    # Raw anchors remain an inventory of the input file, not initialized state.
    if mode == "absent":
        assert result.hydrol.vegstress is None
        assert "vegstress" in result.hydrol.missing
        with pytest.raises(ValueError, match="missing=\\['vegstress'\\]"):
            sechiba_finalize_source_state_from_restart(raw)
    else:
        np.testing.assert_array_equal(result.hydrol.vegstress, raw.fields["vegstress"])
    for name, value in raw.fields.items():
        if name != "vegstress":
            np.testing.assert_array_equal(result.sechiba_restart_state.fields[name], value)
    # Actual initializer output goes directly through strict93 and all seven
    # finalizers. The test supplies neither a replacement owner nor writeback.
    source = sechiba_finalize_source_state_from_restart(result.sechiba_restart_state)
    exported = paper_sechiba_restart_state_from_finalize_state(source, kjit=0)
    output = tmp_path / "paper-initialized.nc"
    write_sechiba_restart_state(output, state=exported, physical_state=fixture.physical,
                               schema_path=fixture.schema)
    written = read_sechiba_restart_state(output)
    assert len(written.fields) == 93
    np.testing.assert_array_equal(written.fields["vegstress"], expected)
    assert len(transitions) == 1  # serialization never replays initialization
    assert hashlib.sha256(path.read_bytes()).hexdigest() == input_sha
    print("PRODUCTION_ENTRY " + json.dumps({"route": "paper_file_initialization", "mode": mode,
        "transition_calls": len(transitions), "fields_written": len(written.fields),
        "PFT14": expected[:, 13].tolist(), "input_file_unchanged": True}))


def _dispatcher_physical_boundaries(monkeypatch, fixture):
    aliases = {"mc": "moistc", "mcl": "moistcl"}
    values = {name: None for name in hydrol.HydrolColdStartState._fields}
    for name in values:
        field = aliases.get(name, name)
        if field in fixture.fields:
            values[name] = fixture.fields[field].copy()
    values.update(us=np.zeros((2, 14, 6, 11)), zforce=False,
        vegtot=np.sum(fixture.fields["veget"], axis=1), explicit_snow=None, provenance=(), notes=())
    cold = hydrol.HydrolColdStartState(**values)
    calls = []

    def boundary(name, result):
        def run(*args, **kwargs):
            calls.append(name)
            return result
        return run

    monkeypatch.setattr(hydrol, "_hydrol_init_usda_parameters", boundary("soil_parameters", {}))
    monkeypatch.setattr(hydrol, "hydrol_cold_start_state", boundary("cold", cold))
    for attribute, label in (("hydrol_var_init", "var_init"), ("hydrol_vegupd_static_state", "vegetation")):
        monkeypatch.setattr(hydrol, attribute, boundary(label, SimpleNamespace(_asdict=lambda: {})))
    return calls


@pytest.mark.parametrize("mode", ["absent", "all999999", "ordinary", "partial_land", "cold_api"])
def test_production_default_dispatcher_selects_hydrol_and_writes_scientific_state(
    monkeypatch, tmp_path, restart_fixture, mode,
):
    from jax_orchidee.sechiba.initialize import (
        SechibaInitDimensions, SechibaInitializationError, SechibaInitializationOwners,
        sechiba_initialize_pft14_explicit,
    )

    fixture = restart_fixture
    calls = _dispatcher_physical_boundaries(monkeypatch, fixture)
    restart = None
    if mode != "cold_api":
        path = tmp_path / "dispatcher-input.nc"
        _file_variant(fixture.original, path, mode)
        restart = read_hydrol_init_restart_state(path, npts=2, nvm=14)
        expected = _source_stress_expectation(fixture.fields, fixture.pref, restart["vegstress"])
    else:
        expected = np.zeros((2, 14))
    dimensions = SechibaInitDimensions(npts=2, nvm=14, nslm=11, nstm=6,
        **{name: 1 for name in ("nnobio", "itimetide", "nflow", "nexp", "nctext", "ncarb", "nparts",
                               "nelements", "nlitt", "ndeep", "ndoc", "npool", "nleafages", "nlai",
                               "ngrnd", "nsnow", "npco2")})
    callbacks = []

    def after_hydrol(state, results):
        callbacks.append("condveg_inputs")
        np.testing.assert_array_equal(state["vegstress"], expected)
        assert "state" not in state and "span_state" not in state
        assert isinstance(results["hydrol_initialize"], hydrol.HydrolInitPFT14Result)
        return {}

    # Omit hydrol_initialize: exercise the actual production default selection.
    owners = SechibaInitializationOwners(
        slowproc_initialize=lambda: {"veget_max": fixture.fields["veget_max"],
                                    "co2_flux": np.zeros((2, 14)), "rot_cmd_max": 1},
        diffuco_initialize=lambda: {},
        enerbil_initialize=lambda: {"qsurf": np.asarray([0.01, 0.02])},
        condveg_initialize=lambda: {"z0m": np.ones(2), "z0h": np.ones(2), "emis": np.ones(2)},
        thermosoil_initialize=lambda: {},
    )
    owner_inputs = {name: {} for name in ("slowproc_initialize", "diffuco_initialize",
                                         "enerbil_initialize", "thermosoil_initialize")}
    owner_inputs["condveg_initialize"] = after_hydrol
    owner_inputs["hydrol_initialize"] = dict(
        veget=fixture.fields["veget"], veget_max=fixture.fields["veget_max"],
        soiltile=fixture.fields["resdist"], pref_soil_veg=fixture.pref,
        njsc=fixture.fields["njsc"], znh_m=np.linspace(0.01, 2.0, 11),
        dnh_m=np.ones(11), dlh_m=np.ones(11), humcste=np.ones(14),
        altmax=np.zeros((2, 14)), restart_state=restart,
    )
    common = dict(dimensions=dimensions, forcing={"qair": np.ones(2)},
        pft_parameters={"veget_max_default": fixture.fields["veget_max"]},
        owner_inputs=owner_inputs, owners=owners)
    result = sechiba_initialize_pft14_explicit(l_first=True, **common)
    assert calls == ["soil_parameters", "cold", "var_init", "vegetation"]
    assert callbacks == ["condveg_inputs"]
    np.testing.assert_array_equal(result.state["vegstress"], expected)
    np.testing.assert_array_equal(result.owner_results["hydrol_initialize"].span_state["vegstress"], expected)
    if mode == "all999999":
        with pytest.raises(SechibaInitializationError, match="repeated call"):
            sechiba_initialize_pft14_explicit(l_first=result.l_first, **common)
        assert len(calls) == 4 and len(callbacks) == 1
    print("PRODUCTION_ENTRY " + json.dumps({"route": "default_dispatcher", "mode": mode,
        "full_owner_calls": 1, "physical_boundaries_replaced": calls,
        "scientific_writeback_callback_calls": len(callbacks), "PFT14": expected[:, 13].tolist()}))


@pytest.mark.parametrize("mode", ["new_native_run", "continuous"])
@pytest.mark.parametrize("stress_mode", ["all999999", "partial_land", "unresolved_us"])
def test_production_year_entry_selects_only_the_new_process_initialization_phase(
    monkeypatch, tmp_path, restart_fixture, mode, stress_mode,
):
    from dataclasses import make_dataclass
    import jax_orchidee.driver.orchestration as orch
    from jax_orchidee.driver.source_clock import SourceClockRuntime, source_clock_from_inputs

    fixture = restart_fixture
    for year in (1961, 1962):
        with Dataset(tmp_path / f"forcing-{year}.nc", "w") as nc:
            nc.createDimension("time", 3)
            time = nc.createVariable("time", "f8", ("time",))
            time.units = f"seconds since {year}-01-01 00:00:00"
            time.calendar = "gregorian"
            time[:] = [0.0, 1800.0, 3600.0]
    initial = source_clock_from_inputs(tmp_path / "forcing-1961.nc", dt_sechiba=1800.0,
                                       run_id="old-native-run", time_length=1)
    previous_runtime = SourceClockRuntime(initial.binding, np.int64(17520), np.int64(17519), np.int64(17520))
    if mode == "continuous":
        setup = source_clock_from_inputs(tmp_path / "forcing-1962.nc", dt_sechiba=1800.0,
            run_id="old-native-run", time_length=1, run_mode="continuous", previous_runtime=previous_runtime)
    else:
        setup = source_clock_from_inputs(tmp_path / "forcing-1962.nc", dt_sechiba=1800.0,
            run_id="new-native-run", time_length=1)
    stress = np.full((2, 14), 999999.0)
    if stress_mode == "partial_land":
        stress[1] = 0.17
    source = sechiba_finalize_source_state_from_restart(
        SechibaRestartState({**fixture.fields, "vegstress": stress}, {}))
    if stress_mode == "unresolved_us":
        source["us"] = source["us"].copy()
        source["us"][:, :, :, 0] = 999999.0
    packet = orch.DriverPreviousStepStatePacket(tstep=17519,
        fields_by_component={"sechiba_finalize_state": source,
                             "hydrol_previous_step_state": {"nroot": np.ones((2, 14, 11))}},
        provenance_by_component={}, source_clock_runtime=previous_runtime)
    context_values = dict(config_path=tmp_path / "config.yaml", run_def_path=tmp_path / "run.def",
        run_def_values={}, first_step_bundle=object(), reference_run_dir=None,
        source_clock_setup=setup, run_scalars=SimpleNamespace(pref_soil_veg=fixture.pref),
        runtime=SimpleNamespace(dt_stomate=86400.0, dt_sechiba=1800.0), dt_sechiba=1800.0)
    TinyContext = make_dataclass("TinyContext", [(name, object) for name in context_values])
    context = TinyContext(**context_values)
    # Orthogonal clock admission, completeness checks and weather IO are
    # replaced. The full production year-entry/rebase and phase choice run.
    monkeypatch.setattr(orch, "source_clock_setup_from_config", lambda *a, **kw: setup)
    monkeypatch.setattr(orch, "driver_year_handoff_state_gaps", lambda state: ())
    monkeypatch.setattr(orch, "_require_source_day_alignment", lambda *a, **kw: None)
    monkeypatch.setattr(orch, "_paper_1961_step_bundle_from_context", lambda *a, **kw: object())
    monkeypatch.setattr(orch, "build_intersurf_first_step_payload", lambda *a, **kw: object())
    monkeypatch.setattr(orch, "restart_thermosoil_initialization_supported", lambda context: False)
    actual_transition = hydrol.hydrol_init_vegstress
    calls = []

    def transition(**kwargs):
        calls.append("stress_initialization")
        return actual_transition(**kwargs)

    monkeypatch.setattr(hydrol, "hydrol_init_vegstress", transition)
    # Stop at the first runtime delegate: no timestep or physical kernel runs.
    def runtime(*args, **kwargs):
        calls.append("runtime_delegate")
        return kwargs["previous_state"]

    monkeypatch.setattr(orch, "paper_1961_driver_later_day_runtime_result", runtime)
    if stress_mode == "unresolved_us" and mode == "new_native_run":
        with pytest.raises(NotImplementedError, match="upstream US_INIT"):
            orch.paper_1961_driver_restart_year_start_day_result("unused.yaml",
                previous_year_end_state=packet, year=1962, prepared_context=context, run_mode=mode)
        assert calls == []
        np.testing.assert_array_equal(source["vegstress"], stress)
        print("PRODUCTION_ENTRY " + json.dumps({"route": "year_entry", "mode": mode,
            "stress_mode": stress_mode, "calls": calls, "explicit_upstream_boundary": "US_INIT"}))
        return
    result = orch.paper_1961_driver_restart_year_start_day_result("unused.yaml",
        previous_year_end_state=packet, year=1962, prepared_context=context, run_mode=mode)
    expected = (_source_stress_expectation(fixture.fields, fixture.pref, stress)
                if mode == "new_native_run" else stress)
    np.testing.assert_array_equal(result.fields_by_component["sechiba_finalize_state"]["vegstress"], expected)
    assert calls == (["stress_initialization", "runtime_delegate"] if mode == "new_native_run"
                     else ["runtime_delegate"])
    np.testing.assert_array_equal(source["vegstress"], stress)
    assert result.tstep == -1 and packet.tstep == 17519
    for name, value in source.items():
        if name != "vegstress":
            np.testing.assert_array_equal(result.fields_by_component["sechiba_finalize_state"][name], value)
    print("PRODUCTION_ENTRY " + json.dumps({"route": "year_entry", "mode": mode,
        "stress_mode": stress_mode, "calls": calls, "PFT14": expected[:, 13].tolist(),
        "original_packet_unchanged": True, "model_steps": 0}))


def test_production_complete_checkpoint_decode_preserves_already_initialized_state(
    monkeypatch, tmp_path, restart_fixture,
):
    import pickle
    from jax_orchidee.driver.orchestration import DriverPreviousStepStatePacket
    from jax_orchidee.runners.multiyear import _read_year_end_state_cache

    source = sechiba_finalize_source_state_from_restart(SechibaRestartState(
        {**restart_fixture.fields, "vegstress": np.full((2, 14), 999999.0)}, {}))
    assert len(source) == 93
    state = DriverPreviousStepStatePacket(tstep=17519,
        fields_by_component={"sechiba_finalize_state": source}, provenance_by_component={})
    path = tmp_path / "complete-checkpoint.pkl"
    with path.open("wb") as stream:
        pickle.dump({"state": state, "start_year": 1961, "end_year": 1961}, stream)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(hydrol, "hydrol_init_vegstress",
                        lambda **kwargs: pytest.fail("complete checkpoint decode replayed initialization"))
    result = _read_year_end_state_cache(path)
    restored = result["state"]
    assert restored.tstep == state.tstep
    for name, value in source.items():
        np.testing.assert_array_equal(restored.fields_by_component["sechiba_finalize_state"][name], value)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    print("PRODUCTION_ENTRY " + json.dumps({"route": "complete_checkpoint_decode",
        "transition_calls": 0, "input_file_unchanged": True}))


def test_production_owner_preserves_explicit_none_rejection(monkeypatch, restart_fixture):
    with pytest.raises(ValueError, match="vegstress"):
        _initialize(monkeypatch, restart_fixture, {"vegstress": None})


@pytest.mark.parametrize("route", ["file_entry", "full_owner"])
@pytest.mark.parametrize("marker_scope", ["whole_layer", "partial_coordinate"])
def test_production_materialized_us_boundary_is_explicit_and_local(
    monkeypatch, tmp_path, restart_fixture, route, marker_scope,
):
    from jax_orchidee.sechiba.restart_io import read_sechiba_init_restart_state

    fixture = restart_fixture
    us = fixture.fields["us"].copy()
    if marker_scope == "whole_layer":
        us[:, :, :, 0] = 999999.0
    else:
        # Isolate the layer-ALL guard with an exact integer-plus-zero sum.
        # Mixed large/fractional reduction order is a separate numeric test.
        us[0, 13, 3, :] = 0.0
        us[0, 13, 3, 0] = 999999.0
    stress = np.full((2, 14), 999999.0)

    def invoke():
        if route == "full_owner":
            return _initialize(monkeypatch, fixture, {"us": us, "vegstress": stress}).state["vegstress"]
        path = tmp_path / "us-boundary.nc"
        write_sechiba_restart_state(path,
            state=SechibaRestartState({**fixture.fields, "us": us, "vegstress": stress}, {}),
            physical_state=fixture.physical, schema_path=fixture.schema)
        return read_sechiba_init_restart_state(path, npts=2, nvm=14,
            pref_soil_veg=fixture.pref).fields["vegstress"]

    if marker_scope == "whole_layer":
        with pytest.raises(NotImplementedError, match="upstream US_INIT"):
            invoke()
    else:
        expected = _source_stress_expectation({"us": us}, fixture.pref, stress)
        np.testing.assert_array_equal(invoke(), expected)
    print("PRODUCTION_ENTRY " + json.dumps({"route": route, "marker_scope": marker_scope,
        "upstream_US_INIT_boundary": marker_scope == "whole_layer"}))


def test_production_in_memory_later_day_keeps_state_before_its_runtime_delegate(
    monkeypatch, restart_fixture,
):
    import jax_orchidee.driver.orchestration as orch

    source = sechiba_finalize_source_state_from_restart(SechibaRestartState(
        {**restart_fixture.fields, "vegstress": np.full((2, 14), 999999.0)}, {}))
    packet = orch.DriverPreviousStepStatePacket(tstep=47,
        fields_by_component={"sechiba_finalize_state": source}, provenance_by_component={})
    context = SimpleNamespace(run_def_path="unused.def", run_def_values={}, cwrr_grid=object(),
        runtime=SimpleNamespace(dt_sechiba=1800.0, dt_stomate=86400.0))
    monkeypatch.setattr(orch, "_source_clock_setup",
                        lambda context: SimpleNamespace(binding=SimpleNamespace(one_day_seconds=86400.0)))
    monkeypatch.setattr(orch, "_require_source_day_alignment", lambda *a, **kw: None)
    monkeypatch.setattr(hydrol, "hydrol_init_vegstress",
                        lambda **kw: pytest.fail("in-memory later-day entry replayed initialization"))
    monkeypatch.setattr(hydrol, "hydrol_init_restored_humrelv",
                        lambda **kw: pytest.fail("in-memory later-day entry reapplied file US_INIT admission"))
    observed = []

    class ReachedRuntimeDelegate(Exception):
        pass

    def runtime_inputs(**kwargs):
        observed.append(kwargs["current_state"])
        raise ReachedRuntimeDelegate

    monkeypatch.setattr(orch, "_paper_1961_later_day_transition_inputs", runtime_inputs)
    with pytest.raises(ReachedRuntimeDelegate):
        orch.paper_1961_driver_later_day_runtime_result("unused.yaml", previous_state=packet,
            day_index=2, year=1961, start_tstep=48, prepared_context=context)
    assert observed == [packet] and observed[0] is packet
    for name, value in source.items():
        np.testing.assert_array_equal(observed[0].fields_by_component["sechiba_finalize_state"][name], value)
    print("PRODUCTION_ENTRY " + json.dumps({"route": "in_memory_later_day_entry",
        "transition_calls": 0, "runtime_boundary_calls": len(observed), "model_steps": 0}))
