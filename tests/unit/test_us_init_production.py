"""Source-derived US_INIT/entry controls; all physical/runtime work is replaced."""
from dataclasses import make_dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace

from netCDF4 import Dataset
import numpy as np
import pytest

from jax_orchidee.driver.init import parse_run_def, read_run_scalars, reference_run_def_path
from jax_orchidee.sechiba import hydrol
from jax_orchidee.sechiba.io_contracts import SetvarContractError
from jax_orchidee.sechiba.restart_export import paper_sechiba_restart_state_from_finalize_state
from jax_orchidee.sechiba.restart_io import (
    SechibaRestartState, read_hydrol_init_restart_state, read_sechiba_init_restart_state,
    read_sechiba_restart_state, sechiba_finalize_source_state_from_restart, write_sechiba_restart_state,
)
from tests.unit.test_vegstress_read_path import (
    restart_fixture, _dispatcher_physical_boundaries, _paper_entry_files,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
MARKER = 999999.0


def _us_case(mode, shape, scalar):
    """Explicit source branch cases; expected layer indices are declared here."""
    original = np.full(shape, 0.125)
    selected = []
    if mode in {"missing", "all"}:
        original[:] = MARKER
        selected = list(range(shape[3]))
    elif mode == "mixed":
        original[:, :, :, [0, 3]] = MARKER
        selected = [0, 3]
        original[0, :, :, 1] = MARKER
        original[:, 0, :, 2] = MARKER
        original[:, :, 0, 4] = MARKER
    elif mode == "partial_land":
        original[0, :, :, 0] = MARKER
    elif mode == "partial_pft":
        original[:, 0, :, 0] = MARKER
    elif mode == "partial_soil":
        original[:, :, 0, 0] = MARKER
    elif mode != "ordinary":
        raise AssertionError(mode)
    expected = original.copy()
    expected[:, :, :, selected] = scalar
    return original, expected, selected


def _sum_source(us):
    # hydrol3040 SUM over levels; binary-exact inputs keep order irrelevant.
    result = np.empty(us.shape[:3])
    for index in np.ndindex(result.shape):
        result[index] = sum(float(us[(*index, level)]) for level in range(us.shape[3]))
    return result


def _stress_source(us, pref, original):
    humrelv = _sum_source(us)
    selected = np.empty(original.shape)
    for point, pft in np.ndindex(original.shape):
        selected[point, pft] = humrelv[point, pft, int(pref[pft]) - 1]
    return original + selected if np.all(original == MARKER) else original.copy()


class RecordingParameters(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = []

    def __contains__(self, key):
        self.reads.append(key)
        return super().__contains__(key)


@pytest.mark.parametrize("mode,token,scalar", [
    ("all", None, 0.0), ("all", "0.25", 0.25), ("all", "-2.0", -2.0),
    ("ordinary", "invalid", 0.0), ("partial_land", "invalid", 0.0),
    ("partial_pft", "invalid", 0.0), ("partial_soil", "invalid", 0.0),
    ("mixed", "2.5", 2.5),
])
def test_source_us_layers_are_local_and_configuration_is_lazy(mode, token, scalar):
    original, expected, selected = _us_case(mode, (2, 3, 2, 5), scalar)
    before = original.copy()
    values = RecordingParameters({} if token is None else {"US_INIT": token})
    result = hydrol.hydrol_init_us_state(us=original, run_def_values=values)
    np.testing.assert_array_equal(result.us, expected)
    np.testing.assert_array_equal(result.humrelv, _sum_source(expected))
    np.testing.assert_array_equal(original, before)
    assert values.reads == ["US_INIT"] * len(selected)


@pytest.mark.parametrize("invalid", ["bad", "0.25 0.5", "1d-1", [0.25], True, None, 0.25j])
def test_configuration_domain_is_checked_only_when_source_reads_it(invalid):
    original = np.full((2, 3, 2, 5), 0.125)
    np.testing.assert_array_equal(hydrol.hydrol_init_us_state(
        us=original, run_def_values={"US_INIT": invalid}).us, original)
    original[:, :, :, 2] = MARKER
    with pytest.raises(SetvarContractError, match="scalar REAL"):
        hydrol.hydrol_init_us_state(us=original, run_def_values={"US_INIT": invalid})


@pytest.mark.parametrize("invalid", [None, np.zeros((2, 3, 2)), np.zeros((2, 3, 2, 0))])
def test_us_input_domain_is_explicit(invalid):
    with pytest.raises((ValueError, TypeError)):
        hydrol.hydrol_init_us_state(us=invalid)


def _run_def(tmp_path, token):
    lines = reference_run_def_path(CONFIG).read_text(encoding="utf8").splitlines()
    lines = [line for line in lines if line.split("=", 1)[0].strip() != "US_INIT"]
    if token is not None:
        lines.append(f"US_INIT = {token}")
    path = tmp_path / "used-run.def"
    path.write_text("\n".join(lines) + "\n", encoding="utf8")
    return path, read_run_scalars(CONFIG, path)


def _write_us_file(tmp_path, fixture, mode, scalar):
    original, expected, _ = _us_case(mode, fixture.fields["us"].shape, scalar)
    fields = {**fixture.fields, "us": original, "vegstress": np.full((2, 14), MARKER)}
    complete = tmp_path / "us-input-complete.nc"
    write_sechiba_restart_state(complete, state=SechibaRestartState(fields, {}),
        physical_state=fixture.physical, schema_path=fixture.schema)
    path = tmp_path / "us-input.nc"
    with Dataset(complete) as source, Dataset(path, "w") as target:
        for name, dimension in source.dimensions.items():
            target.createDimension(name, None if dimension.isunlimited() else len(dimension))
        target.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
        for name, variable in source.variables.items():
            if name == "us" and mode == "missing":
                continue
            attrs = {name: variable.getncattr(name) for name in variable.ncattrs()}
            fill = attrs.pop("_FillValue", None)
            output = target.createVariable(name, variable.dtype, variable.dimensions, fill_value=fill)
            output.setncatts(attrs)
            output[:] = variable[:]
    return path, original, expected


def _paper_boundaries(monkeypatch):
    import jax_orchidee.coupled as coupled
    import jax_orchidee.driver.restart_state as entry
    monkeypatch.setattr(entry, "read_stomate_restart_entry_state", lambda *a, **kw: SimpleNamespace(
        biomass=np.zeros((2, 14, 2)), carbon_32l=np.zeros((2, 3, 14, 1)),
        carbon=np.zeros((2, 3, 14)), litter=np.zeros((2, 2, 14))))
    monkeypatch.setattr(entry, "read_stomate_daily_accumulator_state", lambda *a, **kw: SimpleNamespace(t2m_daily=np.ones(2)))
    monkeypatch.setattr(entry, "read_stomate_restart_season_state", lambda *a, **kw: SimpleNamespace(tsoil_month=np.ones((2, 11))))
    monkeypatch.setattr(coupled, "paper_case_season_time_scales", lambda *a, **kw: SimpleNamespace(tau_longterm_max=1095.0))
    for name in ("read_stomate_readstart_states_from_template", "read_driver_albedo_restart",
                 "read_sechiba_static_restart", "read_diffuco_first_step_restart_state",
                 "read_enerbil_soil_thermal_state_restart", "read_thermosoil_restart_state",
                 "read_slowproc_restart_entry_state"):
        monkeypatch.setattr(entry, name, lambda *a, **kw: SimpleNamespace())
    for name in ("_hydrol_init_usda_parameters", "hydrol_cold_start_state", "hydrol_var_init", "hydrol_vegupd_static_state"):
        monkeypatch.setattr(hydrol, name, lambda *a, **kw: pytest.fail("file reader ran a physical delegate"))
    return entry


def _consumer(raw_anchors, initialized_us):
    return hydrol.assemble_hydrol_first_step_precall_payload(
        enerbil_payload={}, hydrol_restart=raw_anchors, initialized_us=initialized_us,
        slowproc_restart=SimpleNamespace(), pref_soil_veg=np.asarray([1] + [3] * 12 + [4]),
        nstm=6, ext_coeff_vegetfrac=np.ones(14), ok_freeze_cwrr=False,
        static_payload_template={"veget": np.zeros((2, 14))},
    ).payload


@pytest.mark.parametrize("mode,token,scalar,stress_mode", [
    ("missing", None, 0.0, "all999999"), ("missing", "0.25", 0.25, "all999999"),
    ("all", "0.25", 0.25, "all999999"), ("mixed", "0.25", 0.25, "all999999"),
    ("partial_land", "bad", 0.0, "all999999"), ("ordinary", "bad", 0.0, "all999999"),
    ("all", "-2.0", -2.0, "ordinary"), ("mixed", "0.25", 0.25, "partial_land"),
])
def test_real_paper_file_entry_keeps_materialized_us_through_consumer_and_writeback(
    monkeypatch, tmp_path, restart_fixture, mode, token, scalar, stress_mode,
):
    fixture = restart_fixture
    run_def, scalars = _run_def(tmp_path, token)
    variant, original_us, expected_us = _write_us_file(tmp_path, fixture, mode, scalar)
    path = _paper_entry_files(tmp_path, SimpleNamespace(original=variant), stress_mode)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = read_sechiba_restart_state(path)
    entry = _paper_boundaries(monkeypatch)
    actual = hydrol.hydrol_init_us_state
    transitions = []

    def transition(**kwargs):
        transitions.append(kwargs["run_def_values"])
        return actual(**kwargs)

    monkeypatch.setattr(hydrol, "hydrol_init_us_state", transition)
    result = entry.reference_case_first_step_restart_state(CONFIG, root=ROOT, run_dir=tmp_path,
                                                           run_def_path=run_def)
    assert len(transitions) == 1 and transitions[0] == scalars.raw_run_def
    if mode == "missing":
        assert result.hydrol.us is None and "us" not in raw.fields
    else:
        np.testing.assert_array_equal(result.hydrol.us, original_us)
    source = sechiba_finalize_source_state_from_restart(result.sechiba_restart_state)
    np.testing.assert_array_equal(source["us"], expected_us)
    expected_stress = _stress_source(expected_us, scalars.pref_soil_veg, raw.fields["vegstress"])
    np.testing.assert_array_equal(source["vegstress"], expected_stress)
    # Compose the real entry's snapshot with the actual HYDROL consumer. The
    # full paper scaffold uses this same initialized_us argument; no model runs.
    payload = _consumer(result.hydrol, source["us"])
    np.testing.assert_array_equal(payload["us"], expected_us)
    np.testing.assert_array_equal(payload["humrelv"], _sum_source(expected_us))
    for name, value in raw.fields.items():
        if name not in {"us", "vegstress"}:
            np.testing.assert_array_equal(result.sechiba_restart_state.fields[name], value)
    output = tmp_path / "initialized-writeback.nc"
    write_sechiba_restart_state(output,
        state=paper_sechiba_restart_state_from_finalize_state(source, kjit=0),
        physical_state=fixture.physical, schema_path=fixture.schema)
    readback = read_sechiba_restart_state(output)
    np.testing.assert_array_equal(readback.fields["us"], expected_us)
    np.testing.assert_array_equal(readback.fields["vegstress"], expected_stress)
    assert len(readback.fields) == 93 and len(transitions) == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_missing_us_requires_explicit_runtime_dimensions_and_preserves_raw_codec(tmp_path, restart_fixture):
    path, _, expected = _write_us_file(tmp_path, restart_fixture, "missing", 0.0)
    raw = read_sechiba_restart_state(path)
    assert "us" not in raw.fields
    with pytest.raises(ValueError, match="missing=.*us"):
        sechiba_finalize_source_state_from_restart(raw)
    with pytest.raises(ValueError, match="explicit positive source nstm and nslm"):
        read_hydrol_init_restart_state(path, npts=2, nvm=14)
    mapping = read_hydrol_init_restart_state(path, npts=2, nvm=14, nstm=6, nslm=11)
    np.testing.assert_array_equal(mapping["us"], np.full(expected.shape, MARKER))
    with pytest.raises(SetvarContractError, match="scalar REAL"):
        read_sechiba_init_restart_state(path, npts=2, nvm=14, nstm=6, nslm=11,
            pref_soil_veg=restart_fixture.pref, run_def_values={"US_INIT": "bad"})
    with pytest.raises(ValueError, match="soil axis"):
        read_sechiba_init_restart_state(restart_fixture.original, npts=2, nvm=14,
            pref_soil_veg=restart_fixture.pref, nstm=5, nslm=11)


@pytest.mark.parametrize("mode", ["cold_none", "omitted_us", "file_missing", "mixed"])
def test_default_dispatcher_forwards_real_parameters_and_writes_full_us_state(
    monkeypatch, tmp_path, restart_fixture, mode,
):
    from jax_orchidee.sechiba.initialize import (
        SechibaInitDimensions, SechibaInitializationError, SechibaInitializationOwners,
        sechiba_initialize_pft14_explicit,
    )
    fixture = restart_fixture
    _, scalars = _run_def(tmp_path, "0.25")
    calls = _dispatcher_physical_boundaries(monkeypatch, fixture)
    expected_us = np.full((2, 14, 6, 11), 0.25)
    restart = None
    if mode == "omitted_us":
        restart = {"vegstress": np.full((2, 14), MARKER)}
    elif mode in {"file_missing", "mixed"}:
        path, _, expected_us = _write_us_file(tmp_path, fixture,
            "missing" if mode == "file_missing" else "mixed", 0.25)
        restart = read_hydrol_init_restart_state(path, npts=2, nvm=14, nstm=6, nslm=11)
    expected_humrelv = _sum_source(expected_us)
    original_stress = np.zeros((2, 14)) if mode == "cold_none" else np.full((2, 14), MARKER)
    expected_stress = (_stress_source(expected_us, scalars.pref_soil_veg, original_stress)
                       if mode != "cold_none" else expected_humrelv[:, :, 0])
    observed = []

    def var_init(**kwargs):
        calls.append("var_init")
        np.testing.assert_array_equal(kwargs["humrelv"], expected_humrelv)
        return SimpleNamespace(_asdict=lambda: {})

    monkeypatch.setattr(hydrol, "hydrol_var_init", var_init)

    def after_hydrol(state, results):
        observed.append("scientific_state")
        np.testing.assert_array_equal(state["us"], expected_us)
        np.testing.assert_array_equal(state["humrelv"], expected_humrelv)
        np.testing.assert_array_equal(state["vegstressv"], expected_humrelv)
        np.testing.assert_array_equal(state["vegstress"], expected_stress)
        return {}

    dimensions = SechibaInitDimensions(npts=2, nvm=14, nslm=11, nstm=6,
        **{name: 1 for name in ("nnobio", "itimetide", "nflow", "nexp", "nctext", "ncarb", "nparts",
                               "nelements", "nlitt", "ndeep", "ndoc", "npool", "nleafages", "nlai", "ngrnd", "nsnow", "npco2")})
    owners = SechibaInitializationOwners(
        slowproc_initialize=lambda: {"veget_max": fixture.fields["veget_max"], "co2_flux": np.zeros((2, 14)), "rot_cmd_max": 1},
        diffuco_initialize=lambda: {}, enerbil_initialize=lambda: {"qsurf": np.zeros(2)},
        condveg_initialize=lambda: {"z0m": np.ones(2), "z0h": np.ones(2), "emis": np.ones(2)},
        thermosoil_initialize=lambda: {})
    inputs = {name: {} for name in ("slowproc_initialize", "diffuco_initialize", "enerbil_initialize", "thermosoil_initialize")}
    inputs["condveg_initialize"] = after_hydrol
    inputs["hydrol_initialize"] = dict(veget=fixture.fields["veget"], veget_max=fixture.fields["veget_max"],
        soiltile=fixture.fields["resdist"], pref_soil_veg=scalars.pref_soil_veg, njsc=fixture.fields["njsc"],
        znh_m=np.linspace(0.01, 2.0, 11), dnh_m=np.ones(11), dlh_m=np.ones(11), humcste=np.ones(14),
        altmax=np.zeros((2, 14)), restart_state=restart)
    kwargs = dict(dimensions=dimensions, forcing={"qair": np.ones(2)},
        pft_parameters={"veget_max_default": fixture.fields["veget_max"]},
        owner_inputs=inputs, owners=owners, run_def_values=scalars.raw_run_def)
    result = sechiba_initialize_pft14_explicit(l_first=True, **kwargs)
    assert calls == ["soil_parameters", "cold", "var_init", "vegetation"]
    assert observed == ["scientific_state"]
    np.testing.assert_array_equal(result.owner_results["hydrol_initialize"].span_state["us"], expected_us)
    with pytest.raises(SechibaInitializationError, match="repeated call"):
        sechiba_initialize_pft14_explicit(l_first=result.l_first, **kwargs)
    assert len(calls) == 4 and len(observed) == 1


@pytest.mark.parametrize("mode", ["new_native_run", "continuous"])
@pytest.mark.parametrize("us_mode", ["all", "mixed"])
def test_year_entry_updates_consumer_mirror_only_at_new_native_phase(
    monkeypatch, tmp_path, restart_fixture, mode, us_mode,
):
    import jax_orchidee.driver.orchestration as orch
    from jax_orchidee.driver.source_clock import SourceClockRuntime, source_clock_from_inputs
    fixture = restart_fixture
    run_def, scalars = _run_def(tmp_path, "0.25" if mode == "new_native_run" else "invalid")
    for year in (1961, 1962):
        with Dataset(tmp_path / f"forcing-{year}.nc", "w") as nc:
            nc.createDimension("time", 3)
            axis = nc.createVariable("time", "f8", ("time",))
            axis.units, axis.calendar = f"seconds since {year}-01-01 00:00:00", "gregorian"
            axis[:] = [0.0, 1800.0, 3600.0]
    initial = source_clock_from_inputs(tmp_path / "forcing-1961.nc", dt_sechiba=1800.0, run_id="old", time_length=1)
    previous_runtime = SourceClockRuntime(initial.binding, np.int64(17520), np.int64(17519), np.int64(17520))
    setup = source_clock_from_inputs(tmp_path / "forcing-1962.nc", dt_sechiba=1800.0,
        run_id="old" if mode == "continuous" else "new", time_length=1,
        run_mode=mode, previous_runtime=previous_runtime if mode == "continuous" else None)
    us, initialized, _ = _us_case(us_mode, fixture.fields["us"].shape, 0.25)
    source = sechiba_finalize_source_state_from_restart(SechibaRestartState(
        {**fixture.fields, "us": us, "vegstress": np.full((2, 14), MARKER)}, {}))
    mirror = {"us": us.copy(), "humrelv": _sum_source(us), "nroot": np.ones((2, 14, 11))}
    packet = orch.DriverPreviousStepStatePacket(tstep=17519,
        fields_by_component={"sechiba_finalize_state": source, "hydrol_previous_step_state": mirror},
        provenance_by_component={}, source_clock_runtime=previous_runtime)
    values = dict(config_path=CONFIG, run_def_path=run_def, run_def_values=parse_run_def(run_def),
        first_step_bundle=object(), reference_run_dir=None, source_clock_setup=setup, run_scalars=scalars,
        runtime=SimpleNamespace(dt_stomate=86400.0, dt_sechiba=1800.0), dt_sechiba=1800.0)
    context = make_dataclass("TinyContext", [(name, object) for name in values])(**values)
    monkeypatch.setattr(orch, "source_clock_setup_from_config", lambda *a, **kw: setup)
    monkeypatch.setattr(orch, "driver_year_handoff_state_gaps", lambda state: ())
    monkeypatch.setattr(orch, "_require_source_day_alignment", lambda *a, **kw: None)
    monkeypatch.setattr(orch, "_paper_1961_step_bundle_from_context", lambda *a, **kw: object())
    monkeypatch.setattr(orch, "build_intersurf_first_step_payload", lambda *a, **kw: object())
    monkeypatch.setattr(orch, "restart_thermosoil_initialization_supported", lambda context: False)
    expected = initialized if mode == "new_native_run" else us
    actual = hydrol.hydrol_init_us_state
    transitions = []

    def transition(**kwargs):
        transitions.append(kwargs["run_def_values"])
        return actual(**kwargs)

    monkeypatch.setattr(hydrol, "hydrol_init_us_state", transition)

    def runtime(*args, **kwargs):
        state = kwargs["previous_state"]
        namespace, _ = orch._hydrol_state_namespace_from_previous_state(state)
        np.testing.assert_array_equal(namespace.us, expected)
        payload = _consumer(namespace, None)
        np.testing.assert_array_equal(payload["us"], expected)
        np.testing.assert_array_equal(payload["humrelv"], _sum_source(expected))
        return state

    monkeypatch.setattr(orch, "paper_1961_driver_later_day_runtime_result", runtime)
    result = orch.paper_1961_driver_restart_year_start_day_result(CONFIG,
        previous_year_end_state=packet, prepared_context=context, year=1962, run_mode=mode)
    np.testing.assert_array_equal(result.fields_by_component["sechiba_finalize_state"]["us"], expected)
    expected_stress = (_stress_source(expected, scalars.pref_soil_veg, source["vegstress"])
                       if mode == "new_native_run" else source["vegstress"])
    np.testing.assert_array_equal(result.fields_by_component["sechiba_finalize_state"]["vegstress"], expected_stress)
    assert len(transitions) == (1 if mode == "new_native_run" else 0)
    np.testing.assert_array_equal(source["us"], us)
    np.testing.assert_array_equal(mirror["us"], us)
    np.testing.assert_array_equal(mirror["humrelv"], _sum_source(us))
