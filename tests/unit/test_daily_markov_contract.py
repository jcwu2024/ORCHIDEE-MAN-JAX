from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.driver.bundle import ccanopy_from_co2
from jax_orchidee.driver.orchestration import (
    _paper_compiled_forcing_day,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference
from research.daily_coarse_graining import daily_markov_contract as markov
from research.daily_coarse_graining import teacher_shards


def _packet(value: float, *, flag: bool = True, include_nroot: bool = False):
    lai = np.zeros((1, 14), dtype=np.float64)
    lai[:, 13] = value
    hydrol = {
        "mc": np.asarray([[[value]]]),
        "njsc": np.asarray([1], dtype=np.int32),
    }
    if include_nroot:
        hydrol["nroot"] = np.zeros((1, 14, 2), dtype=np.float64)
        hydrol["nroot"][:, 13, :] = np.asarray([0.3, 0.7])
    biomass = np.zeros((1, 14, 1, 1), dtype=np.float64)
    biomass[:, 13, :, :] = value
    pft_present = np.zeros((1, 14), dtype=bool)
    pft_present[:, 13] = flag
    return SimpleNamespace(
        fields_by_component={
            "driver_previous_step_state": {
                "albedo": np.asarray([[value, value + 0.5]])
            },
            "diffuco_previous_step_state": {"lai": lai.copy()},
            "enerbil_previous_step_state": {},
            "hydrol_previous_step_state": hydrol,
            "thermosoil_previous_step_state": {},
            "slowproc_stomate_previous_step_state": {
                "lai": lai.copy(),
                "biomass": biomass,
                "daily_accumulators": {"counter": np.asarray([value])},
                "pft_present": pft_present,
            },
            "sechiba_finalize_state": {
                "mc": np.asarray([[[value]]]),
                "diagnostic_only": np.asarray([1000.0 + value]),
            },
        }
    )


def _record(end_state, *, day_index: int = 1):
    final = {"t2mdiag": np.asarray([290.0]), "temp_sol": np.asarray([291.0])}
    gpp_daily = np.zeros((1, 14), dtype=np.float64)
    gpp_daily[:, 13] = 2.0
    slow_axes = markov._slow_axis_lookup()

    def slow_value(name):
        return np.zeros(
            tuple(14 if axis == "nvm" else 1 for axis in slow_axes.get(name, ("npts",))),
            dtype=np.float64,
        )

    ok_updates = {
        name: slow_value(name)
        for name in markov.FAST_DAY_OK_LEAK_FIELDS
        if name != "deepC_peat"
    }
    return SimpleNamespace(
        year=1961,
        day_index=day_index,
        daily_fold=SimpleNamespace(
            daily_fields={
                "gpp_daily": gpp_daily,
                "precip_daily": np.asarray([3.0]),
            }
        ),
        half_hour_transition=SimpleNamespace(
            current_state=end_state,
            completed_entry_payloads=[final],
        ),
        expected_result=SimpleNamespace(day_end_state=end_state),
        ok_leak_updates=ok_updates,
        ok_leak_result=SimpleNamespace(
            soilcarbon=SimpleNamespace(
                perma_peat=SimpleNamespace(
                    deepc_peat=slow_value("deepC_peat")
                )
            )
        ),
    )


def _native_spec():
    return markov.NativeForcingSpec(
        fields=markov.NATIVE_FORCING_FIELDS,
        field_shape=(1, 1),
        source_records_per_day=5,
        source_interval_seconds=21600.0,
        model_interval_seconds=1800.0,
        split=12,
        precipitation_spread_steps=6,
    )


def _condition_leaves(name: str, width: int, temporal_role: str):
    if width == 0:
        return ()
    return (
        markov.ConditionLeafSpec(
            name=name,
            shape=(width,),
            dtype="float64",
            start=0,
            stop=width,
            temporal_role=temporal_role,
            source="test",
        ),
    )


def test_contract_uses_canonical_state_and_excludes_packet_mirrors():
    start = _packet(1.0)
    end = _packet(2.0, flag=False, include_nroot=True)
    end.fields_by_component["slowproc_stomate_previous_step_state"]["biomass"][
        :, 5, :, :
    ] = 123.0
    record = _record(end)
    contract = markov.build_daily_markov_contract(
        start,
        record,
        parameter_leaves=_condition_leaves("parameter", 4, "landpoint_parameter"),
        landpoint_static_leaves=_condition_leaves("static", 7, "landpoint_static"),
        annual_condition_leaves=_condition_leaves("co2", 1, "annual_exogenous"),
        native_forcing_spec=_native_spec(),
    )
    keys = {leaf.key for leaf in contract.state_leaves}
    assert "slowproc_stomate_previous_step_state.lai" in keys
    assert "diffuco_previous_step_state.lai" not in keys
    assert not any(key.startswith("sechiba_finalize_state.") for key in keys)
    assert "hydrol_previous_step_state.njsc" in keys
    assert "hydrol_previous_step_state.nroot" in keys
    assert contract.active_pft_indices == (0, 13)
    assert contract.sha256 == contract.sha256

    markov.assert_markov_continuity([start], [record], end, contract)
    trajectory, discrete = markov.build_state_trajectory([start, end], contract)
    assert trajectory.shape == (2, contract.continuous_state_width)
    np.testing.assert_array_equal(
        discrete["slowproc_stomate_previous_step_state.pft_present"][:, 0, 1],
        np.asarray([True, False]),
    )
    reconstructed_fields = markov.reconstruct_state_fields(
        trajectory[1],
        {name: value[1] for name, value in discrete.items()},
        contract,
        template_fields=end.fields_by_component,
    )
    reconstructed = SimpleNamespace(fields_by_component=reconstructed_fields)
    roundtrip, roundtrip_discrete = markov.extract_state(reconstructed, contract)
    np.testing.assert_array_equal(roundtrip, trajectory[1])
    for name, value in discrete.items():
        np.testing.assert_array_equal(roundtrip_discrete[name], value[1])
    np.testing.assert_array_equal(
        reconstructed_fields["diffuco_previous_step_state"]["lai"],
        reconstructed_fields["slowproc_stomate_previous_step_state"]["lai"],
    )
    assert not np.any(
        reconstructed_fields["slowproc_stomate_previous_step_state"]
        ["daily_accumulators"]["counter"]
    )
    assert reconstructed_fields["slowproc_stomate_previous_step_state"][
        "biomass"
    ][0, 5, 0, 0] == 123.0


def test_markov_continuity_rejects_a_different_next_day_state():
    start = _packet(1.0)
    expected = _packet(2.0, include_nroot=True)
    observed = _packet(3.0, include_nroot=True)
    record = _record(expected)
    contract = markov.build_daily_markov_contract(
        start,
        record,
        parameter_leaves=(),
        landpoint_static_leaves=(),
        annual_condition_leaves=(),
        native_forcing_spec=_native_spec(),
    )
    with np.testing.assert_raises_regex(ValueError, "continuity failed"):
        markov.assert_markov_continuity([start], [record], observed, contract)


def test_native_forcing_window_reconstructs_teacher_units_and_spreading(monkeypatch):
    spec = _native_spec()
    raw_by_field = {
        name: (np.arange(5, dtype=np.float64) + 10.0 * index).reshape(5, 1)
        for index, name in enumerate(spec.fields)
    }
    window = np.concatenate([raw_by_field[name] for name in spec.fields], axis=1)
    domain = SimpleNamespace(nbindex=1)
    first = SimpleNamespace(
        domain=domain,
        forcing=SimpleNamespace(Height_Lev1=2.0, Height_Levuv=10.0),
        static_trace_fields=SimpleNamespace(
            salinity=np.asarray([0.25]), tide_height=np.asarray([1.5])
        ),
    )
    context = SimpleNamespace(first_step_bundle=first, config_path=Path("unused"))
    monkeypatch.setattr(markov, "read_annual_co2", lambda *_args: 400.0)
    monkeypatch.setattr(markov, "_shortwave_factors", lambda *_args, **_kwargs: np.ones((48, 1)))
    result = markov.reconstruct_compiled_forcing_day(
        window, spec, context, year=1961, day_index=1
    )

    first_weight = 1.0 / 12.0
    expected_tair = raw_by_field["Tair"][0] + (
        raw_by_field["Tair"][1] - raw_by_field["Tair"][0]
    ) * first_weight
    np.testing.assert_array_equal(np.asarray(result.temp_air[0]), expected_tair)
    np.testing.assert_array_equal(
        np.asarray(result.precip_rain[0]), raw_by_field["Rainf"][1] * 2.0 * 1800.0
    )
    np.testing.assert_array_equal(np.asarray(result.precip_rain[6]), np.asarray([0.0]))
    np.testing.assert_array_equal(
        np.asarray(result.u[0]),
        (raw_by_field["Wind_N"][0] + first_weight).reshape(spec.field_shape),
    )
    np.testing.assert_array_equal(
        np.asarray(result.v[0]),
        (raw_by_field["Wind_E"][0] + first_weight).reshape(spec.field_shape),
    )
    np.testing.assert_array_equal(
        np.asarray(result.ccanopy[0]), ccanopy_from_co2(400.0, 1)
    )

    monkeypatch.setattr(
        markov,
        "read_annual_co2",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected file lookup")),
    )
    explicit = markov.reconstruct_compiled_forcing_day(
        window,
        spec,
        context,
        year=2099,
        day_index=1,
        annual_co2_ppm=412.5,
    )
    np.testing.assert_array_equal(
        np.asarray(explicit.ccanopy[0]), ccanopy_from_co2(412.5, 1)
    )


def test_native_record_indices_preserve_the_fortran_first_day_boundary_rule():
    np.testing.assert_array_equal(
        markov.forcing_record_indices_for_day(day_index=1, raw_steps=1460, split=12),
        np.asarray([3, 0, 1, 2, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        markov.forcing_record_indices_for_day(day_index=2, raw_steps=1460, split=12),
        np.asarray([3, 4, 5, 6, 7], dtype=np.int32),
    )


def test_v2_shard_reader_exposes_state_to_next_state_without_stored_masks(tmp_path):
    days = 3
    path = tmp_path / "shard.npz"
    np.savez(
        path,
        state_trajectory=np.arange((days + 1) * 4, dtype=np.float64).reshape(days + 1, 4),
        fast_day_target=np.ones((days, 6), dtype=np.float64),
        forcing_native=np.ones((days, 5, 9), dtype=np.float64),
        forcing_record_indices=np.arange(days * 5, dtype=np.int32).reshape(days, 5),
        parameters=np.ones(2),
        landpoint_static=np.ones(3),
        annual_conditions=np.asarray([317.27]),
        diagnostics=np.ones((days, 2)),
        year=np.asarray(1961, dtype=np.int32),
        day_index=np.arange(1, days + 1, dtype=np.int32),
        state_discrete__flag=np.asarray([[True], [False], [True], [True]]),
    )
    shard = markov.load_markov_shard(path)
    sample = shard.sample(1)
    np.testing.assert_array_equal(sample["state"], shard.state_trajectory[1])
    np.testing.assert_array_equal(sample["next_state"], shard.state_trajectory[2])
    assert sample["discrete_state"]["flag"].item() is False
    assert sample["annual_conditions"].item() == 317.27
    assert sample["fast_day_target"].shape == (6,)
    assert sample["year"] == 1961
    assert sample["state_finite"].all()
    assert not any("finite" in name for name in np.load(path).files)


def test_compiled_day_argument_ownership_manifest_covers_the_real_signature():
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "jax_orchidee/driver/orchestration.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)
    outer = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_paper_compiled_later_day_block_executable"
    )
    transition = next(
        node
        for node in outer.body
        if isinstance(node, ast.FunctionDef) and node.name == "transition"
    )
    actual = tuple(argument.arg for argument in transition.args.args)
    manifest = json.loads(
        (
            root
            / "manifests/coarse_graining/daily_markov_input_ownership_v2.json"
        ).read_text(encoding="utf-8")
    )
    assert tuple(manifest["compiled_transition_arguments"]) == actual
    assert all(
        item["owner"] and item["representation"]
        for item in manifest["compiled_transition_arguments"].values()
    )


def test_teacher_shard_builder_emits_only_the_v2_markov_arrays(monkeypatch):
    start = _packet(1.0)
    end = _packet(2.0, include_nroot=True)
    record = _record(end)
    monkeypatch.setattr(
        teacher_shards,
        "_parameter_groups",
        lambda _context: (("parameter", np.asarray([1.0, 2.0])),),
    )
    monkeypatch.setattr(
        teacher_shards,
        "_landpoint_static_groups",
        lambda _context: (("static", np.asarray([3.0])),),
    )
    monkeypatch.setattr(
        teacher_shards,
        "_annual_condition_groups",
        lambda _context, **_kwargs: (("annual_co2_ppm", np.asarray([317.27])),),
    )
    monkeypatch.setattr(
        teacher_shards,
        "native_forcing_days",
        lambda _context, **_kwargs: (
            np.ones((1, 5, 9), dtype=np.float64),
            np.asarray([[3, 0, 1, 2, 3]], dtype=np.int32),
            _native_spec(),
        ),
    )
    arrays, contract = teacher_shards.build_shard_arrays(
        [start], [object()], [record], object(), final_state=end
    )
    streamed, streamed_contract, streamed_final = (
        teacher_shards.build_shard_arrays_from_blocks(
            [([start], (object(),), (record,), end)], object()
        )
    )
    start_continuous, start_discrete = markov.extract_state(
        start, contract, allow_year_start_missing=True
    )
    compact_block = SimpleNamespace(
        year=1961,
        day_indices=np.asarray([1], dtype=np.int32),
        state_rows=start_continuous[None, :],
        discrete_rows={
            name: value[None, ...] for name, value in start_discrete.items()
        },
        fast_day_targets=arrays["fast_day_target"],
        diagnostics=arrays["diagnostics"],
        final_state=end,
        contract=contract,
    )
    compact, compact_contract, compact_final = (
        teacher_shards.build_shard_arrays_from_compact_blocks(
            [compact_block], object()
        )
    )
    assert contract.schema_version == markov.CONTRACT_SCHEMA_VERSION
    assert streamed_contract.metadata() == contract.metadata()
    assert compact_contract.metadata() == contract.metadata()
    assert streamed_final is end
    assert compact_final is end
    assert streamed.keys() == arrays.keys()
    for name in arrays:
        np.testing.assert_array_equal(streamed[name], arrays[name])
        np.testing.assert_array_equal(compact[name], arrays[name])
    assert arrays["state_trajectory"].shape[0] == 2
    assert arrays["fast_day_target"].shape == (1, contract.fast_day_target_width)
    assert arrays["forcing_native"].shape == (1, 5, 9)
    assert arrays["annual_conditions"].shape == (1,)
    np.testing.assert_array_equal(arrays["year"], np.asarray(1961, dtype=np.int32))
    assert arrays["diagnostics"].shape[0] == 1
    forbidden = {
        "day_start_state",
        "forcing_48",
        "teacher_target",
        "day_start_state_finite",
        "forcing_48_finite",
        "teacher_target_finite",
    }
    assert forbidden.isdisjoint(arrays)


def test_v3_schema_has_a_material_size_reduction_against_the_v1_year():
    days = 365
    arrays = {
        "state_trajectory": np.empty((days + 1, 3724), dtype=np.float64),
        "fast_day_target": np.empty((days, 4000), dtype=np.float64),
        "forcing_native": np.empty((days, 5, 9), dtype=np.float64),
        "annual_conditions": np.empty(1, dtype=np.float64),
        "diagnostics": np.empty((days, 90), dtype=np.float64),
        "year": np.asarray(1961, dtype=np.int32),
        "day_index": np.empty(days, dtype=np.int32),
    }
    assert markov.estimated_uncompressed_bytes(arrays) < 0.11 * 224_333_416


@pytest.mark.external_data
def test_real_landpoints_share_the_named_parameter_and_static_schema():
    root = Path(__file__).resolve().parents[2]
    schemas = []
    widths = []
    for landpoint_id in ("001.0-071.0", "003.0-077.0"):
        reference = resolve_paper_landpoint_reference(root, landpoint_id)
        if reference.used_run_def is None or reference.output_dir is None:
            pytest.skip(f"reference assets are unavailable for {landpoint_id}")
        context = prepare_paper_1961_driver_context(
            root / "configs/orchidee_man_250919.yaml",
            used_run_def_path=reference.used_run_def,
            reference_run_dir=reference.output_dir,
        )
        parameters, parameter_leaves = teacher_shards._pack_condition_groups(
            teacher_shards._parameter_groups(context),
            temporal_role="landpoint_parameter",
            source="test",
        )
        static, static_leaves = teacher_shards._pack_condition_groups(
            teacher_shards._landpoint_static_groups(context),
            temporal_role="landpoint_static",
            source="test",
        )
        schemas.append(
            tuple(
                (leaf.name, leaf.shape, leaf.start, leaf.stop)
                for leaf in (*parameter_leaves, *static_leaves)
            )
        )
        widths.append((parameters.size, static.size))
    assert schemas[0] == schemas[1]
    assert widths == [(84, 735), (84, 735)]


@pytest.mark.external_data
def test_real_paper_forcing_is_reconstructed_from_native_records():
    root = Path(__file__).resolve().parents[2]
    config = root / "configs/orchidee_man_250919.yaml"
    used_run_def = root / "outputs/server_1961_trace_full_20260623/run/used_run.def"
    if not used_run_def.exists():
        pytest.skip("local paper forcing inputs are unavailable")
    context = prepare_paper_1961_driver_context(
        config, used_run_def_path=used_run_def
    )
    days = (1, 2, 365)
    native, _indices, spec = markov.native_forcing_days(
        context, year=1961, day_indices=days
    )
    for offset, day in enumerate(days):
        reconstructed = markov.reconstruct_compiled_forcing_day(
            native[offset], spec, context, year=1961, day_index=day
        )
        reference = _paper_compiled_forcing_day(
            context,
            year=1961,
            start_tstep=(day - 1) * 48,
            steps_per_stomate=48,
        )
        for name in reconstructed._fields:
            np.testing.assert_allclose(
                np.asarray(getattr(reconstructed, name)),
                np.asarray(getattr(reference, name)),
                rtol=0.0,
                atol=1.0e-12,
                err_msg=f"day={day} field={name}",
            )
