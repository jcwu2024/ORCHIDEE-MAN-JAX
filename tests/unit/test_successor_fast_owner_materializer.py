import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_markov_contract import (
    FastDayTargetLeafSpec,
    NativeForcingSpec,
    _compiled_select_pft_axes,
)
from research.daily_coarse_graining.daily_operator.fast_owner_reconstruction import (
    DEFAULT_CONTRACT_PATH,
    EXPECTED_CONTRACT_SHA256,
    ROLE_CARRY,
    ROLE_MIRROR,
    ROLE_NEURAL,
    SuccessorFastOwnerMaterializer,
    load_fast_owner_reconstruction_contract,
)
from research.daily_coarse_graining.daily_operator.process_derivations import (
    NativeForcingFieldLayout,
)
from research.daily_coarse_graining.daily_operator.types import NativeForcingInput
from scripts.hpc.run_successor_process_head_canonical_gate import (
    _all_teacher_fast_owner_outputs,
)


def _set_path(root, path, value):
    target = root
    for name in path[:-1]:
        target = target.setdefault(name, {})
    target[path[-1]] = value


def _get_path(root, path):
    value = root
    for name in path:
        value = value[name]
    return value


def _fixture():
    raw = json.loads(Path(DEFAULT_CONTRACT_PATH).read_text(encoding="utf-8"))
    specs = []
    for record in raw["leaves"]:
        shape = tuple(record["shape"])
        axes = tuple(record["axis_names"])
        full_shape = list(shape)
        selected = ()
        if "nvm" in axes:
            axis = axes.index("nvm")
            if shape[axis] == 2:
                full_shape[axis] = 14
                selected = (0, 13)
        specs.append(
            FastDayTargetLeafSpec(
                family=record["family"],
                component=record["component"],
                path=tuple(record["path"]),
                shape=shape,
                full_shape=tuple(full_shape),
                dtype="float64",
                start=record["start"],
                stop=record["stop"],
                owner=record["canonical_owner"],
                axis_names=axes,
                selected_pft_indices=selected,
            )
        )
    native = NativeForcingSpec(
        fields=(
            "Tair",
            "PSurf",
            "Qair",
            "Wind_E",
            "Wind_N",
            "Rainf",
            "Snowf",
            "SWdown",
            "LWdown",
        ),
        field_shape=(1,),
        source_records_per_day=5,
        source_interval_seconds=21600.0,
        model_interval_seconds=1800.0,
        split=12,
        precipitation_spread_steps=1,
    )
    markov = SimpleNamespace(
        sha256=raw["dataset"]["markov_contract_sha256"],
        fast_day_target_leaves=tuple(specs),
        active_pft_indices=(0, 13),
        native_forcing=native,
    )
    owner = load_fast_owner_reconstruction_contract(markov_contract=markov)
    layout = NativeForcingFieldLayout(
        rain_start=5,
        rain_stop=6,
        snow_start=6,
        snow_stop=7,
        temperature_start=0,
        temperature_stop=1,
    )
    materializer = SuccessorFastOwnerMaterializer(
        markov_contract=markov,
        owner_contract=owner,
        forcing_layout=layout,
        forcing_indices={name: index for index, name in enumerate(native.fields)},
    )

    fields = {}
    groups = {
        "daily_interface": {},
        "ok_leak": {},
        "final_diagnostics": {},
    }
    initial = {}
    by_key = {leaf.key: leaf for leaf in specs}
    for index, leaf in enumerate(specs):
        value = jnp.full(leaf.full_shape, float(index + 1), dtype=jnp.float64)
        target = fields.setdefault(leaf.component, {}) if leaf.component else groups[leaf.family]
        _set_path(target, leaf.path, value)
        initial[leaf.key] = value

    outputs = {}
    for index, leaf in enumerate(owner.leaves):
        if leaf.role != ROLE_NEURAL:
            continue
        outputs[leaf.output_key] = jnp.full(
            leaf.shape,
            float(index) + 0.25,
            dtype=jnp.float64,
        )
        if leaf.defined_output_key is not None:
            outputs[leaf.defined_output_key] = jnp.asarray(
                [[1.0, -1.0]],
                dtype=jnp.float64,
            )

    values = np.zeros((5, 9), dtype=np.float64)
    values[:, 0] = np.asarray([280.0, 282.0, 284.0, 286.0, 288.0])
    values[:, 1] = 101325.0
    values[:, 2] = 0.01
    values[:, 3] = 3.0
    values[:, 4] = 4.0
    values[:, 5] = 1.0e-5
    values[:, 6] = 2.0e-6
    forcing = NativeForcingInput(
        values=jnp.asarray(values),
        record_time_seconds=jnp.arange(5, dtype=jnp.float64) * 21600.0,
        duration_seconds=jnp.asarray([0.0, 21600.0, 21600.0, 21600.0, 21600.0]),
        predecessor_mask=jnp.asarray([True, False, False, False, False]),
        record_mask=jnp.ones((5,), dtype=jnp.bool_),
    )
    return materializer, by_key, fields, groups, initial, outputs, forcing


def _value(result, spec):
    root = (
        result.fields_by_component[spec.component]
        if spec.component
        else {
            "daily_interface": result.daily_fields,
            "ok_leak": result.ok_leak_updates,
            "final_diagnostics": result.final_diagnostics,
        }[spec.family]
    )
    return _get_path(root, spec.path)


def _pack(result, specs):
    return jnp.concatenate(tuple(_compiled_select_pft_axes(_value(result, spec), spec).reshape(-1) for spec in specs))


def test_loader_binds_the_frozen_schema_to_all_160_markov_leaves():
    materializer, by_key, *_ = _fixture()

    assert materializer.owner_contract.contract_sha256 == EXPECTED_CONTRACT_SHA256
    assert len(materializer.owner_contract.leaves) == len(by_key) == 160
    assert set(materializer.owner_contract.leaves_by_key) == set(by_key)


def test_all_teacher_gate_routes_only_same_row_process_and_fast_owner_targets():
    materializer, _, _, _, _, _, _ = _fixture()
    fast_target = np.arange(2855, dtype=np.float64)
    rveget = materializer.owner_contract.leaves_by_key["diffuco_previous_step_state.rveget"]
    fast_target[rveget.stop - 1] = 1.0e20
    example = SimpleNamespace(
        targets=SimpleNamespace(
            supervision={"daily.fast_gpp": SimpleNamespace(value=np.asarray([[3.0, 4.0]], dtype=np.float64))}
        )
    )

    outputs = _all_teacher_fast_owner_outputs(
        example,
        {"fast_day_target": fast_target},
        materializer.owner_contract,
    )

    assert set(outputs) == {
        "daily.fast_gpp",
        *(leaf.output_key for leaf in materializer.owner_contract.leaves if leaf.output_key is not None),
        rveget.defined_output_key,
    }
    np.testing.assert_array_equal(outputs["daily.fast_gpp"], [[3.0, 4.0]])
    np.testing.assert_array_equal(
        outputs[rveget.output_key],
        fast_target[rveget.start : rveget.stop].reshape(rveget.shape),
    )
    np.testing.assert_array_equal(
        outputs[rveget.defined_output_key],
        [[1.0, -1.0]],
    )


def test_materializer_closes_all_160_leaves_with_exact_mirrors_masks_and_derivations():
    materializer, by_key, fields, groups, initial, outputs, forcing = _fixture()

    result = materializer.materialize(
        fields_by_component=fields,
        daily_fields=groups["daily_interface"],
        ok_leak_updates=groups["ok_leak"],
        final_diagnostics=groups["final_diagnostics"],
        named_outputs=outputs | {"typed.water.runoff": jnp.asarray([-12345.0])},
        forcing=forcing,
        measurement_height=jnp.asarray(2.0),
    )
    packed = np.asarray(_pack(result, tuple(by_key.values())))

    assert packed.shape == (2855,)
    assert np.isfinite(packed).all()
    assert np.asarray(_value(result, by_key["daily_interface.tmc_topgrass_daily"])) == pytest.approx(0.0)
    rveget = np.asarray(_value(result, by_key["diffuco_previous_step_state.rveget"]))
    assert rveget[0, 0] != pytest.approx(1.0e20)
    assert rveget[0, 13] == pytest.approx(1.0e20)
    owner = materializer.owner_contract.leaves_by_key
    roughheight_pft = np.asarray(_value(result, by_key["diffuco_previous_step_state.roughheight_pft"]))
    assert roughheight_pft[0, 0] == pytest.approx(1.0e20)
    assert roughheight_pft[0, 13] == pytest.approx(
        np.asarray(outputs[owner["diffuco_previous_step_state.roughheight_pft"].output_key])[0, 1]
    )
    for key in (
        "daily_interface.wspeed_daily",
        "daily_interface.resp_maint_radia",
        "final_diagnostics.temp_sol",
        "diffuco_previous_step_state.control_inudate",
    ):
        leaf = owner[key]
        assert leaf.role == ROLE_NEURAL
        np.testing.assert_array_equal(
            _compiled_select_pft_axes(_value(result, by_key[key]), by_key[key]),
            outputs[leaf.output_key],
        )
    for leaf in materializer.owner_contract.leaves:
        if leaf.role == ROLE_MIRROR:
            np.testing.assert_array_equal(
                _value(result, by_key[leaf.key]),
                _value(result, by_key[leaf.canonical_owner]),
            )
        if leaf.role == ROLE_CARRY:
            np.testing.assert_array_equal(
                _value(result, by_key[leaf.key]),
                initial[leaf.key],
            )
    assert owner["hydrol_previous_step_state.runoff"].canonical_owner == ("hydrol_previous_step_state.runoff_per_soil")
    assert not np.any(packed == -12345.0)


def test_materializer_is_jittable_and_differentiable_and_rejects_missing_endpoints():
    materializer, by_key, fields, groups, _, outputs, forcing = _fixture()
    specs = tuple(by_key.values())

    def objective(scale):
        scaled = {key: value if key.endswith("defined_logit") else value * scale for key, value in outputs.items()}
        result = materializer.materialize(
            fields_by_component=fields,
            daily_fields=groups["daily_interface"],
            ok_leak_updates=groups["ok_leak"],
            final_diagnostics=groups["final_diagnostics"],
            named_outputs=scaled,
            forcing=forcing,
            measurement_height=jnp.asarray(2.0),
        )
        packed = _pack(result, specs)
        return jnp.sum(jnp.where(jnp.abs(packed) < 1.0e19, packed, 0.0))

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(gradient))
    assert float(gradient) != 0.0

    missing = dict(outputs)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing successor endpoint"):
        materializer.materialize(
            fields_by_component=fields,
            daily_fields=groups["daily_interface"],
            ok_leak_updates=groups["ok_leak"],
            final_diagnostics=groups["final_diagnostics"],
            named_outputs=missing,
            forcing=forcing,
            measurement_height=jnp.asarray(2.0),
        )
