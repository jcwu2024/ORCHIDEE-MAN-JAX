import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator import successor_training as training
from research.daily_coarse_graining.daily_operator import target_oriented_attention as attention
from scripts.hpc import audit_conditioned_attention_thermal_source as audit

jax.config.update("jax_enable_x64", True)


class Group(NamedTuple):
    other: object


class Thermal(NamedTuple):
    thermal: object


class Fast(NamedTuple):
    state: object


class Result(NamedTuple):
    next_state: object
    fast_update: object


def test_same_input_source_capture_and_state_direction(monkeypatch):
    keys = [f"{owner}_previous_step_state.{field}" for owner in ("hydrol", "diffuco")
            for field in ("snowheat", "snowtemp")]
    leaves = [NS(key=key, start=i, stop=i + 1) for i, key in enumerate(keys)]
    contract = NS(state_leaves=leaves)
    fields = [NS(name=name, constraint="signed_unit", value_representation="direct", shape=(1,))
              for name in audit.FIELDS]
    normalization = NS(mean={f.name: np.zeros(1) for f in fields}, scale={f.name: np.ones(1) for f in fields})
    head = NS(contract=contract, learned_schema=NS(fields=fields), output_normalization=normalization,
              input_assembler=NS(thermal_keys=keys[:2]))
    original = attention.normalized_output_to_physical

    def dynamic(parameters, state, retained):
        decoded = [attention.normalized_output_to_physical(
            jnp.asarray(state[i:i + 1] * parameters[i], dtype=jnp.float32), jnp.ones(1, dtype=bool),
            field=field, normalization=normalization, inputs=None,
        )[0] for i, field in enumerate(fields)]
        heat, temp = state[0] + 1e9 * decoded[0], state[1] + 100 * decoded[1]
        return Result(Group(jnp.array([heat, temp, heat, temp])), Fast(Thermal(jnp.array([temp, heat]))))

    monkeypatch.setattr(training, "rollout_input_from_template", lambda template, state, discrete, contract: state)
    bootstrap = {"context": {"runtime": {"jax": jax, "contract": contract}, "template": None},
                 "steps": {"transition_builder": lambda template: dynamic}}
    detailed, lean, tangent = audit.make_kernels(bootstrap)
    state = jnp.array([1000.0, 270.0, 1000.0, 270.0])
    parameters = jnp.array([1e-9, 1e-5])
    day = (None, {}, None)
    result, captured = detailed(parameters, state, day)
    assert attention.normalized_output_to_physical is original
    np.testing.assert_allclose(result.next_state.other, lean(parameters, state, day), rtol=1e-12)
    report = audit.source_rows(state, result, captured, head, audit.locations(contract))
    json.dumps(report, allow_nan=False)
    for row in report.values():
        assert row["decode_host_max_abs_error"] < 1e-12
        assert row["affine_host_max_abs_error"] < 1e-9
        assert row["tail_minus_fast_max_abs_error"] == 0
    direction = jnp.array([1.0, 0.1, 1.0, 0.1])
    primal, response = tangent(parameters, state, direction, day)
    secant = (lean(parameters, state + direction, day) - lean(parameters, state - direction, day)) / 2
    np.testing.assert_allclose(primal, result.next_state.other, rtol=1e-12)
    np.testing.assert_allclose(response, secant, rtol=1e-3)


def test_launcher_delivers_all_required_parser_arguments():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text()
    start = shared.index('elif [[ "$PYTHON_MODULE" == "scripts.hpc.run_target_oriented_coupled_gpp_conditioned_attention_persistent_training"')
    branch = shared[start:shared.index("elif [[", start + 8)]
    assert '"scripts.hpc.audit_conditioned_attention_thermal_source" ||' in branch
    command = shared[shared.index('"$PYTHON" -m "$PYTHON_MODULE"'):]
    assert all(o in branch or o in command for a in audit.window._parser()._actions if a.required for o in a.option_strings)
    wrapper = (root / "scripts/hpc/audit_conditioned_attention_thermal_source.sh").read_text()
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-30m}" in wrapper
    assert "sbatch" not in wrapper


def test_runner_does_not_precreate_initializer_output(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "report.json").write_text('{"identity": {}}')
    contract = tmp_path / "contract.json"
    contract.write_text('{"contract_sha256": "fixture"}')
    args = NS(conditioned_run_dir=parent, output=tmp_path / "result.json",
              parent_persistent_training_report=None, parent_artifact_aware_recovery=None,
              artifact_aware_persistent_contract=contract, conditioning_contract=None)
    monkeypatch.setattr(audit.window, "load_recovery", lambda *a, **kw: {"completed_logical_update": 3, "trainable": {}})
    monkeypatch.setattr(audit.window, "_load_parent_state", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(audit.window, "load_coupled_gpp_conditioned_attention_contract", lambda *a: None)

    def bootstrap(*a, **kw):
        assert not args.output.exists()
        return {"context": {"runtime": {"jax": jax}}}

    monkeypatch.setattr(audit.window, "_conditioned_bootstrap", bootstrap)
    monkeypatch.setattr(audit.window, "_load_artifact", lambda *a: None)
    monkeypatch.setattr(audit, "experiment", lambda *a, **kw: {"windows": []})
    assert audit.run(args)["status"] == "completed"
    assert json.loads(args.output.read_text())["status"] == "completed"
