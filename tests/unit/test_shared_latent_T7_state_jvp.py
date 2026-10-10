"""CPU contracts for the bounded T7 state-JVP diagnostic."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc import probe_shared_latent_T7_state_jvp as probe
from scripts.hpc import run_shared_latent_T7_state_jvp as runner
from scripts.hpc import shared_latent_T7_batch_ad as batch_ad
from scripts.hpc import shared_latent_T7_state_jvp as contract


def _objective(parameters, state, discrete, forcing, target, context):
    del discrete, context
    residual = parameters["scale"] * state + forcing - target
    return jnp.sum(residual * residual), {
        "residual": residual, "hard": {"mask_mismatch": jnp.asarray(0, dtype=jnp.int32)}}


def test_direction_layout_and_jvp_routes_match_normalized_saved_vjp_projections():
    jax.config.update("jax_enable_x64", True)
    params = {"scale": jnp.asarray(0.7, dtype=jnp.float32)}
    cases = tuple((jnp.full((3854,), 0.2 + index * 0.1, jnp.float64),
                   jnp.asarray(index, jnp.int32),
                   jnp.full((3854,), 0.03 * index, jnp.float64),
                   jnp.full((3854,), 0.11, jnp.float64),
                   {"window": jnp.asarray(index, jnp.int32)}) for index in range(4))
    stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *cases)
    directions = contract.direction_matrix()
    assert directions.shape == (4, 4, 3854)
    assert directions.dtype == np.float64
    for i, coordinate in enumerate(contract.STATE_COORDINATES):
        assert np.count_nonzero(directions[i]) == 1
        assert directions[i, 3, coordinate] == 1
        assert np.count_nonzero(directions[i, :3]) == 0

    b1_fn, b4_fn = probe.route_functions(jax, _objective)
    b1_results = [b1_fn(params, *cases[3], jnp.asarray(row)) for row in directions]
    b4_results = [b4_fn(params, *stacked, jnp.asarray(row)) for row in directions]
    b1_jvps = np.asarray([result[1] for result in b1_results])
    b4_jvps = np.asarray([result[1] for result in b4_results])
    b1_vjp = jax.grad(lambda state: _objective(params, state, *cases[3][1:])[0])(cases[3][0])
    mean = batch_ad.make_batch_objective(jax, _objective)
    b4_vjp = jax.grad(lambda states: mean(params, states, *stacked[1:])[0])(stacked[0])
    np.testing.assert_allclose(b1_jvps, contract.project_b1_state_vjp(b1_vjp), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(b4_jvps, contract.project_b4_state_vjp(b4_vjp), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(b1_jvps, b4_jvps, rtol=1e-6, atol=1e-7)
    b1_bridge = probe.compare_primal_bridge(jax, "B1", b1_results[0][0], _objective(params, *cases[3]))
    saved_b4 = batch_ad.make_batch_objective(jax, _objective)(params, *stacked)
    b4_bridge = probe.compare_primal_bridge(jax, "B4", b4_results[0][0], saved_b4)
    assert b1_bridge["passed"] and b4_bridge["passed"]
    assert b1_bridge["hard_physical_mask_gates"]["mask_mismatch"]["zero"]


def test_b1_completion_jvp_preserves_raw_and_mean_scaled_tangent():
    jax.config.update("jax_enable_x64", True)
    params = {"scale": jnp.asarray(0.7, dtype=jnp.float32)}
    state = jnp.arange(3854, dtype=jnp.float64)
    args = (jnp.asarray(0, jnp.int32), jnp.ones(3854, jnp.float64),
            jnp.zeros(3854, jnp.float64), {"window": jnp.asarray(0, jnp.int32)})
    direction = contract.direction_matrix()[0]
    b1_fn, _ = probe.route_functions(jax, _objective, retain_unscaled_b1=True)
    (primal, auxiliary), (raw_tangent, scaled_tangent) = b1_fn(params, state, *args, direction)
    assert primal.shape == () and auxiliary["hard"]["mask_mismatch"] == 0
    np.testing.assert_array_equal(np.asarray(scaled_tangent) * 4, np.asarray(raw_tangent))


def test_saved_jvp_abi_helper_handles_nested_argument_pytrees():
    from types import SimpleNamespace

    arguments = ({"state": jnp.ones((2,), dtype=jnp.float64)},
                 (jnp.asarray(1, dtype=jnp.int32), jnp.ones((), dtype=jnp.float32)))
    supplied = (tuple(arguments), {})
    args_info = jax.tree_util.tree_map(lambda value: SimpleNamespace(_aval=SimpleNamespace(
        shape=value.shape, dtype=value.dtype)), supplied)
    executable = SimpleNamespace(args_info=args_info)
    serialized = (b"executable", jax.tree_util.tree_structure(supplied), b"metadata")
    receipt = probe.validate_serialized_call_abi(jax, serialized, executable, arguments, "B4 JVP")
    assert receipt["argument_count"] == 2
    assert receipt["argument_leaf_count"] == 3
    assert receipt["argument_leaf_shapes"] == [[2], [], []]
    assert receipt["argument_leaf_dtypes"] == ["float64", "int32", "float32"]


def test_strict_projection_comparison_and_paid_ranges_are_separate():
    b1 = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    b4 = b1.copy()
    assert contract.compare_projections(b1, b4)["passed"]
    b4[1] += 1e-3
    failed = contract.compare_projections(b1, b4)
    assert failed["passed"] is False
    assert failed["failed_coordinates"] == [2382]

    records = [{"sample": 3, "coordinate": coordinate,
                "same_session_paid_pairs_only": {
                    "reference": {"minimum": 0.1, "maximum": 0.2},
                    "actual": {"minimum": 0.11, "maximum": 0.21}}}
               for coordinate in contract.STATE_COORDINATES[:3]]
    report = contract.paid_pair_envelope_report({"B1": b1, "B4": b4}, {
        "schema_version": "latent_T7_state_repeat_analysis_v1", "rtol": 1e-5, "atol": 1e-6,
        "coordinates": records})
    assert report["is_acceptance_gate"] is False
    assert report["routes"]["B1"]["434"]["inside_observed_range"]
    assert report["routes"]["B1"]["200"]["envelope_available"] is False


def test_projection_scaling_and_dtype_contract():
    gradient = np.zeros(3854, dtype=np.float64)
    gradient[[434, 2382, 2621, 200]] = [0.4, 0.8, 1.2, 1.6]
    np.testing.assert_array_equal(contract.project_b1_state_vjp(gradient), [0.1, 0.2, 0.3, 0.4])
    batched = np.zeros((4, 3854), dtype=np.float64)
    batched[3, [434, 2382, 2621, 200]] = [0.1, 0.2, 0.3, 0.4]
    np.testing.assert_array_equal(contract.project_b4_state_vjp(batched), [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="FP64"):
        contract.direction_matrix(dtype=np.float32)


def test_occupation_caps_and_saved_artifact_preservation(tmp_path):
    plan = contract.validate_occupation_plan(total_seconds=3600, lower_seconds=180,
        backend_seconds=900, route_seconds=1500, load_seconds=300, calls=32)
    assert plan["JVP_calls"] == 32
    for changes in ({"total_seconds": 3601}, {"backend_seconds": 901}, {"calls": 31}):
        values = dict(total_seconds=3600, lower_seconds=180, backend_seconds=900,
                      route_seconds=1500, load_seconds=300, calls=32)
        values.update(changes)
        with pytest.raises(ValueError):
            contract.validate_occupation_plan(**values)
    path = tmp_path / "saved.pkl"
    probe._save(SimpleNamespace(atomic_pickle=lambda p, v: p.write_bytes(json.dumps(v).encode())), path, {"kept": True})
    assert json.loads(path.read_text()) == {"kept": True}
    with pytest.raises(FileExistsError):
        probe._save(None, path, {})


def test_b4_only_recovery_caps_are_partial_and_fit_the_original_grant():
    plan = contract.validate_occupation_plan(total_seconds=2400, lower_seconds=180,
        backend_seconds=1500, route_seconds=1800, load_seconds=300, calls=16,
        route_mode="B4-only")
    assert plan["route_mode"] == "B4-only"
    assert plan["JVP_calls"] == 16
    assert plan["new_JVP_graph_compiles_per_route"] == 1
    with pytest.raises(ValueError, match="occupation"):
        contract.validate_occupation_plan(total_seconds=2400, lower_seconds=180,
            backend_seconds=1500, route_seconds=1800, load_seconds=300, calls=32,
            route_mode="B4-only")


def test_b1_completion_caps_reuse_saved_b4_graph_and_keep_fresh_grant():
    plan = contract.validate_occupation_plan(total_seconds=3600, lower_seconds=180,
        backend_seconds=1800, route_seconds=2400, load_seconds=300, calls=32,
        route_mode="B1-completion")
    assert plan["saved_executable_load_count"] == 3
    assert plan["new_JVP_graph_compiles"] == {"B1": 1, "B4": 0}
    with pytest.raises(ValueError, match="cap"):
        contract.validate_occupation_plan(total_seconds=3600, lower_seconds=180,
            backend_seconds=1801, route_seconds=2400, load_seconds=300, calls=32,
            route_mode="B1-completion")


def test_runner_resolves_only_isolated_overlay_and_preserves_unapproved_template():
    args = SimpleNamespace(stage="execute", output_prefix="new", runtime_preflight=runner.ROOT / "preflight.json",
                           state_input_metadata=runner.DIAG / "T7-state-inputs-20261009.json",
                           approval_report=runner.DIAG / "unapproved.json")
    command = runner.child_command(args)
    assert "probe_shared_latent_T7_state_jvp.py" in command[2]
    assert str(runner.OVERLAY) in command[2]
    assert "--approval-report" in command
    assert command[command.index("--wall-seconds") + 1] == "3060"


def test_recovery_runner_builds_b4_only_with_reused_preflight_and_reserved_cap(capsys):
    args = SimpleNamespace(stage="execute", output_prefix=runner.RECOVERY_PREFIX,
        runtime_preflight=runner.DIAG / (runner.PREFIX + ".runtime-preflight.json"),
        state_input_metadata=runner.DIAG / "T7-state-inputs-20261009.json",
        approval_report=runner.DIAG / (runner.PREFIX + ".approved.json"),
        repair_record=runner.DIAG / (runner.RECOVERY_PREFIX + ".recovery-budget.json"))
    command = runner.child_command(args)
    parsed = probe.parser().parse_args(command[3:])
    assert command[command.index("--route-mode") + 1] == "B4-only"
    assert command[command.index("--gpu-seconds-cap") + 1] == "2400"
    assert command[command.index("--backend-seconds") + 1] == "1500"
    assert command[command.index("--route-seconds") + 1] == "1800"
    assert command[command.index("--repair-record") + 1] == str(args.repair_record)
    assert str(args.runtime_preflight) in command
    assert parsed.route_mode == "B4-only"
    assert parsed.gpu_seconds_cap == 2400 and parsed.backend_seconds == 1500
    assert probe.main(command[3:] + ["--plan-only"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["route_measurement_status"]["B1_JVP"] == "unmeasured"
    assert receipt["planned_JVP_calls"] == 16
    assert receipt["planned_saved_VJP_calls"] == 4
    assert receipt["gpu_backend_started"] is False


def test_b1_completion_plan_is_false_approval_and_reuses_saved_b4_jvp(capsys, tmp_path):
    args = SimpleNamespace(stage="execute", output_prefix=runner.COMPLETION_PREFIX,
        runtime_preflight=runner.DIAG / "T7-state-jvp-20261009.runtime-preflight.json",
        state_input_metadata=runner.DIAG / "T7-state-inputs-20261009.json",
        approval_report=runner.DIAG / (runner.COMPLETION_PREFIX + ".approval.json"),
        b1_completion=True)
    command = runner.child_command(args)
    parsed = probe.parser().parse_args(command[3:] + ["--plan-only"])
    assert parsed.route_mode == "B1-completion"
    assert parsed.backend_seconds == 1800 and parsed.route_seconds == 2400
    assert parsed.gpu_seconds_cap == 3600 and parsed.wall_seconds == 3300
    assert "--b4-jvp-executable" in command
    result = probe.run(parsed)
    assert result["GPU_executed"] is False
    assert result["planned_model_compiles"] == {"B1_JVP": 1, "B4_JVP": 0}
    assert result["saved_executable_load_count"] == 3
    assert result["saved_B4_JVP_reused_without_recompile"] is True
    assert result["planned_JVP_calls"] == 32 and result["planned_saved_VJP_calls"] == 4
    assert result["optimizer_updates"] == 0
    template_path = Path("outputs/diagnostics/shared-latent-reader-20261004") / (
        runner.COMPLETION_PREFIX + ".approval.json")
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert template["approved"] is False
    template["output_report"] = str(runner.DIAG / (runner.COMPLETION_PREFIX + ".result.json"))
    approval_path = tmp_path / "false-template.json"
    approval_path.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(PermissionError, match="separately approved"):
        probe._approval(SimpleNamespace(approval_report=approval_path,
            output=runner.DIAG / (runner.COMPLETION_PREFIX + ".result.json"),
            route_mode="B1-completion"))


def test_saved_gpu_compiler_options_lower_and_compile_on_cpu():
    # These options are part of the corrected T7 saved-graph mathematical identity.
    saved_identity = {"jax_version": "0.4.38", "compiler_options": {
        "xla_gpu_enable_llvm_module_compilation_parallelism": True,
        "xla_gpu_force_compilation_parallelism": 4}}
    options = {"xla_gpu_enable_llvm_module_compilation_parallelism": True,
               "xla_gpu_force_compilation_parallelism": 4}
    assert saved_identity["jax_version"] == "0.4.38"
    assert saved_identity["compiler_options"] == options

    compiled = jax.jit(lambda value: value * 2, compiler_options=options).lower(
        jnp.ones((4,), dtype=jnp.float32)).compile()
    np.testing.assert_array_equal(compiled(jnp.ones((4,), dtype=jnp.float32)), [2, 2, 2, 2])


def test_execute_runner_imports_approval_before_preflight_without_running_it(tmp_path, monkeypatch):
    approval_path = tmp_path / "approved.json"
    output_path = runner.DIAG / (runner.PREFIX + ".result.json")
    approval_path.write_text(json.dumps({
        "schema_version": "latent_T7_state_jvp_approval_v1",
        "approved": True,
        "approval_origin": "explicit_human_user",
        "approval_note": "test-only approval fixture",
        "funding_scope": "latent_T7_state_jvp_20261009",
        "gpu_count": 1,
        "node": "gln01",
        "limit_gpu_seconds": 3600,
        "model_compile_cap_per_route": 1,
        "optimizer_update_cap": 0,
        "entry_script": "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-state-jvp-20261009/probe_shared_latent_T7_state_jvp.py",
        "working_directory": "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1",
        "output_report": str(output_path),
    }))
    original_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda path: True
        if path in (runner.CODE, runner.OVERLAY) else original_is_dir(path))
    monkeypatch.setattr(runner.os, "chdir", lambda path: None)
    monkeypatch.setattr(sys, "argv", ["runner", "--stage", "execute",
        "--approval-report", str(approval_path), "--runtime-preflight", str(tmp_path / "preflight.json")])

    class PreflightIntercepted(RuntimeError):
        pass

    def intercept_preflight(command, **kwargs):
        assert "check_daily_training_runtime.py" in command[2]
        raise PreflightIntercepted

    monkeypatch.setattr(runner.subprocess, "run", intercept_preflight)
    with pytest.raises(PreflightIntercepted):
        runner.main()


def test_probe_cli_receipt_propagates_numerical_failure_exit_code(monkeypatch, capsys):
    args = SimpleNamespace(output=Path("T7.result.json"))
    monkeypatch.setattr(probe, "parser", lambda: SimpleNamespace(parse_args=lambda argv=None: args))
    monkeypatch.setattr(probe, "run", lambda _: {
        "status": "numerical_failed", "GPU_executed": True,
        "model_compiles": {"B1_JVP": 1, "B4_JVP": 1}, "JVP_calls": 32,
        "saved_VJP_calls": 4, "optimizer_updates": 0, "training_updates": 0,
        "comparisons": np.asarray([1.0, 2.0]),
    })

    assert probe.main([]) == 3
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "numerical_failed"
    assert receipt["JVP_calls"] == 32
    assert "comparisons" not in receipt


def test_partial_b4_route_receipt_keeps_missing_b1_unmeasured(monkeypatch, capsys):
    args = SimpleNamespace(output=Path("T7-B4-recovery.result.json"))
    monkeypatch.setattr(probe, "parser", lambda: SimpleNamespace(parse_args=lambda argv=None: args))
    monkeypatch.setattr(probe, "run", lambda _: {
        "status": "partial_diagnostic_complete", "GPU_executed": True,
        "backend_compile_attempted": True, "real_model_evaluated": True,
        "route_mode": "B4-only", "route_measurement_status": {
            "B1_JVP": "unmeasured", "B4_JVP": "measured"},
        "model_compiles": {"B1_JVP": 0, "B4_JVP": 1}, "JVP_calls": 16,
        "saved_VJP_calls": 4, "optimizer_updates": 0, "training_updates": 0})

    assert probe.main([]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["route_measurement_status"]["B1_JVP"] == "unmeasured"
    assert receipt["model_compiles"] == {"B1_JVP": 0, "B4_JVP": 1}
    assert receipt["JVP_calls"] == 16 and receipt["saved_VJP_calls"] == 4


def test_b4_batch_primal_bridge_stays_with_b4_auxiliary_tree():
    batch_aux = {"prediction": jnp.arange(4, dtype=jnp.float64),
        "hard": {"mask_mismatch": jnp.asarray(0, dtype=jnp.int32)}}
    single_aux = {"prediction": jnp.arange(3854, dtype=jnp.float64),
        "hard": {"mask_mismatch": jnp.asarray(0, dtype=jnp.int32)}}
    b4_primal = (jnp.asarray(2.0, dtype=jnp.float64), batch_aux)
    b1_primal = (jnp.asarray(2.0, dtype=jnp.float64), single_aux)
    assert probe.primal_bridge_routes("B4") == ("B4",)
    assert probe.compare_primal_bridge(jax, "B4", b4_primal, b4_primal)["passed"]
    with pytest.raises(ValueError, match="shape, dtype"):
        probe.compare_primal_bridge(jax, "B4", b4_primal, b1_primal)
    with pytest.raises(ValueError, match="known JVP route"):
        probe.primal_bridge_routes("B3")


def test_recovery_approval_binds_original_output_and_immutable_approval_hash(tmp_path):
    import hashlib

    approval_path = tmp_path / "T7-state-jvp-20261009.approved.json"
    output = tmp_path / "T7-state-jvp-20261009-B4-recovery.result.json"
    repair_path = tmp_path / "T7-state-jvp-20261009-B4-recovery.recovery-budget.json"
    approval = {"schema_version": "latent_T7_state_jvp_approval_v1",
        "funding_scope": "latent_T7_state_jvp_20261009", "gpu_count": 1, "node": "gln01",
        "limit_gpu_seconds": 3600, "model_compile_cap_per_route": 1, "optimizer_update_cap": 0,
        "entry_script": "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-state-jvp-20261009/probe_shared_latent_T7_state_jvp.py",
        "working_directory": "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1",
        "output_report": str(output.parent / "T7-state-jvp-20261009.result.json"),
        "approved": True, "approval_origin": "explicit_human_user", "approval_note": "existing scope"}
    approval_path.write_text(json.dumps(approval))
    repair = {"schema_version": "latent_T7_state_jvp_recovery_budget_v1", "route_mode": "B4-only",
        "recovery_output": str(output), "original_approval_sha256": hashlib.sha256(approval_path.read_bytes()).hexdigest(),
        "approved_total_grant_gpu_seconds": 3600, "previously_charged_gpu_seconds": 1149.853161,
        "remaining_before_recovery_gpu_seconds": 2450.146839, "reserved_recovery_gpu_seconds": 2400,
        "protected_remainder_gpu_seconds": 50.146839, "lower_seconds": 180, "backend_seconds": 1500,
        "route_seconds": 1800, "saved_executable_load_seconds_each": 300,
        "reuse_runtime_preflight": True, "JVP_calls": 16, "saved_VJP_calls": 4, "optimizer_updates": 0}
    repair_path.write_text(json.dumps(repair))
    before = approval_path.read_bytes()
    probe._approval(SimpleNamespace(approval_report=approval_path, output=output,
        route_mode="B4-only", repair_record=repair_path))
    assert approval_path.read_bytes() == before
