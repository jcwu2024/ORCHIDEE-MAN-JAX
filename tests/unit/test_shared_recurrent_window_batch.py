"""Lightweight native-AD and failure-boundary tests for two-window execution."""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import shared_process_head
from scripts.hpc import probe_shared_recurrent_window_batch as probe


def native_objective(parameters, initial, discrete, days, targets, context):
    def step(state, day):
        forcing, mask = day
        next_state = jnp.where(mask, jnp.tanh(parameters * state + forcing), state)
        return next_state, next_state

    _, predictions = jax.lax.scan(jax.checkpoint(step), initial, days)
    loss = jnp.mean((predictions - targets) ** 2) + context * jnp.sum(initial**2)
    return loss, {"flux": loss, "hard": {"physical": jnp.sum(discrete)},
                  "declared_mask_change": jnp.asarray(0, dtype=jnp.int32)}


def cases():
    return (
        (jnp.array([0.2, 0.4]), jnp.array([0, 0]),
         (jnp.arange(14, dtype=jnp.float32).reshape(7, 2) / 10,
          jnp.array([[True, False], [False, True], [True, True], [False, False],
                     [True, False], [False, True], [True, True]])),
         jnp.full((7, 2), 0.3), jnp.asarray(0.01)),
        (jnp.array([-0.1, 0.7]), jnp.array([0, 0]),
         (jnp.arange(14, dtype=jnp.float32).reshape(7, 2) / -13,
          jnp.array([[False, True], [True, True], [False, False], [True, False],
                     [True, True], [True, False], [False, True]])),
         jnp.full((7, 2), -0.2), jnp.asarray(0.03)),
    )


def test_native_scan_parameter_and_initial_state_gradients_match_two_distinct_masked_windows():
    examples = cases()
    parameters = jnp.array([0.8, 0.9])
    stacked = probe.stack_cases(jax, examples)
    batched = probe.make_batched_objective(jax, jnp, native_objective)
    candidate = jax.jit(jax.value_and_grad(batched, has_aux=True))(parameters, stacked)
    backward = jax.value_and_grad(native_objective, has_aux=True)
    reference = probe.aggregate_results(jax, jnp, *(backward(parameters, *case) for case in examples))
    for actual, expected in zip(jax.tree_util.tree_leaves(candidate), jax.tree_util.tree_leaves(reference), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    # Differentiate the unchanged initial-state boundary, not Teacher-forced daily states.
    def batch_initial(initials):
        return batched(parameters, (initials, *stacked[1:]))[0]

    actual = jax.grad(batch_initial)(stacked[0])
    expected = jnp.stack([jax.grad(lambda initial: native_objective(parameters, initial, *case[1:])[0])(case[0]) / 2
                          for case in examples])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_aggregation_sums_counts_and_means_losses_and_gradients():
    def result(value):
        return ((jnp.asarray(value, dtype=jnp.float32),
                 {"flux": jnp.asarray(value, dtype=jnp.float32),
                  "hard": {"physical": jnp.asarray(value, dtype=jnp.int32)},
                  "declared_mask_change": jnp.asarray(value, dtype=jnp.int32)}),
                {"weight": jnp.array([value], dtype=jnp.float32)})

    aggregate = probe.aggregate_results(jax, jnp, result(2), result(4))
    assert float(aggregate[0][0]) == 3
    assert int(aggregate[0][1]["hard"]["physical"]) == 6
    assert int(aggregate[0][1]["declared_mask_change"]) == 6
    np.testing.assert_array_equal(aggregate[1]["weight"], [3])


def test_legal_declared_mask_changes_are_summed_and_compared_not_a_hard_zero_gate():
    helper, daily = probe.load_helpers()
    result = ((jnp.asarray(1.0), {"flux": jnp.asarray(1.0), "hard": {"physical": jnp.asarray(0)},
                               "declared_mask_change": jnp.asarray(3)}), {"weight": jnp.asarray(2.0)})
    aggregate = probe.aggregate_results(jax, jnp, result, result)
    assert int(aggregate[0][1]["declared_mask_change"]) == 6
    assert not any(probe.physical_counts(daily, aggregate).values())
    comparisons, hard, passed = helper.compare_candidate(daily, jax, aggregate, aggregate)
    assert passed and comparisons["auxiliary"]["passed"] and hard == {"physical": 0}


@pytest.mark.parametrize("change", ["shape", "dtype", "tree"])
def test_stack_rejects_case_geometry_drift(change):
    first, second = cases()
    second = list(second)
    if change == "shape":
        second[0] = jnp.zeros(3)
    elif change == "dtype":
        second[0] = jnp.zeros(2, dtype=jnp.int32)
    else:
        second[1] = {"different": second[1]}
    with pytest.raises(ValueError, match="differ"):
        probe.stack_cases(jax, (first, tuple(second)))


def test_compile_requires_saved_reference_and_restores_both_patches_after_failure(tmp_path):
    _, daily = probe.load_helpers()
    original_raw = shared_process_head.shared_transformer_raw_outputs
    original_context = shared_process_head.assemble_shared_transformer_context
    adapter, backbone = object(), object()
    path = tmp_path / "reference.pkl"
    calls = []

    class Backward:
        def lower(self, *arguments):
            calls.append(arguments)
            assert shared_process_head.shared_transformer_raw_outputs is adapter
            assert shared_process_head.assemble_shared_transformer_context is backbone
            return self

        def compile(self):
            assert shared_process_head.shared_transformer_raw_outputs is adapter
            assert shared_process_head.assemble_shared_transformer_context is backbone
            raise RuntimeError("backend failure")

    def patches():
        return daily.candidate_process_head_patches(shared_process_head, adapter, backbone)

    with pytest.raises(FileNotFoundError, match="saved before"):
        probe.compile_candidate(Backward(), "parameters", "stacked", path, patches(), lambda *a, **k: None)
    assert not calls
    path.touch()
    with pytest.raises(RuntimeError, match="backend failure"):
        probe.compile_candidate(Backward(), "parameters", "stacked", path, patches(), lambda *a, **k: None)
    assert calls == [("parameters", "stacked")]
    assert shared_process_head.shared_transformer_raw_outputs is original_raw
    assert shared_process_head.assemble_shared_transformer_context is original_context


def test_benchmark_preserves_first_result_before_warm_timings():
    calls = []
    preserved = []

    def call():
        if calls:
            assert preserved
        calls.append(1)
        return jnp.asarray(1.0)

    _, seconds = probe.benchmark(call, jax, 2, preserved.append)
    assert len(calls) == 3 and len(seconds) == 2 and len(preserved) == 1


def test_minimal_parser_has_no_teacher_preparation_arguments():
    names = ("source-profile", "input-bundle", "prepared-context", "combined-report",
             "candidate-module-file", "backbone-module-file", "output")
    args = probe.parser().parse_args([part for name in names for part in (f"--{name}", f"{name}.pkl")]
                                    + ["--inspect-only", "--reference-result", "recovered.pkl"])
    assert args.inspect_only and args.reference_result == Path("recovered.pkl")
    assert not hasattr(args, "dataset_manifest")


def test_abstract_forward_uses_checkpoint_shapes_and_scoped_candidate_patches(monkeypatch):
    from scripts.hpc import run_shared_learning

    _, daily = probe.load_helpers()
    adapter, backbone = object(), object()
    original_raw = shared_process_head.shared_transformer_raw_outputs
    original_context = shared_process_head.assemble_shared_transformer_context
    calls = []

    def loader(path, *, expected_method):
        assert path == Path("update0.pkl") and expected_method == {"declared": True}
        return {"completed_updates": 0, "parameters": jnp.array([0.8, 0.9])}

    def objective(parameters, *case):
        assert shared_process_head.shared_transformer_raw_outputs is adapter
        assert shared_process_head.assemble_shared_transformer_context is backbone
        calls.append(1)
        return native_objective(parameters, *case)

    monkeypatch.setattr(run_shared_learning, "load_learning_checkpoint", loader)
    result = probe.inspect_abstract_forward(jax, daily, {
        "checkpoint": Path("update0.pkl"), "reference": {"method": {"declared": True}},
        "stacked": probe.stack_cases(jax, cases()), "objective": objective,
        "adapter": adapter, "backbone": backbone,
    })
    assert calls and result["abstract_forward_validated"] and result["abstract_loss_shape"] == []
    assert not result["gpu_gradient_compiled"]
    assert shared_process_head.shared_transformer_raw_outputs is original_raw
    assert shared_process_head.assemble_shared_transformer_context is original_context
