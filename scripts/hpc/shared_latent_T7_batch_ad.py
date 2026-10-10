"""Sample batching of the unchanged public checkpointed recurrent objective.

This execution adapter adds no process physics. The source-backed canonical
step, physical masks and loss remain owned by shared_recurrent_training and
RolloutLoss.bind_checkpointed_loss_with_aux. Shared parameters are differentiated
after sample averaging, so no B4 parameter-gradient tree is produced.
"""
from __future__ import annotations

import numpy as np


def make_batch_objective(jax, objective):
    """Mean four independent full recurrences, retaining each sample's aux."""
    import jax.numpy as jnp
    mapped = jax.vmap(objective, in_axes=(None, 0, 0, 0, 0, 0))

    def mean_objective(parameters, states, discrete, days, targets, contexts):
        losses, auxiliary = mapped(parameters, states, discrete, days, targets, contexts)
        return jnp.mean(losses), auxiliary

    return mean_objective


def make_batch_value_and_grad(jax, objective):
    return jax.value_and_grad(make_batch_objective(jax, objective), argnums=(0, 1), has_aux=True)


def merge_serial_results(jax, results):
    """Device-side baseline work: sample mean param AD and state stack/B."""
    import jax.numpy as jnp
    if not results:
        raise ValueError("nonempty sample results required")
    size = len(results)
    loss = sum(row[0][0] for row in results) / size
    auxiliary = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *(row[0][1] for row in results))
    parameters = jax.tree_util.tree_map(lambda *values: sum(values) / size, *(row[1][0] for row in results))
    states = jnp.stack([row[1][1] for row in results]) / size
    return (loss, auxiliary), (parameters, states)


def validate_batch_abi(jax, serialized, executable, arguments, *, batch_size=4, state_width=3854):
    """Validate saved shapes/dtypes/trees without tracing or compiling."""
    if len(arguments) != 6 or tuple(arguments[1].shape) != (batch_size, state_width):
        raise ValueError("B4 full-state argument count/state shape differs")
    supplied = (tuple(arguments), {})
    if not isinstance(serialized, tuple) or len(serialized) != 3:
        raise ValueError("serialized JAX executable triple required")
    if serialized[1] != jax.tree_util.tree_structure(supplied):
        raise ValueError("B4 serialized input treedef differs")
    infos, tree = jax.tree_util.tree_flatten(executable.args_info)
    values, expected_tree = jax.tree_util.tree_flatten(supplied)
    if tree != expected_tree:
        raise ValueError("B4 saved args_info treedef differs")
    for info, value in zip(infos, values, strict=True):
        if tuple(info._aval.shape) != tuple(value.shape) or np.dtype(info._aval.dtype) != np.dtype(value.dtype):
            raise ValueError("B4 saved input shape/dtype differs")


def shape_memory_estimate(jax, result_shape, arguments):
    """Array sizes only; reverse tapes/workspace are explicitly unestimated."""
    def bytes_for(tree):
        return sum(int(np.prod(value.shape)) * np.dtype(value.dtype).itemsize
                   for value in jax.tree_util.tree_leaves(tree))
    (loss, auxiliary), (parameters, states) = result_shape
    return {"input_bytes": bytes_for(arguments), "output_bytes": bytes_for(result_shape),
            "parameter_gradient_bytes": bytes_for(parameters), "state_gradient_bytes": bytes_for(states),
            "auxiliary_bytes": bytes_for(auxiliary), "loss_shape": list(loss.shape),
            "state_gradient_shape": list(states.shape),
            "parameter_gradient_leaves": len(jax.tree_util.tree_leaves(parameters)),
            "parameter_gradient_coordinates": sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(parameters)),
            "reverse_tape_workspace_and_compiler_peak_estimated": False,
            "linear_memory_scaling_guaranteed": False}


def validate_batch_result(jax, host, parameters, initial_states, gates):
    """Retain per-sample aux and apply scalar hard gates without averaging them."""
    (loss, auxiliary), (gradient, states) = host
    batch_size, width = np.shape(initial_states)
    if np.shape(loss) != () or not np.isfinite(loss):
        raise ValueError("B4 scalar loss invalid")
    for value in jax.tree_util.tree_leaves(auxiliary):
        if np.shape(value)[:1] != (batch_size,) or not np.all(np.isfinite(value)):
            raise ValueError("per-sample auxiliary shape/nonfinite failure")
    hard = auxiliary.get("hard")
    if not isinstance(hard, dict) or not hard:
        raise ValueError("per-sample public physical/mask hard gates missing")
    for name, value in hard.items():
        a = np.asarray(value)
        if a.shape != (batch_size,) or np.any(a != 0):
            raise ValueError("per-sample physical/mask hard gate failed: " + name)
    gate_aux = jax.tree_util.tree_map(lambda value: np.mean(value, axis=0), auxiliary)
    gate_aux["hard"] = {name: np.sum(value, axis=0) for name, value in hard.items()}
    valid = gates.validate_result(jax, ((loss, gate_aux), gradient), parameters, count=66123713, leaves=355)
    states = np.asarray(states)
    if (states.shape != (batch_size, width) or states.dtype != np.asarray(initial_states).dtype
            or not np.all(np.isfinite(states)) or np.any(np.count_nonzero(states, axis=1) == 0)):
        raise ValueError("full per-sample state gradient shape/dtype/finite/nonzero gate failed")
    return {**valid, "initial_state_shape": list(states.shape), "initial_state_finite": True,
            "initial_state_nonzero_per_sample": np.count_nonzero(states, axis=1).tolist(),
            "per_sample_hard_counts": {name: np.asarray(value).tolist() for name, value in hard.items()}}


def compare_full_results(jax, actual, reference, pair, contract):
    """Fixed strict tolerance, complete failure ranges and exact state values."""
    helper = pair.adjacent("inspect_shared_leaf_cuda_acceptance")
    groups = {"loss": (actual[0][0], reference[0][0]),
              "auxiliary": (actual[0][1], reference[0][1]),
              "all_parameter_gradients": (actual[1][0], reference[1][0]),
              "all_initial_state_gradients": (actual[1][1], reference[1][1])}
    comparisons = {name: helper.compare_tree(jax, a, r) for name, (a, r) in groups.items()}
    ranges = {name: pair.failed_coordinate_ranges(jax, a, r) for name, (a, r) in groups.items()}
    a, r = np.asarray(actual[1][1]), np.asarray(reference[1][1])
    coordinates = [{"sample": int(sample), "coordinate": int(coordinate),
                    "bindings": helper.coordinate_binding(contract, int(coordinate)),
                    "actual": float(a[sample, coordinate]), "reference": float(r[sample, coordinate]),
                    "absolute_difference": float(abs(a[sample, coordinate]-r[sample, coordinate])),
                    "allowance": float(1e-6 + 1e-5*abs(r[sample, coordinate]))}
                   for sample, coordinate in np.argwhere(~np.isclose(a, r, rtol=1e-5, atol=1e-6))]
    return {**comparisons, "complete_failed_coordinate_ranges": ranges,
            "state_failure_coordinates": coordinates, "rtol": 1e-5, "atol": 1e-6,
            "passed": all(row["passed"] for row in comparisons.values())}
