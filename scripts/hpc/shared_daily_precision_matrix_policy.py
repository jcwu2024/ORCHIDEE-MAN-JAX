"""Keep native F16_F16_F32 while expressing rank-one linears as 1-row GEMMs.

Actual V100/JAX0.4.38 rejects the preset for rank-one lhs[384]@kernel[384,384],
but supports the same kernel with lhs[1,384], including complete native VJP.
The original kernel, bias, output shape and native derivative inputs remain.
"""
from __future__ import annotations
from contextlib import contextmanager
from collections import Counter

import probe_shared_neural_precision as base

CONTRACT = base.CONTRACT
vector_metrics = base.vector_metrics
parameter_groups = base.parameter_groups
LINEAR_SHAPES = Counter()


def algorithm_linear(parameters, value):
    import jax.numpy as jnp
    x = jnp.asarray(value, jnp.float32)
    shape = (tuple(x.shape), tuple(parameters["kernel"].shape), x.ndim == 1)
    LINEAR_SHAPES[shape] += 1
    if x.ndim == 1:
        return base.algorithm_linear(parameters, x.reshape((1, x.shape[-1])))[0]
    return base.algorithm_linear(parameters, x)


def algorithm_ffn(parameters, value):
    import jax
    return algorithm_linear(parameters["second"], jax.nn.gelu(algorithm_linear(parameters["first"], value)))


def policy_evidence():
    return {
        "algorithm": "unchanged_native_DotAlgorithmPreset.F16_F16_F32",
        "rank1_matching_rule": "only value.ndim == 1: reshape(K)->(1,K), same kernel and bias; restore output via [0]",
        "FP32_fallback": False, "custom_AD": False,
        "observed_trace_shapes": [{"lhs": list(lhs), "kernel": list(kernel),
                                   "matrixized": matrixized, "trace_calls": count}
                                  for (lhs, kernel, matrixized), count in sorted(LINEAR_SHAPES.items())],
    }


@contextmanager
def neural_policy(modules, *, mixed):
    saved = []
    try:
        if mixed:
            for module in modules:
                for name, replacement in (("_linear", algorithm_linear), ("_ffn", algorithm_ffn)):
                    if hasattr(module, name):
                        saved.append((module, name, getattr(module, name)))
                        setattr(module, name, replacement)
        yield
    finally:
        for module, name, previous in reversed(saved):
            setattr(module, name, previous)
