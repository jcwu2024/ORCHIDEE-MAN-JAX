"""Exact-primal JAX primitives for source formulas with singular tangents."""

from __future__ import annotations

import jax.numpy as jnp
from jax import config
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

config.update("jax_enable_x64", True)


_stable_ratio_p = Primitive("orchidee_source_ratio")
_stable_ratio_p.def_impl(lambda numerator, denominator: numerator / denominator)
_stable_ratio_p.def_abstract_eval(lambda numerator, _denominator: numerator)
mlir.register_lowering(
    _stable_ratio_p,
    mlir.lower_fun(
        lambda numerator, denominator: numerator / denominator,
        multiple_results=False,
    ),
)
batching.defbroadcasting(_stable_ratio_p)


def _stable_ratio_jvp(primals, tangents):
    numerator, denominator = primals
    numerator_tangent, denominator_tangent = map(ad.instantiate_zeros, tangents)
    ratio = _stable_ratio_p.bind(numerator, denominator)
    representable = jnp.abs(denominator) >= jnp.sqrt(
        jnp.finfo(denominator.dtype).tiny
    )
    safe_ratio = jnp.where(representable, ratio, 0.0)
    safe_denominator = jnp.where(representable, denominator, 1.0)
    safe_numerator_tangent = jnp.where(representable, numerator_tangent, 0.0)
    safe_denominator_tangent = jnp.where(
        representable, denominator_tangent, 0.0
    )
    tangent = (
        safe_numerator_tangent - safe_ratio * safe_denominator_tangent
    ) / safe_denominator
    return ratio, tangent


ad.primitive_jvps[_stable_ratio_p] = _stable_ratio_jvp


def source_ratio_with_finite_tangent(numerator, denominator):
    """Lower to the source quotient while masking unrepresentable tangents."""

    return _stable_ratio_p.bind(numerator, denominator)


_finite_zero_sqrt_p = Primitive("orchidee_source_sqrt")
_finite_zero_sqrt_p.def_impl(jnp.sqrt)
_finite_zero_sqrt_p.def_abstract_eval(lambda value: value)
mlir.register_lowering(
    _finite_zero_sqrt_p,
    mlir.lower_fun(jnp.sqrt, multiple_results=False),
)
batching.defvectorized(_finite_zero_sqrt_p)


def _finite_zero_sqrt_jvp(primals, tangents):
    (value,) = primals
    tangent = ad.instantiate_zeros(tangents[0])
    result = _finite_zero_sqrt_p.bind(value)
    safe_result = jnp.where(value == 0.0, 1.0, result)
    derivative = jnp.where(value == 0.0, 0.0, 0.5 / safe_result)
    return result, derivative * tangent


ad.primitive_jvps[_finite_zero_sqrt_p] = _finite_zero_sqrt_jvp


def source_sqrt_with_finite_zero_tangent(value):
    """Lower to source sqrt while choosing the zero subgradient at zero."""

    return _finite_zero_sqrt_p.bind(value)
