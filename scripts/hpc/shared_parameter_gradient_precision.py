"""Proposed parameter-only F16 GEMM rule; no default or real-model admission.

Neural provenance: strong_baseline_architecture.py::_linear886-887,
_ffn897-898. Physics, public loss and optimizer are outside this policy.
Primal and input tangent retain the original FP32 matmul. Native custom_jvp
transpose uses the requested algorithm only for the kernel tangent term.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero


def vector_metrics(jax,actual,reference):
    """Reuse the existing host metric implementation, never its precision policy."""
    name = 'parameter_gradient_original_metric_support'
    if name not in sys.modules:
        path = Path(__file__).with_name('probe_shared_neural_precision.py')
        spec = importlib.util.spec_from_file_location(name,path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].vector_metrics(jax,actual,reference)

CONTRACT = {'exact_rtol':1e-5,'exact_atol':1e-6,'loss_relative_limit':1e-3,
            'gradient_relative_l2_limit':1e-2,'gradient_cosine_minimum':0.9999,
            'near_zero_reference_norm':1e-12,'near_zero_absolute_max_limit':1e-6,
            'state_derivatives':'strict_coordinate_gate_no_self_drift_waiver',
            'scope':'parameter_gradient_engineering_diagnostic_not_adoption'}

TARGETS = ('shared_transformer','shared_backbone_execution',
           'shared_backbone_projection_reuse','shared_decoder_macro_packed')
FFN_TARGETS = ('shared_transformer','shared_backbone_projection_reuse','shared_decoder_macro_packed')
PARAMETER_TANGENT_SCALE = 4096.0


def _kernel_tangent_dot(value,kernel_tangent):
    """Only this tangent branch is matrixized; no FP32 fallback exists."""
    rows = value.reshape((-1,value.shape[-1]))
    scale = jnp.asarray(PARAMETER_TANGENT_SCALE,dtype=jnp.float32)
    result = jax.lax.dot_general(rows,kernel_tangent/scale,(((1,),(0,)),((),())),
        precision=jax.lax.DotAlgorithmPreset.F16_F16_F32,
        preferred_element_type=jnp.float32)
    return (result*scale).reshape(value.shape[:-1]+(kernel_tangent.shape[-1],))


@jax.custom_jvp
def _linear_parts(value,kernel,bias):
    return value@kernel+bias


@partial(_linear_parts.defjvp,symbolic_zeros=True)
def _linear_parts_jvp(primals,tangents):
    value,kernel,bias = primals
    value_t,kernel_t,bias_t = tangents
    primal = value@kernel+bias
    terms = []
    if not isinstance(value_t,SymbolicZero):
        terms.append(value_t@kernel)
    if not isinstance(kernel_t,SymbolicZero):
        terms.append(_kernel_tangent_dot(value,kernel_t))
    if not isinstance(bias_t,SymbolicZero):
        terms.append(jnp.broadcast_to(bias_t,primal.shape))
    tangent = sum(terms[1:],terms[0]) if terms else jnp.zeros_like(primal)
    return primal,tangent


def parameter_gradient_linear(parameters,value):
    value = jnp.asarray(value,dtype=jnp.float32)
    kernel,bias = parameters['kernel'],parameters['bias']
    if kernel.dtype!=jnp.float32 or bias.dtype!=jnp.float32 or kernel.ndim!=2 or value.ndim<1:
        raise ValueError('requires original FP32 kernel/bias and rank>=1 input')
    if value.shape[-1]!=kernel.shape[0] or bias.shape!=(kernel.shape[1],):
        raise ValueError('original linear shape contract differs')
    return _linear_parts(value,kernel,bias)


def parameter_gradient_ffn(parameters,value):
    return parameter_gradient_linear(parameters['second'],
        jax.nn.gelu(parameter_gradient_linear(parameters['first'],value)))


@contextmanager
def neural_policy(modules,*,mixed):
    """Independent scoped alias overlay, restoring all functions on every exit."""
    indexed = {module.__name__.rsplit('.',1)[-1]:module for module in modules}
    if len(indexed)!=len(modules) or set(indexed)!=set(TARGETS):
        raise ValueError('requires exact active four-module scope')
    package = indexed['shared_transformer'].__name__.rsplit('.',1)[0]
    original = sys.modules[package+'.strong_baseline_architecture']
    saved_linear = [(module,module._linear) for module in modules]
    saved_ffn = [(indexed[name],indexed[name]._ffn) for name in FFN_TARGETS]
    if any(fn is not original._linear for _,fn in saved_linear) or any(fn is not original._ffn for _,fn in saved_ffn):
        raise ValueError('requires unchanged original FP32 aliases')
    macro = indexed['shared_decoder_macro_packed']
    adapter = sys.modules.get(package+'.shared_decoder_expert_execution')
    decoder = getattr(adapter,'expert_decode_fields',None)
    if decoder is None or decoder.__globals__ is not macro.__dict__:
        raise ValueError('actual decoder does not resolve macro globals')
    try:
        if mixed:
            for module,_ in saved_linear:
                module._linear = parameter_gradient_linear
            for module,_ in saved_ffn:
                module._ffn = parameter_gradient_ffn
        yield
    finally:
        for module,fn in saved_ffn:
            module._ffn = fn
        for module,fn in saved_linear:
            module._linear = fn


def policy_evidence():
    return {'mode':'proposed_fixed_S4096_parameter_tangent_only','AD':'native_custom_jvp_transpose',
            'parameter_derivative':'new numerical approximation to original FP32 derivative',
            'finite_precision_tangent_linearity':'approximate; no exact adjoint consistency claim',
            'kernel_tangent_algorithm':'F16_F16_F32','primal':'original_FP32_matmul',
            'input_tangent':'original_FP32_matmul','bias_gradient':'FP32',
            'kernel_storage':'FP32','kernel_gradient_accumulation':'FP32','physics':'FP64',
            'rank1_rule':'matrixize only kernel tangent term, preserving primal/input matmul shape',
            'linear_aliases':list(TARGETS),'ffn_aliases':list(FFN_TARGETS),
            'canonical_tokenizer':'unchanged original strong-baseline functions',
            'full_neural_coverage':False,'FP32_fallback':False,'loss_scaling':False,
            'parameter_tangent_local_scale':PARAMETER_TANGENT_SCALE,
            'kernel_JVP':'S * dot_F16_F16_F32(x, W_t/S)',
            'kernel_VJP':'dot_F16_F16_F32(x.T, S*g)/S',
            'scientific_interface_limit':'reverse parameter-gradient diagnostic only; candidate parameter JVP forbidden as correctness oracle',
            'production_adoption':False,'real_model_or_GPU_verified':False}


def alias_metadata(modules):
    return policy_evidence() | {'patched_aliases':[{'module':module.__name__,'alias':'_linear'}
                                                  for module in modules]
        + [{'module':module.__name__,'alias':'_ffn'} for module in modules
           if module.__name__.rsplit('.',1)[-1] in FFN_TARGETS]}
