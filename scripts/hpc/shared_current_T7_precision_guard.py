"""Disconnected CPU method guards; current real-package graph is NOT admitted.

This module imports no JAX/model owner and provides no real-model harness.
The temporary execution scope accepts synthetic fixture modules only. Static
source conversion is an audit aid, not evidence of full integration or runtime
precision coverage. No production defaults or physical formula are changed.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import re
from typing import Any

import numpy as np

CURRENT_MODULES = (
    'input_conditioning', 'strong_baseline_architecture', 'shared_transformer',
    'shared_decoder_dense', 'shared_decoder_macro_packed',
    'shared_decoder_expert_execution', 'shared_backbone_projection_reuse',
    'shared_backbone_execution', 'shared_latent_reader',
)
SOURCE_HEAD = 'c599236bcf57b745e9bafed678298010f4c3b58f'
R4_EPSILON = 2.0**-23
REAL_PACKAGE_SUPPORTED = False
CURRENT_GRAPH_ADMITTED = False


class NeuralDtypes(ast.NodeTransformer):
    """Same bounded jnp.float32 attribute conversion as the old diagnostic.

    Literal constants, np.float32, integer/Boolean dtypes and physical source
    are untouched. This is only for explicitly selected neural source functions;
    it must never be applied to physical finfo/epsilon expressions.
    """
    def __init__(self):
        self.lines: list[int] = []

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'jnp' and node.attr == 'float32':
            self.lines.append(node.lineno)
            node.attr = 'float64'
        return self.generic_visit(node)


def transform_neural_functions(source: str):
    """Return transformed top-level function AST and changed source lines.

    Classes, module-level constants and all non-function statements are excluded
    from execution. The caller may inspect this AST without any owner import.
    """
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    converter = NeuralDtypes()
    transformed = converter.visit(ast.Module(body=functions, type_ignores=[]))
    return ast.fix_missing_locations(transformed), tuple(sorted(converter.lines))


def exact_array_widen(value: np.ndarray | np.generic):
    """Exactly widen FP32 storage and preserve every other numeric/mask dtype.

    Reject object/string arrays. A roundtrip byte check includes signed zeros
    and NaN payloads, and fails closed if conversion cannot retain source bits.
    """
    if not isinstance(value, (np.ndarray, np.generic)):
        raise TypeError('NumPy arrays/scalars only; no device conversion')
    original = np.asarray(value)
    if original.dtype.kind not in 'fbiu':
        raise ValueError('Only numeric arrays and Boolean masks supported')
    if original.dtype == np.dtype('float32'):
        widened = original.astype(np.float64)
        if widened.astype(original.dtype).tobytes() != original.tobytes():
            raise ValueError('FP32 widening does not retain all source bits')
    else:
        widened = original.copy()
    return widened


def exact_r4_constant_bridge(value):
    """Widen the declared R4 threshold value, never use FP64 machine epsilon."""
    original = np.asarray(value)
    if original.shape != () or original.dtype.kind != 'f' or float(original) != R4_EPSILON:
        raise ValueError('Canonical R4 threshold must remain exactly 2**-23')
    widened = np.float64(original)
    if float(widened) != float(original):
        raise ValueError('R4 threshold changed')
    return widened


def r4_zero_branch(values, *, epsilon):
    """Tiny fixture for source's strict abs(value)<epsilon, not a process port."""
    threshold = exact_r4_constant_bridge(epsilon)
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.abs(values) < threshold, 0.0, values)


def validate_current_identity(identity: dict[str, Any], *, expected_normalization_sha256: str):
    """Require independently supplied normalization identity; never infer it.

    This validates a declared small metadata contract only. It cannot prove the
    checkpoint/input contents or admit a current model execution.
    """
    if not isinstance(expected_normalization_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_normalization_sha256):
        raise ValueError('Explicit valid expected normalization SHA256 required')
    expected = dict(parameter_count=66123713, parameter_leaves=355, artifact_id=299,
                    window=0, horizon=7, architecture='shared_latent32x384_reader64x2',
                    latent_slots=32, reader_width=64, reader_layers=2,
                    latent_reader_execution='field', source_head=SOURCE_HEAD,
                    normalization_sha256=expected_normalization_sha256)
    for key, value in expected.items():
        actual = identity.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError('Current identity differs: '+key)
    return {'metadata_contract_passed': True, 'current_graph_admitted': False,
            'real_package_supported': False}


@contextmanager
def temporary_fixture_fp64_scope(module, source: str, *, enabled: bool = False):
    """Default disconnected; explicit synthetic fixture-only temporary patch.

    No real package module can be patched. Imported aliases and real class
    methods are deliberately unsupported; integration requires separate work.
    Always restore fixture function bindings, including exceptional exits.
    """
    if not enabled:
        yield {'enabled': False, 'changed_lines': (), 'real_package_supported': False}
        return
    if not getattr(module, '__name__', '').startswith('_precision_guard_fixture_'):
        raise ValueError('Only explicitly named synthetic fixture modules supported')
    tree, lines = transform_neural_functions(source)
    names = [node.name for node in tree.body]
    saved = {name: module.__dict__.get(name) for name in names}
    missing = {name for name in names if name not in module.__dict__}
    # Execute into an isolated fixture namespace before changing any binding.
    namespace = module.__dict__.copy()
    exec(compile(tree, '<precision-guard-fixture>', 'exec'), namespace)
    try:
        for name in names:
            setattr(module, name, namespace[name])
        yield {'enabled': True, 'changed_lines': lines, 'real_package_supported': False}
    finally:
        for name in names:
            if name in missing:
                delattr(module, name)
            else:
                setattr(module, name, saved[name])
