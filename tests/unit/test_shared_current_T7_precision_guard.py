"""Tiny CPU/AST fixtures only; no real neural import or scientific assets."""
import ast
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from scripts.hpc import shared_current_T7_precision_guard as guard


def identity():
    return dict(parameter_count=66123713, parameter_leaves=355, artifact_id=299,
                window=0, horizon=7, architecture='shared_latent32x384_reader64x2',
                latent_slots=32, reader_width=64, reader_layers=2,
                latent_reader_execution='field', source_head=guard.SOURCE_HEAD,
                normalization_sha256='a'*64)


def test_exact_widen_preserves_source_bits_and_masks():
    raw = np.array([0, 0x80000000, 0x3f800001, 0x7fc00001, 0x7f800000], dtype=np.uint32).view(np.float32)
    result = guard.exact_array_widen(raw)
    assert result.dtype == np.float64
    assert result.astype(np.float32).tobytes() == raw.tobytes()
    for array in (np.array([True, False]), np.array([-1, 42], dtype=np.int32), np.array([0.25], dtype=np.float64)):
        preserved = guard.exact_array_widen(array)
        assert preserved.dtype == array.dtype and preserved.tobytes() == array.tobytes()
        assert not np.shares_memory(preserved, array)
    with pytest.raises(ValueError):
        guard.exact_array_widen(np.array([object()], dtype=object))


def test_narrowing_counterexample_exposes_reader_restore_failure():
    decoded = np.array([1.0 + 2**-30], dtype=np.float64)
    raw = np.zeros(1, dtype=np.float32)
    raw[0] = decoded[0]
    assert raw[0] != decoded[0]
    restored = np.zeros(1, dtype=np.float64)
    restored[0] = decoded[0]
    assert restored[0] == decoded[0]


def test_r4_bridge_and_strict_threshold_branch_are_preserved():
    original = np.float32(2**-23)
    widened = guard.exact_r4_constant_bridge(original)
    assert widened.dtype == np.float64 and widened == original == 2**-23
    values = np.array([-2**-23, -(2**-24), 0, 2**-24, 2**-23], dtype=np.float64)
    np.testing.assert_array_equal(guard.r4_zero_branch(values, epsilon=original), [-(2**-23), 0, 0, 0, 2**-23])
    np.testing.assert_array_equal(guard.r4_zero_branch(values, epsilon=widened), guard.r4_zero_branch(values, epsilon=original))
    with pytest.raises(ValueError):
        guard.exact_r4_constant_bridge(np.finfo(np.float64).eps)
    with pytest.raises(ValueError):
        guard.r4_zero_branch(values, epsilon=np.float64(2**-24))


def test_identity_positive_is_metadata_only():
    result = guard.validate_current_identity(identity(), expected_normalization_sha256='a'*64)
    assert result['metadata_contract_passed'] and not result['current_graph_admitted']
    assert not result['real_package_supported']


@pytest.mark.parametrize('key,value', [
    ('parameter_count', 64884481), ('parameter_leaves', 300), ('artifact_id', 742),
    ('window', 1), ('horizon', 30), ('architecture', 'original'),
    ('latent_slots', 16), ('reader_width', 32), ('reader_layers', 1),
    ('latent_reader_execution', 'expert'), ('source_head', 'b'*40),
    ('normalization_sha256', 'b'*64), ('artifact_id', True),
])
def test_identity_rejects_wrong_current_scope(key, value):
    data = identity()
    data[key] = value
    with pytest.raises(ValueError, match=key):
        guard.validate_current_identity(data, expected_normalization_sha256='a'*64)


@pytest.mark.parametrize('expected', ['', None, 'a'*63, 'z'*64])
def test_identity_requires_independent_valid_normalization_hash(expected):
    with pytest.raises(ValueError):
        guard.validate_current_identity(identity(), expected_normalization_sha256=expected)


def fixture():
    source = 'def restore(value):\n    threshold = 2**-23\n    return jnp.asarray(value, dtype=jnp.float32), threshold\n'
    module = ModuleType('_precision_guard_fixture_restore')
    module.jnp = np
    exec(source, module.__dict__)
    return module, source


def test_default_disconnected_and_explicit_fixture_scope_restores_after_error():
    module, source = fixture()
    original = module.restore
    with guard.temporary_fixture_fp64_scope(module, source) as manifest:
        assert not manifest['enabled'] and module.restore is original
        assert module.restore([1.0])[0].dtype == np.float32
    with pytest.raises(RuntimeError):
        with guard.temporary_fixture_fp64_scope(module, source, enabled=True):
            assert module.restore([1.0])[0].dtype == np.float64
            assert module.restore([1.0])[1] == 2**-23
            raise RuntimeError('fixture failure')
    assert module.restore is original
    with guard.temporary_fixture_fp64_scope(module, source, enabled=True):
        assert module.restore([1.0])[0].dtype == np.float64
    assert module.restore is original


def test_real_package_patch_is_explicitly_unsupported():
    module = ModuleType('research.daily_coarse_graining.daily_operator.shared_latent_reader')
    with pytest.raises(ValueError, match='synthetic'):
        with guard.temporary_fixture_fp64_scope(module, 'def f():\n    return 1\n', enabled=True):
            pass


def test_actual_reader_ast_covers_restore_and_preserves_non_dtype_nodes():
    root = Path(__file__).resolve().parents[2]
    source = (root/'research/daily_coarse_graining/daily_operator/shared_latent_reader.py').read_text()
    original = ast.parse(source)
    tree, lines = guard.transform_neural_functions(source)
    assert 'shared_latent_reader' in guard.CURRENT_MODULES and len(guard.CURRENT_MODULES) == 9
    restore = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == '_restore_field')
    dtype = next(node for node in ast.walk(restore) if isinstance(node, ast.Attribute) and node.attr == 'float32')
    assert dtype.lineno in lines and dtype.lineno == 277
    changed = next(node for node in tree.body if node.name == '_restore_field')
    assert any(isinstance(node, ast.Attribute) and node.attr == 'float64' for node in ast.walk(changed))
    # Reverse only the recorded transformed attributes: full structural AST
    # equality then proves constants, branches, formulas and other dtypes intact.
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'jnp' and node.attr == 'float64' and node.lineno in lines:
            node.attr = 'float32'
    functions = ast.Module(body=[node for node in original.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))], type_ignores=[])
    assert ast.dump(tree, include_attributes=True) == ast.dump(functions, include_attributes=True)
    assert not guard.REAL_PACKAGE_SUPPORTED and not guard.CURRENT_GRAPH_ADMITTED
