"""Tiny CPU/AST and canonical adapter fixtures; no real-model graph/data."""
import ast
import importlib
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
    assert result['real_package_supported']


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
    assert guard.REAL_PACKAGE_SUPPORTED and not guard.CURRENT_GRAPH_ADMITTED


@pytest.fixture
def canonical_modules():
    # Imports definitions only: no model init/forward, shape evaluation or graph.
    modules = tuple(importlib.import_module(guard.PACKAGE+name) for name in guard.CURRENT_MODULES)
    jax = modules[1].jax
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    try:
        yield modules
    finally:
        jax.config.update('jax_enable_x64', previous)


def test_actual_loaded_precision_scope_tiny_restore_and_aliases(canonical_modules):
    modules = canonical_modules
    baseline, reader = modules[1], modules[-1]
    process_head = importlib.import_module(guard.PACKAGE+'shared_process_head')
    season = importlib.import_module('jax_orchidee.stomate.season')
    permafrost = importlib.import_module('jax_orchidee.stomate.permafrost')
    policy = baseline.NeuralPrecisionPolicy
    field = type('TinyField', (), {'shape': (1,)})()
    decoded = np.array([1.0 + 2**-30], dtype=np.float64)
    active = np.array([True])
    original_restore = reader._restore_field
    original_linear = baseline._linear
    original_class = baseline.StrongBaselineLayout
    original_policy = policy.attention_score_dtype
    original_pooling = policy.source_family_pooling_dtype
    original_season = season.FORTRAN_EPSILON_R4
    original_permafrost = permafrost.FORTRAN_EPSILON_R4
    original_defaults = permafrost.microactem_temperature_control.__kwdefaults__
    # Capture all actual package imported function aliases for exact restoration.
    import sys
    aliases = [(owner, name, value) for owner in list(sys.modules.values())
               if owner is not None and getattr(owner, '__name__', '').startswith(guard.PACKAGE)
               for name, value in list(vars(owner).items()) if value is original_linear]
    head_aliases = [(name, value) for name, value in vars(process_head).items()
                    if callable(value) and any(value is getattr(module, name, None) for module in modules)]
    assert head_aliases
    before, _ = original_restore(field, np.array([0]), active, decoded)
    assert before.dtype == np.float32 and float(before[0]) != decoded[0]
    with guard.current_neural_fp64_context(modules) as (replacements, manifest):
        assert not replacements and not manifest['enabled']
        assert reader._restore_field is original_restore
    for exceptional in (False, True):
        try:
            with guard.current_neural_fp64_context(modules, enabled=True) as (replacements, manifest):
                result, mask = reader._restore_field(field, np.array([0]), active, decoded)
                assert result.dtype == np.float64 and float(result[0]) == decoded[0]
                np.testing.assert_array_equal(mask, active)
                assert baseline.StrongBaselineLayout is original_class
                assert policy().attention_score_dtype == np.float64
                assert policy().source_family_pooling_dtype == np.float64
                assert reader._linear is baseline._linear is replacements[id(original_linear)]
                for owner, name, _ in aliases:
                    assert getattr(owner, name) is baseline._linear
                assert any(getattr(process_head, name) is replacements[id(old)]
                           for name, old in head_aliases if id(old) in replacements)
                assert len(manifest['modules']) == 9
                assert 277 in manifest['modules'][-1]['fp32_dtype_lines_changed']
                assert all(len(row['source_sha256']) == 64 for row in manifest['modules'])
                assert manifest['current_graph_admitted'] is False
                assert season.FORTRAN_EPSILON_R4.dtype == np.float64
                assert permafrost.FORTRAN_EPSILON_R4.dtype == np.float64
                assert permafrost.microactem_temperature_control.__kwdefaults__['epsilon'] == 2**-23
                assert permafrost.microactem_temperature_control.__kwdefaults__['epsilon'].dtype == np.float64
                assert season.FORTRAN_EPSILON_R4 == original_season == 2**-23
                np.testing.assert_array_equal(guard.r4_zero_branch([2**-24, 2**-23], epsilon=season.FORTRAN_EPSILON_R4), [0, 2**-23])
                if exceptional:
                    raise RuntimeError('actual scope fixture failure')
        except RuntimeError:
            if not exceptional:
                raise
        assert reader._restore_field is original_restore
        assert baseline._linear is original_linear
        assert baseline.StrongBaselineLayout is original_class
        assert policy.attention_score_dtype is original_policy
        assert policy.source_family_pooling_dtype is original_pooling
        assert season.FORTRAN_EPSILON_R4 is original_season
        assert permafrost.FORTRAN_EPSILON_R4 is original_permafrost
        assert permafrost.microactem_temperature_control.__kwdefaults__ is original_defaults
        for owner, name, old in aliases:
            assert getattr(owner, name) is old
        for name, old in head_aliases:
            assert getattr(process_head, name) is old


def test_actual_context_rejects_order_and_wrong_dtype(canonical_modules, monkeypatch):
    modules = canonical_modules
    original = modules[-1]._restore_field
    with pytest.raises(ValueError, match='set/order'):
        with guard.current_neural_fp64_context(tuple(reversed(modules)), enabled=True):
            pass
    fake_jnp = type('FakeDtypes', (), {'float32': np.float16, 'float64': np.float64})()
    monkeypatch.setattr(modules[0], 'jnp', fake_jnp)
    with pytest.raises(ValueError, match='Unknown neural dtype'):
        with guard.current_neural_fp64_context(modules, enabled=True):
            pass
    assert modules[-1]._restore_field is original


def test_recursive_context_widen_preserves_classes_aliases_bits_and_originals():
    from dataclasses import dataclass
    from typing import NamedTuple
    from pathlib import PurePosixPath

    @dataclass(frozen=True)
    class Calibration:
        center: object
        scale: object
        mask: object

    class Head(NamedTuple):
        normalization: object
        layout: object

    bits = np.array([0, 0x80000000, 0x3f800001, 0x7fc00001], dtype=np.uint32)
    center = bits.view(np.float32)
    mask = np.array([True, False])
    indices = np.array([1, -1], dtype=np.int32)
    calibration = Calibration(center, center, mask)
    head = Head(calibration, {'indices': indices})
    metadata = PurePosixPath('/WORK/saved/normalization')
    source = {'head': head, 'candidate_head': head, 'normalization': calibration,
              'arrays': [center, mask, indices], 'path': metadata, 'label': 'unchanged',
              'string_array': np.array(['metadata']), 'float64': np.array([2.0])}
    original_bits = center.tobytes()
    prepared, rows = guard.widen_context(source)
    assert prepared is not source
    assert type(prepared['head']) is Head and type(prepared['normalization']) is Calibration
    assert prepared['head'] is prepared['candidate_head']
    assert prepared['head'].normalization is prepared['normalization']
    assert prepared['normalization'].center is prepared['normalization'].scale is prepared['arrays'][0]
    assert prepared['normalization'].mask is prepared['arrays'][1]
    assert prepared['head'].layout['indices'] is prepared['arrays'][2]
    widened = prepared['normalization'].center
    assert widened.dtype == np.float64 and widened.astype(np.float32).tobytes() == original_bits
    assert prepared['arrays'][1].dtype == mask.dtype and prepared['arrays'][1].tobytes() == mask.tobytes()
    assert prepared['arrays'][2].dtype == indices.dtype and prepared['arrays'][2].tobytes() == indices.tobytes()
    assert prepared['path'] is metadata and prepared['label'] == source['label']
    assert prepared['string_array'] is source['string_array']
    assert prepared['float64'].dtype == np.float64
    assert len(rows) == 1 and rows[0]['original_bits_preserved']
    assert rows[0]['path'] == 'context.head.normalization.center'
    assert rows[0]['original_sha256'] != rows[0]['prepared_sha256']
    assert center.tobytes() == original_bits and source['head'] is head and calibration.center is center
    assert center.dtype == np.float32  # Source requires no restoration: never patched.


def test_context_copy_preserves_mutable_cycles_and_rejects_tuple_cycles():
    source = {'values': []}
    source['self'] = source
    source['values'].append(source)
    prepared, rows = guard.widen_context(source)
    assert prepared['self'] is prepared and prepared['values'][0] is prepared and not rows
    bridge = []
    immutable = (bridge,)
    bridge.append(immutable)
    with pytest.raises(ValueError, match='Cyclic immutable'):
        guard.widen_context(immutable)


def test_context_widen_actual_metadata_class_without_model(canonical_modules):
    policy_class = canonical_modules[1].NeuralPrecisionPolicy
    policy = policy_class(attention_normalization_epsilon=np.float32(1e-6))
    prepared, rows = guard.widen_context({'policy': policy, 'alias': policy})
    assert type(prepared['policy']) is policy_class
    assert prepared['policy'] is prepared['alias'] and prepared['policy'] is not policy
    assert np.asarray(prepared['policy'].attention_normalization_epsilon).dtype == np.float64
    assert np.asarray(policy.attention_normalization_epsilon).dtype == np.float32
    assert len(rows) == 1 and rows[0]['original_bits_preserved']


def test_actual_historical_expert_hash_aliases_classes_and_restoration(canonical_modules, monkeypatch):
    import sys
    modules = canonical_modules
    original = modules[5]
    fullname = original.__name__
    package = importlib.import_module(guard.PACKAGE.removesuffix('.'))
    dense = modules[3]
    dense_class = dense.DenseDecoderField
    old_function = original.shared_transformer_raw_outputs_expert128
    head = importlib.import_module(guard.PACKAGE+'shared_process_head')
    monkeypatch.setattr(head, '_precision_fixture_expert_alias', old_function, raising=False)
    monkeypatch.setattr(head, '_precision_fixture_expert_module', original, raising=False)
    root = Path(__file__).resolve().parents[2]
    cache_directory = root/'outputs/T7-current-model-precision-admission-preparation-20261014/v2/source-cache'
    for exceptional in (False, True):
        try:
            with guard.historical_expert_source_context(root, cache_directory) as metadata:
                loaded = sys.modules[fullname]
                assert loaded is not original
                assert package.shared_decoder_expert_execution is loaded
                assert head._precision_fixture_expert_alias is loaded.shared_transformer_raw_outputs_expert128
                assert head._precision_fixture_expert_module is loaded
                assert modules[3] is sys.modules[dense.__name__]
                assert loaded.DenseDecoderField is dense.DenseDecoderField is dense_class
                assert metadata['source_sha256'] == guard.HISTORICAL_EXPERT_SHA256
                assert guard.hashlib.sha256(Path(loaded.__file__).read_bytes()).hexdigest() == guard.HISTORICAL_EXPERT_SHA256
                assert metadata['module_functions'] == ['shared_transformer_raw_outputs_expert128']
                assert not metadata['sibling_modules_reloaded'] and not metadata['classes_redefined']
                current = tuple(importlib.import_module(guard.PACKAGE+name) for name in guard.CURRENT_MODULES)
                with guard.current_neural_fp64_context(current, enabled=True) as (_, manifest):
                    expert_manifest = manifest['modules'][5]
                    assert expert_manifest['source_sha256'] == guard.HISTORICAL_EXPERT_SHA256
                    assert loaded.DenseDecoderField is dense_class
                assert head._precision_fixture_expert_alias is loaded.shared_transformer_raw_outputs_expert128
                if exceptional:
                    raise RuntimeError('historical fixture failure')
        except RuntimeError:
            if not exceptional:
                raise
        assert sys.modules[fullname] is original
        assert package.shared_decoder_expert_execution is original
        assert head._precision_fixture_expert_alias is old_function
        assert head._precision_fixture_expert_module is original
        assert dense.DenseDecoderField is dense_class


def test_historical_expert_rejects_cache_outside_outputs(canonical_modules, tmp_path):
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError):
        with guard.historical_expert_source_context(root, tmp_path):
            pass
