"""Default-off current neural diagnostic precision; graph is NOT admitted.

This module imports no JAX/model owner and provides no real-model harness.
Explicit runtime context accepts the audited loaded neural module set, redirects
its loaded aliases and restores all bindings and dtype getters. No real-model
harness is provided. Integration tests cover tiny adapters, not the full graph.
No production defaults, serialized class identities or physical formula change.
"""
from __future__ import annotations

import ast
import __future__
from contextlib import contextmanager
import copy
import hashlib
import importlib
import importlib.util
import inspect
from pathlib import Path
import re
import subprocess
import sys
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
PACKAGE = 'research.daily_coarse_graining.daily_operator.'
REAL_PACKAGE_SUPPORTED = True
CURRENT_GRAPH_ADMITTED = False
_ACTIVE = False
_HISTORICAL_ACTIVE = False
HISTORICAL_EXPERT_SHA256 = '5d77141bad1359d3a71174ba03eba57527d7b0e94cf3e42bce73a2dc650c740d'
HISTORICAL_EXPERT_REVISION = 'e8db045'
HISTORICAL_EXPERT_PATH = 'research/daily_coarse_graining/daily_operator/shared_decoder_expert_execution.py'


@contextmanager
def historical_expert_source_context(root, cache_directory):
    """Install the exact admitted expert Git blob temporarily, without reloads.

    This explicitly invoked diagnostic context replaces only the canonical
    expert module and its loaded package function aliases. Dense/other sibling
    module classes remain the existing objects. Cache bytes must stay beneath
    this checkout's outputs; existing cache files are verified, never replaced.
    Enter before loading assets/entering current_neural_fp64_context and exit
    after the precision context so reverse-order restoration is transactional.
    """
    global _HISTORICAL_ACTIVE
    if _HISTORICAL_ACTIVE or _ACTIVE:
        raise ValueError('Historical source context must precede precision context; nesting unsupported')
    root = Path(root).resolve(strict=True)
    cache_directory = Path(cache_directory).resolve()
    cache_directory.relative_to((root/'outputs').resolve())
    git_root = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
        check=True, capture_output=True, text=True).stdout.strip()
    if Path(git_root).resolve() != root:
        raise ValueError('Historical source root must be the declared checkout')
    source = subprocess.run(['git', '-C', str(root), 'show',
        HISTORICAL_EXPERT_REVISION+':'+HISTORICAL_EXPERT_PATH],
        check=True, capture_output=True).stdout
    if hashlib.sha256(source).hexdigest() != HISTORICAL_EXPERT_SHA256:
        raise ValueError('Pinned historical expert Git blob hash differs')
    if any(isinstance(node, ast.ClassDef) for node in ast.walk(ast.parse(source))):
        raise ValueError('Historical expert must not introduce serialized classes')
    cache = cache_directory/HISTORICAL_EXPERT_REVISION/'shared_decoder_expert_execution.py'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        if cache.read_bytes() != source:
            raise ValueError('Existing historical source cache differs; never overwritten')
    else:
        # Exclusive creation also fails closed on concurrent materialization.
        with cache.open('xb') as stream:
            stream.write(source)
    fullname = PACKAGE+'shared_decoder_expert_execution'
    original = importlib.import_module(fullname)
    package = importlib.import_module(PACKAGE.removesuffix('.'))
    old_attribute = getattr(package, 'shared_decoder_expert_execution')
    old_functions = {name: value for name, value in vars(original).items()
        if inspect.isfunction(value) and value.__module__ == fullname}
    spec = importlib.util.spec_from_file_location(fullname, cache)
    if spec is None or spec.loader is None:
        raise ValueError('Historical source module loader unavailable')
    replacement = importlib.util.module_from_spec(spec)
    bindings = []
    _HISTORICAL_ACTIVE = True
    try:
        sys.modules[fullname] = replacement
        setattr(package, 'shared_decoder_expert_execution', replacement)
        # Disallow creation of __pycache__ beside the immutable cached source.
        # Compile source directly in the canonical module namespace instead.
        exec(compile(source, str(cache), 'exec'), replacement.__dict__)
        new_functions = {name: value for name, value in vars(replacement).items()
            if inspect.isfunction(value) and value.__module__ == fullname}
        mapping = {id(old): new_functions[name] for name, old in old_functions.items() if name in new_functions}
        mapping[id(original)] = replacement
        missing = {id(old): name for name, old in old_functions.items() if name not in new_functions}
        aliases = []
        for owner in list(sys.modules.values()):
            if owner is None or not getattr(owner, '__name__', '').startswith(PACKAGE):
                continue
            for name, old in list(vars(owner).items()):
                if id(old) in missing:
                    raise ValueError('Loaded alias requires a nonhistorical expert function: '+missing[id(old)])
                new = mapping.get(id(old))
                if new is not None and new is not old:
                    bindings.append((owner, name, old))
                    setattr(owner, name, new)
                    aliases.append(dict(module=owner.__name__, name=name,
                        binding_kind='module' if old is original else 'function'))
        yield dict(module=fullname, old_source_path=original.__file__,
            old_source_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
            source_path=str(cache), source_sha256=HISTORICAL_EXPERT_SHA256,
            git_revision=HISTORICAL_EXPERT_REVISION, git_path=HISTORICAL_EXPERT_PATH,
            module_functions=sorted(new_functions), redirected_aliases=aliases,
            sibling_modules_reloaded=False, classes_redefined=False)
    finally:
        for owner, name, old in reversed(bindings):
            setattr(owner, name, old)
        sys.modules[fullname] = original
        setattr(package, 'shared_decoder_expert_execution', old_attribute)
        _HISTORICAL_ACTIVE = False


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


def widen_context(tree):
    """Copy context/head metadata and exactly widen floating calibration leaves.

    Preserve dataclass/namedtuple identities, shared object aliases, dictionary
    keys, nonnumeric metadata and Boolean/integer values/dtypes. This never
    mutates the source context or initializes/calls a model. Array-like device
    leaves are read through NumPy, as in the old diagnostic context bridge.
    Cyclic immutable tuple containers are unsupported and fail explicitly.
    """
    memo = {}
    active_tuples = set()
    rows = []

    def walk(value, path):
        if id(value) in memo:
            return memo[id(value)]
        if isinstance(value, (np.ndarray, np.generic)) or (hasattr(value, 'dtype') and hasattr(value, 'shape')):
            original = np.asarray(value)
            if original.dtype.kind not in 'fbiu':
                return value  # Metadata arrays are not numerical calibration.
            prepared = exact_array_widen(original)
            memo[id(value)] = prepared
            if original.dtype == np.dtype('float32'):
                rows.append(dict(path=path, shape=list(original.shape),
                    original_dtype='float32', prepared_dtype='float64', exact_widen=True,
                    original_bits_preserved=True,
                    original_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                    prepared_sha256=hashlib.sha256(prepared.tobytes()).hexdigest()))
            return prepared
        if isinstance(value, dict):
            prepared = value.copy()
            memo[id(value)] = prepared
            for key, child in value.items():
                prepared[key] = walk(child, path+'.'+str(key))
            return prepared
        # Local import is stdlib only; keeps this helper independent of owners.
        import dataclasses
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            prepared = copy.copy(value)
            memo[id(value)] = prepared
            for field in dataclasses.fields(value):
                object.__setattr__(prepared, field.name,
                    walk(getattr(value, field.name), path+'.'+field.name))
            if type(prepared) is not type(value):
                raise ValueError('Context serialized class identity changed')
            return prepared
        if isinstance(value, list):
            prepared = copy.copy(value)
            memo[id(value)] = prepared
            for index, child in enumerate(value):
                prepared[index] = walk(child, path+'.'+str(index))
            return prepared
        if isinstance(value, tuple):
            if id(value) in active_tuples:
                raise ValueError('Cyclic immutable context tuple unsupported')
            active_tuples.add(id(value))
            try:
                children = [walk(child, path+'.'+(value._fields[index] if hasattr(value, '_fields') else str(index)))
                            for index, child in enumerate(value)]
                prepared = type(value)(*children) if hasattr(value, '_fields') else type(value)(children)
                memo[id(value)] = prepared
                return prepared
            finally:
                active_tuples.remove(id(value))
        return value

    return walk(tree, 'context'), rows


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
            'real_package_supported': REAL_PACKAGE_SUPPORTED}


@contextmanager
def current_neural_fp64_context(modules=(), *, enabled: bool = False):
    """Patch audited already-loaded canonical modules only, explicitly opt-in.

    Yields (old-function-id replacements, manifest). Parameters/calibration
    arrays must separately be exactly widened by the caller. The context does
    not initialize a model, trace a graph, or certify runtime dtype coverage.
    Loaded aliases inside PACKAGE are redirected; arbitrary external captured
    function objects and class methods other than the two policy getters are
    outside the supported scope. No module is implicitly loaded except the
    old exact physical-constant bridge's season/permafrost owners.
    """
    global _ACTIVE
    if not enabled:
        yield {}, dict(enabled=False, modules=[], current_graph_admitted=False)
        return
    modules = tuple(modules)
    if tuple(getattr(module, '__name__', '') for module in modules) != tuple(PACKAGE+name for name in CURRENT_MODULES):
        raise ValueError('Exact loaded current neural module set/order required')
    if _ACTIVE:
        raise ValueError('Nested current precision contexts are unsupported')
    if not modules[1].jax.config.x64_enabled:
        raise ValueError('Explicit x64-enabled runtime required')
    prepared = []
    for module in modules:
        if sys.modules.get(module.__name__) is not module:
            raise ValueError('Module is not the actual loaded canonical owner')
        for dtype in ('float32', 'float64'):
            value = getattr(module.jnp, dtype, None)
            if value is None or np.dtype(value) != np.dtype(dtype):
                raise ValueError('Unknown neural dtype: '+dtype)
        source = inspect.getsource(module)
        tree, lines = transform_neural_functions(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ('jnp', 'np') and node.attr in ('float16', 'bfloat16'):
                raise ValueError('Unsupported neural low-precision dtype')
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'finfo':
                raise ValueError('finfo threshold source is outside neural dtype overlay')
        originals = {node.name: module.__dict__[node.name] for node in tree.body}
        if not all(inspect.isfunction(value) for value in originals.values()):
            raise ValueError('Loaded neural function identities differ')
        prepared.append((module, source, tree, lines, originals))
    from scripts.hpc.shared_neural_fp64_diagnostic import exact_physical_constant_widening
    bindings = []
    properties = []
    replacements = {}
    manifest = dict(enabled=True, modules=[], alias_bindings=[],
                    real_package_supported=True, current_graph_admitted=False,
                    coverage='Loaded top-level neural functions and package aliases; two dtype getters and exact R4 bridge. Full graph unverified.')
    _ACTIVE = True
    try:
        for module, source, tree, lines, originals in prepared:
            # Register restoration before exec so partial-definition failures
            # cannot leak changes into the canonical loaded namespace.
            bindings.extend((module, name, old) for name, old in originals.items())
            exec(compile(tree, module.__file__, 'exec', flags=__future__.annotations.compiler_flag), module.__dict__)
            for name, old in originals.items():
                replacements[id(old)] = module.__dict__[name]
            manifest['modules'].append(dict(module=module.__name__, source=module.__file__,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                fp32_dtype_lines_changed=list(lines), functions_recompiled=sorted(originals)))
        policy = modules[1].NeuralPrecisionPolicy
        for name in ('attention_score_dtype', 'source_family_pooling_dtype'):
            old = getattr(policy, name)
            if not isinstance(old, property):
                raise ValueError('Unknown precision-policy getter')
            properties.append((policy, name, old))
            setattr(policy, name, property(lambda self, jnp=modules[1].jnp: jnp.float64))
        for owner in list(sys.modules.values()):
            if owner is None or not getattr(owner, '__name__', '').startswith(PACKAGE):
                continue
            for name, old in list(vars(owner).items()):
                new = replacements.get(id(old))
                if new is not None and new is not old:
                    bindings.append((owner, name, old))
                    setattr(owner, name, new)
                    manifest['alias_bindings'].append(dict(module=owner.__name__, name=name))
        with exact_physical_constant_widening() as physical:
            if not all(row['exact_value_unchanged'] and row['prepared_value'] == R4_EPSILON for row in physical):
                raise ValueError('Physical R4 threshold identity differs')
            manifest['exact_physical_constant_bridge'] = physical
            yield replacements, manifest
    finally:
        for owner, name, value in reversed(bindings):
            setattr(owner, name, value)
        for owner, name, value in reversed(properties):
            setattr(owner, name, value)
        _ACTIVE = False


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
